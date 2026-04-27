from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from sim.api import SimulationConfig, SimulationSession
from sim.game.defensive_target import DefensiveTargetIntentProvider
from sim.game.manual import KeyboardCommandState, ManualGameCommandProvider
from sim.game.training import RPOTrainingConfig, RPOTrainingTracker
from sim.presets.thrusters import resolve_thruster_max_thrust_n_from_specs

SPEED_MULTIPLIER_OPTIONS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0)


def _max_accel_from_config(config: SimulationConfig, controlled_object_id: str) -> float:
    section = config.scenario.chaser if controlled_object_id == "chaser" else config.scenario.target
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    if "player_max_accel_km_s2" in game_cfg:
        return float(game_cfg["player_max_accel_km_s2"])
    params = dict((section.mission_strategy.params if section.mission_strategy is not None else {}) or {})
    if "max_accel_km_s2" in params:
        return float(params["max_accel_km_s2"])
    orbit_params = dict((section.orbit_control.params if section.orbit_control is not None else {}) or {})
    if "max_accel_km_s2" in orbit_params:
        return float(orbit_params["max_accel_km_s2"])
    specs = dict(section.specs or {})
    max_thrust_n = resolve_thruster_max_thrust_n_from_specs(specs)
    dry_mass_kg = specs.get("dry_mass_kg", specs.get("mass_kg"))
    fuel_mass_kg = specs.get("fuel_mass_kg", 0.0)
    if max_thrust_n is not None and dry_mass_kg is not None:
        wet_mass_kg = float(dry_mass_kg) + float(fuel_mass_kg or 0.0)
        if wet_mass_kg > 0.0:
            return float(max_thrust_n) / wet_mass_kg / 1e3
    return 2.0e-5


def _game_control_mode(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("control_mode", "attitude_thrust") or "attitude_thrust").strip().lower()


def _game_difficulty(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("difficulty", "easy") or "easy").strip().lower()


def _game_defensive_target_provider(config: SimulationConfig) -> DefensiveTargetIntentProvider | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = dict(game_cfg.get("defensive_target", {}) or {})
    if not bool(raw.get("enabled", False)):
        return None
    return DefensiveTargetIntentProvider(
        chaser_object_id=str(raw.get("chaser_object_id", "chaser") or "chaser"),
        trigger_range_km=float(raw.get("trigger_range_km", 1.2) or 1.2),
        trigger_closing_speed_km_s=float(raw.get("trigger_closing_speed_km_s", 0.00025) or 0.00025),
        keepout_radius_km=float(raw.get("keepout_radius_km", 0.25) or 0.25),
        max_accel_km_s2=float(raw.get("max_accel_km_s2", 7.5e-6) or 7.5e-6),
        max_delta_v_m_s=_optional_float(raw.get("max_delta_v_m_s")),
        cross_track_bias=float(raw.get("cross_track_bias", 0.65) or 0.65),
        pulse_period_s=float(raw.get("pulse_period_s", 120.0) or 120.0),
    )


def _game_ric_reference_object_id(config: SimulationConfig, default: str) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("ric_reference_object_id", default) or default)


def _coast_prediction_orbit_fraction(difficulty: str) -> float:
    table = {
        "easy": 1.0,
        "medium": 0.5,
        "normal": 0.5,
        "hard": 0.25,
        "extreme": 0.0,
        "expert": 0.0,
    }
    key = str(difficulty or "easy").strip().lower()
    if key not in table:
        raise ValueError("metadata.game.difficulty must be one of: easy, medium, hard, extreme")
    return table[key]


def _wall_step_s(dt_s: float, speed_multiple: float) -> float:
    return float(dt_s) / max(float(speed_multiple), 1.0e-9)


def _coerce_speed_multiple(speed_multiple: float) -> float:
    value = float(speed_multiple)
    return min(SPEED_MULTIPLIER_OPTIONS, key=lambda option: abs(option - value))


def _adjust_speed_multiple(speed_multiple: float, change: int) -> float:
    current = _coerce_speed_multiple(speed_multiple)
    idx = SPEED_MULTIPLIER_OPTIONS.index(current)
    idx = int(np.clip(idx + int(change), 0, len(SPEED_MULTIPLIER_OPTIONS) - 1))
    return SPEED_MULTIPLIER_OPTIONS[idx]


def _command_status(state: KeyboardCommandState, *, control_mode: str = "attitude_thrust") -> str:
    if control_mode in {"ric", "ric_translation", "translation"}:
        sim_state = "PAUSED" if state.paused else "RUNNING"
        return (
            "W/S radial +/-R  A/D in-track +/-I  Left/Right cross-track +/-C\n"
            "Use small pulses, then coast and watch the target-centered RIC motion.\n"
            f"{sim_state}  R={state.pitch:+.0f} I={state.yaw:+.0f} C={state.roll:+.0f} throttle={state.throttle:.2f}"
        )
    burn = "FIRE" if state.firing else "coast"
    return (
        "W/S pitch  A/D yaw  Left/Right roll  Space fire  R reset  Esc quit\n"
        "Keys work in the figure window or this terminal; terminal input is pulse/repeat based.\n"
        f"pitch={state.pitch:+.0f} yaw={state.yaw:+.0f} roll={state.roll:+.0f} thrust={burn}"
    )


def run_game_mode(
    config_path: str | Path,
    *,
    controlled_object_id: str = "chaser",
    attitude_rate_deg_s: float = 45.0,
    realtime: bool = True,
    speed_multiple: float = 1.0,
) -> None:
    from sim.game.pygame_dashboard import PygameRPODashboard

    config = SimulationConfig.from_yaml(config_path)
    control_mode = _game_control_mode(config)
    difficulty = _game_difficulty(config)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    ric_reference_object_id = _game_ric_reference_object_id(config, training_cfg.target_object_id)
    trainer = RPOTrainingTracker(training_cfg)
    command_state = KeyboardCommandState()
    session, _, snapshot = _start_game_attempt(
        config,
        command_state=command_state,
        training_cfg=training_cfg,
        controlled_object_id=controlled_object_id,
        attitude_rate_deg_s=attitude_rate_deg_s,
        control_mode=control_mode,
        ric_reference_object_id=ric_reference_object_id,
    )

    anim_cfg = dict(config.scenario.outputs.animations or {})
    current_speed_multiple = _coerce_speed_multiple(speed_multiple)
    dashboard = PygameRPODashboard(
        target_object_id=str(anim_cfg.get("battlespace_dashboard_target_object_id", "target")),
        chaser_object_id=str(anim_cfg.get("battlespace_dashboard_chaser_object_id", "chaser")),
        reference_object_id=ric_reference_object_id,
        keepout_radius_km=training_cfg.keepout_radius_km,
        goal_radius_km=training_cfg.goal_radius_km,
        goal_relative_ric_km=training_cfg.goal_relative_ric_km,
        goal_nmt_radial_amplitude_km=training_cfg.goal_nmt_radial_amplitude_km,
        goal_nmt_cross_track_amplitude_km=training_cfg.goal_nmt_cross_track_amplitude_km,
        goal_nmt_cross_track_phase_deg=training_cfg.goal_nmt_cross_track_phase_deg,
        goal_nmt_center_ric_km=training_cfg.goal_nmt_center_ric_km,
        coast_prediction_orbit_fraction=_coast_prediction_orbit_fraction(difficulty),
        fullscreen=True,
    )
    dashboard.push_snapshot(snapshot)
    trainer.record(snapshot)
    score = trainer.score()
    dashboard.draw(
        command_status=_command_status(command_state, control_mode=control_mode),
        coach_hint=trainer.current_hint(),
        mission_state=_mission_state(score),
        mission_metrics=_mission_metrics(training_cfg, score),
        speed_multiple=current_speed_multiple,
        debrief_lines=_score_debrief_lines(score),
    )

    pygame = dashboard.pygame
    dt_s = float(config.scenario.simulator.dt_s)
    wall_step_s = _wall_step_s(dt_s, current_speed_multiple)
    last_step_wall = perf_counter()
    try:
        while (not command_state.quit_requested) and (not dashboard.closed):
            _poll_pygame_input(pygame, command_state, control_mode=control_mode)
            if command_state.quit_requested:
                break
            if command_state.speed_multiplier_change:
                current_speed_multiple = _adjust_speed_multiple(current_speed_multiple, command_state.speed_multiplier_change)
                wall_step_s = _wall_step_s(dt_s, current_speed_multiple)
                command_state.speed_multiplier_change = 0
                last_step_wall = perf_counter()
            if command_state.restart_requested:
                session, _, snapshot = _start_game_attempt(
                    config,
                    command_state=command_state,
                    training_cfg=training_cfg,
                    controlled_object_id=controlled_object_id,
                    attitude_rate_deg_s=attitude_rate_deg_s,
                    control_mode=control_mode,
                    ric_reference_object_id=ric_reference_object_id,
                )
                trainer.clear()
                dashboard.clear()
                dashboard.push_snapshot(snapshot)
                trainer.record(snapshot)
                command_state.restart_requested = False
                command_state.step_requested = False
                command_state.speed_multiplier_change = 0
                command_state.paused = False
                last_step_wall = perf_counter()
            now = perf_counter()
            pre_score = trainer.score()
            mission_decided = bool(pre_score.level_passed or pre_score.level_failed)
            if _game_loop_should_exit(session_done=session.done, score=pre_score):
                break
            if mission_decided:
                command_state.paused = True
            step_due = (not realtime) or (now - last_step_wall >= wall_step_s)
            should_step = (
                ((not command_state.paused and step_due) or bool(command_state.step_requested))
                and not mission_decided
                and not session.done
            )
            if should_step:
                last_step_wall = now
                snapshot = session.step()
                dashboard.push_snapshot(snapshot)
                trainer.record(snapshot)
                command_state.step_requested = False
            score = trainer.score()
            if score.level_passed or score.level_failed:
                command_state.paused = True
            dashboard.draw(
                command_status=_command_status(command_state, control_mode=control_mode),
                coach_hint=trainer.current_hint(),
                mission_state=_mission_state(score),
                mission_metrics=_mission_metrics(training_cfg, score),
                speed_multiple=current_speed_multiple,
                debrief_lines=_score_debrief_lines(score),
            )
            dashboard.tick(60.0)
    finally:
        dashboard.close()
        if training_cfg.enabled:
            print(trainer.debrief_text())


def _start_game_attempt(
    config: SimulationConfig,
    *,
    command_state: KeyboardCommandState,
    training_cfg: RPOTrainingConfig,
    controlled_object_id: str,
    attitude_rate_deg_s: float,
    control_mode: str,
    ric_reference_object_id: str,
) -> tuple[SimulationSession, ManualGameCommandProvider, Any]:
    session = SimulationSession.from_config(config)
    provider = ManualGameCommandProvider(
        command_state=command_state,
        max_accel_km_s2=_max_accel_from_config(config, controlled_object_id),
        attitude_rate_deg_s=attitude_rate_deg_s,
        controlled_object_id=controlled_object_id,
        control_mode=control_mode,
        reference_object_id=ric_reference_object_id,
    )
    session.set_external_intent_provider(controlled_object_id, provider)
    defensive_target_provider = _game_defensive_target_provider(config)
    if defensive_target_provider is not None:
        session.set_external_intent_provider(training_cfg.target_object_id, defensive_target_provider)
    snapshot = session.reset()
    if snapshot is None:
        raise RuntimeError("Game mode requires a single-run scenario.")
    provider.reset_target_to_current(snapshot.truth[controlled_object_id])
    return session, provider, snapshot


def _poll_pygame_input(pygame: Any, state: KeyboardCommandState, *, control_mode: str = "attitude_thrust") -> None:
    ric_mode = str(control_mode or "").strip().lower() in {"ric", "ric_translation", "translation"}
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state.quit_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            state.quit_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            state.restart_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and ric_mode:
            state.paused = not bool(state.paused)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_PERIOD:
            state.step_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            state.speed_multiplier_change += 1
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
            state.speed_multiplier_change -= 1

    keys = pygame.key.get_pressed()
    state.pitch = 0.0
    state.yaw = 0.0
    state.roll = 0.0
    if keys[pygame.K_w]:
        state.pitch += 1.0
    if keys[pygame.K_s]:
        state.pitch -= 1.0
    if keys[pygame.K_d]:
        state.yaw += 1.0
    if keys[pygame.K_a]:
        state.yaw -= 1.0
    if keys[pygame.K_RIGHT]:
        state.roll += 1.0
    if keys[pygame.K_LEFT]:
        state.roll -= 1.0
    state.firing = False if ric_mode else bool(keys[pygame.K_SPACE])


def _mission_state(score: Any) -> str:
    if bool(getattr(score, "level_passed", False)):
        return "passed"
    if bool(getattr(score, "level_failed", False)):
        return "failed"
    return "active"


def _game_loop_should_exit(*, session_done: bool, score: Any) -> bool:
    terminal_score = bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False))
    return bool(session_done) and not terminal_score


def _mission_metrics(config: RPOTrainingConfig, score: Any) -> tuple[str, ...]:
    metrics: list[str] = []
    if config.max_time_s is not None:
        remain = max(float(config.max_time_s) - float(getattr(score, "elapsed_s", 0.0)), 0.0)
        metrics.append(f"Time {remain:4.0f}s")
    if config.max_delta_v_m_s is not None:
        remain = max(float(config.max_delta_v_m_s) - float(getattr(score, "approximate_delta_v_m_s", 0.0)), 0.0)
        metrics.append(f"dV {remain:4.1f} m/s")
    if config.goal_nmt_radial_amplitude_km is None and config.keepout_radius_km is not None:
        final_range = float(getattr(score, "final_range_km", float("nan")))
        margin = final_range - float(config.keepout_radius_km)
        metrics.append(f"KO {_fmt_distance(margin)}")
    if config.goal_nmt_element_tolerance_km is not None:
        tol = float(config.goal_nmt_element_tolerance_km)
        r_err = float(getattr(score, "final_nmt_radial_amplitude_error_km", float("nan")))
        c_err = float(getattr(score, "final_nmt_cross_track_amplitude_error_km", float("nan")))
        metrics.append(f"R amp { _fmt_metric(r_err)}/{tol:.2f} km")
        metrics.append(f"C amp { _fmt_metric(c_err)}/{tol:.2f} km")
    if config.goal_nmt_velocity_tolerance_km_s is not None:
        tol = float(config.goal_nmt_velocity_tolerance_km_s)
        err = float(getattr(score, "final_nmt_drift_velocity_error_km_s", float("nan")))
        metrics.append(f"Drift { _fmt_metric(err, precision=4)}/{tol:.4f}")
    if config.goal_nmt_radial_amplitude_km is None and config.goal_radius_km is not None:
        err = float(getattr(score, "final_goal_error_km", float("nan")))
        metrics.append(f"Goal {_fmt_distance(err)}/{_fmt_distance(float(config.goal_radius_km))}")
    if config.goal_nmt_radial_amplitude_km is None and config.max_goal_speed_km_s is not None:
        speed = float(getattr(score, "final_relative_speed_km_s", float("nan")))
        metrics.append(f"Speed {_fmt_speed(speed)}/{_fmt_speed(float(config.max_goal_speed_km_s))}")
    return tuple(metrics)


def _score_debrief_lines(score: Any) -> tuple[str, ...]:
    if not (bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False))):
        return ()
    lines = [
        f"Scenario      {str(getattr(score, 'scenario_id', '') or '--')}",
        f"Elapsed       {float(getattr(score, 'elapsed_s', float('nan'))):.1f} s",
        f"Closest App   {_fmt_distance(float(getattr(score, 'closest_approach_km', float('nan'))))}",
        f"Final Range   {_fmt_distance(float(getattr(score, 'final_range_km', float('nan'))))}",
        f"Goal Error    {_fmt_distance(float(getattr(score, 'final_goal_error_km', float('nan'))))}",
        f"Final Speed   {_fmt_speed(float(getattr(score, 'final_relative_speed_km_s', float('nan'))))}",
        f"Keepout Time  {float(getattr(score, 'time_inside_keepout_s', 0.0)):.1f} s",
        f"Approx dV     {float(getattr(score, 'approximate_delta_v_m_s', 0.0)):.2f} m/s",
    ]
    for reason in tuple(getattr(score, "pass_fail_reasons", ()) or ())[:3]:
        lines.append(f"Result        {reason}")
    return tuple(lines)


def _fmt_metric(value: float, *, precision: int = 2) -> str:
    if not np.isfinite(float(value)):
        return "--"
    return f"{float(value):.{precision}f}"


def _fmt_distance(value_km: float) -> str:
    if not np.isfinite(float(value_km)):
        return "--"
    if abs(float(value_km)) < 0.1:
        return f"{float(value_km) * 1000.0:.0f} m"
    return f"{float(value_km):.2f} km"


def _fmt_speed(value_km_s: float) -> str:
    if not np.isfinite(float(value_km_s)):
        return "--"
    if abs(float(value_km_s)) < 0.01:
        return f"{float(value_km_s) * 1000.0:.2f} m/s"
    return f"{float(value_km_s):.4f} km/s"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
