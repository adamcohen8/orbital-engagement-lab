from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from sim.api import SimulationConfig, SimulationSession
from sim.config import object_section
from sim.game.arcade import (
    _arcade_mission_metrics,
    _arcade_round_briefing_lines,
    _arcade_round_rng,
    _arcade_round_time_bonus_s,
    _arcade_round_weighted_score,
    _arcade_score,
    _game_arcade_enabled,
    _game_arcade_initial_time_s,
    _game_defensive_target_provider,
    _game_random_direction_defensive_target_provider,
    _new_arcade_seed,
    _score_time_used_s,
)
from sim.game.audio import (
    ARCADE_ROUND_CLEAR_SOUND_PATH,
    _play_game_sound_effect,
    _stop_game_music,
    _sync_game_music,
)
from sim.game.debrief import game_debrief_path, tracker_replay_history, write_game_debrief
from sim.game.defensive_target import DefensiveTargetIntentProvider
from sim.game.manual import KeyboardCommandState, ManualGameCommandProvider
from sim.game.recording import GameFrameRecorder, game_recording_path
from sim.game.session import (
    _attempt_config_for_training_clock,
    _install_chaser_delta_v_limiter,
)
from sim.game.training import RPOTrainingConfig, RPOTrainingTracker
from sim.presets.thrusters import resolve_thruster_max_thrust_n_from_specs

SPEED_MULTIPLIER_OPTIONS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)
MAX_REALTIME_STEPS_PER_FRAME = 12
DASHBOARD_FPS = 60.0
HIGH_SPEED_DASHBOARD_FPS = 30.0
HIGH_SPEED_MANEUVER_THRESHOLD = 50.0
MANEUVER_CONTROL_SPEED = 10.0


@dataclass(frozen=True)
class GameRunResult:
    config_path: Path
    difficulty: str
    level_passed: bool
    arcade_score: int = 0
    arcade_seed: int | None = None
    recording_path: Path | None = None
    debrief_path: Path | None = None


def _max_accel_from_config(config: SimulationConfig, controlled_object_id: str) -> float:
    section = object_section(config.scenario, str(controlled_object_id))
    if section is None:
        raise ValueError(f"Unknown controlled object '{controlled_object_id}'.")
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
    training_cfg = dict(game_cfg.get("training", {}) or {})
    default = "ric_translation" if training_cfg else "attitude_thrust"
    return str(game_cfg.get("control_mode", default) or default).strip().lower()


def _game_controlled_object_id(config: SimulationConfig, default: str = "chaser") -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("controlled_object_id", default) or default)


def _game_difficulty(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("difficulty", "easy") or "easy").strip().lower()


def _game_plot_overlays_in_zoom(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("plot_overlays_in_zoom", True))


def _game_plot_overlays_in_zoom_by_plane(config: SimulationConfig) -> dict[str, bool]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = game_cfg.get("plot_overlays_in_zoom_by_plane", {}) or {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, bool] = {}
    for plane, value in raw.items():
        key = str(plane or "").strip().upper()
        if key in {"RI", "RC", "IC"}:
            parsed[key] = bool(value)
    return parsed


def _positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def _game_plot_axis_scale(config: SimulationConfig) -> dict[str, tuple[float, float]]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = game_cfg.get("plot_axis_scale", {}) or {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, tuple[float, float]] = {}
    for plane, value in raw.items():
        key = str(plane or "").strip().upper()
        if key not in {"RI", "RC", "IC"}:
            continue
        if isinstance(value, dict):
            pair = (value.get("x", 1.0), value.get("y", 1.0))
        else:
            try:
                pair = tuple(value)  # type: ignore[arg-type]
            except TypeError:
                continue
            if len(pair) != 2:
                continue
        try:
            x_scale = float(pair[0])
            y_scale = float(pair[1])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x_scale) or x_scale <= 0.0:
            x_scale = 1.0
        if not np.isfinite(y_scale) or y_scale <= 0.0:
            y_scale = 1.0
        parsed[key] = (x_scale, y_scale)
    return parsed


def _game_plot_fixed_axis_half_span_km(config: SimulationConfig) -> dict[str, tuple[float | None, float | None]]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = game_cfg.get("plot_fixed_axis_half_span_km", {}) or {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, tuple[float | None, float | None]] = {}
    for plane, value in raw.items():
        key = str(plane or "").strip().upper()
        if key not in {"RI", "RC", "IC"}:
            continue
        if isinstance(value, dict):
            pair = (value.get("x"), value.get("y"))
        else:
            try:
                pair = tuple(value)  # type: ignore[arg-type]
            except TypeError:
                continue
            if len(pair) != 2:
                continue
        x_span = _positive_float_or_none(pair[0])
        y_span = _positive_float_or_none(pair[1])
        if x_span is not None or y_span is not None:
            parsed[key] = (x_span, y_span)
    return parsed


def _game_plot_equal_axis_scale_planes(config: SimulationConfig) -> tuple[str, ...]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = game_cfg.get("plot_equal_axis_scale_planes", ())
    if isinstance(raw, str):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError:
            return ()
    planes: list[str] = []
    for value in values:
        plane = str(value or "").strip().upper()
        if plane in {"RI", "RC", "IC"} and plane not in planes:
            planes.append(plane)
    return tuple(planes)


def _game_proximity_ring_plot_planes(config: SimulationConfig) -> tuple[str, ...]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = game_cfg.get("proximity_ring_plot_planes", ("RI", "RC", "IC"))
    if isinstance(raw, str):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError:
            return ("RI", "RC", "IC")
    planes: list[str] = []
    for value in values:
        plane = str(value or "").strip().upper()
        if plane in {"RI", "RC", "IC"} and plane not in planes:
            planes.append(plane)
    return tuple(planes) if planes else ("RI", "RC", "IC")


def _game_camera_mode(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("camera_mode", "reference") or "reference")


def _game_show_target_hcw_path(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("show_target_hcw_path", False))


def _game_coast_chaser_after_delta_v_budget(config: RPOTrainingConfig) -> bool:
    return bool(getattr(config, "coast_chaser_after_delta_v_budget", False))


def _game_ric_reference_object_id(config: SimulationConfig, default: str) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("ric_reference_object_id", default) or default)


def _dashboard_object_ids(training_cfg: RPOTrainingConfig, anim_cfg: dict[str, Any]) -> tuple[str, str]:
    return (
        str(anim_cfg.get("battlespace_dashboard_target_object_id", training_cfg.target_object_id)),
        str(anim_cfg.get("battlespace_dashboard_chaser_object_id", training_cfg.chaser_object_id)),
    )


def _training_briefing_lines(
    config: SimulationConfig, training_cfg: RPOTrainingConfig, *, difficulty: str
) -> tuple[str, ...]:
    if not training_cfg.enabled:
        return ()
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = dict(game_cfg.get("training", {}) or {})
    lines = [
        str(training_cfg.scenario_id or config.scenario.scenario_name or "RPO training"),
        f"Assists: {str(difficulty or 'easy').title()}",
    ]
    if training_cfg.learning_goal:
        lines.append(f"Objective: {training_cfg.learning_goal}")
    player_brief = str(raw.get("player_brief", "") or "").strip()
    if player_brief:
        lines.append(f"Plan: {player_brief}")
    pass_criteria = _as_str_tuple(raw.get("pass_criteria"))
    for item in pass_criteria[:4]:
        lines.append(f"Gate: {item}")
    return tuple(lines)


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


def _has_maneuver_input(state: KeyboardCommandState, *, control_mode: str = "attitude_thrust") -> bool:
    axes_active = any(abs(float(value)) > 1.0e-9 for value in (state.pitch, state.yaw, state.roll))
    if str(control_mode or "").strip().lower() in {"ric", "ric_translation", "translation"}:
        return bool(axes_active and float(state.throttle) > 0.0)
    return bool(axes_active or (state.firing and float(state.throttle) > 0.0))


def _speed_after_maneuver_input(
    speed_multiple: float,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
) -> float:
    speed = _coerce_speed_multiple(speed_multiple)
    if speed >= HIGH_SPEED_MANEUVER_THRESHOLD and _has_maneuver_input(state, control_mode=control_mode):
        return MANEUVER_CONTROL_SPEED
    return speed


def _dashboard_fps_for_speed(speed_multiple: float, *, recording: bool = False) -> float:
    if bool(recording):
        return DASHBOARD_FPS
    if float(speed_multiple) >= 100.0:
        return HIGH_SPEED_DASHBOARD_FPS
    if float(speed_multiple) >= 50.0:
        return 45.0
    return DASHBOARD_FPS


def _realtime_steps_due(
    *,
    now_s: float,
    last_step_wall_s: float,
    wall_step_s: float,
    max_steps: int = MAX_REALTIME_STEPS_PER_FRAME,
) -> tuple[int, float]:
    wall_step = float(max(wall_step_s, 1.0e-9))
    elapsed = max(float(now_s) - float(last_step_wall_s), 0.0)
    due = int(elapsed // wall_step)
    if due <= 0:
        return 0, float(last_step_wall_s)
    cap = max(int(max_steps), 1)
    steps = min(due, cap)
    if due > cap:
        return steps, float(now_s)
    return steps, float(last_step_wall_s) + float(steps) * wall_step


def _command_status(state: KeyboardCommandState, *, control_mode: str = "attitude_thrust") -> str:
    if control_mode in {"ric", "ric_translation", "translation"}:
        sim_state = "PAUSED" if state.paused else "RUNNING"
        return (
            "W/S radial +/-R  A/D in-track +/-I  Left/Right cross-track +/-C  M music\n"
            "Use small pulses, then coast and watch the target-centered RIC motion.\n"
            f"{sim_state}  R={state.pitch:+.0f} I={state.yaw:+.0f} C={state.roll:+.0f} throttle={state.throttle:.2f}"
        )
    burn = "FIRE" if state.firing else "coast"
    return (
        "W/S pitch  A/D yaw  Left/Right roll  Space fire  M music  R reset  Esc quit\n"
        "Keys work in the figure window or this terminal; terminal input is pulse/repeat based.\n"
        f"pitch={state.pitch:+.0f} yaw={state.yaw:+.0f} roll={state.roll:+.0f} thrust={burn}"
    )


def run_game_mode(
    config_path: str | Path,
    *,
    controlled_object_id: str | None = None,
    attitude_rate_deg_s: float = 45.0,
    realtime: bool = True,
    speed_multiple: float = 1.0,
    difficulty_override: str | None = None,
    music_enabled: bool = True,
    record_video: bool = False,
    recording_output_dir: str | Path | None = None,
    recording_fps: float = DASHBOARD_FPS,
    arcade_seed: int | None = None,
    debrief_output_dir: str | Path | None = None,
) -> GameRunResult:
    from sim.game.pygame_dashboard import PygameRPODashboard

    config = SimulationConfig.from_yaml(config_path)
    controlled_object_id = _game_controlled_object_id(config, default=controlled_object_id or "chaser")
    control_mode = _game_control_mode(config)
    difficulty = str(difficulty_override or _game_difficulty(config)).strip().lower()
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    arcade_enabled = _game_arcade_enabled(config)
    arcade_seed_value = _new_arcade_seed() if arcade_enabled and arcade_seed is None else arcade_seed
    arcade_round_index = 1
    arcade_total_score = 0
    arcade_remaining_time_s = _game_arcade_initial_time_s(config, training_cfg) if arcade_enabled else None
    if arcade_enabled:
        training_cfg = replace(training_cfg, max_time_s=arcade_remaining_time_s)
    ric_reference_object_id = _game_ric_reference_object_id(config, training_cfg.target_object_id)
    current_speed_multiple = _coerce_speed_multiple(speed_multiple)
    trainer = RPOTrainingTracker(training_cfg)
    command_state = KeyboardCommandState()
    command_state.paused = bool(training_cfg.enabled)
    briefing_open = bool(training_cfg.enabled)
    briefing_lines = _training_briefing_lines(config, training_cfg, difficulty=difficulty)
    session, _, snapshot = _start_game_attempt(
        config,
        command_state=command_state,
        training_cfg=training_cfg,
        controlled_object_id=controlled_object_id,
        attitude_rate_deg_s=attitude_rate_deg_s,
        control_mode=control_mode,
        ric_reference_object_id=ric_reference_object_id,
        defensive_target_provider=(
            _game_random_direction_defensive_target_provider(
                config,
                rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
            )
            if arcade_enabled and arcade_seed_value is not None
            else None
        ),
    )

    anim_cfg = dict(config.scenario.outputs.animations or {})
    dashboard_target_id, dashboard_chaser_id = _dashboard_object_ids(training_cfg, anim_cfg)
    dashboard = PygameRPODashboard(
        target_object_id=dashboard_target_id,
        chaser_object_id=dashboard_chaser_id,
        reference_object_id=ric_reference_object_id,
        keepout_radius_km=training_cfg.keepout_radius_km,
        goal_range_km=training_cfg.goal_range_km,
        goal_range_tolerance_km=training_cfg.goal_range_tolerance_km,
        goal_radius_km=training_cfg.goal_radius_km,
        goal_relative_ric_km=training_cfg.goal_relative_ric_km,
        goal_nmt_radial_amplitude_km=training_cfg.goal_nmt_radial_amplitude_km,
        goal_nmt_cross_track_amplitude_km=training_cfg.goal_nmt_cross_track_amplitude_km,
        goal_nmt_cross_track_phase_deg=training_cfg.goal_nmt_cross_track_phase_deg,
        goal_nmt_center_ric_km=training_cfg.goal_nmt_center_ric_km,
        coast_prediction_orbit_fraction=_coast_prediction_orbit_fraction(difficulty),
        forbidden_regions=training_cfg.forbidden_regions,
        approach_gates=training_cfg.approach_gates,
        inspection_gates=training_cfg.inspection_gates,
        plot_overlays_in_zoom=_game_plot_overlays_in_zoom(config),
        plot_overlays_in_zoom_by_plane=_game_plot_overlays_in_zoom_by_plane(config),
        plot_axis_scale=_game_plot_axis_scale(config),
        plot_fixed_axis_half_span_km=_game_plot_fixed_axis_half_span_km(config),
        plot_equal_axis_scale_planes=_game_plot_equal_axis_scale_planes(config),
        proximity_ring_plot_planes=_game_proximity_ring_plot_planes(config),
        camera_mode=_game_camera_mode(config),
        show_target_coast_prediction=_game_show_target_hcw_path(config),
        fullscreen=True,
    )
    recording_attempt = 1
    recording_path: Path | None = None
    debrief_path: Path | None = None
    recorder: GameFrameRecorder | None = None
    try:
        recorder = _start_game_recorder(
            enabled=record_video,
            config=config,
            difficulty=difficulty,
            attempt_index=recording_attempt,
            output_dir=recording_output_dir,
            fps=recording_fps,
        )
        dashboard.push_snapshot(snapshot)
        trainer.record(snapshot)
        score = trainer.score()
        dashboard.draw(
            command_status=_command_status(command_state, control_mode=control_mode),
            coach_hint=trainer.current_hint(),
            mission_state=_mission_state(score),
            mission_metrics=_arcade_mission_metrics(
                _mission_metrics(training_cfg, score),
                enabled=arcade_enabled,
                round_index=arcade_round_index,
                total_score=arcade_total_score,
            ),
            objective_checklist=_mission_checklist(training_cfg, score),
            speed_multiple=current_speed_multiple,
            briefing_lines=briefing_lines if briefing_open else (),
            debrief_lines=_score_debrief_lines(score, config=training_cfg, difficulty=difficulty),
        )
        _capture_recording_frame(recorder, dashboard)

        pygame = dashboard.pygame
        active_game_music_path: Path | None = _sync_game_music(
            pygame,
            score,
            training_cfg=training_cfg,
            music_enabled=music_enabled,
            active_path=None,
        )
        dt_s = float(config.scenario.simulator.dt_s)
        wall_step_s = _wall_step_s(dt_s, current_speed_multiple)
        last_step_wall = perf_counter()
        while (not command_state.quit_requested) and (not dashboard.closed):
            _poll_pygame_input(pygame, command_state, control_mode=control_mode)
            if command_state.quit_requested:
                break
            if briefing_open and not command_state.paused:
                briefing_open = False
                last_step_wall = perf_counter()
            if command_state.speed_multiplier_change:
                previous_speed_multiple = current_speed_multiple
                current_speed_multiple = _adjust_speed_multiple(
                    current_speed_multiple, command_state.speed_multiplier_change
                )
                if not np.isclose(current_speed_multiple, previous_speed_multiple):
                    trainer.record_speed_multiplier_change()
                wall_step_s = _wall_step_s(dt_s, current_speed_multiple)
                command_state.speed_multiplier_change = 0
                last_step_wall = perf_counter()
            maneuver_speed_multiple = _speed_after_maneuver_input(
                current_speed_multiple,
                command_state,
                control_mode=control_mode,
            )
            if not np.isclose(maneuver_speed_multiple, current_speed_multiple):
                current_speed_multiple = maneuver_speed_multiple
                wall_step_s = _wall_step_s(dt_s, current_speed_multiple)
                last_step_wall = perf_counter()
            if command_state.music_toggle_requested:
                music_enabled = not bool(music_enabled)
                command_state.music_toggle_requested = False
                active_game_music_path = _sync_game_music(
                    pygame,
                    trainer.score(),
                    training_cfg=training_cfg,
                    music_enabled=music_enabled,
                    active_path=active_game_music_path,
                )
            if command_state.restart_requested:
                if recorder is not None:
                    recorder.discard()
                _stop_game_music(pygame)
                active_game_music_path = None
                if arcade_enabled:
                    arcade_round_index = 1
                    arcade_total_score = 0
                    arcade_remaining_time_s = _game_arcade_initial_time_s(config, training_cfg)
                    training_cfg = replace(training_cfg, max_time_s=arcade_remaining_time_s)
                    trainer = RPOTrainingTracker(training_cfg)
                recording_attempt += 1
                recorder = _start_game_recorder(
                    enabled=record_video,
                    config=config,
                    difficulty=difficulty,
                    attempt_index=recording_attempt,
                    output_dir=recording_output_dir,
                    fps=recording_fps,
                )
                recording_path = None
                debrief_path = None
                session, _, snapshot = _start_game_attempt(
                    config,
                    command_state=command_state,
                    training_cfg=training_cfg,
                    controlled_object_id=controlled_object_id,
                    attitude_rate_deg_s=attitude_rate_deg_s,
                    control_mode=control_mode,
                    ric_reference_object_id=ric_reference_object_id,
                    defensive_target_provider=(
                        _game_random_direction_defensive_target_provider(
                        config,
                            rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
                        )
                        if arcade_enabled and arcade_seed_value is not None
                        else None
                    ),
                )
                trainer.clear()
                dashboard.clear()
                dashboard.push_snapshot(snapshot)
                trainer.record(snapshot)
                command_state.restart_requested = False
                command_state.step_requested = False
                command_state.speed_multiplier_change = 0
                command_state.music_toggle_requested = False
                command_state.paused = bool(training_cfg.enabled)
                briefing_open = bool(training_cfg.enabled)
                dt_s = float(config.scenario.simulator.dt_s)
                wall_step_s = _wall_step_s(dt_s, current_speed_multiple)
                last_step_wall = perf_counter()
            now = perf_counter()
            pre_score = trainer.score()
            mission_decided = bool(pre_score.level_passed or pre_score.level_failed)
            if _game_loop_should_exit(session_done=session.done, score=pre_score):
                break
            if mission_decided:
                command_state.paused = True
            if command_state.paused and not command_state.step_requested:
                last_step_wall = now
            steps_to_run = 0
            if not mission_decided and not session.done:
                if command_state.step_requested:
                    steps_to_run = 1
                    last_step_wall = now
                elif not command_state.paused:
                    if realtime:
                        steps_to_run, last_step_wall = _realtime_steps_due(
                            now_s=now,
                            last_step_wall_s=last_step_wall,
                            wall_step_s=wall_step_s,
                        )
                    else:
                        steps_to_run = 1
                        last_step_wall = now
            score = _step_game_attempt(
                session=session,
                dashboard=dashboard,
                trainer=trainer,
                steps_to_run=steps_to_run,
                initial_score=pre_score,
            )
            command_state.step_requested = False
            if score.level_passed or score.level_failed:
                command_state.paused = True
            if arcade_enabled and bool(getattr(score, "level_passed", False)):
                if music_enabled:
                    _play_game_sound_effect(pygame, ARCADE_ROUND_CLEAR_SOUND_PATH, volume=0.74)
                round_score = _arcade_round_weighted_score(
                    training_cfg,
                    score,
                    difficulty=difficulty,
                    round_index=arcade_round_index,
                )
                arcade_total_score += int(round_score)
                time_used = _score_time_used_s(score)
                round_bonus_s = _arcade_round_time_bonus_s(config, training_cfg, score)
                assert arcade_remaining_time_s is not None
                arcade_remaining_time_s = (
                    max(float(arcade_remaining_time_s) - time_used, 0.0)
                    + round_bonus_s
                )
                cleared_round_index = arcade_round_index
                arcade_round_index += 1
                training_cfg = replace(training_cfg, max_time_s=arcade_remaining_time_s)
                trainer = RPOTrainingTracker(training_cfg)
                session, _, snapshot = _start_game_attempt(
                    config,
                    command_state=command_state,
                    training_cfg=training_cfg,
                    controlled_object_id=controlled_object_id,
                    attitude_rate_deg_s=attitude_rate_deg_s,
                    control_mode=control_mode,
                    ric_reference_object_id=ric_reference_object_id,
                    defensive_target_provider=_game_random_direction_defensive_target_provider(
                        config,
                        rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
                    ),
                )
                dashboard.clear()
                dashboard.push_snapshot(snapshot)
                trainer.record(snapshot)
                score = trainer.score()
                briefing_lines = _arcade_round_briefing_lines(
                    cleared_round_index=cleared_round_index,
                    next_round_index=arcade_round_index,
                    round_score=int(round_score),
                    total_score=arcade_total_score,
                    time_used_s=time_used,
                    bonus_time_s=round_bonus_s,
                    next_time_budget_s=arcade_remaining_time_s,
                )
                command_state.paused = True
                command_state.step_requested = False
                command_state.restart_requested = False
                command_state.speed_multiplier_change = 0
                command_state.music_toggle_requested = False
                briefing_open = True
                active_game_music_path = None
                last_step_wall = perf_counter()
            active_game_music_path = _sync_game_music(
                pygame,
                score,
                training_cfg=training_cfg,
                music_enabled=music_enabled,
                active_path=active_game_music_path,
            )
            dashboard.draw(
                command_status=_command_status(command_state, control_mode=control_mode),
                coach_hint=trainer.current_hint(),
                mission_state=_mission_state(score),
                mission_metrics=_arcade_mission_metrics(
                    _mission_metrics(training_cfg, score),
                    enabled=arcade_enabled,
                    round_index=arcade_round_index,
                    total_score=arcade_total_score,
                ),
                objective_checklist=_mission_checklist(training_cfg, score),
                speed_multiple=current_speed_multiple,
                briefing_lines=briefing_lines if briefing_open else (),
                debrief_lines=_score_debrief_lines(score, config=training_cfg, difficulty=difficulty),
            )
            _capture_recording_frame(recorder, dashboard)
            if recorder is not None and (score.level_passed or score.level_failed) and not recorder.saved:
                recording_path = recorder.finish()
                if recording_path is not None:
                    print(f"Saved game recording: {recording_path}")
            if (score.level_passed or score.level_failed) and debrief_path is None:
                debrief_path = write_game_debrief(
                    game_debrief_path(
                        scenario_id=training_cfg.scenario_id,
                        difficulty=difficulty,
                        attempt_index=recording_attempt,
                        output_dir=debrief_output_dir,
                    ),
                    config=training_cfg,
                    score=score,
                    difficulty=difficulty,
                    objective_checklist=_mission_checklist(training_cfg, score),
                    arcade_score=arcade_total_score if arcade_enabled else _arcade_score(
                        training_cfg, score, difficulty=difficulty
                    ),
                    arcade_seed=arcade_seed_value if arcade_enabled else None,
                    arcade_round_index=arcade_round_index if arcade_enabled else None,
                    recording_path=recording_path,
                    replay_history=tracker_replay_history(trainer),
                )
                print(f"Saved game debrief: {debrief_path}")
            dashboard.tick(_dashboard_fps_for_speed(current_speed_multiple, recording=recorder is not None))
    finally:
        if recorder is not None and not recorder.saved:
            recorder.discard()
        _stop_game_music(getattr(dashboard, "pygame", None))
        dashboard.close()
        if training_cfg.enabled:
            print(trainer.debrief_text())
    final_arcade_score = arcade_total_score
    if arcade_enabled and bool(getattr(score, "level_passed", False)):
        final_arcade_score += _arcade_round_weighted_score(
            training_cfg,
            score,
            difficulty=difficulty,
            round_index=arcade_round_index,
        )
    return GameRunResult(
        config_path=Path(config_path),
        difficulty=difficulty,
        level_passed=bool(score.level_passed),
        arcade_score=final_arcade_score if arcade_enabled else _arcade_score(training_cfg, score, difficulty=difficulty),
        arcade_seed=arcade_seed_value if arcade_enabled else None,
        recording_path=recording_path,
        debrief_path=debrief_path,
    )


def _start_game_recorder(
    *,
    enabled: bool,
    config: SimulationConfig,
    difficulty: str,
    attempt_index: int,
    output_dir: str | Path | None,
    fps: float,
) -> GameFrameRecorder | None:
    if not bool(enabled):
        return None
    path = game_recording_path(
        scenario_name=str(config.scenario.scenario_name or "game"),
        difficulty=difficulty,
        attempt_index=attempt_index,
        output_dir=output_dir,
    )
    return GameFrameRecorder.start(path, fps=fps)


def _capture_recording_frame(recorder: GameFrameRecorder | None, dashboard: Any) -> None:
    if recorder is None or recorder.saved:
        return
    recorder.capture_surface(dashboard.screen)


def _step_game_attempt(
    *,
    session: SimulationSession,
    dashboard: Any,
    trainer: RPOTrainingTracker,
    steps_to_run: int,
    initial_score: Any | None = None,
) -> Any:
    score = trainer.score() if initial_score is None else initial_score
    for _ in range(max(int(steps_to_run), 0)):
        if session.done:
            break
        snapshot = session.step()
        dashboard.push_snapshot(snapshot)
        trainer.record(snapshot)
        score = trainer.score()
        if bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False)):
            break
    return score


def _start_game_attempt(
    config: SimulationConfig,
    *,
    command_state: KeyboardCommandState,
    training_cfg: RPOTrainingConfig,
    controlled_object_id: str,
    attitude_rate_deg_s: float,
    control_mode: str,
    ric_reference_object_id: str,
    defensive_target_provider: DefensiveTargetIntentProvider | None = None,
) -> tuple[SimulationSession, ManualGameCommandProvider, Any]:
    session = SimulationSession.from_config(_attempt_config_for_training_clock(config, training_cfg))
    provider = ManualGameCommandProvider(
        command_state=command_state,
        max_accel_km_s2=_max_accel_from_config(config, controlled_object_id),
        attitude_rate_deg_s=attitude_rate_deg_s,
        controlled_object_id=controlled_object_id,
        control_mode=control_mode,
        reference_object_id=ric_reference_object_id,
    )
    session.set_external_intent_provider(controlled_object_id, provider)
    target_provider = defensive_target_provider
    if target_provider is None:
        target_provider = _game_defensive_target_provider(config)
    if target_provider is not None:
        session.set_external_intent_provider(training_cfg.target_object_id, target_provider)
    snapshot = session.reset()
    if snapshot is None:
        raise RuntimeError("Game mode requires a single-run scenario.")
    _install_chaser_delta_v_limiter(session, training_cfg=training_cfg, dt_s=float(config.scenario.simulator.dt_s))
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
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            state.music_toggle_requested = True
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
        elapsed = float(getattr(score, "elapsed_s", 0.0))
        if config.survival_goal:
            ratio = elapsed / max(float(config.max_time_s), 1.0e-9)
            metrics.append(
                f"{_status_tag(elapsed >= float(config.max_time_s), ratio >= 0.5)} Survive {elapsed:4.0f}/{float(config.max_time_s):.0f}s"
            )
        else:
            remain = max(float(config.max_time_s) - elapsed, 0.0)
            ratio = remain / max(float(config.max_time_s), 1.0e-9)
            metrics.append(f"{_status_tag(remain > 0.0, ratio > 0.2)} Time {remain:4.0f}s")
    if config.max_delta_v_m_s is not None:
        remain = max(float(config.max_delta_v_m_s) - float(getattr(score, "approximate_delta_v_m_s", 0.0)), 0.0)
        ratio = remain / max(float(config.max_delta_v_m_s), 1.0e-9)
        if config.fail_on_delta_v_budget:
            tag = _status_tag(remain > 0.0, ratio > 0.2)
        else:
            tag = "OK"
        suffix = " coast" if not config.fail_on_delta_v_budget and remain <= 0.0 else ""
        metrics.append(f"{tag} Chaser dV {remain:5.2f} m/s{suffix}")
    if config.max_target_delta_v_m_s is not None:
        remain = max(float(config.max_target_delta_v_m_s) - float(getattr(score, "target_delta_v_m_s", 0.0)), 0.0)
        ratio = remain / max(float(config.max_target_delta_v_m_s), 1.0e-9)
        metrics.append(f"{_status_tag(remain > 0.0, ratio > 0.2)} Target dV {remain:5.2f} m/s")
    if config.required_burn_axes:
        satisfied = set(getattr(score, "burn_axes_satisfied", ()))
        parts = [
            f"{_burn_axis_short_label(axis)}{'+' if axis in satisfied else '-'}" for axis in config.required_burn_axes
        ]
        all_done = len(satisfied.intersection(config.required_burn_axes)) >= len(config.required_burn_axes)
        metrics.append(f"{'OK' if all_done else 'WARN'} Burns {'/'.join(parts)}")
    if config.required_phase_burns:
        satisfied = set(getattr(score, "phase_burns_satisfied", ()))
        done = len(satisfied.intersection(burn.name for burn in config.required_phase_burns))
        total = len(config.required_phase_burns)
        metrics.append(f"{'OK' if done >= total else 'WARN'} Phase {done}/{total}")
    if config.require_speed_multiplier_change:
        changed = bool(getattr(score, "speed_multiplier_changed", False))
        metrics.append(f"{'OK' if changed else 'WARN'} Speed x")
    if config.goal_nmt_element_tolerance_km is not None:
        tol = float(config.goal_nmt_element_tolerance_km)
        r_err = float(getattr(score, "final_nmt_radial_amplitude_error_km", float("nan")))
        c_err = float(getattr(score, "final_nmt_cross_track_amplitude_error_km", float("nan")))
        metrics.append(f"{_status_tag(r_err <= tol, r_err <= 0.75 * tol)} R amp {_fmt_metric(r_err)}/{tol:.2f} km")
        metrics.append(f"{_status_tag(c_err <= tol, c_err <= 0.75 * tol)} C amp {_fmt_metric(c_err)}/{tol:.2f} km")
    if config.goal_nmt_velocity_tolerance_km_s is not None:
        tol = float(config.goal_nmt_velocity_tolerance_km_s)
        err = float(getattr(score, "final_nmt_drift_velocity_error_km_s", float("nan")))
        metrics.append(f"{_status_tag(err <= tol, err <= 0.75 * tol)} Drift {_fmt_metric(err, precision=4)}/{tol:.4f}")
    if config.goal_nmt_radial_amplitude_km is None and config.goal_range_km is not None:
        err = float(getattr(score, "final_goal_error_km", float("nan")))
        tol = config.goal_range_tolerance_km
        if tol is not None:
            tol_float = float(tol)
            metrics.append(
                f"{_status_tag(err <= tol_float, err <= 0.75 * tol_float)} Range {_fmt_distance(err)}/{_fmt_distance(tol_float)}"
            )
        else:
            final_range = float(getattr(score, "final_range_km", float("nan")))
            target_range = float(config.goal_range_km)
            inside_range = final_range <= target_range
            metrics.append(
                f"{_status_tag(inside_range, inside_range)} Range {_fmt_distance(final_range)}/{_fmt_distance(target_range)}"
            )
    elif config.goal_nmt_radial_amplitude_km is None and config.goal_radius_km is not None:
        err = float(getattr(score, "final_goal_error_km", float("nan")))
        tol = float(config.goal_radius_km)
        metrics.append(f"{_status_tag(err <= tol, err <= 0.75 * tol)} Goal {_fmt_distance(err)}/{_fmt_distance(tol)}")
    if config.goal_nmt_radial_amplitude_km is None and config.max_goal_speed_km_s is not None:
        speed = float(getattr(score, "final_relative_speed_km_s", float("nan")))
        tol = float(config.max_goal_speed_km_s)
        metrics.append(f"{_status_tag(speed <= tol, speed <= 0.75 * tol)} Speed {_fmt_speed(speed)}/{_fmt_speed(tol)}")
    if config.hard_speed_limit_radius_km is not None and config.hard_speed_limit_km_s is not None:
        violated = bool(getattr(score, "hard_speed_limit_violation", False))
        metrics.append(
            f"{_status_tag(not violated, not violated)} Prox V <= {_fmt_speed(float(config.hard_speed_limit_km_s))}"
        )
    if config.goal_nmt_radial_amplitude_km is None and config.keepout_radius_km is not None:
        final_range = float(getattr(score, "final_range_km", float("nan")))
        margin = final_range - float(config.keepout_radius_km)
        metrics.append(f"{_status_tag(margin >= 0.0, margin > 0.1)} KO {_fmt_distance(margin)}")
    if config.forbidden_regions:
        clear = not bool(getattr(score, "forbidden_region_violation", False))
        metrics.append(f"{_status_tag(clear, clear)} FR {'clear' if clear else 'violated'}")
    if config.inspection_gates:
        total = int(getattr(score, "inspection_gates_total", len(config.inspection_gates)))
        satisfied = int(getattr(score, "inspection_gates_satisfied", 0))
        tag = "OK" if satisfied >= total else "WARN"
        metrics.append(f"{tag} Inspect {satisfied}/{total}")
    if config.approach_gates:
        total = int(getattr(score, "approach_gates_total", len(config.approach_gates)))
        satisfied = int(getattr(score, "approach_gates_satisfied", 0))
        required = any(gate.required for gate in config.approach_gates)
        if not required and not bool(getattr(score, "approach_gate_violation", False)):
            return tuple(metrics)
        if bool(getattr(score, "approach_gate_violation", False)):
            tag = "FAIL"
        elif satisfied >= total:
            tag = "OK"
        else:
            tag = "WARN"
        metrics.append(f"{tag} Gates {satisfied}/{total}")
    return tuple(metrics)


def _mission_checklist(config: RPOTrainingConfig, score: Any) -> tuple[str, ...]:
    checklist: list[str] = []
    if config.required_burn_axes:
        satisfied = set(getattr(score, "burn_axes_satisfied", ()))
        for axis in config.required_burn_axes:
            checklist.append(f"{'OK' if axis in satisfied else 'WARN'} {_burn_axis_display_label(axis)} burn")
    if config.required_phase_burns:
        satisfied = set(getattr(score, "phase_burns_satisfied", ()))
        for burn in config.required_phase_burns:
            checklist.append(f"{'OK' if burn.name in satisfied else 'WARN'} {burn.label}")
    if config.require_speed_multiplier_change:
        changed = bool(getattr(score, "speed_multiplier_changed", False))
        checklist.append(f"{'OK' if changed else 'WARN'} Change speed")
    if config.inspection_gates:
        total = int(getattr(score, "inspection_gates_total", len(config.inspection_gates)))
        satisfied = int(getattr(score, "inspection_gates_satisfied", 0))
        checklist.append(f"{'OK' if satisfied >= total else 'WARN'} Inspect gates {satisfied}/{total}")
    if config.survival_goal and config.max_time_s is not None:
        elapsed = float(getattr(score, "elapsed_s", 0.0))
        checklist.append(f"{'OK' if elapsed >= float(config.max_time_s) else 'WARN'} Survive timer")
    elif config.goal_range_km is not None:
        final_range = float(getattr(score, "final_range_km", float("nan")))
        target_range = float(config.goal_range_km)
        checklist.append(f"{'OK' if final_range <= target_range else 'WARN'} Reach range")
    elif config.goal_radius_km is not None:
        err = float(getattr(score, "final_goal_error_km", float("nan")))
        tol = float(config.goal_radius_km)
        checklist.append(f"{'OK' if err <= tol else 'WARN'} Reach goal")
    elif config.goal_nmt_radial_amplitude_km is not None:
        passed = bool(getattr(score, "level_passed", False))
        checklist.append(f"{'OK' if passed else 'WARN'} Match NMT")
    if config.keepout_radius_km is not None:
        clear = not bool(getattr(score, "keepout_violation", False))
        checklist.append(f"{'OK' if clear else 'FAIL'} Keepout clear")
    if config.max_delta_v_m_s is not None and config.fail_on_delta_v_budget:
        used = float(getattr(score, "approximate_delta_v_m_s", 0.0))
        checklist.append(f"{'OK' if used <= float(config.max_delta_v_m_s) else 'FAIL'} Chaser dV")
    if config.max_target_delta_v_m_s is not None:
        used = float(getattr(score, "target_delta_v_m_s", 0.0))
        checklist.append(f"{'OK' if used <= float(config.max_target_delta_v_m_s) else 'FAIL'} Target dV")
    return tuple(checklist[:5])


def _status_tag(ok: bool, strong: bool) -> str:
    if not bool(ok):
        return "FAIL"
    if not bool(strong):
        return "WARN"
    return "OK"


def _burn_axis_short_label(axis: str) -> str:
    labels = {"radial": "R", "in_track": "I", "cross_track": "C"}
    return labels.get(str(axis), str(axis)[:1].upper())


def _burn_axis_display_label(axis: str) -> str:
    labels = {"radial": "Radial", "in_track": "In-track", "cross_track": "Cross-track"}
    return labels.get(str(axis), str(axis).replace("_", " ").title())


def _score_debrief_lines(
    score: Any,
    *,
    config: RPOTrainingConfig | None = None,
    difficulty: str = "easy",
) -> tuple[str, ...]:
    if not (bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False))):
        return ()
    arcade_score = _arcade_score(config, score, difficulty=difficulty) if config is not None else 0
    lines = [
        f"Scenario      {str(getattr(score, 'scenario_id', '') or '--')}",
        f"Score         {arcade_score:,}" if arcade_score > 0 else "",
        f"Elapsed       {float(getattr(score, 'elapsed_s', float('nan'))):.1f} s",
        f"Closest App   {_fmt_distance(float(getattr(score, 'closest_approach_km', float('nan'))))}",
        f"Final Range   {_fmt_distance(float(getattr(score, 'final_range_km', float('nan'))))}",
        f"Goal Error    {_fmt_distance(float(getattr(score, 'final_goal_error_km', float('nan'))))}",
        f"Final Speed   {_fmt_speed(float(getattr(score, 'final_relative_speed_km_s', float('nan'))))}",
        f"Keepout Time  {float(getattr(score, 'time_inside_keepout_s', 0.0)):.1f} s",
        f"Approx dV     {float(getattr(score, 'approximate_delta_v_m_s', 0.0)):.2f} m/s",
        f"Target dV     {float(getattr(score, 'target_delta_v_m_s', 0.0)):.2f} m/s",
    ]
    lines = [line for line in lines if line]
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


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)
