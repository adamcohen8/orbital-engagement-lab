# ruff: noqa: F401,F403,F405,I001
from .runner_common import *
from .runner_models import *
from .runner_config import *
from .tutorial_runtime import *
from .recording_runtime import *
from sim.flight_software import GamePilotInputProfile, GamePilotMode
from sim.game.fsw_inputs import (
    GameOperatorController,
    GameOperatorInputAdapter,
    GamePilotInputAdapter,
)
from sim.game.operator import OPERATOR_IMPULSE_DURATION_S


def _step_game_attempt(
    *,
    session: GamePhysicsSession,
    dashboard: Any,
    trainer: RPOTrainingTracker,
    steps_to_run: int,
    initial_score: Any | None = None,
    dt_s: float | None = None,
    max_step_dt_s: float | None = None,
    control_telemetry_provider: Any | None = None,
    operator_command_provider: Any | None = None,
    operator_burn_transition_callback: Any | None = None,
) -> Any:
    score = trainer.score() if initial_score is None else initial_score
    for _ in range(max(int(steps_to_run), 0)):
        if dt_s is None:
            step_dts: tuple[float | None, ...] = (None,)
        else:
            current_time_s = _game_attempt_current_time_s(session, dashboard)
            step_dts = _split_game_step_dt(
                float(dt_s),
                max_step_dt_s=max_step_dt_s,
                current_time_s=current_time_s,
                operator_command_provider=operator_command_provider,
            )
        for step_dt in step_dts:
            if session.done:
                break
            if step_dt is not None:
                training_cfg = getattr(trainer, "config", None)
                if training_cfg is not None:
                    _set_chaser_delta_v_limiter_dt(session, training_cfg=training_cfg, dt_s=float(step_dt))
            simulation_t0 = perf_counter()
            snapshot = session.step() if step_dt is None else session.step(dt_s=float(step_dt))
            presentation_controller = getattr(dashboard, "presentation_controller", None)
            if presentation_controller is not None:
                presentation_controller.record_simulation_step(perf_counter() - simulation_t0)
            if operator_command_provider is not None and hasattr(operator_command_provider, "observe_time"):
                operator_command_provider.observe_time(float(snapshot.time_s))
            dashboard.push_snapshot(snapshot)
            operator_transition_duration_s = _trigger_operator_projection_transition(
                dashboard, operator_command_provider
            )
            if operator_transition_duration_s is not None and callable(operator_burn_transition_callback):
                operator_burn_transition_callback(float(operator_transition_duration_s))
            if control_telemetry_provider is None:
                trainer.record(snapshot)
            else:
                trainer.record(snapshot, control_telemetry_provider=control_telemetry_provider)
            score = trainer.score()
            if hasattr(session, "record_scoring"):
                session.record_scoring(score)
            if bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False)):
                break
        if session.done or bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False)):
            break
    return score


def _split_game_step_dt(
    dt_s: float,
    *,
    max_step_dt_s: float | None = None,
    current_time_s: float | None = None,
    operator_command_provider: Any | None = None,
) -> tuple[float, ...]:
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        return ()
    max_step = _positive_float_or_none(max_step_dt_s)
    boundaries = {0.0, dt}
    if max_step is not None:
        boundary = float(max_step)
        while boundary < dt - 1.0e-12:
            boundaries.add(boundary)
            boundary += float(max_step)
    if current_time_s is not None and operator_command_provider is not None:
        plan = getattr(operator_command_provider, "plan", None)
        burns = tuple(getattr(plan, "burns", ()) or ())
        next_index = int(max(getattr(operator_command_provider, "_next_burn_index", 0), 0))
        impulse_duration_s = max(float(getattr(operator_command_provider, "impulse_duration_s", 0.0)), 0.0)
        start_t_s = float(current_time_s)
        for burn in burns[next_index:]:
            offset_s = float(getattr(burn, "time_s", 0.0)) - start_t_s
            if offset_s > dt + 1.0e-9:
                break
            burn_start_s = float(np.clip(offset_s, 0.0, dt))
            if 1.0e-12 < burn_start_s < dt - 1.0e-12:
                boundaries.add(burn_start_s)
            burn_stop_s = burn_start_s + impulse_duration_s
            if burn_start_s < dt - 1.0e-12 and 1.0e-12 < burn_stop_s < dt - 1.0e-12:
                boundaries.add(burn_stop_s)
    ordered = sorted(boundaries)
    return tuple(
        float(stop - start) for start, stop in zip(ordered, ordered[1:], strict=False) if stop - start > 1.0e-12
    )


def _game_attempt_current_time_s(session: Any, dashboard: Any) -> float | None:
    dashboard_times = getattr(dashboard, "t_s", None)
    if dashboard_times:
        return float(dashboard_times[-1])
    engine = getattr(session, "_engine", None)
    if engine is None:
        return None
    try:
        return float(engine.t_s[int(engine.current_index)])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _start_game_attempt(
    config: SimulationConfig,
    *,
    command_state: KeyboardCommandState,
    training_cfg: RPOTrainingConfig,
    controlled_object_id: str,
    attitude_rate_deg_s: float,
    control_mode: str,
    ric_reference_object_id: str,
    operator_burn_plan: OperatorBurnPlan | None = None,
    operator_actuator_error_fraction: float = 0.0,
    defensive_target_profile: dict[str, Any] | None = None,
) -> tuple[GamePhysicsSession, Any, Any]:
    config = _force_game_acceleration_off_config(config)
    root = config.to_dict()
    root.setdefault("metadata", {}).setdefault("game", {})["ric_reference_object_id"] = str(
        ric_reference_object_id
    )
    config = SimulationConfig.from_dict(root, source_path=config.source_path)
    session = GamePhysicsSession(
        _attempt_config_for_training_clock(config, training_cfg),
        retained_history_samples=_game_retained_history_samples(config),
    )
    snapshot = session.reset()
    if snapshot is None:
        raise RuntimeError("Game mode requires a single-run scenario.")
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    if defensive_target_profile is not None:
        target_id = str(training_cfg.target_object_id or "target")
        target_runtime = session._engine.agents[target_id].flight_software_runtime
        if target_runtime is not None and defensive_target_profile.get("max_delta_v_m_s") is not None:
            target_runtime.max_delta_v_m_s = float(defensive_target_profile["max_delta_v_m_s"])
    profile_mode = (
        GamePilotMode.AERODYNAMIC
        if str(control_mode).strip().lower() in AERODYNAMIC_CONTROL_MODES
        else GamePilotMode.TRANSLATION
        if str(control_mode).strip().lower() in TRANSLATION_CONTROL_MODES
        else GamePilotMode.ATTITUDE_THRUST
    )
    profile = GamePilotInputProfile(str(game_cfg["input_profile"]), profile_mode)
    if operator_burn_plan is None:
        provider = GamePilotInputAdapter(
            profile,
            source_id=f"game/{controlled_object_id}/pilot",
            boot_id="game-input-0",
        )
        runtime = session._engine.agents[controlled_object_id].flight_software_runtime
        if runtime is None:
            raise RuntimeError("Pilot mode requires a v2 flight-software runtime.")
        aero_cfg = _game_aerodynamic_control_config(config)
        provider.bind_physical_runtime(
            runtime,
            ballistic_coefficient_min_kg_m2=aero_cfg["ballistic_coefficient_min_kg_m2"],
            ballistic_coefficient_max_kg_m2=aero_cfg["ballistic_coefficient_max_kg_m2"],
            drag_coefficient=aero_cfg["drag_coefficient"],
            lift_coefficient=aero_cfg["lift_coefficient"],
            lift_area_m2=aero_cfg["lift_area_m2"],
        )
        session.add_fsw_input_publisher(
            controlled_object_id,
            lambda at: (
                (event,)
                if (
                    event := provider.sample_control_interval_if_changed(
                        command_state,
                        at=at,
                        control_interval_s=runtime.task_period_ns * 1.0e-9,
                    )
                )
                is not None
                else ()
            ),
        )
    else:
        operator_adapter = GameOperatorInputAdapter(
            source_id=f"game/{controlled_object_id}/operator",
            boot_id="game-input-0",
        )
        provider = GameOperatorController(
            operator_burn_plan,
            operator_adapter,
            impulse_duration_s=OPERATOR_IMPULSE_DURATION_S,
            actuator_error_fraction=operator_actuator_error_fraction,
        )
        runtime = session._engine.agents[controlled_object_id].flight_software_runtime
        if runtime is None:
            raise RuntimeError("Operator mode requires a v2 flight-software runtime.")
        events = operator_adapter.scheduled_burn_events(
            operator_burn_plan.burns,
            clock_id=f"{controlled_object_id}/onboard",
            tick_period_ns=runtime.tick_period_ns,
            impulse_duration_s=provider.impulse_duration_s,
            actuator_error_fraction=provider.actuator_error_fraction,
        )
        for event in events:
            session.publish_fsw_input(controlled_object_id, event)
    _install_chaser_delta_v_limiter(session, training_cfg=training_cfg, dt_s=float(config.scenario.simulator.dt_s))
    return session, provider, snapshot


def _poll_pygame_input(
    pygame: Any,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
    briefing_open: bool = False,
    terminal_open: bool = False,
    frame_convention: FrameConvention | dict[str, Any] | None = None,
) -> None:
    game_input.poll_pygame_input(
        pygame,
        state,
        control_mode=control_mode,
        briefing_open=briefing_open,
        terminal_open=terminal_open,
        frame_convention=frame_convention,
    )


def _request_pilot_input_poll_for_transition(
    session: GamePhysicsSession,
    provider: Any,
    command_state: KeyboardCommandState,
    *,
    controlled_object_id: str,
) -> bool:
    observe_transition = getattr(provider, "live_control_state_changed", None)
    if not callable(observe_transition) or not bool(observe_transition(command_state)):
        return False
    session.request_fsw_input_publisher_poll(controlled_object_id)
    return True


def _pygame_focus_lost(pygame: Any, event: Any) -> bool:
    return game_input.pygame_focus_lost(pygame, event)


def _opposing_key_axis(keys: Any, *, positive_key: Any, negative_key: Any) -> float:
    return game_input.opposing_key_axis(keys, positive_key=positive_key, negative_key=negative_key)


def _mission_state(score: Any) -> str:
    return mission_state_for_dashboard(phase_from_score(score))


def _game_loop_should_exit(*, session_done: bool, score: Any) -> bool:
    terminal_score = bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False))
    return bool(session_done) and not terminal_score


__all__ = [name for name in globals() if not name.startswith("__")]
