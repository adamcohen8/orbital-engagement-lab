# ruff: noqa: F401,F403,F405,I001
from .runner_common import *
from .runner_models import *
from .runner_config import *
from .tutorial_runtime import *
from .recording_runtime import *

def _step_game_attempt(
    *,
    session: GamePhysicsSession,
    dashboard: Any,
    trainer: RPOTrainingTracker,
    steps_to_run: int,
    initial_score: Any | None = None,
    dt_s: float | None = None,
    max_step_dt_s: float | None = None,
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
            snapshot = session.step() if step_dt is None else session.step(dt_s=float(step_dt))
            dashboard.push_snapshot(snapshot)
            operator_transition_duration_s = _trigger_operator_projection_transition(dashboard, operator_command_provider)
            if operator_transition_duration_s is not None and callable(operator_burn_transition_callback):
                operator_burn_transition_callback(float(operator_transition_duration_s))
            trainer.record(snapshot)
            score = trainer.score()
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
        float(stop - start)
        for start, stop in zip(ordered, ordered[1:], strict=False)
        if stop - start > 1.0e-12
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
    defensive_target_provider: DefensiveTargetIntentProvider | None = None,
) -> tuple[GamePhysicsSession, Any, Any]:
    config = _force_game_acceleration_off_config(config)
    session = GamePhysicsSession(
        _attempt_config_for_training_clock(config, training_cfg),
        retained_history_samples=_game_retained_history_samples(config),
    )
    if operator_burn_plan is None:
        provider: Any = ManualGameCommandProvider(
            command_state=command_state,
            max_accel_km_s2=_max_accel_from_config(config, controlled_object_id),
            attitude_rate_deg_s=attitude_rate_deg_s,
            controlled_object_id=controlled_object_id,
            control_mode=control_mode,
            reference_object_id=ric_reference_object_id,
        )
    else:
        provider = OperatorBurnCommandProvider(
            operator_burn_plan,
            controlled_object_id=controlled_object_id,
            reference_object_id=ric_reference_object_id,
            control_mode=control_mode,
            relative_frame=_game_relative_frame(config),
            actuator_error_fraction=operator_actuator_error_fraction,
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
    if hasattr(provider, "reset_target_to_current"):
        provider.reset_target_to_current(snapshot.truth[controlled_object_id])
    return session, provider, snapshot


def _poll_pygame_input(
    pygame: Any,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
    briefing_open: bool = False,
    terminal_open: bool = False,
) -> None:
    game_input.poll_pygame_input(
        pygame,
        state,
        control_mode=control_mode,
        briefing_open=briefing_open,
        terminal_open=terminal_open,
    )


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
