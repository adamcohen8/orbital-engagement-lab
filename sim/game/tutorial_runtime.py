# ruff: noqa: F401,F403,F405,I001
from .runner_common import *
from .runner_models import *
from .runner_config import *
from .launcher_common import _pos_in_bounds

def _clear_dashboard_tutorial_path(dashboard: Any) -> None:
    dashboard.tutorial_target_path_ric = np.empty((0, 6), dtype=float)
    if hasattr(dashboard, "_frame_cache_dirty"):
        dashboard._frame_cache_dirty = True


def _sync_guided_tutorial_path_for_mode(
    dashboard: Any,
    trainer: RPOTrainingTracker,
    training_cfg: RPOTrainingConfig,
    guided_tutorial: GuidedTutorialRuntime,
    *,
    game_mode: str,
) -> None:
    if _normalize_game_mode(game_mode) == "operator":
        return
    _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)

def _operator_coast_prediction_orbit_fraction(game_mode: str, difficulty: str) -> float:
    if _normalize_game_mode(game_mode) == "operator":
        return 1.0
    return _coast_prediction_orbit_fraction(difficulty)


def _wall_step_s(dt_s: float, speed_multiple: float) -> float:
    return float(dt_s) / max(float(speed_multiple), 1.0e-9)


def _coerce_speed_multiple(speed_multiple: float, *, options: tuple[float, ...] | None = None) -> float:
    value = float(speed_multiple)
    choices = tuple(options or SPEED_MULTIPLIER_OPTIONS)
    return min(choices, key=lambda option: abs(option - value))


def _adjust_speed_multiple(
    speed_multiple: float,
    change: int,
    *,
    options: tuple[float, ...] | None = None,
) -> float:
    choices = tuple(options or SPEED_MULTIPLIER_OPTIONS)
    current = _coerce_speed_multiple(speed_multiple, options=choices)
    idx = choices.index(current)
    idx = int(np.clip(idx + int(change), 0, len(choices) - 1))
    return choices[idx]


def _has_maneuver_input(state: KeyboardCommandState, *, control_mode: str = "attitude_thrust") -> bool:
    axes_active = any(abs(float(value)) > 1.0e-9 for value in (state.pitch, state.yaw, state.roll))
    mode = str(control_mode or "").strip().lower()
    if mode in AERODYNAMIC_CONTROL_MODES:
        return bool(abs(float(state.pitch)) > 1.0e-9 or abs(float(state.roll)) > 1.0e-9)
    if mode in TRANSLATION_CONTROL_MODES:
        return bool(axes_active and float(state.throttle) > 0.0)
    return bool(axes_active or (state.firing and float(state.throttle) > 0.0))


def _speed_after_maneuver_input(
    speed_multiple: float,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
    options: tuple[float, ...] | None = None,
    maneuver_control_speed_multiple: float | None = None,
) -> float:
    speed = _coerce_speed_multiple(speed_multiple, options=options)
    if _has_maneuver_input(state, control_mode=control_mode):
        configured_control_speed = _positive_float_or_none(maneuver_control_speed_multiple)
        control_speed = MANEUVER_CONTROL_SPEED if configured_control_speed is None else configured_control_speed
        if speed > control_speed:
            return _coerce_speed_multiple(control_speed, options=options)
    return speed


def _effective_speed_multiple_for_control(
    config: SimulationConfig,
    selected_speed_multiple: float,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
    options: tuple[float, ...] | None = None,
) -> float:
    if not _game_two_rail_speed_control_enabled(config):
        return _coerce_speed_multiple(selected_speed_multiple, options=options)
    return _speed_after_maneuver_input(
        selected_speed_multiple,
        state,
        control_mode=control_mode,
        options=options,
        maneuver_control_speed_multiple=_game_maneuver_control_speed_multiple(config),
    )


def _operator_burn_cinematic_should_arm(
    provider: Any | None,
    *,
    current_sim_time_s: float,
    dt_s: float,
    frame_horizon_s: float | None = None,
    lookahead_s: float = OPERATOR_BURN_CINEMATIC_LOOKAHEAD_S,
) -> bool:
    if provider is None or not hasattr(provider, "next_burn_time_s"):
        return False
    next_burn_time_s = provider.next_burn_time_s()
    if next_burn_time_s is None:
        return False
    time_to_burn_s = float(next_burn_time_s) - float(current_sim_time_s)
    step_window_s = 2.0 * max(float(dt_s), 0.0)
    frame_window_s = 0.0
    if frame_horizon_s is not None:
        frame_window_s = max(float(frame_horizon_s), 0.0) + max(float(lookahead_s), 0.0)
    trigger_window_s = max(float(lookahead_s), step_window_s, frame_window_s)
    return bool(time_to_burn_s >= -1.0e-9 and time_to_burn_s <= trigger_window_s + 1.0e-9)


def _update_operator_burn_cinematic(
    runtime: OperatorBurnCinematicRuntime,
    provider: Any | None,
    *,
    now_wall_s: float,
    current_sim_time_s: float,
    dt_s: float,
    frame_horizon_s: float | None = None,
) -> None:
    if runtime.active and runtime.hold_until_wall_s is not None and float(now_wall_s) > float(runtime.hold_until_wall_s):
        runtime.reset()
    if runtime.active:
        return
    if _operator_burn_cinematic_should_arm(
        provider,
        current_sim_time_s=current_sim_time_s,
        dt_s=dt_s,
        frame_horizon_s=frame_horizon_s,
    ):
        runtime.active = True
        runtime.hold_until_wall_s = None


def _operator_burn_cinematic_speed_multiple(
    speed_multiple: float,
    runtime: OperatorBurnCinematicRuntime,
    *,
    options: tuple[float, ...] | None = None,
) -> float:
    selected = _coerce_speed_multiple(speed_multiple, options=options)
    if not runtime.active:
        return selected
    cinematic = _coerce_speed_multiple(OPERATOR_BURN_CINEMATIC_SPEED_MULTIPLE, options=options)
    return min(selected, cinematic)


def _operator_burn_cinematic_hold_for_animation(
    runtime: OperatorBurnCinematicRuntime,
    *,
    now_wall_s: float,
    duration_s: float,
) -> None:
    runtime.active = True
    runtime.hold_until_wall_s = float(now_wall_s) + max(float(duration_s), 0.0)


def _operator_terminal_animation_pending(
    *,
    game_mode: str,
    score: Any,
    runtime: OperatorBurnCinematicRuntime,
) -> bool:
    if _normalize_game_mode(game_mode) != "operator":
        return False
    terminal_score = bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False))
    return bool(terminal_score and runtime.active)


def _phase_from_score_with_operator_animation(
    score: Any,
    *,
    briefing_open: bool = False,
    paused: bool = False,
    game_mode: str = "pilot",
    operator_burn_cinematic: OperatorBurnCinematicRuntime | None = None,
) -> GamePhase:
    if operator_burn_cinematic is not None and _operator_terminal_animation_pending(
        game_mode=game_mode,
        score=score,
        runtime=operator_burn_cinematic,
    ):
        return GamePhase.PLAYING
    return phase_from_score(score, briefing_open=briefing_open, paused=paused)


def _operator_burn_visual_duration_s(delta_v_m_s: float) -> float:
    try:
        magnitude = float(delta_v_m_s)
    except (TypeError, ValueError):
        magnitude = 0.0
    if not np.isfinite(magnitude):
        magnitude = 0.0
    duration = OPERATOR_BURN_VISUAL_DURATION_BASE_S + OPERATOR_BURN_VISUAL_DURATION_PER_M_S * max(magnitude, 0.0)
    return float(
        np.clip(
            duration,
            OPERATOR_BURN_VISUAL_DURATION_MIN_S,
            OPERATOR_BURN_VISUAL_DURATION_MAX_S,
        )
    )


def _clear_two_rail_released_maneuver_input(
    config: SimulationConfig,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
) -> bool:
    if not _game_two_rail_speed_control_enabled(config):
        return False
    if _has_maneuver_input(state, control_mode=control_mode):
        return False
    pending = (
        float(getattr(state, "pitch_sim_s", 0.0)),
        float(getattr(state, "yaw_sim_s", 0.0)),
        float(getattr(state, "roll_sim_s", 0.0)),
        float(getattr(state, "firing_sim_s", 0.0)),
    )
    if not any(abs(value) > 1.0e-12 for value in pending):
        return False
    state.clear_timed_input()
    return True


def _timed_maneuver_pending_sim_s(
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
) -> float:
    if not bool(getattr(state, "use_timing_accumulator", False)):
        return 0.0
    mode = str(control_mode or "").strip().lower()
    if mode in TRANSLATION_CONTROL_MODES:
        if float(getattr(state, "throttle", 0.0)) <= 0.0:
            return 0.0
        return max(
            abs(float(getattr(state, "pitch_sim_s", 0.0))),
            abs(float(getattr(state, "yaw_sim_s", 0.0))),
            abs(float(getattr(state, "roll_sim_s", 0.0))),
        )
    return max(float(getattr(state, "firing_sim_s", 0.0)), 0.0)


def _manual_maneuver_active_for_mode(
    game_mode: str,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
) -> bool:
    if _normalize_game_mode(game_mode) == "operator":
        return False
    return _has_maneuver_input(state, control_mode=control_mode)


def _pending_maneuver_sim_s_for_mode(
    game_mode: str,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
) -> float:
    if _normalize_game_mode(game_mode) == "operator":
        return 0.0
    return _timed_maneuver_pending_sim_s(state, control_mode=control_mode)


def _effective_speed_multiple_for_mode(
    config: SimulationConfig,
    selected_speed_multiple: float,
    state: KeyboardCommandState,
    *,
    game_mode: str,
    control_mode: str = "attitude_thrust",
    options: tuple[float, ...] | None = None,
) -> float:
    if _normalize_game_mode(game_mode) == "operator":
        return _coerce_speed_multiple(selected_speed_multiple, options=options)
    return _effective_speed_multiple_for_control(
        config,
        selected_speed_multiple,
        state,
        control_mode=control_mode,
        options=options,
    )


def _guided_tutorial_current_stage(training_cfg: RPOTrainingConfig, runtime: GuidedTutorialRuntime) -> Any | None:
    if runtime.awaiting_speed_step:
        return None
    stages = tuple(training_cfg.guided_tutorial_burns)
    idx = int(runtime.stage_index)
    if idx < 0 or idx >= len(stages):
        return None
    return stages[idx]


def _guided_tutorial_axis_value(state: KeyboardCommandState, axis: str) -> float:
    if axis == "radial":
        return float(state.pitch)
    if axis == "in_track":
        return float(state.yaw)
    if axis == "cross_track":
        return float(state.roll)
    return 0.0


def _guided_tutorial_input_matches(state: KeyboardCommandState, stage: Any) -> bool:
    expected_axis = str(stage.axis)
    expected_sign = 1.0 if int(stage.sign) >= 0 else -1.0
    if _guided_tutorial_axis_value(state, expected_axis) * expected_sign <= 0.5:
        return False
    for axis in ("radial", "in_track", "cross_track"):
        if axis == expected_axis:
            continue
        if abs(_guided_tutorial_axis_value(state, axis)) > 0.5:
            return False
    return True


def _guided_tutorial_wrong_input_active(state: KeyboardCommandState, stage: Any) -> bool:
    if not any(abs(_guided_tutorial_axis_value(state, axis)) > 0.5 for axis in ("radial", "in_track", "cross_track")):
        return False
    return not _guided_tutorial_input_matches(state, stage)


def _guided_tutorial_expected_key(stage: Any) -> str:
    axis = str(getattr(stage, "axis", ""))
    sign = 1 if int(getattr(stage, "sign", 1)) >= 0 else -1
    return {
        ("radial", 1): "W",
        ("radial", -1): "S",
        ("in_track", 1): "D",
        ("in_track", -1): "A",
        ("cross_track", 1): "Right",
        ("cross_track", -1): "Left",
    }.get((axis, sign), "the highlighted control")


def _guided_tutorial_target_path(
    rel0: np.ndarray,
    mean_motion_rad_s: float,
    stage: Any,
    *,
    samples: int = 181,
) -> np.ndarray:
    from sim.game.pygame_dashboard import _cw_coast_state

    n = float(mean_motion_rad_s)
    if not np.isfinite(n) or n <= 0.0:
        return np.empty((0, 6), dtype=float)
    state0 = np.array(rel0, dtype=float).reshape(6).copy()
    axis_idx = {"radial": 3, "in_track": 4, "cross_track": 5}.get(str(stage.axis))
    if axis_idx is None:
        return np.empty((0, 6), dtype=float)
    state0[axis_idx] += (1.0 if int(stage.sign) >= 0 else -1.0) * float(stage.delta_v_m_s) / 1000.0
    horizon_s = 2.0 * np.pi / n
    times = np.linspace(0.0, horizon_s, max(int(samples), 2), dtype=float)
    return np.vstack([_cw_coast_state(state0, float(t), n) for t in times])


def _guided_tutorial_delta_v_m_s(trainer: RPOTrainingTracker, stage: Any) -> float:
    if len(trainer.t_s) < 2 or len(trainer.thrust_ric_hist) < 2:
        return 0.0
    axis_idx = {"radial": 0, "in_track": 1, "cross_track": 2}.get(str(stage.axis))
    if axis_idx is None:
        return 0.0
    t = np.array(trainer.t_s, dtype=float).reshape(-1)
    thrust = np.vstack(trainer.thrust_ric_hist)
    n = min(t.size, thrust.shape[0])
    if n < 2:
        return 0.0
    component = (1.0 if int(stage.sign) >= 0 else -1.0) * thrust[1:n, axis_idx]
    dt = np.diff(t[:n])
    valid = np.isfinite(component) & np.isfinite(dt) & (dt > 0.0) & (component > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(component[valid] * dt[valid]) * 1000.0)


def _guided_tutorial_stage_hint(stage: Any | None, runtime: GuidedTutorialRuntime) -> str:
    if stage is None:
        return ""
    if runtime.wrong_key_active:
        return f"Wrong key - hold {_guided_tutorial_expected_key(stage)} for {stage.display_label}."
    hint = str(getattr(stage, "hint", "") or "").strip()
    if not hint:
        hint = f"Hold {stage.display_label} until the burn reaches the green target path."
    progress = float(max(runtime.active_stage_delta_v_m_s, 0.0))
    target = float(getattr(stage, "delta_v_m_s", 0.0))
    if target > 0.0:
        return f"{hint} Burn progress: {progress:.2f}/{target:.2f} m/s."
    return hint


def _guided_tutorial_speed_step_hint(training_cfg: RPOTrainingConfig, current_speed_multiple: float) -> str:
    step = training_cfg.guided_tutorial_speed_step
    if step is None:
        return ""
    hint = step.hint or (
        "Want to go faster? Hit the up arrow key to increase the speed multiple. "
        f"Try going up to {step.target_speed_multiplier:g}x."
    )
    return f"{hint} Current speed: {float(current_speed_multiple):g}x."


def _guided_tutorial_speed_step_reached(training_cfg: RPOTrainingConfig, current_speed_multiple: float) -> bool:
    step = training_cfg.guided_tutorial_speed_step
    if step is None:
        return True
    return float(current_speed_multiple) + 1.0e-9 >= float(step.target_speed_multiplier)


def _guided_tutorial_speed_step_follows_burn(training_cfg: RPOTrainingConfig, completed_stage: Any | None) -> bool:
    step = training_cfg.guided_tutorial_speed_step
    if step is None or completed_stage is None:
        return False
    after_name = str(step.after_burn_name or "").strip()
    if not after_name:
        return False
    return str(getattr(completed_stage, "name", "") or "") == after_name


def _guided_tutorial_update_dashboard_path(
    dashboard: Any,
    trainer: RPOTrainingTracker,
    training_cfg: RPOTrainingConfig,
    runtime: GuidedTutorialRuntime,
) -> None:
    stage = _guided_tutorial_current_stage(training_cfg, runtime)
    path = np.empty((0, 6), dtype=float)
    if stage is None:
        runtime.stage_start_rel_ric = None
        runtime.stage_start_mean_motion_rad_s = None
    elif trainer.rel_ric_hist and trainer.mean_motion_hist:
        if runtime.stage_start_rel_ric is None or runtime.stage_start_mean_motion_rad_s is None:
            runtime.stage_start_rel_ric = np.array(trainer.rel_ric_hist[-1], dtype=float).reshape(6)
            runtime.stage_start_mean_motion_rad_s = float(trainer.mean_motion_hist[-1])
        path = _guided_tutorial_target_path(
            runtime.stage_start_rel_ric,
            runtime.stage_start_mean_motion_rad_s,
            stage,
        )
    dashboard.tutorial_target_path_ric = path
    if hasattr(dashboard, "_frame_cache_dirty"):
        dashboard._frame_cache_dirty = True


def _guided_tutorial_complete_active_stage(
    trainer: RPOTrainingTracker,
    training_cfg: RPOTrainingConfig,
    runtime: GuidedTutorialRuntime,
) -> bool:
    stage = _guided_tutorial_current_stage(training_cfg, runtime)
    if stage is None:
        return False
    runtime.active_stage_delta_v_m_s = _guided_tutorial_delta_v_m_s(trainer, stage)
    if runtime.active_stage_delta_v_m_s + 1.0e-9 < float(stage.delta_v_m_s):
        return False
    trainer.mark_guided_tutorial_burn_complete(stage.name)
    runtime.stage_index += 1
    runtime.active_stage_delta_v_m_s = 0.0
    runtime.stage_start_rel_ric = None
    runtime.stage_start_mean_motion_rad_s = None
    return True


def _reset_guided_tutorial_stage_attempt(
    *,
    attempt_config: SimulationConfig,
    command_state: KeyboardCommandState,
    trainer: RPOTrainingTracker,
    dashboard: Any,
    training_cfg: RPOTrainingConfig,
    controlled_object_id: str,
    attitude_rate_deg_s: float,
    control_mode: str,
    ric_reference_object_id: str,
) -> tuple[GamePhysicsSession, Any]:
    command_state.reset_axes()
    session, _, snapshot = _lazy_start_game_attempt(
        attempt_config,
        command_state=command_state,
        training_cfg=training_cfg,
        controlled_object_id=controlled_object_id,
        attitude_rate_deg_s=attitude_rate_deg_s,
        control_mode=control_mode,
        ric_reference_object_id=ric_reference_object_id,
    )
    trainer.clear(reset_guided_tutorial_progress=False)
    dashboard.clear()
    _lazy_sync_dashboard_training_config(dashboard, training_cfg)
    _lazy_sync_dashboard_round_config(dashboard, attempt_config)
    dashboard.push_snapshot(snapshot)
    trainer.record(snapshot)
    return session, snapshot


def _sandbox_setup_text_values(setup: SandboxSetupValues) -> list[str]:
    return [f"{float(getattr(setup, field)):.6g}" for _, _, field in _SANDBOX_SETUP_FIELDS]


def _sandbox_setup_from_text_values(values: list[str]) -> tuple[SandboxSetupValues | None, str]:
    parsed: dict[str, float] = {}
    for idx, (_, unit, field) in enumerate(_SANDBOX_SETUP_FIELDS):
        raw = str(values[idx] if idx < len(values) else "").strip()
        try:
            value = float(raw)
        except ValueError:
            suffix = f" ({unit})" if unit else ""
            return None, f"Enter a numeric value for {_SANDBOX_SETUP_FIELDS[idx][0]}{suffix}."
        if not np.isfinite(value):
            return None, f"{_SANDBOX_SETUP_FIELDS[idx][0]} must be finite."
        parsed[field] = value
    if parsed["target_a_km"] <= 0.0:
        return None, "Target Semimajor Axis must be positive."
    if not (0.0 <= parsed["target_ecc"] < 1.0):
        return None, "Target Eccentricity must satisfy 0 <= e < 1."
    if not (0.0 <= parsed["target_inc_deg"] <= 180.0):
        return None, "Target Inclination must satisfy 0 <= i <= 180 degrees."
    return SandboxSetupValues(**parsed), ""


def _sandbox_setup_layout(
    pygame: Any,
    screen_width: int,
    screen_height: int,
) -> tuple[tuple[Any, Any], list[Any], Any, Any]:
    margin_x = 54
    gap = 24
    panel_top = 112
    footer_top = max(int(screen_height) - 88, panel_top + 330)
    panel_width = max((int(screen_width) - 2 * margin_x - gap) // 2, 300)
    panel_height = max(footer_top - panel_top - 18, 320)
    left_panel = pygame.Rect(margin_x, panel_top, panel_width, panel_height)
    right_panel = pygame.Rect(margin_x + panel_width + gap, panel_top, panel_width, panel_height)
    row_height = max(40, min(58, (panel_height - 82) // 6))
    field_rects: list[Any] = []
    for panel in (left_panel, right_panel):
        input_x = panel.x + max(int(panel.width * 0.57), 170)
        input_width = max(panel.right - input_x - 18, 96)
        for row in range(6):
            field_rects.append(
                pygame.Rect(
                    input_x,
                    panel.y + 60 + row * row_height,
                    input_width,
                    max(row_height - 10, 30),
                )
            )
    launch_rect = pygame.Rect(int(screen_width) - 180, int(screen_height) - 70, 126, 36)
    cancel_rect = pygame.Rect(launch_rect.x - 130, launch_rect.y, 120, 36)
    return (left_panel, right_panel), field_rects, launch_rect, cancel_rect


def _sandbox_setup_next_field(active_index: int, key: Any, pygame: Any) -> int:
    index = int(active_index) % len(_SANDBOX_SETUP_FIELDS)
    row = index % 6
    column = index // 6
    if key == pygame.K_UP:
        return column * 6 + (row - 1) % 6
    if key == pygame.K_DOWN:
        return column * 6 + (row + 1) % 6
    if key == getattr(pygame, "K_LEFT", object()):
        return row
    if key == getattr(pygame, "K_RIGHT", object()):
        return 6 + row
    return index


def _run_sandbox_setup_form(
    dashboard: Any,
    *,
    config: SimulationConfig,
    speed_multiple: float,
    level_title: str,
) -> SandboxSetupValues | None:
    pygame = dashboard.pygame
    values = _sandbox_setup_text_values(_sandbox_setup_from_config(config))
    active_idx = 0
    error = ""
    allowed_chars = set("0123456789+-.eE")
    del speed_multiple
    from sim.game.launcher_widgets import _draw_sandbox_setup_screen

    get_grab = getattr(pygame.event, "get_grab", None)
    set_grab = getattr(pygame.event, "set_grab", None)
    mouse_api = getattr(pygame, "mouse", None)
    get_visible = getattr(mouse_api, "get_visible", None)
    set_visible = getattr(mouse_api, "set_visible", None)
    previous_grab = bool(get_grab()) if callable(get_grab) else False
    previous_visible = bool(get_visible()) if callable(get_visible) else True
    if callable(set_grab):
        set_grab(False)
    if callable(set_visible):
        set_visible(True)
    try:
        while not getattr(dashboard, "closed", False):
            width, height = dashboard.screen.get_size()
            panels, field_rects, launch_rect, cancel_rect = _sandbox_setup_layout(pygame, width, height)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == getattr(pygame, "MOUSEBUTTONDOWN", object()):
                    if int(getattr(event, "button", 0)) != 1:
                        continue
                    mouse_pos = getattr(event, "pos", None)
                    if mouse_pos is None:
                        get_pos = getattr(mouse_api, "get_pos", None)
                        mouse_pos = get_pos() if callable(get_pos) else (-1, -1)
                    if _pos_in_bounds(mouse_pos, (launch_rect.x, launch_rect.y, launch_rect.w, launch_rect.h)):
                        setup, error = _sandbox_setup_from_text_values(values)
                        if setup is not None:
                            return setup
                        continue
                    if _pos_in_bounds(mouse_pos, (cancel_rect.x, cancel_rect.y, cancel_rect.w, cancel_rect.h)):
                        return None
                    for idx, rect in enumerate(field_rects):
                        if _pos_in_bounds(mouse_pos, (rect.x, rect.y, rect.w, rect.h)):
                            active_idx = idx
                            error = ""
                            break
                    continue
                if event.type != pygame.KEYDOWN:
                    continue
                key = getattr(event, "key", None)
                if key == pygame.K_ESCAPE:
                    return None
                if key in {pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}:
                    setup, error = _sandbox_setup_from_text_values(values)
                    if setup is not None:
                        return setup
                    continue
                if key == pygame.K_TAB:
                    backwards = bool(getattr(event, "mod", 0) & getattr(pygame, "KMOD_SHIFT", 0))
                    active_idx = (active_idx + (-1 if backwards else 1)) % len(values)
                    error = ""
                    continue
                if key in {
                    pygame.K_DOWN,
                    pygame.K_UP,
                    getattr(pygame, "K_LEFT", object()),
                    getattr(pygame, "K_RIGHT", object()),
                }:
                    active_idx = _sandbox_setup_next_field(active_idx, key, pygame)
                    error = ""
                    continue
                if key == pygame.K_BACKSPACE:
                    values[active_idx] = values[active_idx][:-1]
                    error = ""
                    continue
                if key == getattr(pygame, "K_DELETE", object()):
                    values[active_idx] = ""
                    error = ""
                    continue
                text = str(getattr(event, "unicode", "") or "")
                if text and all(ch in allowed_chars for ch in text):
                    values[active_idx] += text
                    error = ""
            setup, validation_error = _sandbox_setup_from_text_values(values)
            if validation_error:
                error = validation_error
            _draw_sandbox_setup_screen(
                pygame,
                dashboard.screen,
                level_title=level_title,
                values=values,
                active_index=active_idx,
                panels=panels,
                field_rects=field_rects,
                launch_rect=launch_rect,
                cancel_rect=cancel_rect,
                validation_message=error,
                can_launch=setup is not None,
                font=dashboard.font,
                small_font=dashboard.small_font,
                title_font=dashboard.large_font,
            )
            display_flip = getattr(getattr(pygame, "display", None), "flip", None)
            if callable(display_flip):
                display_flip()
            dashboard.tick(60.0)
        return None
    finally:
        if callable(set_grab):
            set_grab(previous_grab)
        if callable(set_visible):
            set_visible(previous_visible)


def _camera_rule_toggle_enabled_for_dashboard(dashboard: Any, training_cfg: RPOTrainingConfig) -> bool:
    return bool(getattr(training_cfg, "sandbox_mode", False)) or bool(
        getattr(dashboard, "camera_rule_toggle_enabled", False)
    )


def _camera_rule_status(dashboard: Any, training_cfg: RPOTrainingConfig) -> str:
    if not _camera_rule_toggle_enabled_for_dashboard(dashboard, training_cfg):
        return ""
    mode = str(getattr(dashboard, "_camera_rule_mode_key", lambda: "current_pair")())
    label = "Full Trajectory" if mode == "full_trajectory" else "Satellites Only"
    return f"C Camera: {label}"


def _coach_hint_with_camera_rule(hint: str, dashboard: Any, training_cfg: RPOTrainingConfig) -> str:
    status = _camera_rule_status(dashboard, training_cfg)
    if not status:
        return hint
    base = str(hint or "").strip()
    if not base:
        return status
    return f"{base} {status}."


def _dashboard_fps_for_speed(
    speed_multiple: float,
    *,
    recording: bool = False,
    static_screen: bool = False,
    recording_fps: float = GAME_RECORDING_FPS,
    fps_cap: float | None = None,
    high_speed_fps: float | None = None,
    high_speed_fps_max_multiple: float | None = None,
) -> float:
    if bool(recording):
        return float(max(recording_fps, 1.0))
    cap = _positive_float_or_none(fps_cap)
    if float(speed_multiple) >= 100.0:
        override_limit = _positive_float_or_none(high_speed_fps_max_multiple)
        override_fps = _positive_float_or_none(high_speed_fps)
        if override_fps is not None and (override_limit is None or float(speed_multiple) <= override_limit + 1.0e-9):
            fps = override_fps
        else:
            fps = HIGH_SPEED_DASHBOARD_FPS
    elif float(speed_multiple) >= 50.0:
        fps = MEDIUM_HIGH_SPEED_DASHBOARD_FPS
    else:
        fps = DASHBOARD_FPS
    if bool(static_screen):
        fps = min(float(fps), float(STATIC_DASHBOARD_FPS))
    if cap is not None:
        fps = min(float(fps), float(cap))
    return float(max(fps, 1.0))


def _presentation_fps_for_frame(
    controller: PresentationFrameController | None,
    speed_multiple: float,
    *,
    recording: bool = False,
    static_screen: bool = False,
    recording_fps: float = GAME_RECORDING_FPS,
    fps_cap: float | None = None,
    high_speed_fps: float | None = None,
    high_speed_fps_max_multiple: float | None = None,
) -> float:
    compatibility_fps = _dashboard_fps_for_speed(
        speed_multiple,
        recording=recording,
        static_screen=static_screen,
        recording_fps=recording_fps,
        fps_cap=fps_cap,
        high_speed_fps=high_speed_fps,
        high_speed_fps_max_multiple=high_speed_fps_max_multiple,
    )
    if controller is None:
        return compatibility_fps
    return controller.target_fps(
        compatibility_fps=compatibility_fps,
        recording=recording,
        recording_fps=recording_fps,
        static_screen=static_screen,
    )


def _clip_recording_status(
    controller: GameClipRecordingController,
    *,
    started_wall_s: float | None,
    now_wall_s: float,
    status_message: str = "",
    status_until_wall_s: float = 0.0,
) -> str:
    if controller.recording:
        elapsed = 0.0 if started_wall_s is None else max(float(now_wall_s) - float(started_wall_s), 0.0)
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"REC {minutes:02d}:{seconds:02d}  G/F9 discard  Enter save"
    if status_message and float(now_wall_s) < float(status_until_wall_s):
        return status_message
    return ""


def _pause_teaching_overlay_enabled(
    phase: GamePhase,
    training_cfg: RPOTrainingConfig,
    guided_tutorial: GuidedTutorialRuntime,
) -> bool:
    return bool(
        phase == GamePhase.PAUSED
        and _guided_tutorial_current_stage(training_cfg, guided_tutorial) is None
        and not bool(guided_tutorial.awaiting_speed_step)
    )


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


def _realtime_steps_due_with_backlog(
    *,
    now_s: float,
    last_step_wall_s: float,
    wall_step_s: float,
    max_steps: int = MAX_REALTIME_STEPS_PER_FRAME,
) -> tuple[int, float, int]:
    wall_step = float(max(wall_step_s, 1.0e-9))
    elapsed = max(float(now_s) - float(last_step_wall_s), 0.0)
    due = int(elapsed // wall_step)
    if due <= 0:
        return 0, float(last_step_wall_s), 0
    cap = max(int(max_steps), 1)
    steps = min(due, cap)
    discarded = max(due - cap, 0)
    if due > cap:
        return steps, float(now_s), discarded
    return steps, float(last_step_wall_s) + float(steps) * wall_step, 0


def _dashboard_snapshot_age_s(dashboard: Any, *, now_s: float | None = None) -> float | None:
    samples = getattr(dashboard, "sample_wall_s", ())
    if not samples:
        return None
    current = perf_counter() if now_s is None else float(now_s)
    return max(current - float(samples[-1]), 0.0)


def _command_status(state: KeyboardCommandState, *, control_mode: str = "attitude_thrust") -> str:
    mode = str(control_mode or "").strip().lower()
    if mode in AERODYNAMIC_CONTROL_MODES:
        return "W/S Increase/Decrease BC  Left/Right Lift CCW/CW  Space Pause  C Camera  M Music"
    if mode in CISLUNAR_TRANSLATION_MODES:
        return "W/S Earth-Moon Y  A/D Tangential X  Left/Right Normal Z  C Camera  M Music"
    if mode in MOON_RIC_TRANSLATION_MODES:
        return "W/S R about Moon  A/D I about Moon  Left/Right C about Moon  C Camera  O/P ECI  M Music"
    if mode in TRANSLATION_CONTROL_MODES:
        return "W/S R  A/D I  Left/Right C  C Camera  O/P ECI  M Music"
    burn = "FIRE" if state.firing else "Coast"
    return (
        "W/S Pitch  A/D Yaw  Left/Right Roll  Space Fire  M Music  R Reset  Esc Quit\n"
        "Keys work in the figure window or this terminal; terminal input is pulse/repeat based.\n"
        f"Pitch={state.pitch:+.0f} Yaw={state.yaw:+.0f} Roll={state.roll:+.0f} Thrust={burn}"
    )


def _operator_next_burn_status(provider: Any) -> str:
    next_burn_getter = getattr(provider, "next_burn", None)
    burn = next_burn_getter() if callable(next_burn_getter) else None
    if burn is None:
        return "Next Burn: None"
    delta_v = np.asarray(getattr(burn, "delta_v_ric_m_s", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
    if delta_v.size < 3:
        delta_v = np.pad(delta_v, (0, 3 - delta_v.size), mode="constant")
    return (
        f"Next Burn: T+{float(getattr(burn, 'time_s', 0.0)):g}s | "
        f"{float(delta_v[0]):g} m/s R, {float(delta_v[1]):g} m/s I, {float(delta_v[2]):g} m/s C"
    )


def _game_command_status(
    state: KeyboardCommandState,
    *,
    control_mode: str,
    game_mode: str,
    command_provider: Any,
) -> str:
    if _normalize_game_mode(game_mode) == "operator":
        return _operator_next_burn_status(command_provider)
    if str(control_mode or "").strip().lower() in AERODYNAMIC_CONTROL_MODES:
        return (
            f"{_command_status(state, control_mode=control_mode)}\n"
            f"BC={float(getattr(command_provider, 'ballistic_coefficient_kg_m2', 0.0)):.1f} kg/m^2  "
            f"Lift={float(getattr(command_provider, 'lift_bank_angle_deg', 0.0)):+.0f} deg"
        )
    return _command_status(state, control_mode=control_mode)


def _live_prediction_accel_ric(
    state: KeyboardCommandState,
    *,
    control_mode: str,
    max_accel_km_s2: float,
) -> np.ndarray:
    if bool(state.paused) or str(control_mode or "").strip().lower() not in TRANSLATION_CONTROL_MODES:
        return np.zeros(3, dtype=float)
    if float(state.throttle) <= 0.0:
        return np.zeros(3, dtype=float)
    accel = np.array(
        [
            float(np.clip(state.pitch, -1.0, 1.0)),
            float(np.clip(state.yaw, -1.0, 1.0)),
            float(np.clip(state.roll, -1.0, 1.0)),
        ],
        dtype=float,
    )
    nrm = float(np.linalg.norm(accel))
    if nrm > 1.0:
        accel /= nrm
    return accel * float(max(max_accel_km_s2, 0.0)) * float(np.clip(state.throttle, 0.0, 1.0))


def _live_prediction_burn(
    state: KeyboardCommandState,
    *,
    control_mode: str,
    max_accel_km_s2: float,
    elapsed_wall_s: float,
    speed_multiple: float,
    dt_s: float,
) -> tuple[np.ndarray, float]:
    mode = str(control_mode or "").strip().lower()
    if bool(state.paused) or mode not in TRANSLATION_CONTROL_MODES:
        return np.zeros(3, dtype=float), 0.0
    if float(state.throttle) <= 0.0:
        return np.zeros(3, dtype=float), 0.0
    if bool(getattr(state, "use_timing_accumulator", False)):
        if not _has_maneuver_input(state, control_mode=control_mode):
            return np.zeros(3, dtype=float), 0.0
        accel = _live_prediction_accel_ric(
            state,
            control_mode=control_mode,
            max_accel_km_s2=max_accel_km_s2,
        )
        elapsed = 0.0
        if float(np.linalg.norm(accel)) > 0.0:
            elapsed = min(
                max(float(elapsed_wall_s), 0.0) * max(float(speed_multiple), 0.0),
                max(float(dt_s), 0.0),
            )
        return accel, elapsed

    accel = _live_prediction_accel_ric(
        state,
        control_mode=control_mode,
        max_accel_km_s2=max_accel_km_s2,
    )
    elapsed = 0.0
    if float(np.linalg.norm(accel)) > 0.0:
        elapsed = min(
            max(float(elapsed_wall_s), 0.0) * max(float(speed_multiple), 0.0),
            max(float(dt_s), 0.0),
        )
    return accel, elapsed


def _sync_live_prediction_burn(
    dashboard: Any,
    state: KeyboardCommandState,
    *,
    control_mode: str,
    max_accel_km_s2: float,
    elapsed_wall_s: float,
    speed_multiple: float,
    dt_s: float,
) -> None:
    if not hasattr(dashboard, "set_live_prediction_burn"):
        return
    accel, elapsed = _live_prediction_burn(
        state,
        control_mode=control_mode,
        max_accel_km_s2=max_accel_km_s2,
        elapsed_wall_s=elapsed_wall_s,
        speed_multiple=speed_multiple,
        dt_s=dt_s,
    )
    dashboard.set_live_prediction_burn(accel, elapsed)


def _clear_live_prediction_burn(dashboard: Any) -> None:
    if hasattr(dashboard, "set_live_prediction_burn"):
        dashboard.set_live_prediction_burn(np.zeros(3, dtype=float), 0.0)


def _sync_live_prediction_burn_for_mode(
    dashboard: Any,
    state: KeyboardCommandState,
    *,
    game_mode: str,
    control_mode: str,
    max_accel_km_s2: float,
    elapsed_wall_s: float,
    speed_multiple: float,
    dt_s: float,
) -> None:
    if _normalize_game_mode(game_mode) == "operator":
        return
    _sync_live_prediction_burn(
        dashboard,
        state,
        control_mode=control_mode,
        max_accel_km_s2=max_accel_km_s2,
        elapsed_wall_s=elapsed_wall_s,
        speed_multiple=speed_multiple,
        dt_s=dt_s,
    )


def _trigger_operator_projection_transition(dashboard: Any, provider: Any | None) -> float | None:
    if provider is None or not hasattr(dashboard, "set_operator_projection_transition"):
        return None
    if getattr(provider, "last_executed_burn", None) is None:
        return None
    delta_v = getattr(provider, "last_executed_delta_v_ric_m_s", None)
    if delta_v is None:
        return None
    delta_v_ric_km_s = np.asarray(delta_v, dtype=float).reshape(3) / 1000.0
    if not np.all(np.isfinite(delta_v_ric_km_s)) or float(np.linalg.norm(delta_v_ric_km_s)) <= 0.0:
        return None
    rel_hist = getattr(dashboard, "rel_hist", ())
    if not rel_hist:
        return None
    post_burn_rel = np.asarray(rel_hist[-1], dtype=float).reshape(6)
    if not np.all(np.isfinite(post_burn_rel)):
        return None
    pre_burn_rel = post_burn_rel.copy()
    pre_burn_rel[3:6] -= delta_v_ric_km_s
    duration_s = _operator_burn_visual_duration_s(float(np.linalg.norm(delta_v_ric_km_s)) * 1000.0)
    dashboard.set_operator_projection_transition(pre_burn_rel, post_burn_rel, duration_s=duration_s)
    return duration_s


def _lazy_start_game_attempt(*args, **kwargs):
    from .attempt_lifecycle import _start_game_attempt

    return _start_game_attempt(*args, **kwargs)


def _lazy_sync_dashboard_training_config(*args, **kwargs):
    from .recording_runtime import _sync_dashboard_training_config

    return _sync_dashboard_training_config(*args, **kwargs)


def _lazy_sync_dashboard_round_config(*args, **kwargs):
    from .recording_runtime import _sync_dashboard_round_config

    return _sync_dashboard_round_config(*args, **kwargs)

__all__ = [name for name in globals() if not name.startswith("__")]
