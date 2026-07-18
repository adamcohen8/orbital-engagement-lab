# ruff: noqa: F401,F403,F405,I001
from .runner_common import *
from .runner_models import *

def _force_game_acceleration_off_config(config: SimulationConfig) -> SimulationConfig:
    return (
        config.with_value("simulator.acceleration.mode", "off")
        .with_value("simulator.acceleration.warmup", False)
        .with_value("simulator.acceleration.env_override", False)
    )


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


def _game_relative_frame(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("relative_frame", "ric") or "ric").strip().lower()


def _game_target_sprite_path(config: SimulationConfig) -> Path | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = str(game_cfg.get("target_sprite_path", "") or "").strip()
    return Path(raw) if raw else None


def _game_chaser_sprite_path(config: SimulationConfig) -> Path | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = str(game_cfg.get("chaser_sprite_path", "") or "").strip()
    return Path(raw) if raw else None










def _game_target_sprite_diameter_km(config: SimulationConfig) -> float:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return float(game_cfg.get("target_sprite_diameter_km", 0.006) or 0.006)


def _game_chaser_sprite_diameter_km(config: SimulationConfig) -> float:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return float(game_cfg.get("chaser_sprite_diameter_km", 0.006) or 0.006)




def _game_controlled_object_id(config: SimulationConfig, default: str = "chaser") -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("controlled_object_id", default) or default)


def _training_config_with_sun_environment(training_cfg: RPOTrainingConfig, config: SimulationConfig) -> RPOTrainingConfig:
    constraints = tuple(getattr(training_cfg, "sun_angle_constraints", ()) or ())
    if not constraints:
        return training_cfg
    env = dict(config.scenario.simulator.environment or {})
    if config.scenario.simulator.initial_jd_utc is not None:
        env.setdefault("jd_utc_start", float(config.scenario.simulator.initial_jd_utc))
    return replace(
        training_cfg,
        sun_angle_constraints=tuple(constraint.with_sun_environment(env) for constraint in constraints),
    )


def _game_difficulty(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("difficulty", "easy") or "easy").strip().lower()


def _normalize_game_mode(value: Any) -> str:
    key = str(value or "pilot").strip().lower()
    if key in {"operator", "op", "script", "scripted"}:
        return "operator"
    return "pilot"


def _display_game_mode(value: Any) -> str:
    return "Operator Mode" if _normalize_game_mode(value) == "operator" else "Pilot Mode"


def _operator_actuator_error_fraction(difficulty: str) -> float:
    key = str(difficulty or "easy").strip().lower()
    if key not in OPERATOR_ACTUATOR_ERROR_BY_DIFFICULTY:
        raise ValueError("operator difficulty must be one of: easy, medium, hard, extreme")
    return float(OPERATOR_ACTUATOR_ERROR_BY_DIFFICULTY[key])


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


def _game_plot_prediction_in_zoom(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("plot_prediction_in_zoom", False))


def _game_plot_prediction_zoom_max_span_km(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("plot_prediction_zoom_max_span_km"))


def _game_plot_prediction_full_trajectory_only(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("plot_prediction_full_trajectory_only", False))


def _game_cr3bp_coast_prediction_horizon_s(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("cr3bp_coast_prediction_horizon_s"))


def _game_cr3bp_active_prediction_horizon_s(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("cr3bp_active_prediction_horizon_s"))


def _game_cr3bp_coast_prediction_horizon_mode(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    value = str(game_cfg.get("cr3bp_coast_prediction_horizon_mode", "default") or "default")
    key = value.strip().lower().replace("-", "_")
    if key in {"time_remaining", "remaining_time", "mission_remaining", "mission_time_remaining"}:
        return "time_remaining"
    return "default"


def _game_cr3bp_coast_prediction_dt_s(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("cr3bp_coast_prediction_dt_s"))


def _game_cr3bp_prediction_coast_update_interval_s(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("cr3bp_prediction_coast_update_interval_s"))


def _game_cr3bp_projection_mode(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    value = str(game_cfg.get("cr3bp_projection_mode", "nonlinear") or "nonlinear").strip().lower().replace("-", "_")
    if value in {"linear", "linearized", "stm", "variational"}:
        return "linearized"
    return "nonlinear"


def _game_target_coast_prediction_horizon_s(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("target_coast_prediction_horizon_s"))


def _game_target_coast_prediction_dt_s(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("target_coast_prediction_dt_s"))


def _game_dashboard_fps_cap(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("dashboard_fps_cap"))


def _game_dashboard_high_speed_fps(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("dashboard_high_speed_fps"))


def _game_dashboard_high_speed_fps_max_multiple(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("dashboard_high_speed_fps_max_multiple"))


def _game_retained_history_samples(config: SimulationConfig) -> int:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    try:
        value = int(game_cfg.get("retained_history_samples", 4096))
    except (TypeError, ValueError):
        return 4096
    return int(max(2, value))


def _speed_dt_schedule_from_raw(raw: Any) -> tuple[tuple[float, float], ...]:
    if raw is None or raw is False:
        return ()
    if not isinstance(raw, dict):
        return tuple(SPEED_DT_SCHEDULE)
    rows: list[tuple[float, float]] = []
    for speed, dt_s in raw.items():
        try:
            speed_value = float(speed)
        except (TypeError, ValueError):
            continue
        dt_value = _positive_float_or_none(dt_s)
        if np.isfinite(speed_value) and speed_value > 0.0 and dt_value is not None:
            rows.append((speed_value, dt_value))
    return tuple(sorted(rows, key=lambda item: item[0]))


def _game_speed_dt_schedule(config: SimulationConfig) -> tuple[tuple[float, float], ...]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    if "speed_dt_s" not in game_cfg:
        return tuple(SPEED_DT_SCHEDULE)
    return _speed_dt_schedule_from_raw(game_cfg.get("speed_dt_s"))


def _game_coast_speed_dt_schedule(config: SimulationConfig) -> tuple[tuple[float, float], ...]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    if "coast_speed_dt_s" not in game_cfg:
        return _game_speed_dt_schedule(config)
    return _speed_dt_schedule_from_raw(game_cfg.get("coast_speed_dt_s"))


def _game_tick_dt_s_from_schedule(
    config: SimulationConfig,
    speed_multiple: float,
    schedule: tuple[tuple[float, float], ...],
    *,
    cap_to_base_dt: bool = True,
) -> float:
    base_dt_s = float(config.scenario.simulator.dt_s)
    if not schedule:
        return base_dt_s
    current_speed = float(speed_multiple)
    chosen = base_dt_s
    for threshold_speed, dt_s in schedule:
        if current_speed >= threshold_speed - 1.0e-9:
            chosen = float(dt_s)
    if bool(cap_to_base_dt):
        chosen = min(chosen, base_dt_s)
    return float(max(chosen, 1.0e-9))


def _game_tick_dt_s(config: SimulationConfig, speed_multiple: float) -> float:
    return _game_tick_dt_s_from_schedule(config, speed_multiple, _game_speed_dt_schedule(config))


def _game_coast_tick_dt_s(config: SimulationConfig, speed_multiple: float) -> float:
    return _game_tick_dt_s_from_schedule(
        config,
        speed_multiple,
        _game_coast_speed_dt_schedule(config),
        cap_to_base_dt=not _game_two_rail_speed_control_enabled(config),
    )


def _game_active_tick_dt_s(
    config: SimulationConfig,
    speed_multiple: float,
    *,
    maneuver_active: bool,
) -> float:
    if bool(maneuver_active):
        return _game_tick_dt_s(config, speed_multiple)
    return _game_coast_tick_dt_s(config, speed_multiple)


def _game_maneuver_input_max_pending_steps(config: SimulationConfig) -> int:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    try:
        value = int(game_cfg.get("maneuver_input_max_pending_steps"))
    except (TypeError, ValueError):
        return 1
    return max(value, 1)


def _game_timed_input_accumulator_enabled(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("timed_input_accumulator", True))


def _game_visual_extrapolation_enabled(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("visual_extrapolation_enabled", True))


def _game_two_rail_speed_control_enabled(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("two_rail_speed_control", False))


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
    return _game_plane_tuple(raw)


def _game_target_centered_plot_planes(config: SimulationConfig) -> tuple[str, ...]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = game_cfg.get("target_centered_plot_planes", ())
    return _game_plane_tuple(raw)


def _game_target_centered_plot_axes(config: SimulationConfig) -> dict[str, tuple[str, ...]]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = game_cfg.get("target_centered_plot_axes", {}) or {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, tuple[str, ...]] = {}
    for plane, value in raw.items():
        key = str(plane or "").strip().upper()
        if key not in {"RI", "RC", "IC"}:
            continue
        if isinstance(value, str):
            values = [value]
        else:
            try:
                values = list(value)
            except TypeError:
                continue
        axes: list[str] = []
        for raw_axis in values:
            axis = str(raw_axis or "").strip().lower()
            if axis in {"x", "y"} and axis not in axes:
                axes.append(axis)
        if axes:
            parsed[key] = tuple(axes)
    return parsed


def _game_plane_tuple(raw: Any) -> tuple[str, ...]:
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


def _game_camera_rule_mode(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    value = str(game_cfg.get("camera_rule_mode", "default") or "default").strip().lower()
    if value in {"full", "full_trajectory", "trajectory", "trail", "trail_projection"}:
        return "full_trajectory"
    if value in {"current_pair", "pair", "satellites", "satellites_only", "current"}:
        return "current_pair"
    return "default"


def _game_camera_rule_toggle_enabled(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("camera_rule_toggle_enabled", False))


def _game_level_title(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    title = str(game_cfg.get("level_name", "") or "").strip()
    if title:
        return title
    training_cfg = dict(game_cfg.get("training", {}) or {})
    scenario_id = str(training_cfg.get("scenario_id", config.scenario.scenario_name or "") or "")
    parts = scenario_id.split("_")
    if len(parts) >= 3 and parts[0] == "rpo" and parts[1].isdigit():
        return f"Level {int(parts[1])} - {' '.join(parts[2:]).title()}"
    return str(config.scenario.scenario_name or "Level").replace("_", " ").title()


def _game_show_target_hcw_path(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("show_target_hcw_path", False))


def _game_coast_prediction_model(config: SimulationConfig) -> str:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return str(game_cfg.get("coast_prediction_model", "hcw") or "hcw").strip().lower()


def _game_initial_speed_multiple(config: SimulationConfig, requested_speed_multiple: float | None) -> float:
    options = _game_speed_multiplier_options(config)
    if requested_speed_multiple is not None:
        return _coerce_speed_multiple(float(requested_speed_multiple), options=options)
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    configured = _positive_float_or_none(game_cfg.get("initial_speed_multiple"))
    return _coerce_speed_multiple(1.0 if configured is None else configured, options=options)


def _game_maneuver_control_speed_multiple(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("maneuver_control_speed_multiple"))


def _game_max_autonomy_step_s(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("max_autonomy_step_s"))


def _game_speed_multiplier_options(config: SimulationConfig) -> tuple[float, ...]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = game_cfg.get("speed_multiplier_options")
    if raw is None:
        return SPEED_MULTIPLIER_OPTIONS
    if isinstance(raw, (str, bytes)):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError:
            return SPEED_MULTIPLIER_OPTIONS
    parsed: list[float] = []
    for value in values:
        numeric = _positive_float_or_none(value)
        if numeric is not None and not any(np.isclose(numeric, existing) for existing in parsed):
            parsed.append(float(numeric))
    if not parsed:
        return SPEED_MULTIPLIER_OPTIONS
    return tuple(sorted(parsed))


def _game_sandbox_enabled(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    training_cfg = dict(game_cfg.get("training", {}) or {})
    return bool(game_cfg.get("sandbox", False) or training_cfg.get("sandbox_mode", False))


def _ric_primer_enabled(training_cfg: RPOTrainingConfig, *, arcade_enabled: bool = False) -> bool:
    if bool(arcade_enabled):
        return False
    return (
        bool(training_cfg.enabled)
        and str(training_cfg.scenario_id or "").strip() == "rpo_00_tutorial"
        and bool(training_cfg.guided_tutorial_burns)
        and not bool(getattr(training_cfg, "sandbox_mode", False))
    )


def _operator_tutorial_enabled(
    game_mode: str,
    training_cfg: RPOTrainingConfig,
    *,
    arcade_enabled: bool = False,
) -> bool:
    return bool(
        _normalize_game_mode(game_mode) == "operator"
        and not bool(arcade_enabled)
        and bool(training_cfg.enabled)
        and str(training_cfg.scenario_id or "").strip() == "rpo_00_tutorial"
        and not bool(getattr(training_cfg, "sandbox_mode", False))
    )


def _operator_tutorial_stages() -> tuple[OperatorTutorialStage, ...]:
    return (
        OperatorTutorialStage("plus_in_track", "+I Burn", 1, 1),
        OperatorTutorialStage("minus_in_track", "-I Burn", 1, -1),
        OperatorTutorialStage("plus_radial", "+R Burn", 0, 1),
        OperatorTutorialStage("minus_radial", "-R Burn", 0, -1),
        OperatorTutorialStage("plus_cross_track", "+C Burn", 2, 1),
        OperatorTutorialStage("minus_cross_track", "-C Burn", 2, -1),
    )


def _operator_tutorial_current_stage(runtime: OperatorTutorialRuntime) -> OperatorTutorialStage | None:
    stages = _operator_tutorial_stages()
    idx = int(runtime.stage_index)
    if idx < 0 or idx >= len(stages):
        return None
    return stages[idx]


def _operator_tutorial_demo_title(runtime: OperatorTutorialRuntime) -> str:
    stage = _operator_tutorial_current_stage(runtime)
    if stage is None:
        return "Operator Tutorial"
    return f"Demo {int(runtime.stage_index) + 1}/{len(_operator_tutorial_stages())}: {stage.display_label}"


def _operator_tutorial_status(runtime: OperatorTutorialRuntime) -> str:
    stage = _operator_tutorial_current_stage(runtime)
    if stage is None:
        return "Operator tutorial complete."
    return (
        f"{_operator_tutorial_demo_title(runtime)}. "
        f"Observe {OPERATOR_TUTORIAL_STAGE_DURATION_S:.0f}s at {OPERATOR_TUTORIAL_PLAYBACK_SPEED_MULTIPLE:.0f}x, "
        "then the next scripted burn will load."
    )

def _operator_tutorial_complete_score(score: Any) -> Any:
    try:
        elapsed_s = float(getattr(score, "elapsed_s", 0.0))
        return replace(
            score,
            achieved_time_s=elapsed_s,
            goal_met=True,
            level_passed=True,
            level_failed=False,
            pass_fail_reasons=("Operator tutorial complete.",),
            hints=(),
        )
    except (TypeError, ValueError):
        return score


def _sandbox_setup_from_config(config: SimulationConfig) -> SandboxSetupValues:
    chaser = config.scenario.objects.get("chaser")
    target = config.scenario.objects.get("target")
    rel_state = [0.0, -3.0, 0.0, 0.0, 0.0, 0.0]
    coes = {"a_km": 7000.0, "ecc": 0.0, "true_anomaly_deg": 0.0}
    if chaser is not None:
        rel = dict(chaser.initial_state.get("relative_to_target_ric", {}) or {})
        raw_state = list(rel.get("state", rel_state) or rel_state)
        if len(raw_state) >= 6:
            rel_state = [float(value) for value in raw_state[:6]]
    if target is not None:
        coes.update(dict(target.initial_state.get("coes", {}) or {}))
    return SandboxSetupValues(
        radial_km=float(rel_state[0]),
        in_track_km=float(rel_state[1]),
        cross_track_km=float(rel_state[2]),
        radial_rate_m_s=float(rel_state[3]) * 1000.0,
        in_track_rate_m_s=float(rel_state[4]) * 1000.0,
        cross_track_rate_m_s=float(rel_state[5]) * 1000.0,
        target_a_km=float(coes.get("a_km", 7000.0) or 7000.0),
        target_ecc=float(coes.get("ecc", 0.0) or 0.0),
        target_true_anomaly_deg=float(coes.get("true_anomaly_deg", 0.0) or 0.0),
    )


def _sandbox_coast_prediction_model(setup: SandboxSetupValues) -> str:
    return "hcw" if abs(float(setup.target_ecc)) <= 1.0e-12 else "tschauner_hempel"


def _apply_sandbox_setup_to_config(config: SimulationConfig, setup: SandboxSetupValues) -> SimulationConfig:
    root = config.to_dict()
    game = root.setdefault("metadata", {}).setdefault("game", {})
    game["coast_prediction_model"] = _sandbox_coast_prediction_model(setup)
    game["sandbox"] = True
    game["target_centered_plot_planes"] = ["RI", "RC"]
    game.setdefault("camera_rule_mode", "full_trajectory")
    training = game.setdefault("training", {})
    training["sandbox_mode"] = True
    training["max_time_s"] = 20000.0
    training.pop("max_delta_v_m_s", None)
    simulator = root.setdefault("simulator", {})
    simulator["duration_s"] = 20000.0
    simulator["dt_s"] = 1.0
    chaser = root.setdefault("objects", {}).setdefault("chaser", {})
    chaser_initial = chaser.setdefault("initial_state", {})
    chaser_relative = dict(chaser_initial.get("relative_to_target_ric", {}) or {})
    chaser_relative["frame"] = "rect"
    chaser_relative["state"] = setup.relative_ric_state_km_s
    chaser_initial["relative_to_target_ric"] = chaser_relative
    chaser_initial.setdefault("relative_to", "target")
    target = root["objects"].setdefault("target", {})
    target_initial = target.setdefault("initial_state", {})
    target_coes = dict(target_initial.get("coes", {}) or {})
    target_coes["a_km"] = float(setup.target_a_km)
    target_coes["ecc"] = float(setup.target_ecc)
    target_coes["true_anomaly_deg"] = float(setup.target_true_anomaly_deg)
    target_initial["coes"] = target_coes
    return SimulationConfig.from_dict(root)


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
    config: SimulationConfig,
    training_cfg: RPOTrainingConfig,
    *,
    difficulty: str,
    game_mode: str = "pilot",
    frame_convention: FrameConvention | dict[str, Any] | None = None,
    operator_burn_plan: OperatorBurnPlan | None = None,
) -> tuple[str, ...]:
    if not training_cfg.enabled:
        return ()
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = dict(game_cfg.get("training", {}) or {})
    mode_key = _normalize_game_mode(game_mode)
    lines = [
        str(training_cfg.scenario_id or config.scenario.scenario_name or "RPO training"),
        (
            f"Actuator Error: {_operator_actuator_error_fraction(difficulty) * 100.0:g}%"
            if mode_key == "operator"
            else f"Assists: {str(difficulty or 'easy').title()}"
        ),
        f"Mode: {_display_game_mode(game_mode)}",
    ]
    if mode_key == "operator":
        lines.extend(f"Operator: {line}" for line in operator_plan_summary(operator_burn_plan or OperatorBurnPlan()))
    if training_cfg.learning_goal:
        lines.append(f"Objective: {training_cfg.learning_goal}")
    player_brief = str(raw.get("player_brief", "") or "").strip()
    if player_brief:
        lines.append(f"Plan: {player_brief}")
    axis_descriptions = dict(raw.get("axis_descriptions", {}) or {})
    axis_labels = (("radial", "R"), ("in_track", "I"), ("cross_track", "C"))
    for axis, short_label in axis_labels:
        text = str(axis_descriptions.get(axis, "") or "").strip()
        if text:
            lines.append(f"Axis {short_label}: {text}")
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
    if str(control_mode or "").strip().lower() in TRANSLATION_CONTROL_MODES:
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

__all__ = [name for name in globals() if not name.startswith("__")]
