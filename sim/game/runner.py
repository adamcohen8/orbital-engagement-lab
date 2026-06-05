from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from sim.api import SimulationConfig, SimulationSession
from sim.config import object_section
from sim.game import input as game_input
from sim.game import recording_controller as game_recording
from sim.game.arcade import (
    _arcade_mission_metrics,
    _arcade_round_briefing_lines,
    _arcade_round_initial_state_rng,
    _arcade_round_is_boss,
    _arcade_round_music_track,
    _arcade_round_rng,
    _arcade_round_simulation_config,
    _arcade_round_time_bonus_s,
    _arcade_round_training_config,
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
    GAME_MUSIC_DIR,
    _stop_game_music,
)
from sim.game.audio_controller import GameAudioController
from sim.game.debrief import (
    game_debrief_path,
    next_game_debrief_attempt_index,
    open_game_debrief_folder,
    tracker_replay_history,
    write_game_debrief,
)
from sim.game.defensive_target import DefensiveTargetIntentProvider
from sim.game.formatting import format_distance_km, format_speed_km_s, format_speed_m_s
from sim.game.manual import KeyboardCommandState, ManualGameCommandProvider
from sim.game.recording_controller import GameClipRecordingController, GameRecordingController
from sim.game.session import (
    _attempt_config_for_training_clock,
    _install_chaser_delta_v_limiter,
)
from sim.game.state import (
    GamePhase,
    mission_state_for_dashboard,
    phase_from_score,
    phase_is_terminal,
    phase_shows_briefing,
)
from sim.game.training import RPOTrainingConfig, RPOTrainingTracker
from sim.game.tuning import (
    BRIEFING_SCROLL_STEP_PX,
    DASHBOARD_FPS,
    GAME_RECORDING_FPS,
    HIGH_SPEED_DASHBOARD_FPS,
    MANEUVER_CONTROL_SPEED,
    MAX_REALTIME_STEPS_PER_FRAME,
    MEDIUM_HIGH_SPEED_DASHBOARD_FPS,
    SPEED_MULTIPLIER_OPTIONS,
)
from sim.presets.thrusters import resolve_thruster_max_thrust_n_from_specs

FULL_ATTEMPT_RECORDING_PAD_S = 3.0
RIC_PRIMER_STAGE_COUNT = 3


@dataclass(frozen=True)
class GameRunResult:
    config_path: Path
    difficulty: str
    level_passed: bool
    arcade_score: int = 0
    arcade_seed: int | None = None
    recording_path: Path | None = None
    debrief_path: Path | None = None


@dataclass
class GuidedTutorialRuntime:
    stage_index: int = 0
    active_stage_delta_v_m_s: float = 0.0
    stage_start_rel_ric: np.ndarray | None = None
    stage_start_mean_motion_rad_s: float | None = None
    awaiting_speed_step: bool = False
    wrong_key_active: bool = False


@dataclass
class RICPrimerRuntime:
    stage_index: int = 0
    elapsed_s: float = 0.0

    def reset(self) -> None:
        self.stage_index = 0
        self.elapsed_s = 0.0


@dataclass(frozen=True)
class SandboxSetupValues:
    radial_km: float = 0.0
    in_track_km: float = -3.0
    cross_track_km: float = 0.0
    radial_rate_m_s: float = 0.0
    in_track_rate_m_s: float = 0.0
    cross_track_rate_m_s: float = 0.0
    target_a_km: float = 7000.0
    target_ecc: float = 0.0
    target_true_anomaly_deg: float = 0.0

    @property
    def relative_ric_state_km_s(self) -> list[float]:
        return [
            float(self.radial_km),
            float(self.in_track_km),
            float(self.cross_track_km),
            float(self.radial_rate_m_s) / 1000.0,
            float(self.in_track_rate_m_s) / 1000.0,
            float(self.cross_track_rate_m_s) / 1000.0,
        ]


_SANDBOX_SETUP_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("Radial R", "km", "radial_km"),
    ("In-Track I", "km", "in_track_km"),
    ("Cross-Track C", "km", "cross_track_km"),
    ("Radial Rate dR", "m/s", "radial_rate_m_s"),
    ("In-Track Rate dI", "m/s", "in_track_rate_m_s"),
    ("Cross-Track Rate dC", "m/s", "cross_track_rate_m_s"),
    ("Target Semimajor Axis", "km", "target_a_km"),
    ("Target Eccentricity", "", "target_ecc"),
    ("Target True Anomaly", "deg", "target_true_anomaly_deg"),
)


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


def _game_plot_prediction_in_zoom(config: SimulationConfig) -> bool:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return bool(game_cfg.get("plot_prediction_in_zoom", False))


def _game_plot_prediction_zoom_max_span_km(config: SimulationConfig) -> float | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return _positive_float_or_none(game_cfg.get("plot_prediction_zoom_max_span_km"))


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
    if speed > MANEUVER_CONTROL_SPEED and _has_maneuver_input(state, control_mode=control_mode):
        return MANEUVER_CONTROL_SPEED
    return speed


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
) -> tuple[SimulationSession, Any]:
    command_state.reset_axes()
    session, _, snapshot = _start_game_attempt(
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
    _sync_dashboard_training_config(dashboard, training_cfg)
    _sync_dashboard_round_config(dashboard, attempt_config)
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
    return SandboxSetupValues(**parsed), ""


def _sandbox_setup_briefing_lines(values: list[str], *, active_index: int, error: str = "") -> tuple[str, ...]:
    lines = [
        "Sandbox Setup",
        "Edit the initial relative state and target orbit, then press Enter or Space to launch.",
        "Positions are km. Relative rates are m/s. Target anomaly is degrees.",
    ]
    if error:
        lines.append(f"Input Error: {error}")
    for idx, (label, unit, _) in enumerate(_SANDBOX_SETUP_FIELDS):
        marker = ">" if idx == int(active_index) else " "
        suffix = f" {unit}" if unit else ""
        value = values[idx] if idx < len(values) else ""
        lines.append(f"{marker} {label}: {value}{suffix}")
    lines.append("Tab/Up/Down Change Field. Backspace Edits. Enter Starts. Esc Cancels.")
    return tuple(lines)


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
    while not getattr(dashboard, "closed", False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == getattr(pygame, "MOUSEWHEEL", object()):
                dashboard.scroll_briefing(-int(getattr(event, "y", 0)) * BRIEFING_SCROLL_STEP_PX)
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
            if key in {pygame.K_TAB, pygame.K_DOWN}:
                active_idx = (active_idx + 1) % len(values)
                error = ""
                continue
            if key == pygame.K_UP:
                active_idx = (active_idx - 1) % len(values)
                error = ""
                continue
            if key == getattr(pygame, "K_PAGEUP", object()):
                dashboard.scroll_briefing(-BRIEFING_SCROLL_STEP_PX * 4)
                continue
            if key == getattr(pygame, "K_PAGEDOWN", object()):
                dashboard.scroll_briefing(BRIEFING_SCROLL_STEP_PX * 4)
                continue
            if key == getattr(pygame, "K_HOME", object()):
                dashboard.scroll_briefing(-1000000)
                continue
            if key == getattr(pygame, "K_END", object()):
                dashboard.scroll_briefing(1000000)
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
        dashboard.draw(
            command_status="Sandbox Setup",
            coach_hint=error or "Choose a starting state, then launch and experiment freely.",
            mission_state="active",
            level_title=level_title,
            mission_metrics=("INFO Sandbox",),
            objective_checklist=(),
            speed_multiple=speed_multiple,
            briefing_lines=_sandbox_setup_briefing_lines(values, active_index=active_idx, error=error),
        )
        dashboard.tick(30.0)
    return None


def _camera_rule_status(dashboard: Any, training_cfg: RPOTrainingConfig) -> str:
    if not bool(getattr(training_cfg, "sandbox_mode", False)):
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
    recording_fps: float = GAME_RECORDING_FPS,
) -> float:
    if bool(recording):
        return float(max(recording_fps, 1.0))
    if float(speed_multiple) >= 100.0:
        return HIGH_SPEED_DASHBOARD_FPS
    if float(speed_multiple) >= 50.0:
        return MEDIUM_HIGH_SPEED_DASHBOARD_FPS
    return DASHBOARD_FPS


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
            "W/S Radial +/-R  A/D In-Track +/-I  Left/Right Cross-Track +/-C  M Music\n"
            "Use small pulses, then coast and watch the target-centered RIC motion.\n"
            f"{sim_state}  R={state.pitch:+.0f} I={state.yaw:+.0f} C={state.roll:+.0f} Throttle={state.throttle:.2f}"
        )
    burn = "FIRE" if state.firing else "Coast"
    return (
        "W/S Pitch  A/D Yaw  Left/Right Roll  Space Fire  M Music  R Reset  Esc Quit\n"
        "Keys work in the figure window or this terminal; terminal input is pulse/repeat based.\n"
        f"Pitch={state.pitch:+.0f} Yaw={state.yaw:+.0f} Roll={state.roll:+.0f} Thrust={burn}"
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
    recording_fps: float = GAME_RECORDING_FPS,
    arcade_seed: int | None = None,
    debrief_output_dir: str | Path | None = None,
) -> GameRunResult:
    from sim.game.pygame_dashboard import PygameRPODashboard

    config = _force_game_acceleration_off_config(SimulationConfig.from_yaml(config_path))
    controlled_object_id = _game_controlled_object_id(config, default=controlled_object_id or "chaser")
    control_mode = _game_control_mode(config)
    difficulty = str(difficulty_override or _game_difficulty(config)).strip().lower()
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    base_training_cfg = training_cfg
    arcade_enabled = _game_arcade_enabled(config)
    arcade_seed_value = _new_arcade_seed() if arcade_enabled and arcade_seed is None else arcade_seed
    arcade_round_index = 1
    arcade_total_score = 0
    arcade_remaining_time_s = _game_arcade_initial_time_s(config, training_cfg) if arcade_enabled else None
    if arcade_enabled:
        training_cfg = _arcade_round_training_config(
            config,
            training_cfg,
            round_index=arcade_round_index,
            max_time_s=arcade_remaining_time_s,
        )
    debrief_enabled = _game_debrief_enabled(
        config,
        training_cfg,
        arcade_enabled=arcade_enabled,
    )
    attempt_config = _arcade_round_simulation_config(
        config,
        training_cfg,
        round_index=arcade_round_index,
        rng=(
            _arcade_round_initial_state_rng(int(arcade_seed_value), arcade_round_index)
            if arcade_enabled and arcade_seed_value is not None
            else None
        ),
    )
    ric_reference_object_id = _game_ric_reference_object_id(config, training_cfg.target_object_id)
    level_title = _game_level_title(config)
    current_speed_multiple = _coerce_speed_multiple(speed_multiple)
    trainer = RPOTrainingTracker(training_cfg)
    command_state = KeyboardCommandState()
    command_state.use_timing_accumulator = True
    guided_tutorial = GuidedTutorialRuntime()
    ric_primer = RICPrimerRuntime()
    ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled)
    command_state.paused = bool(training_cfg.enabled)
    phase = GamePhase.BRIEFING if training_cfg.enabled else GamePhase.PLAYING
    briefing_lines = _training_briefing_lines(config, training_cfg, difficulty=difficulty)
    session, _, snapshot = _start_game_attempt(
        attempt_config,
        command_state=command_state,
        training_cfg=training_cfg,
        controlled_object_id=controlled_object_id,
        attitude_rate_deg_s=attitude_rate_deg_s,
        control_mode=control_mode,
        ric_reference_object_id=ric_reference_object_id,
        defensive_target_provider=(
            _game_random_direction_defensive_target_provider(
                config,
                round_index=arcade_round_index,
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
        hard_speed_limit_radius_km=training_cfg.hard_speed_limit_radius_km,
        hard_speed_limit_km_s=training_cfg.hard_speed_limit_km_s,
        goal_relative_ric_km=training_cfg.goal_relative_ric_km,
        goal_nmt_radial_amplitude_km=training_cfg.goal_nmt_radial_amplitude_km,
        goal_nmt_cross_track_amplitude_km=training_cfg.goal_nmt_cross_track_amplitude_km,
        goal_nmt_cross_track_phase_deg=training_cfg.goal_nmt_cross_track_phase_deg,
        goal_nmt_center_ric_km=training_cfg.goal_nmt_center_ric_km,
        goal_nmt_element_tolerance_km=training_cfg.goal_nmt_element_tolerance_km,
        coast_prediction_orbit_fraction=_coast_prediction_orbit_fraction(difficulty),
        coast_prediction_model=_game_coast_prediction_model(attempt_config),
        forbidden_regions=training_cfg.forbidden_regions,
        approach_gates=training_cfg.approach_gates,
        inspection_gates=training_cfg.inspection_gates,
        plot_overlays_in_zoom=_game_plot_overlays_in_zoom(config),
        plot_overlays_in_zoom_by_plane=_game_plot_overlays_in_zoom_by_plane(config),
        plot_prediction_in_zoom=_game_plot_prediction_in_zoom(config),
        plot_prediction_zoom_max_span_km=_game_plot_prediction_zoom_max_span_km(config),
        plot_axis_scale=_game_plot_axis_scale(config),
        plot_fixed_axis_half_span_km=_game_plot_fixed_axis_half_span_km(config),
        plot_equal_axis_scale_planes=_game_plot_equal_axis_scale_planes(config),
        target_centered_plot_planes=_game_target_centered_plot_planes(config),
        target_centered_plot_axes=_game_target_centered_plot_axes(config),
        proximity_ring_plot_planes=_game_proximity_ring_plot_planes(config),
        camera_mode=_game_camera_mode(config),
        camera_rule_mode=_game_camera_rule_mode(config),
        show_target_coast_prediction=_game_show_target_hcw_path(config),
        fullscreen=True,
    )
    _sync_dashboard_training_config(dashboard, training_cfg)
    _sync_dashboard_round_config(dashboard, attempt_config)
    recording_attempt = 1
    recording_path: Path | None = None
    debrief_path: Path | None = None
    debrief_folder_to_open: Path | None = None
    recording_controller = GameRecordingController(
        enabled=record_video,
        config=config,
        difficulty=difficulty,
        attempt_index=recording_attempt,
        output_dir=recording_output_dir,
        fps=recording_fps,
    )
    clip_recording_controller = GameClipRecordingController(
        config=config,
        difficulty=difficulty,
        output_dir=recording_output_dir,
        fps=recording_fps,
    )
    clip_recording_started_wall: float | None = None
    clip_recording_status_message = ""
    clip_recording_status_until = 0.0
    audio_controller: GameAudioController | None = None
    try:
        if _game_sandbox_enabled(config):
            dashboard.push_snapshot(snapshot)
            setup = _run_sandbox_setup_form(
                dashboard,
                config=config,
                speed_multiple=current_speed_multiple,
                level_title=level_title,
            )
            if setup is None:
                training_cfg = RPOTrainingConfig(enabled=False)
                return GameRunResult(
                    config_path=Path(config_path),
                    difficulty=difficulty,
                    level_passed=False,
                    arcade_score=0,
                    arcade_seed=arcade_seed_value if arcade_enabled else None,
                )
            config = _apply_sandbox_setup_to_config(config, setup)
            training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
            base_training_cfg = training_cfg
            debrief_enabled = _game_debrief_enabled(
                config,
                training_cfg,
                arcade_enabled=arcade_enabled,
            )
            attempt_config = _arcade_round_simulation_config(
                config,
                training_cfg,
                round_index=arcade_round_index,
                rng=(
                    _arcade_round_initial_state_rng(int(arcade_seed_value), arcade_round_index)
                    if arcade_enabled and arcade_seed_value is not None
                    else None
                ),
            )
            ric_reference_object_id = _game_ric_reference_object_id(config, training_cfg.target_object_id)
            level_title = _game_level_title(config)
            trainer = RPOTrainingTracker(training_cfg)
            guided_tutorial = GuidedTutorialRuntime()
            ric_primer = RICPrimerRuntime()
            ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled)
            command_state.paused = bool(training_cfg.enabled)
            phase = GamePhase.BRIEFING if training_cfg.enabled else GamePhase.PLAYING
            briefing_lines = _training_briefing_lines(config, training_cfg, difficulty=difficulty)
            session, _, snapshot = _start_game_attempt(
                attempt_config,
                command_state=command_state,
                training_cfg=training_cfg,
                controlled_object_id=controlled_object_id,
                attitude_rate_deg_s=attitude_rate_deg_s,
                control_mode=control_mode,
                ric_reference_object_id=ric_reference_object_id,
                defensive_target_provider=(
                    _game_random_direction_defensive_target_provider(
                        config,
                        round_index=arcade_round_index,
                        rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
                    )
                    if arcade_enabled and arcade_seed_value is not None
                    else None
                ),
            )
            dashboard.clear()
            dashboard.reference_object_id = ric_reference_object_id
            _sync_dashboard_training_config(dashboard, training_cfg)
            _sync_dashboard_round_config(dashboard, attempt_config)
            dashboard.camera_rule_mode = _game_camera_rule_mode(config)
            recording_controller = GameRecordingController(
                enabled=record_video,
                config=config,
                difficulty=difficulty,
                attempt_index=recording_attempt,
                output_dir=recording_output_dir,
                fps=recording_fps,
            )
            clip_recording_controller = GameClipRecordingController(
                config=config,
                difficulty=difficulty,
                output_dir=recording_output_dir,
                fps=recording_fps,
            )
            clip_recording_started_wall = None
            clip_recording_status_message = ""
            clip_recording_status_until = 0.0
        recording_controller.start()
        dashboard.push_snapshot(snapshot)
        trainer.record(snapshot)
        _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
        score = trainer.score()
        phase = phase_from_score(score, briefing_open=phase_shows_briefing(phase), paused=command_state.paused)
        dashboard.draw(
            command_status=_command_status(command_state, control_mode=control_mode),
            coach_hint=_coach_hint_with_camera_rule(
                _guided_tutorial_stage_hint(
                    _guided_tutorial_current_stage(training_cfg, guided_tutorial), guided_tutorial
                )
                or trainer.current_hint(),
                dashboard,
                training_cfg,
            ),
            mission_state=mission_state_for_dashboard(phase),
            level_title=level_title,
            mission_metrics=_arcade_mission_metrics(
                _mission_metrics(training_cfg, score),
                enabled=arcade_enabled,
                round_index=arcade_round_index,
                total_score=arcade_total_score,
                is_boss=_arcade_round_is_boss(config, arcade_round_index),
            ),
            objective_checklist=_mission_checklist(training_cfg, score),
            speed_multiple=current_speed_multiple,
            recording_status=_clip_recording_status(
                clip_recording_controller,
                started_wall_s=clip_recording_started_wall,
                now_wall_s=perf_counter(),
                status_message=clip_recording_status_message,
                status_until_wall_s=clip_recording_status_until,
            ),
            briefing_lines=briefing_lines if phase_shows_briefing(phase) else (),
            debrief_lines=_score_debrief_lines(score, config=training_cfg, difficulty=difficulty),
            debrief_available=debrief_enabled,
        )
        recording_controller.capture(dashboard)
        if phase_shows_briefing(phase):
            recording_controller.capture_hold(dashboard, duration_s=FULL_ATTEMPT_RECORDING_PAD_S)
        clip_recording_controller.capture(dashboard)

        pygame = dashboard.pygame
        audio_controller = GameAudioController(pygame=pygame, music_enabled=music_enabled)
        audio_controller.sync(
            score,
            training_cfg=training_cfg,
            override_level_path=_arcade_round_music_path(config, arcade_round_index) if arcade_enabled else None,
        )
        dt_s = float(config.scenario.simulator.dt_s)
        wall_step_s = _wall_step_s(dt_s, current_speed_multiple)
        last_step_wall = perf_counter()
        last_input_wall = last_step_wall
        while (not command_state.quit_requested) and (not dashboard.closed):
            briefing_open = phase_shows_briefing(phase)
            debrief_hotkey_enabled = phase_is_terminal(phase)
            _poll_pygame_input(pygame, command_state, control_mode=control_mode, briefing_open=briefing_open)
            if not debrief_hotkey_enabled:
                command_state.open_debrief_requested = False
            input_now = perf_counter()
            input_elapsed_wall = max(float(input_now) - float(last_input_wall), 0.0)
            last_input_wall = input_now
            if command_state.quit_requested:
                break
            if briefing_open and command_state.briefing_scroll_px:
                dashboard.scroll_briefing(command_state.briefing_scroll_px)
            if briefing_open and not command_state.paused:
                if ric_primer_enabled:
                    phase = GamePhase.PRIMER
                    ric_primer.reset()
                    command_state.paused = True
                    command_state.step_requested = False
                    command_state.reset_axes()
                else:
                    phase = GamePhase.PLAYING
                dashboard.reset_briefing_scroll()
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            if command_state.music_toggle_requested:
                command_state.music_toggle_requested = False
                audio_controller.toggle(
                    trainer.score(),
                    training_cfg=training_cfg,
                    override_level_path=(
                        _arcade_round_music_path(config, arcade_round_index) if arcade_enabled else None
                    ),
                )
            if phase == GamePhase.PRIMER:
                if command_state.restart_requested:
                    ric_primer.reset()
                    command_state.restart_requested = False
                    command_state.paused = True
                elif not command_state.paused:
                    ric_primer.stage_index += 1
                    ric_primer.elapsed_s = 0.0
                    if ric_primer.stage_index >= RIC_PRIMER_STAGE_COUNT:
                        phase = GamePhase.PLAYING
                        command_state.paused = _guided_tutorial_current_stage(
                            training_cfg,
                            guided_tutorial,
                        ) is not None
                    else:
                        command_state.paused = True
                if phase == GamePhase.PRIMER:
                    ric_primer.elapsed_s += input_elapsed_wall
                    command_state.reset_axes()
                    command_state.step_requested = False
                    command_state.speed_multiplier_change = 0
                    command_state.camera_rule_toggle_requested = False
                    command_state.clip_record_toggle_requested = False
                    command_state.clip_record_save_requested = False
                    command_state.open_debrief_requested = False
                    dashboard.draw_ric_primer(
                        stage_index=ric_primer.stage_index,
                        elapsed_s=ric_primer.elapsed_s,
                        recording_status=_clip_recording_status(
                            clip_recording_controller,
                            started_wall_s=clip_recording_started_wall,
                            now_wall_s=perf_counter(),
                            status_message=clip_recording_status_message,
                            status_until_wall_s=clip_recording_status_until,
                        ),
                    )
                    recording_controller.capture(dashboard)
                    clip_recording_controller.capture(dashboard)
                    dashboard.tick(_dashboard_fps_for_speed(current_speed_multiple))
                    continue
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
                last_input_wall = last_step_wall
            if command_state.camera_rule_toggle_requested:
                if bool(getattr(training_cfg, "sandbox_mode", False)) and hasattr(dashboard, "toggle_camera_rule_mode"):
                    dashboard.toggle_camera_rule_mode()
                command_state.camera_rule_toggle_requested = False
            if guided_tutorial.awaiting_speed_step and _guided_tutorial_speed_step_reached(
                training_cfg,
                current_speed_multiple,
            ):
                trainer.mark_guided_tutorial_speed_complete()
                guided_tutorial.awaiting_speed_step = False
                session, snapshot = _reset_guided_tutorial_stage_attempt(
                    attempt_config=attempt_config,
                    command_state=command_state,
                    trainer=trainer,
                    dashboard=dashboard,
                    training_cfg=training_cfg,
                    controlled_object_id=controlled_object_id,
                    attitude_rate_deg_s=attitude_rate_deg_s,
                    control_mode=control_mode,
                    ric_reference_object_id=ric_reference_object_id,
                )
                command_state.paused = _guided_tutorial_current_stage(training_cfg, guided_tutorial) is not None
                command_state.step_requested = False
                _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            maneuver_speed_multiple = _speed_after_maneuver_input(
                current_speed_multiple,
                command_state,
                control_mode=control_mode,
            )
            if not np.isclose(maneuver_speed_multiple, current_speed_multiple):
                current_speed_multiple = maneuver_speed_multiple
                wall_step_s = _wall_step_s(dt_s, current_speed_multiple)
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            if command_state.clip_record_toggle_requested:
                command_state.clip_record_toggle_requested = False
                if clip_recording_controller.recording:
                    clip_recording_controller.discard()
                    clip_recording_started_wall = None
                    clip_recording_status_message = "Clip discarded"
                    clip_recording_status_until = perf_counter() + 2.5
                elif phase_shows_briefing(phase) or phase_is_terminal(phase):
                    clip_recording_status_message = "Clip starts during play"
                    clip_recording_status_until = perf_counter() + 2.5
                else:
                    recorder = clip_recording_controller.start_next()
                    if recorder is not None:
                        clip_recording_started_wall = perf_counter()
                        clip_recording_status_message = ""
                        clip_recording_status_until = 0.0
                    else:
                        clip_recording_started_wall = None
                        clip_recording_status_message = "Clip recording unavailable"
                        clip_recording_status_until = perf_counter() + 2.5
            if command_state.clip_record_save_requested:
                command_state.clip_record_save_requested = False
                if clip_recording_controller.recording:
                    clip_path = clip_recording_controller.finish(
                        base_training_cfg,
                        override_level_path=(
                            _arcade_round_music_path(config, arcade_round_index) if arcade_enabled else None
                        ),
                    )
                    clip_recording_started_wall = None
                    if clip_path is not None:
                        print(f"Saved game clip recording: {clip_path}")
                        clip_recording_status_message = "Clip saved"
                    else:
                        clip_recording_status_message = "Clip save failed"
                    clip_recording_status_until = perf_counter() + 2.5
                else:
                    clip_recording_status_message = "No active clip"
                    clip_recording_status_until = perf_counter() + 2.5
            if command_state.restart_requested:
                recording_controller.discard()
                clip_recording_controller.discard()
                clip_recording_started_wall = None
                clip_recording_status_message = ""
                clip_recording_status_until = 0.0
                audio_controller.stop()
                if arcade_enabled:
                    arcade_round_index = 1
                    arcade_total_score = 0
                    arcade_remaining_time_s = _game_arcade_initial_time_s(config, base_training_cfg)
                    training_cfg = _arcade_round_training_config(
                        config,
                        base_training_cfg,
                        round_index=arcade_round_index,
                        max_time_s=arcade_remaining_time_s,
                    )
                    attempt_config = _arcade_round_simulation_config(
                        config,
                        training_cfg,
                        round_index=arcade_round_index,
                        rng=(
                            _arcade_round_initial_state_rng(int(arcade_seed_value), arcade_round_index)
                            if arcade_seed_value is not None
                            else None
                        ),
                    )
                    trainer = RPOTrainingTracker(training_cfg)
                    guided_tutorial = GuidedTutorialRuntime()
                    ric_primer = RICPrimerRuntime()
                    ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled)
                recording_controller.restart()
                recording_attempt = recording_controller.attempt_index
                recording_path = None
                debrief_path = None
                session, _, snapshot = _start_game_attempt(
                    attempt_config,
                    command_state=command_state,
                    training_cfg=training_cfg,
                    controlled_object_id=controlled_object_id,
                    attitude_rate_deg_s=attitude_rate_deg_s,
                    control_mode=control_mode,
                    ric_reference_object_id=ric_reference_object_id,
                    defensive_target_provider=(
                        _game_random_direction_defensive_target_provider(
                            config,
                            round_index=arcade_round_index,
                            rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
                        )
                        if arcade_enabled and arcade_seed_value is not None
                        else None
                    ),
                )
                trainer.clear()
                guided_tutorial = GuidedTutorialRuntime()
                ric_primer = RICPrimerRuntime()
                ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled)
                dashboard.clear()
                _sync_dashboard_training_config(dashboard, training_cfg)
                _sync_dashboard_round_config(dashboard, attempt_config)
                dashboard.push_snapshot(snapshot)
                trainer.record(snapshot)
                _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
                command_state.restart_requested = False
                command_state.step_requested = False
                command_state.speed_multiplier_change = 0
                command_state.camera_rule_toggle_requested = False
                command_state.music_toggle_requested = False
                command_state.clip_record_toggle_requested = False
                command_state.clip_record_save_requested = False
                command_state.open_debrief_requested = False
                command_state.paused = bool(training_cfg.enabled)
                phase = GamePhase.BRIEFING if training_cfg.enabled else GamePhase.PLAYING
                dashboard.reset_briefing_scroll()
                dt_s = float(config.scenario.simulator.dt_s)
                wall_step_s = _wall_step_s(dt_s, current_speed_multiple)
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            now = perf_counter()
            pre_score = trainer.score()
            phase = phase_from_score(
                pre_score,
                briefing_open=phase_shows_briefing(phase),
                paused=command_state.paused,
            )
            mission_decided = phase_is_terminal(phase)
            if _game_loop_should_exit(session_done=session.done, score=pre_score):
                break
            if mission_decided:
                command_state.paused = True
            guided_stage = _guided_tutorial_current_stage(training_cfg, guided_tutorial)
            if (
                guided_stage is not None
                and not briefing_open
                and not mission_decided
                and not session.done
            ):
                guided_input_ok = _guided_tutorial_input_matches(command_state, guided_stage)
                guided_tutorial.wrong_key_active = _guided_tutorial_wrong_input_active(command_state, guided_stage)
                command_state.paused = not guided_input_ok
                if not guided_input_ok:
                    command_state.step_requested = False
            elif guided_tutorial.awaiting_speed_step and not briefing_open and not mission_decided:
                guided_tutorial.wrong_key_active = False
                command_state.reset_axes()
                command_state.paused = True
                command_state.step_requested = False
            else:
                guided_tutorial.wrong_key_active = False
            if (
                not briefing_open
                and not mission_decided
                and not session.done
                and not command_state.paused
            ):
                command_state.accumulate_timed_input(
                    input_elapsed_wall,
                    speed_multiple=current_speed_multiple,
                    control_mode=control_mode,
                )
            elif command_state.paused and not command_state.step_requested:
                command_state.clear_timed_input()
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
            guided_stage_completed = False
            completed_guided_stage: Any | None = None
            if (
                steps_to_run > 0
                and _guided_tutorial_current_stage(training_cfg, guided_tutorial) is not None
                and not bool(getattr(score, "level_failed", False))
            ):
                completed_guided_stage = _guided_tutorial_current_stage(training_cfg, guided_tutorial)
                guided_stage_completed = _guided_tutorial_complete_active_stage(
                    trainer,
                    training_cfg,
                    guided_tutorial,
                )
                score = trainer.score()
            if guided_stage_completed:
                if (
                    _guided_tutorial_speed_step_follows_burn(training_cfg, completed_guided_stage)
                    and not trainer.guided_tutorial_speed_satisfied()
                ):
                    guided_tutorial.awaiting_speed_step = True
                    guided_tutorial.wrong_key_active = False
                    session, _ = _reset_guided_tutorial_stage_attempt(
                        attempt_config=attempt_config,
                        command_state=command_state,
                        trainer=trainer,
                        dashboard=dashboard,
                        training_cfg=training_cfg,
                        controlled_object_id=controlled_object_id,
                        attitude_rate_deg_s=attitude_rate_deg_s,
                        control_mode=control_mode,
                        ric_reference_object_id=ric_reference_object_id,
                    )
                    command_state.reset_axes()
                    command_state.paused = True
                    command_state.step_requested = False
                    _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
                    score = trainer.score()
                    last_step_wall = perf_counter()
                    last_input_wall = last_step_wall
                else:
                    session, snapshot = _reset_guided_tutorial_stage_attempt(
                        attempt_config=attempt_config,
                        command_state=command_state,
                        trainer=trainer,
                        dashboard=dashboard,
                        training_cfg=training_cfg,
                        controlled_object_id=controlled_object_id,
                        attitude_rate_deg_s=attitude_rate_deg_s,
                        control_mode=control_mode,
                        ric_reference_object_id=ric_reference_object_id,
                    )
                    command_state.paused = _guided_tutorial_current_stage(training_cfg, guided_tutorial) is not None
                    guided_tutorial.wrong_key_active = False
                    command_state.step_requested = False
                    _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
                    score = trainer.score()
                    last_step_wall = perf_counter()
                    last_input_wall = last_step_wall
            command_state.step_requested = False
            if score.level_passed or score.level_failed:
                command_state.paused = True
                phase = phase_from_score(score, paused=True)
            else:
                phase = phase_from_score(
                    score,
                    briefing_open=phase_shows_briefing(phase),
                    paused=command_state.paused,
                )
            if arcade_enabled and bool(getattr(score, "level_passed", False)):
                audio_controller.play_round_clear()
                round_score = _arcade_round_weighted_score(
                    training_cfg,
                    score,
                    difficulty=difficulty,
                    round_index=arcade_round_index,
                    arcade_config=config,
                )
                arcade_total_score += int(round_score)
                time_used = _score_time_used_s(score)
                round_bonus_s = _arcade_round_time_bonus_s(
                    config,
                    training_cfg,
                    score,
                    round_index=arcade_round_index,
                )
                assert arcade_remaining_time_s is not None
                arcade_remaining_time_s = (
                    max(float(arcade_remaining_time_s) - time_used, 0.0)
                    + round_bonus_s
                )
                cleared_round_index = arcade_round_index
                arcade_round_index += 1
                training_cfg = _arcade_round_training_config(
                    config,
                    base_training_cfg,
                    round_index=arcade_round_index,
                    max_time_s=arcade_remaining_time_s,
                )
                attempt_config = _arcade_round_simulation_config(
                    config,
                    training_cfg,
                    round_index=arcade_round_index,
                    rng=_arcade_round_initial_state_rng(int(arcade_seed_value), arcade_round_index),
                )
                trainer = RPOTrainingTracker(training_cfg)
                guided_tutorial = GuidedTutorialRuntime()
                ric_primer = RICPrimerRuntime()
                ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled)
                session, _, snapshot = _start_game_attempt(
                    attempt_config,
                    command_state=command_state,
                    training_cfg=training_cfg,
                    controlled_object_id=controlled_object_id,
                    attitude_rate_deg_s=attitude_rate_deg_s,
                    control_mode=control_mode,
                    ric_reference_object_id=ric_reference_object_id,
                    defensive_target_provider=_game_random_direction_defensive_target_provider(
                        config,
                        round_index=arcade_round_index,
                        rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
                    ),
                )
                dashboard.clear()
                _sync_dashboard_training_config(dashboard, training_cfg)
                _sync_dashboard_round_config(dashboard, attempt_config)
                dashboard.push_snapshot(snapshot)
                trainer.record(snapshot)
                _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
                score = trainer.score()
                briefing_lines = _arcade_round_briefing_lines(
                    cleared_round_index=cleared_round_index,
                    next_round_index=arcade_round_index,
                    round_score=int(round_score),
                    total_score=arcade_total_score,
                    time_used_s=time_used,
                    bonus_time_s=round_bonus_s,
                    next_time_budget_s=arcade_remaining_time_s,
                    next_goal_range_km=training_cfg.goal_range_km,
                    next_is_boss=_arcade_round_is_boss(config, arcade_round_index),
                )
                command_state.paused = True
                command_state.step_requested = False
                command_state.restart_requested = False
                command_state.speed_multiplier_change = 0
                command_state.music_toggle_requested = False
                command_state.open_debrief_requested = False
                phase = GamePhase.ARCADE_TRANSITION
                dashboard.reset_briefing_scroll()
                audio_controller.clear_active_path()
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            recording_music_path = _arcade_round_music_path(config, arcade_round_index) if arcade_enabled else None
            audio_controller.sync(
                score,
                training_cfg=training_cfg,
                override_level_path=recording_music_path,
            )
            phase = phase_from_score(
                score,
                briefing_open=phase_shows_briefing(phase),
                paused=command_state.paused,
            )
            _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
            dashboard.draw(
                command_status=_command_status(command_state, control_mode=control_mode),
                coach_hint=_coach_hint_with_camera_rule(
                    (
                        _guided_tutorial_speed_step_hint(training_cfg, current_speed_multiple)
                        if guided_tutorial.awaiting_speed_step
                        else ""
                    )
                    or _guided_tutorial_stage_hint(
                        _guided_tutorial_current_stage(training_cfg, guided_tutorial), guided_tutorial
                    )
                    or trainer.current_hint(),
                    dashboard,
                    training_cfg,
                ),
                mission_state=mission_state_for_dashboard(phase),
                level_title=level_title,
                mission_metrics=_arcade_mission_metrics(
                    _mission_metrics(training_cfg, score),
                    enabled=arcade_enabled,
                    round_index=arcade_round_index,
                    total_score=arcade_total_score,
                    is_boss=_arcade_round_is_boss(config, arcade_round_index),
                ),
                objective_checklist=_mission_checklist(training_cfg, score),
                speed_multiple=current_speed_multiple,
                recording_status=_clip_recording_status(
                    clip_recording_controller,
                    started_wall_s=clip_recording_started_wall,
                    now_wall_s=perf_counter(),
                    status_message=clip_recording_status_message,
                    status_until_wall_s=clip_recording_status_until,
                ),
                briefing_lines=briefing_lines if phase_shows_briefing(phase) else (),
                debrief_lines=_score_debrief_lines(score, config=training_cfg, difficulty=difficulty),
                debrief_available=debrief_enabled,
            )
            recording_controller.capture(dashboard)
            clip_recording_controller.capture(dashboard)
            recorder = recording_controller.recorder
            if recorder is not None and (score.level_passed or score.level_failed) and not recorder.saved:
                recording_controller.capture_hold(dashboard, duration_s=FULL_ATTEMPT_RECORDING_PAD_S)
                recording_path = recording_controller.finish(
                    base_training_cfg,
                    override_level_path=recording_music_path,
                )
                if recording_path is not None:
                    print(f"Saved game recording: {recording_path}")
                else:
                    recorder = None
            if debrief_enabled and (score.level_passed or score.level_failed) and debrief_path is None:
                debrief_attempt = next_game_debrief_attempt_index(
                    scenario_id=training_cfg.scenario_id,
                    output_dir=debrief_output_dir,
                )
                debrief_path = write_game_debrief(
                    game_debrief_path(
                        scenario_id=training_cfg.scenario_id,
                        difficulty=difficulty,
                        attempt_index=debrief_attempt,
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
            if command_state.open_debrief_requested and phase_is_terminal(phase) and debrief_path is not None:
                debrief_folder_to_open = debrief_path.parent
                command_state.quit_requested = True
                break
            dashboard.tick(
                _dashboard_fps_for_speed(
                    current_speed_multiple,
                    recording=recording_controller.recorder is not None or clip_recording_controller.recording,
                    recording_fps=recording_fps,
                )
            )
    finally:
        recorder = recording_controller.recorder
        if recorder is not None and not recorder.saved:
            recording_controller.discard()
        if clip_recording_controller.recording:
            clip_recording_controller.discard()
        if audio_controller is not None:
            audio_controller.stop()
        else:
            _stop_game_music(getattr(dashboard, "pygame", None))
        dashboard.close()
        if debrief_folder_to_open is not None:
            opened = open_game_debrief_folder(debrief_folder_to_open)
            if opened:
                print(f"Opened game debrief folder: {debrief_folder_to_open}")
            else:
                print(f"Game debrief folder: {debrief_folder_to_open}")
        if training_cfg.enabled:
            print(trainer.debrief_text())
    final_arcade_score = arcade_total_score
    if arcade_enabled and bool(getattr(score, "level_passed", False)):
        final_arcade_score += _arcade_round_weighted_score(
            training_cfg,
            score,
            difficulty=difficulty,
            round_index=arcade_round_index,
            arcade_config=config,
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
) -> Any:
    return game_recording.start_game_recorder(
        enabled=enabled,
        config=config,
        difficulty=difficulty,
        attempt_index=attempt_index,
        output_dir=output_dir,
        fps=fps,
    )


def _start_game_clip_recorder(
    *,
    enabled: bool,
    config: SimulationConfig,
    difficulty: str,
    clip_index: int,
    output_dir: str | Path | None,
    fps: float,
) -> Any:
    return game_recording.start_game_clip_recorder(
        enabled=enabled,
        config=config,
        difficulty=difficulty,
        clip_index=clip_index,
        output_dir=output_dir,
        fps=fps,
    )


def _capture_recording_frame(recorder: Any, dashboard: Any) -> None:
    game_recording.capture_recording_frame(recorder, dashboard)


def _safe_capture_recording_frame(recorder: Any, dashboard: Any) -> Any:
    return game_recording.safe_capture_recording_frame(recorder, dashboard)


def _finish_game_recording(
    recorder: Any,
    training_cfg: RPOTrainingConfig,
    *,
    override_level_path: Path | None = None,
) -> Path | None:
    return game_recording.finish_game_recording(recorder, training_cfg, override_level_path=override_level_path)


def _discard_recorder_safely(recorder: Any) -> None:
    game_recording.discard_recorder_safely(recorder)


def _add_level_music_to_recording(
    recording_path: Path,
    training_cfg: RPOTrainingConfig,
    *,
    override_level_path: Path | None = None,
) -> Path:
    return game_recording.add_level_music_to_recording(
        recording_path,
        training_cfg,
        override_level_path=override_level_path,
    )


def _sync_dashboard_training_config(dashboard: Any, training_cfg: RPOTrainingConfig) -> None:
    dashboard.keepout_radius_km = training_cfg.keepout_radius_km
    dashboard.goal_range_km = training_cfg.goal_range_km
    dashboard.goal_range_tolerance_km = training_cfg.goal_range_tolerance_km
    dashboard.goal_radius_km = training_cfg.goal_radius_km
    dashboard.hard_speed_limit_radius_km = training_cfg.hard_speed_limit_radius_km
    dashboard.hard_speed_limit_km_s = training_cfg.hard_speed_limit_km_s
    dashboard.goal_relative_ric_km = training_cfg.goal_relative_ric_km
    dashboard.goal_nmt_radial_amplitude_km = training_cfg.goal_nmt_radial_amplitude_km
    dashboard.goal_nmt_cross_track_amplitude_km = training_cfg.goal_nmt_cross_track_amplitude_km
    dashboard.goal_nmt_cross_track_phase_deg = training_cfg.goal_nmt_cross_track_phase_deg
    dashboard.goal_nmt_center_ric_km = training_cfg.goal_nmt_center_ric_km
    dashboard.goal_nmt_element_tolerance_km = training_cfg.goal_nmt_element_tolerance_km
    if hasattr(dashboard, "_frame_cache_dirty"):
        dashboard._frame_cache_dirty = True


def _sync_dashboard_round_config(dashboard: Any, config: SimulationConfig) -> None:
    dashboard.coast_prediction_model = _game_coast_prediction_model(config)
    dashboard.camera_rule_mode = _game_camera_rule_mode(config)
    if hasattr(dashboard, "_prediction_cache"):
        dashboard._prediction_cache = {}
    if hasattr(dashboard, "_frame_cache_dirty"):
        dashboard._frame_cache_dirty = True


def _arcade_round_music_path(config: SimulationConfig, round_index: int) -> Path | None:
    track = _arcade_round_music_track(config, round_index)
    if track is None:
        return None
    path = Path(track)
    return path if path.is_absolute() else GAME_MUSIC_DIR / path


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
    config = _force_game_acceleration_off_config(config)
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


def _poll_pygame_input(
    pygame: Any,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
    briefing_open: bool = False,
) -> None:
    game_input.poll_pygame_input(pygame, state, control_mode=control_mode, briefing_open=briefing_open)


def _pygame_focus_lost(pygame: Any, event: Any) -> bool:
    return game_input.pygame_focus_lost(pygame, event)


def _opposing_key_axis(keys: Any, *, positive_key: Any, negative_key: Any) -> float:
    return game_input.opposing_key_axis(keys, positive_key=positive_key, negative_key=negative_key)


def _mission_state(score: Any) -> str:
    return mission_state_for_dashboard(phase_from_score(score))


def _game_loop_should_exit(*, session_done: bool, score: Any) -> bool:
    terminal_score = bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False))
    return bool(session_done) and not terminal_score


def _game_debrief_enabled(
    config: SimulationConfig,
    training_cfg: RPOTrainingConfig,
    *,
    arcade_enabled: bool | None = None,
) -> bool:
    if bool(arcade_enabled) or (arcade_enabled is None and _game_arcade_enabled(config)):
        return False
    if bool(getattr(training_cfg, "sandbox_mode", False)):
        return False
    return bool(getattr(training_cfg, "enabled", False))


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
        suffix = " Coast" if not config.fail_on_delta_v_budget and remain <= 0.0 else ""
        metrics.append(f"{tag} Chaser dV {format_speed_m_s(remain)}{suffix}")
    elif config.sandbox_mode:
        used = float(getattr(score, "approximate_delta_v_m_s", 0.0))
        metrics.append(f"INFO dV Used {format_speed_m_s(used)}")
    if config.max_target_delta_v_m_s is not None:
        remain = max(float(config.max_target_delta_v_m_s) - float(getattr(score, "target_delta_v_m_s", 0.0)), 0.0)
        ratio = remain / max(float(config.max_target_delta_v_m_s), 1.0e-9)
        metrics.append(f"{_status_tag(remain > 0.0, ratio > 0.2)} Target dV {format_speed_m_s(remain)}")
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
        metrics.append(f"{'OK' if changed else 'WARN'} Speed X")
    if config.required_coast_after_burn_s is not None:
        coasted = bool(getattr(score, "coast_after_burn_satisfied", False))
        metrics.append(f"{'OK' if coasted else 'WARN'} Coast {float(config.required_coast_after_burn_s):.0f}s")
    if config.guided_tutorial_burns:
        done = len(getattr(score, "guided_tutorial_burns_satisfied", ()))
        total = int(getattr(score, "guided_tutorial_burns_total", len(config.guided_tutorial_burns)))
        if config.guided_tutorial_speed_step is not None:
            done += 1 if bool(getattr(score, "guided_tutorial_speed_satisfied", False)) else 0
            total += 1
        metrics.append(f"{'OK' if done >= total else 'WARN'} Tutor {done}/{total}")
    if config.goal_nmt_element_tolerance_km is not None:
        tol = float(config.goal_nmt_element_tolerance_km)
        r_err = float(getattr(score, "final_nmt_radial_amplitude_error_km", float("nan")))
        c_err = float(getattr(score, "final_nmt_cross_track_amplitude_error_km", float("nan")))
        metrics.append(
            f"{_status_tag(r_err <= tol, r_err <= 0.75 * tol)} R Amp {_fmt_distance(r_err)}/{_fmt_distance(tol)}"
        )
        metrics.append(
            f"{_status_tag(c_err <= tol, c_err <= 0.75 * tol)} C Amp {_fmt_distance(c_err)}/{_fmt_distance(tol)}"
        )
    if config.goal_nmt_velocity_tolerance_km_s is not None:
        tol = float(config.goal_nmt_velocity_tolerance_km_s)
        err = float(getattr(score, "final_nmt_drift_velocity_error_km_s", float("nan")))
        metrics.append(f"{_status_tag(err <= tol, err <= 0.75 * tol)} Drift {_fmt_speed(err)}/{_fmt_speed(tol)}")
    if config.max_cross_track_amplitude_km is not None:
        tol = float(config.max_cross_track_amplitude_km)
        amp = float(getattr(score, "final_nmt_cross_track_amplitude_km", float("nan")))
        metrics.append(f"{_status_tag(amp <= tol, amp <= 0.75 * tol)} C Amp {_fmt_distance(amp)}/{_fmt_distance(tol)}")
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
        metrics.append(f"{_status_tag(clear, clear)} FR {'Clear' if clear else 'Violated'}")
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
    if config.sandbox_mode:
        return ("INFO Experiment Freely",)
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
    if config.required_coast_after_burn_s is not None:
        coasted = bool(getattr(score, "coast_after_burn_satisfied", False))
        checklist.append(f"{'OK' if coasted else 'WARN'} Coast after burn")
    if config.guided_tutorial_burns:
        satisfied = set(getattr(score, "guided_tutorial_burns_satisfied", ()))
        for stage in config.guided_tutorial_burns:
            checklist.append(f"{'OK' if stage.name in satisfied else 'WARN'} {stage.display_label}")
    if config.guided_tutorial_speed_step is not None:
        satisfied = bool(getattr(score, "guided_tutorial_speed_satisfied", False))
        checklist.append(f"{'OK' if satisfied else 'WARN'} {config.guided_tutorial_speed_step.label}")
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
    if config.max_cross_track_amplitude_km is not None:
        amp = float(getattr(score, "final_nmt_cross_track_amplitude_km", float("nan")))
        checklist.append(f"{'OK' if amp <= float(config.max_cross_track_amplitude_km) else 'WARN'} Damp C Amp")
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
        f"Approx dV     {format_speed_m_s(float(getattr(score, 'approximate_delta_v_m_s', 0.0)))}",
        f"Target dV     {format_speed_m_s(float(getattr(score, 'target_delta_v_m_s', 0.0)))}",
    ]
    lines = [line for line in lines if line]
    for reason in tuple(getattr(score, "pass_fail_reasons", ()) or ())[:3]:
        lines.append(f"Result        {reason}")
    return tuple(lines)


def _fmt_distance(value_km: float) -> str:
    return format_distance_km(value_km)


def _fmt_speed(value_km_s: float) -> str:
    return format_speed_km_s(value_km_s)


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
