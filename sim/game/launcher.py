from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.estimation.relative_th_ekf import ya_closed_form_transition_matrix
from sim.game.fonts import game_font
from sim.game.formatting import format_distance_km, format_speed_km_s, format_speed_m_s
from sim.game.frame_convention import (
    FRAME_CONVENTION_PRESET_OEL_DEFAULT,
    FRAME_CONVENTION_PRESET_SPACE_FORCE,
    FrameConvention,
    frame_convention_display_axis_sign,
    frame_convention_from_preset,
    frame_convention_preset,
    frame_convention_to_yaml,
    normalize_frame_convention,
)
from sim.game.operator import OperatorBurn, OperatorBurnPlan, parse_operator_burn_plan, validate_operator_burn_plan
from sim.game.pygame_dashboard import (
    CHASER_SPRITE_PATH,
    MIN_PLOT_SPAN_KM,
    PLOT_OVERLAY_MARGIN,
    TARGET_SPRITE_PATH,
    PygameRPODashboard,
    _coast_prediction_model_key,
    _cw_coast_states,
    _cylinder_projection_polygon_ric,
    _finite_projected_region_bounds,
    _game_asset_path_or_default,
    _new_history_ring,
    _plane_key_for_axes,
    _region_visible_on_plane,
    _satellite_pair_camera_center,
    _sphere_projection_polygon_ric,
)
from sim.game.training import (
    OPERATOR_RELAXED_REQUIRED_BURN_AXIS_SCENARIO_IDS,
    RPOTrainingConfig,
    nmt_curve_points_km,
    training_config_for_game_mode,
)

GAME_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
LAUNCHER_MUSIC_PATH = Path(__file__).resolve().parent / "music" / "01_insert_coin_to_orbit.wav"
START_SCREEN_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "OEL_RPO_Trainer.png"
GAME_PROGRESS_PATH_ENV = "OEL_GAME_PROGRESS_PATH"
GAME_SETTINGS_PATH_ENV = "OEL_GAME_SETTINGS_PATH"
DIFFICULTY_OPTIONS: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
GAME_MODE_OPTIONS: tuple[str, ...] = ("pilot", "operator")
DOWNLOADABLE_GAME_EXCLUDED_SCENARIO_IDS: frozenset[str] = frozenset(
    {
        "rpo_arcade_pursuit",
    }
)
OPERATOR_MODE_EXCLUDED_SCENARIO_IDS: frozenset[str] = frozenset(
    {
        "rpo_10_defensive_target_demo",
    }
)
OPTION_X = 54
OPTION_Y = 136
OPTION_WIDTH = 398
OPTION_HEIGHT = 64
OPTION_ROW_HEIGHT = 78
PANEL_TOP = 124
MIN_PANEL_HEIGHT = 480
FOOTER_HEIGHT = 76
FOOTER_BOTTOM_MARGIN = 22
PREVIEW_PADDING = 20
PREVIEW_LINE_HEIGHT = 18
PREVIEW_SECTION_TITLE_GAP = 22
PREVIEW_SECTION_GAP = 10
PREVIEW_SCROLL_STEP_PX = PREVIEW_LINE_HEIGHT * 3
CLEAR_PROGRESS_RECT = (846, 36, 150, 30)
MUSIC_RECT = (518, 36, 144, 30)
RECORD_VIDEO_RECT = (682, 36, 144, 30)
MODE_TOGGLE_WIDTH = 148
MODE_TOGGLE_HEIGHT = 34
MODE_TOGGLE_RIGHT_MARGIN = 42
SETTINGS_BUTTON_SIZE = 34
MODE_SETTINGS_GAP = 10
OPERATOR_BURN_MAX_ROWS = 24
OPERATOR_BURN_HEADERS: tuple[str, ...] = ("T (s)", "R (m/s)", "I (m/s)", "C (m/s)")
OPERATOR_BURN_ROW_HEIGHT = 36
OPERATOR_BURN_TABLE_MIN_VISIBLE_ROWS = 2
OPERATOR_BURN_MARKER_COLOR = (255, 154, 64)
FRAME_DIALOG_WIDTH = 620
FRAME_DIALOG_HEIGHT = 432
FRAME_DIALOG_CHOICE_HEIGHT = 76


@dataclass(frozen=True)
class GameScenarioOption:
    path: Path
    scenario_id: str
    title: str
    description: str
    learning_goal: str
    player_brief: str
    pass_criteria: tuple[str, ...]
    instructor_notes: tuple[str, ...]
    difficulty: str
    time_budget_s: float | None
    delta_v_budget_m_s: float | None
    goal_speed_km_s: float | None
    target_delta_v_budget_m_s: float | None
    completed_difficulties: tuple[str, ...]
    high_score: int
    level_number: int
    goal_range_km: float | None = None
    controlled_object_id: str = "chaser"
    target_object_id: str = "target"


@dataclass(frozen=True)
class GameProgressRecord:
    completed_difficulties: tuple[str, ...] = ()
    high_score: int = 0


@dataclass(frozen=True)
class GameSettings:
    frame_convention: FrameConvention = FrameConvention()
    ask_frame_convention_on_launch: bool = True
    last_game_mode: str | None = None
    operator_burn_scripts: dict[str, OperatorBurnPlan] = field(default_factory=dict)


@dataclass(frozen=True)
class GameLaunchSelection:
    path: Path
    difficulty: str
    music_enabled: bool = True
    record_video: bool = False
    mode: str = "pilot"
    frame_convention: FrameConvention = FrameConvention()
    operator_burn_plan: OperatorBurnPlan | None = None
    skip_initial_briefing: bool = False


@dataclass(frozen=True)
class OperatorPlotContext:
    initial_relative_ric_km_s: tuple[float, float, float, float, float, float] | None = None
    training_config: RPOTrainingConfig | None = None
    mean_motion_rad_s: float | None = None
    coast_prediction_model: str = "hcw"
    reference_state_eci_km_s: tuple[float, float, float, float, float, float] | None = None
    initial_coast_ric_km_s: tuple[tuple[float, float, float, float, float, float], ...] = ()
    pilot_initial_snapshot: Any | None = None
    pilot_dashboard_kwargs: dict[str, Any] = field(default_factory=dict)
    camera_mode: str = "reference"
    target_centered_plot_planes: tuple[str, ...] = ()
    target_centered_plot_axes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    plot_overlays_in_zoom: bool = True
    plot_overlays_in_zoom_by_plane: dict[str, bool] = field(default_factory=dict)
    plot_axis_scale: dict[str, tuple[float, float]] = field(default_factory=dict)
    plot_fixed_axis_half_span_km: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    plot_equal_axis_scale_planes: tuple[str, ...] = ()
    proximity_ring_plot_planes: tuple[str, ...] = ("RI", "RC", "IC")
    _planned_trajectory_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _planned_trajectory_time_cache: dict[tuple[Any, ...], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _preview_dashboard: Any | None = field(default=None, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class _RectSpec:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class OperatorDisplayState:
    previous_size: tuple[int, int]


@dataclass(frozen=True)
class OperatorTrajectoryProbe:
    state_ric_km_s: tuple[float, float, float, float, float, float]
    time_s: float
    plan_key: tuple[Any, ...]


def discover_game_scenarios(config_dir: Path | None = None) -> tuple[GameScenarioOption, ...]:
    return discover_game_scenarios_for_mode(config_dir, mode="pilot")


def discover_game_scenarios_for_mode(config_dir: Path | None = None, *, mode: str = "pilot") -> tuple[GameScenarioOption, ...]:
    root = Path(config_dir) if config_dir is not None else GAME_CONFIG_DIR
    mode_key = _normalize_game_mode(mode)
    progress = _load_game_progress()
    options: list[GameScenarioOption] = []
    for path in sorted(root.glob("game_training_rpo_*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        option = _scenario_option_from_yaml(path, raw, progress_by_scenario=progress, mode=mode_key)
        if option.scenario_id in DOWNLOADABLE_GAME_EXCLUDED_SCENARIO_IDS:
            continue
        if mode_key == "operator" and option.scenario_id in OPERATOR_MODE_EXCLUDED_SCENARIO_IDS:
            continue
        options.append(option)
    return tuple(sorted(options, key=_scenario_sort_key))


def choose_game_scenario(config_dir: Path | None = None) -> Path | None:
    selection = choose_game_launch(config_dir)
    return None if selection is None else selection.path


def choose_game_launch(
    config_dir: Path | None = None,
    *,
    show_start_screen: bool = True,
    initial_mode: str = "pilot",
) -> GameLaunchSelection | None:
    settings = _load_game_settings()
    mode_key = _normalize_game_mode(settings.last_game_mode or initial_mode)
    options = discover_game_scenarios_for_mode(config_dir, mode=mode_key)
    if not options:
        raise RuntimeError(f"No game training configs found in {Path(config_dir) if config_dir else GAME_CONFIG_DIR}.")
    return _run_launcher(options, show_start_screen=show_start_screen, initial_mode=mode_key)


def plan_operator_burns_for_config(
    pygame: Any,
    screen: Any,
    clock: Any,
    config_path: str | Path,
    *,
    font: Any,
    small_font: Any,
    title_font: Any,
    initial_plan: OperatorBurnPlan | None = None,
    difficulty: str = "easy",
    frame_convention: FrameConvention | dict[str, Any] | None = None,
    read_only: bool = False,
    demo_title: str = "",
    launch_label: str = "Launch",
) -> OperatorBurnPlan | None:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    option = _scenario_option_from_yaml(path, raw, mode="operator")
    previous_grab = bool(pygame.event.get_grab())
    previous_visible = bool(pygame.mouse.get_visible())
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    try:
        return _run_operator_plan_screen(
            pygame,
            screen,
            clock,
            option=option,
            font=font,
            small_font=small_font,
            title_font=title_font,
            initial_plan=initial_plan,
            difficulty=difficulty,
            frame_convention=frame_convention,
            read_only=read_only,
            demo_title=demo_title,
            launch_label=launch_label,
        )
    finally:
        pygame.event.set_grab(previous_grab)
        pygame.mouse.set_visible(previous_visible)


def _scenario_option_from_yaml(
    path: Path,
    raw: dict[str, Any],
    *,
    progress_by_scenario: dict[str, dict[str, GameProgressRecord]] | None = None,
    mode: str = "pilot",
) -> GameScenarioOption:
    metadata = dict(raw.get("metadata", {}) or {})
    game = dict(metadata.get("game", {}) or {})
    training = dict(game.get("training", {}) or {})
    scenario_id = str(training.get("scenario_id", raw.get("scenario_name", path.stem)) or path.stem)
    mode_key = _normalize_game_mode(mode)
    if progress_by_scenario is not None and scenario_id in progress_by_scenario:
        record = progress_by_scenario[scenario_id].get(mode_key, GameProgressRecord())
        completed_difficulties = tuple(record.completed_difficulties)
        high_score = int(record.high_score)
    else:
        completed_difficulties = _completed_difficulties_from_game(game)
        high_score = _high_score_from_game(game)
    level_number = _level_number_from_scenario_id(scenario_id)
    level_name = str(game.get("level_name", "") or "").strip()
    if scenario_id == "rpo_00_tutorial":
        level_name = "Level 0 - Operator Tutorial" if mode_key == "operator" else "Level 0 - Pilot Tutorial"
    target_delta_v_budget = _optional_float(training.get("max_target_delta_v_m_s"))
    if target_delta_v_budget is None:
        target_delta_v_budget = _optional_float(dict(game.get("defensive_target", {}) or {}).get("max_delta_v_m_s"))
    pass_criteria = _as_str_tuple(training.get("pass_criteria"))
    player_brief = str(training.get("player_brief", "") or "")
    if mode_key == "operator" and scenario_id in OPERATOR_RELAXED_REQUIRED_BURN_AXIS_SCENARIO_IDS:
        pass_criteria = _operator_relaxed_burn_axis_pass_criteria(pass_criteria)
        player_brief = _operator_relaxed_burn_axis_player_brief(player_brief)
    return GameScenarioOption(
        path=path,
        scenario_id=scenario_id,
        title=level_name or _title_from_scenario_id(scenario_id, level_number=level_number),
        description=str(raw.get("scenario_description", "") or ""),
        learning_goal=str(training.get("learning_goal", "") or ""),
        player_brief=player_brief,
        pass_criteria=pass_criteria,
        instructor_notes=_as_str_tuple(training.get("instructor_notes")),
        difficulty=str(game.get("difficulty", "") or ""),
        time_budget_s=_optional_float(training.get("max_time_s")),
        delta_v_budget_m_s=_optional_float(training.get("max_delta_v_m_s")),
        goal_speed_km_s=_optional_float(training.get("max_goal_speed_km_s")),
        target_delta_v_budget_m_s=target_delta_v_budget,
        completed_difficulties=completed_difficulties,
        high_score=high_score,
        level_number=level_number,
        goal_range_km=_optional_float(training.get("goal_range_km")),
        controlled_object_id=str(game.get("controlled_object_id", "chaser") or "chaser"),
        target_object_id=str(training.get("target_object_id", "target") or "target"),
    )


def _operator_relaxed_burn_axis_pass_criteria(pass_criteria: tuple[str, ...]) -> tuple[str, ...]:
    relaxed_prefixes = (
        "perform at least one radial burn",
        "perform at least one in-track burn",
    )
    return tuple(
        item
        for item in pass_criteria
        if not any(item.strip().lower().startswith(prefix) for prefix in relaxed_prefixes)
    )


def _operator_relaxed_burn_axis_player_brief(player_brief: str) -> str:
    return player_brief.replace("First test radial and in-track burns, then ", "")


def record_game_progress(
    config_path: str | Path,
    difficulty: str,
    score: int | None = None,
    *,
    completed: bool = True,
    mode: str = "pilot",
) -> None:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenario_id = _scenario_id_from_yaml(path, raw)
    progress = _load_game_progress()
    mode_key = _normalize_game_mode(mode)
    by_mode = dict(progress.get(scenario_id, {}))
    current = by_mode.get(mode_key, GameProgressRecord())
    completed_difficulties = list(current.completed_difficulties)
    normalized = _normalize_difficulty(difficulty)
    if bool(completed) and normalized not in completed_difficulties:
        completed_difficulties.append(normalized)
    high_score = int(current.high_score)
    if score is not None:
        high_score = max(high_score, int(max(score, 0)))
    by_mode[mode_key] = GameProgressRecord(
        completed_difficulties=tuple(item for item in DIFFICULTY_OPTIONS if item in completed_difficulties),
        high_score=high_score,
    )
    progress[scenario_id] = by_mode
    _save_game_progress(progress)


def clear_game_progress(config_dir: Path | None = None) -> None:
    root = Path(config_dir) if config_dir is not None else GAME_CONFIG_DIR
    progress = _load_game_progress()
    changed = False
    for path in sorted(root.glob("game_training_rpo_*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        scenario_id = _scenario_id_from_yaml(path, raw)
        empty = {mode: GameProgressRecord() for mode in GAME_MODE_OPTIONS}
        if scenario_id not in progress or progress.get(scenario_id) != empty:
            progress[scenario_id] = empty
            changed = True
    if changed:
        _save_game_progress(progress)


def _scenario_id_from_yaml(path: Path, raw: dict[str, Any]) -> str:
    metadata = dict(raw.get("metadata", {}) or {})
    game = dict(metadata.get("game", {}) or {})
    training = dict(game.get("training", {}) or {})
    return str(training.get("scenario_id", raw.get("scenario_name", path.stem)) or path.stem)


def _game_progress_path() -> Path:
    override = os.environ.get(GAME_PROGRESS_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".orbital_engagement_lab" / "game_progress.yaml"


def _game_settings_path() -> Path:
    override = os.environ.get(GAME_SETTINGS_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".orbital_engagement_lab" / "game_settings.yaml"


def _load_game_settings() -> GameSettings:
    path = _game_settings_path()
    if not path.exists():
        return GameSettings()
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except OSError:
        return GameSettings()
    if not isinstance(raw, dict):
        return GameSettings()
    return GameSettings(
        frame_convention=normalize_frame_convention(raw.get("frame_convention", {})),
        ask_frame_convention_on_launch=bool(raw.get("ask_frame_convention_on_launch", True)),
        last_game_mode=_game_mode_or_none(raw.get("last_game_mode")),
        operator_burn_scripts=_operator_burn_scripts_from_yaml(raw.get("operator_burn_scripts", {})),
    )


def _save_game_settings(settings: GameSettings) -> None:
    path = _game_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_convention": frame_convention_to_yaml(settings.frame_convention),
        "ask_frame_convention_on_launch": bool(settings.ask_frame_convention_on_launch),
    }
    if settings.last_game_mode is not None:
        payload["last_game_mode"] = _normalize_game_mode(settings.last_game_mode)
    if settings.operator_burn_scripts:
        payload["operator_burn_scripts"] = {
            str(scenario_id): _operator_burn_plan_to_yaml(plan)
            for scenario_id, plan in sorted(settings.operator_burn_scripts.items())
        }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _game_mode_or_none(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in GAME_MODE_OPTIONS:
        return key
    return None


def _save_last_game_mode(settings: GameSettings, mode: Any) -> GameSettings:
    updated = replace(settings, last_game_mode=_normalize_game_mode(mode))
    _save_game_settings(updated)
    return updated


def _frame_convention_dialog_settings(
    settings: GameSettings,
    *,
    frame_convention: FrameConvention,
    dont_ask_again: bool,
    selected_mode: str,
) -> GameSettings:
    return replace(
        settings,
        frame_convention=frame_convention,
        ask_frame_convention_on_launch=not bool(dont_ask_again),
        last_game_mode=_normalize_game_mode(selected_mode),
    )


def _load_saved_operator_burn_plan(scenario_id: Any) -> OperatorBurnPlan | None:
    key = str(scenario_id or "").strip()
    if not key:
        return None
    return _load_game_settings().operator_burn_scripts.get(key)


def _save_operator_burn_plan(scenario_id: Any, plan: OperatorBurnPlan) -> None:
    key = str(scenario_id or "").strip()
    if not key:
        return
    settings = _load_game_settings()
    scripts = dict(settings.operator_burn_scripts)
    scripts[key] = plan
    _save_game_settings(replace(settings, operator_burn_scripts=scripts))


def _operator_burn_plan_to_yaml(plan: OperatorBurnPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for burn in plan.burns:
        rows.append(
            {
                "time_s": float(burn.time_s),
                "delta_v_ric_m_s": [float(value) for value in burn.delta_v_ric_m_s],
            }
        )
    return rows


def _operator_burn_scripts_from_yaml(raw: Any) -> dict[str, OperatorBurnPlan]:
    if not isinstance(raw, dict):
        return {}
    scripts: dict[str, OperatorBurnPlan] = {}
    for scenario_id, value in raw.items():
        key = str(scenario_id or "").strip()
        if not key:
            continue
        plan = _operator_burn_plan_from_yaml(value)
        if plan is not None:
            scripts[key] = plan
    return scripts


def _operator_burn_plan_from_yaml(raw: Any) -> OperatorBurnPlan | None:
    if raw is None:
        return OperatorBurnPlan()
    try:
        items = list(raw)
    except TypeError:
        return None
    burns: list[OperatorBurn] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        try:
            time_s = float(item.get("time_s", 0.0))
            delta_v_values = list(item.get("delta_v_ric_m_s", (0.0, 0.0, 0.0)))
            if len(delta_v_values) != 3:
                return None
            delta_v_ric_m_s = tuple(float(value) for value in delta_v_values)
        except (TypeError, ValueError):
            return None
        burns.append(OperatorBurn(time_s=time_s, delta_v_ric_m_s=delta_v_ric_m_s))
    return OperatorBurnPlan(burns=tuple(sorted(burns, key=lambda burn: burn.time_s)))


def _load_game_progress() -> dict[str, dict[str, GameProgressRecord]]:
    path = _game_progress_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenarios = dict(raw.get("scenarios", {}) or {}) if isinstance(raw, dict) else {}
    progress: dict[str, dict[str, GameProgressRecord]] = {}
    for scenario_id, item in scenarios.items():
        if not isinstance(item, dict):
            continue
        progress[str(scenario_id)] = _progress_modes_from_yaml_item(item)
    return progress


def _save_game_progress(progress: dict[str, dict[str, GameProgressRecord]]) -> None:
    path = _game_progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = {}
    for scenario_id, by_mode in sorted(progress.items()):
        scenarios[scenario_id] = {}
        for mode in GAME_MODE_OPTIONS:
            record = dict(by_mode).get(mode, GameProgressRecord())
            completed_set = set(record.completed_difficulties)
            scenarios[scenario_id][mode] = {
                "completed_difficulties": [item for item in DIFFICULTY_OPTIONS if item in completed_set],
                "high_score": int(max(record.high_score, 0)),
            }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"scenarios": scenarios}, f, sort_keys=False)


def _progress_modes_from_yaml_item(item: dict[str, Any]) -> dict[str, GameProgressRecord]:
    if any(mode in item for mode in GAME_MODE_OPTIONS):
        return {
            mode: _progress_record_from_yaml_item(dict(item.get(mode, {}) or {}))
            for mode in GAME_MODE_OPTIONS
        }
    return {
        "pilot": _progress_record_from_yaml_item(item),
        "operator": GameProgressRecord(),
    }


def _progress_record_from_yaml_item(item: dict[str, Any]) -> GameProgressRecord:
    completed = item.get("completed_difficulties", ())
    if isinstance(completed, str):
        completed = (completed,)
    completed_set = {_normalize_difficulty(value) for value in completed}
    return GameProgressRecord(
        completed_difficulties=tuple(value for value in DIFFICULTY_OPTIONS if value in completed_set),
        high_score=max(int(item.get("high_score", 0) or 0), 0),
    )


def _level_number_from_scenario_id(scenario_id: str) -> int:
    parts = str(scenario_id).split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if digits:
            return int(digits)
    return 999


def _scenario_sort_key(option: GameScenarioOption) -> tuple[int, str]:
    if option.scenario_id == "rpo_11_evasive_target_survival":
        return (10, option.scenario_id)
    if option.scenario_id == "rpo_10_defensive_target_demo":
        return (11, option.scenario_id)
    if option.scenario_id == "rpo_bonus_cislunar_rendezvous":
        return (12, option.scenario_id)
    if option.scenario_id == "rpo_arcade_pursuit":
        return (13, option.scenario_id)
    if option.scenario_id == "rpo_sandbox":
        return (14, option.scenario_id)
    return (option.level_number, option.scenario_id)


def _title_from_scenario_id(scenario_id: str, *, level_number: int) -> str:
    parts = str(scenario_id).split("_")
    if len(parts) >= 3 and parts[0] == "rpo" and parts[1].isdigit():
        name = " ".join(parts[2:]).title()
        return f"Level {level_number} - {name}"
    return str(scenario_id).replace("_", " ").title()


def _run_launcher(
    options: tuple[GameScenarioOption, ...],
    *,
    show_start_screen: bool = True,
    initial_mode: str = "pilot",
) -> GameLaunchSelection | None:
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError("Game launcher requires `pygame`. Install with `pip install .[game]`.") from exc

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1040, 680), pygame.RESIZABLE)
    pygame.display.set_caption("Orbital Engagement Lab - Level Select")
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    clock = pygame.time.Clock()
    font = game_font(pygame, 19)
    small_font = game_font(pygame, 15)
    title_font = game_font(pygame, 32)
    hero_font = game_font(pygame, 46)
    selected = 0
    scroll_offset = 0
    preview_scroll_px = 0
    difficulty_idx = _difficulty_index(options[selected].difficulty)
    selected_mode = _normalize_game_mode(initial_mode)
    music_enabled = _start_launcher_music(pygame)
    start_artwork = _load_start_screen_artwork(pygame)
    record_video = False
    settings = _load_game_settings()
    frame_convention = settings.frame_convention
    frame_dialog_open = bool(settings.ask_frame_convention_on_launch)
    frame_dialog_required = frame_dialog_open
    frame_dialog_dont_ask_again = False
    start_screen_open = bool(show_start_screen)

    try:
        while True:
            width, height = screen.get_size()
            if start_screen_open:
                for event in pygame.event.get():
                    action = _start_screen_event_action(pygame, event)
                    if action == "quit":
                        return None
                    if action == "begin":
                        start_screen_open = False
                        break
                if start_screen_open:
                    _draw_start_screen(
                        pygame,
                        screen,
                        artwork=start_artwork,
                        hero_font=hero_font,
                        font=font,
                        small_font=small_font,
                    )
                    pygame.display.flip()
                    clock.tick(60)
                    continue

            if frame_dialog_open:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            if frame_dialog_required:
                                return None
                            frame_dialog_open = False
                        if event.key in {pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}:
                            settings = _frame_convention_dialog_settings(
                                settings,
                                frame_convention=frame_convention,
                                dont_ask_again=frame_dialog_dont_ask_again,
                                selected_mode=selected_mode,
                            )
                            _save_game_settings(settings)
                            frame_dialog_open = False
                            frame_dialog_required = False
                        elif event.key in {pygame.K_1, pygame.K_o}:
                            frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_OEL_DEFAULT)
                        elif event.key in {pygame.K_2, pygame.K_s}:
                            frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)
                        elif event.key == pygame.K_d:
                            frame_dialog_dont_ask_again = not frame_dialog_dont_ask_again
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        action = _frame_convention_dialog_action(
                            event.pos,
                            width=width,
                            height=height,
                        )
                        if action == FRAME_CONVENTION_PRESET_OEL_DEFAULT:
                            frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_OEL_DEFAULT)
                        elif action == FRAME_CONVENTION_PRESET_SPACE_FORCE:
                            frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)
                        elif action == "dont_ask_again":
                            frame_dialog_dont_ask_again = not frame_dialog_dont_ask_again
                        elif action == "continue":
                            settings = _frame_convention_dialog_settings(
                                settings,
                                frame_convention=frame_convention,
                                dont_ask_again=frame_dialog_dont_ask_again,
                                selected_mode=selected_mode,
                            )
                            _save_game_settings(settings)
                            frame_dialog_open = False
                            frame_dialog_required = False
                if frame_dialog_open:
                    _draw_launcher(
                        pygame,
                        screen,
                        options=options,
                        selected=selected,
                        scroll_offset=scroll_offset,
                        selected_difficulty=DIFFICULTY_OPTIONS[difficulty_idx],
                        music_enabled=music_enabled,
                        preview_scroll_px=preview_scroll_px,
                        record_video=record_video,
                        selected_mode=selected_mode,
                        font=font,
                        small_font=small_font,
                        title_font=title_font,
                    )
                    _draw_frame_convention_dialog(
                        pygame,
                        screen,
                        convention=frame_convention,
                        dont_ask_again=frame_dialog_dont_ask_again,
                        font=font,
                        small_font=small_font,
                        title_font=title_font,
                    )
                    pygame.display.flip()
                    clock.tick(60)
                    continue

            preview_bounds = _preview_bounds(width, height)
            for event in pygame.event.get():
                selected_difficulty = DIFFICULTY_OPTIONS[difficulty_idx]
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key in {pygame.K_DOWN, pygame.K_s}:
                        new_selected = _advance_launcher_selection(selected, 1, count=len(options))
                        if new_selected != selected:
                            preview_scroll_px = 0
                        selected = new_selected
                    elif event.key in {pygame.K_UP, pygame.K_w}:
                        new_selected = _advance_launcher_selection(selected, -1, count=len(options))
                        if new_selected != selected:
                            preview_scroll_px = 0
                        selected = new_selected
                    elif event.key in {pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}:
                        selection = _selection_for_launch(
                            pygame,
                            screen,
                            clock,
                            option=options[selected],
                            difficulty=selected_difficulty,
                            music_enabled=music_enabled,
                            record_video=record_video,
                            mode=selected_mode,
                            frame_convention=frame_convention,
                            font=font,
                            small_font=small_font,
                            title_font=title_font,
                        )
                        if selection is not None:
                            settings = _save_last_game_mode(settings, selected_mode)
                            return selection
                        screen = pygame.display.get_surface() or pygame.display.set_mode((1040, 680), pygame.RESIZABLE)
                    elif event.key == pygame.K_v:
                        record_video = not record_video
                    elif event.key == pygame.K_m:
                        music_enabled = _toggle_launcher_music(pygame, music_enabled=music_enabled)
                    elif event.key == pygame.K_o:
                        selected_mode = _toggle_game_mode(selected_mode)
                        settings = _save_last_game_mode(settings, selected_mode)
                        options = discover_game_scenarios_for_mode(options[0].path.parent, mode=selected_mode)
                        selected = min(selected, len(options) - 1)
                        preview_scroll_px = 0
                        difficulty_idx = _difficulty_index(options[selected].difficulty)
                    elif event.key in {pygame.K_LEFT, pygame.K_a}:
                        difficulty_idx = _advance_launcher_selection(difficulty_idx, -1, count=len(DIFFICULTY_OPTIONS))
                    elif event.key in {pygame.K_RIGHT, pygame.K_d}:
                        difficulty_idx = _advance_launcher_selection(difficulty_idx, 1, count=len(DIFFICULTY_OPTIONS))
                    elif event.key == pygame.K_1:
                        difficulty_idx = 0
                    elif event.key == pygame.K_2:
                        difficulty_idx = 1
                    elif event.key == pygame.K_3:
                        difficulty_idx = 2
                    elif event.key == pygame.K_4:
                        difficulty_idx = 3
                    scroll_offset = _scroll_for_selection(
                        selected, scroll_offset, count=len(options), screen_height=height
                    )
                if event.type == pygame.MOUSEWHEEL:
                    if _pos_in_bounds(pygame.mouse.get_pos(), preview_bounds):
                        preview_scroll_px = _clamp_preview_scroll_px(
                            preview_scroll_px - int(event.y) * PREVIEW_SCROLL_STEP_PX,
                            option=options[selected],
                            font=font,
                            small_font=small_font,
                            preview_bounds=preview_bounds,
                        )
                    else:
                        scroll_offset = int(
                            max(0, min(scroll_offset - int(event.y), _max_scroll_offset(len(options), height)))
                        )
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    idx = _option_index_at_pos(pygame.mouse.get_pos(), count=len(options), scroll_offset=scroll_offset)
                    if idx is not None:
                        if idx == selected:
                            selection = _selection_for_launch(
                                pygame,
                                screen,
                                clock,
                                option=options[selected],
                                difficulty=selected_difficulty,
                                music_enabled=music_enabled,
                                record_video=record_video,
                                mode=selected_mode,
                                frame_convention=frame_convention,
                                font=font,
                                small_font=small_font,
                                title_font=title_font,
                            )
                            if selection is not None:
                                settings = _save_last_game_mode(settings, selected_mode)
                                return selection
                            screen = pygame.display.get_surface() or pygame.display.set_mode((1040, 680), pygame.RESIZABLE)
                        preview_scroll_px = 0
                        selected = idx
                    else:
                        mouse_pos = pygame.mouse.get_pos()
                        if _mode_toggle_at_pos(mouse_pos, width=width, height=height):
                            selected_mode = _toggle_game_mode(selected_mode)
                            settings = _save_last_game_mode(settings, selected_mode)
                            options = discover_game_scenarios_for_mode(options[0].path.parent, mode=selected_mode)
                            selected = min(selected, len(options) - 1)
                            preview_scroll_px = 0
                            difficulty_idx = _difficulty_index(options[selected].difficulty)
                            continue
                        if _settings_button_at_pos(mouse_pos, width=width, height=height):
                            frame_dialog_open = True
                            frame_dialog_required = False
                            frame_dialog_dont_ask_again = not bool(settings.ask_frame_convention_on_launch)
                            continue
                        if _record_video_at_pos(mouse_pos):
                            record_video = not record_video
                            continue
                        if _music_at_pos(mouse_pos):
                            music_enabled = _toggle_launcher_music(pygame, music_enabled=music_enabled)
                            continue
                        if _clear_progress_at_pos(mouse_pos):
                            clear_game_progress(options[0].path.parent)
                            options = discover_game_scenarios_for_mode(options[0].path.parent, mode=selected_mode)
                            selected = min(selected, len(options) - 1)
                            preview_scroll_px = 0
                            difficulty_idx = _difficulty_index(options[selected].difficulty)
                            scroll_offset = _scroll_for_selection(
                                selected, scroll_offset, count=len(options), screen_height=height
                            )
                            continue
                        difficulty = _difficulty_at_pos(mouse_pos)
                        if difficulty is not None:
                            difficulty_idx = _difficulty_index(difficulty)
                if event.type == pygame.MOUSEMOTION:
                    idx = _option_index_at_pos(event.pos, count=len(options), scroll_offset=scroll_offset)
                    if idx is not None and idx != selected:
                        preview_scroll_px = 0
                        selected = idx
            scroll_offset = _scroll_for_selection(selected, scroll_offset, count=len(options), screen_height=height)
            selected_difficulty = DIFFICULTY_OPTIONS[difficulty_idx]
            preview_scroll_px = _clamp_preview_scroll_px(
                preview_scroll_px,
                option=options[selected],
                font=font,
                small_font=small_font,
                preview_bounds=preview_bounds,
            )

            _draw_launcher(
                pygame,
                screen,
                options=options,
                selected=selected,
                scroll_offset=scroll_offset,
                selected_difficulty=selected_difficulty,
                music_enabled=music_enabled,
                preview_scroll_px=preview_scroll_px,
                record_video=record_video,
                selected_mode=selected_mode,
                font=font,
                small_font=small_font,
                title_font=title_font,
            )
            pygame.display.flip()
            clock.tick(60)
    finally:
        _stop_launcher_music(pygame)
        pygame.display.quit()
        pygame.quit()


def _start_screen_event_action(pygame: Any, event: Any) -> str:
    if event.type == pygame.QUIT:
        return "quit"
    if event.type == pygame.KEYDOWN:
        if getattr(event, "key", None) == pygame.K_ESCAPE:
            return "quit"
        return "begin"
    return "ignore"


def _option_index_at_pos(pos: tuple[int, int], *, count: int, scroll_offset: int = 0) -> int | None:
    x, y = pos
    if x < OPTION_X or x > OPTION_X + OPTION_WIDTH:
        return None
    if y < OPTION_Y:
        return None
    row = int((y - OPTION_Y) // OPTION_ROW_HEIGHT)
    idx = int(scroll_offset) + row
    row_y = OPTION_Y + row * OPTION_ROW_HEIGHT
    if 0 <= idx < count and row_y <= y <= row_y + OPTION_HEIGHT:
        return idx
    return None


def _visible_option_count(screen_height: int) -> int:
    list_bottom = PANEL_TOP + _launcher_panel_height(screen_height)
    return max(1, int((list_bottom - OPTION_Y) // OPTION_ROW_HEIGHT))


def _launcher_panel_height(screen_height: int) -> int:
    return max(int(screen_height) - PANEL_TOP - FOOTER_HEIGHT, MIN_PANEL_HEIGHT)


def _max_scroll_offset(count: int, screen_height: int) -> int:
    return max(int(count) - _visible_option_count(screen_height), 0)


def _advance_launcher_selection(selected: int, step: int, *, count: int) -> int:
    if int(count) <= 0:
        return 0
    return int((int(selected) + int(step)) % int(count))


def _scroll_for_selection(selected: int, scroll_offset: int, *, count: int, screen_height: int) -> int:
    max_scroll = _max_scroll_offset(count, screen_height)
    visible = _visible_option_count(screen_height)
    scroll = int(max(0, min(scroll_offset, max_scroll)))
    if selected < scroll:
        return int(max(0, selected))
    if selected >= scroll + visible:
        return int(min(max_scroll, selected - visible + 1))
    return scroll


def _preview_bounds(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    panel_height = _launcher_panel_height(screen_height)
    return (490, PANEL_TOP, max(int(screen_width) - 532, 420), panel_height)


def _pos_in_bounds(pos: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    px, py = pos
    x, y, w, h = bounds
    return x <= px <= x + w and y <= py <= y + h


def _frame_convention_dialog_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    width = min(FRAME_DIALOG_WIDTH, max(int(screen_width) - 80, 360))
    height = min(FRAME_DIALOG_HEIGHT, max(int(screen_height) - 80, 360))
    return (
        max((int(screen_width) - width) // 2, 0),
        max((int(screen_height) - height) // 2, 0),
        width,
        height,
    )


def _frame_convention_dialog_choice_rects(screen_width: int, screen_height: int) -> dict[str, tuple[int, int, int, int]]:
    x, y, w, _h = _frame_convention_dialog_rect(screen_width, screen_height)
    button_w = max(w - 84, 260)
    button_h = FRAME_DIALOG_CHOICE_HEIGHT
    return {
        FRAME_CONVENTION_PRESET_OEL_DEFAULT: (x + 42, y + 122, button_w, button_h),
        FRAME_CONVENTION_PRESET_SPACE_FORCE: (x + 42, y + 212, button_w, button_h),
    }


def _frame_convention_dialog_checkbox_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    x, y, _w, h = _frame_convention_dialog_rect(screen_width, screen_height)
    return (x + 42, y + h - 74, 22, 22)


def _frame_convention_dialog_continue_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    x, y, w, h = _frame_convention_dialog_rect(screen_width, screen_height)
    return (x + w - 178, y + h - 80, 136, 34)


def _frame_convention_dialog_action(
    pos: tuple[int, int],
    *,
    width: int,
    height: int,
) -> str | None:
    for action, rect in _frame_convention_dialog_choice_rects(width, height).items():
        if _pos_in_bounds(pos, rect):
            return action
    checkbox = _frame_convention_dialog_checkbox_rect(width, height)
    if _pos_in_bounds(pos, (checkbox[0], checkbox[1], 190, checkbox[3])):
        return "dont_ask_again"
    if _pos_in_bounds(pos, _frame_convention_dialog_continue_rect(width, height)):
        return "continue"
    return None


def _clamp_preview_scroll_px(
    scroll_px: int,
    *,
    option: GameScenarioOption,
    font: Any,
    small_font: Any,
    preview_bounds: tuple[int, int, int, int],
) -> int:
    _, _, width, height = preview_bounds
    content_width = max(int(width) - PREVIEW_PADDING * 2, 1)
    viewport_height = max(int(height) - PREVIEW_PADDING * 2, 1)
    content_height = _preview_content_height(option, font=font, small_font=small_font, width_px=content_width)
    max_scroll = max(int(content_height) - viewport_height, 0)
    return int(max(0, min(int(scroll_px), max_scroll)))


def _difficulty_index(value: str) -> int:
    key = str(value or "easy").strip().lower()
    if key == "normal":
        key = "medium"
    if key == "expert":
        key = "extreme"
    if key in DIFFICULTY_OPTIONS:
        return DIFFICULTY_OPTIONS.index(key)
    return 0


def _difficulty_at_pos(pos: tuple[int, int]) -> str | None:
    x, y = pos
    if y < 86 or y > 112:
        return None
    for idx, difficulty in enumerate(DIFFICULTY_OPTIONS):
        left = 642 + idx * 86
        right = left + 76
        if left <= x <= right:
            return difficulty
    return None


def _clear_progress_at_pos(pos: tuple[int, int]) -> bool:
    x, y, w, h = CLEAR_PROGRESS_RECT
    px, py = pos
    return x <= px <= x + w and y <= py <= y + h


def _record_video_at_pos(pos: tuple[int, int]) -> bool:
    x, y, w, h = RECORD_VIDEO_RECT
    px, py = pos
    return x <= px <= x + w and y <= py <= y + h


def _music_at_pos(pos: tuple[int, int]) -> bool:
    x, y, w, h = MUSIC_RECT
    px, py = pos
    return x <= px <= x + w and y <= py <= y + h


def _mode_toggle_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    settings_x, settings_y, _settings_w, _settings_h = _settings_button_rect(screen_width, screen_height)
    x = max(settings_x - MODE_SETTINGS_GAP - MODE_TOGGLE_WIDTH, 0)
    y = settings_y
    return (x, y, MODE_TOGGLE_WIDTH, MODE_TOGGLE_HEIGHT)


def _mode_toggle_at_pos(pos: tuple[int, int], *, width: int, height: int) -> bool:
    return _pos_in_bounds(pos, _mode_toggle_rect(width, height))


def _settings_button_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    x = max(int(screen_width) - MODE_TOGGLE_RIGHT_MARGIN - SETTINGS_BUTTON_SIZE, 0)
    y = max(int(screen_height) - FOOTER_BOTTOM_MARGIN - SETTINGS_BUTTON_SIZE, PANEL_TOP)
    return (x, y, SETTINGS_BUTTON_SIZE, SETTINGS_BUTTON_SIZE)


def _settings_button_at_pos(pos: tuple[int, int], *, width: int, height: int) -> bool:
    return _pos_in_bounds(pos, _settings_button_rect(width, height))


def _enter_operator_fullscreen(pygame: Any, screen: Any) -> OperatorDisplayState:
    previous_size = tuple(int(value) for value in screen.get_size())
    flags = pygame.FULLSCREEN | pygame.SCALED
    _reset_pygame_display(pygame)
    pygame.display.set_mode((1280, 720), flags)
    pygame.display.set_caption("Orbital Engagement Lab - Operator Mode")
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    return OperatorDisplayState(previous_size=(previous_size[0], previous_size[1]))


def _restore_operator_display(pygame: Any, state: OperatorDisplayState) -> None:
    width = max(int(state.previous_size[0]), 640)
    height = max(int(state.previous_size[1]), 480)
    _reset_pygame_display(pygame)
    pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Orbital Engagement Lab - Level Select")
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)


def _reset_pygame_display(pygame: Any) -> None:
    pygame.display.quit()
    pygame.display.init()
    pygame.font.init()


def _selection_for_launch(
    pygame: Any,
    screen: Any,
    clock: Any,
    *,
    option: GameScenarioOption,
    difficulty: str,
    music_enabled: bool,
    record_video: bool,
    mode: str,
    frame_convention: FrameConvention,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> GameLaunchSelection | None:
    mode_key = _normalize_game_mode(mode)
    operator_tutorial = mode_key == "operator" and option.scenario_id == "rpo_00_tutorial"
    return GameLaunchSelection(
        path=option.path,
        difficulty=difficulty,
        music_enabled=music_enabled,
        record_video=record_video,
        mode=mode_key,
        frame_convention=frame_convention,
        operator_burn_plan=None,
        skip_initial_briefing=mode_key == "operator" and not operator_tutorial,
    )


def _run_operator_plan_screen(
    pygame: Any,
    screen: Any,
    clock: Any,
    *,
    option: GameScenarioOption,
    font: Any,
    small_font: Any,
    title_font: Any,
    initial_plan: OperatorBurnPlan | None = None,
    difficulty: str = "easy",
    frame_convention: FrameConvention | dict[str, Any] | None = None,
    read_only: bool = False,
    demo_title: str = "",
    launch_label: str = "Launch",
) -> OperatorBurnPlan | None:
    frame_convention = normalize_frame_convention(frame_convention)
    seed_plan = initial_plan if initial_plan is not None else _load_saved_operator_burn_plan(option.scenario_id)
    rows: list[list[str]] = _operator_rows_from_plan(seed_plan)
    active_cell: tuple[int, int] = (0, 0)
    table_scroll_row = 0
    objectives_visible = True
    objectives_scroll_px = 0
    equation_sheet_visible = False
    equation_sheet_scroll_px = 0
    plot_context = _operator_plot_context(option.path, difficulty=difficulty)
    trajectory_probe: OperatorTrajectoryProbe | None = None
    while True:
        width, height = screen.get_size()
        button_gap = 10
        launch_rect = pygame.Rect(width - 52 - 126, height - 70, 126, 36)
        cancel_rect = pygame.Rect(launch_rect.x - button_gap - 120, height - 70, 120, 36)
        equation_sheet_rect = pygame.Rect(*_operator_equation_sheet_button_rect(width, height))
        objectives_rect = pygame.Rect(*_operator_objectives_button_rect(width, height))
        objectives_overlay_rect = pygame.Rect(*_operator_objectives_overlay_rect(width, height))
        objectives_content_rect = _operator_objectives_content_rect(objectives_overlay_rect)
        objectives_content_height = _operator_objectives_content_height(
            option,
            plot_context.training_config,
            font=small_font,
            width_px=objectives_content_rect[2],
        )
        objectives_scroll_px = _clamp_operator_objectives_scroll_px(
            objectives_scroll_px,
            content_height=objectives_content_height,
            viewport_height=objectives_content_rect[3],
        )
        equation_sheet_content_rect = _operator_objectives_content_rect(objectives_overlay_rect)
        equation_sheet_content_height = _operator_equation_sheet_content_height(
            small_font,
            width_px=equation_sheet_content_rect[2],
        )
        equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
            equation_sheet_scroll_px,
            content_height=equation_sheet_content_height,
            viewport_height=equation_sheet_content_rect[3],
        )
        table_rect = _operator_burn_table_rect(width, height)
        add_burn_rect = _operator_add_burn_button_rect(pygame, table_rect)
        table_scroll_row = int(max(0, min(table_scroll_row, _operator_max_table_scroll_row(len(rows), table_rect))))
        field_rects = _operator_burn_field_rects(
            pygame,
            table_rect,
            row_count=len(rows),
            scroll_row=table_scroll_row,
        )
        delete_rects = _operator_burn_delete_rects(
            pygame,
            table_rect,
            row_count=len(rows),
            scroll_row=table_scroll_row,
        )
        plan, errors = _operator_plan_from_rows(rows, option=option)
        can_launch = not errors
        current_plan_key = _operator_planned_trajectory_cache_key(plot_context, plan) if can_launch else None
        if trajectory_probe is not None and trajectory_probe.plan_key != current_plan_key:
            trajectory_probe = None
        validation_message = errors[0] if errors else _operator_plan_status(plan, option=option)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if equation_sheet_visible and event.key in {getattr(pygame, "K_PAGEDOWN", object())}:
                    equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
                        equation_sheet_scroll_px + PREVIEW_SCROLL_STEP_PX,
                        content_height=equation_sheet_content_height,
                        viewport_height=equation_sheet_content_rect[3],
                    )
                    continue
                if equation_sheet_visible and event.key in {getattr(pygame, "K_PAGEUP", object())}:
                    equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
                        equation_sheet_scroll_px - PREVIEW_SCROLL_STEP_PX,
                        content_height=equation_sheet_content_height,
                        viewport_height=equation_sheet_content_rect[3],
                    )
                    continue
                if objectives_visible and event.key in {getattr(pygame, "K_PAGEDOWN", object())}:
                    objectives_scroll_px = _clamp_operator_objectives_scroll_px(
                        objectives_scroll_px + PREVIEW_SCROLL_STEP_PX,
                        content_height=objectives_content_height,
                        viewport_height=objectives_content_rect[3],
                    )
                    continue
                if objectives_visible and event.key in {getattr(pygame, "K_PAGEUP", object())}:
                    objectives_scroll_px = _clamp_operator_objectives_scroll_px(
                        objectives_scroll_px - PREVIEW_SCROLL_STEP_PX,
                        content_height=objectives_content_height,
                        viewport_height=objectives_content_rect[3],
                    )
                    continue
                if event.key == pygame.K_TAB:
                    if read_only:
                        continue
                    active_cell = _operator_next_cell(
                        active_cell,
                        rows=rows,
                        backwards=bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT),
                    )
                    table_scroll_row = _operator_scroll_for_active_row(
                        active_cell[0],
                        table_scroll_row,
                        row_count=len(rows),
                        table_rect=table_rect,
                    )
                    continue
                if event.key in {pygame.K_RIGHT, pygame.K_d}:
                    if read_only:
                        continue
                    active_cell = (active_cell[0], min(active_cell[1] + 1, 3))
                    continue
                if event.key in {pygame.K_LEFT, pygame.K_a}:
                    if read_only:
                        continue
                    active_cell = (active_cell[0], max(active_cell[1] - 1, 0))
                    continue
                if event.key in {pygame.K_DOWN, pygame.K_s}:
                    if read_only:
                        continue
                    active_cell = (min(active_cell[0] + 1, len(rows) - 1), active_cell[1])
                    table_scroll_row = _operator_scroll_for_active_row(
                        active_cell[0],
                        table_scroll_row,
                        row_count=len(rows),
                        table_rect=table_rect,
                    )
                    continue
                if event.key in {pygame.K_UP, pygame.K_w}:
                    if read_only:
                        continue
                    active_cell = (max(active_cell[0] - 1, 0), active_cell[1])
                    table_scroll_row = _operator_scroll_for_active_row(
                        active_cell[0],
                        table_scroll_row,
                        row_count=len(rows),
                        table_rect=table_rect,
                    )
                    continue
                if event.key in {pygame.K_RETURN, pygame.K_KP_ENTER}:
                    if can_launch:
                        if not read_only:
                            _save_operator_burn_plan(option.scenario_id, plan)
                        return plan
                    continue
                if event.key == pygame.K_BACKSPACE:
                    if read_only:
                        continue
                    row_idx, col_idx = active_cell
                    rows[row_idx][col_idx] = rows[row_idx][col_idx][:-1]
                    trajectory_probe = None
                    continue
                if event.key == pygame.K_DELETE:
                    if read_only:
                        continue
                    row_idx, col_idx = active_cell
                    rows[row_idx][col_idx] = ""
                    trajectory_probe = None
                    continue
                event_text = getattr(event, "unicode", "")
                if not read_only and event_text and _operator_field_accepts_text(event_text):
                    row_idx, col_idx = active_cell
                    rows[row_idx][col_idx] += event_text
                    trajectory_probe = None
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if equation_sheet_visible and _pos_in_bounds(
                    mouse_pos,
                    (
                        objectives_overlay_rect.x,
                        objectives_overlay_rect.y,
                        objectives_overlay_rect.w,
                        objectives_overlay_rect.h,
                    ),
                ) and event.button in {4, 5}:
                    equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
                        equation_sheet_scroll_px
                        + (PREVIEW_SCROLL_STEP_PX if event.button == 5 else -PREVIEW_SCROLL_STEP_PX),
                        content_height=equation_sheet_content_height,
                        viewport_height=equation_sheet_content_rect[3],
                    )
                    continue
                if objectives_visible and _pos_in_bounds(
                    mouse_pos,
                    (
                        objectives_overlay_rect.x,
                        objectives_overlay_rect.y,
                        objectives_overlay_rect.w,
                        objectives_overlay_rect.h,
                    ),
                ) and event.button in {4, 5}:
                    objectives_scroll_px = _clamp_operator_objectives_scroll_px(
                        objectives_scroll_px + (PREVIEW_SCROLL_STEP_PX if event.button == 5 else -PREVIEW_SCROLL_STEP_PX),
                        content_height=objectives_content_height,
                        viewport_height=objectives_content_rect[3],
                    )
                    continue
                if _pos_in_bounds(mouse_pos, table_rect) and event.button in {4, 5}:
                    table_scroll_row = int(
                        max(
                            0,
                            min(
                                table_scroll_row + (1 if event.button == 5 else -1),
                                _operator_max_table_scroll_row(len(rows), table_rect),
                            ),
                        )
                    )
                    continue
                if event.button != 1:
                    continue
                if _pos_in_bounds(mouse_pos, (objectives_rect.x, objectives_rect.y, objectives_rect.w, objectives_rect.h)):
                    objectives_visible = not objectives_visible
                    if objectives_visible:
                        equation_sheet_visible = False
                        objectives_scroll_px = 0
                    continue
                if _pos_in_bounds(
                    mouse_pos,
                    (equation_sheet_rect.x, equation_sheet_rect.y, equation_sheet_rect.w, equation_sheet_rect.h),
                ):
                    equation_sheet_visible = not equation_sheet_visible
                    if equation_sheet_visible:
                        objectives_visible = False
                        equation_sheet_scroll_px = 0
                    continue
                if _pos_in_bounds(mouse_pos, (launch_rect.x, launch_rect.y, launch_rect.w, launch_rect.h)) and can_launch:
                    if not read_only:
                        _save_operator_burn_plan(option.scenario_id, plan)
                    return plan
                if _pos_in_bounds(mouse_pos, (cancel_rect.x, cancel_rect.y, cancel_rect.w, cancel_rect.h)):
                    return None
                if can_launch and current_plan_key is not None and not objectives_visible and not equation_sheet_visible:
                    probe_handled, probe_state, probe_time_s = _operator_trajectory_probe_from_click(
                        plot_context,
                        plan,
                        mouse_pos,
                        selected_probe=trajectory_probe,
                    )
                    if probe_handled:
                        trajectory_probe = (
                            OperatorTrajectoryProbe(
                                state_ric_km_s=tuple(float(value) for value in probe_state.reshape(6)),
                                time_s=float(probe_time_s or 0.0),
                                plan_key=current_plan_key,
                            )
                            if probe_state is not None
                            else None
                        )
                        continue
                if (
                    _pos_in_bounds(mouse_pos, (add_burn_rect.x, add_burn_rect.y, add_burn_rect.w, add_burn_rect.h))
                    and len(rows) < OPERATOR_BURN_MAX_ROWS
                    and not read_only
                ):
                    rows.append(["", "", "", ""])
                    trajectory_probe = None
                    active_cell = (len(rows) - 1, 0)
                    table_scroll_row = _operator_scroll_for_active_row(
                        active_cell[0],
                        table_scroll_row,
                        row_count=len(rows),
                        table_rect=table_rect,
                    )
                    continue
                clicked_delete_row = _operator_delete_row_at_pos(mouse_pos, delete_rects, table_rect=table_rect)
                if clicked_delete_row is not None and not read_only:
                    if len(rows) > 1:
                        row_idx = int(clicked_delete_row)
                        del rows[row_idx]
                        trajectory_probe = None
                        active_cell = (min(row_idx, len(rows) - 1), min(active_cell[1], 3))
                        table_scroll_row = _operator_scroll_for_active_row(
                            active_cell[0],
                            table_scroll_row,
                            row_count=len(rows),
                            table_rect=table_rect,
                        )
                    else:
                        rows[0] = ["", "", "", ""]
                        trajectory_probe = None
                        active_cell = (0, 0)
                        table_scroll_row = 0
                    continue
                clicked_cell = _operator_cell_at_pos(mouse_pos, field_rects, table_rect=table_rect)
                if clicked_cell is not None and not read_only:
                    active_cell = clicked_cell
            if event.type == getattr(pygame, "MOUSEWHEEL", object()):
                mouse_pos = pygame.mouse.get_pos()
                if equation_sheet_visible and _pos_in_bounds(
                    mouse_pos,
                    (
                        objectives_overlay_rect.x,
                        objectives_overlay_rect.y,
                        objectives_overlay_rect.w,
                        objectives_overlay_rect.h,
                    ),
                ):
                    equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
                        equation_sheet_scroll_px - int(event.y) * PREVIEW_SCROLL_STEP_PX,
                        content_height=equation_sheet_content_height,
                        viewport_height=equation_sheet_content_rect[3],
                    )
                elif objectives_visible and _pos_in_bounds(
                    mouse_pos,
                    (
                        objectives_overlay_rect.x,
                        objectives_overlay_rect.y,
                        objectives_overlay_rect.w,
                        objectives_overlay_rect.h,
                    ),
                ):
                    objectives_scroll_px = _clamp_operator_objectives_scroll_px(
                        objectives_scroll_px - int(event.y) * PREVIEW_SCROLL_STEP_PX,
                        content_height=objectives_content_height,
                        viewport_height=objectives_content_rect[3],
                    )
                elif _pos_in_bounds(mouse_pos, table_rect):
                    table_scroll_row = int(
                        max(
                            0,
                            min(
                                table_scroll_row - int(event.y),
                                _operator_max_table_scroll_row(len(rows), table_rect),
                            ),
                        )
                    )

        _draw_operator_plan_screen(
            pygame,
            screen,
            option=option,
            plan=plan if can_launch else OperatorBurnPlan(),
            rows=rows,
            active_cell=active_cell,
            field_rects=field_rects,
            delete_rects=delete_rects,
            table_scroll_row=table_scroll_row,
            plot_context=plot_context,
            trajectory_probe=trajectory_probe,
            frame_convention=frame_convention,
            validation_message=validation_message,
            can_launch=can_launch,
            launch_rect=launch_rect,
            cancel_rect=cancel_rect,
            add_burn_rect=add_burn_rect,
            objectives_rect=objectives_rect,
            equation_sheet_rect=equation_sheet_rect,
            equation_sheet_visible=equation_sheet_visible,
            equation_sheet_scroll_px=equation_sheet_scroll_px,
            objectives_visible=objectives_visible,
            objectives_scroll_px=objectives_scroll_px,
            read_only=read_only,
            demo_title=demo_title,
            launch_label=launch_label,
            font=font,
            small_font=small_font,
            title_font=title_font,
        )
        pygame.display.flip()
        clock.tick(60)


def _run_operator_prebrief_screen(
    pygame: Any,
    screen: Any,
    clock: Any,
    *,
    option: GameScenarioOption,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> bool:
    scroll_px = 0
    while True:
        width, height = screen.get_size()
        continue_rect = pygame.Rect(width - 214, height - 70, 162, 36)
        cancel_rect = pygame.Rect(width - 354, height - 70, 120, 36)
        content_rect = _operator_prebrief_content_rect(width, height)
        content_height = _operator_prebrief_content_height(option, font=font, small_font=small_font, width_px=content_rect[2])
        max_scroll = max(content_height - content_rect[3], 0)
        scroll_px = int(max(0, min(scroll_px, max_scroll)))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in {pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}:
                    return True
                if event.key in {pygame.K_DOWN, pygame.K_s, getattr(pygame, "K_PAGEDOWN", object())}:
                    scroll_px += PREVIEW_SCROLL_STEP_PX
                if event.key in {pygame.K_UP, pygame.K_w, getattr(pygame, "K_PAGEUP", object())}:
                    scroll_px -= PREVIEW_SCROLL_STEP_PX
            if event.type == pygame.MOUSEWHEEL:
                scroll_px -= int(event.y) * PREVIEW_SCROLL_STEP_PX
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if _pos_in_bounds(mouse_pos, (continue_rect.x, continue_rect.y, continue_rect.w, continue_rect.h)):
                    return True
                if _pos_in_bounds(mouse_pos, (cancel_rect.x, cancel_rect.y, cancel_rect.w, cancel_rect.h)):
                    return False
        _draw_operator_prebrief_screen(
            pygame,
            screen,
            option=option,
            scroll_px=scroll_px,
            continue_rect=continue_rect,
            cancel_rect=cancel_rect,
            font=font,
            small_font=small_font,
            title_font=title_font,
        )
        pygame.display.flip()
        clock.tick(60)


def _operator_plan_from_text(text: str, *, option: GameScenarioOption) -> tuple[OperatorBurnPlan, tuple[str, ...]]:
    try:
        plan = parse_operator_burn_plan(text)
    except ValueError as exc:
        return OperatorBurnPlan(), (str(exc),)
    errors = validate_operator_burn_plan(
        plan,
        total_delta_v_budget_m_s=_operator_delta_v_budget_m_s(option),
        max_time_s=option.time_budget_s,
    )
    return plan, errors


def _operator_plan_from_rows(rows: list[list[str]], *, option: GameScenarioOption) -> tuple[OperatorBurnPlan, tuple[str, ...]]:
    burns: list[OperatorBurn] = []
    for row_idx, row in enumerate(rows, start=1):
        values = [str(row[col] if col < len(row) else "").strip() for col in range(4)]
        if not any(values):
            continue
        if not values[0]:
            return OperatorBurnPlan(), (f"Burn {row_idx}: enter a time.",)
        parsed: list[float] = []
        for col_idx, value in enumerate(values):
            if col_idx > 0 and not value:
                parsed.append(0.0)
                continue
            try:
                parsed.append(float(value))
            except ValueError:
                return OperatorBurnPlan(), (f"Burn {row_idx}: {OPERATOR_BURN_HEADERS[col_idx]} must be numeric.",)
        burns.append(
            OperatorBurn(
                time_s=float(parsed[0]),
                delta_v_ric_m_s=(float(parsed[1]), float(parsed[2]), float(parsed[3])),
            )
        )
    plan = OperatorBurnPlan(burns=tuple(sorted(burns, key=lambda burn: burn.time_s)))
    errors = validate_operator_burn_plan(
        plan,
        total_delta_v_budget_m_s=_operator_delta_v_budget_m_s(option),
        max_time_s=option.time_budget_s,
    )
    return plan, errors


def _operator_rows_from_plan(plan: OperatorBurnPlan | None) -> list[list[str]]:
    if plan is None or not plan.burns:
        return [["", "", "", ""]]
    rows: list[list[str]] = []
    for burn in plan.burns[:OPERATOR_BURN_MAX_ROWS]:
        r, i, c = burn.delta_v_ric_m_s
        rows.append([
            f"{float(burn.time_s):g}",
            f"{float(r):g}",
            f"{float(i):g}",
            f"{float(c):g}",
        ])
    return rows or [["", "", "", ""]]


def _operator_plan_status(plan: OperatorBurnPlan, *, option: GameScenarioOption) -> str:
    budget = _operator_delta_v_budget_m_s(option)
    if budget is None:
        budget_text = "no total dV limit"
    else:
        budget_text = f"{budget:.1f} m/s total dV budget"
    return f"{len(plan.burns)} burns | {plan.total_delta_v_m_s:.2f} m/s planned | 5.0 m/s max per burn | {budget_text}"


def _operator_delta_v_budget_m_s(option: GameScenarioOption) -> float | None:
    controlled_id = str(option.controlled_object_id or "").strip()
    target_id = str(option.target_object_id or "").strip()
    if controlled_id and target_id and controlled_id == target_id and option.target_delta_v_budget_m_s is not None:
        return option.target_delta_v_budget_m_s
    return option.delta_v_budget_m_s


def _operator_field_accepts_text(value: str) -> bool:
    return all(char.isdigit() or char in ".-+" for char in str(value))


def _operator_next_cell(
    active_cell: tuple[int, int],
    *,
    rows: list[list[str]],
    backwards: bool = False,
) -> tuple[int, int]:
    row, col = active_cell
    linear = row * 4 + col + (-1 if backwards else 1)
    linear = max(0, min(linear, max(len(rows) * 4 - 1, 0)))
    return (linear // 4, linear % 4)


def _operator_cell_at_pos(
    pos: tuple[int, int],
    field_rects: list[list[Any]],
    *,
    table_rect: Any | None = None,
) -> tuple[int, int] | None:
    if table_rect is not None:
        table = _RectSpec(*table_rect) if not hasattr(table_rect, "height") else table_rect
        body_top = int(table.y) + 52
        if not _pos_in_bounds(pos, (int(table.x), body_top, int(table.width), max(int(table.height) - 74, 1))):
            return None
    for row_idx, row in enumerate(field_rects):
        for col_idx, rect in enumerate(row):
            if _pos_in_bounds(pos, (rect.x, rect.y, rect.w, rect.h)):
                return (row_idx, col_idx)
    return None


def _operator_burn_table_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    left, _right = _operator_game_plot_panel_rects(screen_width, screen_height)
    button_group_left = int(screen_width) - 52 - 126 - 10 - 120
    y = left.bottom + 10
    bottom_margin = 20
    min_height = 74 + OPERATOR_BURN_TABLE_MIN_VISIBLE_ROWS * OPERATOR_BURN_ROW_HEIGHT
    height = max(int(screen_height) - y - bottom_margin, min_height)
    width = max(min(button_group_left - left.x - 24, int(screen_width) - left.x - 72), left.width)
    return (left.x, y, width, height)


def _operator_add_burn_button_rect(pygame: Any, table_rect: Any) -> Any:
    table = pygame.Rect(*table_rect)
    return pygame.Rect(table.right - 34, table.y + 7, 24, 24)


def _operator_plan_panel_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    return (42, 104, max(int(screen_width) - 84, 620), max(int(screen_height) - 200, 360))


def _operator_game_plot_panel_rects(screen_width: int, screen_height: int) -> tuple[Any, Any]:
    left = _RectSpec(
        36,
        88,
        max((int(screen_width) - 108) // 2, 200),
        max(int(screen_height) - 256, 250),
    )
    right = _RectSpec(left.right + 36, left.y, left.width, left.height)
    return left, right


def _operator_objectives_button_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    del screen_height
    width = 196
    height = 36
    return (max(int(screen_width) - width - 52, 0), 34, width, height)


def _operator_equation_sheet_button_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    width = 120 + 10 + 126
    height = 36
    right = int(screen_width) - 52
    return (max(right - width, 0), max(int(screen_height) - 116, 0), width, height)


def _operator_objectives_overlay_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    left, right = _operator_game_plot_panel_rects(screen_width, screen_height)
    x = int(left.x) + 28
    y = int(left.y) + 34
    w = max(int(right.right) - x - 28, 320)
    h = max(min(int(left.height) - 68, int(screen_height) - y - 142), 220)
    return (x, y, w, h)


def _operator_objectives_content_rect(overlay_rect: Any) -> tuple[int, int, int, int]:
    rect = _RectSpec(*overlay_rect) if not hasattr(overlay_rect, "height") else overlay_rect
    inset = 22
    title_height = 52
    scrollbar_gutter = 18
    x = int(rect.x) + inset
    y = int(rect.y) + title_height
    width = max(int(rect.width) - inset * 2 - scrollbar_gutter, 60)
    height = max(int(rect.height) - title_height - 18, 40)
    return (x, y, width, height)


def _operator_objectives_content_height(
    option: GameScenarioOption,
    training_config: RPOTrainingConfig | None,
    *,
    font: Any,
    width_px: int,
) -> int:
    y = 0
    y += len(_wrapped_budget_lines(option, font, width_px)) * PREVIEW_LINE_HEIGHT
    if y:
        y += PREVIEW_SECTION_GAP
    y = _section_height(option.learning_goal, font, y, width_px)
    y = _section_height(option.player_brief or option.description, font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.pass_criteria, font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.instructor_notes, font, y + PREVIEW_SECTION_GAP, width_px)
    numeric_targets = _operator_objective_numeric_targets(option, training_config)
    if numeric_targets:
        y = _bullets_height(numeric_targets, font, y + PREVIEW_SECTION_GAP, width_px)
    return max(y, 1)


def _clamp_operator_objectives_scroll_px(scroll_px: int, *, content_height: int, viewport_height: int) -> int:
    max_scroll = max(int(content_height) - max(int(viewport_height), 1), 0)
    return int(max(0, min(int(scroll_px), max_scroll)))


def _operator_objective_numeric_targets(
    option: GameScenarioOption,
    training_config: RPOTrainingConfig | None,
) -> tuple[str, ...]:
    cfg = training_config
    targets: list[str] = []
    if cfg is not None and cfg.enabled:
        if _operator_is_finite(cfg.goal_nmt_radial_amplitude_km):
            targets.append(f"Desired radial amplitude: {format_distance_km(float(cfg.goal_nmt_radial_amplitude_km))}")
        if _operator_is_finite(cfg.goal_nmt_cross_track_amplitude_km) and abs(
            float(cfg.goal_nmt_cross_track_amplitude_km)
        ) > 0.0:
            targets.append(
                f"Desired cross-track amplitude: {format_distance_km(float(cfg.goal_nmt_cross_track_amplitude_km))}"
            )
        if _operator_is_finite(cfg.goal_nmt_element_tolerance_km):
            targets.append(f"NMT amplitude tolerance: {format_distance_km(float(cfg.goal_nmt_element_tolerance_km))}")
        if _operator_is_finite(cfg.goal_nmt_velocity_tolerance_km_s):
            targets.append(
                f"NMT velocity tolerance: {format_speed_km_s(float(cfg.goal_nmt_velocity_tolerance_km_s))}"
            )
        if _operator_is_finite(cfg.goal_nmt_cross_track_phase_deg) and abs(
            float(cfg.goal_nmt_cross_track_phase_deg)
        ) > 0.0:
            targets.append(f"Desired cross-track phase: {_operator_format_degrees(cfg.goal_nmt_cross_track_phase_deg)}")
        if _operator_is_finite(cfg.goal_range_km):
            targets.append(f"Goal range: {format_distance_km(float(cfg.goal_range_km))}")
        elif _operator_is_finite(option.goal_range_km):
            targets.append(f"Goal range: {format_distance_km(float(option.goal_range_km))}")
        if _operator_is_finite(cfg.goal_range_tolerance_km):
            targets.append(f"Goal range tolerance: {format_distance_km(float(cfg.goal_range_tolerance_km))}")
        if _operator_is_finite(cfg.goal_radius_km):
            targets.append(f"Goal radius: {format_distance_km(float(cfg.goal_radius_km))}")
        if _operator_is_finite(cfg.max_goal_speed_km_s):
            targets.append(f"Max goal speed: {format_speed_km_s(float(cfg.max_goal_speed_km_s))}")
        if _operator_is_finite(cfg.keepout_radius_km):
            targets.append(f"Keepout radius: {format_distance_km(float(cfg.keepout_radius_km))}")
        if _operator_is_finite(cfg.hard_speed_limit_radius_km) and _operator_is_finite(cfg.hard_speed_limit_km_s):
            targets.append(
                "Speed gate: "
                f"{format_speed_km_s(float(cfg.hard_speed_limit_km_s))} inside "
                f"{format_distance_km(float(cfg.hard_speed_limit_radius_km))}"
            )
        if _operator_is_finite(cfg.max_cross_track_amplitude_km):
            targets.append(f"Max cross-track amplitude: {format_distance_km(float(cfg.max_cross_track_amplitude_km))}")
        if _operator_is_finite(cfg.max_target_reference_range_km):
            targets.append(
                f"Max target reference range: {format_distance_km(float(cfg.max_target_reference_range_km))}"
            )
        for gate in cfg.approach_gates:
            gate_targets = [
                f"R {format_distance_km(float(gate.radial_ric_km))}",
                f"+/- {format_distance_km(float(gate.radial_tolerance_km))}",
            ]
            if _operator_is_finite(gate.max_abs_intrack_km):
                gate_targets.append(f"|I| <= {format_distance_km(float(gate.max_abs_intrack_km))}")
            if _operator_is_finite(gate.max_abs_cross_track_km):
                gate_targets.append(f"|C| <= {format_distance_km(float(gate.max_abs_cross_track_km))}")
            if _operator_is_finite(gate.max_total_speed_km_s):
                gate_targets.append(f"speed <= {format_speed_km_s(float(gate.max_total_speed_km_s))}")
            targets.append(f"{gate.name}: " + ", ".join(gate_targets))
        for gate in cfg.inspection_gates:
            center = ", ".join(format_distance_km(float(value)) for value in np.asarray(gate.center_ric_km).reshape(3))
            half_width = ", ".join(
                format_distance_km(float(value)) for value in np.asarray(gate.half_width_ric_km).reshape(3)
            )
            text = f"{gate.name}: center ({center}), half-width ({half_width})"
            if _operator_is_finite(gate.max_total_speed_km_s):
                text += f", speed <= {format_speed_km_s(float(gate.max_total_speed_km_s))}"
            targets.append(text)
        for constraint in cfg.sun_angle_constraints:
            text = f"{constraint.name}: half-angle {_operator_format_degrees(constraint.allowed_half_angle_deg)}"
            if _operator_is_finite(constraint.min_range_km):
                text += f", min range {format_distance_km(float(constraint.min_range_km))}"
            if _operator_is_finite(constraint.max_range_km):
                text += f", max range {format_distance_km(float(constraint.max_range_km))}"
            targets.append(text)
        for burn in cfg.required_phase_burns:
            targets.append(
                f"{burn.label}: |R| {format_distance_km(float(burn.radial_abs_km))} "
                f"+/- {format_distance_km(float(burn.radial_tolerance_km))}, "
                f"|I| <= {format_distance_km(float(burn.max_abs_intrack_km))}"
            )
        if _operator_is_finite(cfg.max_time_s):
            targets.append(f"Time budget: {_operator_format_seconds(cfg.max_time_s)}")
        if _operator_is_finite(cfg.max_delta_v_m_s):
            targets.append(f"Chaser delta-v budget: {format_speed_m_s(float(cfg.max_delta_v_m_s))}")
        if _operator_is_finite(cfg.max_target_delta_v_m_s):
            targets.append(f"Target delta-v budget: {format_speed_m_s(float(cfg.max_target_delta_v_m_s))}")
    else:
        if _operator_is_finite(option.goal_range_km):
            targets.append(f"Goal range: {format_distance_km(float(option.goal_range_km))}")
        if _operator_is_finite(option.goal_speed_km_s):
            targets.append(f"Max goal speed: {format_speed_km_s(float(option.goal_speed_km_s))}")
        if _operator_is_finite(option.time_budget_s):
            targets.append(f"Time budget: {_operator_format_seconds(float(option.time_budget_s))}")
        if _operator_is_finite(option.delta_v_budget_m_s):
            targets.append(f"Chaser delta-v budget: {format_speed_m_s(float(option.delta_v_budget_m_s))}")
        if _operator_is_finite(option.target_delta_v_budget_m_s):
            targets.append(f"Target delta-v budget: {format_speed_m_s(float(option.target_delta_v_budget_m_s))}")
    return tuple(dict.fromkeys(targets))


def _operator_equation_sheet_content_height(font: Any, *, width_px: int) -> int:
    text_width = _operator_equation_sheet_text_width(width_px)
    y = 0
    y = _section_height(
        "Circular-chief HCW intuition for the local RIC frame.",
        font,
        y,
        text_width,
    )
    y = _operator_lines_height(
        PygameRPODashboard._pause_overlay_equation_lines(),
        font,
        y + PREVIEW_SECTION_GAP,
        text_width,
    )
    y = _operator_lines_height(
        PygameRPODashboard._pause_overlay_takeaway_lines(),
        font,
        y + PREVIEW_SECTION_GAP,
        text_width,
    )
    return max(y, 1)


def _operator_equation_sheet_text_width(width_px: int) -> int:
    column_gap = 28
    if int(width_px) >= 760:
        return max((int(width_px) - column_gap) // 2, 240)
    return int(width_px)


def _operator_lines_height(items: tuple[str, ...], font: Any, y: int, width_px: int) -> int:
    y += PREVIEW_SECTION_TITLE_GAP
    for item in items:
        y += len(_wrap_text_px(item, font, width_px)) * PREVIEW_LINE_HEIGHT
    return y


def _operator_is_finite(value: Any) -> bool:
    return value is not None and bool(np.isfinite(float(value)))


def _operator_format_seconds(value_s: float) -> str:
    value = float(value_s)
    if not np.isfinite(value):
        return "--"
    return f"{value:.0f} s"


def _operator_format_degrees(value_deg: float) -> str:
    value = float(value_deg)
    if not np.isfinite(value):
        return "--"
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{text} deg"


def _operator_table_visible_rows(table_rect: Any) -> int:
    table = _RectSpec(*table_rect) if not hasattr(table_rect, "height") else table_rect
    return max(1, int((int(table.height) - 74) // OPERATOR_BURN_ROW_HEIGHT))


def _operator_max_table_scroll_row(row_count: int, table_rect: Any) -> int:
    return max(int(row_count) - _operator_table_visible_rows(table_rect), 0)


def _operator_scroll_for_active_row(
    active_row: int,
    scroll_row: int,
    *,
    row_count: int,
    table_rect: Any,
) -> int:
    max_scroll = _operator_max_table_scroll_row(row_count, table_rect)
    scroll = int(max(0, min(int(scroll_row), max_scroll)))
    visible = _operator_table_visible_rows(table_rect)
    if int(active_row) < scroll:
        return int(active_row)
    if int(active_row) >= scroll + visible:
        return int(min(max_scroll, int(active_row) - visible + 1))
    return scroll


def _operator_burn_field_rects(
    pygame: Any,
    table_rect: Any,
    *,
    row_count: int,
    scroll_row: int = 0,
) -> list[list[Any]]:
    table = pygame.Rect(*table_rect)
    gap = 8
    label_width = 24
    scrollbar_width = 14
    delete_width = 30
    usable = max(table.width - label_width - scrollbar_width - delete_width - gap * 4, 1)
    widths = [
        int(round(usable * 0.23)),
        int(round(usable * 0.25)),
        int(round(usable * 0.25)),
        usable - int(round(usable * 0.23)) - int(round(usable * 0.25)) * 2,
    ]
    rects: list[list[Any]] = []
    y = table.y + 52 - int(scroll_row) * OPERATOR_BURN_ROW_HEIGHT
    for _ in range(int(row_count)):
        x = table.x + label_width
        row: list[Any] = []
        for width in widths:
            row.append(pygame.Rect(x, y, max(int(width), 48), 28))
            x += int(width) + gap
        rects.append(row)
        y += OPERATOR_BURN_ROW_HEIGHT
    return rects


def _operator_burn_delete_rects(
    pygame: Any,
    table_rect: Any,
    *,
    row_count: int,
    scroll_row: int = 0,
) -> list[Any]:
    table = pygame.Rect(*table_rect)
    rects: list[Any] = []
    y = table.y + 54 - int(scroll_row) * OPERATOR_BURN_ROW_HEIGHT
    x = table.right - 42
    for _ in range(int(row_count)):
        rects.append(pygame.Rect(x, y, 22, 22))
        y += OPERATOR_BURN_ROW_HEIGHT
    return rects


def _operator_delete_row_at_pos(
    pos: tuple[int, int],
    delete_rects: list[Any],
    *,
    table_rect: Any | None = None,
) -> int | None:
    if table_rect is not None:
        table = _RectSpec(*table_rect) if not hasattr(table_rect, "height") else table_rect
        body_top = int(table.y) + 52
        if not _pos_in_bounds(pos, (int(table.x), body_top, int(table.width), max(int(table.height) - 74, 1))):
            return None
    for row_idx, rect in enumerate(delete_rects):
        if _pos_in_bounds(pos, (rect.x, rect.y, rect.w, rect.h)):
            return int(row_idx)
    return None


def _operator_initial_relative_ric_state(config_path: Path) -> tuple[float, float, float] | None:
    context = _operator_plot_context(config_path)
    if context.initial_relative_ric_km_s is None:
        return None
    return tuple(context.initial_relative_ric_km_s[:3])


def _operator_plot_context(config_path: Path, *, difficulty: str = "easy") -> OperatorPlotContext:
    try:
        with Path(config_path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except OSError:
        return OperatorPlotContext()
    metadata = dict(raw.get("metadata", {}) or {})
    try:
        training_config = RPOTrainingConfig.from_metadata(metadata)
        training_config = training_config_for_game_mode(training_config, game_mode="operator")
    except (TypeError, ValueError):
        training_config = None
    game = dict(metadata.get("game", {}) or {})
    training = dict(game.get("training", {}) or {})
    common_context: dict[str, Any] = {
        "training_config": training_config,
        "camera_mode": str(game.get("camera_mode", "reference") or "reference"),
        "coast_prediction_model": str(game.get("coast_prediction_model", "hcw") or "hcw").strip().lower(),
        "target_centered_plot_planes": _operator_game_plane_tuple(game.get("target_centered_plot_planes", ())),
        "target_centered_plot_axes": _operator_game_target_centered_plot_axes(game),
        "plot_overlays_in_zoom": bool(game.get("plot_overlays_in_zoom", True)),
        "plot_overlays_in_zoom_by_plane": _operator_game_plot_overlays_in_zoom_by_plane(game),
        "plot_axis_scale": _operator_game_plot_axis_scale(game),
        "plot_fixed_axis_half_span_km": _operator_game_plot_fixed_axis_half_span_km(game),
        "plot_equal_axis_scale_planes": _operator_game_plane_tuple(game.get("plot_equal_axis_scale_planes", ())),
        "proximity_ring_plot_planes": _operator_game_plane_tuple(
            game.get("proximity_ring_plot_planes", ("RI", "RC", "IC"))
        )
        or ("RI", "RC", "IC"),
    }
    preview_training_cfg, preview_snapshot, preview_dashboard_kwargs = _operator_pilot_first_frame_preview(
        Path(config_path),
        difficulty=difficulty,
    )
    if preview_training_cfg is not None:
        common_context["training_config"] = preview_training_cfg
    if preview_dashboard_kwargs:
        common_context["coast_prediction_model"] = str(
            preview_dashboard_kwargs.get("coast_prediction_model", common_context["coast_prediction_model"])
        )
    reference_state = _operator_reference_state_from_preview(preview_snapshot, preview_dashboard_kwargs)
    if reference_state is not None:
        common_context["reference_state_eci_km_s"] = reference_state
    common_context["pilot_initial_snapshot"] = preview_snapshot
    common_context["pilot_dashboard_kwargs"] = preview_dashboard_kwargs
    chaser_id = str(training.get("chaser_object_id", game.get("controlled_object_id", "chaser")) or "chaser")
    objects = dict(raw.get("objects", {}) or {})
    chaser = dict(objects.get(chaser_id, {}) or {})
    initial = dict(chaser.get("initial_state", {}) or {})
    relative = dict(initial.get("relative_to_target_ric", {}) or initial.get("relative_ric", {}) or {})
    state = relative.get("state")
    if state is None:
        return OperatorPlotContext(**common_context)
    try:
        values = [float(value) for value in state[:6]]
    except (TypeError, ValueError):
        return OperatorPlotContext(**common_context)
    if len(values) < 3:
        return OperatorPlotContext(**common_context)
    while len(values) < 6:
        values.append(0.0)
    rel6 = (values[0], values[1], values[2], values[3], values[4], values[5])
    mean_motion = _operator_target_mean_motion_rad_s(raw, training_config=training_config)
    return OperatorPlotContext(
        **common_context,
        initial_relative_ric_km_s=rel6,
        mean_motion_rad_s=mean_motion,
        initial_coast_ric_km_s=_operator_initial_coast_path(rel6, mean_motion_rad_s=mean_motion),
    )


def _operator_reference_state_from_preview(
    snapshot: Any | None,
    dashboard_kwargs: dict[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    if snapshot is None or not dashboard_kwargs:
        return None
    reference_id = str(dashboard_kwargs.get("reference_object_id", "") or dashboard_kwargs.get("target_object_id", ""))
    if not reference_id:
        return None
    truth = getattr(snapshot, "truth", {}) or {}
    try:
        state = truth.get(reference_id)
    except AttributeError:
        return None
    if state is None:
        return None
    arr = np.asarray(state, dtype=float).reshape(-1)
    if arr.size < 6 or not np.all(np.isfinite(arr[:6])):
        return None
    return tuple(float(value) for value in arr[:6])


def _operator_pilot_first_frame_preview(
    config_path: Path,
    *,
    difficulty: str = "easy",
) -> tuple[RPOTrainingConfig | None, Any | None, dict[str, Any]]:
    try:
        from sim.api import SimulationConfig
        from sim.game.manual import KeyboardCommandState
        from sim.game.runner import (
            _coast_prediction_orbit_fraction,
            _dashboard_object_ids,
            _force_game_acceleration_off_config,
            _game_camera_mode,
            _game_camera_rule_mode,
            _game_camera_rule_toggle_enabled,
            _game_chaser_sprite_diameter_km,
            _game_chaser_sprite_path,
            _game_coast_prediction_model,
            _game_control_mode,
            _game_controlled_object_id,
            _game_cr3bp_active_prediction_horizon_s,
            _game_cr3bp_coast_prediction_dt_s,
            _game_cr3bp_coast_prediction_horizon_mode,
            _game_cr3bp_coast_prediction_horizon_s,
            _game_cr3bp_projection_mode,
            _game_plot_axis_scale,
            _game_plot_equal_axis_scale_planes,
            _game_plot_fixed_axis_half_span_km,
            _game_plot_overlays_in_zoom,
            _game_plot_overlays_in_zoom_by_plane,
            _game_plot_prediction_full_trajectory_only,
            _game_plot_prediction_in_zoom,
            _game_plot_prediction_zoom_max_span_km,
            _game_proximity_ring_plot_planes,
            _game_relative_frame,
            _game_ric_reference_object_id,
            _game_show_target_hcw_path,
            _game_target_centered_plot_axes,
            _game_target_centered_plot_planes,
            _game_target_coast_prediction_dt_s,
            _game_target_coast_prediction_horizon_s,
            _game_target_sprite_diameter_km,
            _game_target_sprite_path,
            _game_timed_input_accumulator_enabled,
            _game_visual_extrapolation_enabled,
            _start_game_attempt,
            _training_config_with_sun_environment,
        )
    except Exception:
        return None, None, {}

    try:
        config = _force_game_acceleration_off_config(SimulationConfig.from_yaml(config_path))
        training_cfg = _training_config_with_sun_environment(
            RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {})),
            config,
        )
        training_cfg = training_config_for_game_mode(training_cfg, game_mode="operator")
        controlled_object_id = _game_controlled_object_id(config)
        command_state = KeyboardCommandState()
        command_state.use_timing_accumulator = _game_timed_input_accumulator_enabled(config)
        _session, _provider, snapshot = _start_game_attempt(
            config,
            command_state=command_state,
            training_cfg=training_cfg,
            controlled_object_id=controlled_object_id,
            attitude_rate_deg_s=45.0,
            control_mode=_game_control_mode(config),
            ric_reference_object_id=_game_ric_reference_object_id(config, training_cfg.target_object_id),
            operator_burn_plan=None,
        )
        anim_cfg = dict(config.scenario.outputs.animations or {})
        dashboard_target_id, dashboard_chaser_id = _dashboard_object_ids(training_cfg, anim_cfg)
        dashboard_kwargs = {
            "target_object_id": dashboard_target_id,
            "chaser_object_id": dashboard_chaser_id,
            "controlled_object_id": controlled_object_id,
            "reference_object_id": _game_ric_reference_object_id(config, training_cfg.target_object_id),
            "relative_frame": _game_relative_frame(config),
            "keepout_radius_km": training_cfg.keepout_radius_km,
            "goal_range_km": training_cfg.goal_range_km,
            "goal_range_tolerance_km": training_cfg.goal_range_tolerance_km,
            "goal_radius_km": training_cfg.goal_radius_km,
            "hard_speed_limit_radius_km": training_cfg.hard_speed_limit_radius_km,
            "hard_speed_limit_km_s": training_cfg.hard_speed_limit_km_s,
            "goal_relative_ric_km": training_cfg.goal_relative_ric_km,
            "goal_nmt_radial_amplitude_km": training_cfg.goal_nmt_radial_amplitude_km,
            "goal_nmt_cross_track_amplitude_km": training_cfg.goal_nmt_cross_track_amplitude_km,
            "goal_nmt_cross_track_phase_deg": training_cfg.goal_nmt_cross_track_phase_deg,
            "goal_nmt_center_ric_km": training_cfg.goal_nmt_center_ric_km,
            "goal_nmt_element_tolerance_km": training_cfg.goal_nmt_element_tolerance_km,
            "coast_prediction_orbit_fraction": _coast_prediction_orbit_fraction(difficulty),
            "coast_prediction_model": _game_coast_prediction_model(config),
            "cr3bp_projection_mode": _game_cr3bp_projection_mode(config),
            "cr3bp_coast_prediction_horizon_s": _game_cr3bp_coast_prediction_horizon_s(config) or 21600.0,
            "cr3bp_active_prediction_horizon_s": _game_cr3bp_active_prediction_horizon_s(config),
            "cr3bp_coast_prediction_horizon_mode": _game_cr3bp_coast_prediction_horizon_mode(config),
            "cr3bp_coast_prediction_dt_s": _game_cr3bp_coast_prediction_dt_s(config) or 300.0,
            "target_coast_prediction_horizon_s": _game_target_coast_prediction_horizon_s(config),
            "target_coast_prediction_dt_s": _game_target_coast_prediction_dt_s(config),
            "forbidden_regions": training_cfg.forbidden_regions,
            "approach_gates": training_cfg.approach_gates,
            "inspection_gates": training_cfg.inspection_gates,
            "sun_angle_constraints": training_cfg.sun_angle_constraints,
            "plot_overlays_in_zoom": _game_plot_overlays_in_zoom(config),
            "plot_overlays_in_zoom_by_plane": _game_plot_overlays_in_zoom_by_plane(config),
            "plot_prediction_in_zoom": _game_plot_prediction_in_zoom(config),
            "plot_prediction_zoom_max_span_km": _game_plot_prediction_zoom_max_span_km(config),
            "plot_prediction_full_trajectory_only": _game_plot_prediction_full_trajectory_only(config),
            "plot_axis_scale": _game_plot_axis_scale(config),
            "plot_fixed_axis_half_span_km": _game_plot_fixed_axis_half_span_km(config),
            "plot_equal_axis_scale_planes": _game_plot_equal_axis_scale_planes(config),
            "target_centered_plot_planes": _game_target_centered_plot_planes(config),
            "target_centered_plot_axes": _game_target_centered_plot_axes(config),
            "proximity_ring_plot_planes": _game_proximity_ring_plot_planes(config),
            "target_reference_object_id": training_cfg.target_reference_object_id,
            "camera_mode": _game_camera_mode(config),
            "camera_rule_mode": _game_camera_rule_mode(config),
            "camera_rule_toggle_enabled": _game_camera_rule_toggle_enabled(config),
            "target_sprite_path": _game_target_sprite_path(config),
            "chaser_sprite_path": _game_chaser_sprite_path(config),
            "target_sprite_diameter_km": _game_target_sprite_diameter_km(config),
            "chaser_sprite_diameter_km": _game_chaser_sprite_diameter_km(config),
            "show_target_coast_prediction": _game_show_target_hcw_path(config),
            "visual_extrapolation_enabled": _game_visual_extrapolation_enabled(config),
        }
        return training_cfg, snapshot, dashboard_kwargs
    except Exception:
        return None, None, {}


def _operator_game_plane_tuple(raw: Any) -> tuple[str, ...]:
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


def _operator_game_target_centered_plot_axes(game: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = game.get("target_centered_plot_axes", {}) or {}
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


def _operator_game_plot_overlays_in_zoom_by_plane(game: dict[str, Any]) -> dict[str, bool]:
    raw = game.get("plot_overlays_in_zoom_by_plane", {}) or {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, bool] = {}
    for plane, value in raw.items():
        key = str(plane or "").strip().upper()
        if key in {"RI", "RC", "IC"}:
            parsed[key] = bool(value)
    return parsed


def _operator_game_plot_axis_scale(game: dict[str, Any]) -> dict[str, tuple[float, float]]:
    raw = game.get("plot_axis_scale", {}) or {}
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
                pair = tuple(value)
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


def _operator_game_plot_fixed_axis_half_span_km(game: dict[str, Any]) -> dict[str, tuple[float | None, float | None]]:
    raw = game.get("plot_fixed_axis_half_span_km", {}) or {}
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
                pair = tuple(value)
            except TypeError:
                continue
            if len(pair) != 2:
                continue
        x_span = _operator_positive_float_or_none(pair[0])
        y_span = _operator_positive_float_or_none(pair[1])
        if x_span is not None or y_span is not None:
            parsed[key] = (x_span, y_span)
    return parsed


def _operator_positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def _operator_target_mean_motion_rad_s(
    raw_config: dict[str, Any],
    *,
    training_config: RPOTrainingConfig | None,
) -> float:
    objects = dict(raw_config.get("objects", {}) or {})
    target_id = "target"
    if training_config is not None and training_config.target_object_id:
        target_id = str(training_config.target_object_id)
    target = dict(objects.get(target_id, objects.get("target", {})) or {})
    initial = dict(target.get("initial_state", {}) or {})
    coes = dict(initial.get("coes", {}) or {})
    semi_major_axis_km = _optional_float(coes.get("a_km"))
    if semi_major_axis_km is None or not np.isfinite(semi_major_axis_km) or semi_major_axis_km <= 0.0:
        semi_major_axis_km = 7000.0
    return float(np.sqrt(EARTH_MU_KM3_S2 / (float(semi_major_axis_km) ** 3)))


def _operator_initial_coast_path(
    rel6: tuple[float, float, float, float, float, float],
    *,
    mean_motion_rad_s: float,
) -> tuple[tuple[float, float, float, float, float, float], ...]:
    n = float(mean_motion_rad_s)
    if not np.isfinite(n) or n <= 0.0:
        return ()
    period_s = 2.0 * np.pi / n
    times = np.linspace(0.0, period_s, 241)
    rows = _cw_coast_states(np.array(rel6, dtype=float), times, n)
    rows = rows[np.all(np.isfinite(rows), axis=1)]
    return tuple(tuple(float(value) for value in row[:6]) for row in rows)


def _operator_planned_trajectory(
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
) -> tuple[np.ndarray, np.ndarray]:
    if not plan.burns or plot_context.initial_relative_ric_km_s is None:
        return np.empty((0, 6), dtype=float), np.empty((0, 6), dtype=float)
    n = plot_context.mean_motion_rad_s
    if n is None or not np.isfinite(float(n)) or float(n) <= 0.0:
        return np.empty((0, 6), dtype=float), np.empty((0, 6), dtype=float)
    cache_key = _operator_planned_trajectory_cache_key(plot_context, plan)
    cached = plot_context._planned_trajectory_cache.get(cache_key)
    if cached is not None:
        return cached

    period_s = float(2.0 * np.pi / float(n))
    horizon_s = max(float(plan.burns[-1].time_s), 0.0) + period_s
    sample_dt_s = max(min(period_s / 240.0, 120.0), 5.0)
    state = np.array(plot_context.initial_relative_ric_km_s, dtype=float).reshape(6)
    chief_state = (
        np.array(plot_context.reference_state_eci_km_s, dtype=float).reshape(6)
        if plot_context.reference_state_eci_km_s is not None
        else None
    )
    current_t_s = 0.0
    rows: list[np.ndarray] = [state.copy()]
    time_rows: list[float] = [0.0]
    marker_rows: list[np.ndarray] = []

    for burn in plan.burns:
        burn_t_s = max(float(burn.time_s), current_t_s)
        state, chief_state = _operator_append_planned_coast(
            rows,
            time_rows=time_rows,
            state=state,
            chief_state_eci=chief_state,
            start_t_s=current_t_s,
            stop_t_s=burn_t_s,
            sample_dt_s=sample_dt_s,
            mean_motion_rad_s=float(n),
            coast_prediction_model=plot_context.coast_prediction_model,
        )
        delta_v_km_s = np.asarray(burn.delta_v_ric_m_s, dtype=float).reshape(3) / 1000.0
        if np.all(np.isfinite(delta_v_km_s)):
            state = state.copy()
            state[3:6] += delta_v_km_s
            rows.append(state.copy())
            time_rows.append(float(burn_t_s))
            marker_rows.append(state.copy())
        current_t_s = burn_t_s

    _operator_append_planned_coast(
        rows,
        time_rows=time_rows,
        state=state,
        chief_state_eci=chief_state,
        start_t_s=current_t_s,
        stop_t_s=horizon_s,
        sample_dt_s=sample_dt_s,
        mean_motion_rad_s=float(n),
        coast_prediction_model=plot_context.coast_prediction_model,
    )
    trajectory = np.vstack(rows) if rows else np.empty((0, 6), dtype=float)
    trajectory_times = np.asarray(time_rows, dtype=float).reshape(-1)
    markers = np.vstack(marker_rows) if marker_rows else np.empty((0, 6), dtype=float)
    finite = np.all(np.isfinite(trajectory), axis=1) if trajectory.size else np.empty(0, dtype=bool)
    if trajectory.size:
        trajectory = trajectory[finite]
        trajectory_times = trajectory_times[finite] if trajectory_times.size == finite.size else np.empty(0, dtype=float)
    marker_finite = np.all(np.isfinite(markers), axis=1) if markers.size else np.empty(0, dtype=bool)
    markers = markers[marker_finite] if markers.size else markers
    plot_context._planned_trajectory_cache[cache_key] = (trajectory, markers)
    plot_context._planned_trajectory_time_cache[cache_key] = trajectory_times
    return trajectory, markers


def _operator_planned_trajectory_times(plot_context: OperatorPlotContext, plan: OperatorBurnPlan) -> np.ndarray:
    cache_key = _operator_planned_trajectory_cache_key(plot_context, plan)
    times = plot_context._planned_trajectory_time_cache.get(cache_key)
    if times is None:
        _operator_planned_trajectory(plot_context, plan)
        times = plot_context._planned_trajectory_time_cache.get(cache_key)
    return np.asarray(times, dtype=float).reshape(-1) if times is not None else np.empty(0, dtype=float)


def _operator_append_planned_coast(
    rows: list[np.ndarray],
    *,
    time_rows: list[float] | None = None,
    state: np.ndarray,
    chief_state_eci: np.ndarray | None,
    start_t_s: float,
    stop_t_s: float,
    sample_dt_s: float,
    mean_motion_rad_s: float,
    coast_prediction_model: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    state_arr = np.array(state, dtype=float).reshape(6)
    chief_arr = None if chief_state_eci is None else np.array(chief_state_eci, dtype=float).reshape(6)
    duration_s = max(float(stop_t_s) - float(start_t_s), 0.0)
    if duration_s <= 1.0e-9:
        return state_arr, chief_arr
    interior_count = max(int(np.ceil(duration_s / max(float(sample_dt_s), 1.0e-6))), 1)
    times = np.linspace(0.0, duration_s, interior_count + 1, dtype=float)[1:]
    coast, chief_rows = _operator_planned_coast_states(
        state_arr,
        times,
        mean_motion_rad_s=float(mean_motion_rad_s),
        coast_prediction_model=coast_prediction_model,
        chief_state_eci=chief_arr,
    )
    for row in coast:
        rows.append(np.array(row, dtype=float).reshape(6).copy())
    if time_rows is not None:
        for t_s in times:
            time_rows.append(float(start_t_s) + float(t_s))
    chief_end = chief_rows[-1].copy() if chief_rows.size else _operator_propagate_reference_state(chief_arr, duration_s)
    return np.array(coast[-1], dtype=float).reshape(6), chief_end


def _operator_planned_coast_states(
    state: np.ndarray,
    times_s: np.ndarray,
    *,
    mean_motion_rad_s: float,
    coast_prediction_model: str,
    chief_state_eci: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float), np.empty((0, 6), dtype=float)
    chief_rows = _operator_reference_coast_states(chief_state_eci, times)
    if _operator_uses_ya_planned_coast(coast_prediction_model) and chief_state_eci is not None:
        try:
            rows = []
            for t_s, chief_end in zip(times, chief_rows, strict=True):
                phi = ya_closed_form_transition_matrix(float(t_s), chief_state_eci, chief_end)
                rows.append(phi @ np.asarray(state, dtype=float).reshape(6))
            result = np.vstack(rows)
            if result.shape == (times.size, 6) and np.all(np.isfinite(result)):
                return result, chief_rows
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
    return _cw_coast_states(np.array(state, dtype=float).reshape(6), times, float(mean_motion_rad_s)), chief_rows


def _operator_uses_ya_planned_coast(coast_prediction_model: str) -> bool:
    return _coast_prediction_model_key(coast_prediction_model) in {"elliptic_linear", "tschauner_hempel", "ts"}


def _operator_reference_coast_states(chief_state_eci: np.ndarray | None, times_s: np.ndarray) -> np.ndarray:
    if chief_state_eci is None:
        return np.empty((0, 6), dtype=float)
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float)
    state = np.asarray(chief_state_eci, dtype=float).reshape(6).copy()
    rows: list[np.ndarray] = []
    previous_t_s = 0.0
    for target_t_s in times:
        duration_s = max(float(target_t_s) - previous_t_s, 0.0)
        propagated = _operator_propagate_reference_state(state, duration_s)
        if propagated is not None:
            state = propagated
        rows.append(state.copy())
        previous_t_s = max(float(target_t_s), previous_t_s)
    return np.vstack(rows) if rows else np.empty((0, 6), dtype=float)


def _operator_propagate_reference_state(chief_state_eci: np.ndarray | None, dt_s: float) -> np.ndarray | None:
    if chief_state_eci is None:
        return None
    state = np.asarray(chief_state_eci, dtype=float).reshape(6).copy()
    remaining_s = max(float(dt_s), 0.0)
    if remaining_s <= 1.0e-9:
        return state
    max_step_s = 10.0
    zero_accel = np.zeros(3, dtype=float)
    while remaining_s > 1.0e-9:
        step_s = min(max_step_s, remaining_s)
        state = propagate_two_body_rk4(state, step_s, EARTH_MU_KM3_S2, zero_accel)
        remaining_s -= step_s
    return state


def _operator_planned_trajectory_cache_key(
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
) -> tuple[Any, ...]:
    initial = tuple(round(float(value), 12) for value in (plot_context.initial_relative_ric_km_s or ()))
    n = None if plot_context.mean_motion_rad_s is None else round(float(plot_context.mean_motion_rad_s), 15)
    reference = tuple(round(float(value), 12) for value in (plot_context.reference_state_eci_km_s or ()))
    burns = tuple(
        (
            round(float(burn.time_s), 6),
            tuple(round(float(value), 9) for value in burn.delta_v_ric_m_s),
        )
        for burn in plan.burns
    )
    return (initial, n, _coast_prediction_model_key(plot_context.coast_prediction_model), reference, burns)


def _operator_nmt_points(training_cfg: RPOTrainingConfig) -> np.ndarray:
    if training_cfg.goal_nmt_radial_amplitude_km is None:
        return np.empty((0, 3), dtype=float)
    return nmt_curve_points_km(
        radial_amplitude_km=float(training_cfg.goal_nmt_radial_amplitude_km),
        cross_track_amplitude_km=float(training_cfg.goal_nmt_cross_track_amplitude_km),
        cross_track_phase_deg=float(training_cfg.goal_nmt_cross_track_phase_deg),
        center_ric_km=np.array(training_cfg.goal_nmt_center_ric_km, dtype=float).reshape(3),
        samples=241,
    )


def _operator_nmt_boundary_points(training_cfg: RPOTrainingConfig) -> tuple[np.ndarray, ...]:
    radial = training_cfg.goal_nmt_radial_amplitude_km
    tol = training_cfg.goal_nmt_element_tolerance_km or training_cfg.goal_nmt_tolerance_km
    if radial is None or tol is None:
        return ()
    outer = float(radial) + max(float(tol), 0.0)
    inner = max(float(radial) - max(float(tol), 0.0), 0.0)
    points = [
        nmt_curve_points_km(
            radial_amplitude_km=outer,
            cross_track_amplitude_km=float(training_cfg.goal_nmt_cross_track_amplitude_km) + max(float(tol), 0.0),
            cross_track_phase_deg=float(training_cfg.goal_nmt_cross_track_phase_deg),
            center_ric_km=np.array(training_cfg.goal_nmt_center_ric_km, dtype=float).reshape(3),
            samples=241,
        )
    ]
    if inner > 0.0:
        points.append(
            nmt_curve_points_km(
                radial_amplitude_km=inner,
                cross_track_amplitude_km=max(
                    float(training_cfg.goal_nmt_cross_track_amplitude_km) - max(float(tol), 0.0),
                    0.0,
                ),
                cross_track_phase_deg=float(training_cfg.goal_nmt_cross_track_phase_deg),
                center_ric_km=np.array(training_cfg.goal_nmt_center_ric_km, dtype=float).reshape(3),
                samples=241,
            )
        )
    return tuple(points)


def _operator_plot_half_span(
    rows: list[np.ndarray],
    *,
    x_axis: int,
    y_axis: int,
    ring_radii: list[float | None],
) -> float:
    max_abs = 0.5
    for row in rows:
        arr = np.array(row, dtype=float)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[1] <= max(int(x_axis), int(y_axis)):
            continue
        projected = arr[:, [int(x_axis), int(y_axis)]]
        finite = projected[np.all(np.isfinite(projected), axis=1)]
        if finite.size:
            max_abs = max(max_abs, float(np.max(np.abs(finite))))
    for radius in ring_radii:
        if radius is not None and np.isfinite(float(radius)):
            max_abs = max(max_abs, float(radius))
    return max_abs * 1.25


def _operator_camera_center_ric(
    plot_context: OperatorPlotContext,
    *,
    chaser_current: np.ndarray,
    target_current: np.ndarray,
    x_axis: int,
    y_axis: int,
) -> np.ndarray:
    mode = str(plot_context.camera_mode or "reference").strip().lower()
    target = np.array(target_current, dtype=float).reshape(3)
    chaser = np.array(chaser_current, dtype=float).reshape(3)
    if mode not in {"target_pair", "satellite_pair", "pair"}:
        return np.zeros(3, dtype=float)
    plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
    if plane in plot_context.target_centered_plot_planes:
        return target
    if {int(x_axis), int(y_axis)} == {0, 2}:
        return np.zeros(3, dtype=float)
    center = _satellite_pair_camera_center(
        chaser=chaser,
        target=target,
        x_axis=int(x_axis),
        y_axis=int(y_axis),
        keep_rc_reference_centered=True,
    )
    override_axes = tuple(str(value or "").strip().lower() for value in plot_context.target_centered_plot_axes.get(plane, ()))
    if "x" in override_axes:
        center[int(x_axis)] = float(target[int(x_axis)])
    if "y" in override_axes:
        center[int(y_axis)] = float(target[int(y_axis)])
    return center


def _operator_forbidden_region_projection_points(
    training_cfg: RPOTrainingConfig,
    *,
    x_axis: int,
    y_axis: int,
    offset: np.ndarray,
) -> list[np.ndarray]:
    pts: list[np.ndarray] = []
    for region in training_cfg.forbidden_regions:
        if not _region_visible_on_plane(region, x_axis=int(x_axis), y_axis=int(y_axis)):
            continue
        if region.kind == "annular_sector":
            polygon = region.sector_polygon_ric()
            if polygon.size:
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                pts.append(polygon[:, [int(x_axis), int(y_axis)]])
            continue
        if region.kind == "cylinder":
            polygon = _cylinder_projection_polygon_ric(region, x_axis=int(x_axis), y_axis=int(y_axis))
            if polygon.size:
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                pts.append(polygon[:, [int(x_axis), int(y_axis)]])
            continue
        if region.kind == "sphere":
            polygon = _sphere_projection_polygon_ric(region, x_axis=int(x_axis), y_axis=int(y_axis))
            if polygon.size:
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                pts.append(polygon[:, [int(x_axis), int(y_axis)]])
            continue
        bounds = _finite_projected_region_bounds(region, x_axis=int(x_axis), y_axis=int(y_axis))
        if bounds is None:
            continue
        lo, hi = bounds
        corners = np.array(
            [
                [lo[0], lo[1]],
                [lo[0], hi[1]],
                [hi[0], lo[1]],
                [hi[0], hi[1]],
            ],
            dtype=float,
        )
        corners += np.array([offset[int(x_axis)], offset[int(y_axis)]], dtype=float).reshape(1, 2)
        pts.append(corners)
    return pts


def _operator_minimum_plot_span_km(
    plot_context: OperatorPlotContext,
    *,
    x_axis: int,
    y_axis: int,
    target_current: np.ndarray,
    nmt: np.ndarray,
    nmt_bounds: tuple[np.ndarray, ...],
) -> float:
    plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
    include_overlays = bool(
        plot_context.plot_overlays_in_zoom_by_plane.get(plane, plot_context.plot_overlays_in_zoom)
    )
    if not include_overlays:
        return MIN_PLOT_SPAN_KM
    origin_offset = np.array(target_current, dtype=float).reshape(3)
    overlay_points = _operator_forbidden_region_projection_points(
        plot_context.training_config or RPOTrainingConfig(enabled=False),
        x_axis=int(x_axis),
        y_axis=int(y_axis),
        offset=origin_offset,
    )
    if nmt.size:
        overlay_points.append(nmt[:, [int(x_axis), int(y_axis)]] + origin_offset[[int(x_axis), int(y_axis)]].reshape(1, 2))
    for boundary in nmt_bounds:
        if boundary.size:
            overlay_points.append(
                boundary[:, [int(x_axis), int(y_axis)]]
                + origin_offset[[int(x_axis), int(y_axis)]].reshape(1, 2)
            )
    finite: list[np.ndarray] = []
    for points in overlay_points:
        projected = np.array(points, dtype=float).reshape(-1, 2)
        projected = projected[np.all(np.isfinite(projected), axis=1)]
        if projected.size:
            finite.append(projected)
    span = MIN_PLOT_SPAN_KM
    if finite:
        span = max(span, float(np.max(np.abs(np.vstack(finite)))) * PLOT_OVERLAY_MARGIN)
    return span


def _operator_axis_scales_for_plot(
    plot_context: OperatorPlotContext,
    plot: Any,
    *,
    pts: list[np.ndarray],
    min_span_km: float,
    x_axis: int,
    y_axis: int,
    screen_size: tuple[int, int],
) -> tuple[float, float]:
    plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
    axis_scale_x, axis_scale_y = plot_context.plot_axis_scale.get(plane, (1.0, 1.0))
    fixed_x, fixed_y = plot_context.plot_fixed_axis_half_span_km.get(plane, (None, None))
    if fixed_x is None and fixed_y is None:
        base_scale = _operator_scale_for_plot(pts=pts, min_span_km=min_span_km, screen_size=screen_size)
        scale_x = base_scale * axis_scale_x
        scale_y = base_scale * axis_scale_y
    else:
        scale_x = _operator_scale_for_axis(pts=pts, axis_index=0, min_span_km=min_span_km, plot_px=plot.width)
        scale_y = _operator_scale_for_axis(pts=pts, axis_index=1, min_span_km=min_span_km, plot_px=plot.height)
        scale_x *= axis_scale_x
        scale_y *= axis_scale_y
        if fixed_x is not None:
            scale_x = _operator_scale_for_fixed_half_span(plot_px=plot.width, half_span_km=fixed_x)
        if fixed_y is not None:
            scale_y = _operator_scale_for_fixed_half_span(plot_px=plot.height, half_span_km=fixed_y)
    if plane in plot_context.plot_equal_axis_scale_planes:
        scale_x = scale_y
    return (scale_x, scale_y)


def _operator_scale_for_plot(*, pts: list[np.ndarray], min_span_km: float, screen_size: tuple[int, int]) -> float:
    finite: list[np.ndarray] = []
    for arr in pts:
        projected = np.array(arr, dtype=float).reshape(-1, 2)
        projected = projected[np.all(np.isfinite(projected), axis=1)]
        if projected.size:
            finite.append(projected)
    span = float(max(min_span_km, MIN_PLOT_SPAN_KM))
    if finite:
        span = max(float(np.max(np.abs(np.vstack(finite)))) * 1.2, span)
    width, height = screen_size
    px_span = max(min(width, height) * 0.28, 80.0)
    return float(px_span / max(span, 1.0e-9))


def _operator_scale_for_axis(*, pts: list[np.ndarray], axis_index: int, min_span_km: float, plot_px: float) -> float:
    finite: list[np.ndarray] = []
    for arr in pts:
        projected = np.array(arr, dtype=float).reshape(-1, 2)
        projected = projected[np.all(np.isfinite(projected), axis=1)]
        if projected.size:
            finite.append(projected[:, int(axis_index)])
    span = float(max(min_span_km, MIN_PLOT_SPAN_KM))
    if finite:
        span = max(float(np.max(np.abs(np.concatenate(finite)))) * 1.2, span)
    px_span = max(float(plot_px) * 0.42, 80.0)
    return float(px_span / max(span, 1.0e-9))


def _operator_scale_for_fixed_half_span(*, plot_px: float, half_span_km: float) -> float:
    span = float(half_span_km)
    if not np.isfinite(span) or span <= 0.0:
        span = MIN_PLOT_SPAN_KM
    return float(max(float(plot_px) * 0.5, 1.0) / span)


def _operator_rows_to_px(rows: np.ndarray, *, to_px: Any) -> list[tuple[int, int]]:
    arr = np.array(rows, dtype=float)
    if arr.size == 0:
        return []
    return [to_px(row[:3]) for row in arr.reshape(-1, arr.shape[-1]) if row.size >= 3]


def _operator_plot_transform_to_px(
    transform: dict[str, Any],
    point_ric_km: np.ndarray,
    *,
    x_axis: int,
    y_axis: int,
) -> tuple[int, int] | None:
    try:
        plot = transform["plot"]
        center_x = int(plot[0]) + int(plot[2]) // 2
        center_y = int(plot[1]) + int(plot[3]) // 2
        camera_center = np.asarray(transform["camera_center"], dtype=float).reshape(3)
        scale_x = float(transform["scale_x"])
        scale_y = float(transform["scale_y"])
        x_display_sign = float(transform["x_display_sign"])
        y_display_sign = float(transform["y_display_sign"])
    except (KeyError, TypeError, ValueError):
        return None
    shifted = np.asarray(point_ric_km, dtype=float).reshape(-1)[:3] - camera_center
    return (
        center_x + int(round(float(shifted[int(x_axis)]) * x_display_sign * scale_x)),
        center_y - int(round(float(shifted[int(y_axis)]) * y_display_sign * scale_y)),
    )


def _operator_trajectory_probe_from_click(
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
    pos: tuple[int, int],
    *,
    selected_probe: OperatorTrajectoryProbe | None = None,
) -> tuple[bool, np.ndarray | None, float | None]:
    dashboard = plot_context._preview_dashboard
    transforms = getattr(dashboard, "_frame_cache", {}).get("plot_transforms", {}) if dashboard is not None else {}
    if not transforms:
        return False, None, None
    trajectory, _markers = _operator_planned_trajectory(plot_context, plan)
    trajectory_times = _operator_planned_trajectory_times(plot_context, plan)
    trajectory_rows = (
        np.asarray(trajectory, dtype=float).reshape(-1, 6)
        if np.asarray(trajectory).size
        else np.empty((0, 6), dtype=float)
    )
    if trajectory_rows.size:
        finite = np.all(np.isfinite(trajectory_rows), axis=1)
        trajectory_rows = trajectory_rows[finite]
        trajectory_times = trajectory_times[finite] if trajectory_times.size == finite.size else np.empty(0, dtype=float)
    if trajectory_rows.size == 0 and selected_probe is None:
        return False, None, None
    for x_axis, y_axis in ((1, 0), (2, 0)):
        transform = transforms.get((x_axis, y_axis))
        if not isinstance(transform, dict):
            continue
        try:
            plot = transform["plot"]
            plot_bounds = (int(plot[0]), int(plot[1]), int(plot[2]), int(plot[3]))
        except (KeyError, TypeError, ValueError):
            continue
        if not _pos_in_bounds(pos, plot_bounds):
            continue
        if selected_probe is not None:
            selected_px = _operator_plot_transform_to_px(
                transform,
                np.asarray(selected_probe.state_ric_km_s, dtype=float)[:3],
                x_axis=x_axis,
                y_axis=y_axis,
            )
            if selected_px is not None and _operator_pixel_distance(pos, selected_px) <= 10.0:
                return True, None, None
        if trajectory_rows.size == 0:
            return False, None, None
        projected_rows = [
            (px, row, time_s)
            for row, time_s in zip(trajectory_rows, trajectory_times, strict=False)
            if (px := _operator_plot_transform_to_px(transform, row[:3], x_axis=x_axis, y_axis=y_axis)) is not None
        ]
        if not projected_rows:
            return False, None, None
        points = np.array([px for px, _row, _time_s in projected_rows], dtype=float)
        distances = np.linalg.norm(points - np.array(pos, dtype=float).reshape(1, 2), axis=1)
        closest_idx = int(np.argmin(distances))
        if float(distances[closest_idx]) <= 10.0:
            _px, row, time_s = projected_rows[closest_idx]
            return True, np.asarray(row, dtype=float).reshape(6).copy(), float(time_s)
        return False, None, None
    return False, None, None


def _operator_pixel_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def _draw_operator_grid(
    pygame: Any,
    screen: Any,
    plot: Any,
    *,
    center: tuple[int, int],
    scale_x: float,
    scale_y: float,
    half_span_km: float,
) -> None:
    pygame.draw.rect(screen, (8, 11, 16), plot)
    spacing_km = _operator_grid_spacing_km(half_span_km)
    limit = int(np.ceil(float(half_span_km) / spacing_km))
    for idx in range(-limit, limit + 1):
        x = center[0] + int(round(idx * spacing_km * scale_x))
        y = center[1] - int(round(idx * spacing_km * scale_y))
        if plot.left <= x <= plot.right:
            pygame.draw.line(screen, (24, 32, 44), (x, plot.top), (x, plot.bottom), width=1)
        if plot.top <= y <= plot.bottom:
            pygame.draw.line(screen, (24, 32, 44), (plot.left, y), (plot.right, y), width=1)


def _operator_grid_spacing_km(half_span_km: float) -> float:
    raw = max(float(half_span_km) / 3.0, 0.1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        spacing = multiplier * magnitude
        if spacing >= raw:
            return float(spacing)
    return float(10.0 * magnitude)


def _draw_operator_dashed_polyline(
    pygame: Any,
    screen: Any,
    points: list[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    for idx in range(len(points) - 1):
        if idx % 2 == 0:
            pygame.draw.line(screen, color, points[idx], points[idx + 1], width=width)


def _draw_operator_velocity_vector(
    pygame: Any,
    screen: Any,
    origin: tuple[int, int],
    velocity_ric_km_s: np.ndarray,
    *,
    x_axis: int,
    y_axis: int,
    x_display_sign: float = 1.0,
    y_display_sign: float = 1.0,
    color: tuple[int, int, int] = (86, 202, 245),
    length_px: float = 28.0,
) -> None:
    end = _operator_velocity_vector_endpoint(
        origin,
        velocity_ric_km_s,
        x_axis=x_axis,
        y_axis=y_axis,
        x_display_sign=x_display_sign,
        y_display_sign=y_display_sign,
        length_px=length_px,
    )
    if end is None:
        return
    pygame.draw.line(screen, color, origin, end, width=2)
    _draw_operator_arrowhead(pygame, screen, origin=origin, end=end, color=color)


def _draw_operator_arrowhead(
    pygame: Any,
    screen: Any,
    *,
    origin: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    direction = np.array([float(end[0] - origin[0]), float(end[1] - origin[1])], dtype=float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-9:
        return
    unit = direction / norm
    normal = np.array([-unit[1], unit[0]], dtype=float)
    tip = np.array([float(end[0]), float(end[1])], dtype=float)
    wing_center = tip - unit * 8.0
    for sign in (-1.0, 1.0):
        wing = wing_center + normal * sign * 5.0
        pygame.draw.line(
            screen,
            color,
            (int(round(tip[0])), int(round(tip[1]))),
            (int(round(wing[0])), int(round(wing[1]))),
            width=2,
        )


def _draw_operator_probe_dot(
    pygame: Any,
    screen: Any,
    center: tuple[int, int],
    *,
    label: str | None = None,
    font: Any | None = None,
) -> None:
    pygame.draw.circle(screen, (86, 202, 245), center, 6)
    pygame.draw.circle(screen, (8, 11, 16), center, 6, width=1)
    if label and font is not None:
        _text(screen, font, label, (center[0] + 9, center[1] - 18), (86, 202, 245))


def _operator_probe_time_label(time_s: float | None) -> str | None:
    if time_s is None or not np.isfinite(float(time_s)):
        return None
    value = max(float(time_s), 0.0)
    return f"T={value:.0f}s"


def _operator_velocity_vector_endpoint(
    origin: tuple[int, int],
    velocity_ric_km_s: np.ndarray,
    *,
    x_axis: int,
    y_axis: int,
    x_display_sign: float = 1.0,
    y_display_sign: float = 1.0,
    length_px: float = 28.0,
) -> tuple[int, int] | None:
    vel = np.array(velocity_ric_km_s, dtype=float).reshape(3)
    projected = vel[[int(x_axis), int(y_axis)]]
    projected = projected * np.array([float(x_display_sign), float(y_display_sign)], dtype=float)
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return None
    direction = projected / norm
    return (
        int(round(origin[0] + direction[0] * float(length_px))),
        int(round(origin[1] - direction[1] * float(length_px))),
    )


def _draw_operator_translucent_polygon(
    pygame: Any,
    screen: Any,
    plot: Any,
    points: list[tuple[int, int]],
    *,
    color: tuple[int, int, int, int],
) -> None:
    if len(points) < 3:
        return
    overlay = pygame.Surface((plot.width, plot.height), pygame.SRCALPHA)
    shifted = [(int(x - plot.x), int(y - plot.y)) for x, y in points]
    pygame.draw.polygon(overlay, color, shifted)
    screen.blit(overlay, plot.topleft)


def _draw_operator_forbidden_regions(
    pygame: Any,
    screen: Any,
    plot: Any,
    training_cfg: RPOTrainingConfig,
    *,
    x_axis: int,
    y_axis: int,
    to_px: Any,
    offset: np.ndarray,
) -> None:
    for region in training_cfg.forbidden_regions:
        if not _region_visible_on_plane(region, x_axis=int(x_axis), y_axis=int(y_axis)):
            continue
        if region.kind == "annular_sector":
            polygon = region.sector_polygon_ric()
            if not polygon.size:
                continue
            polygon = polygon.copy()
            polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
            points = [to_px(row) for row in polygon]
            if len(points) >= 3:
                _draw_operator_translucent_polygon(pygame, screen, plot, points, color=(168, 44, 54, 58))
                pygame.draw.lines(screen, (230, 80, 92), True, points, width=1)
            continue
        if region.kind == "cylinder":
            polygon = _cylinder_projection_polygon_ric(region, x_axis=int(x_axis), y_axis=int(y_axis))
            if not polygon.size:
                continue
            polygon = polygon.copy()
            polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
            points = [to_px(row) for row in polygon]
            if len(points) >= 3:
                _draw_operator_translucent_polygon(pygame, screen, plot, points, color=(168, 44, 54, 58))
                pygame.draw.lines(screen, (230, 80, 92), True, points, width=1)
            continue
        if region.kind == "sphere":
            polygon = _sphere_projection_polygon_ric(region, x_axis=int(x_axis), y_axis=int(y_axis))
            if not polygon.size:
                continue
            polygon = polygon.copy()
            polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
            points = [to_px(row) for row in polygon]
            if len(points) >= 3:
                _draw_operator_translucent_polygon(pygame, screen, plot, points, color=(168, 44, 54, 58))
                pygame.draw.lines(screen, (230, 80, 92), True, points, width=1)
            continue
        bounds = _finite_projected_region_bounds(region, x_axis=int(x_axis), y_axis=int(y_axis))
        if bounds is None:
            continue
        lo, hi = bounds
        p_min = np.zeros(3, dtype=float)
        p_max = np.zeros(3, dtype=float)
        p_min[int(x_axis)] = lo[0] + float(offset[int(x_axis)])
        p_min[int(y_axis)] = lo[1] + float(offset[int(y_axis)])
        p_max[int(x_axis)] = hi[0] + float(offset[int(x_axis)])
        p_max[int(y_axis)] = hi[1] + float(offset[int(y_axis)])
        a = to_px(p_min)
        b = to_px(p_max)
        rect = pygame.Rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1]))
        clipped = rect.clip(plot)
        if clipped.width <= 0 or clipped.height <= 0:
            continue
        fill = pygame.Surface((clipped.width, clipped.height), pygame.SRCALPHA)
        fill.fill((168, 44, 54, 58))
        screen.blit(fill, (clipped.x, clipped.y))
        pygame.draw.rect(screen, (230, 80, 92), clipped, width=1)


def _start_launcher_music(pygame: Any) -> bool:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(LAUNCHER_MUSIC_PATH))
        pygame.mixer.music.set_volume(0.55)
        pygame.mixer.music.play(-1)
    except (OSError, pygame.error):
        return False
    return True


def _stop_launcher_music(pygame: Any) -> None:
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except pygame.error:
        return


def _toggle_launcher_music(pygame: Any, *, music_enabled: bool) -> bool:
    if music_enabled:
        _stop_launcher_music(pygame)
        return False
    return _start_launcher_music(pygame)


def _load_start_screen_artwork(pygame: Any) -> Any | None:
    try:
        return pygame.image.load(str(START_SCREEN_LOGO_PATH)).convert()
    except (OSError, pygame.error):
        return None


def _start_artwork_rect(
    image_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_w, image_h = image_size
    screen_w, screen_h = screen_size
    if image_w <= 0 or image_h <= 0 or screen_w <= 0 or screen_h <= 0:
        return (0, 0, 0, 0)
    scale = min(screen_w / float(image_w), screen_h / float(image_h))
    width = max(int(round(image_w * scale)), 1)
    height = max(int(round(image_h * scale)), 1)
    x = int((screen_w - width) // 2)
    y = int((screen_h - height) // 2)
    return (x, y, width, height)


def _draw_start_screen(
    pygame: Any,
    screen: Any,
    *,
    artwork: Any | None = None,
    hero_font: Any,
    font: Any,
    small_font: Any,
) -> None:
    width, height = screen.get_size()
    if artwork is not None:
        rect_tuple = _start_artwork_rect(artwork.get_size(), (width, height))
        artwork_rect = pygame.Rect(*rect_tuple)
        if artwork_rect.width > 0 and artwork_rect.height > 0:
            scaled_artwork = pygame.transform.smoothscale(
                artwork,
                (artwork_rect.width, artwork_rect.height),
            )
            screen.blit(scaled_artwork, artwork_rect)
            return

    screen.fill((8, 12, 18))
    center = (width // 2, height // 2)
    accent = (96, 174, 224)
    pygame.draw.line(screen, (26, 38, 52), (center[0], 34), (center[0], height - 72), width=1)
    pygame.draw.line(screen, (26, 38, 52), (72, center[1] + 4), (width - 72, center[1] + 4), width=1)
    for radius in (170, 255, 340):
        rect = pygame.Rect(0, 0, radius * 2, int(radius * 0.72))
        rect.center = (center[0], center[1] - 30)
        pygame.draw.ellipse(screen, (22, 38, 56), rect, width=1)
    _text_centered(screen, hero_font, "Orbital Engagement Lab", (center[0], center[1] - 150), (238, 244, 250))
    title_y = center[1] - 82
    _text_centered(screen, hero_font, "RPO TRAINER", (center[0] + 3, title_y + 3), (10, 42, 68))
    _text_centered(screen, hero_font, "RPO TRAINER", (center[0], title_y), (210, 246, 255))
    _text_centered(screen, small_font, "RENDEZVOUS  PROXIMITY  OPERATIONS", (center[0], title_y + 42), (96, 174, 224))
    prompt_rect = pygame.Rect(0, 0, 360, 44)
    prompt_y = min(max(title_y + 104, center[1] + 150), height - 88)
    prompt_rect.center = (center[0], prompt_y)
    pygame.draw.rect(screen, (18, 34, 48), prompt_rect, border_radius=6)
    pygame.draw.rect(screen, accent, prompt_rect, width=1, border_radius=6)
    _text_centered(screen, font, "HIT ANY KEY TO BEGIN", prompt_rect.center, (238, 244, 250))
    _text_centered(screen, small_font, "Esc Quits", (center[0], prompt_rect.bottom + 22), (152, 166, 186))


def _draw_frame_convention_dialog(
    pygame: Any,
    screen: Any,
    *,
    convention: FrameConvention,
    dont_ask_again: bool,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))
    rect = pygame.Rect(*_frame_convention_dialog_rect(width, height))
    pygame.draw.rect(screen, (18, 24, 32), rect, border_radius=8)
    pygame.draw.rect(screen, (238, 184, 92), rect, width=1, border_radius=8)
    _text(screen, title_font, "Coordinate Frame Convention", (rect.x + 34, rect.y + 28), (238, 242, 248))
    _text(
        screen,
        small_font,
        "Choose the RIC display and input convention for this computer.",
        (rect.x + 36, rect.y + 78),
        (162, 178, 198),
    )

    choices = _frame_convention_dialog_choice_rects(width, height)
    preset = frame_convention_preset(convention)
    _draw_frame_convention_choice(
        pygame,
        screen,
        pygame.Rect(*choices[FRAME_CONVENTION_PRESET_OEL_DEFAULT]),
        "OEL Default",
        "Positive in-track points right; cross-track follows the OEL classroom display.",
        selected=preset == FRAME_CONVENTION_PRESET_OEL_DEFAULT,
        font=font,
        small_font=small_font,
    )
    _draw_frame_convention_choice(
        pygame,
        screen,
        pygame.Rect(*choices[FRAME_CONVENTION_PRESET_SPACE_FORCE]),
        "Space Force",
        "Positive in-track points left; cross-track is provisional until confirmed.",
        selected=preset == FRAME_CONVENTION_PRESET_SPACE_FORCE,
        font=font,
        small_font=small_font,
    )

    checkbox = pygame.Rect(*_frame_convention_dialog_checkbox_rect(width, height))
    pygame.draw.rect(screen, (12, 16, 22), checkbox, border_radius=4)
    pygame.draw.rect(screen, (238, 184, 92), checkbox, width=1, border_radius=4)
    if dont_ask_again:
        pygame.draw.line(screen, (238, 184, 92), (checkbox.x + 5, checkbox.centery), (checkbox.centerx - 1, checkbox.bottom - 6), width=2)
        pygame.draw.line(screen, (238, 184, 92), (checkbox.centerx - 1, checkbox.bottom - 6), (checkbox.right - 5, checkbox.y + 5), width=2)
    _text(screen, font, "Don't ask again", (checkbox.right + 10, checkbox.y + 2), (230, 238, 245))

    _draw_dialog_button(
        pygame,
        screen,
        pygame.Rect(*_frame_convention_dialog_continue_rect(width, height)),
        "Continue",
        font=font,
        enabled=True,
        primary=True,
    )


def _draw_frame_convention_choice(
    pygame: Any,
    screen: Any,
    rect: Any,
    label: str,
    detail: str,
    *,
    selected: bool,
    font: Any,
    small_font: Any,
) -> None:
    fill = (62, 48, 28) if selected else (12, 16, 22)
    stroke = (238, 184, 92) if selected else (70, 82, 100)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    _text(screen, font, label, (rect.x + 18, rect.y + 8), (246, 240, 226) if selected else (230, 238, 245))
    detail_color = (214, 202, 176) if selected else (156, 170, 190)
    detail_lines = _wrap_text_px(detail, small_font, max(int(rect.width) - 36, 40))[:2]
    for idx, line in enumerate(detail_lines):
        _text(screen, small_font, line, (rect.x + 18, rect.y + 32 + idx * 18), detail_color)


def _draw_launcher(
    pygame: Any,
    screen: Any,
    *,
    options: tuple[GameScenarioOption, ...],
    selected: int,
    scroll_offset: int,
    selected_difficulty: str,
    music_enabled: bool,
    preview_scroll_px: int,
    record_video: bool,
    selected_mode: str,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill((12, 16, 22))
    _text(screen, title_font, "Orbital Engagement Lab", (54, 36), (238, 242, 248))
    _draw_music_button(pygame, screen, enabled=music_enabled, mode=selected_mode, font=small_font)
    _draw_record_video_button(pygame, screen, enabled=record_video, mode=selected_mode, font=small_font)
    _draw_clear_progress_button(pygame, screen, mode=selected_mode, font=small_font)
    _draw_mode_toggle(pygame, screen, mode=selected_mode, font=small_font)
    _draw_settings_button(pygame, screen, mode=selected_mode)
    _text(screen, font, "Select RPO Training Level", (56, 78), (172, 186, 206))
    _draw_difficulty_picker(
        pygame,
        screen,
        selected_difficulty=selected_difficulty,
        selected_mode=selected_mode,
        font=small_font,
    )

    footer_y = max(height - small_font.get_height() - FOOTER_BOTTOM_MARGIN, PANEL_TOP + 16)
    mode_rect = _mode_toggle_rect(width, height)
    footer_text = "Up/Down Select   Left/Right Difficulty   O Mode   M Music   V Video   Enter Launch   Esc Quit"
    footer_max_width = max(mode_rect[0] - 72, 120)
    _text(
        screen,
        small_font,
        _fit_text_px(footer_text, small_font, footer_max_width),
        (56, footer_y),
        (220, 160, 160),
    )

    panel_height = _launcher_panel_height(height)
    list_rect = pygame.Rect(42, PANEL_TOP, 424, panel_height)
    preview_rect = pygame.Rect(*_preview_bounds(width, height))
    pygame.draw.rect(screen, (18, 24, 32), list_rect, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), list_rect, width=1, border_radius=8)

    visible = _visible_option_count(height)
    operator_mode = _normalize_game_mode(selected_mode) == "operator"
    for row, option in enumerate(options[scroll_offset : scroll_offset + visible]):
        idx = scroll_offset + row
        y = OPTION_Y + row * OPTION_ROW_HEIGHT
        rect = pygame.Rect(OPTION_X, y, OPTION_WIDTH, OPTION_HEIGHT)
        is_selected = idx == selected
        if operator_mode:
            fill = (62, 48, 28) if is_selected else (20, 27, 36)
            stroke = (238, 184, 92) if is_selected else (62, 50, 32)
        else:
            fill = (28, 48, 66) if is_selected else (20, 27, 36)
            stroke = (96, 174, 224) if is_selected else (48, 60, 76)
        pygame.draw.rect(screen, fill, rect, border_radius=8)
        pygame.draw.rect(screen, stroke, rect, width=2 if is_selected else 1, border_radius=8)
        _text(screen, font, option.title, (rect.x + 18, rect.y + 12), (238, 244, 250))
        if _show_progress_text(option):
            _text(
                screen,
                small_font,
                (
                    "Progress: "
                    f"{_progress_stars(option.completed_difficulties)}   "
                    f"High: {_format_high_score(option.high_score)}"
                ),
                (rect.x + 18, rect.y + 38),
                (162, 178, 198),
            )

    if len(options) > visible:
        _draw_scrollbar(
            pygame,
            screen,
            list_rect,
            count=len(options),
            visible=visible,
            scroll_offset=scroll_offset,
            mode=selected_mode,
        )

    _draw_preview(
        pygame,
        screen,
        preview_rect,
        option=options[selected],
        scroll_px=preview_scroll_px,
        mode=selected_mode,
        font=font,
        small_font=small_font,
    )


def _draw_scrollbar(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    count: int,
    visible: int,
    scroll_offset: int,
    mode: str = "pilot",
) -> None:
    track = pygame.Rect(rect.right - 12, rect.y + 12, 4, max(rect.height - 24, 20))
    pygame.draw.rect(screen, (38, 48, 62), track, border_radius=2)
    frac = min(float(visible) / max(float(count), 1.0), 1.0)
    thumb_h = max(int(track.height * frac), 28)
    max_scroll = max(int(count) - int(visible), 1)
    travel = max(track.height - thumb_h, 0)
    thumb_y = track.y + int(round(travel * min(max(float(scroll_offset) / float(max_scroll), 0.0), 1.0)))
    thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
    thumb_color = (238, 184, 92) if _normalize_game_mode(mode) == "operator" else (96, 174, 224)
    pygame.draw.rect(screen, thumb_color, thumb, border_radius=2)


def _draw_operator_plan_screen(
    pygame: Any,
    screen: Any,
    *,
    option: GameScenarioOption,
    plan: OperatorBurnPlan,
    rows: list[list[str]],
    active_cell: tuple[int, int],
    field_rects: list[list[Any]],
    delete_rects: list[Any],
    table_scroll_row: int,
    plot_context: OperatorPlotContext,
    trajectory_probe: OperatorTrajectoryProbe | None,
    frame_convention: FrameConvention,
    validation_message: str,
    can_launch: bool,
    launch_rect: Any,
    cancel_rect: Any,
    add_burn_rect: Any,
    objectives_rect: Any,
    equation_sheet_rect: Any,
    equation_sheet_visible: bool,
    equation_sheet_scroll_px: int,
    objectives_visible: bool,
    objectives_scroll_px: int,
    read_only: bool,
    demo_title: str,
    launch_label: str,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill((12, 16, 22))
    mode_title = "Operator Mode"
    _text(screen, title_font, mode_title, (54, 34), (238, 242, 248))
    title_x = 54 + _text_width(title_font, mode_title) + 34
    _text(
        screen,
        title_font,
        _fit_text_px(demo_title or option.title, title_font, max(width - title_x - 54, 120)),
        (title_x, 34),
        (238, 242, 248),
    )

    table = pygame.Rect(*_operator_burn_table_rect(width, height))
    plot_left, plot_right = _operator_game_plot_panel_rects(width, height)
    ri_rect = pygame.Rect(plot_left.x, plot_left.y, plot_left.width, plot_left.height)
    rc_rect = pygame.Rect(plot_right.x, plot_right.y, plot_right.width, plot_right.height)
    _draw_operator_initial_plot(
        pygame,
        screen,
        ri_rect,
        title="Initial RI",
        plot_context=plot_context,
        plan=plan,
        trajectory_probe=trajectory_probe,
        x_axis=1,
        y_axis=0,
        frame_convention=frame_convention,
        font=font,
        small_font=small_font,
    )
    _draw_operator_initial_plot(
        pygame,
        screen,
        rc_rect,
        title="Initial RC",
        plot_context=plot_context,
        plan=plan,
        trajectory_probe=trajectory_probe,
        x_axis=2,
        y_axis=0,
        frame_convention=frame_convention,
        font=font,
        small_font=small_font,
    )
    _draw_dialog_button(
        pygame,
        screen,
        objectives_rect,
        "Hide Mission Brief" if objectives_visible else "Show Mission Brief",
        font=small_font,
        enabled=True,
        primary=False,
    )
    _draw_dialog_button(
        pygame,
        screen,
        equation_sheet_rect,
        "Hide Equation Sheet" if equation_sheet_visible else "Show Equation Sheet",
        font=small_font,
        enabled=True,
        primary=False,
    )
    if objectives_visible:
        _draw_operator_objectives_overlay(
            pygame,
            screen,
            pygame.Rect(*_operator_objectives_overlay_rect(width, height)),
            option=option,
            training_config=plot_context.training_config,
            scroll_px=objectives_scroll_px,
            font=font,
            small_font=small_font,
        )
    if equation_sheet_visible:
        _draw_operator_equation_sheet_overlay(
            pygame,
            screen,
            pygame.Rect(*_operator_objectives_overlay_rect(width, height)),
            scroll_px=equation_sheet_scroll_px,
            font=font,
            small_font=small_font,
        )

    status_color = (162, 232, 174) if can_launch else (245, 126, 126)
    _draw_operator_burn_table(
        pygame,
        screen,
        table,
        rows=rows,
        active_cell=active_cell,
        field_rects=field_rects,
        delete_rects=delete_rects,
        add_burn_rect=add_burn_rect,
        add_enabled=len(rows) < OPERATOR_BURN_MAX_ROWS and not read_only,
        read_only=read_only,
        scroll_row=table_scroll_row,
        validation_message=validation_message,
        validation_color=status_color,
        font=small_font,
    )

    _draw_dialog_button(pygame, screen, cancel_rect, "Cancel", font=small_font, enabled=True, primary=False)
    _draw_dialog_button(pygame, screen, launch_rect, launch_label, font=small_font, enabled=can_launch, primary=True)


def _draw_operator_objectives_overlay(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    option: GameScenarioOption,
    training_config: RPOTrainingConfig | None,
    scroll_px: int,
    font: Any,
    small_font: Any,
) -> None:
    pygame.draw.rect(screen, (14, 19, 27), rect, border_radius=8)
    pygame.draw.rect(screen, (238, 184, 92), rect, width=1, border_radius=8)
    inset = 22
    x = int(rect.x) + inset
    _text(screen, font, "Mission Brief", (x, int(rect.y) + 18), (238, 244, 250))
    content_tuple = _operator_objectives_content_rect(rect)
    content = pygame.Rect(*content_tuple)
    width_px = max(content.width, 60)
    content_height = _operator_objectives_content_height(
        option,
        training_config,
        font=small_font,
        width_px=width_px,
    )
    scroll_px = _clamp_operator_objectives_scroll_px(
        scroll_px,
        content_height=content_height,
        viewport_height=content.height,
    )
    previous_clip = screen.get_clip()
    screen.set_clip(content)
    y = int(content.y) - int(scroll_px)
    max_y = y + int(content_height) + PREVIEW_LINE_HEIGHT
    for line in _wrapped_budget_lines(option, small_font, width_px):
        if y + PREVIEW_LINE_HEIGHT > max_y:
            break
        _text(screen, small_font, line, (x, y), (162, 178, 198))
        y += PREVIEW_LINE_HEIGHT
    if y > int(content.y) - int(scroll_px):
        y += PREVIEW_SECTION_GAP
    y = _draw_section(screen, small_font, "Objective", option.learning_goal, x, y, width_px, max_y)
    y = _draw_section(
        screen,
        small_font,
        "Brief",
        option.player_brief or option.description,
        x,
        y + PREVIEW_SECTION_GAP,
        width_px,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Pass Criteria",
        option.pass_criteria,
        x,
        y + PREVIEW_SECTION_GAP,
        width_px,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Instructor Notes",
        option.instructor_notes,
        x,
        y + PREVIEW_SECTION_GAP,
        width_px,
        max_y,
    )
    numeric_targets = _operator_objective_numeric_targets(option, training_config)
    if numeric_targets:
        y += PREVIEW_SECTION_GAP
        y = _draw_bullets(
            screen,
            small_font,
            "Numeric Targets",
            numeric_targets,
            x,
            y,
            width_px,
            max_y,
        )
    screen.set_clip(previous_clip)
    _draw_operator_objectives_scrollbar(
        pygame,
        screen,
        content,
        content_height=content_height,
        scroll_px=scroll_px,
    )


def _draw_operator_equation_sheet_overlay(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    scroll_px: int,
    font: Any,
    small_font: Any,
) -> None:
    pygame.draw.rect(screen, (14, 19, 27), rect, border_radius=8)
    pygame.draw.rect(screen, (238, 184, 92), rect, width=1, border_radius=8)
    inset = 22
    x = int(rect.x) + inset
    _text(screen, font, "Equation Sheet", (x, int(rect.y) + 18), (238, 244, 250))
    content_tuple = _operator_objectives_content_rect(rect)
    content = pygame.Rect(*content_tuple)
    width_px = max(content.width, 60)
    content_height = _operator_equation_sheet_content_height(small_font, width_px=width_px)
    scroll_px = _clamp_operator_objectives_scroll_px(
        scroll_px,
        content_height=content_height,
        viewport_height=content.height,
    )
    previous_clip = screen.get_clip()
    screen.set_clip(content)
    y = int(content.y) - int(scroll_px)
    max_y = y + int(content_height) + PREVIEW_LINE_HEIGHT
    text_width = _operator_equation_sheet_text_width(width_px)
    y = _draw_section(
        screen,
        small_font,
        "RIC Motion Card",
        "Circular-chief HCW intuition for the local RIC frame.",
        x,
        y,
        text_width,
        max_y,
    )
    y = _draw_operator_lines_section(
        screen,
        small_font,
        "HCW Equations",
        PygameRPODashboard._pause_overlay_equation_lines(),
        x,
        y + PREVIEW_SECTION_GAP,
        text_width,
        max_y,
    )
    y = _draw_operator_lines_section(
        screen,
        small_font,
        "Useful Intuition",
        PygameRPODashboard._pause_overlay_takeaway_lines(),
        x,
        y + PREVIEW_SECTION_GAP,
        text_width,
        max_y,
    )
    screen.set_clip(previous_clip)
    _draw_operator_equation_sheet_ric_diagram(
        pygame,
        screen,
        content,
        font=font,
        small_font=small_font,
    )
    _draw_operator_objectives_scrollbar(
        pygame,
        screen,
        content,
        content_height=content_height,
        scroll_px=scroll_px,
    )


def _draw_operator_equation_sheet_ric_diagram(
    pygame: Any,
    screen: Any,
    content: Any,
    *,
    font: Any,
    small_font: Any,
) -> None:
    width_px = int(content.width)
    if width_px < 760:
        return
    column_gap = 28
    text_width = _operator_equation_sheet_text_width(width_px)
    diagram_x = int(content.x) + text_width + column_gap
    diagram = pygame.Rect(
        diagram_x,
        int(content.y),
        max(int(content.right) - diagram_x, 220),
        int(content.height),
    )
    dashboard = _new_operator_preview_dashboard(pygame, screen, font=font, small_font=small_font)
    dashboard._target_sprite = dashboard._load_marker_sprite(TARGET_SPRITE_PATH)
    dashboard._chaser_sprite = dashboard._load_marker_sprite(CHASER_SPRITE_PATH)
    dashboard._draw_pause_ric_diagram(diagram)


def _draw_operator_lines_section(
    screen: Any,
    font: Any,
    title: str,
    items: tuple[str, ...],
    x: int,
    y: int,
    width_px: int,
    max_y: int,
) -> int:
    if y + PREVIEW_SECTION_TITLE_GAP > max_y:
        return max_y
    _text(screen, font, title, (x, y), (238, 244, 250))
    y += PREVIEW_SECTION_TITLE_GAP
    for item in items:
        for line in _wrap_text_px(item, font, width_px):
            if y + PREVIEW_LINE_HEIGHT > max_y:
                return max_y
            _text(screen, font, line, (x, y), (182, 194, 210))
            y += PREVIEW_LINE_HEIGHT
    return y


def _draw_operator_objectives_scrollbar(
    pygame: Any,
    screen: Any,
    content_rect: Any,
    *,
    content_height: int,
    scroll_px: int,
) -> None:
    viewport_height = max(int(content_rect.height), 1)
    if int(content_height) <= viewport_height:
        return
    track = pygame.Rect(content_rect.right + 8, content_rect.y, 4, viewport_height)
    pygame.draw.rect(screen, (38, 48, 62), track, border_radius=2)
    frac = min(float(viewport_height) / max(float(content_height), 1.0), 1.0)
    thumb_h = max(int(track.height * frac), 28)
    max_scroll = max(int(content_height) - viewport_height, 1)
    travel = max(track.height - thumb_h, 0)
    thumb_y = track.y + int(round(travel * min(max(float(scroll_px) / float(max_scroll), 0.0), 1.0)))
    thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
    pygame.draw.rect(screen, (238, 184, 92), thumb, border_radius=2)


def _draw_operator_prebrief_screen(
    pygame: Any,
    screen: Any,
    *,
    option: GameScenarioOption,
    scroll_px: int,
    continue_rect: Any,
    cancel_rect: Any,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill((12, 16, 22))
    title = "Operator Prebrief"
    _text(screen, title_font, title, (54, 34), (238, 242, 248))
    title_x = 54 + _text_width(title_font, title) + 34
    _text(
        screen,
        title_font,
        _fit_text_px(option.title, title_font, max(width - title_x - 54, 120)),
        (title_x, 34),
        (238, 242, 248),
    )

    panel = pygame.Rect(42, 104, max(width - 84, 620), max(height - 200, 360))
    pygame.draw.rect(screen, (18, 24, 32), panel, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), panel, width=1, border_radius=8)
    content_tuple = _operator_prebrief_content_rect(width, height)
    content = pygame.Rect(*content_tuple)
    previous_clip = screen.get_clip()
    screen.set_clip(content)
    y = content.y - int(scroll_px)
    max_y = content.y + _operator_prebrief_content_height(option, font=font, small_font=small_font, width_px=content.width)
    _text(screen, font, "Mission Brief", (content.x, y), (238, 244, 250))
    y += 34
    for line in _wrapped_budget_lines(option, small_font, content.width):
        _text(screen, small_font, line, (content.x, y), (162, 178, 198))
        y += PREVIEW_LINE_HEIGHT
    y += PREVIEW_SECTION_GAP
    y = _draw_section(screen, small_font, "Objective", option.learning_goal, content.x, y, content.width, max_y)
    y = _draw_section(
        screen,
        small_font,
        "Brief",
        option.player_brief or option.description,
        content.x,
        y + PREVIEW_SECTION_GAP,
        content.width,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Pass Criteria",
        option.pass_criteria,
        content.x,
        y + PREVIEW_SECTION_GAP,
        content.width,
        max_y,
    )
    _draw_bullets(
        screen,
        small_font,
        "Instructor Notes",
        option.instructor_notes,
        content.x,
        y + PREVIEW_SECTION_GAP,
        content.width,
        max_y,
    )
    screen.set_clip(previous_clip)
    _draw_preview_scrollbar(
        pygame,
        screen,
        content.inflate(PREVIEW_PADDING * 2, PREVIEW_PADDING * 2),
        content_height=_operator_prebrief_content_height(option, font=font, small_font=small_font, width_px=content.width),
        scroll_px=scroll_px,
    )
    _draw_dialog_button(pygame, screen, cancel_rect, "Cancel", font=small_font, enabled=True, primary=False)
    _draw_dialog_button(pygame, screen, continue_rect, "Script Burns", font=small_font, enabled=True, primary=True)


def _operator_prebrief_content_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    panel = _operator_plan_panel_rect(screen_width, screen_height)
    return (panel[0] + 24, panel[1] + 24, max(panel[2] - 48, 1), max(panel[3] - 58, 1))


def _operator_prebrief_content_height(
    option: GameScenarioOption,
    *,
    font: Any,
    small_font: Any,
    width_px: int,
) -> int:
    y = 0
    y += 34
    y += len(_wrapped_budget_lines(option, small_font, width_px)) * PREVIEW_LINE_HEIGHT
    y += PREVIEW_SECTION_GAP
    y = _section_height(option.learning_goal, small_font, y, width_px)
    y = _section_height(option.player_brief or option.description, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.pass_criteria, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.instructor_notes, small_font, y + PREVIEW_SECTION_GAP, width_px)
    return max(y, _text_height(font))


def _draw_dialog_button(
    pygame: Any,
    screen: Any,
    rect: Any,
    label: str,
    *,
    font: Any,
    enabled: bool,
    primary: bool,
) -> None:
    if primary and enabled:
        fill = (36, 72, 52)
        stroke = (108, 232, 142)
    elif primary:
        fill = (38, 42, 48)
        stroke = (76, 84, 96)
    else:
        fill = (32, 38, 48)
        stroke = (120, 132, 150)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    color = (230, 238, 245) if enabled else (126, 136, 150)
    _text_centered(screen, font, label, rect.center, color)


def _draw_operator_burn_table(
    pygame: Any,
    screen: Any,
    table: Any,
    *,
    rows: list[list[str]],
    active_cell: tuple[int, int],
    field_rects: list[list[Any]],
    delete_rects: list[Any],
    add_burn_rect: Any,
    add_enabled: bool,
    scroll_row: int,
    validation_message: str,
    validation_color: tuple[int, int, int],
    font: Any,
    read_only: bool = False,
) -> None:
    pygame.draw.rect(screen, (18, 24, 32), table, border_radius=6)
    pygame.draw.rect(screen, (70, 82, 100), table, width=1, border_radius=6)
    _text(screen, font, "Script impulsive RIC burns", (table.x + 12, table.y + 8), (238, 244, 250))
    _draw_operator_icon_button(pygame, screen, add_burn_rect, "+", font=font, enabled=add_enabled)
    for col_idx, header in enumerate(OPERATOR_BURN_HEADERS):
        if not field_rects:
            continue
        rect = field_rects[0][col_idx]
        _text(screen, font, header, (rect.x + 4, table.y + 32), (162, 178, 198))
    body_clip = pygame.Rect(table.x + 1, table.y + 52, table.width - 14, max(table.height - 74, 1))
    previous_clip = screen.get_clip()
    screen.set_clip(body_clip)
    for row_idx, row in enumerate(rows):
        if row_idx >= len(field_rects):
            break
        first_rect = field_rects[row_idx][0]
        if first_rect.bottom < body_clip.top or first_rect.top > body_clip.bottom:
            continue
        _text(screen, font, f"{row_idx + 1}", (table.x + 4, field_rects[row_idx][0].y + 7), (126, 138, 154))
        for col_idx, rect in enumerate(field_rects[row_idx]):
            active = (row_idx, col_idx) == active_cell
            fill = (22, 34, 46) if active else (18, 24, 32)
            stroke = (238, 184, 92) if active else (82, 96, 116)
            pygame.draw.rect(screen, fill, rect, border_radius=4)
            pygame.draw.rect(screen, stroke, rect, width=1, border_radius=4)
            value = str(row[col_idx] if col_idx < len(row) else "")
            _text(screen, font, _fit_text_px(value, font, rect.width - 12), (rect.x + 6, rect.y + 7), (230, 238, 245))
        if row_idx < len(delete_rects):
            _draw_operator_icon_button(pygame, screen, delete_rects[row_idx], "X", font=font, enabled=not read_only)
    screen.set_clip(previous_clip)
    _text(
        screen,
        font,
        _fit_text_px(validation_message, font, table.width - 28),
        (table.x + 12, table.bottom - 20),
        validation_color,
    )
    _draw_operator_table_scrollbar(pygame, screen, table, row_count=len(rows), scroll_row=scroll_row)


def _draw_operator_icon_button(
    pygame: Any,
    screen: Any,
    rect: Any,
    label: str,
    *,
    font: Any,
    enabled: bool,
) -> None:
    fill = (32, 38, 48) if enabled else (24, 28, 34)
    stroke = (120, 132, 150) if enabled else (72, 82, 96)
    pygame.draw.rect(screen, fill, rect, border_radius=4)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=4)
    color = (230, 238, 245) if enabled else (126, 136, 150)
    _text_centered(screen, font, str(label), rect.center, color)


def _draw_operator_table_scrollbar(
    pygame: Any,
    screen: Any,
    table: Any,
    *,
    row_count: int,
    scroll_row: int,
) -> None:
    visible = _operator_table_visible_rows(table)
    if int(row_count) <= visible:
        return
    track = pygame.Rect(table.right - 9, table.y + 54, 4, max(table.height - 78, 1))
    pygame.draw.rect(screen, (38, 48, 62), track, border_radius=2)
    frac = min(float(visible) / max(float(row_count), 1.0), 1.0)
    thumb_h = max(int(track.height * frac), 18)
    max_scroll = max(int(row_count) - int(visible), 1)
    travel = max(track.height - thumb_h, 0)
    thumb_y = track.y + int(round(travel * min(max(float(scroll_row) / float(max_scroll), 0.0), 1.0)))
    pygame.draw.rect(screen, (96, 174, 224), pygame.Rect(track.x, thumb_y, track.width, thumb_h), border_radius=2)


def _draw_operator_pilot_first_frame_plot(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    title: str,
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
    trajectory_probe: OperatorTrajectoryProbe | None,
    x_axis: int,
    y_axis: int,
    frame_convention: FrameConvention,
    font: Any,
    small_font: Any,
) -> bool:
    dashboard = _operator_preview_dashboard(
        pygame,
        screen,
        plot_context=plot_context,
        frame_convention=frame_convention,
        font=font,
        small_font=small_font,
    )
    if dashboard is None:
        return False
    dashboard._draw_panel(rect, title, x_axis=int(x_axis), y_axis=int(y_axis))
    trajectory, markers = _operator_planned_trajectory(plot_context, plan)
    _draw_operator_planned_trajectory_overlay(
        pygame,
        screen,
        dashboard,
        trajectory=trajectory,
        markers=markers,
        selected_state_ric_km_s=(
            np.asarray(trajectory_probe.state_ric_km_s, dtype=float) if trajectory_probe is not None else None
        ),
        selected_time_s=trajectory_probe.time_s if trajectory_probe is not None else None,
        x_axis=int(x_axis),
        y_axis=int(y_axis),
        font=small_font,
    )
    rel = np.asarray(dashboard._frame_cache.get("rel", np.empty((0, 6))), dtype=float)
    if rel.ndim == 2 and rel.shape[0] > 0 and rel.shape[1] >= 6:
        readout_state = (
            np.asarray(trajectory_probe.state_ric_km_s, dtype=float).reshape(6)
            if trajectory_probe is not None
            else rel[-1]
        )
        _draw_operator_initial_state_readout(
            screen,
            rect,
            rel_state_ric_km_s=readout_state,
            x_axis=x_axis,
            y_axis=y_axis,
            font=font,
            color=(86, 202, 245) if trajectory_probe is not None else (162, 178, 198),
        )
    return True


def _operator_preview_dashboard(
    pygame: Any,
    screen: Any,
    *,
    plot_context: OperatorPlotContext,
    frame_convention: FrameConvention,
    font: Any,
    small_font: Any,
) -> Any | None:
    snapshot = plot_context.pilot_initial_snapshot
    if snapshot is None or not plot_context.pilot_dashboard_kwargs:
        return None
    dashboard = plot_context._preview_dashboard
    if dashboard is None:
        dashboard = _new_operator_preview_dashboard(pygame, screen, font=font, small_font=small_font)
        for key, value in plot_context.pilot_dashboard_kwargs.items():
            setattr(dashboard, key, value)
        dashboard._target_sprite = dashboard._load_marker_sprite(
            _game_asset_path_or_default(getattr(dashboard, "target_sprite_path", None), TARGET_SPRITE_PATH)
        )
        dashboard._chaser_sprite = dashboard._load_marker_sprite(
            _game_asset_path_or_default(getattr(dashboard, "chaser_sprite_path", None), CHASER_SPRITE_PATH)
        )
        dashboard.push_snapshot(snapshot)
        dashboard._prepare_frame_cache()
        object.__setattr__(plot_context, "_preview_dashboard", dashboard)
    dashboard._plot_panel_title_gap_px = 8
    dashboard.pygame = pygame
    dashboard.screen = screen
    dashboard.font = font
    dashboard.small_font = small_font
    dashboard.large_font = font
    dashboard.frame_convention = normalize_frame_convention(frame_convention)
    dashboard._render_motion_enabled = False
    return dashboard


def _draw_operator_planned_trajectory_overlay(
    pygame: Any,
    screen: Any,
    dashboard: Any,
    *,
    trajectory: np.ndarray,
    markers: np.ndarray,
    selected_state_ric_km_s: np.ndarray | None,
    selected_time_s: float | None,
    x_axis: int,
    y_axis: int,
    font: Any,
) -> None:
    if np.asarray(trajectory, dtype=float).size == 0 and np.asarray(markers, dtype=float).size == 0:
        return
    transforms = getattr(dashboard, "_frame_cache", {}).get("plot_transforms", {})
    transform = transforms.get((int(x_axis), int(y_axis)))
    if not isinstance(transform, dict):
        return
    try:
        plot = pygame.Rect(*transform["plot"])
        camera_center = np.asarray(transform["camera_center"], dtype=float).reshape(3)
        scale_x = float(transform["scale_x"])
        scale_y = float(transform["scale_y"])
        x_display_sign = float(transform["x_display_sign"])
        y_display_sign = float(transform["y_display_sign"])
    except (KeyError, TypeError, ValueError):
        return

    def to_px(point: np.ndarray) -> tuple[int, int]:
        shifted = np.asarray(point, dtype=float).reshape(-1)[:3] - camera_center
        return (
            plot.centerx + int(round(float(shifted[int(x_axis)]) * x_display_sign * scale_x)),
            plot.centery - int(round(float(shifted[int(y_axis)]) * y_display_sign * scale_y)),
        )

    previous_clip = screen.get_clip()
    screen.set_clip(plot)
    planned = np.asarray(trajectory, dtype=float).reshape(-1, 6) if np.asarray(trajectory).size else np.empty((0, 6))
    planned = planned[np.all(np.isfinite(planned), axis=1)] if planned.size else planned
    if planned.shape[0] >= 2:
        points = [to_px(row[:3]) for row in planned]
        pygame.draw.lines(screen, (238, 184, 92), False, points, width=2)
        _draw_operator_dashed_polyline(pygame, screen, points, color=(255, 224, 142), width=1)
    marker_rows = np.asarray(markers, dtype=float).reshape(-1, 6) if np.asarray(markers).size else np.empty((0, 6))
    marker_rows = marker_rows[np.all(np.isfinite(marker_rows), axis=1)] if marker_rows.size else marker_rows
    for marker_idx, marker in enumerate(marker_rows, start=1):
        marker_px = to_px(marker[:3])
        _draw_operator_velocity_vector(
            pygame,
            screen,
            marker_px,
            marker[3:6],
            x_axis=x_axis,
            y_axis=y_axis,
            x_display_sign=x_display_sign,
            y_display_sign=y_display_sign,
            color=(86, 202, 245),
            length_px=30.0,
        )
        pygame.draw.circle(screen, OPERATOR_BURN_MARKER_COLOR, marker_px, 5)
        pygame.draw.circle(screen, (12, 16, 22), marker_px, 5, width=1)
        _text(screen, font, str(marker_idx), (marker_px[0] + 7, marker_px[1] - 14), OPERATOR_BURN_MARKER_COLOR)
    if selected_state_ric_km_s is not None:
        _draw_operator_probe_dot(
            pygame,
            screen,
            to_px(np.asarray(selected_state_ric_km_s, dtype=float)[:3]),
            label=_operator_probe_time_label(selected_time_s),
            font=font,
        )
    screen.set_clip(previous_clip)


def _new_operator_preview_dashboard(pygame: Any, screen: Any, *, font: Any, small_font: Any) -> Any:
    dashboard = object.__new__(PygameRPODashboard)
    for item in fields(PygameRPODashboard):
        if item.default_factory is not MISSING:  # type: ignore[attr-defined]
            value = item.default_factory()  # type: ignore[misc]
        elif item.default is not MISSING:
            value = item.default
        else:
            continue
        setattr(dashboard, item.name, value)
    dashboard.pygame = pygame
    dashboard.screen = screen
    dashboard.clock = None
    dashboard.font = font
    dashboard.small_font = small_font
    dashboard.large_font = font
    dashboard.closed = False
    dashboard.t_s = []
    dashboard.sample_wall_s = []
    dashboard.rel_hist = []
    dashboard.target_rel_hist = []
    dashboard.target_reference_rel_hist = []
    dashboard.target_eci_hist = []
    dashboard.chaser_eci_hist = []
    dashboard.thrust_hist = []
    dashboard.thrust_ric_hist = []
    max_rows = int(max(getattr(dashboard, "max_history", 900), 2))
    dashboard._rel_array = _new_history_ring(6, max_rows)
    dashboard._target_rel_array = _new_history_ring(6, max_rows)
    dashboard._target_reference_rel_array = _new_history_ring(6, max_rows)
    dashboard._target_eci_array = _new_history_ring(6, max_rows)
    dashboard._chaser_eci_array = _new_history_ring(6, max_rows)
    dashboard._thrust_ric_array = _new_history_ring(3, max_rows)
    dashboard.mean_motion_rad_s = None
    dashboard.reference_state_eci = None
    dashboard.target_orbit_reference_state_eci = None
    dashboard.target_true_anomaly_deg = None
    dashboard.briefing_scroll_px = 0
    dashboard.mission_banner_scroll_px = 0
    dashboard._frame_cache = {}
    dashboard._raw_frame_cache = {}
    dashboard._frame_cache_dirty = True
    dashboard._render_motion_enabled = False
    dashboard._render_wall_time_s = 0.0
    dashboard._render_speed_multiple = 1.0
    dashboard._prediction_cache = {}
    dashboard._briefing_layout_cache = {}
    dashboard._mission_banner_layout_cache = {}
    dashboard._text_cache = {}
    dashboard._operator_projection_transition = None
    dashboard._target_sprite = None
    dashboard._chaser_sprite = None
    dashboard._sprite_scale_cache = {}
    return dashboard


def _draw_operator_initial_plot(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    title: str,
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
    trajectory_probe: OperatorTrajectoryProbe | None,
    x_axis: int,
    y_axis: int,
    frame_convention: FrameConvention,
    font: Any,
    small_font: Any,
) -> None:
    if _draw_operator_pilot_first_frame_plot(
        pygame,
        screen,
        rect,
        title=title,
        plot_context=plot_context,
        plan=plan,
        trajectory_probe=trajectory_probe,
        x_axis=x_axis,
        y_axis=y_axis,
        frame_convention=frame_convention,
        font=font,
        small_font=small_font,
    ):
        return
    pygame.draw.rect(screen, (20, 27, 36), rect, border_radius=10)
    pygame.draw.rect(screen, (80, 92, 110), rect, width=1, border_radius=10)
    _text(screen, font, title, (rect.x + 14, rect.y + 10), (230, 235, 242))
    plot = rect.inflate(-48, -72)
    plot.y += 8
    pygame.draw.rect(screen, (8, 11, 16), plot)
    pygame.draw.rect(screen, (72, 84, 102), plot, width=1)
    rel6 = plot_context.initial_relative_ric_km_s
    if rel6 is None:
        _text(screen, font, "Initial RIC unavailable", (plot.x, plot.centery - 8), (245, 126, 126))
        return
    training_cfg = plot_context.training_config or RPOTrainingConfig(enabled=False)
    readout_state = (
        np.asarray(trajectory_probe.state_ric_km_s, dtype=float).reshape(6)
        if trajectory_probe is not None
        else np.asarray(rel6, dtype=float).reshape(6)
    )
    rel = np.array(readout_state[:3], dtype=float)
    initial_rel = np.array(rel6[:3], dtype=float)
    initial_rel_vel = np.array(rel6[3:6], dtype=float)
    coast = np.array(plot_context.initial_coast_ric_km_s, dtype=float).reshape(-1, 6)
    nmt = _operator_nmt_points(training_cfg)
    nmt_bounds = _operator_nmt_boundary_points(training_cfg)
    target_current = np.zeros(3, dtype=float)
    camera_center = _operator_camera_center_ric(
        plot_context,
        chaser_current=initial_rel,
        target_current=target_current,
        x_axis=x_axis,
        y_axis=y_axis,
    )
    projected_sets: list[np.ndarray] = [
        (initial_rel - camera_center).reshape(1, 3)[:, [int(x_axis), int(y_axis)]],
        (target_current - camera_center).reshape(1, 3)[:, [int(x_axis), int(y_axis)]],
    ]
    if coast.size:
        projected_sets.append((coast[:, :3] - camera_center.reshape(1, 3))[:, [int(x_axis), int(y_axis)]])
    if nmt.size and not nmt_bounds:
        projected_sets.append((nmt - camera_center.reshape(1, 3))[:, [int(x_axis), int(y_axis)]])
    projected_sets.extend(
        (boundary - camera_center.reshape(1, 3))[:, [int(x_axis), int(y_axis)]]
        for boundary in nmt_bounds
        if boundary.size
    )
    min_span = _operator_minimum_plot_span_km(
        plot_context,
        x_axis=x_axis,
        y_axis=y_axis,
        target_current=target_current,
        nmt=nmt,
        nmt_bounds=nmt_bounds,
    )
    scale_x, scale_y = _operator_axis_scales_for_plot(
        plot_context,
        plot,
        pts=projected_sets,
        min_span_km=min_span,
        x_axis=x_axis,
        y_axis=y_axis,
        screen_size=screen.get_size(),
    )
    center = (plot.centerx, plot.centery)
    x_display_sign = frame_convention_display_axis_sign(frame_convention, x_axis)
    y_display_sign = frame_convention_display_axis_sign(frame_convention, y_axis)

    def to_px(point: np.ndarray | tuple[float, float, float]) -> tuple[int, int]:
        arr = np.array(point, dtype=float).reshape(-1)[:3] - camera_center
        return (
            center[0] + int(round(float(arr[int(x_axis)]) * x_display_sign * scale_x)),
            center[1] - int(round(float(arr[int(y_axis)]) * y_display_sign * scale_y)),
        )

    def circle_rect(center_px: tuple[int, int], radius_km: float) -> Any:
        radius_x_px = max(1, int(round(float(radius_km) * scale_x)))
        radius_y_px = max(1, int(round(float(radius_km) * scale_y)))
        return pygame.Rect(center_px[0] - radius_x_px, center_px[1] - radius_y_px, radius_x_px * 2, radius_y_px * 2)

    previous_clip = screen.get_clip()
    screen.set_clip(plot)
    half_span = max(min_span, MIN_PLOT_SPAN_KM)
    if projected_sets:
        finite_span = [
            np.max(np.abs(np.array(points, dtype=float).reshape(-1, 2)))
            for points in projected_sets
            if np.array(points, dtype=float).size
        ]
        if finite_span:
            half_span = max(half_span, float(max(finite_span)) * 1.2)
    _draw_operator_grid(pygame, screen, plot, center=center, scale_x=scale_x, scale_y=scale_y, half_span_km=half_span)
    target_px = to_px(target_current)
    _draw_operator_forbidden_regions(
        pygame,
        screen,
        plot,
        training_cfg,
        x_axis=x_axis,
        y_axis=y_axis,
        to_px=to_px,
        offset=target_current,
    )
    plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
    if plane in plot_context.proximity_ring_plot_planes:
        if training_cfg.keepout_radius_km is not None and float(training_cfg.keepout_radius_km) > 0.0:
            pygame.draw.ellipse(
                screen,
                (190, 68, 68),
                circle_rect(target_px, float(training_cfg.keepout_radius_km)),
                width=2,
            )
        if training_cfg.goal_range_km is not None and float(training_cfg.goal_range_km) > 0.0:
            outer = float(training_cfg.goal_range_km)
            inner = outer
            if training_cfg.goal_range_tolerance_km is not None:
                tol = max(float(training_cfg.goal_range_tolerance_km), 0.0)
                inner = max(outer - tol, 0.0)
                outer += tol
            pygame.draw.ellipse(screen, (78, 178, 112), circle_rect(target_px, outer), width=2)
            if inner > 0.0 and not np.isclose(inner, outer):
                pygame.draw.ellipse(screen, (78, 178, 112), circle_rect(target_px, inner), width=2)
        if training_cfg.goal_radius_km is not None and float(training_cfg.goal_radius_km) > 0.0:
            goal_center = np.array(training_cfg.goal_relative_ric_km, dtype=float).reshape(3)
            pygame.draw.ellipse(
                screen,
                (78, 178, 112),
                circle_rect(to_px(goal_center), float(training_cfg.goal_radius_km)),
                width=2,
            )
        if (
            training_cfg.hard_speed_limit_radius_km is not None
            and float(training_cfg.hard_speed_limit_radius_km) > 0.0
        ):
            pygame.draw.ellipse(
                screen,
                (232, 194, 74),
                circle_rect(target_px, float(training_cfg.hard_speed_limit_radius_km)),
                width=2,
            )
    for boundary in nmt_bounds:
        boundary_points = _operator_rows_to_px(boundary, to_px=to_px)
        if len(boundary_points) >= 2:
            pygame.draw.lines(screen, (78, 178, 112), True, boundary_points, width=2)
    nmt_points = _operator_rows_to_px(nmt, to_px=to_px)
    if len(nmt_points) >= 2 and not nmt_bounds:
        _draw_operator_dashed_polyline(pygame, screen, nmt_points, color=(120, 236, 154), width=2)
    coast_points = _operator_rows_to_px(coast[:, :3], to_px=to_px) if coast.size else []
    if len(coast_points) >= 2:
        _draw_operator_dashed_polyline(pygame, screen, coast_points, color=(96, 174, 224), width=2)
    trajectory, markers = _operator_planned_trajectory(plot_context, plan)
    trajectory_points = _operator_rows_to_px(trajectory[:, :3], to_px=to_px) if trajectory.size else []
    if len(trajectory_points) >= 2:
        pygame.draw.lines(screen, (238, 184, 92), False, trajectory_points, width=2)
        _draw_operator_dashed_polyline(pygame, screen, trajectory_points, color=(255, 224, 142), width=1)
    if markers.size:
        for marker_idx, marker in enumerate(markers.reshape(-1, 6), start=1):
            marker_px = to_px(marker[:3])
            _draw_operator_velocity_vector(
                pygame,
                screen,
                marker_px,
                marker[3:6],
                x_axis=x_axis,
                y_axis=y_axis,
                x_display_sign=x_display_sign,
                y_display_sign=y_display_sign,
                color=(86, 202, 245),
                length_px=30.0,
            )
            pygame.draw.circle(screen, OPERATOR_BURN_MARKER_COLOR, marker_px, 5)
            pygame.draw.circle(screen, (12, 16, 22), marker_px, 5, width=1)
            _text(screen, font, str(marker_idx), (marker_px[0] + 7, marker_px[1] - 14), OPERATOR_BURN_MARKER_COLOR)
    if trajectory_probe is not None:
        _draw_operator_probe_dot(
            pygame,
            screen,
            to_px(rel),
            label=_operator_probe_time_label(trajectory_probe.time_s),
            font=font,
        )
    pygame.draw.line(screen, (90, 104, 124), (plot.left, center[1]), (plot.right, center[1]), width=1)
    pygame.draw.line(screen, (90, 104, 124), (center[0], plot.top), (center[0], plot.bottom), width=1)
    pygame.draw.circle(screen, (96, 174, 224), target_px, 5)
    chaser_px = to_px(initial_rel)
    pygame.draw.circle(screen, (245, 205, 92), chaser_px, 6)
    pygame.draw.circle(screen, (10, 14, 20), chaser_px, 6, width=1)
    _draw_operator_velocity_vector(
        pygame,
        screen,
        chaser_px,
        initial_rel_vel,
        x_axis=x_axis,
        y_axis=y_axis,
        x_display_sign=x_display_sign,
        y_display_sign=y_display_sign,
    )
    screen.set_clip(previous_clip)

    _draw_operator_signed_axis_labels(
        screen,
        plot,
        x_axis=x_axis,
        y_axis=y_axis,
        x_display_sign=x_display_sign,
        y_display_sign=y_display_sign,
        font=font,
    )
    _draw_operator_initial_state_readout(
        screen,
        rect,
        rel_state_ric_km_s=readout_state,
        x_axis=x_axis,
        y_axis=y_axis,
        font=font,
        color=(86, 202, 245) if trajectory_probe is not None else (162, 178, 198),
    )


def _draw_operator_initial_state_readout(
    screen: Any,
    rect: Any,
    *,
    rel_state_ric_km_s: np.ndarray,
    x_axis: int,
    y_axis: int,
    font: Any,
    color: tuple[int, int, int] = (162, 178, 198),
) -> None:
    state = np.asarray(rel_state_ric_km_s, dtype=float).reshape(-1)
    if state.size < 6:
        return
    rel = state[:3]
    rel_vel = state[3:6]
    show_velocity_readout = int(x_axis) == 2 and int(y_axis) == 0
    readout_labels = ("dR", "dI", "dC") if show_velocity_readout else ("R", "I", "C")
    readout_values = rel_vel * 1000.0 if show_velocity_readout else rel
    readout_unit = "m/s" if show_velocity_readout else "km"
    _text(
        screen,
        font,
        (
            f"{readout_labels[0]} {readout_values[0]:.2f} {readout_unit}  "
            f"{readout_labels[1]} {readout_values[1]:.2f} {readout_unit}  "
            f"{readout_labels[2]} {readout_values[2]:.2f} {readout_unit}"
        ),
        (rect.x + 12, rect.bottom - 22),
        color,
    )


def _operator_axis_symbol(axis: int) -> str:
    labels = ("R", "I", "C")
    idx = int(axis)
    return labels[idx] if 0 <= idx < len(labels) else ""


def _operator_signed_axis_label(axis: int, sign: int) -> str:
    symbol = _operator_axis_symbol(axis)
    if not symbol:
        return ""
    prefix = "+" if int(sign) >= 0 else "-"
    return f"{prefix}{symbol}"


def _draw_operator_signed_axis_labels(
    screen: Any,
    plot: Any,
    *,
    x_axis: int,
    y_axis: int,
    x_display_sign: float,
    y_display_sign: float,
    font: Any,
) -> None:
    color = (162, 178, 198)
    x_plus = _operator_signed_axis_label(x_axis, 1)
    x_minus = _operator_signed_axis_label(x_axis, -1)
    y_plus = _operator_signed_axis_label(y_axis, 1)
    y_minus = _operator_signed_axis_label(y_axis, -1)

    x_plus_right = float(x_display_sign) >= 0.0
    x_left_label = x_minus if x_plus_right else x_plus
    x_right_label = x_plus if x_plus_right else x_minus
    _text(screen, font, x_left_label, (plot.left + 8, plot.centery + 6), color)
    _text(screen, font, x_right_label, (plot.right - _text_width(font, x_right_label) - 8, plot.centery + 6), color)

    y_plus_top = float(y_display_sign) >= 0.0
    y_top_label = y_plus if y_plus_top else y_minus
    y_bottom_label = y_minus if y_plus_top else y_plus
    _text(screen, font, y_top_label, (plot.centerx + 8, plot.top + 4), color)
    _text(screen, font, y_bottom_label, (plot.centerx + 8, plot.bottom - 24), color)


def _launcher_mode_palette(mode: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if _normalize_game_mode(mode) == "operator":
        return (62, 48, 28), (238, 184, 92)
    return (28, 48, 66), (96, 174, 224)


def _draw_clear_progress_button(pygame: Any, screen: Any, *, mode: str = "pilot", font: Any) -> None:
    rect = pygame.Rect(*CLEAR_PROGRESS_RECT)
    fill, stroke = _launcher_mode_palette(mode)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    _text(screen, font, "Clear Progress", (rect.x + 15, rect.y + 8), (230, 238, 245))


def _draw_record_video_button(pygame: Any, screen: Any, *, enabled: bool, mode: str = "pilot", font: Any) -> None:
    rect = pygame.Rect(*RECORD_VIDEO_RECT)
    fill, stroke = _launcher_mode_palette(mode)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    label = "Video: ON" if enabled else "Video: OFF"
    _text(screen, font, label, (rect.x + 18, rect.y + 8), (246, 238, 242))


def _draw_music_button(pygame: Any, screen: Any, *, enabled: bool, mode: str = "pilot", font: Any) -> None:
    rect = pygame.Rect(*MUSIC_RECT)
    fill, stroke = _launcher_mode_palette(mode)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    label = "Music: ON" if enabled else "Music: OFF"
    _text(screen, font, label, (rect.x + 18, rect.y + 8), (238, 244, 250))


def _draw_mode_toggle(pygame: Any, screen: Any, *, mode: str, font: Any) -> None:
    width, height = screen.get_size()
    rect = pygame.Rect(*_mode_toggle_rect(width, height))
    mode_key = _normalize_game_mode(mode)
    is_operator = mode_key == "operator"
    fill = (62, 48, 28) if is_operator else (28, 52, 66)
    stroke = (238, 184, 92) if is_operator else (96, 174, 224)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    label = "Operator Mode" if is_operator else "Pilot Mode"
    _text_centered(screen, font, label, rect.center, (246, 240, 226))


def _draw_settings_button(pygame: Any, screen: Any, *, mode: str) -> None:
    width, height = screen.get_size()
    rect = pygame.Rect(*_settings_button_rect(width, height))
    is_operator = _normalize_game_mode(mode) == "operator"
    fill = (62, 48, 28) if is_operator else (28, 52, 66)
    stroke = (238, 184, 92) if is_operator else (96, 174, 224)
    icon = (246, 240, 226)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    center = rect.center
    for dx, dy in ((0, -9), (6, -6), (9, 0), (6, 6), (0, 9), (-6, 6), (-9, 0), (-6, -6)):
        pygame.draw.line(screen, icon, center, (center[0] + dx, center[1] + dy), width=1)
    pygame.draw.circle(screen, fill, center, 7)
    pygame.draw.circle(screen, icon, center, 7, width=2)
    pygame.draw.circle(screen, icon, center, 2)


def _draw_difficulty_picker(
    pygame: Any,
    screen: Any,
    *,
    selected_difficulty: str,
    selected_mode: str = "pilot",
    font: Any,
) -> None:
    operator_mode = _normalize_game_mode(selected_mode) == "operator"
    label = "Error" if operator_mode else "Assists"
    label_right = 574 + _text_width(font, "Error")
    label_x = 574 if operator_mode else label_right - _text_width(font, label)
    _text(screen, font, label, (label_x, 92), (172, 186, 206))
    for idx, difficulty in enumerate(DIFFICULTY_OPTIONS):
        rect = pygame.Rect(642 + idx * 86, 86, 76, 26)
        is_selected = difficulty == selected_difficulty
        if operator_mode:
            fill = (62, 48, 28) if is_selected else (18, 24, 32)
            stroke = (238, 184, 92) if is_selected else (112, 86, 46)
        else:
            fill = (36, 72, 52) if is_selected else (18, 24, 32)
            stroke = (108, 232, 142) if is_selected else (70, 82, 100)
        pygame.draw.rect(screen, fill, rect, border_radius=6)
        pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
        _text(screen, font, difficulty.title(), (rect.x + 9, rect.y + 6), (230, 238, 245))


def _draw_preview(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    option: GameScenarioOption,
    scroll_px: int = 0,
    mode: str = "pilot",
    font: Any,
    small_font: Any,
) -> None:
    pygame.draw.rect(screen, (18, 24, 32), rect, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), rect, width=1, border_radius=8)
    content_rect = rect.inflate(-PREVIEW_PADDING * 2, -PREVIEW_PADDING * 2)
    content_height = _preview_content_height(option, font=font, small_font=small_font, width_px=content_rect.width)
    max_scroll = max(int(content_height) - int(content_rect.height), 0)
    scroll = int(max(0, min(int(scroll_px), max_scroll)))
    previous_clip = screen.get_clip()
    screen.set_clip(content_rect.clip(rect))
    y = content_rect.y - scroll
    max_y = content_rect.y + max(content_height, content_rect.height)
    _text(screen, font, _fit_text_px(option.title, font, content_rect.width), (content_rect.x, y), (238, 244, 250))
    y += 34
    for line in _wrapped_budget_lines(option, small_font, content_rect.width):
        _text(screen, small_font, line, (content_rect.x, y), (162, 178, 198))
        y += PREVIEW_LINE_HEIGHT
    y += PREVIEW_SECTION_GAP
    high = _format_high_score(option.high_score)
    if _show_progress_text(option) and high != "--":
        _text(screen, small_font, f"High Score: {high}", (content_rect.x, y), (245, 205, 92))
        y += PREVIEW_LINE_HEIGHT + PREVIEW_SECTION_GAP
    y = _draw_section(screen, small_font, "Objective", option.learning_goal, content_rect.x, y, content_rect.width, max_y)
    y = _draw_section(
        screen,
        small_font,
        "Brief",
        option.player_brief or option.description,
        content_rect.x,
        y + PREVIEW_SECTION_GAP,
        content_rect.width,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Pass Criteria",
        option.pass_criteria,
        content_rect.x,
        y + PREVIEW_SECTION_GAP,
        content_rect.width,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Instructor Notes",
        option.instructor_notes,
        content_rect.x,
        y + PREVIEW_SECTION_GAP,
        content_rect.width,
        max_y,
    )
    screen.set_clip(previous_clip)
    _draw_preview_scrollbar(pygame, screen, rect, content_height=content_height, scroll_px=scroll, mode=mode)


def _draw_preview_scrollbar(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    content_height: int,
    scroll_px: int,
    mode: str = "pilot",
) -> None:
    viewport_height = max(int(rect.height) - PREVIEW_PADDING * 2, 1)
    if int(content_height) <= viewport_height:
        return
    track = pygame.Rect(rect.right - 12, rect.y + PREVIEW_PADDING, 4, viewport_height)
    pygame.draw.rect(screen, (38, 48, 62), track, border_radius=2)
    frac = min(float(viewport_height) / max(float(content_height), 1.0), 1.0)
    thumb_h = max(int(track.height * frac), 28)
    max_scroll = max(int(content_height) - viewport_height, 1)
    travel = max(track.height - thumb_h, 0)
    thumb_y = track.y + int(round(travel * min(max(float(scroll_px) / float(max_scroll), 0.0), 1.0)))
    thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
    thumb_color = (238, 184, 92) if _normalize_game_mode(mode) == "operator" else (96, 174, 224)
    pygame.draw.rect(screen, thumb_color, thumb, border_radius=2)


def _preview_content_height(option: GameScenarioOption, *, font: Any, small_font: Any, width_px: int) -> int:
    y = 0
    y += 34
    y += len(_wrapped_budget_lines(option, small_font, width_px)) * PREVIEW_LINE_HEIGHT
    y += PREVIEW_SECTION_GAP
    if _show_progress_text(option) and option.high_score > 0:
        y += PREVIEW_LINE_HEIGHT + PREVIEW_SECTION_GAP
    y = _section_height(option.learning_goal, small_font, y, width_px)
    y = _section_height(option.player_brief or option.description, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.pass_criteria, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.instructor_notes, small_font, y + PREVIEW_SECTION_GAP, width_px)
    return max(y, _text_height(font))


def _section_height(body: str, font: Any, y: int, width_px: int) -> int:
    y += PREVIEW_SECTION_TITLE_GAP
    y += len(_wrap_text_px(body, font, width_px)) * PREVIEW_LINE_HEIGHT
    return y


def _show_progress_text(option: GameScenarioOption) -> bool:
    return str(option.scenario_id) != "rpo_00_tutorial"


def _bullets_height(items: tuple[str, ...], font: Any, y: int, width_px: int) -> int:
    if not items:
        return y
    y += PREVIEW_SECTION_TITLE_GAP
    bullet_width = max(width_px - _text_width(font, "- "), 1)
    for item in items:
        y += len(_wrap_text_px(item, font, bullet_width)) * PREVIEW_LINE_HEIGHT
    return y


def _text(screen: Any, font: Any, text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
    if not text:
        return
    surf = font.render(str(text), True, color)
    screen.blit(surf, pos)


def _text_centered(screen: Any, font: Any, text: str, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    if not text:
        return
    surf = font.render(str(text), True, color)
    screen.blit(surf, (int(center[0] - surf.get_width() // 2), int(center[1] - surf.get_height() // 2)))


def _draw_section(screen: Any, font: Any, title: str, body: str, x: int, y: int, width_px: int, max_y: int) -> int:
    if y + PREVIEW_SECTION_TITLE_GAP > max_y:
        return max_y
    _text(screen, font, title, (x, y), (238, 244, 250))
    y += PREVIEW_SECTION_TITLE_GAP
    lines = _wrap_text_px(body, font, width_px)
    for line in _lines_that_fit(lines, font, width_px, y, max_y):
        _text(screen, font, line, (x, y), (182, 194, 210))
        y += PREVIEW_LINE_HEIGHT
    return y


def _draw_bullets(
    screen: Any,
    font: Any,
    title: str,
    items: tuple[str, ...],
    x: int,
    y: int,
    width_px: int,
    max_y: int,
) -> int:
    if not items:
        return y
    if y + PREVIEW_SECTION_TITLE_GAP > max_y:
        return max_y
    _text(screen, font, title, (x, y), (238, 244, 250))
    y += PREVIEW_SECTION_TITLE_GAP
    for item in items:
        bullet_width = max(width_px - _text_width(font, "- "), 1)
        wrapped = _wrap_text_px(item, font, bullet_width)
        for idx, line in enumerate(wrapped):
            if y + PREVIEW_LINE_HEIGHT > max_y:
                ellipsis_y = y - PREVIEW_LINE_HEIGHT
                if idx > 0 and ellipsis_y >= 0:
                    ellipsis = _fit_text_px("...", font, width_px)
                    _text(screen, font, ellipsis, (x, y - PREVIEW_LINE_HEIGHT), (182, 194, 210))
                return max_y
            prefix = "- " if idx == 0 else "  "
            _text(screen, font, prefix + line, (x, y), (182, 194, 210))
            y += PREVIEW_LINE_HEIGHT
    return y


def _budget_line(option: GameScenarioOption) -> str:
    parts = []
    if option.time_budget_s is not None:
        parts.append(f"Time: {option.time_budget_s:.0f}s")
    if option.delta_v_budget_m_s is not None:
        parts.append(f"Chaser dV: {format_speed_m_s(option.delta_v_budget_m_s)}")
    if option.goal_speed_km_s is not None:
        parts.append(f"Speed Gate: {format_speed_km_s(option.goal_speed_km_s)}")
    if option.target_delta_v_budget_m_s is not None:
        parts.append(f"Target dV: {format_speed_m_s(option.target_delta_v_budget_m_s)}")
    return "   ".join(parts)


def _wrapped_budget_lines(option: GameScenarioOption, font: Any, width_px: int) -> list[str]:
    return _wrap_text_px(_budget_line(option), font, width_px)


def _completed_difficulties_from_game(game: dict[str, Any]) -> tuple[str, ...]:
    progress = dict(game.get("progress", {}) or {})
    completed = progress.get("completed_difficulties", ())
    if isinstance(completed, str):
        completed = (completed,)
    values = {_normalize_difficulty(item) for item in completed}
    return tuple(item for item in DIFFICULTY_OPTIONS if item in values)


def _high_score_from_game(game: dict[str, Any]) -> int:
    progress = dict(game.get("progress", {}) or {})
    return max(int(progress.get("high_score", 0) or 0), 0)


def _normalize_difficulty(value: Any) -> str:
    key = str(value or "easy").strip().lower()
    if key == "normal":
        return "medium"
    if key == "expert":
        return "extreme"
    if key in DIFFICULTY_OPTIONS:
        return key
    return "easy"


def _normalize_game_mode(value: Any) -> str:
    key = str(value or "pilot").strip().lower()
    if key in {"operator", "op", "script", "scripted"}:
        return "operator"
    return "pilot"


def _toggle_game_mode(value: Any) -> str:
    return "operator" if _normalize_game_mode(value) == "pilot" else "pilot"


def _progress_stars(completed_difficulties: tuple[str, ...]) -> str:
    highest = -1
    for difficulty in completed_difficulties:
        if difficulty in DIFFICULTY_OPTIONS:
            highest = max(highest, DIFFICULTY_OPTIONS.index(difficulty))
    earned = highest + 1
    return "★" * earned + "☆" * (len(DIFFICULTY_OPTIONS) - earned)


def _format_high_score(score: int) -> str:
    value = int(max(score, 0))
    return f"{value:,}" if value > 0 else "--"


def _wrap_text(value: str, max_chars: int) -> list[str]:
    words = str(value or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _wrap_text_px(value: str, font: Any, width_px: int) -> list[str]:
    words = str(value or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if _text_width(font, candidate) <= width_px:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = _fit_text_px(word, font, width_px) if _text_width(font, word) > width_px else word
    if current:
        lines.append(current)
    return lines or [""]


def _lines_that_fit(lines: list[str], font: Any, width_px: int, y: int, max_y: int) -> list[str]:
    if y >= max_y:
        return []
    available = max((max_y - y) // PREVIEW_LINE_HEIGHT, 0)
    if available <= 0:
        return []
    if len(lines) <= available:
        return lines
    kept = lines[:available]
    kept[-1] = _fit_text_px("...", font, width_px)
    return kept


def _fit_text_px(value: str, font: Any, width_px: int) -> str:
    text = " ".join(str(value or "").split())
    if _text_width(font, text) <= width_px:
        return text
    ellipsis = "..."
    if _text_width(font, ellipsis) > width_px:
        return ""
    lo = 0
    hi = len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if _text_width(font, candidate) <= width_px:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ellipsis


def _text_width(font: Any, text: str) -> int:
    if hasattr(font, "size"):
        return int(font.size(str(text))[0])
    surf = font.render(str(text), True, (255, 255, 255))
    if hasattr(surf, "get_width"):
        return int(surf.get_width())
    return len(str(text)) * 8


def _text_height(font: Any) -> int:
    if hasattr(font, "get_height"):
        return int(font.get_height())
    if hasattr(font, "size"):
        return int(font.size("Hg")[1])
    return PREVIEW_LINE_HEIGHT


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _truncate(value: str, max_chars: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 3, 0)].rstrip() + "..."
