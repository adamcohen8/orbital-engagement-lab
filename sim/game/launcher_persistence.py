# ruff: noqa: F401,F403,F405,I001
from .launcher_common import *
from .launcher_models import *
from .scenario_catalog import *

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

__all__ = [name for name in globals() if not name.startswith("__")]
