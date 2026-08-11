# ruff: noqa: F401,F403,F405,I001
from .launcher_common import *
from .launcher_models import *

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
        return (14, option.scenario_id)
    if option.scenario_id == "rpo_sandbox":
        return (15, option.scenario_id)
    return (option.level_number, option.scenario_id)


def _title_from_scenario_id(scenario_id: str, *, level_number: int) -> str:
    parts = str(scenario_id).split("_")
    if len(parts) >= 3 and parts[0] == "rpo" and parts[1].isdigit():
        name = " ".join(parts[2:]).title()
        return f"Level {level_number} - {name}"
    return str(scenario_id).replace("_", " ").title()

__all__ = [name for name in globals() if not name.startswith("__")]
