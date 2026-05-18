from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sim.game.training import RPOTrainingConfig


def game_debrief_path(
    *,
    scenario_id: str,
    difficulty: str,
    attempt_index: int,
    output_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    root = Path(output_dir) if output_dir is not None else Path("outputs") / "game_debriefs"
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    scenario = _slug(scenario_id or "game")
    diff = _slug(difficulty or "easy")
    return root / f"{scenario}_{diff}_{stamp}_attempt{max(int(attempt_index), 1):02d}.json"


def write_game_debrief(
    path: str | Path,
    *,
    config: RPOTrainingConfig,
    score: Any,
    difficulty: str,
    objective_checklist: tuple[str, ...] = (),
    arcade_score: int = 0,
    arcade_seed: int | None = None,
    arcade_round_index: int | None = None,
    recording_path: str | Path | None = None,
    replay_history: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = game_debrief_payload(
        config=config,
        score=score,
        difficulty=difficulty,
        objective_checklist=objective_checklist,
        arcade_score=arcade_score,
        arcade_seed=arcade_seed,
        arcade_round_index=arcade_round_index,
        recording_path=recording_path,
        replay_history=replay_history,
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def game_debrief_payload(
    *,
    config: RPOTrainingConfig,
    score: Any,
    difficulty: str,
    objective_checklist: tuple[str, ...] = (),
    arcade_score: int = 0,
    arcade_seed: int | None = None,
    arcade_round_index: int | None = None,
    recording_path: str | Path | None = None,
    replay_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scenario_id": str(config.scenario_id or getattr(score, "scenario_id", "") or ""),
        "learning_goal": str(config.learning_goal or getattr(score, "learning_goal", "") or ""),
        "difficulty": str(difficulty or "easy"),
        "level_passed": bool(getattr(score, "level_passed", False)),
        "level_failed": bool(getattr(score, "level_failed", False)),
        "score": {
            "arcade_score": int(max(arcade_score, 0)),
            "arcade_seed": None if arcade_seed is None else int(arcade_seed),
            "arcade_round_index": None if arcade_round_index is None else int(arcade_round_index),
        },
        "metrics": {
            "samples": int(getattr(score, "samples", 0)),
            "elapsed_s": _float_or_none(getattr(score, "elapsed_s", None)),
            "achieved_time_s": _float_or_none(getattr(score, "achieved_time_s", None)),
            "closest_approach_km": _float_or_none(getattr(score, "closest_approach_km", None)),
            "final_range_km": _float_or_none(getattr(score, "final_range_km", None)),
            "final_goal_error_km": _float_or_none(getattr(score, "final_goal_error_km", None)),
            "final_relative_speed_km_s": _float_or_none(getattr(score, "final_relative_speed_km_s", None)),
            "time_inside_keepout_s": _float_or_none(getattr(score, "time_inside_keepout_s", None)),
            "approximate_delta_v_m_s": _float_or_none(getattr(score, "approximate_delta_v_m_s", None)),
            "target_delta_v_m_s": _float_or_none(getattr(score, "target_delta_v_m_s", None)),
            "min_goal_error_km": _float_or_none(getattr(score, "min_goal_error_km", None)),
        },
        "violations": {
            "keepout": bool(getattr(score, "keepout_violation", False)),
            "hard_speed_limit": bool(getattr(score, "hard_speed_limit_violation", False)),
            "forbidden_region": bool(getattr(score, "forbidden_region_violation", False)),
            "forbidden_region_names": list(getattr(score, "forbidden_region_names", ()) or ()),
            "approach_gate": bool(getattr(score, "approach_gate_violation", False)),
            "approach_gate_names": list(getattr(score, "approach_gate_names", ()) or ()),
        },
        "objectives": {
            "checklist": list(objective_checklist),
            "pass_fail_reasons": list(getattr(score, "pass_fail_reasons", ()) or ()),
            "hints": list(getattr(score, "hints", ()) or ()),
            "burn_axes_satisfied": list(getattr(score, "burn_axes_satisfied", ()) or ()),
            "speed_multiplier_changed": bool(getattr(score, "speed_multiplier_changed", False)),
            "approach_gates_satisfied": int(getattr(score, "approach_gates_satisfied", 0)),
            "approach_gates_total": int(getattr(score, "approach_gates_total", 0)),
            "inspection_gates_satisfied": int(getattr(score, "inspection_gates_satisfied", 0)),
            "inspection_gates_total": int(getattr(score, "inspection_gates_total", 0)),
            "inspection_gate_names": list(getattr(score, "inspection_gate_names", ()) or ()),
        },
        "artifacts": {
            "recording_path": None if recording_path is None else str(recording_path),
        },
        "replay": replay_history or {},
    }


def tracker_replay_history(tracker: Any) -> dict[str, Any]:
    return {
        "time_s": _array_list(getattr(tracker, "t_s", [])),
        "relative_ric": _array_list(getattr(tracker, "rel_ric_hist", [])),
        "chaser_thrust_ric_km_s2": _array_list(getattr(tracker, "thrust_ric_hist", [])),
        "target_thrust_eci_km_s2": _array_list(getattr(tracker, "target_thrust_hist", [])),
    }


def _array_list(value: Any) -> list[Any]:
    arr = np.array(value, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 0:
        return [float(arr)]
    return arr.tolist()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    out = []
    last_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_sep = False
        elif not last_sep:
            out.append("_")
            last_sep = True
    return "".join(out).strip("_") or "game"
