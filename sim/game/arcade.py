from __future__ import annotations

from typing import Any

import numpy as np

from sim.api import SimulationConfig
from sim.game.defensive_target import DefensiveTargetIntentProvider
from sim.game.training import RPOTrainingConfig

DIFFICULTY_SCORE_MULTIPLIERS: dict[str, int] = {"easy": 1, "medium": 2, "hard": 3, "extreme": 4}


def _game_arcade_config(config: SimulationConfig) -> dict[str, Any]:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    return dict(game_cfg.get("arcade", {}) or {})


def _game_arcade_enabled(config: SimulationConfig) -> bool:
    return bool(_game_arcade_config(config).get("enabled", False))


def _game_arcade_initial_time_s(config: SimulationConfig, training_cfg: RPOTrainingConfig) -> float:
    arcade = _game_arcade_config(config)
    fallback = training_cfg.max_time_s if training_cfg.max_time_s is not None else config.scenario.simulator.duration_s
    return float(max(float(arcade.get("initial_time_s", fallback) or fallback), 0.0))


def _game_arcade_round_bonus_time_s(config: SimulationConfig) -> float:
    arcade = _game_arcade_config(config)
    return float(max(float(arcade.get("round_bonus_time_s", 0.0) or 0.0), 0.0))


def _game_arcade_delta_v_bonus_time_per_m_s(config: SimulationConfig) -> float:
    arcade = _game_arcade_config(config)
    return float(max(float(arcade.get("delta_v_bonus_time_per_m_s", 0.0) or 0.0), 0.0))


def _new_arcade_seed() -> int:
    return int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def _arcade_round_rng(arcade_seed: int, round_index: int) -> np.random.Generator:
    seed_seq = np.random.SeedSequence([int(arcade_seed), int(max(round_index, 1))])
    return np.random.default_rng(seed_seq)


def _game_defensive_target_provider(config: SimulationConfig) -> DefensiveTargetIntentProvider | None:
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    raw = dict(game_cfg.get("defensive_target", {}) or {})
    if not bool(raw.get("enabled", False)):
        return None
    return DefensiveTargetIntentProvider(
        chaser_object_id=str(raw.get("chaser_object_id", "chaser") or "chaser"),
        trigger_range_km=float(raw.get("trigger_range_km", 1.2) or 1.2),
        trigger_closing_speed_km_s=float(raw.get("trigger_closing_speed_km_s", 0.00025) or 0.00025),
        keepout_radius_km=float(raw.get("keepout_radius_km", 0.25) or 0.25),
        max_accel_km_s2=float(raw.get("max_accel_km_s2", 7.5e-6) or 7.5e-6),
        max_delta_v_m_s=_optional_float(raw.get("max_delta_v_m_s")),
        cross_track_bias=float(raw.get("cross_track_bias", 0.65) or 0.65),
        pulse_period_s=float(raw.get("pulse_period_s", 120.0) or 120.0),
    )


def _game_random_direction_defensive_target_provider(
    config: SimulationConfig,
    *,
    rng: np.random.Generator,
) -> DefensiveTargetIntentProvider | None:
    provider = _game_defensive_target_provider(config)
    if provider is None:
        return None
    direction = rng.normal(size=3)
    nrm = float(np.linalg.norm(direction))
    if nrm <= 0.0 or not np.isfinite(nrm):
        direction = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        direction = direction / nrm
    provider.fixed_direction_ric = tuple(float(x) for x in direction)
    return provider


def _arcade_mission_metrics(
    metrics: tuple[str, ...],
    *,
    enabled: bool,
    round_index: int,
    total_score: int,
) -> tuple[str, ...]:
    if not bool(enabled):
        return metrics
    return (f"OK Round {int(max(round_index, 1))}", f"OK Score {int(max(total_score, 0)):,}", *tuple(metrics))


def _arcade_round_briefing_lines(
    *,
    cleared_round_index: int,
    next_round_index: int,
    round_score: int,
    total_score: int,
    time_used_s: float,
    bonus_time_s: float,
    next_time_budget_s: float,
) -> tuple[str, ...]:
    return (
        f"Round {int(max(cleared_round_index, 1))} Cleared",
        f"Round score: {int(max(round_score, 0)):,}. Total score: {int(max(total_score, 0)):,}.",
        f"Time used: {float(max(time_used_s, 0.0)):.0f} s. Bonus awarded: {float(max(bonus_time_s, 0.0)):.0f} s.",
        f"Round {int(max(next_round_index, 1))} starts with {float(max(next_time_budget_s, 0.0)):.0f} s remaining.",
        "Fuel resets. The target picks a new fixed evasion direction.",
    )


def _score_time_used_s(score: Any) -> float:
    achieved_time = getattr(score, "achieved_time_s", None)
    if achieved_time is not None:
        return float(max(float(achieved_time), 0.0))
    return float(max(float(getattr(score, "elapsed_s", 0.0)), 0.0))


def _arcade_round_weighted_score(
    config: RPOTrainingConfig,
    score: Any,
    *,
    difficulty: str,
    round_index: int,
) -> int:
    return int(max(round_index, 1)) * _arcade_score(config, score, difficulty=difficulty)


def _arcade_round_time_bonus_s(config: SimulationConfig, training_cfg: RPOTrainingConfig, score: Any) -> float:
    baseline = _game_arcade_round_bonus_time_s(config)
    if training_cfg.max_delta_v_m_s is None:
        return baseline
    used_delta_v_m_s = float(getattr(score, "approximate_delta_v_m_s", 0.0))
    remaining_delta_v_m_s = max(float(training_cfg.max_delta_v_m_s) - used_delta_v_m_s, 0.0)
    return baseline + remaining_delta_v_m_s * _game_arcade_delta_v_bonus_time_per_m_s(config)


def _arcade_score(config: RPOTrainingConfig, score: Any, *, difficulty: str) -> int:
    if not bool(getattr(score, "level_passed", False)):
        return 0
    seconds_remaining = 0.0
    if config.max_time_s is not None:
        achieved_time = getattr(score, "achieved_time_s", None)
        elapsed = float(getattr(score, "elapsed_s", 0.0))
        time_used = elapsed if achieved_time is None else float(achieved_time)
        seconds_remaining = max(float(config.max_time_s) - time_used, 0.0)
    delta_v_remaining_mm_s = 0.0
    if config.max_delta_v_m_s is not None:
        used_delta_v_m_s = float(getattr(score, "approximate_delta_v_m_s", 0.0))
        delta_v_remaining_mm_s = max(float(config.max_delta_v_m_s) - used_delta_v_m_s, 0.0) * 1000.0
    if config.max_target_delta_v_m_s is not None:
        used_target_delta_v_m_s = float(getattr(score, "target_delta_v_m_s", 0.0))
        delta_v_remaining_mm_s += max(float(config.max_target_delta_v_m_s) - used_target_delta_v_m_s, 0.0) * 1000.0
    multiplier = _difficulty_score_multiplier(difficulty)
    return int(round((seconds_remaining + delta_v_remaining_mm_s) * multiplier))


def _difficulty_score_multiplier(difficulty: str) -> int:
    key = str(difficulty or "easy").strip().lower()
    if key == "normal":
        key = "medium"
    if key == "expert":
        key = "extreme"
    return DIFFICULTY_SCORE_MULTIPLIERS.get(key, 1)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
