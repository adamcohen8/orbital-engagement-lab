from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from sim.api import SimulationConfig
from sim.dynamics.orbit.elements import coes_mapping_to_rv_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.game.defensive_target import DefensiveTargetIntentProvider
from sim.game.formatting import format_distance_km
from sim.game.training import RPOTrainingConfig
from sim.utils.frames import ric_rect_state_to_eci

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


def _game_arcade_goal_range_step_km(config: SimulationConfig) -> float:
    arcade = _game_arcade_config(config)
    return float(max(float(arcade.get("goal_range_step_km", 0.0) or 0.0), 0.0))


def _game_arcade_min_goal_range_km(config: SimulationConfig) -> float:
    arcade = _game_arcade_config(config)
    return float(max(float(arcade.get("min_goal_range_km", 0.0) or 0.0), 0.0))


def _game_arcade_boss_config(config: SimulationConfig) -> dict[str, Any]:
    raw = _game_arcade_config(config).get("boss", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _game_arcade_boss_round_interval(config: SimulationConfig) -> int:
    arcade = _game_arcade_config(config)
    try:
        value = int(arcade.get("boss_round_interval", 0) or 0)
    except (TypeError, ValueError):
        value = 0
    return max(value, 0)


def _arcade_round_is_boss(config: SimulationConfig, round_index: int) -> bool:
    interval = _game_arcade_boss_round_interval(config)
    return bool(interval > 0 and int(round_index) > 0 and int(round_index) % interval == 0)


def _arcade_round_score_multiplier(config: SimulationConfig, round_index: int) -> float:
    if not _arcade_round_is_boss(config, round_index):
        return 1.0
    raw = _game_arcade_boss_config(config).get("score_multiplier", 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 1.0
    return float(max(value, 0.0))


def _arcade_round_music_track(config: SimulationConfig, round_index: int) -> str | None:
    if not _arcade_round_is_boss(config, round_index):
        return None
    track = str(_game_arcade_boss_config(config).get("music_track", "") or "").strip()
    return track or None


def _arcade_round_coast_prediction_model(config: SimulationConfig, round_index: int) -> str | None:
    if not _arcade_round_is_boss(config, round_index):
        return None
    model = str(_game_arcade_boss_config(config).get("coast_prediction_model", "") or "").strip()
    return model or None


def _arcade_round_training_config(
    config: SimulationConfig,
    training_cfg: RPOTrainingConfig,
    *,
    round_index: int,
    max_time_s: float | None,
) -> RPOTrainingConfig:
    kwargs: dict[str, Any] = {"max_time_s": max_time_s}
    goal_range = training_cfg.goal_range_km
    step = _game_arcade_goal_range_step_km(config)
    if goal_range is not None and step > 0.0:
        minimum = _game_arcade_min_goal_range_km(config)
        decrement = max(int(round_index) - 1, 0) * step
        kwargs["goal_range_km"] = max(float(goal_range) - decrement, minimum)
    return replace(training_cfg, **kwargs)


def _arcade_round_simulation_config(
    config: SimulationConfig,
    training_cfg: RPOTrainingConfig,
    *,
    round_index: int,
    rng: np.random.Generator | None,
) -> SimulationConfig:
    boss_round = _arcade_round_is_boss(config, round_index)
    randomize_initial_state = int(round_index) > 1 and _game_arcade_random_initial_state_enabled(config)
    if not boss_round and not randomize_initial_state:
        return config
    if rng is None:
        rng = np.random.default_rng()

    root = config.to_dict()
    if boss_round:
        _apply_arcade_boss_round_config(root, config, training_cfg, rng=rng)
    if not randomize_initial_state:
        return SimulationConfig.from_dict(root)
    attempt_config = SimulationConfig.from_dict(root)
    rel_state = _sample_arcade_random_initial_state(attempt_config, training_cfg, rng=rng)
    chaser_id = str(training_cfg.chaser_object_id or "chaser")
    target_id = str(training_cfg.target_object_id or "target")
    chaser = root.setdefault("objects", {}).setdefault(chaser_id, {})
    initial_state = chaser.setdefault("initial_state", {})
    rel_block = dict(initial_state.get("relative_to_target_ric", {}) or {})
    rel_block["frame"] = "rect"
    rel_block["state"] = [float(value) for value in rel_state]
    initial_state["relative_to_target_ric"] = rel_block
    initial_state["relative_to"] = target_id
    return SimulationConfig.from_dict(root)


def _new_arcade_seed() -> int:
    return int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def _arcade_round_rng(arcade_seed: int, round_index: int) -> np.random.Generator:
    seed_seq = np.random.SeedSequence([int(arcade_seed), int(max(round_index, 1))])
    return np.random.default_rng(seed_seq)


def _arcade_round_initial_state_rng(arcade_seed: int, round_index: int) -> np.random.Generator:
    seed_seq = np.random.SeedSequence([int(arcade_seed), int(max(round_index, 1)), 715_827_883])
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
    is_boss: bool = False,
) -> tuple[str, ...]:
    if not bool(enabled):
        return metrics
    prefix = [f"OK Round {int(max(round_index, 1))}", f"OK Score {int(max(total_score, 0)):,}"]
    if bool(is_boss):
        prefix.append("WARN Boss")
    return (*prefix, *tuple(metrics))


def _arcade_round_briefing_lines(
    *,
    cleared_round_index: int,
    next_round_index: int,
    round_score: int,
    total_score: int,
    time_used_s: float,
    bonus_time_s: float,
    next_time_budget_s: float,
    next_goal_range_km: float | None = None,
    next_is_boss: bool = False,
) -> tuple[str, ...]:
    lines = [
        f"Round {int(max(cleared_round_index, 1))} Cleared",
        f"Round score: {int(max(round_score, 0)):,}. Total score: {int(max(total_score, 0)):,}.",
        f"Time used: {float(max(time_used_s, 0.0)):.0f} s. Bonus awarded: {float(max(bonus_time_s, 0.0)):.0f} s.",
        f"Round {int(max(next_round_index, 1))} starts with {float(max(next_time_budget_s, 0.0)):.0f} s remaining.",
    ]
    if bool(next_is_boss):
        lines.append("Boss round: elliptical target orbit, randomized true anomaly, TH projection.")
    if next_goal_range_km is not None:
        lines.append(f"Next pursuit target: close within {format_distance_km(float(next_goal_range_km))}.")
    lines.append("Fuel resets. The target picks a new fixed evasion direction.")
    return tuple(lines)


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
    arcade_config: SimulationConfig | None = None,
) -> int:
    multiplier = 1.0 if arcade_config is None else _arcade_round_score_multiplier(arcade_config, round_index)
    return int(round(int(max(round_index, 1)) * _arcade_score(config, score, difficulty=difficulty) * multiplier))


def _arcade_round_time_bonus_s(
    config: SimulationConfig,
    training_cfg: RPOTrainingConfig,
    score: Any,
    *,
    round_index: int | None = None,
) -> float:
    baseline = _game_arcade_round_bonus_time_s(config)
    if round_index is not None and _arcade_round_is_boss(config, int(round_index)):
        boss_raw = _game_arcade_boss_config(config).get("bonus_time_s", 0.0)
        try:
            baseline += max(float(boss_raw), 0.0)
        except (TypeError, ValueError):
            pass
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


def _apply_arcade_boss_round_config(
    root: dict[str, Any],
    config: SimulationConfig,
    training_cfg: RPOTrainingConfig,
    *,
    rng: np.random.Generator,
) -> None:
    target_id = str(training_cfg.target_object_id or "target")
    target = root.setdefault("objects", {}).setdefault(target_id, {})
    initial_state = dict(target.get("initial_state", {}) or {})
    coes = _arcade_boss_target_coes(config, rng=rng)
    initial_state["coes"] = coes
    initial_state.pop("position_eci_km", None)
    initial_state.pop("velocity_eci_km_s", None)
    target["initial_state"] = initial_state
    target["reference_orbit"] = {**dict(target.get("reference_orbit", {}) or {}), "enabled": True}

    boss_model = str(_game_arcade_boss_config(config).get("coast_prediction_model", "") or "").strip()
    if boss_model:
        game_cfg = dict(root.setdefault("metadata", {}).setdefault("game", {}) or {})
        game_cfg["coast_prediction_model"] = boss_model
        root["metadata"]["game"] = game_cfg


def _arcade_boss_target_coes(config: SimulationConfig, *, rng: np.random.Generator) -> dict[str, float]:
    raw = _game_arcade_boss_config(config)
    base = raw.get("target_coes", {}) or {}
    coes = dict(base) if isinstance(base, dict) else {}
    if not coes:
        target_id = str(
            dict(dict(config.scenario.metadata.get("game", {}) or {}).get("training", {}) or {}).get(
                "target_object_id",
                "target",
            )
            or "target"
        )
        target_section = config.scenario.objects.get(target_id)
        if target_section is not None:
            initial = dict(target_section.initial_state or {})
            coes = dict(initial.get("coes", {}) or {})
    if not coes:
        coes = {
            "a_km": 9000.0,
            "ecc": 0.25,
            "inc_deg": 45.0,
            "raan_deg": 0.0,
            "argp_deg": 0.0,
            "true_anomaly_deg": 0.0,
        }
    anomaly_range = _range_pair(raw.get("true_anomaly_range_deg"), (0.0, 360.0))
    coes["true_anomaly_deg"] = float(rng.uniform(anomaly_range[0], anomaly_range[1]) % 360.0)
    return {str(key): float(value) for key, value in coes.items()}


def _game_arcade_random_initial_state_config(config: SimulationConfig) -> dict[str, Any]:
    raw = _game_arcade_config(config).get("random_initial_state", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _game_arcade_random_initial_state_enabled(config: SimulationConfig) -> bool:
    return bool(_game_arcade_random_initial_state_config(config).get("enabled", False))


def _sample_arcade_random_initial_state(
    config: SimulationConfig,
    training_cfg: RPOTrainingConfig,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = _game_arcade_random_initial_state_config(config)
    radial_range = _range_pair(raw.get("radial_range_km"), (-1.0, 1.0))
    in_track_range = _range_pair(raw.get("in_track_range_km"), (-10.0, 10.0))
    cross_track_range = _range_pair(raw.get("cross_track_range_km"), (-1.0, 1.0))
    cross_track_rate_range = _range_pair(raw.get("cross_track_rate_range_km_s"), (-0.001, 0.001))
    min_range_km = float(max(float(raw.get("min_range_km", 5.0) or 5.0), 0.0))

    target_id = str(training_cfg.target_object_id or "target")
    target_section = config.scenario.objects.get(target_id)
    if target_section is None:
        raise ValueError(f"Cannot randomize arcade initial state: unknown target '{target_id}'.")
    target_r_eci_km, target_v_eci_km_s = _object_initial_eci_state(dict(target_section.initial_state or {}))

    for _ in range(1000):
        rel_position = np.array(
            [
                rng.uniform(radial_range[0], radial_range[1]),
                rng.uniform(in_track_range[0], in_track_range[1]),
                rng.uniform(cross_track_range[0], cross_track_range[1]),
            ],
            dtype=float,
        )
        if float(np.linalg.norm(rel_position)) < min_range_km:
            continue
        cross_track_rate = float(rng.uniform(cross_track_rate_range[0], cross_track_rate_range[1]))
        in_track_rate = _energy_matched_in_track_rate_km_s(
            rel_position,
            cross_track_rate_km_s=cross_track_rate,
            target_r_eci_km=target_r_eci_km,
            target_v_eci_km_s=target_v_eci_km_s,
        )
        if np.isfinite(in_track_rate):
            return np.array(
                [rel_position[0], rel_position[1], rel_position[2], 0.0, in_track_rate, cross_track_rate],
                dtype=float,
            )
    raise RuntimeError("Unable to sample a valid arcade initial state after 1000 attempts.")


def _energy_matched_in_track_rate_km_s(
    rel_position_ric_km: np.ndarray,
    *,
    cross_track_rate_km_s: float,
    target_r_eci_km: np.ndarray,
    target_v_eci_km_s: np.ndarray,
) -> float:
    rel_position = np.array(rel_position_ric_km, dtype=float).reshape(3)
    target_r = np.array(target_r_eci_km, dtype=float).reshape(3)
    target_v = np.array(target_v_eci_km_s, dtype=float).reshape(3)
    base_rel = np.array([rel_position[0], rel_position[1], rel_position[2], 0.0, 0.0, cross_track_rate_km_s])
    unit_i_rel = np.array([rel_position[0], rel_position[1], rel_position[2], 0.0, 1.0, cross_track_rate_km_s])
    base_state = ric_rect_state_to_eci(base_rel, target_r, target_v)
    unit_i_state = ric_rect_state_to_eci(unit_i_rel, target_r, target_v)

    r_chaser = base_state[:3]
    v_base = base_state[3:]
    v_axis = unit_i_state[3:] - v_base
    target_energy = _specific_orbital_energy_km2_s2(target_r, target_v)

    qa = 0.5 * float(np.dot(v_axis, v_axis))
    qb = float(np.dot(v_base, v_axis))
    qc = (
        0.5 * float(np.dot(v_base, v_base))
        - EARTH_MU_KM3_S2 / max(float(np.linalg.norm(r_chaser)), 1e-9)
        - target_energy
    )
    if qa <= 0.0:
        return float("nan")
    discriminant = qb * qb - 4.0 * qa * qc
    if discriminant < 0.0:
        return float("nan")
    root = float(np.sqrt(max(discriminant, 0.0)))
    candidates = ((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa))
    return float(min(candidates, key=lambda value: abs(float(value))))


def _specific_orbital_energy_km2_s2(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> float:
    r_norm = max(float(np.linalg.norm(np.array(r_eci_km, dtype=float).reshape(3))), 1e-9)
    v = np.array(v_eci_km_s, dtype=float).reshape(3)
    return 0.5 * float(np.dot(v, v)) - EARTH_MU_KM3_S2 / r_norm


def _object_initial_eci_state(initial_state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "position_eci_km" in initial_state:
        pos = np.array(initial_state.get("position_eci_km", [7000.0, 0.0, 0.0]), dtype=float).reshape(3)
        if "velocity_eci_km_s" in initial_state:
            vel = np.array(initial_state["velocity_eci_km_s"], dtype=float).reshape(3)
        else:
            speed = float(np.sqrt(EARTH_MU_KM3_S2 / max(float(np.linalg.norm(pos)), EARTH_RADIUS_KM + 1.0)))
            vel = np.array([0.0, speed, 0.0], dtype=float)
        return pos, vel
    coes = initial_state.get("coes")
    if isinstance(coes, dict):
        return coes_mapping_to_rv_eci(coes)
    pos = np.array([7000.0, 0.0, 0.0], dtype=float)
    vel = np.array([0.0, float(np.sqrt(EARTH_MU_KM3_S2 / np.linalg.norm(pos))), 0.0], dtype=float)
    return pos, vel


def _range_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    try:
        pair = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        pair = default
    if len(pair) != 2:
        pair = default
    try:
        lower = float(pair[0])
        upper = float(pair[1])
    except (TypeError, ValueError):
        lower, upper = default
    if not np.isfinite(lower) or not np.isfinite(upper):
        lower, upper = default
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper
