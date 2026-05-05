from __future__ import annotations

from typing import Any

import numpy as np


def _finite_range_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n_rel = int(min(a.shape[0], b.shape[0]))
    if n_rel <= 0:
        return np.array([], dtype=float)
    dr = a[:n_rel, :3] - b[:n_rel, :3]
    rng_km = np.linalg.norm(dr, axis=1)
    return rng_km[np.isfinite(rng_km)]


def _candidate_truth_pairs(tb: dict[str, Any], primary_pair: tuple[str, str] | None = None) -> list[tuple[str, str]]:
    if primary_pair is not None and primary_pair[0] in tb and primary_pair[1] in tb:
        return [primary_pair]
    if "chaser" in tb and "target" in tb:
        return [("chaser", "target")]
    object_ids = sorted(str(key) for key in tb.keys())
    return [(a, b) for idx, a in enumerate(object_ids) for b in object_ids[idx + 1 :]]


def _primary_pair_from_payload(run_output: dict[str, Any]) -> tuple[str, str] | None:
    summary = dict(run_output.get("summary", {}) or {})
    raw = summary.get("primary_object_pair")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        a_id = str(raw[0])
        b_id = str(raw[1])
        if a_id and b_id and a_id != b_id:
            return a_id, b_id
    return None


def closest_approach_from_run_payload(run_output: dict[str, Any]) -> float:
    closest_approach_km = float("nan")
    try:
        tb = dict(run_output.get("truth_by_object", {}) or {})
        pair_mins: list[float] = []
        for a_id, b_id in _candidate_truth_pairs(tb, _primary_pair_from_payload(run_output)):
            a = np.array(tb.get(a_id, []), dtype=float)
            b = np.array(tb.get(b_id, []), dtype=float)
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or b.shape[0] == 0:
                continue
            finite = _finite_range_km(a, b)
            if finite.size > 0:
                pair_mins.append(float(np.min(finite)))
        if pair_mins:
            closest_approach_km = float(min(pair_mins))
    except (TypeError, ValueError, KeyError, IndexError):
        closest_approach_km = float("nan")
    return closest_approach_km


def relative_range_series_from_run_payload(run_output: dict[str, Any]) -> dict[str, np.ndarray] | None:
    try:
        tb = dict(run_output.get("truth_by_object", {}) or {})
        t_s = np.array(run_output.get("time_s", []), dtype=float).reshape(-1)
        pairs = _candidate_truth_pairs(tb, _primary_pair_from_payload(run_output))
        if t_s.ndim != 1 or t_s.size == 0 or not pairs:
            return None
        a_id, b_id = pairs[0]
        a = np.array(tb.get(a_id, []), dtype=float)
        b = np.array(tb.get(b_id, []), dtype=float)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or b.shape[0] == 0:
            return None
        n_rel = int(min(t_s.size, a.shape[0], b.shape[0]))
        dr = a[:n_rel, :3] - b[:n_rel, :3]
        return {
            "time_s": np.array(t_s[:n_rel], dtype=float),
            "range_km": np.array(np.linalg.norm(dr, axis=1), dtype=float),
        }
    except (TypeError, ValueError, KeyError, IndexError):
        return None


_closest_approach_from_run_payload = closest_approach_from_run_payload
_relative_range_series_from_run_payload = relative_range_series_from_run_payload
