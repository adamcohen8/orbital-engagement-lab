from __future__ import annotations

import re
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


def relative_motion_summary_from_run_payload(run_output: dict[str, Any]) -> dict[str, Any]:
    try:
        tb = dict(run_output.get("truth_by_object", {}) or {})
        pairs = _candidate_truth_pairs(tb, _primary_pair_from_payload(run_output))
        if not pairs:
            return {}
        a_id, b_id = pairs[0]
        a = np.array(tb.get(a_id, []), dtype=float)
        b = np.array(tb.get(b_id, []), dtype=float)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or b.shape[0] == 0:
            return {"pair": [a_id, b_id], "samples": 0, "finite_fraction": 0.0}
        n_rel = int(min(a.shape[0], b.shape[0]))
        if n_rel <= 0 or a.shape[1] < 6 or b.shape[1] < 6:
            return {"pair": [a_id, b_id], "samples": n_rel, "finite_fraction": 0.0}
        rel_pos_km = a[:n_rel, :3] - b[:n_rel, :3]
        rel_vel_km_s = a[:n_rel, 3:6] - b[:n_rel, 3:6]
        finite_rows = np.all(np.isfinite(np.hstack((rel_pos_km, rel_vel_km_s))), axis=1)
        range_km = np.linalg.norm(rel_pos_km, axis=1)
        speed_km_s = np.linalg.norm(rel_vel_km_s, axis=1)
        finite_range = range_km[finite_rows & np.isfinite(range_km)]
        finite_speed = speed_km_s[finite_rows & np.isfinite(speed_km_s)]
        return {
            "pair": [a_id, b_id],
            "samples": n_rel,
            "finite_fraction": float(np.mean(finite_rows)),
            "initial_range_km": float(range_km[0]) if np.isfinite(range_km[0]) else float("nan"),
            "final_range_km": float(range_km[-1]) if np.isfinite(range_km[-1]) else float("nan"),
            "closest_approach_km": float(np.min(finite_range)) if finite_range.size else float("nan"),
            "initial_relative_speed_m_s": float(speed_km_s[0] * 1000.0)
            if np.isfinite(speed_km_s[0])
            else float("nan"),
            "final_relative_speed_m_s": float(speed_km_s[-1] * 1000.0)
            if np.isfinite(speed_km_s[-1])
            else float("nan"),
            "max_relative_speed_m_s": float(np.max(finite_speed) * 1000.0) if finite_speed.size else float("nan"),
        }
    except (TypeError, ValueError, KeyError, IndexError):
        return {}


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


def _deep_get(root: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = root
    for tok in str(path or "").split("."):
        if not tok:
            return default
        if "[" in tok and tok.endswith("]"):
            key, idx_txt = tok[:-1].split("[", 1)
            try:
                idx = int(idx_txt)
            except ValueError:
                return default
            if key:
                if not isinstance(cur, dict) or key not in cur:
                    return default
                cur = cur[key]
            if not isinstance(cur, list) or idx < 0 or idx >= len(cur):
                return default
            cur = cur[idx]
            continue
        if not isinstance(cur, dict) or tok not in cur:
            return default
        cur = cur[tok]
    return cur


def _metric_name_from_path(path: str) -> str:
    text = str(path or "").strip()
    text = text.removeprefix("summary.").removeprefix("payload.").removeprefix("derived.")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return text or "metric"


STUDY_METRIC_PRESETS: dict[str, list[dict[str, str]]] = {
    "final_state": [
        {"name": "duration_s", "path": "summary.duration_s"},
        {"name": "terminated_early", "path": "summary.terminated_early"},
        {"name": "termination_reason", "path": "summary.termination_reason"},
    ],
    "relative_motion": [
        {"name": "closest_approach_km", "path": "derived.closest_approach_km"},
    ],
    "reentry": [
        {"name": "reentry_peak_g_load", "path": "derived.reentry_peak_g_load"},
        {"name": "reentry_peak_heat_rate_w_m2", "path": "derived.reentry_peak_heat_rate_w_m2"},
        {"name": "reentry_final_heat_load_j_m2", "path": "derived.reentry_final_heat_load_j_m2"},
        {"name": "reentry_min_altitude_km", "path": "derived.reentry_min_altitude_km"},
    ],
    "rocket_ascent": [
        {"name": "rocket_insertion_achieved", "path": "summary.rocket_insertion_achieved"},
        {"name": "rocket_final_altitude_km", "path": "summary.rocket_metrics_summary.final_altitude_km"},
        {"name": "rocket_max_dynamic_pressure_pa", "path": "summary.rocket_metrics_summary.max_dynamic_pressure_pa"},
    ],
    "attitude_control": [
        {"name": "attitude_guardrail_events", "path": "derived.attitude_guardrail_events"},
    ],
    "ground_access": [
        {"name": "ground_access_total_windows", "path": "derived.ground_access_total_windows"},
        {"name": "ground_access_total_duration_s", "path": "derived.ground_access_total_duration_s"},
    ],
    "orbit_elements": [
        {"name": "final_altitude_km_min", "path": "derived.final_altitude_km_min"},
        {"name": "final_altitude_km_max", "path": "derived.final_altitude_km_max"},
    ],
}


def normalize_study_metric_specs(raw_metrics: list[Any] | tuple[Any, ...] | None) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for raw in list(raw_metrics or []):
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            preset = STUDY_METRIC_PRESETS.get(text)
            if preset is not None:
                specs.extend(dict(item) for item in preset)
            else:
                specs.append({"name": text, "path": text})
            continue
        if isinstance(raw, dict):
            preset_name = str(raw.get("preset", "") or raw.get("name", "") or "").strip()
            if "path" not in raw and preset_name in STUDY_METRIC_PRESETS:
                specs.extend(dict(item) for item in STUDY_METRIC_PRESETS[preset_name])
                continue
            path = str(raw.get("path", "") or raw.get("metric", "") or "").strip()
            if not path:
                continue
            name = str(raw.get("name", "") or "").strip() or _metric_name_from_path(path)
            specs.append({"name": name, "path": path})
    return specs


def _numeric_values(values: list[Any]) -> np.ndarray:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool):
            out.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            fv = float(value)
            if np.isfinite(fv):
                out.append(fv)
    return np.array(out, dtype=float)


def _reentry_summary_values(run_output: dict[str, Any], key: str) -> list[Any]:
    summary = dict(run_output.get("summary", {}) or {})
    by_object = dict(summary.get("reentry_summary_by_object", {}) or {})
    return [dict(obj_summary or {}).get(key) for obj_summary in by_object.values()]


def extract_study_metric(run_output: dict[str, Any], metric_path: str) -> Any:
    path = str(metric_path or "").strip()
    if not path:
        return None
    if path == "derived.closest_approach_km":
        existing = run_output.get("closest_approach_km")
        if isinstance(existing, (int, float, np.integer, np.floating)) and np.isfinite(float(existing)):
            return float(existing)
        value = closest_approach_from_run_payload(run_output)
        return float(value) if np.isfinite(value) else None
    if path in {
        "derived.initial_range_km",
        "derived.final_range_km",
        "derived.final_relative_speed_m_s",
        "derived.max_relative_speed_m_s",
    }:
        existing = _deep_get(dict(run_output.get("derived", {}) or {}), path[len("derived.") :], default=None)
        if isinstance(existing, (int, float, np.integer, np.floating)) and np.isfinite(float(existing)):
            return float(existing)
        summary = relative_motion_summary_from_run_payload(run_output)
        value = summary.get(path.removeprefix("derived."))
        return float(value) if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) else None
    if path == "derived.attitude_guardrail_events":
        guardrail_map = dict(dict(run_output.get("summary", {}) or {}).get("attitude_guardrail_stats", {}) or {})
        return int(sum(int(v) for v in guardrail_map.values())) if guardrail_map else 0
    if path.startswith("derived.reentry_"):
        reentry_map = {
            "derived.reentry_peak_g_load": ("peak_g_load", "max"),
            "derived.reentry_peak_heat_rate_w_m2": ("peak_heat_rate_w_m2", "max"),
            "derived.reentry_final_heat_load_j_m2": ("final_heat_load_j_m2", "max"),
            "derived.reentry_min_altitude_km": ("min_altitude_km", "min"),
        }
        key_mode = reentry_map.get(path)
        if key_mode is None:
            return None
        key, mode = key_mode
        arr = _numeric_values(_reentry_summary_values(run_output, key))
        if arr.size == 0:
            return None
        return float(np.min(arr) if mode == "min" else np.max(arr))
    if path == "derived.ground_access_total_windows":
        access_summary = dict(dict(run_output.get("summary", {}) or {}).get("ground_station_access_summary", {}) or {})
        values: list[Any] = []
        for station_summary in access_summary.values():
            station_map = dict(station_summary or {})
            for object_summary in station_map.values():
                obj = dict(object_summary or {})
                values.append(obj.get("access_window_count", obj.get("access_samples")))
        return int(np.sum(_numeric_values(values))) if values else 0
    if path == "derived.ground_access_total_duration_s":
        access_summary = dict(dict(run_output.get("summary", {}) or {}).get("ground_station_access_summary", {}) or {})
        values = []
        for station_summary in access_summary.values():
            station_map = dict(station_summary or {})
            for object_summary in station_map.values():
                obj = dict(object_summary or {})
                values.append(obj.get("total_access_duration_s", obj.get("access_duration_s")))
        numeric = _numeric_values(values)
        return float(np.sum(numeric)) if numeric.size else 0.0
    if path in {"derived.final_altitude_km_min", "derived.final_altitude_km_max"}:
        values: list[float] = []
        for hist in dict(run_output.get("truth_by_object", {}) or {}).values():
            arr = np.array(hist, dtype=float)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
                r_km = float(np.linalg.norm(arr[-1, :3]))
                values.append(r_km - 6378.137)
        finite = _numeric_values(values)
        if finite.size == 0:
            return None
        return float(np.min(finite) if path.endswith("_min") else np.max(finite))
    if path.startswith("summary."):
        return _deep_get(dict(run_output.get("summary", {}) or {}), path[len("summary.") :], default=None)
    if path.startswith("payload."):
        return _deep_get(run_output, path[len("payload.") :], default=None)
    return _deep_get(run_output, path, default=None)


def extract_study_metrics(run_output: dict[str, Any], metric_specs: list[Any]) -> dict[str, Any]:
    specs = normalize_study_metric_specs(metric_specs)
    return {str(spec["name"]): extract_study_metric(run_output, str(spec["path"])) for spec in specs}


def _coerce_gate_value(value: Any) -> float | str | bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        fv = float(value)
        return fv if np.isfinite(fv) else None
    if value is None:
        return None
    return str(value)


def _compare_gate(actual: Any, op: str, expected: Any) -> bool:
    op_norm = str(op or "").strip().lower()
    if op_norm in {"=", "==", "eq"}:
        return actual == expected
    if op_norm in {"!=", "<>", "ne"}:
        return actual != expected
    if op_norm in {"in"}:
        return actual in list(expected or []) if isinstance(expected, (list, tuple, set)) else False
    if op_norm in {"not_in", "not in"}:
        return actual not in list(expected or []) if isinstance(expected, (list, tuple, set)) else True
    actual_num = _coerce_gate_value(actual)
    expected_num = _coerce_gate_value(expected)
    if not isinstance(actual_num, (int, float)) or isinstance(actual_num, bool):
        return False
    if not isinstance(expected_num, (int, float)) or isinstance(expected_num, bool):
        return False
    a = float(actual_num)
    b = float(expected_num)
    if op_norm in {"<", "lt"}:
        return a < b
    if op_norm in {"<=", "lte", "le"}:
        return a <= b
    if op_norm in {">", "gt"}:
        return a > b
    if op_norm in {">=", "gte", "ge"}:
        return a >= b
    raise ValueError(f"Unsupported study metric gate op: {op}")


def evaluate_study_metric_gates(run_output: dict[str, Any], gates: list[Any] | tuple[Any, ...] | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw in list(gates or []):
        if not isinstance(raw, dict):
            continue
        metric_path = str(raw.get("metric", "") or raw.get("path", "") or "").strip()
        if not metric_path:
            continue
        op = str(raw.get("op", "<=") or "<=")
        expected = raw.get("value")
        actual = extract_study_metric(run_output, metric_path)
        try:
            passed = _compare_gate(actual, op, expected)
            error = ""
        except ValueError as exc:
            passed = False
            error = str(exc)
        results.append(
            {
                "name": str(raw.get("name", "") or _metric_name_from_path(metric_path)),
                "metric": metric_path,
                "op": op,
                "value": expected,
                "actual": actual,
                "pass": bool(passed),
                "error": error,
            }
        )
    return results


_closest_approach_from_run_payload = closest_approach_from_run_payload
_relative_range_series_from_run_payload = relative_range_series_from_run_payload
_extract_study_metric = extract_study_metric
_extract_study_metrics = extract_study_metrics
