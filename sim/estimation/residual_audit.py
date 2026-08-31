from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


def build_residual_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    decision_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Summarize standardized residual evidence without discarding source rows."""

    normalized = [dict(record) for record in records]
    return {
        "schema_version": 1,
        "residual_space": "cholesky_whitened_when_available_else_diagonal_normalized",
        "overall": _statistics(normalized),
        "by_partition": _group_statistics(normalized, "partition"),
        "by_measurement_type": _group_statistics(normalized, "measurement_type"),
        "by_station": _group_statistics(normalized, "station_id"),
        "by_arc": _group_statistics(normalized, "arc_id"),
        "decisions": _decision_statistics(decision_records),
    }


def residual_records_from_vectors(
    values: Sequence[float] | np.ndarray,
    *,
    partition: str,
    measurement_type: str,
    component_count: int,
    arc_id: str,
) -> list[dict[str, Any]]:
    residual = np.asarray(values, dtype=float).reshape(-1)
    if component_count <= 0:
        raise ValueError("component_count must be positive.")
    return [
        {
            "residual_index": int(index),
            "observation_index": int(index // component_count),
            "component_index": int(index % component_count),
            "partition": str(partition),
            "measurement_type": str(measurement_type),
            "arc_id": str(arc_id),
            "whitened_residual": float(value),
        }
        for index, value in enumerate(residual)
    ]


def _group_statistics(records: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        raw = record.get(key)
        if raw in (None, ""):
            continue
        groups.setdefault(str(raw), []).append(record)
    return [{key: value, **_statistics(groups[value])} for value in sorted(groups)]


def _statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.array([value for record in records if (value := _residual_value(record)) is not None], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "residual_count": 0,
            "mean": None,
            "standard_deviation": None,
            "rms": None,
            "median": None,
            "p95_absolute": None,
            "max_absolute": None,
            "lag1_autocorrelation": None,
            "durbin_watson": None,
            "whiteness_status": "insufficient_samples",
        }
    centered = finite - float(np.mean(finite))
    centered_energy = float(np.dot(centered, centered))
    raw_energy = float(np.dot(finite, finite))
    lag1 = None
    durbin_watson = None
    whiteness_status = "insufficient_samples"
    if finite.size >= 2:
        lag1 = None if centered_energy <= 0.0 else float(np.dot(centered[:-1], centered[1:]) / centered_energy)
        durbin_watson = None if raw_energy <= 0.0 else float(np.dot(np.diff(finite), np.diff(finite)) / raw_energy)
        whiteness_status = "computed"
    return {
        "residual_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "standard_deviation": float(np.std(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "median": float(np.median(finite)),
        "p95_absolute": float(np.percentile(np.abs(finite), 95.0)),
        "max_absolute": float(np.max(np.abs(finite))),
        "lag1_autocorrelation": lag1,
        "durbin_watson": durbin_watson,
        "whiteness_status": whiteness_status,
    }


def _residual_value(record: Mapping[str, Any]) -> float | None:
    for key in ("whitened_residual", "normalized_residual", "standardized_residual"):
        if record.get(key) is not None:
            return float(record[key])
    return None


def _decision_statistics(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in decisions]
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(str(reason) for reason in list(row.get("reasons", []) or []))
    accepted = sum(bool(row.get("accepted", False)) for row in rows)
    downweighted = sum(float(row.get("robust_weight", 1.0)) < 1.0 - 1.0e-12 for row in rows)
    return {
        "decision_count": len(rows),
        "accepted_count": accepted,
        "rejected_count": len(rows) - accepted,
        "downweighted_count": downweighted,
        "reason_counts": dict(sorted(reason_counts.items())),
    }
