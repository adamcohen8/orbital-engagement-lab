"""Content-bound cadence and resolution sensitivity evidence for coverage products."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.analysis.coverage_queries import CoverageProduct, validate_global_coverage_product

COVERAGE_SENSITIVITY_CONTRACT_VERSION = "oel.coverage-sensitivity-evidence.v0.2"


@dataclass(frozen=True)
class CoverageSensitivityCriteria:
    maximum_mean_fraction_absolute_delta: float
    maximum_ever_fraction_absolute_delta: float
    maximum_complete_revisit_mean_absolute_delta_s: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_mean_fraction_absolute_delta",
            "maximum_ever_fraction_absolute_delta",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite within [0, 1].")
            object.__setattr__(self, field_name, value)
        if self.maximum_complete_revisit_mean_absolute_delta_s is not None:
            value = float(self.maximum_complete_revisit_mean_absolute_delta_s)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "maximum_complete_revisit_mean_absolute_delta_s must be nonnegative and finite."
                )
            object.__setattr__(
                self,
                "maximum_complete_revisit_mean_absolute_delta_s",
                value,
            )


@dataclass(frozen=True)
class CoverageSensitivityResult:
    comparison_id: str
    comparison_kind: str
    baseline_analysis_id: str
    refined_analysis_id: str
    baseline_semantic_sha256: str
    refined_semantic_sha256: str
    baseline_order: int
    refined_order: int
    baseline_sample_count: int
    refined_sample_count: int
    matched_assumptions_sha256: str
    metrics: dict[str, Any]
    criteria: CoverageSensitivityCriteria
    passed: bool
    disposition: str
    semantic_sha256: str


def _mean_covered_fraction(product: CoverageProduct) -> float:
    duration = float(product.times_s[-1] - product.times_s[0])
    return float(
        np.dot(product.instantaneous_covered_fraction[:-1], np.diff(product.times_s))
        / duration
    )


def _ever_fraction(product: CoverageProduct) -> float:
    return float(np.count_nonzero(product.cell_metrics.interval_count > 0) / product.cell_metrics.interval_count.size)


def _mean_complete_revisit(product: CoverageProduct) -> float | None:
    values = product.cell_metrics.max_complete_revisit_gap_s
    finite = values[np.isfinite(values)]
    return None if not finite.size else float(np.mean(finite))


def _matched_assumption_record(product: CoverageProduct) -> dict[str, Any]:
    if not is_dataclass(product.config):
        raise ValueError("Sensitivity products require typed dataclass configurations.")
    record = asdict(product.config)
    for field_name in (
        "analysis_id",
        "order",
        "chunk_size",
        "max_working_memory_bytes",
        "max_cell_time_comparisons",
        "max_asset_cell_time_values",
    ):
        record.pop(field_name, None)
    return {
        "config_type": type(product.config).__qualname__,
        "domain_disposition": product.summary.get("domain_disposition"),
        "scientific_config": record,
    }


def evaluate_coverage_sensitivity(
    *,
    comparison_id: str,
    comparison_kind: str,
    baseline: CoverageProduct,
    refined: CoverageProduct,
    criteria: CoverageSensitivityCriteria,
) -> CoverageSensitivityResult:
    """Compare already-computed products without claiming unsupported convergence."""

    identity = str(comparison_id or "").strip()
    if not identity:
        raise ValueError("comparison_id must be a non-empty string.")
    kind = str(comparison_kind or "").strip().lower()
    if kind not in {"cadence", "resolution"}:
        raise ValueError("comparison_kind must be cadence or resolution.")
    validate_global_coverage_product(baseline)
    validate_global_coverage_product(refined)
    if baseline.config.analysis_id == refined.config.analysis_id:
        raise ValueError("Sensitivity products must use distinct analysis IDs.")
    if baseline.summary.get("domain_disposition") != refined.summary.get("domain_disposition"):
        raise ValueError("Sensitivity products must have the same global domain disposition.")
    if baseline.times_s[0] != refined.times_s[0] or baseline.times_s[-1] != refined.times_s[-1]:
        raise ValueError("Sensitivity products must share the exact evaluated horizon.")
    baseline_assumptions = _matched_assumption_record(baseline)
    refined_assumptions = _matched_assumption_record(refined)
    if baseline_assumptions != refined_assumptions:
        raise ValueError("Sensitivity products must preserve all non-refinement scientific assumptions.")
    matched_assumptions_hash = hashlib.sha256(
        json.dumps(
            baseline_assumptions,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if kind == "cadence":
        if baseline.config.order != refined.config.order:
            raise ValueError("Cadence sensitivity requires the same HEALPix order.")
        if refined.times_s.size <= baseline.times_s.size:
            raise ValueError("Cadence refinement must contain more samples than the baseline.")
        if not np.all(np.isin(baseline.times_s, refined.times_s)):
            raise ValueError("Cadence refinement must retain every baseline epoch exactly.")
    else:
        if refined.config.order <= baseline.config.order:
            raise ValueError("Resolution refinement must increase HEALPix order.")
        if not np.array_equal(baseline.times_s, refined.times_s):
            raise ValueError("Resolution refinement must preserve every analysis epoch exactly.")
    baseline_mean = _mean_covered_fraction(baseline)
    refined_mean = _mean_covered_fraction(refined)
    baseline_ever = _ever_fraction(baseline)
    refined_ever = _ever_fraction(refined)
    baseline_revisit = _mean_complete_revisit(baseline)
    refined_revisit = _mean_complete_revisit(refined)
    revisit_delta = (
        None
        if baseline_revisit is None or refined_revisit is None
        else abs(refined_revisit - baseline_revisit)
    )
    metrics = {
        "time_weighted_mean_covered_fraction": {
            "baseline": baseline_mean,
            "refined": refined_mean,
            "absolute_delta": abs(refined_mean - baseline_mean),
        },
        "ever_covered_fraction": {
            "baseline": baseline_ever,
            "refined": refined_ever,
            "absolute_delta": abs(refined_ever - baseline_ever),
        },
        "mean_complete_revisit_gap_s": {
            "baseline": baseline_revisit,
            "refined": refined_revisit,
            "absolute_delta": revisit_delta,
        },
    }
    passed = (
        metrics["time_weighted_mean_covered_fraction"]["absolute_delta"]
        <= criteria.maximum_mean_fraction_absolute_delta
        and metrics["ever_covered_fraction"]["absolute_delta"]
        <= criteria.maximum_ever_fraction_absolute_delta
    )
    if criteria.maximum_complete_revisit_mean_absolute_delta_s is not None:
        passed = bool(
            passed
            and revisit_delta is not None
            and revisit_delta <= criteria.maximum_complete_revisit_mean_absolute_delta_s
        )
    record = {
        "contract_version": COVERAGE_SENSITIVITY_CONTRACT_VERSION,
        "comparison_id": identity,
        "comparison_kind": kind,
        "baseline_analysis_id": baseline.config.analysis_id,
        "refined_analysis_id": refined.config.analysis_id,
        "baseline_semantic_sha256": baseline.interval_semantic_sha256,
        "refined_semantic_sha256": refined.interval_semantic_sha256,
        "baseline_order": baseline.config.order,
        "refined_order": refined.config.order,
        "baseline_sample_count": int(baseline.times_s.size),
        "refined_sample_count": int(refined.times_s.size),
        "matched_assumptions_sha256": matched_assumptions_hash,
        "metrics": metrics,
        "criteria": asdict(criteria),
        "passed": passed,
        "disposition": "within_declared_limits" if passed else "exceeds_declared_limits",
    }
    semantic_hash = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return CoverageSensitivityResult(
        comparison_id=identity,
        comparison_kind=kind,
        baseline_analysis_id=baseline.config.analysis_id,
        refined_analysis_id=refined.config.analysis_id,
        baseline_semantic_sha256=baseline.interval_semantic_sha256,
        refined_semantic_sha256=refined.interval_semantic_sha256,
        baseline_order=baseline.config.order,
        refined_order=refined.config.order,
        baseline_sample_count=int(baseline.times_s.size),
        refined_sample_count=int(refined.times_s.size),
        matched_assumptions_sha256=matched_assumptions_hash,
        metrics=metrics,
        criteria=criteria,
        passed=passed,
        disposition=record["disposition"],
        semantic_sha256=semantic_hash,
    )


def write_coverage_sensitivity_evidence(
    result: CoverageSensitivityResult,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Coverage sensitivity evidence already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = asdict(result)
    record["contract_version"] = COVERAGE_SENSITIVITY_CONTRACT_VERSION
    record["claim_limits"] = [
        "This packet compares two supplied finite-horizon sampled products.",
        "Passing caller-declared limits is sensitivity evidence, not independent validation.",
        "No convergence rate, steady-state behavior, or operational assurance is inferred.",
    ]
    destination.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "COVERAGE_SENSITIVITY_CONTRACT_VERSION",
    "CoverageSensitivityCriteria",
    "CoverageSensitivityResult",
    "evaluate_coverage_sensitivity",
    "write_coverage_sensitivity_evidence",
]
