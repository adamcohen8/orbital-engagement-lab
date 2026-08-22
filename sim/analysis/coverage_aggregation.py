"""Phase 5 deterministic multi-asset coverage aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sim.analysis.coverage_queries import CoverageProduct, validate_global_coverage_product
from sim.analysis.global_coverage import CoverageCellMetrics, summarize_sampled_coverage_mask
from sim.analysis.healpix import HEALPIX_GRID_ID, WGS84_SURFACE_AREA_KM2, healpix_npix

CONSTELLATION_COVERAGE_CONTRACT_VERSION = "oel.constellation-coverage-aggregation.v0.2"


@dataclass(frozen=True)
class ConstellationCoverageConfig:
    analysis_id: str
    member_analysis_ids: tuple[str, ...]
    order: int
    service_definition_id: str
    required_multiplicity: int = 1
    max_asset_cell_time_values: int = 500_000_000

    def __post_init__(self) -> None:
        analysis_id = str(self.analysis_id or "").strip()
        if not analysis_id:
            raise ValueError("analysis_id must be a non-empty string.")
        service_definition_id = str(self.service_definition_id or "").strip()
        if not service_definition_id:
            raise ValueError("service_definition_id must be a non-empty string.")
        members = tuple(sorted(str(value or "").strip() for value in self.member_analysis_ids))
        if len(members) < 2 or any(not value for value in members) or len(set(members)) != len(members):
            raise ValueError("member_analysis_ids must contain at least two unique non-empty IDs.")
        if isinstance(self.order, (bool, np.bool_)) or int(self.order) != self.order:
            raise ValueError("order must be an integer.")
        multiplicity = self.required_multiplicity
        if (
            isinstance(multiplicity, (bool, np.bool_))
            or int(multiplicity) != multiplicity
            or not 1 <= int(multiplicity) <= len(members)
        ):
            raise ValueError("required_multiplicity must be within [1, member count].")
        resource_limit = self.max_asset_cell_time_values
        if (
            isinstance(resource_limit, (bool, np.bool_))
            or int(resource_limit) != resource_limit
            or int(resource_limit) <= 0
        ):
            raise ValueError("max_asset_cell_time_values must be a positive integer.")
        object.__setattr__(self, "analysis_id", analysis_id)
        object.__setattr__(self, "service_definition_id", service_definition_id)
        object.__setattr__(self, "member_analysis_ids", members)
        if int(self.order) not in range(5, 9):
            raise ValueError("Constellation coverage v0.2 supports HEALPix orders 5 through 8.")
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "required_multiplicity", int(multiplicity))
        object.__setattr__(self, "max_asset_cell_time_values", int(resource_limit))


@dataclass(frozen=True)
class ConstellationCoverageResult:
    config: ConstellationCoverageConfig
    times_s: np.ndarray
    covered_cell_count: np.ndarray
    instantaneous_covered_fraction: np.ndarray
    cell_geodetic_latitude_deg: np.ndarray
    cell_longitude_deg: np.ndarray
    cell_metrics: CoverageCellMetrics
    mean_multiplicity_per_cell: np.ndarray
    max_multiplicity_per_cell: np.ndarray
    maximum_multiplicity_by_sample: np.ndarray
    active_asset_count_by_sample: np.ndarray
    multiplicity_histogram: np.ndarray
    summary: dict[str, Any]
    resource_estimate: dict[str, int]
    member_domain_disposition: str
    member_semantic_sha256: tuple[str, ...]
    interval_semantic_sha256: str


@dataclass(frozen=True)
class ConstellationCoverageArtifacts:
    output_dir: Path
    manifest_json: Path
    summary_json: Path
    samples_csv: Path
    cells_csv: Path
    intervals_npz: Path


def _dense_mask(product: CoverageProduct) -> np.ndarray:
    sample_count = product.times_s.size
    cell_count = healpix_npix(product.config.order)
    intervals = product.cell_metrics.intervals
    sparse_cells = np.asarray(intervals.cell_index, dtype=np.int64)
    if (
        sparse_cells.ndim != 1
        or np.any(sparse_cells < 0)
        or np.any(sparse_cells >= cell_count)
        or (sparse_cells.size > 1 and np.any(sparse_cells[1:] <= sparse_cells[:-1]))
    ):
        raise ValueError("Coverage member sparse cells must be unique, sorted, and canonical.")
    if intervals.interval_offset.shape != (sparse_cells.size + 1,):
        raise ValueError("Coverage member interval offsets are malformed.")
    mask = np.zeros((sample_count, cell_count), dtype=bool)
    for sparse_index, cell in enumerate(sparse_cells):
        begin = int(intervals.interval_offset[sparse_index])
        end = int(intervals.interval_offset[sparse_index + 1])
        for interval_index in range(begin, end):
            start = int(intervals.start_sample_index[interval_index])
            stop = int(intervals.end_sample_index_exclusive[interval_index])
            if not 0 <= start < stop <= sample_count:
                raise ValueError("Coverage member contains an invalid sparse interval.")
            mask[start:stop, int(cell)] = True
    return mask


def evaluate_constellation_coverage(
    config: ConstellationCoverageConfig,
    members: Iterable[CoverageProduct],
) -> ConstellationCoverageResult:
    """Aggregate matching global products without rerunning member geometry."""

    products = tuple(sorted(tuple(members), key=lambda value: value.config.analysis_id))
    ids = tuple(product.config.analysis_id for product in products)
    if ids != config.member_analysis_ids:
        raise ValueError("Coverage member IDs do not match config.member_analysis_ids.")
    if any(product.config.order != config.order for product in products):
        raise ValueError("All coverage members must use the configured HEALPix order.")
    for product in products:
        validate_global_coverage_product(product)
    supported_dispositions = {
        "global_earth",
        "global_earth_communications_service",
        "global_earth_constellation_aggregate",
    }
    if any(
        product.summary.get("domain_disposition") not in supported_dispositions
        for product in products
    ):
        raise ValueError("Every constellation member must be a completed global-Earth product.")
    member_dispositions = {
        str(product.summary["domain_disposition"])
        for product in products
    }
    if len(member_dispositions) != 1:
        raise ValueError("Constellation members must share one coverage domain disposition.")
    member_disposition = next(iter(member_dispositions))
    if member_disposition == "global_earth_communications_service":
        service_ids = {str(product.summary.get("service_id") or "") for product in products}
        if len(service_ids) != 1 or not next(iter(service_ids)):
            raise ValueError("Communications members must share one non-empty service_id.")
    reference = products[0]
    for product in products[1:]:
        if not np.array_equal(product.times_s, reference.times_s):
            raise ValueError("All coverage members must use identical analysis epochs.")
        if not np.array_equal(
            product.cell_geodetic_latitude_deg,
            reference.cell_geodetic_latitude_deg,
        ) or not np.array_equal(product.cell_longitude_deg, reference.cell_longitude_deg):
            raise ValueError("All coverage members must use identical canonical cells.")
    sample_count = reference.times_s.size
    cell_count = healpix_npix(config.order)
    values = len(products) * sample_count * cell_count
    resources = {
        "member_count": len(products),
        "sample_count": int(sample_count),
        "cell_count": cell_count,
        "asset_cell_time_values": values,
        "estimated_dense_working_bytes": values + sample_count * cell_count * 3,
        "max_asset_cell_time_values": config.max_asset_cell_time_values,
    }
    if values > config.max_asset_cell_time_values:
        raise ValueError("Constellation aggregation exceeds max_asset_cell_time_values.")
    member_masks = np.asarray([_dense_mask(product) for product in products], dtype=np.uint8)
    multiplicity = np.sum(member_masks, axis=0, dtype=np.uint16)
    qualified = multiplicity >= config.required_multiplicity
    metrics = summarize_sampled_coverage_mask(qualified, reference.times_s)
    covered = np.count_nonzero(qualified, axis=1).astype(np.int64)
    mean_multiplicity = np.mean(multiplicity, axis=0)
    max_multiplicity = np.max(multiplicity, axis=0)
    maximum_by_sample = np.max(multiplicity, axis=1)
    active_assets = np.count_nonzero(np.any(member_masks.astype(bool), axis=2), axis=0)
    histogram = np.bincount(
        multiplicity.reshape(-1).astype(np.int64),
        minlength=len(products) + 1,
    )
    member_hashes = tuple(product.interval_semantic_sha256 for product in products)
    semantic_hash = _semantic_hash(
        config,
        reference.times_s,
        metrics,
        multiplicity,
        member_hashes,
    )
    summary = _summary(
        config,
        reference.times_s,
        covered,
        metrics,
        mean_multiplicity,
        max_multiplicity,
        maximum_by_sample,
        active_assets,
        histogram,
        member_disposition,
    )
    return ConstellationCoverageResult(
        config=config,
        times_s=reference.times_s,
        covered_cell_count=covered,
        instantaneous_covered_fraction=covered.astype(float) / cell_count,
        cell_geodetic_latitude_deg=reference.cell_geodetic_latitude_deg,
        cell_longitude_deg=reference.cell_longitude_deg,
        cell_metrics=metrics,
        mean_multiplicity_per_cell=mean_multiplicity,
        max_multiplicity_per_cell=max_multiplicity,
        maximum_multiplicity_by_sample=maximum_by_sample,
        active_asset_count_by_sample=active_assets,
        multiplicity_histogram=histogram,
        summary=summary,
        resource_estimate=resources,
        member_domain_disposition=member_disposition,
        member_semantic_sha256=member_hashes,
        interval_semantic_sha256=semantic_hash,
    )


def _semantic_hash(
    config: ConstellationCoverageConfig,
    times: np.ndarray,
    metrics: CoverageCellMetrics,
    multiplicity: np.ndarray,
    member_hashes: tuple[str, ...],
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "config": asdict(config),
                "contract_version": CONSTELLATION_COVERAGE_CONTRACT_VERSION,
                "grid_identity": HEALPIX_GRID_ID,
                "member_semantic_sha256": member_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    intervals = metrics.intervals
    for name, values, dtype in (
        ("times_s", times, "<f8"),
        ("cell_index", intervals.cell_index, "<i8"),
        ("interval_offset", intervals.interval_offset, "<i8"),
        ("start_sample_index", intervals.start_sample_index, "<i8"),
        ("end_sample_index_exclusive", intervals.end_sample_index_exclusive, "<i8"),
        ("multiplicity", multiplicity, "<u2"),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _summary(
    config: ConstellationCoverageConfig,
    times: np.ndarray,
    covered: np.ndarray,
    metrics: CoverageCellMetrics,
    mean_multiplicity: np.ndarray,
    max_multiplicity: np.ndarray,
    maximum_by_sample: np.ndarray,
    active_assets: np.ndarray,
    histogram: np.ndarray,
    member_disposition: str,
) -> dict[str, Any]:
    cells = healpix_npix(config.order)
    finite_revisit = metrics.max_complete_revisit_gap_s[
        np.isfinite(metrics.max_complete_revisit_gap_s)
    ]
    duration = float(times[-1] - times[0])
    fraction = covered.astype(float) / cells
    revisit_percentiles = {
        str(percentile): (
            None if not finite_revisit.size else float(np.percentile(finite_revisit, percentile))
        )
        for percentile in (10, 50, 90, 95)
    }
    return {
        "contract_version": CONSTELLATION_COVERAGE_CONTRACT_VERSION,
        "analysis_id": config.analysis_id,
        "status": "complete",
        "domain_disposition": "global_earth_constellation_aggregate",
        "member_domain_disposition": member_disposition,
        "service_definition_id": config.service_definition_id,
        "member_analysis_ids": list(config.member_analysis_ids),
        "member_count": len(config.member_analysis_ids),
        "required_multiplicity": config.required_multiplicity,
        "grid_identity": HEALPIX_GRID_ID,
        "order": config.order,
        "cell_count": cells,
        "cell_area_km2": WGS84_SURFACE_AREA_KM2 / cells,
        "sample_count": int(times.size),
        "horizon_start_s": float(times[0]),
        "horizon_end_s": float(times[-1]),
        "time_weighted_mean_covered_fraction": float(
            np.dot(fraction[:-1], np.diff(times)) / duration
        ),
        "never_service_qualified_cell_count": int(np.count_nonzero(metrics.interval_count == 0)),
        "mean_multiplicity": {
            "minimum": float(np.min(mean_multiplicity)),
            "mean": float(np.mean(mean_multiplicity)),
            "maximum": float(np.max(mean_multiplicity)),
        },
        "maximum_multiplicity": int(np.max(max_multiplicity)),
        "maximum_multiplicity_by_sample": {
            "minimum": int(np.min(maximum_by_sample)),
            "mean": float(np.mean(maximum_by_sample)),
            "maximum": int(np.max(maximum_by_sample)),
        },
        "active_asset_count_by_sample": {
            "minimum": int(np.min(active_assets)),
            "mean": float(np.mean(active_assets)),
            "maximum": int(np.max(active_assets)),
        },
        "sampled_dwell_s": {
            "minimum": float(np.min(metrics.dwell_s)),
            "mean": float(np.mean(metrics.dwell_s)),
            "maximum": float(np.max(metrics.dwell_s)),
            "percentiles": {
                str(percentile): float(np.percentile(metrics.dwell_s, percentile))
                for percentile in (10, 50, 90, 95)
            },
        },
        "multiplicity_cell_sample_count": {
            str(index): int(value) for index, value in enumerate(histogram)
        },
        "max_complete_revisit_gap_s": {
            "mean": None if not finite_revisit.size else float(np.mean(finite_revisit)),
            "maximum": None if not finite_revisit.size else float(np.max(finite_revisit)),
            "percentiles": revisit_percentiles,
            "evaluated_cell_count": int(finite_revisit.size),
        },
        "claim_limits": [
            "Aggregation combines supplied sampled member products and does not rerun propagation.",
            "Members must share exact epochs, cell identity, and resolution.",
            "Members share one domain disposition and an explicitly declared service definition.",
            "Complete revisit excludes boundary-censored gaps.",
            "No terminal contention, routing, scheduling, or probabilistic availability is implied.",
        ],
    }


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_constellation_coverage_artifacts(
    result: ConstellationCoverageResult,
    output_dir: str | Path,
) -> ConstellationCoverageArtifacts:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Constellation coverage output already exists: {destination}")
    destination.mkdir(parents=True)
    manifest = destination / "constellation_coverage_manifest.json"
    summary = destination / "constellation_coverage_summary.json"
    samples = destination / "constellation_coverage_samples.csv"
    cells = destination / "constellation_coverage_cells.csv"
    intervals = destination / "constellation_coverage_intervals.npz"
    _json_dump(summary, result.summary)
    with samples.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "sample_index",
                "time_s",
                "qualified_cell_count",
                "qualified_fraction",
                "maximum_multiplicity",
                "active_asset_count",
            )
        )
        for index, time_s in enumerate(result.times_s):
            writer.writerow(
                (
                    index,
                    f"{float(time_s):.17g}",
                    int(result.covered_cell_count[index]),
                    f"{float(result.instantaneous_covered_fraction[index]):.17g}",
                    int(result.maximum_multiplicity_by_sample[index]),
                    int(result.active_asset_count_by_sample[index]),
                )
            )
    metrics = result.cell_metrics
    with cells.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "cell_index",
                "geodetic_latitude_deg",
                "longitude_deg",
                "mean_multiplicity",
                "maximum_multiplicity",
                "sampled_dwell_s",
                "interval_count",
                "max_complete_revisit_gap_s",
            )
        )
        for index in range(metrics.cell_index.size):
            gap = metrics.max_complete_revisit_gap_s[index]
            writer.writerow(
                (
                    int(metrics.cell_index[index]),
                    f"{float(result.cell_geodetic_latitude_deg[index]):.17g}",
                    f"{float(result.cell_longitude_deg[index]):.17g}",
                    f"{float(result.mean_multiplicity_per_cell[index]):.17g}",
                    int(result.max_multiplicity_per_cell[index]),
                    f"{float(metrics.dwell_s[index]):.17g}",
                    int(metrics.interval_count[index]),
                    "" if not np.isfinite(gap) else f"{float(gap):.17g}",
                )
            )
    sparse = metrics.intervals
    np.savez_compressed(
        intervals,
        cell_index=np.asarray(sparse.cell_index, dtype="<i8"),
        interval_offset=np.asarray(sparse.interval_offset, dtype="<i8"),
        start_sample_index=np.asarray(sparse.start_sample_index, dtype="<i8"),
        end_sample_index_exclusive=np.asarray(sparse.end_sample_index_exclusive, dtype="<i8"),
    )
    artifacts = {
        path.name: {"sha256": _file_hash(path)} for path in (summary, samples, cells, intervals)
    }
    _json_dump(
        manifest,
        {
            "contract_version": CONSTELLATION_COVERAGE_CONTRACT_VERSION,
            "analysis_id": result.config.analysis_id,
            "status": "complete",
            "normalized_config": asdict(result.config),
            "member_semantic_sha256": list(result.member_semantic_sha256),
            "semantic_sha256": result.interval_semantic_sha256,
            "resource_estimate": result.resource_estimate,
            "artifacts": artifacts,
            "claim_limits": result.summary["claim_limits"],
        },
    )
    return ConstellationCoverageArtifacts(
        output_dir=destination,
        manifest_json=manifest,
        summary_json=summary,
        samples_csv=samples,
        cells_csv=cells,
        intervals_npz=intervals,
    )


__all__ = [
    "CONSTELLATION_COVERAGE_CONTRACT_VERSION",
    "ConstellationCoverageArtifacts",
    "ConstellationCoverageConfig",
    "ConstellationCoverageResult",
    "evaluate_constellation_coverage",
    "write_constellation_coverage_artifacts",
]
