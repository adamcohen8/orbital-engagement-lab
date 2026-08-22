"""Regional-mask and point-cell queries over a global coverage product."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from sim.analysis.global_coverage import (
    GLOBAL_COVERAGE_CONTRACT_VERSION,
    CoverageCellMetrics,
    SparseCoverageIntervals,
)
from sim.analysis.healpix import (
    HEALPIX_GRID_ID,
    WGS84_SURFACE_AREA_KM2,
    healpix_npix,
    healpix_wgs84_centers,
    wgs84_points_to_healpix_nested,
)

COVERAGE_QUERY_SCHEMA_VERSION = "oel.global-earth-coverage-queries.v0.1"


class _CoverageProductConfig(Protocol):
    analysis_id: str
    order: int


class CoverageProduct(Protocol):
    """Structural query surface shared by basic and rich global products."""

    config: _CoverageProductConfig
    times_s: np.ndarray
    covered_cell_count: np.ndarray
    instantaneous_covered_fraction: np.ndarray
    cell_geodetic_latitude_deg: np.ndarray
    cell_longitude_deg: np.ndarray
    cell_metrics: CoverageCellMetrics
    summary: dict[str, Any]
    interval_semantic_sha256: str


def _required_identity(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


@dataclass(frozen=True)
class CoverageRegionMask:
    """A versioned, provenance-bound, sorted set of canonical coverage cells."""

    region_id: str
    mask_version: str
    provenance: str
    cell_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("region_id", "mask_version", "provenance"):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        raw = np.asarray(self.cell_indices)
        if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
            raise ValueError("region cell_indices must be a one-dimensional integer sequence.")
        cells = raw.astype(np.int64, copy=False)
        if cells.size == 0:
            raise ValueError("region cell_indices must not be empty.")
        if np.any(cells < 0):
            raise ValueError("region cell_indices must be non-negative.")
        if cells.size > 1 and np.any(cells[1:] <= cells[:-1]):
            raise ValueError("region cell_indices must be unique and strictly increasing.")
        object.__setattr__(self, "cell_indices", tuple(int(value) for value in cells))


@dataclass(frozen=True)
class CoveragePointQuery:
    """A WGS84 zero-height point interpreted through its containing cell."""

    point_id: str
    longitude_deg: float
    geodetic_latitude_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", _required_identity(self.point_id, "point_id"))
        longitude = float(self.longitude_deg)
        latitude = float(self.geodetic_latitude_deg)
        if not np.isfinite(longitude) or not np.isfinite(latitude):
            raise ValueError("point longitude and geodetic latitude must be finite.")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("point geodetic latitude must be within [-90, 90] degrees.")
        normalized_longitude = (longitude + 180.0) % 360.0 - 180.0
        object.__setattr__(self, "longitude_deg", float(normalized_longitude))
        object.__setattr__(self, "geodetic_latitude_deg", latitude)


@dataclass(frozen=True)
class CoverageRegionResult:
    mask: CoverageRegionMask
    mask_semantic_sha256: str
    covered_cell_count: np.ndarray
    instantaneous_covered_fraction: np.ndarray
    covered_area_km2: np.ndarray
    cell_metrics: CoverageCellMetrics
    summary: dict[str, Any]


@dataclass(frozen=True)
class CoveragePointResult:
    query: CoveragePointQuery
    cell_index: int
    cell_geodetic_latitude_deg: float
    cell_longitude_deg: float
    covered_by_sample: np.ndarray
    cell_metrics: CoverageCellMetrics
    summary: dict[str, Any]


@dataclass(frozen=True)
class CoverageQueryResult:
    source_analysis_id: str
    source_interval_semantic_sha256: str
    order: int
    times_s: np.ndarray
    regions: tuple[CoverageRegionResult, ...]
    points: tuple[CoveragePointResult, ...]
    query_semantic_sha256: str


@dataclass(frozen=True)
class CoverageQueryArtifacts:
    output_dir: Path
    manifest_json: Path
    queries_json: Path
    region_samples_csv: Path | None
    point_samples_csv: Path | None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _region_mask_hash(mask: CoverageRegionMask, order: int) -> str:
    identity = {
        "grid_identity": HEALPIX_GRID_ID,
        "mask_version": mask.mask_version,
        "order": int(order),
        "provenance": mask.provenance,
        "region_id": mask.region_id,
    }
    cells = np.ascontiguousarray(np.asarray(mask.cell_indices, dtype="<i8"))
    digest = hashlib.sha256(_canonical_json(identity))
    digest.update(b"cell_indices")
    digest.update(cells.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(cells.shape)))
    digest.update(cells.tobytes(order="C"))
    return digest.hexdigest()


def _same_float_array(actual: Any, expected: Any) -> bool:
    return bool(
        np.array_equal(
            np.asarray(actual),
            np.asarray(expected),
            equal_nan=True,
        )
    )


def validate_global_coverage_product(result: CoverageProduct) -> int:
    """Fail closed unless a structural coverage product is internally coherent."""

    order = int(result.config.order)
    if order not in {5, 6, 7, 8}:
        raise ValueError("Coverage queries support only qualified HEALPix orders 5 through 8.")
    npix = healpix_npix(order)
    metrics = result.cell_metrics
    times = np.asarray(result.times_s, dtype=float)
    expected_cells = np.arange(npix, dtype=np.int64)
    supported_global_dispositions = {
        "global_earth",
        "global_earth_communications_service",
        "global_earth_constellation_aggregate",
    }
    if result.summary.get("status") != "complete":
        raise ValueError("Coverage products must have a complete source status.")
    if result.summary.get("domain_disposition") not in supported_global_dispositions:
        raise ValueError("Coverage queries require a completed global-Earth source disposition.")
    if not np.array_equal(metrics.cell_index, expected_cells):
        raise ValueError("Coverage queries require a complete canonical global cell product.")
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("Coverage queries require a valid global analysis time vector.")
    expected_identity = {
        "analysis_id": str(result.config.analysis_id),
        "grid_identity": HEALPIX_GRID_ID,
        "order": order,
        "sample_count": int(times.size),
        "horizon_start_s": float(times[0]),
        "horizon_end_s": float(times[-1]),
    }
    for field_name, expected_value in expected_identity.items():
        if result.summary.get(field_name) != expected_value:
            raise ValueError(f"Coverage summary field {field_name!r} is inconsistent with its product.")
    for config_field in ("service_id", "service_definition_id"):
        if hasattr(result.config, config_field) and result.summary.get(config_field) != getattr(
            result.config,
            config_field,
        ):
            raise ValueError(
                f"Coverage summary {config_field} is inconsistent with its configuration."
            )
    if result.covered_cell_count.shape != times.shape or result.instantaneous_covered_fraction.shape != times.shape:
        raise ValueError("Global covered-cell counts do not match the analysis time vector.")
    for field_name in (
        "dwell_s",
        "interval_count",
        "observed_acquisition_count",
        "max_complete_revisit_gap_s",
        "prefix_boundary_gap_s",
        "suffix_boundary_gap_s",
        "start_censored",
        "end_censored",
    ):
        if np.asarray(getattr(metrics, field_name)).shape != (npix,):
            raise ValueError(f"Global per-cell field {field_name!r} does not match the canonical grid.")
    if result.cell_geodetic_latitude_deg.shape != (npix,) or result.cell_longitude_deg.shape != (npix,):
        raise ValueError("Global cell-center coordinates do not match the canonical grid.")
    centers = healpix_wgs84_centers(order)
    if not np.allclose(
        result.cell_geodetic_latitude_deg,
        np.rad2deg(centers.geodetic_latitude_rad),
        rtol=0.0,
        atol=2.0e-12,
    ) or not np.allclose(
        result.cell_longitude_deg,
        np.rad2deg(centers.longitude_rad),
        rtol=0.0,
        atol=2.0e-12,
    ):
        raise ValueError("Global cell-center coordinates are not the canonical HEALPix centers.")

    sparse = metrics.intervals
    sparse_cells = np.asarray(sparse.cell_index)
    offsets = np.asarray(sparse.interval_offset)
    starts = np.asarray(sparse.start_sample_index)
    ends = np.asarray(sparse.end_sample_index_exclusive)
    if (
        sparse_cells.ndim != 1
        or sparse_cells.dtype.kind not in {"i", "u"}
        or offsets.shape != (sparse_cells.size + 1,)
        or offsets.dtype.kind not in {"i", "u"}
        or starts.ndim != 1
        or starts.dtype.kind not in {"i", "u"}
        or ends.shape != starts.shape
        or ends.dtype.kind not in {"i", "u"}
        or offsets.size == 0
        or int(offsets[0]) != 0
        or int(offsets[-1]) != starts.size
        or np.any(np.diff(offsets) <= 0)
        or np.any(sparse_cells < 0)
        or np.any(sparse_cells >= npix)
        or (sparse_cells.size > 1 and np.any(sparse_cells[1:] <= sparse_cells[:-1]))
        or np.any(starts < 0)
        or np.any(starts >= ends)
        or np.any(ends > times.size)
    ):
        raise ValueError("Global sparse coverage intervals are malformed.")
    for sparse_index in range(sparse_cells.size):
        begin = int(offsets[sparse_index])
        end = int(offsets[sparse_index + 1])
        if end - begin > 1 and np.any(starts[begin + 1 : end] <= ends[begin : end - 1]):
            raise ValueError("Global sparse coverage intervals must be disjoint and maximal per cell.")
    expected_interval_counts = np.zeros(npix, dtype=np.int64)
    expected_interval_counts[sparse_cells] = np.diff(offsets)
    if not np.array_equal(metrics.interval_count, expected_interval_counts):
        raise ValueError("Global sparse intervals are inconsistent with per-cell interval counts.")
    derived_count = _counts_from_sparse(sparse, times.size)
    if not np.array_equal(result.covered_cell_count, derived_count):
        raise ValueError("Global sparse intervals are inconsistent with sampled covered-cell counts.")
    expected_fraction = result.covered_cell_count.astype(float) / npix
    if not np.array_equal(result.instantaneous_covered_fraction, expected_fraction):
        raise ValueError("Global sampled fractions are inconsistent with covered-cell counts.")
    digest = str(result.interval_semantic_sha256 or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Coverage queries require a lowercase source interval SHA-256 digest.")

    delta_t = np.diff(times)
    cumulative_dwell = np.concatenate(([0.0], np.cumsum(delta_t)))
    expected_dwell = np.zeros(npix, dtype=float)
    expected_acquisitions = np.zeros(npix, dtype=np.int64)
    expected_revisit = np.full(npix, np.nan)
    expected_prefix = np.full(npix, times[-1] - times[0])
    expected_suffix = np.full(npix, times[-1] - times[0])
    expected_start_censored = np.zeros(npix, dtype=bool)
    expected_end_censored = np.zeros(npix, dtype=bool)
    for sparse_index, cell_value in enumerate(sparse_cells):
        cell = int(cell_value)
        begin = int(offsets[sparse_index])
        end = int(offsets[sparse_index + 1])
        cell_starts = starts[begin:end].astype(np.int64, copy=False)
        cell_ends = ends[begin:end].astype(np.int64, copy=False)
        dwell_ends = np.minimum(cell_ends, times.size - 1)
        expected_dwell[cell] = float(
            np.sum(cumulative_dwell[dwell_ends] - cumulative_dwell[cell_starts])
        )
        expected_start_censored[cell] = bool(cell_starts[0] == 0)
        expected_end_censored[cell] = bool(cell_ends[-1] == times.size)
        expected_acquisitions[cell] = cell_starts.size - int(expected_start_censored[cell])
        expected_prefix[cell] = (
            np.nan if expected_start_censored[cell] else float(times[cell_starts[0]] - times[0])
        )
        expected_suffix[cell] = (
            np.nan
            if expected_end_censored[cell]
            else float(times[-1] - times[cell_ends[-1]])
        )
        if cell_starts.size > 1:
            expected_revisit[cell] = float(
                np.max(times[cell_starts[1:]] - times[cell_ends[:-1]])
            )
    expected_fields = {
        "dwell_s": expected_dwell,
        "observed_acquisition_count": expected_acquisitions,
        "max_complete_revisit_gap_s": expected_revisit,
        "prefix_boundary_gap_s": expected_prefix,
        "suffix_boundary_gap_s": expected_suffix,
        "start_censored": expected_start_censored,
        "end_censored": expected_end_censored,
    }
    for field_name, expected in expected_fields.items():
        if not _same_float_array(getattr(metrics, field_name), expected):
            raise ValueError(f"Global per-cell field {field_name!r} is inconsistent with sparse intervals.")
    return npix


_validate_source = validate_global_coverage_product


def _subset_metrics(metrics: CoverageCellMetrics, cells: np.ndarray) -> CoverageCellMetrics:
    counts = np.asarray(metrics.interval_count[cells], dtype=np.int64)
    covered_cells = cells[counts > 0]
    source_sparse = metrics.intervals
    positions = np.searchsorted(source_sparse.cell_index, covered_cells)
    if covered_cells.size and (
        np.any(positions >= source_sparse.cell_index.size)
        or not np.array_equal(source_sparse.cell_index[positions], covered_cells)
    ):
        raise ValueError("Global sparse intervals are inconsistent with the per-cell metrics.")

    interval_counts = counts[counts > 0]
    if covered_cells.size and not np.array_equal(
        np.diff(source_sparse.interval_offset)[positions],
        interval_counts,
    ):
        raise ValueError("Global sparse interval offsets are inconsistent with the per-cell metrics.")
    offsets = np.zeros(covered_cells.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(interval_counts, dtype=np.int64)
    start_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    for position in positions:
        start = int(source_sparse.interval_offset[position])
        stop = int(source_sparse.interval_offset[position + 1])
        start_parts.append(source_sparse.start_sample_index[start:stop])
        end_parts.append(source_sparse.end_sample_index_exclusive[start:stop])
    starts = np.concatenate(start_parts) if start_parts else np.empty(0, dtype=np.int64)
    ends = np.concatenate(end_parts) if end_parts else np.empty(0, dtype=np.int64)
    sparse = SparseCoverageIntervals(
        cell_index=np.asarray(covered_cells, dtype="<i8"),
        interval_offset=np.asarray(offsets, dtype="<i8"),
        start_sample_index=np.asarray(starts, dtype="<i8"),
        end_sample_index_exclusive=np.asarray(ends, dtype="<i8"),
    )
    return CoverageCellMetrics(
        cell_index=cells.copy(),
        dwell_s=np.asarray(metrics.dwell_s[cells], dtype=float),
        interval_count=counts,
        observed_acquisition_count=np.asarray(metrics.observed_acquisition_count[cells], dtype=np.int64),
        max_complete_revisit_gap_s=np.asarray(metrics.max_complete_revisit_gap_s[cells], dtype=float),
        prefix_boundary_gap_s=np.asarray(metrics.prefix_boundary_gap_s[cells], dtype=float),
        suffix_boundary_gap_s=np.asarray(metrics.suffix_boundary_gap_s[cells], dtype=float),
        start_censored=np.asarray(metrics.start_censored[cells], dtype=bool),
        end_censored=np.asarray(metrics.end_censored[cells], dtype=bool),
        intervals=sparse,
    )


def _counts_from_sparse(intervals: SparseCoverageIntervals, sample_count: int) -> np.ndarray:
    delta = np.zeros(sample_count + 1, dtype=np.int64)
    np.add.at(delta, intervals.start_sample_index, 1)
    np.add.at(delta, intervals.end_sample_index_exclusive, -1)
    return np.cumsum(delta[:-1], dtype=np.int64)


def _region_summary(
    result: CoverageProduct,
    mask: CoverageRegionMask,
    mask_hash: str,
    metrics: CoverageCellMetrics,
    covered_count: np.ndarray,
) -> dict[str, Any]:
    cell_count = int(metrics.cell_index.size)
    cell_area = WGS84_SURFACE_AREA_KM2 / healpix_npix(result.config.order)
    fraction = covered_count.astype(float) / cell_count
    duration = float(result.times_s[-1] - result.times_s[0])
    time_weighted = float(np.dot(fraction[:-1], np.diff(result.times_s)) / duration)
    ever_covered = metrics.interval_count > 0
    finite_revisit = np.isfinite(metrics.max_complete_revisit_gap_s)
    revisit = metrics.max_complete_revisit_gap_s[finite_revisit]
    return {
        "contract_version": GLOBAL_COVERAGE_CONTRACT_VERSION,
        "query_schema_version": COVERAGE_QUERY_SCHEMA_VERSION,
        "source_analysis_id": result.config.analysis_id,
        "domain_disposition": "region_query",
        "region_id": mask.region_id,
        "mask_version": mask.mask_version,
        "mask_provenance": mask.provenance,
        "mask_semantic_sha256": mask_hash,
        "grid_identity": HEALPIX_GRID_ID,
        "order": int(result.config.order),
        "cell_count": cell_count,
        "cell_area_km2": float(cell_area),
        "region_area_km2": float(cell_count * cell_area),
        "sample_count": int(result.times_s.size),
        "horizon_start_s": float(result.times_s[0]),
        "horizon_end_s": float(result.times_s[-1]),
        "horizon_duration_s": duration,
        "instantaneous_covered_fraction_min": float(np.min(fraction)),
        "instantaneous_covered_fraction_max": float(np.max(fraction)),
        "instantaneous_covered_area_km2_min": float(np.min(covered_count) * cell_area),
        "instantaneous_covered_area_km2_max": float(np.max(covered_count) * cell_area),
        "time_weighted_mean_covered_fraction": time_weighted,
        "time_weighted_mean_covered_area_km2": float(time_weighted * cell_count * cell_area),
        "ever_covered_cell_count": int(np.count_nonzero(ever_covered)),
        "never_covered_cell_count": int(np.count_nonzero(~ever_covered)),
        "ever_covered_fraction": float(np.count_nonzero(ever_covered) / cell_count),
        "never_covered_fraction": float(np.count_nonzero(~ever_covered) / cell_count),
        "dwell_s": {
            "minimum": float(np.min(metrics.dwell_s)),
            "mean": float(np.mean(metrics.dwell_s)),
            "maximum": float(np.max(metrics.dwell_s)),
            "included_cell_count": cell_count,
        },
        "max_complete_revisit_gap_s": {
            "minimum": None if not revisit.size else float(np.min(revisit)),
            "mean": None if not revisit.size else float(np.mean(revisit)),
            "maximum": None if not revisit.size else float(np.max(revisit)),
            "evaluated_cell_count": int(revisit.size),
            "not_evaluated_cell_count": int(cell_count - revisit.size),
            "disposition": "not_evaluated" if not revisit.size else "evaluated",
        },
        "start_censored_cell_count": int(np.count_nonzero(metrics.start_censored)),
        "end_censored_cell_count": int(np.count_nonzero(metrics.end_censored)),
        "claim_limits": [
            "This is an aggregation of declared cells and cannot support a global-Earth claim.",
            "Coverage is sampled at canonical cell centers; no subcell or overlap-area claim is made.",
        ],
    }


def _optional_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _point_summary(
    result: CoverageProduct,
    query: CoveragePointQuery,
    cell_index: int,
    metrics: CoverageCellMetrics,
) -> dict[str, Any]:
    return {
        "contract_version": GLOBAL_COVERAGE_CONTRACT_VERSION,
        "query_schema_version": COVERAGE_QUERY_SCHEMA_VERSION,
        "source_analysis_id": result.config.analysis_id,
        "domain_disposition": "point_cell_query",
        "point_id": query.point_id,
        "longitude_deg": query.longitude_deg,
        "geodetic_latitude_deg": query.geodetic_latitude_deg,
        "ellipsoidal_height_km": 0.0,
        "grid_identity": HEALPIX_GRID_ID,
        "order": int(result.config.order),
        "cell_index": int(cell_index),
        "resolution_dependent_cell_result": True,
        "ever_covered": bool(metrics.interval_count[0] > 0),
        "sampled_dwell_s": float(metrics.dwell_s[0]),
        "interval_count": int(metrics.interval_count[0]),
        "observed_acquisition_count": int(metrics.observed_acquisition_count[0]),
        "max_complete_revisit_gap_s": _optional_float(metrics.max_complete_revisit_gap_s[0]),
        "prefix_boundary_gap_s": _optional_float(metrics.prefix_boundary_gap_s[0]),
        "suffix_boundary_gap_s": _optional_float(metrics.suffix_boundary_gap_s[0]),
        "start_censored": bool(metrics.start_censored[0]),
        "end_censored": bool(metrics.end_censored[0]),
        "claim_limits": [
            "The query inherits its containing cell's result at the declared order.",
            "It does not prove exact subcell point visibility or between-sample access.",
        ],
    }


def _query_hash(
    result: CoverageProduct,
    regions: Iterable[CoverageRegionResult],
    points: Iterable[CoveragePointResult],
) -> str:
    identity = {
        "contract_version": GLOBAL_COVERAGE_CONTRACT_VERSION,
        "grid_identity": HEALPIX_GRID_ID,
        "order": int(result.config.order),
        "points": [
            {
                "cell_index": point.cell_index,
                "geodetic_latitude_deg": point.query.geodetic_latitude_deg,
                "longitude_deg": point.query.longitude_deg,
                "point_id": point.query.point_id,
            }
            for point in points
        ],
        "query_schema_version": COVERAGE_QUERY_SCHEMA_VERSION,
        "regions": [
            {
                "mask_semantic_sha256": region.mask_semantic_sha256,
                "region_id": region.mask.region_id,
            }
            for region in regions
        ],
        "source_analysis_id": result.config.analysis_id,
        "source_interval_semantic_sha256": result.interval_semantic_sha256,
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def evaluate_coverage_queries(
    result: CoverageProduct,
    *,
    region_masks: Iterable[CoverageRegionMask] = (),
    points: Iterable[CoveragePointQuery] = (),
) -> CoverageQueryResult:
    """Evaluate Phase 2 queries without rerunning propagation or geometry."""

    npix = validate_global_coverage_product(result)
    masks = sorted(tuple(region_masks), key=lambda item: item.region_id)
    point_queries = sorted(tuple(points), key=lambda item: item.point_id)
    if not masks and not point_queries:
        raise ValueError("At least one region mask or point query is required.")
    identifiers = [mask.region_id for mask in masks] + [point.point_id for point in point_queries]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Coverage region and point query identifiers must be unique.")

    region_results: list[CoverageRegionResult] = []
    cell_area = WGS84_SURFACE_AREA_KM2 / npix
    for mask in masks:
        cells = np.asarray(mask.cell_indices, dtype=np.int64)
        if int(cells[-1]) >= npix:
            raise ValueError(
                f"Region {mask.region_id!r} contains a cell outside [0, {npix}) for order {result.config.order}."
            )
        metrics = _subset_metrics(result.cell_metrics, cells)
        covered_count = _counts_from_sparse(metrics.intervals, result.times_s.size)
        mask_hash = _region_mask_hash(mask, result.config.order)
        region_results.append(
            CoverageRegionResult(
                mask=mask,
                mask_semantic_sha256=mask_hash,
                covered_cell_count=covered_count,
                instantaneous_covered_fraction=covered_count.astype(float) / cells.size,
                covered_area_km2=covered_count.astype(float) * cell_area,
                cell_metrics=metrics,
                summary=_region_summary(result, mask, mask_hash, metrics, covered_count),
            )
        )

    point_results: list[CoveragePointResult] = []
    for query in point_queries:
        cell_index = int(
            wgs84_points_to_healpix_nested(
                result.config.order,
                query.geodetic_latitude_deg,
                query.longitude_deg,
            ).item()
        )
        cells = np.asarray([cell_index], dtype=np.int64)
        metrics = _subset_metrics(result.cell_metrics, cells)
        covered = _counts_from_sparse(metrics.intervals, result.times_s.size).astype(bool)
        point_results.append(
            CoveragePointResult(
                query=query,
                cell_index=cell_index,
                cell_geodetic_latitude_deg=float(result.cell_geodetic_latitude_deg[cell_index]),
                cell_longitude_deg=float(result.cell_longitude_deg[cell_index]),
                covered_by_sample=covered,
                cell_metrics=metrics,
                summary=_point_summary(result, query, cell_index, metrics),
            )
        )

    region_tuple = tuple(region_results)
    point_tuple = tuple(point_results)
    return CoverageQueryResult(
        source_analysis_id=result.config.analysis_id,
        source_interval_semantic_sha256=result.interval_semantic_sha256,
        order=int(result.config.order),
        times_s=result.times_s.copy(),
        regions=region_tuple,
        points=point_tuple,
        query_semantic_sha256=_query_hash(result, region_tuple, point_tuple),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_coverage_query_artifacts(
    result: CoverageQueryResult,
    output_dir: str | Path,
) -> CoverageQueryArtifacts:
    """Write deterministic Phase 2 query definitions, summaries, and samples."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Coverage query output directory already exists: {destination}")
    destination.mkdir(parents=True)
    queries_path = destination / "coverage_queries.json"
    manifest_path = destination / "coverage_query_manifest.json"
    region_samples_path = destination / "coverage_region_samples.csv" if result.regions else None
    point_samples_path = destination / "coverage_point_samples.csv" if result.points else None

    queries_record = {
        "query_schema_version": COVERAGE_QUERY_SCHEMA_VERSION,
        "source_analysis_id": result.source_analysis_id,
        "source_interval_semantic_sha256": result.source_interval_semantic_sha256,
        "grid_identity": HEALPIX_GRID_ID,
        "order": result.order,
        "query_semantic_sha256": result.query_semantic_sha256,
        "regions": [
            {
                "region_id": region.mask.region_id,
                "mask_version": region.mask.mask_version,
                "provenance": region.mask.provenance,
                "cell_indices": list(region.mask.cell_indices),
                "mask_semantic_sha256": region.mask_semantic_sha256,
                "summary": region.summary,
            }
            for region in result.regions
        ],
        "points": [
            {
                "point_id": point.query.point_id,
                "longitude_deg": point.query.longitude_deg,
                "geodetic_latitude_deg": point.query.geodetic_latitude_deg,
                "cell_index": point.cell_index,
                "cell_center_geodetic_latitude_deg": point.cell_geodetic_latitude_deg,
                "cell_center_longitude_deg": point.cell_longitude_deg,
                "summary": point.summary,
            }
            for point in result.points
        ],
    }
    _write_json(queries_path, queries_record)

    if region_samples_path is not None:
        with region_samples_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "region_id",
                    "sample_index",
                    "time_s",
                    "covered_cell_count",
                    "instantaneous_covered_fraction",
                    "covered_area_km2",
                )
            )
            for region in result.regions:
                for sample_index, time_s in enumerate(result.times_s):
                    writer.writerow(
                        (
                            region.mask.region_id,
                            sample_index,
                            f"{float(time_s):.17g}",
                            int(region.covered_cell_count[sample_index]),
                            f"{float(region.instantaneous_covered_fraction[sample_index]):.17g}",
                            f"{float(region.covered_area_km2[sample_index]):.17g}",
                        )
                    )

    if point_samples_path is not None:
        with point_samples_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("point_id", "cell_index", "sample_index", "time_s", "covered"))
            for point in result.points:
                for sample_index, time_s in enumerate(result.times_s):
                    writer.writerow(
                        (
                            point.query.point_id,
                            point.cell_index,
                            sample_index,
                            f"{float(time_s):.17g}",
                            str(bool(point.covered_by_sample[sample_index])).lower(),
                        )
                    )

    artifacts: dict[str, dict[str, str]] = {
        queries_path.name: {"sha256": _sha256_file(queries_path)},
    }
    if region_samples_path is not None:
        artifacts[region_samples_path.name] = {"sha256": _sha256_file(region_samples_path)}
    if point_samples_path is not None:
        artifacts[point_samples_path.name] = {"sha256": _sha256_file(point_samples_path)}
    _write_json(
        manifest_path,
        {
            "query_schema_version": COVERAGE_QUERY_SCHEMA_VERSION,
            "source_analysis_id": result.source_analysis_id,
            "source_interval_semantic_sha256": result.source_interval_semantic_sha256,
            "grid_identity": HEALPIX_GRID_ID,
            "order": result.order,
            "query_semantic_sha256": result.query_semantic_sha256,
            "region_count": len(result.regions),
            "point_count": len(result.points),
            "artifacts": artifacts,
            "status": "complete",
        },
    )
    return CoverageQueryArtifacts(
        output_dir=destination,
        manifest_json=manifest_path,
        queries_json=queries_path,
        region_samples_csv=region_samples_path,
        point_samples_csv=point_samples_path,
    )


__all__ = [
    "COVERAGE_QUERY_SCHEMA_VERSION",
    "CoveragePointQuery",
    "CoveragePointResult",
    "CoverageQueryArtifacts",
    "CoverageQueryResult",
    "CoverageRegionMask",
    "CoverageRegionResult",
    "evaluate_coverage_queries",
    "validate_global_coverage_product",
    "write_coverage_query_artifacts",
]
