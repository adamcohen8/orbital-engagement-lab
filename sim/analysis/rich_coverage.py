"""Phase 3 rich Earth sensor coverage from deterministic OEL evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.analysis.global_coverage import (
    CoverageCellMetrics,
    SparseCoverageIntervals,
    _combine_chunk_metrics,
    _validated_evidence,
    _validated_quaternion,
    summarize_sampled_coverage_mask,
)
from sim.analysis.healpix import (
    HEALPIX_GRID_ID,
    WGS84_AUTHALIC_RADIUS_KM,
    WGS84_SURFACE_AREA_KM2,
    healpix_npix,
    healpix_wgs84_centers,
)
from sim.analysis.sensor_footprint_geometry import (
    PRIMARY_REASON_NAMES,
    HardFOVPattern,
    SurfaceServiceConstraints,
    evaluate_rich_surface_targets_ecef,
    fov_boundary_rays_sensor,
    intersect_rays_wgs84,
)
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM, ecef_to_geodetic_deg_km
from sim.utils.quaternion import quaternion_to_dcm_bn

RICH_COVERAGE_CONTRACT_VERSION = "oel.rich-earth-coverage-analysis.v0.1"
BOUNDARY_DISPOSITION_NAMES = ("complete", "partial", "no_intersection")

_SUPPORTED_ORDERS = frozenset(range(5, 9))
_ATTITUDE_SOURCE_KINDS = frozenset({"achieved", "replay", "analytic_ideal"})
_ANGULAR_TOLERANCE_RAD = 1.0e-12
_RANGE_TOLERANCE_KM = 1.0e-9


@dataclass(frozen=True)
class RichCoverageConfig:
    analysis_id: str
    source_asset_id: str
    state_provider_id: str
    attitude_source_kind: str
    attitude_provider_id: str
    sensor_id: str
    order: int
    quat_body_from_sensor: tuple[float, float, float, float]
    pattern: HardFOVPattern
    constraints: SurfaceServiceConstraints = SurfaceServiceConstraints()
    max_range_km: float | None = None
    sun_provider_id: str | None = None
    boundary_samples_per_edge: int = 16
    chunk_size: int = 8192
    max_working_memory_bytes: int = 512 * 1024 * 1024
    max_cell_time_comparisons: int = 300_000_000

    def __post_init__(self) -> None:
        for field_name in (
            "analysis_id",
            "source_asset_id",
            "state_provider_id",
            "attitude_provider_id",
            "sensor_id",
        ):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, value)
        source_kind = str(self.attitude_source_kind or "").strip().lower()
        if source_kind not in _ATTITUDE_SOURCE_KINDS:
            choices = ", ".join(sorted(_ATTITUDE_SOURCE_KINDS))
            raise ValueError(f"attitude_source_kind must be one of: {choices}.")
        object.__setattr__(self, "attitude_source_kind", source_kind)
        if isinstance(self.order, (bool, np.bool_)) or int(self.order) != self.order:
            raise ValueError("Rich coverage order must be an integer.")
        if int(self.order) not in _SUPPORTED_ORDERS:
            raise ValueError("Rich coverage v0.1 supports HEALPix orders 5 through 8.")
        object.__setattr__(self, "order", int(self.order))
        mounting = _validated_quaternion(self.quat_body_from_sensor, "quat_body_from_sensor")
        object.__setattr__(self, "quat_body_from_sensor", tuple(float(value) for value in mounting))
        if not isinstance(self.pattern, HardFOVPattern):
            raise ValueError("pattern must be a validated HardFOVPattern.")
        if not isinstance(self.constraints, SurfaceServiceConstraints):
            raise ValueError("constraints must be validated SurfaceServiceConstraints.")
        if self.max_range_km is not None:
            maximum_range = float(self.max_range_km)
            if not np.isfinite(maximum_range) or maximum_range <= 0.0:
                raise ValueError("max_range_km must be positive and finite when provided.")
            object.__setattr__(self, "max_range_km", maximum_range)
        provider = None if self.sun_provider_id is None else str(self.sun_provider_id).strip()
        if self.constraints.illumination_enabled and not provider:
            raise ValueError("sun_provider_id is required when illumination constraints are enabled.")
        object.__setattr__(self, "sun_provider_id", provider or None)
        for field_name, minimum, maximum in (
            ("boundary_samples_per_edge", 2, 4096),
            ("chunk_size", 1, None),
            ("max_working_memory_bytes", 1, None),
            ("max_cell_time_comparisons", 1, None),
        ):
            value = getattr(self, field_name)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must be an integer.") from exc
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(numeric)
                or not numeric.is_integer()
                or numeric < minimum
                or (maximum is not None and numeric > maximum)
            ):
                if maximum is None:
                    raise ValueError(f"{field_name} must be a positive integer.")
                raise ValueError(f"{field_name} must be an integer within [{minimum}, {maximum}].")
            object.__setattr__(self, field_name, int(numeric))


@dataclass(frozen=True)
class FootprintBoundaryEvidence:
    subsatellite_geodetic_latitude_deg: np.ndarray
    subsatellite_longitude_deg: np.ndarray
    boresight_off_nadir_rad: np.ndarray
    boundary_hit: np.ndarray
    boundary_geodetic_latitude_deg: np.ndarray
    boundary_longitude_deg: np.ndarray
    boundary_disposition_code: np.ndarray

    @property
    def ray_count(self) -> int:
        return int(self.boundary_hit.shape[1])


@dataclass(frozen=True)
class RichCoverageResult:
    config: RichCoverageConfig
    frame_metadata: dict[str, Any]
    times_s: np.ndarray
    covered_cell_count: np.ndarray
    instantaneous_covered_fraction: np.ndarray
    cell_geodetic_latitude_deg: np.ndarray
    cell_longitude_deg: np.ndarray
    cell_metrics: CoverageCellMetrics
    primary_reason_count: np.ndarray
    footprint_boundary: FootprintBoundaryEvidence
    summary: dict[str, Any]
    resource_estimate: dict[str, int]
    input_evidence_sha256: str
    interval_semantic_sha256: str


@dataclass(frozen=True)
class RichCoverageArtifacts:
    output_dir: Path
    manifest_json: Path
    summary_json: Path
    samples_csv: Path
    cells_csv: Path | None
    intervals_npz: Path
    footprints_npz: Path
    footprint_plot_png: Path | None


def estimate_rich_coverage_resources(
    config: RichCoverageConfig,
    sample_count: int,
) -> dict[str, int]:
    samples = int(sample_count)
    if samples < 2:
        raise ValueError("sample_count must be at least two.")
    npix = healpix_npix(config.order)
    chunk_cells = min(config.chunk_size, npix)
    dense_comparisons = npix * samples
    boundary_rays = 4 * config.boundary_samples_per_edge
    chunk_boolean_bytes = chunk_cells * samples
    chunk_geometry_bytes = chunk_cells * 20 * 8
    chunk_gate_bytes = chunk_cells * 8
    full_metric_bytes = npix * (8 * 8 + 2)
    input_state_bytes = samples * (3 + 4 + 3 + 9) * 8
    boundary_bytes = samples * boundary_rays * (2 * 8 + 1)
    reason_bytes = samples * len(PRIMARY_REASON_NAMES) * 8
    worst_case_interval_count = npix * ((samples + 1) // 2)
    worst_case_interval_working_bytes = worst_case_interval_count * 16 * 2
    estimated_peak_bytes = (
        chunk_boolean_bytes
        + chunk_geometry_bytes
        + chunk_gate_bytes
        + full_metric_bytes
        + input_state_bytes
        + boundary_bytes
        + reason_bytes
        + worst_case_interval_working_bytes
    )
    return {
        "sample_count": samples,
        "cell_count": npix,
        "cell_time_comparisons": dense_comparisons,
        "chunk_size": chunk_cells,
        "boundary_ray_count": boundary_rays,
        "boundary_ray_intersections": boundary_rays * samples,
        "worst_case_interval_count": worst_case_interval_count,
        "estimated_peak_working_bytes": estimated_peak_bytes,
        "max_working_memory_bytes": config.max_working_memory_bytes,
        "max_cell_time_comparisons": config.max_cell_time_comparisons,
    }


def _validated_sun_evidence(
    config: RichCoverageConfig,
    sample_count: int,
    sun_positions_eci_km: np.ndarray | None,
) -> np.ndarray | None:
    if sun_positions_eci_km is None:
        if config.constraints.illumination_enabled:
            raise ValueError("Explicit Sun ECI positions are required by illumination constraints.")
        if config.sun_provider_id:
            raise ValueError("Sun ECI positions are required when sun_provider_id is declared.")
        return None
    sun = np.asarray(sun_positions_eci_km, dtype=float)
    if sun.shape != (sample_count, 3) or not np.all(np.isfinite(sun)):
        raise ValueError("sun_positions_eci_km must be finite with shape (samples, 3).")
    if not config.sun_provider_id:
        raise ValueError("sun_provider_id is required when Sun evidence is supplied.")
    return sun


def _prepare_ecef_evidence(
    config: RichCoverageConfig,
    times: np.ndarray,
    positions_eci: np.ndarray,
    attitudes_quat_bn: np.ndarray,
    sun_positions_eci: np.ndarray | None,
    frame_context: FrameContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    sample_count = times.size
    positions_ecef = np.empty_like(positions_eci)
    sensor_from_ecef = np.empty((sample_count, 3, 3), dtype=float)
    sun_ecef = None if sun_positions_eci is None else np.empty_like(sun_positions_eci)
    subpoint_latitude = np.empty(sample_count, dtype=float)
    subpoint_longitude = np.empty(sample_count, dtype=float)
    boresight_off_nadir = np.empty(sample_count, dtype=float)
    body_from_sensor = quaternion_to_dcm_bn(np.asarray(config.quat_body_from_sensor, dtype=float))
    sensor_from_body = body_from_sensor.T
    sensor_boresight = np.array([0.0, 0.0, 1.0], dtype=float)
    for sample_index, time_s in enumerate(times):
        ecef_from_eci = eci_to_ecef_rotation_context(float(time_s), frame_context)
        body_from_eci = quaternion_to_dcm_bn(attitudes_quat_bn[sample_index])
        sensor_from_ecef[sample_index] = sensor_from_body @ body_from_eci @ ecef_from_eci.T
        positions_ecef[sample_index] = ecef_from_eci @ positions_eci[sample_index]
        if sun_ecef is not None and sun_positions_eci is not None:
            sun_ecef[sample_index] = ecef_from_eci @ sun_positions_eci[sample_index]
        latitude, longitude, _ = ecef_to_geodetic_deg_km(positions_ecef[sample_index])
        subpoint_latitude[sample_index] = latitude
        subpoint_longitude[sample_index] = (longitude + 180.0) % 360.0 - 180.0
        outward = np.array(
            [
                positions_ecef[sample_index, 0] / (WGS84_A_KM**2),
                positions_ecef[sample_index, 1] / (WGS84_A_KM**2),
                positions_ecef[sample_index, 2] / (WGS84_B_KM**2),
            ],
            dtype=float,
        )
        outward /= np.linalg.norm(outward)
        boresight_ecef = sensor_from_ecef[sample_index].T @ sensor_boresight
        boresight_off_nadir[sample_index] = np.arccos(
            np.clip(float(np.dot(boresight_ecef, -outward)), -1.0, 1.0)
        )
    ellipsoid_level = (
        (positions_ecef[:, 0] / WGS84_A_KM) ** 2
        + (positions_ecef[:, 1] / WGS84_A_KM) ** 2
        + (positions_ecef[:, 2] / WGS84_B_KM) ** 2
    )
    invalid = np.flatnonzero(ellipsoid_level <= 1.0)
    if invalid.size:
        raise ValueError(
            f"Source spacecraft must be outside WGS84; sample {int(invalid[0])} is on or inside the ellipsoid."
        )
    return (
        positions_ecef,
        sensor_from_ecef,
        sun_ecef,
        subpoint_latitude,
        subpoint_longitude,
        boresight_off_nadir,
    )


def _footprint_boundary(
    config: RichCoverageConfig,
    positions_ecef: np.ndarray,
    sensor_from_ecef: np.ndarray,
    subpoint_latitude: np.ndarray,
    subpoint_longitude: np.ndarray,
    boresight_off_nadir: np.ndarray,
) -> FootprintBoundaryEvidence:
    rays_sensor = fov_boundary_rays_sensor(
        config.pattern,
        samples_per_edge=config.boundary_samples_per_edge,
    )
    sample_count = positions_ecef.shape[0]
    ray_count = rays_sensor.shape[0]
    hits = np.zeros((sample_count, ray_count), dtype=bool)
    latitudes = np.full((sample_count, ray_count), np.nan, dtype=float)
    longitudes = np.full((sample_count, ray_count), np.nan, dtype=float)
    disposition = np.empty(sample_count, dtype=np.uint8)
    for sample_index in range(sample_count):
        directions_ecef = rays_sensor @ sensor_from_ecef[sample_index]
        intersections = intersect_rays_wgs84(
            positions_ecef[sample_index],
            directions_ecef,
        )
        hits[sample_index] = intersections.hit
        for ray_index in np.flatnonzero(intersections.hit):
            latitude, longitude, _ = ecef_to_geodetic_deg_km(
                intersections.point_ecef_km[ray_index]
            )
            latitudes[sample_index, ray_index] = latitude
            longitudes[sample_index, ray_index] = (longitude + 180.0) % 360.0 - 180.0
        hit_count = int(np.count_nonzero(intersections.hit))
        if hit_count == ray_count:
            disposition[sample_index] = 0
        elif hit_count:
            disposition[sample_index] = 1
        else:
            disposition[sample_index] = 2
    return FootprintBoundaryEvidence(
        subsatellite_geodetic_latitude_deg=subpoint_latitude,
        subsatellite_longitude_deg=subpoint_longitude,
        boresight_off_nadir_rad=boresight_off_nadir,
        boundary_hit=hits,
        boundary_geodetic_latitude_deg=latitudes,
        boundary_longitude_deg=longitudes,
        boundary_disposition_code=disposition,
    )


def _normalized_scientific_config(config: RichCoverageConfig) -> dict[str, Any]:
    return {
        "analysis_id": config.analysis_id,
        "attitude_provider_id": config.attitude_provider_id,
        "attitude_source_kind": config.attitude_source_kind,
        "boundary_samples_per_edge": config.boundary_samples_per_edge,
        "constraints": asdict(config.constraints),
        "contract_version": RICH_COVERAGE_CONTRACT_VERSION,
        "grid_identity": HEALPIX_GRID_ID,
        "max_range_km": config.max_range_km,
        "order": config.order,
        "pattern": asdict(config.pattern),
        "quat_body_from_sensor": list(config.quat_body_from_sensor),
        "sensor_id": config.sensor_id,
        "source_asset_id": config.source_asset_id,
        "state_provider_id": config.state_provider_id,
        "sun_provider_id": config.sun_provider_id,
    }


def _hash_arrays(identity: dict[str, Any], arrays: tuple[tuple[str, Any, str], ...]) -> str:
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    for name, values, dtype in arrays:
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _input_evidence_hash(
    times: np.ndarray,
    positions_eci: np.ndarray,
    attitudes: np.ndarray,
    sun_positions_eci: np.ndarray | None,
    frame_metadata: dict[str, Any],
) -> str:
    arrays: list[tuple[str, Any, str]] = [
        ("times_s", times, "<f8"),
        ("positions_eci_km", positions_eci, "<f8"),
        ("attitudes_quat_bn", attitudes, "<f8"),
    ]
    if sun_positions_eci is not None:
        arrays.append(("sun_positions_eci_km", sun_positions_eci, "<f8"))
    return _hash_arrays(
        {
            "frame": frame_metadata,
            "schema": "oel.rich-coverage-input-evidence.v1",
        },
        tuple(arrays),
    )


def _semantic_hash(
    config: RichCoverageConfig,
    input_evidence_sha256: str,
    times: np.ndarray,
    intervals: SparseCoverageIntervals,
    primary_reason_count: np.ndarray,
    footprint: FootprintBoundaryEvidence,
) -> str:
    identity = {
        **_normalized_scientific_config(config),
        "input_evidence_sha256": input_evidence_sha256,
    }
    return _hash_arrays(
        identity,
        (
            ("times_s", times, "<f8"),
            ("cell_index", intervals.cell_index, "<i8"),
            ("interval_offset", intervals.interval_offset, "<i8"),
            ("start_sample_index", intervals.start_sample_index, "<i8"),
            ("end_sample_index_exclusive", intervals.end_sample_index_exclusive, "<i8"),
            ("primary_reason_count", primary_reason_count, "<i8"),
            (
                "subsatellite_geodetic_latitude_deg",
                footprint.subsatellite_geodetic_latitude_deg,
                "<f8",
            ),
            ("subsatellite_longitude_deg", footprint.subsatellite_longitude_deg, "<f8"),
            ("boresight_off_nadir_rad", footprint.boresight_off_nadir_rad, "<f8"),
            ("boundary_hit", footprint.boundary_hit, "|u1"),
            (
                "boundary_geodetic_latitude_deg",
                footprint.boundary_geodetic_latitude_deg,
                "<f8",
            ),
            ("boundary_longitude_deg", footprint.boundary_longitude_deg, "<f8"),
            ("boundary_disposition_code", footprint.boundary_disposition_code, "|u1"),
        ),
    )


def _summary(
    config: RichCoverageConfig,
    times: np.ndarray,
    covered_count: np.ndarray,
    metrics: CoverageCellMetrics,
    reason_count: np.ndarray,
    footprint: FootprintBoundaryEvidence,
) -> dict[str, Any]:
    npix = healpix_npix(config.order)
    cell_area = WGS84_SURFACE_AREA_KM2 / npix
    fraction = covered_count.astype(float) / npix
    duration = float(times[-1] - times[0])
    ever_covered = metrics.interval_count > 0
    finite_revisit = np.isfinite(metrics.max_complete_revisit_gap_s)
    revisit = metrics.max_complete_revisit_gap_s[finite_revisit]
    reason_totals = np.sum(reason_count, axis=0, dtype=np.int64)
    boundary_counts = np.bincount(
        footprint.boundary_disposition_code,
        minlength=len(BOUNDARY_DISPOSITION_NAMES),
    )
    return {
        "contract_version": RICH_COVERAGE_CONTRACT_VERSION,
        "analysis_id": config.analysis_id,
        "status": "complete",
        "domain_disposition": "global_earth",
        "source_asset_id": config.source_asset_id,
        "sensor_id": config.sensor_id,
        "attitude_source_kind": config.attitude_source_kind,
        "sun_provider_id": config.sun_provider_id,
        "grid_identity": HEALPIX_GRID_ID,
        "order": config.order,
        "pattern": asdict(config.pattern),
        "constraints": asdict(config.constraints),
        "cell_count": npix,
        "cell_area_km2": float(cell_area),
        "wgs84_surface_area_km2": WGS84_SURFACE_AREA_KM2,
        "authalic_radius_km": WGS84_AUTHALIC_RADIUS_KM,
        "sample_count": int(times.size),
        "horizon_start_s": float(times[0]),
        "horizon_end_s": float(times[-1]),
        "horizon_duration_s": duration,
        "instantaneous_covered_fraction_min": float(np.min(fraction)),
        "instantaneous_covered_fraction_max": float(np.max(fraction)),
        "time_weighted_mean_covered_fraction": float(
            np.dot(fraction[:-1], np.diff(times)) / duration
        ),
        "ever_covered_cell_count": int(np.count_nonzero(ever_covered)),
        "never_covered_cell_count": int(np.count_nonzero(~ever_covered)),
        "ever_covered_fraction": float(np.count_nonzero(ever_covered) / npix),
        "never_covered_fraction": float(np.count_nonzero(~ever_covered) / npix),
        "dwell_s": {
            "minimum": float(np.min(metrics.dwell_s)),
            "mean": float(np.mean(metrics.dwell_s)),
            "maximum": float(np.max(metrics.dwell_s)),
            "included_cell_count": npix,
        },
        "max_complete_revisit_gap_s": {
            "minimum": None if not revisit.size else float(np.min(revisit)),
            "mean": None if not revisit.size else float(np.mean(revisit)),
            "maximum": None if not revisit.size else float(np.max(revisit)),
            "evaluated_cell_count": int(revisit.size),
            "not_evaluated_cell_count": int(npix - revisit.size),
            "disposition": "not_evaluated" if not revisit.size else "evaluated",
        },
        "start_censored_cell_count": int(np.count_nonzero(metrics.start_censored)),
        "end_censored_cell_count": int(np.count_nonzero(metrics.end_censored)),
        "primary_reason_total": {
            name: int(reason_totals[index]) for index, name in enumerate(PRIMARY_REASON_NAMES)
        },
        "boundary_disposition_sample_count": {
            name: int(boundary_counts[index])
            for index, name in enumerate(BOUNDARY_DISPOSITION_NAMES)
        },
        "boundary_ray_count": footprint.ray_count,
        "boresight_off_nadir_rad": {
            "minimum": float(np.min(footprint.boresight_off_nadir_rad)),
            "maximum": float(np.max(footprint.boresight_off_nadir_rad)),
        },
        "sampling_semantics": "instantaneous_cell_center_left_closed_right_open",
        "footprint_semantics": "rich_hard_fov_healpix_cell_center",
        "boundary_semantics": "sampled_fov_boundary_ray_wgs84_intersections",
        "claim_limits": [
            "No exact polygon-overlap area or swept-footprint inference.",
            "Pushbroom geometry does not model scanning, integration time, or image performance.",
            "No terrain, atmosphere, clouds, weather, refraction, tasking, or scheduling.",
            "Boundary curves are review evidence and do not define cell membership.",
            "Decision-grade use requires cadence, resolution, and independent matched-assumption evidence.",
        ],
    }


def evaluate_rich_coverage(
    config: RichCoverageConfig,
    *,
    times_s: np.ndarray,
    positions_eci_km: np.ndarray,
    attitudes_quat_bn: np.ndarray,
    frame_context: FrameContext,
    sun_positions_eci_km: np.ndarray | None = None,
) -> RichCoverageResult:
    """Evaluate Phase 3 rich coverage from explicit deterministic evidence."""

    times, positions, attitudes = _validated_evidence(
        times_s,
        positions_eci_km,
        attitudes_quat_bn,
        frame_context,
    )
    sun_positions = _validated_sun_evidence(config, times.size, sun_positions_eci_km)
    resources = estimate_rich_coverage_resources(config, times.size)
    if resources["estimated_peak_working_bytes"] > resources["max_working_memory_bytes"]:
        raise ValueError(
            "Estimated rich coverage working memory exceeds the configured limit: "
            f"{resources['estimated_peak_working_bytes']} > {resources['max_working_memory_bytes']} bytes."
        )
    if resources["cell_time_comparisons"] > resources["max_cell_time_comparisons"]:
        raise ValueError(
            "Rich coverage cell-time comparisons exceed the configured limit: "
            f"{resources['cell_time_comparisons']} > {resources['max_cell_time_comparisons']}."
        )
    (
        positions_ecef,
        sensor_from_ecef,
        sun_ecef,
        subpoint_latitude,
        subpoint_longitude,
        boresight_off_nadir,
    ) = _prepare_ecef_evidence(
        config,
        times,
        positions,
        attitudes,
        sun_positions,
        frame_context,
    )
    footprint = _footprint_boundary(
        config,
        positions_ecef,
        sensor_from_ecef,
        subpoint_latitude,
        subpoint_longitude,
        boresight_off_nadir,
    )

    npix = healpix_npix(config.order)
    covered_count = np.zeros(times.size, dtype=np.int64)
    reason_count = np.zeros((times.size, len(PRIMARY_REASON_NAMES)), dtype=np.int64)
    metric_chunks: list[CoverageCellMetrics] = []
    latitude_chunks: list[np.ndarray] = []
    longitude_chunks: list[np.ndarray] = []
    for start in range(0, npix, config.chunk_size):
        stop = min(start + config.chunk_size, npix)
        cells = np.arange(start, stop, dtype=np.int64)
        centers = healpix_wgs84_centers(config.order, cells)
        latitude_chunks.append(np.rad2deg(centers.geodetic_latitude_rad))
        longitude_chunks.append(np.rad2deg(centers.longitude_rad))
        mask = np.zeros((times.size, cells.size), dtype=bool)
        for sample_index in range(times.size):
            geometry = evaluate_rich_surface_targets_ecef(
                observer_ecef_km=positions_ecef[sample_index],
                target_ecef_km=centers.ecef_km,
                target_outward_normal_ecef=centers.outward_normal_ecef,
                dcm_sensor_from_ecef=sensor_from_ecef[sample_index],
                pattern=config.pattern,
                constraints=config.constraints,
                max_range_km=config.max_range_km,
                sun_ecef_km=None if sun_ecef is None else sun_ecef[sample_index],
                angular_tolerance_rad=_ANGULAR_TOLERANCE_RAD,
                range_tolerance_km=_RANGE_TOLERANCE_KM,
            )
            mask[sample_index] = geometry.available
            reason_count[sample_index] += np.bincount(
                geometry.primary_reason_code,
                minlength=len(PRIMARY_REASON_NAMES),
            )
        covered_count += np.count_nonzero(mask, axis=1)
        metric_chunks.append(summarize_sampled_coverage_mask(mask, times, cell_indices=cells))

    metrics = _combine_chunk_metrics(metric_chunks)
    frame_metadata = frame_context.metadata(sample_t_s=float(times[0]))
    input_hash = _input_evidence_hash(
        times,
        positions,
        attitudes,
        sun_positions,
        frame_metadata,
    )
    semantic_hash = _semantic_hash(
        config,
        input_hash,
        times,
        metrics.intervals,
        reason_count,
        footprint,
    )
    return RichCoverageResult(
        config=config,
        frame_metadata=frame_metadata,
        times_s=times,
        covered_cell_count=covered_count,
        instantaneous_covered_fraction=covered_count.astype(float) / npix,
        cell_geodetic_latitude_deg=np.concatenate(latitude_chunks),
        cell_longitude_deg=np.concatenate(longitude_chunks),
        cell_metrics=metrics,
        primary_reason_count=reason_count,
        footprint_boundary=footprint,
        summary=_summary(config, times, covered_count, metrics, reason_count, footprint),
        resource_estimate=resources,
        input_evidence_sha256=input_hash,
        interval_semantic_sha256=semantic_hash,
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


def write_rich_coverage_artifacts(
    result: RichCoverageResult,
    output_dir: str | Path,
    *,
    include_cell_csv: bool = True,
    include_footprint_plot: bool = False,
) -> RichCoverageArtifacts:
    """Write deterministic Phase 3 evidence and an optional review overlay."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Rich coverage output directory already exists: {destination}")
    destination.mkdir(parents=True)
    summary_path = destination / "rich_coverage_summary.json"
    samples_path = destination / "rich_coverage_samples.csv"
    cells_path = destination / "rich_coverage_cells.csv" if include_cell_csv else None
    intervals_path = destination / "rich_coverage_intervals.npz"
    footprints_path = destination / "rich_coverage_footprints.npz"
    plot_path = destination / "rich_coverage_footprints.png" if include_footprint_plot else None
    manifest_path = destination / "rich_coverage_analysis_manifest.json"

    _write_json(summary_path, result.summary)
    footprint = result.footprint_boundary
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "sample_index",
                "time_s",
                "covered_cell_count",
                "instantaneous_covered_fraction",
                "covered_area_km2",
                "subsatellite_geodetic_latitude_deg",
                "subsatellite_longitude_deg",
                "boresight_off_nadir_rad",
                "boundary_disposition",
                "boundary_hit_count",
                *[f"reason_{name}_count" for name in PRIMARY_REASON_NAMES],
            )
        )
        cell_area = WGS84_SURFACE_AREA_KM2 / healpix_npix(result.config.order)
        for sample_index, time_s in enumerate(result.times_s):
            writer.writerow(
                (
                    sample_index,
                    f"{float(time_s):.17g}",
                    int(result.covered_cell_count[sample_index]),
                    f"{float(result.instantaneous_covered_fraction[sample_index]):.17g}",
                    f"{float(result.covered_cell_count[sample_index] * cell_area):.17g}",
                    f"{float(footprint.subsatellite_geodetic_latitude_deg[sample_index]):.17g}",
                    f"{float(footprint.subsatellite_longitude_deg[sample_index]):.17g}",
                    f"{float(footprint.boresight_off_nadir_rad[sample_index]):.17g}",
                    BOUNDARY_DISPOSITION_NAMES[int(footprint.boundary_disposition_code[sample_index])],
                    int(np.count_nonzero(footprint.boundary_hit[sample_index])),
                    *[int(value) for value in result.primary_reason_count[sample_index]],
                )
            )

    if cells_path is not None:
        metrics = result.cell_metrics
        with cells_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "cell_index",
                    "geodetic_latitude_deg",
                    "longitude_deg",
                    "sampled_dwell_s",
                    "interval_count",
                    "observed_acquisition_count",
                    "max_complete_revisit_gap_s",
                    "prefix_boundary_gap_s",
                    "suffix_boundary_gap_s",
                    "start_censored",
                    "end_censored",
                )
            )
            for index in range(metrics.cell_index.size):
                values = (
                    metrics.max_complete_revisit_gap_s[index],
                    metrics.prefix_boundary_gap_s[index],
                    metrics.suffix_boundary_gap_s[index],
                )
                writer.writerow(
                    (
                        int(metrics.cell_index[index]),
                        f"{float(result.cell_geodetic_latitude_deg[index]):.17g}",
                        f"{float(result.cell_longitude_deg[index]):.17g}",
                        f"{float(metrics.dwell_s[index]):.17g}",
                        int(metrics.interval_count[index]),
                        int(metrics.observed_acquisition_count[index]),
                        *["" if not np.isfinite(value) else f"{float(value):.17g}" for value in values],
                        str(bool(metrics.start_censored[index])).lower(),
                        str(bool(metrics.end_censored[index])).lower(),
                    )
                )

    intervals = result.cell_metrics.intervals
    np.savez_compressed(
        intervals_path,
        cell_index=np.asarray(intervals.cell_index, dtype="<i8"),
        interval_offset=np.asarray(intervals.interval_offset, dtype="<i8"),
        start_sample_index=np.asarray(intervals.start_sample_index, dtype="<i8"),
        end_sample_index_exclusive=np.asarray(intervals.end_sample_index_exclusive, dtype="<i8"),
    )
    np.savez_compressed(
        footprints_path,
        times_s=np.asarray(result.times_s, dtype="<f8"),
        subsatellite_geodetic_latitude_deg=np.asarray(
            footprint.subsatellite_geodetic_latitude_deg,
            dtype="<f8",
        ),
        subsatellite_longitude_deg=np.asarray(footprint.subsatellite_longitude_deg, dtype="<f8"),
        boresight_off_nadir_rad=np.asarray(footprint.boresight_off_nadir_rad, dtype="<f8"),
        boundary_hit=np.asarray(footprint.boundary_hit, dtype="|u1"),
        boundary_geodetic_latitude_deg=np.asarray(
            footprint.boundary_geodetic_latitude_deg,
            dtype="<f8",
        ),
        boundary_longitude_deg=np.asarray(footprint.boundary_longitude_deg, dtype="<f8"),
        boundary_disposition_code=np.asarray(footprint.boundary_disposition_code, dtype="|u1"),
    )
    if plot_path is not None:
        from sim.analysis.coverage_plotting import write_coverage_footprint_plot

        write_coverage_footprint_plot(result, plot_path)

    artifacts: dict[str, dict[str, Any]] = {
        summary_path.name: {"sha256": _sha256_file(summary_path)},
        samples_path.name: {"sha256": _sha256_file(samples_path)},
        intervals_path.name: {
            "container_sha256": _sha256_file(intervals_path),
            "semantic_sha256": result.interval_semantic_sha256,
        },
        footprints_path.name: {
            "container_sha256": _sha256_file(footprints_path),
            "semantic_sha256": result.interval_semantic_sha256,
        },
    }
    if cells_path is not None:
        artifacts[cells_path.name] = {"sha256": _sha256_file(cells_path)}
    if plot_path is not None:
        artifacts[plot_path.name] = {
            "sha256": _sha256_file(plot_path),
            "disposition": "review_visual_not_scientific_identity",
        }
    _write_json(
        manifest_path,
        {
            "contract_version": RICH_COVERAGE_CONTRACT_VERSION,
            "analysis_id": result.config.analysis_id,
            "status": "complete",
            "normalized_scientific_config": _normalized_scientific_config(result.config),
            "frame": result.frame_metadata,
            "resource_estimate": result.resource_estimate,
            "input_evidence_sha256": result.input_evidence_sha256,
            "semantic_sha256": result.interval_semantic_sha256,
            "artifacts": artifacts,
            "claim_limits": result.summary["claim_limits"],
        },
    )
    return RichCoverageArtifacts(
        output_dir=destination,
        manifest_json=manifest_path,
        summary_json=summary_path,
        samples_csv=samples_path,
        cells_csv=cells_path,
        intervals_npz=intervals_path,
        footprints_npz=footprints_path,
        footprint_plot_png=plot_path,
    )


__all__ = [
    "BOUNDARY_DISPOSITION_NAMES",
    "RICH_COVERAGE_CONTRACT_VERSION",
    "FootprintBoundaryEvidence",
    "RichCoverageArtifacts",
    "RichCoverageConfig",
    "RichCoverageResult",
    "estimate_rich_coverage_resources",
    "evaluate_rich_coverage",
    "write_rich_coverage_artifacts",
]
