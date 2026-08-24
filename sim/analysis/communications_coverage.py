"""Phase 4 RF-qualified global Earth communications coverage."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.analysis.directed_link import TerminalPattern, free_space_link_ledger
from sim.analysis.global_coverage import (
    CoverageCellMetrics,
    _combine_chunk_metrics,
    _validated_evidence,
    _validated_quaternion,
    summarize_sampled_coverage_mask,
)
from sim.analysis.healpix import (
    HEALPIX_GRID_ID,
    WGS84_SURFACE_AREA_KM2,
    cached_healpix_wgs84_centers,
    healpix_npix,
)
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM
from sim.utils.quaternion import quaternion_to_dcm_bn

COMMUNICATIONS_COVERAGE_CONTRACT_VERSION = "oel.global-communications-coverage.v0.1"
COMMUNICATIONS_COVERAGE_REASON_NAMES = (
    "available",
    "earth_blocked",
    "below_elevation_mask",
    "beyond_max_range",
    "source_outside_pattern",
    "earth_terminal_outside_pattern",
    "negative_margin",
)
_ANGULAR_TOLERANCE_RAD = 1.0e-12
_RANGE_TOLERANCE_KM = 1.0e-9


def _required(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


@dataclass(frozen=True)
class EarthTerminalProfile:
    profile_id: str
    provenance: str
    pattern: TerminalPattern
    minimum_elevation_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _required(self.profile_id, "profile_id"))
        object.__setattr__(self, "provenance", _required(self.provenance, "provenance"))
        if not isinstance(self.pattern, TerminalPattern):
            raise ValueError("pattern must be a validated TerminalPattern.")
        elevation = float(self.minimum_elevation_rad)
        if not np.isfinite(elevation) or not -0.5 * np.pi <= elevation <= 0.5 * np.pi:
            raise ValueError("minimum_elevation_rad must be finite within [-pi/2, pi/2].")
        object.__setattr__(self, "minimum_elevation_rad", elevation)


@dataclass(frozen=True)
class CommunicationsCoverageConfig:
    analysis_id: str
    service_id: str
    source_asset_id: str
    state_provider_id: str
    attitude_source_kind: str
    attitude_provider_id: str | None
    source_terminal_id: str
    source_terminal_pattern: TerminalPattern
    quat_body_from_terminal: tuple[float, float, float, float]
    earth_terminal_profile: EarthTerminalProfile
    direction: str
    order: int
    carrier_frequency_hz: float
    tx_power_w: float
    data_rate_bps: float
    system_noise_temperature_k: float
    required_eb_n0_db: float
    tx_line_loss_db: float = 0.0
    rx_line_loss_db: float = 0.0
    misc_loss_db: float = 0.0
    max_range_km: float | None = None
    chunk_size: int = 8192
    max_working_memory_bytes: int = 512 * 1024 * 1024
    max_cell_time_comparisons: int = 300_000_000

    def __post_init__(self) -> None:
        for field_name in (
            "analysis_id",
            "service_id",
            "source_asset_id",
            "state_provider_id",
            "source_terminal_id",
        ):
            object.__setattr__(self, field_name, _required(getattr(self, field_name), field_name))
        source_kind = str(self.attitude_source_kind or "").strip().lower()
        if source_kind not in {"achieved", "replay", "analytic_ideal", "not_required"}:
            raise ValueError(
                "attitude_source_kind must be achieved, replay, analytic_ideal, or not_required."
            )
        object.__setattr__(self, "attitude_source_kind", source_kind)
        if not isinstance(self.source_terminal_pattern, TerminalPattern):
            raise ValueError("source_terminal_pattern must be a validated TerminalPattern.")
        if source_kind == "not_required":
            if not self.source_terminal_pattern.attitude_independent:
                raise ValueError(
                    "attitude_source_kind not_required is valid only for an attitude-independent "
                    "source terminal pattern."
                )
            if self.attitude_provider_id is not None:
                raise ValueError("attitude_provider_id must be absent when attitude is not required.")
        else:
            object.__setattr__(
                self,
                "attitude_provider_id",
                _required(self.attitude_provider_id, "attitude_provider_id"),
            )
        if not isinstance(self.earth_terminal_profile, EarthTerminalProfile):
            raise ValueError("Earth terminal profile must be a validated EarthTerminalProfile.")
        mounting = _validated_quaternion(
            self.quat_body_from_terminal,
            "quat_body_from_terminal",
        )
        object.__setattr__(
            self,
            "quat_body_from_terminal",
            tuple(float(value) for value in mounting),
        )
        direction = str(self.direction or "").strip().lower()
        if direction not in {"spacecraft_to_earth", "earth_to_spacecraft"}:
            raise ValueError("direction must be spacecraft_to_earth or earth_to_spacecraft.")
        object.__setattr__(self, "direction", direction)
        if isinstance(self.order, (bool, np.bool_)) or int(self.order) != self.order:
            raise ValueError("order must be an integer.")
        if int(self.order) not in range(5, 9):
            raise ValueError("Communications coverage v0.1 supports HEALPix orders 5 through 8.")
        object.__setattr__(self, "order", int(self.order))
        for field_name in (
            "carrier_frequency_hz",
            "tx_power_w",
            "data_rate_bps",
            "system_noise_temperature_k",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite.")
            object.__setattr__(self, field_name, value)
        required = float(self.required_eb_n0_db)
        if not np.isfinite(required):
            raise ValueError("required_eb_n0_db must be finite.")
        object.__setattr__(self, "required_eb_n0_db", required)
        for field_name in ("tx_line_loss_db", "rx_line_loss_db", "misc_loss_db"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be nonnegative and finite.")
            object.__setattr__(self, field_name, value)
        if self.max_range_km is not None:
            maximum = float(self.max_range_km)
            if not np.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("max_range_km must be positive and finite.")
            object.__setattr__(self, "max_range_km", maximum)
        for field_name in (
            "chunk_size",
            "max_working_memory_bytes",
            "max_cell_time_comparisons",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer.")
            object.__setattr__(self, field_name, int(value))


@dataclass(frozen=True)
class CommunicationsCoverageResult:
    config: CommunicationsCoverageConfig
    frame_metadata: dict[str, Any]
    times_s: np.ndarray
    covered_cell_count: np.ndarray
    instantaneous_covered_fraction: np.ndarray
    cell_geodetic_latitude_deg: np.ndarray
    cell_longitude_deg: np.ndarray
    cell_metrics: CoverageCellMetrics
    primary_reason_count: np.ndarray
    sample_margin_min_db: np.ndarray
    sample_margin_max_db: np.ndarray
    cell_best_margin_db: np.ndarray
    summary: dict[str, Any]
    resource_estimate: dict[str, int]
    input_evidence_sha256: str
    interval_semantic_sha256: str


@dataclass(frozen=True)
class CommunicationsCoverageArtifacts:
    output_dir: Path
    manifest_json: Path
    summary_json: Path
    samples_csv: Path
    cells_csv: Path
    intervals_npz: Path


def estimate_communications_coverage_resources(
    config: CommunicationsCoverageConfig,
    sample_count: int,
) -> dict[str, int]:
    samples = int(sample_count)
    if samples < 2:
        raise ValueError("sample_count must be at least two.")
    cells = healpix_npix(config.order)
    chunk = min(cells, config.chunk_size)
    comparisons = cells * samples
    worst_intervals = cells * ((samples + 1) // 2)
    estimated = (
        chunk * samples
        + chunk * 28 * 8
        + cells * 10 * 8
        + worst_intervals * 16 * 2
        + samples * (3 + 9) * 8
    )
    return {
        "sample_count": samples,
        "cell_count": cells,
        "cell_time_comparisons": comparisons,
        "chunk_size": chunk,
        "worst_case_interval_count": worst_intervals,
        "estimated_peak_working_bytes": estimated,
        "max_working_memory_bytes": config.max_working_memory_bytes,
        "max_cell_time_comparisons": config.max_cell_time_comparisons,
    }


def _prepared_source(
    config: CommunicationsCoverageConfig,
    times: np.ndarray,
    positions_eci: np.ndarray,
    attitudes: np.ndarray | None,
    frame_context: FrameContext,
) -> tuple[np.ndarray, np.ndarray]:
    positions_ecef = np.empty_like(positions_eci)
    terminal_from_ecef = np.empty((times.size, 3, 3), dtype=float)
    body_from_terminal = quaternion_to_dcm_bn(
        np.asarray(config.quat_body_from_terminal, dtype=float)
    )
    terminal_from_body = body_from_terminal.T
    for index, time_s in enumerate(times):
        ecef_from_eci = eci_to_ecef_rotation_context(float(time_s), frame_context)
        if attitudes is None:
            # The transform is deliberately unused by attitude-independent
            # patterns, but retaining an identity row keeps the vectorized RF
            # path shape-stable.
            terminal_from_ecef[index] = np.eye(3)
        else:
            body_from_eci = quaternion_to_dcm_bn(attitudes[index])
            terminal_from_ecef[index] = terminal_from_body @ body_from_eci @ ecef_from_eci.T
        positions_ecef[index] = ecef_from_eci @ positions_eci[index]
    return positions_ecef, terminal_from_ecef


def _pattern_pass(
    pattern: TerminalPattern,
    direction_terminal: np.ndarray,
    *,
    gate_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if pattern.attitude_independent:
        return np.zeros(direction_terminal.shape[0]), np.ones(direction_terminal.shape[0], dtype=bool)
    if gate_mode == "direct_cosine":
        threshold_cosine = np.cos(float(pattern.half_angle_rad) + _ANGULAR_TOLERANCE_RAD)
        cosine = np.clip(direction_terminal[:, 2], -1.0, 1.0)
        return np.zeros(direction_terminal.shape[0]), cosine >= threshold_cosine
    angle = np.arccos(np.clip(direction_terminal[:, 2], -1.0, 1.0))
    return angle, angle <= float(pattern.half_angle_rad) + _ANGULAR_TOLERANCE_RAD


def _input_hash(
    times: np.ndarray,
    positions: np.ndarray,
    attitudes: np.ndarray | None,
    frame_metadata: dict[str, Any],
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "frame": frame_metadata,
                "schema": "oel.communications-coverage-input.v1",
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    for name, values in (
        ("times_s", times),
        ("positions_eci_km", positions),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest.update(name.encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    if attitudes is None:
        digest.update(b"attitudes_quat_bn:not_required")
    else:
        array = np.ascontiguousarray(np.asarray(attitudes, dtype="<f8"))
        digest.update(b"attitudes_quat_bn")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _semantic_hash(
    config: CommunicationsCoverageConfig,
    input_hash: str,
    times: np.ndarray,
    metrics: CoverageCellMetrics,
    reasons: np.ndarray,
    best_margin: np.ndarray,
    pattern_gate_mode: str = "exact_arccos",
) -> str:
    identity = {
        "config": _scientific_config(config),
        "contract_version": COMMUNICATIONS_COVERAGE_CONTRACT_VERSION,
        "grid_identity": HEALPIX_GRID_ID,
        "input_evidence_sha256": input_hash,
    }
    if pattern_gate_mode != "exact_arccos":
        identity["pattern_gate"] = {
            "mode": pattern_gate_mode,
            "equivalence": "rounding_level",
        }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    intervals = metrics.intervals
    for name, values, dtype in (
        ("times_s", times, "<f8"),
        ("cell_index", intervals.cell_index, "<i8"),
        ("interval_offset", intervals.interval_offset, "<i8"),
        ("start_sample_index", intervals.start_sample_index, "<i8"),
        ("end_sample_index_exclusive", intervals.end_sample_index_exclusive, "<i8"),
        ("primary_reason_count", reasons, "<i8"),
        # RF terms are contractually compared at 1e-10 dB. Normalizing to that
        # envelope prevents shape-dependent vector math below the declared
        # tolerance from changing scientific identity across chunk sizes.
        ("cell_best_margin_db", np.round(best_margin, decimals=10), "<f8"),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _scientific_config(config: CommunicationsCoverageConfig) -> dict[str, Any]:
    record = asdict(config)
    for execution_field in (
        "chunk_size",
        "max_working_memory_bytes",
        "max_cell_time_comparisons",
    ):
        record.pop(execution_field)
    return record


def evaluate_communications_coverage(
    config: CommunicationsCoverageConfig,
    *,
    times_s: np.ndarray,
    positions_eci_km: np.ndarray,
    attitudes_quat_bn: np.ndarray | None,
    frame_context: FrameContext,
    pattern_gate_mode: str = "exact_arccos",
) -> CommunicationsCoverageResult:
    """Evaluate global Earth cells against an explicit terminal service profile.

    ``exact_arccos`` is the bitwise-compatible default pattern gate.  The
    explicit ``direct_cosine`` option records rounding-level provenance in the
    result summary and scientific hash.
    """

    normalized_gate_mode = str(pattern_gate_mode or "").strip().lower()
    if normalized_gate_mode not in {"exact_arccos", "direct_cosine"}:
        raise ValueError("pattern_gate_mode must be exact_arccos or direct_cosine.")
    if config.attitude_source_kind == "not_required":
        if attitudes_quat_bn is not None:
            raise ValueError(
                "attitudes_quat_bn must be absent when attitude_source_kind is not_required."
            )
        identity_attitudes = np.tile((1.0, 0.0, 0.0, 0.0), (np.asarray(times_s).size, 1))
        times, positions, _ = _validated_evidence(
            times_s,
            positions_eci_km,
            identity_attitudes,
            frame_context,
        )
        attitudes = None
    else:
        if attitudes_quat_bn is None:
            raise ValueError("attitudes_quat_bn is required for this source terminal pattern.")
        times, positions, attitudes = _validated_evidence(
            times_s,
            positions_eci_km,
            attitudes_quat_bn,
            frame_context,
        )
    resources = estimate_communications_coverage_resources(config, times.size)
    if resources["estimated_peak_working_bytes"] > resources["max_working_memory_bytes"]:
        raise ValueError("Estimated communications coverage memory exceeds the configured limit.")
    if resources["cell_time_comparisons"] > resources["max_cell_time_comparisons"]:
        raise ValueError("Communications coverage comparisons exceed the configured limit.")
    positions_ecef, terminal_from_ecef = _prepared_source(
        config,
        times,
        positions,
        attitudes,
        frame_context,
    )
    ellipsoid_level = (
        (positions_ecef[:, 0] / WGS84_A_KM) ** 2
        + (positions_ecef[:, 1] / WGS84_A_KM) ** 2
        + (positions_ecef[:, 2] / WGS84_B_KM) ** 2
    )
    invalid_source = np.flatnonzero(ellipsoid_level <= 1.0)
    if invalid_source.size:
        raise ValueError(
            "Communications coverage source must be outside WGS84; "
            f"sample {int(invalid_source[0])} is on or inside the ellipsoid."
        )
    npix = healpix_npix(config.order)
    covered_count = np.zeros(times.size, dtype=np.int64)
    reason_count = np.zeros(
        (times.size, len(COMMUNICATIONS_COVERAGE_REASON_NAMES)),
        dtype=np.int64,
    )
    sample_margin_min = np.full(times.size, np.inf)
    sample_margin_max = np.full(times.size, -np.inf)
    metric_chunks: list[CoverageCellMetrics] = []
    latitude_chunks: list[np.ndarray] = []
    longitude_chunks: list[np.ndarray] = []
    best_margin_chunks: list[np.ndarray] = []
    earth_pattern = config.earth_terminal_profile.pattern
    for start in range(0, npix, config.chunk_size):
        stop = min(start + config.chunk_size, npix)
        cells = np.arange(start, stop, dtype=np.int64)
        centers = cached_healpix_wgs84_centers(config.order, cells)
        latitude_chunks.append(np.rad2deg(centers.geodetic_latitude_rad))
        longitude_chunks.append(np.rad2deg(centers.longitude_rad))
        mask = np.zeros((times.size, cells.size), dtype=bool)
        best_margin = np.full(cells.size, -np.inf)
        for sample_index in range(times.size):
            delta = centers.ecef_km - positions_ecef[sample_index]
            ranges = np.linalg.norm(delta, axis=1)
            source_to_cell = delta / ranges[:, None]
            cell_to_source = -source_to_cell
            visible = (
                np.einsum(
                    "ij,ij->i",
                    centers.outward_normal_ecef,
                    positions_ecef[sample_index] - centers.ecef_km,
                )
                > _RANGE_TOLERANCE_KM
            )
            earth_terminal_cosine = np.einsum(
                "ij,ij->i",
                centers.outward_normal_ecef,
                cell_to_source,
            )
            elevation = np.arcsin(np.clip(earth_terminal_cosine, -1.0, 1.0))
            elevation_pass = (
                elevation
                >= config.earth_terminal_profile.minimum_elevation_rad
                - _ANGULAR_TOLERANCE_RAD
            )
            range_pass = np.ones(cells.size, dtype=bool)
            if config.max_range_km is not None:
                range_pass = ranges <= config.max_range_km + _RANGE_TOLERANCE_KM
            source_direction_terminal = source_to_cell @ terminal_from_ecef[sample_index].T
            _, source_pattern_pass = _pattern_pass(
                config.source_terminal_pattern,
                source_direction_terminal,
                gate_mode=normalized_gate_mode,
            )
            # Only the +Z cosine is needed for the axisymmetric profile.
            if earth_pattern.attitude_independent:
                earth_pattern_pass = np.ones(cells.size, dtype=bool)
            else:
                if normalized_gate_mode == "direct_cosine":
                    threshold_cosine = np.cos(
                        float(earth_pattern.half_angle_rad) + _ANGULAR_TOLERANCE_RAD
                    )
                    earth_pattern_pass = (
                        np.clip(earth_terminal_cosine, -1.0, 1.0)
                        >= threshold_cosine
                    )
                else:
                    earth_off_axis = np.arccos(
                        np.clip(earth_terminal_cosine, -1.0, 1.0)
                    )
                    earth_pattern_pass = (
                        earth_off_axis
                        <= float(earth_pattern.half_angle_rad) + _ANGULAR_TOLERANCE_RAD
                    )
            if config.direction == "spacecraft_to_earth":
                tx_gain = config.source_terminal_pattern.gain_dbi
                rx_gain = earth_pattern.gain_dbi
            else:
                tx_gain = earth_pattern.gain_dbi
                rx_gain = config.source_terminal_pattern.gain_dbi
            rf = free_space_link_ledger(
                ranges,
                carrier_frequency_hz=config.carrier_frequency_hz,
                tx_power_w=config.tx_power_w,
                tx_gain_dbi=tx_gain,
                rx_gain_dbi=rx_gain,
                data_rate_bps=config.data_rate_bps,
                system_noise_temperature_k=config.system_noise_temperature_k,
                required_eb_n0_db=config.required_eb_n0_db,
                tx_line_loss_db=config.tx_line_loss_db,
                rx_line_loss_db=config.rx_line_loss_db,
                misc_loss_db=config.misc_loss_db,
            )
            gates = (
                visible,
                elevation_pass,
                range_pass,
                source_pattern_pass,
                earth_pattern_pass,
                rf.margin_pass,
            )
            available = np.ones(cells.size, dtype=bool)
            reason_code = np.zeros(cells.size, dtype=np.uint8)
            for code, gate in enumerate(gates, start=1):
                failed = available & (~gate)
                reason_code[failed] = code
                available &= gate
            mask[sample_index] = available
            reason_count[sample_index] += np.bincount(
                reason_code,
                minlength=len(COMMUNICATIONS_COVERAGE_REASON_NAMES),
            )
            sample_margin_min[sample_index] = min(
                sample_margin_min[sample_index],
                float(np.min(rf.margin_db)),
            )
            sample_margin_max[sample_index] = max(
                sample_margin_max[sample_index],
                float(np.max(rf.margin_db)),
            )
            best_margin = np.maximum(best_margin, rf.margin_db)
        covered_count += np.count_nonzero(mask, axis=1)
        metric_chunks.append(summarize_sampled_coverage_mask(mask, times, cell_indices=cells))
        best_margin_chunks.append(best_margin)
    metrics = _combine_chunk_metrics(metric_chunks)
    best_margin = np.concatenate(best_margin_chunks)
    frame_metadata = frame_context.metadata(sample_t_s=float(times[0]))
    input_hash = _input_hash(times, positions, attitudes, frame_metadata)
    semantic_hash = _semantic_hash(
        config,
        input_hash,
        times,
        metrics,
        reason_count,
        best_margin,
        normalized_gate_mode,
    )
    summary = _summary(config, times, covered_count, metrics, reason_count, best_margin)
    if normalized_gate_mode != "exact_arccos":
        summary["pattern_gate"] = {
            "mode": normalized_gate_mode,
            "equivalence": "rounding_level",
            "provenance": "explicit_analysis_call",
        }
    return CommunicationsCoverageResult(
        config=config,
        frame_metadata=frame_metadata,
        times_s=times,
        covered_cell_count=covered_count,
        instantaneous_covered_fraction=covered_count.astype(float) / npix,
        cell_geodetic_latitude_deg=np.concatenate(latitude_chunks),
        cell_longitude_deg=np.concatenate(longitude_chunks),
        cell_metrics=metrics,
        primary_reason_count=reason_count,
        sample_margin_min_db=sample_margin_min,
        sample_margin_max_db=sample_margin_max,
        cell_best_margin_db=best_margin,
        summary=summary,
        resource_estimate=resources,
        input_evidence_sha256=input_hash,
        interval_semantic_sha256=semantic_hash,
    )


def _summary(
    config: CommunicationsCoverageConfig,
    times: np.ndarray,
    covered: np.ndarray,
    metrics: CoverageCellMetrics,
    reasons: np.ndarray,
    best_margin: np.ndarray,
) -> dict[str, Any]:
    npix = healpix_npix(config.order)
    duration = float(times[-1] - times[0])
    fraction = covered.astype(float) / npix
    ever = metrics.interval_count > 0
    finite_revisit = metrics.max_complete_revisit_gap_s[
        np.isfinite(metrics.max_complete_revisit_gap_s)
    ]
    return {
        "contract_version": COMMUNICATIONS_COVERAGE_CONTRACT_VERSION,
        "analysis_id": config.analysis_id,
        "service_id": config.service_id,
        "status": "complete",
        "domain_disposition": "global_earth_communications_service",
        "direction": config.direction,
        "rf_roles": {
            "transmitter": (
                "source_terminal"
                if config.direction == "spacecraft_to_earth"
                else "earth_terminal"
            ),
            "receiver": (
                "earth_terminal"
                if config.direction == "spacecraft_to_earth"
                else "source_terminal"
            ),
            "tx_gain_dbi": (
                config.source_terminal_pattern.gain_dbi
                if config.direction == "spacecraft_to_earth"
                else config.earth_terminal_profile.pattern.gain_dbi
            ),
            "rx_gain_dbi": (
                config.earth_terminal_profile.pattern.gain_dbi
                if config.direction == "spacecraft_to_earth"
                else config.source_terminal_pattern.gain_dbi
            ),
        },
        "earth_terminal_profile_id": config.earth_terminal_profile.profile_id,
        "earth_terminal_profile_provenance": config.earth_terminal_profile.provenance,
        "grid_identity": HEALPIX_GRID_ID,
        "order": config.order,
        "cell_count": npix,
        "cell_area_km2": WGS84_SURFACE_AREA_KM2 / npix,
        "sample_count": int(times.size),
        "horizon_start_s": float(times[0]),
        "horizon_end_s": float(times[-1]),
        "time_weighted_mean_covered_fraction": float(
            np.dot(fraction[:-1], np.diff(times)) / duration
        ),
        "ever_service_qualified_cell_count": int(np.count_nonzero(ever)),
        "never_service_qualified_cell_count": int(np.count_nonzero(~ever)),
        "ever_service_qualified_fraction": float(np.count_nonzero(ever) / npix),
        "sampled_dwell_s": {
            "minimum": float(np.min(metrics.dwell_s)),
            "mean": float(np.mean(metrics.dwell_s)),
            "maximum": float(np.max(metrics.dwell_s)),
        },
        "max_complete_revisit_gap_s": {
            "minimum": None if not finite_revisit.size else float(np.min(finite_revisit)),
            "mean": None if not finite_revisit.size else float(np.mean(finite_revisit)),
            "maximum": None if not finite_revisit.size else float(np.max(finite_revisit)),
            "evaluated_cell_count": int(finite_revisit.size),
        },
        "best_margin_db": {
            "minimum": float(np.min(best_margin)),
            "mean": float(np.mean(best_margin)),
            "maximum": float(np.max(best_margin)),
        },
        "primary_reason_total": {
            name: int(np.sum(reasons[:, index], dtype=np.int64))
            for index, name in enumerate(COMMUNICATIONS_COVERAGE_REASON_NAMES)
        },
        "claim_limits": [
            "Every Earth cell uses the declared terminal profile; no actual-site inventory is implied.",
            "Same-epoch one-way free-space service qualification only.",
            "No terrain, atmosphere, weather, interference, protocols, scheduling, or hardware assurance.",
            "Center-of-cell sampling remains resolution and cadence sensitive.",
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


def write_communications_coverage_artifacts(
    result: CommunicationsCoverageResult,
    output_dir: str | Path,
) -> CommunicationsCoverageArtifacts:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Communications coverage output already exists: {destination}")
    destination.mkdir(parents=True)
    manifest = destination / "communications_coverage_manifest.json"
    summary = destination / "communications_coverage_summary.json"
    samples = destination / "communications_coverage_samples.csv"
    cells = destination / "communications_coverage_cells.csv"
    intervals = destination / "communications_coverage_intervals.npz"
    _json_dump(summary, result.summary)
    with samples.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "sample_index",
                "time_s",
                "service_qualified_cell_count",
                "service_qualified_fraction",
                "minimum_margin_db",
                "maximum_margin_db",
                *[f"reason_{name}_count" for name in COMMUNICATIONS_COVERAGE_REASON_NAMES],
            )
        )
        for index, time_s in enumerate(result.times_s):
            writer.writerow(
                (
                    index,
                    f"{float(time_s):.17g}",
                    int(result.covered_cell_count[index]),
                    f"{float(result.instantaneous_covered_fraction[index]):.17g}",
                    f"{float(result.sample_margin_min_db[index]):.17g}",
                    f"{float(result.sample_margin_max_db[index]):.17g}",
                    *[int(value) for value in result.primary_reason_count[index]],
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
                "best_margin_db",
                "sampled_dwell_s",
                "interval_count",
                "max_complete_revisit_gap_s",
                "start_censored",
                "end_censored",
            )
        )
        for index in range(metrics.cell_index.size):
            max_gap = metrics.max_complete_revisit_gap_s[index]
            writer.writerow(
                (
                    int(metrics.cell_index[index]),
                    f"{float(result.cell_geodetic_latitude_deg[index]):.17g}",
                    f"{float(result.cell_longitude_deg[index]):.17g}",
                    f"{float(result.cell_best_margin_db[index]):.17g}",
                    f"{float(metrics.dwell_s[index]):.17g}",
                    int(metrics.interval_count[index]),
                    "" if not np.isfinite(max_gap) else f"{float(max_gap):.17g}",
                    str(bool(metrics.start_censored[index])).lower(),
                    str(bool(metrics.end_censored[index])).lower(),
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
            "contract_version": COMMUNICATIONS_COVERAGE_CONTRACT_VERSION,
            "analysis_id": result.config.analysis_id,
            "service_id": result.config.service_id,
            "status": "complete",
            "normalized_config": asdict(result.config),
            "frame": result.frame_metadata,
            "resource_estimate": result.resource_estimate,
            "input_evidence_sha256": result.input_evidence_sha256,
            "semantic_sha256": result.interval_semantic_sha256,
            "artifacts": artifacts,
            "claim_limits": result.summary["claim_limits"],
        },
    )
    return CommunicationsCoverageArtifacts(
        output_dir=destination,
        manifest_json=manifest,
        summary_json=summary,
        samples_csv=samples,
        cells_csv=cells,
        intervals_npz=intervals,
    )


__all__ = [
    "COMMUNICATIONS_COVERAGE_CONTRACT_VERSION",
    "COMMUNICATIONS_COVERAGE_REASON_NAMES",
    "CommunicationsCoverageArtifacts",
    "CommunicationsCoverageConfig",
    "CommunicationsCoverageResult",
    "EarthTerminalProfile",
    "estimate_communications_coverage_resources",
    "evaluate_communications_coverage",
    "write_communications_coverage_artifacts",
]
