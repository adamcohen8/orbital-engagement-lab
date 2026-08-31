"""Bounded deterministic constellation and ground-network design trades."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.analysis.coverage_aggregation import (
    ConstellationCoverageConfig,
    evaluate_constellation_coverage,
)
from sim.analysis.directed_link import (
    DirectedLinkConfig,
    LinkTerminal,
    TerminalPattern,
    evaluate_directed_link,
    fixed_wgs84_site_history,
)
from sim.analysis.global_coverage import GlobalCoverageConfig
from sim.analysis.healpix import healpix_npix
from sim.analysis.history_adapters import AnalysisHistory, evaluate_history_global_coverage
from sim.analysis.optical_collection import local_nadir_frame_sensor_from_eci
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import FrameContext
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin
from sim.orbital_calculator.core import circular_inclined_elements_to_rv
from sim.utils.io import SafeReadError, read_regular_file_nofollow
from sim.utils.quaternion import dcm_to_quaternion_bn

CONSTELLATION_DESIGN_PROBLEM_SCHEMA = "oel.constellation_design_problem.v1"
CONSTELLATION_DESIGN_EVIDENCE_SCHEMA = "oel.constellation_design_evidence.v1"
MAX_PUBLIC_DESIGNS = 8
MAX_PUBLIC_SATELLITES_PER_DESIGN = 24
MAX_PUBLIC_GROUND_SITES = 8
MAX_PUBLIC_SELECTED_GROUND_SITES = 4
MAX_PUBLIC_SAMPLES = 721
MAX_PUBLIC_COVERAGE_COMPARISONS = 120_000_000
MAX_PUBLIC_LINK_SAMPLES = 100_000
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_BYTES = 24 * 1024 * 1024
_IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)


class ConstellationDesignError(ValueError):
    """Raised when a public constellation-design contract is invalid."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConstellationDesignError(f"{field} must be a JSON object.")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ConstellationDesignError(f"{field} must be a JSON array.")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ConstellationDesignError(f"{field} is missing required fields: {', '.join(missing)}.")
    if unknown:
        raise ConstellationDesignError(f"{field} contains unknown fields: {', '.join(unknown)}.")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstellationDesignError(f"{field} must be a non-empty string.")
    normalized = value.strip()
    if len(normalized) > 128:
        raise ConstellationDesignError(f"{field} exceeds the 128-character public bound.")
    return normalized


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ConstellationDesignError(f"{field} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConstellationDesignError(f"{field} must be finite.") from exc
    if not math.isfinite(result):
        raise ConstellationDesignError(f"{field} must be finite.")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConstellationDesignError(f"{field} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConstellationDesignError(f"{field} must be an integer.") from exc
    if result != value:
        raise ConstellationDesignError(f"{field} must be an integer.")
    return result


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConstellationDesignError(f"Evidence must be finite JSON: {exc}") from exc


def _semantic_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConstellationDesignError(f"Problem must be finite JSON: {exc}") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GroundSite:
    site_id: str
    geodetic_latitude_deg: float
    longitude_deg: float
    ellipsoidal_height_km: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GroundSite:
        raw = _mapping(value, "ground site")
        _exact_fields(
            raw,
            {"site_id", "geodetic_latitude_deg", "longitude_deg", "ellipsoidal_height_km"},
            "ground site",
        )
        latitude = _finite(raw["geodetic_latitude_deg"], "ground_site.geodetic_latitude_deg")
        longitude = _finite(raw["longitude_deg"], "ground_site.longitude_deg")
        height = _finite(raw["ellipsoidal_height_km"], "ground_site.ellipsoidal_height_km")
        if not -90.0 <= latitude <= 90.0:
            raise ConstellationDesignError("Ground-site latitude must be within [-90, 90] degrees.")
        if not -180.0 <= longitude <= 180.0:
            raise ConstellationDesignError("Ground-site longitude must be within [-180, 180] degrees.")
        if height < 0.0:
            raise ConstellationDesignError("Ground-site ellipsoidal height must be nonnegative.")
        return cls(_text(raw["site_id"], "ground_site.site_id"), latitude, longitude, height)


@dataclass(frozen=True)
class ConstellationCandidate:
    design_id: str
    pattern: str
    satellite_count: int
    plane_count: int
    phasing: int
    altitude_km: float
    inclination_deg: float
    raan_start_deg: float
    initial_phase_deg: float
    raan_span_deg: float | None
    ground_site_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ConstellationCandidate:
        raw = _mapping(value, "design")
        _exact_fields(
            raw,
            {
                "design_id",
                "pattern",
                "satellite_count",
                "plane_count",
                "phasing",
                "altitude_km",
                "inclination_deg",
                "raan_start_deg",
                "initial_phase_deg",
                "raan_span_deg",
                "ground_site_ids",
            },
            "design",
        )
        pattern = _text(raw["pattern"], "design.pattern").lower()
        if pattern not in {"walker_delta", "walker_star", "shell"}:
            raise ConstellationDesignError("design.pattern must be walker_delta, walker_star, or shell.")
        satellite_count = _integer(raw["satellite_count"], "design.satellite_count")
        plane_count = _integer(raw["plane_count"], "design.plane_count")
        phasing = _integer(raw["phasing"], "design.phasing")
        if not 2 <= satellite_count <= MAX_PUBLIC_SATELLITES_PER_DESIGN:
            raise ConstellationDesignError(
                f"design.satellite_count must be within [2, {MAX_PUBLIC_SATELLITES_PER_DESIGN}]."
            )
        if not 1 <= plane_count <= satellite_count or satellite_count % plane_count:
            raise ConstellationDesignError("design.plane_count must divide satellite_count.")
        if not 0 <= phasing < plane_count:
            raise ConstellationDesignError("design.phasing must be within [0, plane_count).")
        altitude = _finite(raw["altitude_km"], "design.altitude_km")
        inclination = _finite(raw["inclination_deg"], "design.inclination_deg")
        if not 100.0 <= altitude <= 100_000.0:
            raise ConstellationDesignError("design.altitude_km must be within [100, 100000].")
        if not 0.0 <= inclination <= 180.0:
            raise ConstellationDesignError("design.inclination_deg must be within [0, 180].")
        span_raw = raw["raan_span_deg"]
        if pattern == "shell":
            if span_raw is None:
                raise ConstellationDesignError("A shell design requires raan_span_deg.")
            span = _finite(span_raw, "design.raan_span_deg")
            if not 0.0 < span <= 360.0:
                raise ConstellationDesignError("design.raan_span_deg must be within (0, 360].")
        else:
            if span_raw is not None:
                raise ConstellationDesignError("Walker designs derive their RAAN span and require raan_span_deg=null.")
            span = None
        site_ids = tuple(
            _text(item, "design.ground_site_ids item")
            for item in _sequence(raw["ground_site_ids"], "design.ground_site_ids")
        )
        if not 1 <= len(site_ids) <= MAX_PUBLIC_SELECTED_GROUND_SITES or len(set(site_ids)) != len(site_ids):
            raise ConstellationDesignError(
                f"design.ground_site_ids must contain 1-{MAX_PUBLIC_SELECTED_GROUND_SITES} unique IDs."
            )
        return cls(
            design_id=_text(raw["design_id"], "design.design_id"),
            pattern=pattern,
            satellite_count=satellite_count,
            plane_count=plane_count,
            phasing=phasing,
            altitude_km=altitude,
            inclination_deg=inclination,
            raan_start_deg=_finite(raw["raan_start_deg"], "design.raan_start_deg") % 360.0,
            initial_phase_deg=_finite(raw["initial_phase_deg"], "design.initial_phase_deg") % 360.0,
            raan_span_deg=span,
            ground_site_ids=site_ids,
        )

    @property
    def effective_raan_span_deg(self) -> float:
        if self.pattern == "walker_delta":
            return 360.0
        if self.pattern == "walker_star":
            return 180.0
        assert self.raan_span_deg is not None
        return self.raan_span_deg


@dataclass(frozen=True)
class ConstellationDesignProblem:
    analysis_id: str
    initial_jd_utc: float
    duration_s: float
    sample_step_s: float
    propagation: dict[str, Any]
    coverage: dict[str, Any]
    ground_sites: tuple[GroundSite, ...]
    link_budget: dict[str, float]
    objective: dict[str, float]
    designs: tuple[ConstellationCandidate, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ConstellationDesignProblem:
        raw = _mapping(value, "constellation-design problem")
        _exact_fields(
            raw,
            {
                "schema_version",
                "analysis_id",
                "initial_jd_utc",
                "duration_s",
                "sample_step_s",
                "propagation",
                "coverage",
                "ground_sites",
                "link_budget",
                "objective",
                "designs",
            },
            "constellation-design problem",
        )
        if raw["schema_version"] != CONSTELLATION_DESIGN_PROBLEM_SCHEMA:
            raise ConstellationDesignError(f"Unsupported problem schema: {raw['schema_version']!r}.")
        duration = _finite(raw["duration_s"], "duration_s")
        sample_step = _finite(raw["sample_step_s"], "sample_step_s")
        if duration <= 0.0 or sample_step <= 0.0 or sample_step > duration:
            raise ConstellationDesignError(
                "duration_s and sample_step_s must be positive, with sample_step_s <= duration_s."
            )
        ratio = duration / sample_step
        if abs(ratio - round(ratio)) > 1.0e-10:
            raise ConstellationDesignError("duration_s must be an integer multiple of sample_step_s.")
        sample_count = int(round(ratio)) + 1
        if sample_count > MAX_PUBLIC_SAMPLES:
            raise ConstellationDesignError(f"Problem exceeds the public bound of {MAX_PUBLIC_SAMPLES} samples.")

        propagation_raw = _mapping(raw["propagation"], "propagation")
        _exact_fields(propagation_raw, {"model", "integration_step_s"}, "propagation")
        model = _text(propagation_raw["model"], "propagation.model").lower()
        if model not in {"onp_two_body", "onp_j2"}:
            raise ConstellationDesignError("propagation.model must be onp_two_body or onp_j2.")
        integration_step = _finite(propagation_raw["integration_step_s"], "propagation.integration_step_s")
        if integration_step <= 0.0 or integration_step > sample_step:
            raise ConstellationDesignError("integration_step_s must be positive and no larger than sample_step_s.")

        coverage_raw = _mapping(raw["coverage"], "coverage")
        _exact_fields(coverage_raw, {"order", "half_angle_deg", "required_multiplicity"}, "coverage")
        order = _integer(coverage_raw["order"], "coverage.order")
        half_angle = _finite(coverage_raw["half_angle_deg"], "coverage.half_angle_deg")
        multiplicity = _integer(coverage_raw["required_multiplicity"], "coverage.required_multiplicity")
        if order not in range(5, 9):
            raise ConstellationDesignError("coverage.order must be within [5, 8].")
        if not 0.0 < half_angle < 90.0:
            raise ConstellationDesignError("coverage.half_angle_deg must be within (0, 90).")
        if multiplicity < 1:
            raise ConstellationDesignError("coverage.required_multiplicity must be positive.")

        sites = tuple(GroundSite.from_mapping(item) for item in _sequence(raw["ground_sites"], "ground_sites"))
        if not 1 <= len(sites) <= MAX_PUBLIC_GROUND_SITES:
            raise ConstellationDesignError(f"ground_sites must contain 1-{MAX_PUBLIC_GROUND_SITES} sites.")
        site_ids = [site.site_id for site in sites]
        if len(site_ids) != len(set(site_ids)):
            raise ConstellationDesignError("ground_sites site_id values must be unique.")

        link_raw = _mapping(raw["link_budget"], "link_budget")
        link_fields = {
            "carrier_frequency_hz",
            "tx_power_w",
            "data_rate_bps",
            "system_noise_temperature_k",
            "required_eb_n0_db",
            "tx_gain_dbi",
            "rx_gain_dbi",
            "tx_line_loss_db",
            "rx_line_loss_db",
            "misc_loss_db",
            "minimum_elevation_deg",
        }
        _exact_fields(link_raw, link_fields, "link_budget")
        link = {field: _finite(link_raw[field], f"link_budget.{field}") for field in link_fields}
        for field in ("carrier_frequency_hz", "tx_power_w", "data_rate_bps", "system_noise_temperature_k"):
            if link[field] <= 0.0:
                raise ConstellationDesignError(f"link_budget.{field} must be positive.")
        for field in ("tx_line_loss_db", "rx_line_loss_db", "misc_loss_db"):
            if link[field] < 0.0:
                raise ConstellationDesignError(f"link_budget.{field} must be nonnegative.")
        if not -90.0 <= link["minimum_elevation_deg"] <= 90.0:
            raise ConstellationDesignError("link_budget.minimum_elevation_deg must be within [-90, 90].")

        objective_raw = _mapping(raw["objective"], "objective")
        objective_fields = {
            "coverage_weight",
            "network_weight",
            "satellite_penalty",
            "ground_site_penalty",
            "minimum_coverage_fraction",
            "minimum_network_availability_fraction",
        }
        _exact_fields(objective_raw, objective_fields, "objective")
        objective = {field: _finite(objective_raw[field], f"objective.{field}") for field in objective_fields}
        if any(
            objective[field] < 0.0
            for field in ("coverage_weight", "network_weight", "satellite_penalty", "ground_site_penalty")
        ):
            raise ConstellationDesignError("Objective weights and penalties must be nonnegative.")
        if objective["coverage_weight"] + objective["network_weight"] <= 0.0:
            raise ConstellationDesignError("At least one service objective weight must be positive.")
        for field in ("minimum_coverage_fraction", "minimum_network_availability_fraction"):
            if not 0.0 <= objective[field] <= 1.0:
                raise ConstellationDesignError(f"objective.{field} must be within [0, 1].")

        designs = tuple(ConstellationCandidate.from_mapping(item) for item in _sequence(raw["designs"], "designs"))
        if not 1 <= len(designs) <= MAX_PUBLIC_DESIGNS:
            raise ConstellationDesignError(f"designs must contain 1-{MAX_PUBLIC_DESIGNS} explicit candidates.")
        design_ids = [design.design_id for design in designs]
        if len(design_ids) != len(set(design_ids)):
            raise ConstellationDesignError("design_id values must be unique.")
        known_sites = set(site_ids)
        for design in designs:
            unknown_sites = sorted(set(design.ground_site_ids) - known_sites)
            if unknown_sites:
                raise ConstellationDesignError(
                    f"Design {design.design_id!r} references unknown ground sites: {', '.join(unknown_sites)}."
                )
            if multiplicity > design.satellite_count:
                raise ConstellationDesignError(
                    f"coverage.required_multiplicity exceeds satellite_count for {design.design_id!r}."
                )

        coverage_comparisons = sum(design.satellite_count for design in designs) * sample_count * healpix_npix(order)
        link_samples = sum(design.satellite_count * len(design.ground_site_ids) for design in designs) * sample_count
        if coverage_comparisons > MAX_PUBLIC_COVERAGE_COMPARISONS:
            raise ConstellationDesignError(
                f"Problem requires {coverage_comparisons} coverage comparisons, above the public bound of {MAX_PUBLIC_COVERAGE_COMPARISONS}."
            )
        if link_samples > MAX_PUBLIC_LINK_SAMPLES:
            raise ConstellationDesignError(
                f"Problem requires {link_samples} link samples, above the public bound of {MAX_PUBLIC_LINK_SAMPLES}."
            )
        return cls(
            analysis_id=_text(raw["analysis_id"], "analysis_id"),
            initial_jd_utc=_finite(raw["initial_jd_utc"], "initial_jd_utc"),
            duration_s=duration,
            sample_step_s=sample_step,
            propagation={"model": model, "integration_step_s": integration_step},
            coverage={"order": order, "half_angle_deg": half_angle, "required_multiplicity": multiplicity},
            ground_sites=tuple(sorted(sites, key=lambda item: item.site_id)),
            link_budget={field: link[field] for field in sorted(link)},
            objective={field: objective[field] for field in sorted(objective)},
            designs=tuple(sorted(designs, key=lambda item: item.design_id)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONSTELLATION_DESIGN_PROBLEM_SCHEMA,
            "analysis_id": self.analysis_id,
            "initial_jd_utc": self.initial_jd_utc,
            "duration_s": self.duration_s,
            "sample_step_s": self.sample_step_s,
            "propagation": dict(self.propagation),
            "coverage": dict(self.coverage),
            "ground_sites": [asdict(item) for item in self.ground_sites],
            "link_budget": dict(self.link_budget),
            "objective": dict(self.objective),
            "designs": [asdict(item) for item in self.designs],
        }


@dataclass(frozen=True)
class ConstellationDesignResult:
    problem: ConstellationDesignProblem
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ConstellationDesignArtifacts:
    output_dir: Path
    manifest_json: Path
    problem_json: Path
    evidence_json: Path


def generate_constellation_members(design: ConstellationCandidate) -> list[dict[str, Any]]:
    """Generate canonical circular member states for a Walker or bounded shell design."""

    satellites_per_plane = design.satellite_count // design.plane_count
    members: list[dict[str, Any]] = []
    for plane_index in range(design.plane_count):
        raan_deg = (design.raan_start_deg + plane_index * design.effective_raan_span_deg / design.plane_count) % 360.0
        plane_phase_deg = plane_index * design.phasing * 360.0 / design.satellite_count
        for slot_index in range(satellites_per_plane):
            argument_of_latitude_deg = (
                design.initial_phase_deg + slot_index * 360.0 / satellites_per_plane + plane_phase_deg
            ) % 360.0
            converted = circular_inclined_elements_to_rv(
                EARTH_RADIUS_KM + design.altitude_km,
                design.inclination_deg,
                raan_deg,
                argument_of_latitude_deg,
            )
            member_id = f"{design.design_id}.p{plane_index + 1:02d}.s{slot_index + 1:02d}"
            members.append(
                {
                    "member_id": member_id,
                    "plane_index": plane_index,
                    "slot_index": slot_index,
                    "raan_deg": raan_deg,
                    "argument_of_latitude_deg": argument_of_latitude_deg,
                    "position_eci_km": list(converted.position_eci_km),
                    "velocity_eci_km_s": list(converted.velocity_eci_km_s),
                }
            )
    return members


def _propagate_member(
    problem: ConstellationDesignProblem,
    member: Mapping[str, Any],
) -> AnalysisHistory:
    times = np.arange(0.0, problem.duration_s + 0.5 * problem.sample_step_s, problem.sample_step_s, dtype=float)
    state = np.array([*member["position_eci_km"], *member["velocity_eci_km_s"]], dtype=float)
    states = np.empty((times.size, 6), dtype=float)
    states[0] = state
    plugins = [j2_plugin] if problem.propagation["model"] == "onp_j2" else []
    propagator = OrbitPropagator(model="two_body", integrator="rk4", plugins=plugins)
    context = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=1.0)
    integration_step = float(problem.propagation["integration_step_s"])
    current_time = 0.0
    for sample_index in range(1, times.size):
        target_time = float(times[sample_index])
        while current_time < target_time - 1.0e-12:
            step = min(integration_step, target_time - current_time)
            state = propagator.propagate(
                state,
                step,
                current_time,
                np.zeros(3, dtype=float),
                {},
                context,
            )
            current_time += step
        states[sample_index] = state
    attitudes = np.asarray(
        [dcm_to_quaternion_bn(local_nadir_frame_sensor_from_eci(row)) for row in states],
        dtype=float,
    )
    provider = f"onp:{problem.propagation['model']}:rk4:{integration_step:.12g}s"
    return AnalysisHistory(
        object_id=str(member["member_id"]),
        product_kind="constellation_design_member",
        state_provider_id=provider,
        frame="eci",
        initial_jd_utc=problem.initial_jd_utc,
        times_s=times,
        position_eci_km=states[:, :3],
        velocity_eci_km_s=states[:, 3:],
        attitude_quat_bn=attitudes,
        attitude_source_kind="analytic_ideal",
        attitude_provider_id="ideal_local_nadir_frame.v1",
    )


def _evaluate_design(problem: ConstellationDesignProblem, design: ConstellationCandidate) -> dict[str, Any]:
    members = generate_constellation_members(design)
    histories = [_propagate_member(problem, member) for member in members]
    frame_context = FrameContext(jd_utc_start=problem.initial_jd_utc)
    coverage_products = []
    for history in histories:
        coverage_products.append(
            evaluate_history_global_coverage(
                GlobalCoverageConfig(
                    analysis_id=f"{problem.analysis_id}.{history.object_id}.coverage",
                    source_asset_id=history.object_id,
                    state_provider_id=history.state_provider_id,
                    attitude_source_kind=history.attitude_source_kind,
                    attitude_provider_id=str(history.attitude_provider_id),
                    sensor_id="ideal_nadir_conical_sensor",
                    order=int(problem.coverage["order"]),
                    half_angle_rad=math.radians(float(problem.coverage["half_angle_deg"])),
                    quat_body_from_sensor=_IDENTITY_QUATERNION,
                    max_cell_time_comparisons=MAX_PUBLIC_COVERAGE_COMPARISONS,
                ),
                history=history,
                frame_context=frame_context,
            )
        )
    coverage = evaluate_constellation_coverage(
        ConstellationCoverageConfig(
            analysis_id=f"{problem.analysis_id}.{design.design_id}.coverage",
            member_analysis_ids=tuple(product.config.analysis_id for product in coverage_products),
            order=int(problem.coverage["order"]),
            service_definition_id="ideal_nadir_conical_coverage",
            required_multiplicity=int(problem.coverage["required_multiplicity"]),
            max_asset_cell_time_values=MAX_PUBLIC_COVERAGE_COMPARISONS,
        ),
        coverage_products,
    )

    sites_by_id = {site.site_id: site for site in problem.ground_sites}
    link_results = []
    union_available = np.zeros(histories[0].times_s.shape, dtype=bool)
    for history in histories:
        spacecraft_endpoint = history.link_endpoint(require_attitude=False)
        for site_id in design.ground_site_ids:
            site = sites_by_id[site_id]
            site_endpoint = fixed_wgs84_site_history(
                asset_id=site.site_id,
                state_provider_id="fixed_wgs84_site.v1",
                times_s=history.times_s,
                geodetic_latitude_deg=site.geodetic_latitude_deg,
                longitude_deg=site.longitude_deg,
                ellipsoidal_height_km=site.ellipsoidal_height_km,
                frame_context=frame_context,
            )
            link_id = f"{history.object_id}.to.{site.site_id}"
            link = evaluate_directed_link(
                DirectedLinkConfig(
                    analysis_id=f"{problem.analysis_id}.{design.design_id}.{link_id}",
                    link_id=link_id,
                    tx_terminal=LinkTerminal(
                        terminal_id=f"{history.object_id}.tx",
                        asset_id=history.object_id,
                        parent_frame="body",
                        quat_parent_from_terminal=_IDENTITY_QUATERNION,
                        pattern=TerminalPattern("constant", problem.link_budget["tx_gain_dbi"]),
                    ),
                    rx_terminal=LinkTerminal(
                        terminal_id=f"{site.site_id}.rx",
                        asset_id=site.site_id,
                        parent_frame="enu",
                        quat_parent_from_terminal=_IDENTITY_QUATERNION,
                        pattern=TerminalPattern("constant", problem.link_budget["rx_gain_dbi"]),
                    ),
                    carrier_frequency_hz=problem.link_budget["carrier_frequency_hz"],
                    tx_power_w=problem.link_budget["tx_power_w"],
                    data_rate_bps=problem.link_budget["data_rate_bps"],
                    system_noise_temperature_k=problem.link_budget["system_noise_temperature_k"],
                    required_eb_n0_db=problem.link_budget["required_eb_n0_db"],
                    tx_line_loss_db=problem.link_budget["tx_line_loss_db"],
                    rx_line_loss_db=problem.link_budget["rx_line_loss_db"],
                    misc_loss_db=problem.link_budget["misc_loss_db"],
                    min_fixed_site_elevation_rad=math.radians(problem.link_budget["minimum_elevation_deg"]),
                ),
                tx_history=spacecraft_endpoint,
                rx_history=site_endpoint,
                frame_context=frame_context,
            )
            union_available |= link.samples.available
            link_results.append(
                {
                    "link_id": link_id,
                    "sampled_available_fraction": link.summary["sampled_available_fraction"],
                    "estimated_delivered_data_bits": link.summary["estimated_delivered_data_bits"],
                    "semantic_sha256": link.semantic_sha256,
                }
            )
    intervals = np.diff(histories[0].times_s)
    network_availability = float(np.dot(union_available[:-1].astype(float), intervals) / problem.duration_s)
    coverage_fraction = float(coverage.summary["time_weighted_mean_covered_fraction"])
    score_components = {
        "coverage_service": problem.objective["coverage_weight"] * coverage_fraction,
        "network_service": problem.objective["network_weight"] * network_availability,
        "satellite_penalty": -problem.objective["satellite_penalty"] * design.satellite_count,
        "ground_site_penalty": -problem.objective["ground_site_penalty"] * len(design.ground_site_ids),
    }
    score = math.fsum(score_components.values())
    feasible = (
        coverage_fraction + 1.0e-15 >= problem.objective["minimum_coverage_fraction"]
        and network_availability + 1.0e-15 >= problem.objective["minimum_network_availability_fraction"]
    )
    return {
        "design_id": design.design_id,
        "rank": 0,
        "feasible": feasible,
        "score": score,
        "score_components": score_components,
        "generated_members": members,
        "coverage": {
            "time_weighted_mean_covered_fraction": coverage_fraction,
            "never_service_qualified_cell_count": coverage.summary["never_service_qualified_cell_count"],
            "maximum_multiplicity": coverage.summary["maximum_multiplicity"],
            "sample_times_s": coverage.times_s.tolist(),
            "instantaneous_covered_fraction": coverage.instantaneous_covered_fraction.tolist(),
            "covered_cell_count": coverage.covered_cell_count.tolist(),
            "interval_semantic_sha256": coverage.interval_semantic_sha256,
        },
        "network": {
            "union_sampled_available_fraction": network_availability,
            "sample_times_s": histories[0].times_s.tolist(),
            "union_available_by_sample": union_available.tolist(),
            "selected_ground_site_ids": list(design.ground_site_ids),
            "link_results": sorted(link_results, key=lambda item: item["link_id"]),
            "capacity_disposition": "unconstrained_per-link_estimate_not_a_scheduled_network_capacity_claim",
        },
    }


def solve_constellation_design(
    problem: ConstellationDesignProblem | Mapping[str, Any],
) -> ConstellationDesignResult:
    """Evaluate and rank only the caller-supplied bounded candidate inventory."""

    normalized = (
        problem if isinstance(problem, ConstellationDesignProblem) else ConstellationDesignProblem.from_mapping(problem)
    )
    results = [_evaluate_design(normalized, design) for design in normalized.designs]
    results.sort(key=lambda item: (-int(item["feasible"]), -float(item["score"]), str(item["design_id"])))
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    resource_estimate = {
        "candidate_count": len(normalized.designs),
        "sample_count": int(round(normalized.duration_s / normalized.sample_step_s)) + 1,
        "total_satellite_candidates": sum(item.satellite_count for item in normalized.designs),
        "coverage_cell_time_comparisons": sum(item.satellite_count for item in normalized.designs)
        * (int(round(normalized.duration_s / normalized.sample_step_s)) + 1)
        * healpix_npix(int(normalized.coverage["order"])),
        "link_samples": sum(item.satellite_count * len(item.ground_site_ids) for item in normalized.designs)
        * (int(round(normalized.duration_s / normalized.sample_step_s)) + 1),
    }
    input_digest = _semantic_sha256(normalized.to_dict())
    result_digest = _semantic_sha256(results)
    recommended_design_id = results[0]["design_id"] if results[0]["feasible"] else None
    evidence = {
        "schema_version": CONSTELLATION_DESIGN_EVIDENCE_SCHEMA,
        "analysis_id": normalized.analysis_id,
        "status": "complete",
        "input_semantic_sha256": input_digest,
        "result_semantic_sha256": result_digest,
        "ranking": [item["design_id"] for item in results],
        "recommended_design_id": recommended_design_id,
        "candidate_results": results,
        "resource_estimate": resource_estimate,
        "claim_limits": [
            "Ranking covers only the explicit candidate inventory; this is not a global optimizer.",
            "Circular Walker/shell initialization uses ideal phasing and deterministic ONP propagation.",
            "Coverage uses an ideal nadir-fixed conical sensor over the configured HEALPix grid.",
            "Ground links are same-epoch free-space engineering estimates without weather, terrain, interference, protocols, or scheduling.",
            "Union link availability does not model station capacity, conflicts, crosslinks, or routed data delivery.",
            "Evidence is analysis support, not operational qualification.",
        ],
    }
    return ConstellationDesignResult(normalized, evidence)


def _render_artifacts(result: ConstellationDesignResult) -> dict[str, bytes]:
    return {
        "normalized_problem.json": _json_bytes(result.problem.to_dict()),
        "constellation_design_evidence.json": _json_bytes(result.evidence),
    }


def write_constellation_design_artifacts(
    result: ConstellationDesignResult,
    output_dir: str | Path,
) -> ConstellationDesignArtifacts:
    """Atomically publish a closed, content-bound evidence directory."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise ConstellationDesignError("output_dir must not be a symbolic link.")
    destination = requested.resolve()
    if destination.exists() or destination.is_symlink():
        raise ConstellationDesignError(f"output_dir must be absent; refusing to replace {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = _render_artifacts(result)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.staging-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        receipts = []
        for name, content in rendered.items():
            (staging / name).write_bytes(content)
            receipts.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        manifest = {
            "schema_version": CONSTELLATION_DESIGN_EVIDENCE_SCHEMA,
            "analysis_id": result.problem.analysis_id,
            "status": result.evidence["status"],
            "input_semantic_sha256": result.evidence["input_semantic_sha256"],
            "result_semantic_sha256": result.evidence["result_semantic_sha256"],
            "recommended_design_id": result.evidence["recommended_design_id"],
            "artifacts": receipts,
            "claim_limits": result.evidence["claim_limits"],
        }
        (staging / "constellation_design_manifest.json").write_bytes(_json_bytes(manifest))
        try:
            os.rename(staging, destination)
        except FileExistsError as exc:
            raise ConstellationDesignError(f"output_dir appeared during publication: {destination}.") from exc
    return ConstellationDesignArtifacts(
        destination,
        destination / "constellation_design_manifest.json",
        destination / "normalized_problem.json",
        destination / "constellation_design_evidence.json",
    )


def _strict_json(content: bytes, field: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ConstellationDesignError(f"{field} contains forbidden non-finite constant {value}.")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ConstellationDesignError(f"{field} contains duplicate field {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(content.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConstellationDesignError(f"Could not parse {field}: {exc}") from exc


def verify_constellation_design_artifacts(evidence_dir: str | Path) -> dict[str, Any]:
    """Verify receipts, rerun the normalized problem, and compare authoritative evidence."""

    root = Path(evidence_dir).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ConstellationDesignError("evidence_dir must be a real directory.")
    required = {
        "constellation_design_manifest.json",
        "normalized_problem.json",
        "constellation_design_evidence.json",
    }
    actual = {path.name for path in root.iterdir()}
    if actual != required:
        raise ConstellationDesignError("Constellation-design evidence directory inventory is not exact.")
    try:
        manifest_content = read_regular_file_nofollow(
            root / "constellation_design_manifest.json", min_bytes=1, max_bytes=_MAX_ARTIFACT_BYTES
        )
        manifest = _strict_json(manifest_content, "constellation-design manifest")
    except (OSError, SafeReadError) as exc:
        raise ConstellationDesignError(f"Could not read constellation-design manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ConstellationDesignError("Constellation-design manifest must be a JSON object.")
    expected_manifest_fields = {
        "schema_version",
        "analysis_id",
        "status",
        "input_semantic_sha256",
        "result_semantic_sha256",
        "recommended_design_id",
        "artifacts",
        "claim_limits",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != CONSTELLATION_DESIGN_EVIDENCE_SCHEMA
    ):
        raise ConstellationDesignError("Constellation-design manifest contract is invalid.")
    receipts = manifest.get("artifacts")
    if not isinstance(receipts, list) or len(receipts) != 2:
        raise ConstellationDesignError("Constellation-design manifest requires two artifact receipts.")
    artifact_content: dict[str, bytes] = {}
    total = len(manifest_content)
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != {"path", "bytes", "sha256"}:
            raise ConstellationDesignError("Artifact receipt contract is invalid.")
        name = receipt.get("path")
        if name not in required - {"constellation_design_manifest.json"} or name in artifact_content:
            raise ConstellationDesignError("Artifact receipt inventory is invalid.")
        size = receipt.get("bytes")
        digest = receipt.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= _MAX_ARTIFACT_BYTES:
            raise ConstellationDesignError("Artifact receipt size is invalid.")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ConstellationDesignError("Artifact receipt SHA-256 is invalid.")
        try:
            content = read_regular_file_nofollow(root / name, min_bytes=1, max_bytes=_MAX_ARTIFACT_BYTES)
        except (OSError, SafeReadError) as exc:
            raise ConstellationDesignError(f"Could not read received artifact {name}: {exc}") from exc
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise ConstellationDesignError(f"Artifact receipt mismatch for {name}.")
        total += len(content)
        artifact_content[name] = content
    if total > _MAX_TOTAL_BYTES or set(artifact_content) != required - {"constellation_design_manifest.json"}:
        raise ConstellationDesignError("Constellation-design artifact inventory exceeds its public bounds.")
    problem_payload = _strict_json(artifact_content["normalized_problem.json"], "normalized problem")
    evidence_payload = _strict_json(
        artifact_content["constellation_design_evidence.json"], "constellation-design evidence"
    )
    if not isinstance(problem_payload, dict) or not isinstance(evidence_payload, dict):
        raise ConstellationDesignError("Normalized problem and evidence must be JSON objects.")
    authoritative = solve_constellation_design(ConstellationDesignProblem.from_mapping(problem_payload))
    expected = _render_artifacts(authoritative)
    if artifact_content != expected:
        raise ConstellationDesignError("Retained evidence differs from authoritative deterministic replay.")
    if (
        manifest.get("analysis_id") != authoritative.problem.analysis_id
        or manifest.get("status") != authoritative.evidence["status"]
        or manifest.get("input_semantic_sha256") != authoritative.evidence["input_semantic_sha256"]
        or manifest.get("result_semantic_sha256") != authoritative.evidence["result_semantic_sha256"]
        or manifest.get("recommended_design_id") != authoritative.evidence["recommended_design_id"]
        or manifest.get("claim_limits") != authoritative.evidence["claim_limits"]
    ):
        raise ConstellationDesignError("Manifest claims differ from authoritative deterministic replay.")
    return {
        "schema_version": CONSTELLATION_DESIGN_EVIDENCE_SCHEMA,
        "analysis_id": authoritative.problem.analysis_id,
        "status": "verified",
        "recommended_design_id": authoritative.evidence["recommended_design_id"],
        "input_semantic_sha256": authoritative.evidence["input_semantic_sha256"],
        "result_semantic_sha256": authoritative.evidence["result_semantic_sha256"],
    }


__all__ = [
    "CONSTELLATION_DESIGN_EVIDENCE_SCHEMA",
    "CONSTELLATION_DESIGN_PROBLEM_SCHEMA",
    "MAX_PUBLIC_DESIGNS",
    "MAX_PUBLIC_SATELLITES_PER_DESIGN",
    "ConstellationCandidate",
    "ConstellationDesignArtifacts",
    "ConstellationDesignError",
    "ConstellationDesignProblem",
    "ConstellationDesignResult",
    "GroundSite",
    "generate_constellation_members",
    "solve_constellation_design",
    "verify_constellation_design_artifacts",
    "write_constellation_design_artifacts",
]
