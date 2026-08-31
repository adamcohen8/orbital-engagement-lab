"""Deterministic orbit-coupled solar-array and battery feasibility analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.analysis.history_adapters import AnalysisHistory
from sim.analysis.mission_scheduling import verify_mission_scheduling_artifacts
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM, SUN_RADIUS_KM
from sim.utils.io import SafeReadError, read_regular_file_nofollow
from sim.utils.quaternion import quaternion_to_dcm_bn

SPACECRAFT_POWER_PROBLEM_SCHEMA = "oel.spacecraft_power_problem.v1"
SPACECRAFT_POWER_HISTORY_SCHEMA = "oel.spacecraft_power_history.v1"
SPACECRAFT_POWER_EVIDENCE_SCHEMA = "oel.spacecraft_power_evidence.v1"
SPACECRAFT_POWER_MANIFEST_SCHEMA = "oel.spacecraft_power_manifest.v1"

MAX_POWER_DURATION_S = 7.0 * 86400.0
MAX_POWER_SAMPLES = 200_000
MAX_POWER_ACTIVITIES = 1_000
MAX_INTEGRATION_STEP_S = 60.0
MAX_POWER_JSON_BYTES = 128 * 1024 * 1024
_EPS = 1.0e-12
_DIGEST_PATTERN = set("0123456789abcdef")
_ARTIFACT_FILES = {
    "normalized_problem.json",
    "normalized_history.json",
    "spacecraft_power_summary.json",
    "spacecraft_power_timeseries.csv",
    "spacecraft_power_intervals.csv",
    "spacecraft_power_events.csv",
}


class SpacecraftPowerError(ValueError):
    """Raised when power-analysis inputs or retained evidence are invalid."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpacecraftPowerError(f"{field} must be a non-empty string.")
    return value.strip()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise SpacecraftPowerError(f"{field} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SpacecraftPowerError(f"{field} must be a finite number.") from exc
    if not math.isfinite(result):
        raise SpacecraftPowerError(f"{field} must be a finite number.")
    return result


def _exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise SpacecraftPowerError(f"{field} is missing required fields: {', '.join(missing)}.")
    if extra:
        raise SpacecraftPowerError(f"{field} contains unknown fields: {', '.join(extra)}.")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpacecraftPowerError(f"{field} must be a JSON object.")
    return dict(value)


def _digest(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _valid_digest(value: Any, field: str) -> str:
    digest = _required_text(value, field).lower()
    if len(digest) != 64 or any(character not in _DIGEST_PATTERN for character in digest):
        raise SpacecraftPowerError(f"{field} must be a lowercase SHA-256 digest.")
    return digest


def _unit_vector(value: Any, field: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or np.any(~np.isfinite(vector)):
        raise SpacecraftPowerError(f"{field} must contain three finite values.")
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > 1.0e-10:
        raise SpacecraftPowerError(f"{field} must be normalized within 1e-10.")
    return tuple(float(item) for item in vector)


@dataclass(frozen=True)
class SolarArrayConfig:
    area_m2: float
    efficiency: float
    solar_flux_w_m2: float
    maximum_generation_w: float
    normal_body: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SolarArrayConfig:
        raw = _mapping(value, "solar_array")
        _exact_fields(
            raw,
            {"area_m2", "efficiency", "solar_flux_w_m2", "maximum_generation_w", "normal_body"},
            "solar_array",
        )
        return cls(
            area_m2=_finite(raw["area_m2"], "solar_array.area_m2"),
            efficiency=_finite(raw["efficiency"], "solar_array.efficiency"),
            solar_flux_w_m2=_finite(raw["solar_flux_w_m2"], "solar_array.solar_flux_w_m2"),
            maximum_generation_w=_finite(
                raw["maximum_generation_w"], "solar_array.maximum_generation_w"
            ),
            normal_body=_unit_vector(raw["normal_body"], "solar_array.normal_body"),
        )

    def __post_init__(self) -> None:
        for field in ("area_m2", "efficiency", "solar_flux_w_m2", "maximum_generation_w"):
            object.__setattr__(self, field, _finite(getattr(self, field), f"solar_array.{field}"))
        object.__setattr__(self, "normal_body", _unit_vector(self.normal_body, "solar_array.normal_body"))
        if self.area_m2 <= 0.0 or self.solar_flux_w_m2 <= 0.0 or self.maximum_generation_w <= 0.0:
            raise SpacecraftPowerError("Solar-array area, flux, and maximum generation must be positive.")
        if not 0.0 < self.efficiency <= 1.0:
            raise SpacecraftPowerError("solar_array.efficiency must lie in (0, 1].")


@dataclass(frozen=True)
class BatteryConfig:
    capacity_wh: float
    initial_soc_fraction: float
    minimum_soc_fraction: float
    maximum_soc_fraction: float
    maximum_charge_power_w: float
    maximum_discharge_power_w: float
    charge_efficiency: float
    discharge_efficiency: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BatteryConfig:
        raw = _mapping(value, "battery")
        fields = {
            "capacity_wh",
            "initial_soc_fraction",
            "minimum_soc_fraction",
            "maximum_soc_fraction",
            "maximum_charge_power_w",
            "maximum_discharge_power_w",
            "charge_efficiency",
            "discharge_efficiency",
        }
        _exact_fields(raw, fields, "battery")
        return cls(**{field: _finite(raw[field], f"battery.{field}") for field in fields})

    def __post_init__(self) -> None:
        for field in (
            "capacity_wh",
            "initial_soc_fraction",
            "minimum_soc_fraction",
            "maximum_soc_fraction",
            "maximum_charge_power_w",
            "maximum_discharge_power_w",
            "charge_efficiency",
            "discharge_efficiency",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), f"battery.{field}"))
        if self.capacity_wh <= 0.0:
            raise SpacecraftPowerError("battery.capacity_wh must be positive.")
        if not 0.0 <= self.minimum_soc_fraction < self.maximum_soc_fraction <= 1.0:
            raise SpacecraftPowerError(
                "Battery minimum and maximum SOC must satisfy 0 <= minimum < maximum <= 1."
            )
        if not self.minimum_soc_fraction <= self.initial_soc_fraction <= self.maximum_soc_fraction:
            raise SpacecraftPowerError("Battery initial SOC must lie inside the declared operating range.")
        if self.maximum_charge_power_w <= 0.0 or self.maximum_discharge_power_w <= 0.0:
            raise SpacecraftPowerError("Battery charge and discharge power limits must be positive.")
        if not 0.0 < self.charge_efficiency <= 1.0 or not 0.0 < self.discharge_efficiency <= 1.0:
            raise SpacecraftPowerError("Battery charge and discharge efficiencies must lie in (0, 1].")


@dataclass(frozen=True)
class PowerActivity:
    activity_id: str
    category: str
    start_s: float
    end_s: float
    load_power_w: float
    source_product_sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PowerActivity:
        raw = _mapping(value, "activity")
        _exact_fields(
            raw,
            {"activity_id", "category", "start_s", "end_s", "load_power_w", "source_product_sha256"},
            "activity",
        )
        digest = raw["source_product_sha256"]
        return cls(
            activity_id=_required_text(raw["activity_id"], "activity.activity_id"),
            category=_required_text(raw["category"], "activity.category").lower(),
            start_s=_finite(raw["start_s"], "activity.start_s"),
            end_s=_finite(raw["end_s"], "activity.end_s"),
            load_power_w=_finite(raw["load_power_w"], "activity.load_power_w"),
            source_product_sha256=(
                None if digest is None else _valid_digest(digest, "activity.source_product_sha256")
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity_id", _required_text(self.activity_id, "activity.activity_id"))
        object.__setattr__(self, "category", _required_text(self.category, "activity.category").lower())
        for field in ("start_s", "end_s", "load_power_w"):
            object.__setattr__(self, field, _finite(getattr(self, field), f"activity.{field}"))
        if self.source_product_sha256 is not None:
            object.__setattr__(
                self,
                "source_product_sha256",
                _valid_digest(self.source_product_sha256, "activity.source_product_sha256"),
            )
        if self.end_s <= self.start_s:
            raise SpacecraftPowerError(f"Activity {self.activity_id!r} must have positive duration.")
        if self.load_power_w < 0.0:
            raise SpacecraftPowerError(f"Activity {self.activity_id!r} load power must be nonnegative.")


@dataclass(frozen=True)
class SpacecraftPowerProblem:
    analysis_id: str
    asset_id: str
    epoch_jd_utc: float
    horizon_start_s: float
    horizon_end_s: float
    integration_step_s: float
    transition_time_tolerance_s: float
    transition_max_iterations: int
    shadow_model: str
    ephemeris_model: str
    orientation_mode: str
    solar_array: SolarArrayConfig
    battery: BatteryConfig
    base_load_w: float
    activities: tuple[PowerActivity, ...]
    schema_version: str = SPACECRAFT_POWER_PROBLEM_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SpacecraftPowerProblem:
        raw = _mapping(value, "spacecraft-power problem")
        fields = {
            "schema_version",
            "analysis_id",
            "asset_id",
            "epoch_jd_utc",
            "horizon_start_s",
            "horizon_end_s",
            "integration_step_s",
            "transition_time_tolerance_s",
            "transition_max_iterations",
            "shadow_model",
            "ephemeris_model",
            "orientation_mode",
            "solar_array",
            "battery",
            "base_load_w",
            "activities",
        }
        _exact_fields(raw, fields, "spacecraft-power problem")
        if raw["schema_version"] != SPACECRAFT_POWER_PROBLEM_SCHEMA:
            raise SpacecraftPowerError(f"Unsupported spacecraft-power schema {raw['schema_version']!r}.")
        activities_raw = raw["activities"]
        if not isinstance(activities_raw, list):
            raise SpacecraftPowerError("activities must be a JSON array.")
        if len(activities_raw) > MAX_POWER_ACTIVITIES:
            raise SpacecraftPowerError(f"At most {MAX_POWER_ACTIVITIES} activities are supported.")
        iterations = raw["transition_max_iterations"]
        if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, int):
            raise SpacecraftPowerError("transition_max_iterations must be an integer.")
        return cls(
            schema_version=raw["schema_version"],
            analysis_id=_required_text(raw["analysis_id"], "analysis_id"),
            asset_id=_required_text(raw["asset_id"], "asset_id"),
            epoch_jd_utc=_finite(raw["epoch_jd_utc"], "epoch_jd_utc"),
            horizon_start_s=_finite(raw["horizon_start_s"], "horizon_start_s"),
            horizon_end_s=_finite(raw["horizon_end_s"], "horizon_end_s"),
            integration_step_s=_finite(raw["integration_step_s"], "integration_step_s"),
            transition_time_tolerance_s=_finite(
                raw["transition_time_tolerance_s"], "transition_time_tolerance_s"
            ),
            transition_max_iterations=iterations,
            shadow_model=_required_text(raw["shadow_model"], "shadow_model").lower(),
            ephemeris_model=_required_text(raw["ephemeris_model"], "ephemeris_model").lower(),
            orientation_mode=_required_text(raw["orientation_mode"], "orientation_mode").lower(),
            solar_array=SolarArrayConfig.from_mapping(raw["solar_array"]),
            battery=BatteryConfig.from_mapping(raw["battery"]),
            base_load_w=_finite(raw["base_load_w"], "base_load_w"),
            activities=tuple(PowerActivity.from_mapping(item) for item in activities_raw),
        )

    def __post_init__(self) -> None:
        if self.schema_version != SPACECRAFT_POWER_PROBLEM_SCHEMA:
            raise SpacecraftPowerError(f"Unsupported spacecraft-power schema {self.schema_version!r}.")
        object.__setattr__(self, "analysis_id", _required_text(self.analysis_id, "analysis_id"))
        object.__setattr__(self, "asset_id", _required_text(self.asset_id, "asset_id"))
        for field in (
            "epoch_jd_utc",
            "horizon_start_s",
            "horizon_end_s",
            "integration_step_s",
            "transition_time_tolerance_s",
            "base_load_w",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if (
            isinstance(self.transition_max_iterations, (bool, np.bool_))
            or not isinstance(self.transition_max_iterations, int)
        ):
            raise SpacecraftPowerError("transition_max_iterations must be an integer.")
        object.__setattr__(self, "shadow_model", _required_text(self.shadow_model, "shadow_model").lower())
        object.__setattr__(
            self,
            "ephemeris_model",
            _required_text(self.ephemeris_model, "ephemeris_model").lower(),
        )
        object.__setattr__(
            self,
            "orientation_mode",
            _required_text(self.orientation_mode, "orientation_mode").lower(),
        )
        if not isinstance(self.solar_array, SolarArrayConfig) or not isinstance(self.battery, BatteryConfig):
            raise SpacecraftPowerError("solar_array and battery must be validated configuration objects.")
        try:
            activities = tuple(self.activities)
        except TypeError as exc:
            raise SpacecraftPowerError("activities must be an iterable of PowerActivity values.") from exc
        if len(activities) > MAX_POWER_ACTIVITIES or any(
            not isinstance(item, PowerActivity) for item in activities
        ):
            raise SpacecraftPowerError("activities must contain validated PowerActivity values within the limit.")
        object.__setattr__(self, "activities", activities)
        duration = self.horizon_end_s - self.horizon_start_s
        if not 0.0 < duration <= MAX_POWER_DURATION_S:
            raise SpacecraftPowerError(f"Power horizon must be positive and at most {MAX_POWER_DURATION_S} seconds.")
        if not 0.0 < self.integration_step_s <= MAX_INTEGRATION_STEP_S:
            raise SpacecraftPowerError(
                f"integration_step_s must lie in (0, {MAX_INTEGRATION_STEP_S}]."
            )
        count = int(math.ceil(duration / self.integration_step_s)) + 1
        if count > MAX_POWER_SAMPLES:
            raise SpacecraftPowerError(f"The integration grid may not exceed {MAX_POWER_SAMPLES} samples.")
        if not 0.0 < self.transition_time_tolerance_s < self.integration_step_s:
            raise SpacecraftPowerError(
                "transition_time_tolerance_s must be positive and smaller than integration_step_s."
            )
        if self.transition_max_iterations <= 0:
            raise SpacecraftPowerError("transition_max_iterations must be positive.")
        if self.shadow_model not in {"conical", "cylindrical", "none"}:
            raise SpacecraftPowerError("shadow_model must be conical, cylindrical, or none.")
        if self.ephemeris_model not in {"analytic_simple", "analytic_enhanced"}:
            raise SpacecraftPowerError("ephemeris_model must be analytic_simple or analytic_enhanced.")
        if self.orientation_mode not in {"sun_tracking_ideal", "history_body_fixed"}:
            raise SpacecraftPowerError(
                "orientation_mode must be sun_tracking_ideal or history_body_fixed."
            )
        if self.base_load_w < 0.0:
            raise SpacecraftPowerError("base_load_w must be nonnegative.")
        identifiers = [item.activity_id for item in self.activities]
        if len(identifiers) != len(set(identifiers)):
            raise SpacecraftPowerError("Activity IDs must be unique.")
        for activity in self.activities:
            if activity.start_s < self.horizon_start_s or activity.end_s > self.horizon_end_s:
                raise SpacecraftPowerError(
                    f"Activity {activity.activity_id!r} lies outside the power-analysis horizon."
                )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["solar_array"]["normal_body"] = list(self.solar_array.normal_body)
        value["activities"] = [
            asdict(item) for item in sorted(self.activities, key=lambda item: (item.start_s, item.end_s, item.activity_id))
        ]
        return value


@dataclass(frozen=True)
class PowerSample:
    time_s: float
    illumination_class: str
    shadow_factor: float
    incidence_cosine: float
    generation_power_w: float
    load_power_w: float
    battery_energy_wh: float
    battery_soc_fraction: float
    cumulative_generated_wh: float
    cumulative_load_wh: float
    cumulative_served_load_wh: float
    cumulative_unmet_load_wh: float
    cumulative_curtailed_wh: float


@dataclass(frozen=True)
class PowerInterval:
    interval_index: int
    start_s: float
    end_s: float
    duration_s: float
    illumination_class: str


@dataclass(frozen=True)
class PowerEvent:
    event_index: int
    time_s: float
    event_kind: str
    from_state: str
    to_state: str
    disposition: str
    bracket_start_s: float
    bracket_end_s: float
    original_bracket_start_s: float
    original_bracket_end_s: float
    iterations: int


@dataclass(frozen=True)
class _RefinedTransition:
    time_s: float
    bracket_start_s: float
    bracket_end_s: float
    original_bracket_start_s: float
    original_bracket_end_s: float
    iterations: int


@dataclass(frozen=True)
class SpacecraftPowerResult:
    problem: SpacecraftPowerProblem
    history_semantic_sha256: str
    samples: tuple[PowerSample, ...]
    intervals: tuple[PowerInterval, ...]
    events: tuple[PowerEvent, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class SpacecraftPowerArtifacts:
    output_dir: Path
    manifest_json: Path
    problem_json: Path
    history_json: Path
    summary_json: Path
    timeseries_csv: Path
    intervals_csv: Path
    events_csv: Path


def power_history_to_dict(history: AnalysisHistory) -> dict[str, Any]:
    samples = []
    for index, time_s in enumerate(history.times_s):
        samples.append(
            {
                "time_s": float(time_s),
                "position_eci_km": [float(item) for item in history.position_eci_km[index]],
                "velocity_eci_km_s": [float(item) for item in history.velocity_eci_km_s[index]],
                "attitude_quat_bn": (
                    None
                    if history.attitude_quat_bn is None
                    else [float(item) for item in history.attitude_quat_bn[index]]
                ),
            }
        )
    return {
        "schema_version": SPACECRAFT_POWER_HISTORY_SCHEMA,
        "asset_id": history.object_id,
        "epoch_jd_utc": history.initial_jd_utc,
        "frame": "eci",
        "product_kind": history.product_kind,
        "state_provider_id": history.state_provider_id,
        "attitude_source_kind": history.attitude_source_kind,
        "attitude_provider_id": history.attitude_provider_id,
        "refinement_source": history.refinement_source,
        "samples": samples,
    }


def power_history_from_mapping(value: Mapping[str, Any]) -> AnalysisHistory:
    raw = _mapping(value, "spacecraft-power history")
    fields = {
        "schema_version",
        "asset_id",
        "epoch_jd_utc",
        "frame",
        "product_kind",
        "state_provider_id",
        "attitude_source_kind",
        "attitude_provider_id",
        "refinement_source",
        "samples",
    }
    _exact_fields(raw, fields, "spacecraft-power history")
    if raw["schema_version"] != SPACECRAFT_POWER_HISTORY_SCHEMA:
        raise SpacecraftPowerError(f"Unsupported spacecraft-power history schema {raw['schema_version']!r}.")
    samples = raw["samples"]
    if not isinstance(samples, list) or not 2 <= len(samples) <= MAX_POWER_SAMPLES:
        raise SpacecraftPowerError(f"History samples must contain between 2 and {MAX_POWER_SAMPLES} rows.")
    times: list[float] = []
    positions: list[Sequence[float]] = []
    velocities: list[Sequence[float]] = []
    attitudes: list[Sequence[float]] = []
    attitude_present: bool | None = None
    for index, item in enumerate(samples):
        sample = _mapping(item, f"samples[{index}]")
        _exact_fields(
            sample,
            {"time_s", "position_eci_km", "velocity_eci_km_s", "attitude_quat_bn"},
            f"samples[{index}]",
        )
        times.append(_finite(sample["time_s"], f"samples[{index}].time_s"))
        positions.append(sample["position_eci_km"])
        velocities.append(sample["velocity_eci_km_s"])
        present = sample["attitude_quat_bn"] is not None
        if attitude_present is None:
            attitude_present = present
        elif attitude_present != present:
            raise SpacecraftPowerError("Every history sample must use the same attitude presence.")
        if present:
            attitudes.append(sample["attitude_quat_bn"])
    try:
        return AnalysisHistory(
            object_id=_required_text(raw["asset_id"], "history.asset_id"),
            product_kind=_required_text(raw["product_kind"], "history.product_kind"),
            state_provider_id=_required_text(raw["state_provider_id"], "history.state_provider_id"),
            frame=_required_text(raw["frame"], "history.frame"),
            initial_jd_utc=_finite(raw["epoch_jd_utc"], "history.epoch_jd_utc"),
            times_s=np.asarray(times, dtype=float),
            position_eci_km=np.asarray(positions, dtype=float),
            velocity_eci_km_s=np.asarray(velocities, dtype=float),
            attitude_quat_bn=(np.asarray(attitudes, dtype=float) if attitude_present else None),
            attitude_source_kind=_required_text(
                raw["attitude_source_kind"], "history.attitude_source_kind"
            ),
            attitude_provider_id=(
                None
                if raw["attitude_provider_id"] is None
                else _required_text(raw["attitude_provider_id"], "history.attitude_provider_id")
            ),
            refinement_source=_required_text(raw["refinement_source"], "history.refinement_source"),
        )
    except (TypeError, ValueError) as exc:
        raise SpacecraftPowerError(f"Invalid spacecraft-power history: {exc}") from exc


def _illumination_class(shadow_factor: float) -> str:
    if shadow_factor <= _EPS:
        return "umbra"
    if shadow_factor >= 1.0 - _EPS:
        return "sunlight"
    return "penumbra"


def validate_spacecraft_power_inputs(
    problem: SpacecraftPowerProblem | Mapping[str, Any],
    history: AnalysisHistory | Mapping[str, Any],
) -> tuple[SpacecraftPowerProblem, AnalysisHistory]:
    """Normalize and validate one mutually compatible problem/history pair."""

    parsed = problem if isinstance(problem, SpacecraftPowerProblem) else SpacecraftPowerProblem.from_mapping(problem)
    normalized_history = history if isinstance(history, AnalysisHistory) else power_history_from_mapping(history)
    if normalized_history.object_id != parsed.asset_id:
        raise SpacecraftPowerError("Power problem asset_id does not match the supplied history.")
    if abs(normalized_history.initial_jd_utc - parsed.epoch_jd_utc) > 1.0e-12:
        raise SpacecraftPowerError("Power problem epoch_jd_utc does not match the supplied history.")
    if parsed.horizon_start_s < normalized_history.times_s[0] or parsed.horizon_end_s > normalized_history.times_s[-1]:
        raise SpacecraftPowerError("Power-analysis horizon lies outside the retained history.")
    if parsed.orientation_mode == "history_body_fixed" and normalized_history.attitude_quat_bn is None:
        raise SpacecraftPowerError("history_body_fixed orientation requires retained attitude evidence.")
    return parsed, normalized_history


def _regular_times(problem: SpacecraftPowerProblem, history: AnalysisHistory) -> list[float]:
    values = {problem.horizon_start_s, problem.horizon_end_s}
    count = int(math.ceil((problem.horizon_end_s - problem.horizon_start_s) / problem.integration_step_s))
    values.update(
        min(problem.horizon_start_s + index * problem.integration_step_s, problem.horizon_end_s)
        for index in range(count + 1)
    )
    values.update(
        float(value)
        for value in history.times_s
        if problem.horizon_start_s <= float(value) <= problem.horizon_end_s
    )
    for activity in problem.activities:
        values.update((activity.start_s, activity.end_s))
    result = sorted(values)
    if len(result) > MAX_POWER_SAMPLES:
        raise SpacecraftPowerError(f"The combined power-analysis grid exceeds {MAX_POWER_SAMPLES} samples.")
    return result


def _instantaneous(
    problem: SpacecraftPowerProblem,
    history: AnalysisHistory,
    time_s: float,
) -> tuple[float, float, float, float, str]:
    state = history.state_at(time_s)
    env = {
        "jd_utc_start": problem.epoch_jd_utc,
        "ephemeris_mode": problem.ephemeris_model,
        "srp_shadow_model": problem.shadow_model,
    }
    geometry = resolve_srp_geometry(state.position_eci_km, time_s, env)
    shadow = float(
        srp_shadow_factor(
            state.position_eci_km,
            time_s,
            env,
            srp_geometry=geometry,
        )
    )
    if problem.orientation_mode == "sun_tracking_ideal":
        incidence = 1.0
    else:
        if state.attitude_quat_bn is None:
            raise SpacecraftPowerError(
                "history_body_fixed orientation requires achieved, replay, or analytic-ideal attitude history."
            )
        body_from_eci = quaternion_to_dcm_bn(state.attitude_quat_bn)
        normal_eci = body_from_eci.T @ np.asarray(problem.solar_array.normal_body, dtype=float)
        sun_direction = np.asarray(geometry["sun_dir_sc_eci"], dtype=float)
        incidence = max(0.0, float(np.dot(normal_eci, sun_direction)))
    generation = min(
        problem.solar_array.maximum_generation_w,
        problem.solar_array.area_m2
        * problem.solar_array.efficiency
        * problem.solar_array.solar_flux_w_m2
        * float(geometry["distance_scale"])
        * shadow
        * incidence,
    )
    load = problem.base_load_w + sum(
        item.load_power_w for item in problem.activities if item.start_s <= time_s < item.end_s
    )
    return shadow, incidence, generation, load, _illumination_class(shadow)


def _refine_membership_change(
    problem: SpacecraftPowerProblem,
    history: AnalysisHistory,
    left: float,
    right: float,
    predicate: Any,
    *,
    original_left: float,
    original_right: float,
) -> _RefinedTransition:
    left_value = predicate(_instantaneous(problem, history, left)[0])
    right_value = predicate(_instantaneous(problem, history, right)[0])
    if left_value == right_value:
        raise RuntimeError("Transition refinement requires a bracketed membership change.")
    iterations = 0
    while right - left > problem.transition_time_tolerance_s and iterations < problem.transition_max_iterations:
        midpoint = 0.5 * (left + right)
        middle_value = predicate(_instantaneous(problem, history, midpoint)[0])
        if middle_value == left_value:
            left = midpoint
        else:
            right = midpoint
        iterations += 1
    if right - left > problem.transition_time_tolerance_s:
        raise SpacecraftPowerError(
            "Illumination transition refinement exhausted transition_max_iterations "
            f"with a {right - left:.12g}-second bracket, larger than the declared "
            f"{problem.transition_time_tolerance_s:.12g}-second tolerance."
        )
    return _RefinedTransition(
        time_s=0.5 * (left + right),
        bracket_start_s=left,
        bracket_end_s=right,
        original_bracket_start_s=original_left,
        original_bracket_end_s=original_right,
        iterations=iterations,
    )


def _transition_metric(
    problem: SpacecraftPowerProblem,
    history: AnalysisHistory,
    time_s: float,
    boundary: str,
) -> float:
    """Continuous signed metric; negative means membership in the named shadow phase."""

    state = history.state_at(time_s)
    env = {
        "jd_utc_start": problem.epoch_jd_utc,
        "ephemeris_mode": problem.ephemeris_model,
        "srp_shadow_model": problem.shadow_model,
    }
    geometry = resolve_srp_geometry(state.position_eci_km, time_s, env)
    r_sc = np.asarray(geometry["r_sc_eci_km"], dtype=float)
    r_norm = float(geometry["r_sc_norm_km"])
    if r_norm <= EARTH_RADIUS_KM:
        return -1.0
    if problem.shadow_model == "cylindrical":
        r_sun = np.asarray(geometry["sun_pos_eci_km"], dtype=float)
        sun_norm = float(geometry["sun_pos_norm_km"])
        if sun_norm <= 0.0:
            return 1.0
        s_hat = r_sun / sun_norm
        along = float(np.dot(r_sc, s_hat))
        cross_track2 = max(0.0, float(np.dot(r_sc, r_sc)) - along * along)
        return max(
            along / EARTH_RADIUS_KM,
            (cross_track2 - EARTH_RADIUS_KM**2) / EARTH_RADIUS_KM**2,
        )
    if problem.shadow_model == "none":
        return 1.0
    rho_norm = float(geometry["rho_norm_km"])
    if rho_norm <= 0.0:
        return 1.0
    alpha = float(np.arcsin(np.clip(EARTH_RADIUS_KM / r_norm, -1.0, 1.0)))
    beta = float(np.arcsin(np.clip(SUN_RADIUS_KM / rho_norm, -1.0, 1.0)))
    u_earth = -r_sc / r_norm
    u_sun = np.asarray(geometry["sun_dir_sc_eci"], dtype=float)
    gamma = float(np.arccos(np.clip(float(np.dot(u_earth, u_sun)), -1.0, 1.0)))
    if boundary == "partial_shadow":
        return gamma - (alpha + beta)
    if alpha <= beta:
        return gamma + beta - alpha
    return gamma - (alpha - beta)


def _bounded_extreme(
    function: Any,
    left: float,
    right: float,
    *,
    maximize: bool,
) -> float:
    """Deterministically locate one smooth local extreme inside a bounded bracket."""

    sign = -1.0 if maximize else 1.0
    ratio = 0.5 * (math.sqrt(5.0) - 1.0)
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = sign * float(function(x1))
    f2 = sign * float(function(x2))
    for _ in range(64):
        if right - left <= 1.0e-9:
            break
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = sign * float(function(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = sign * float(function(x2))
    return 0.5 * (left + right)


def _transition_probes(
    problem: SpacecraftPowerProblem,
    history: AnalysisHistory,
    left: float,
    right: float,
    boundary: str,
) -> list[float]:
    """Find sampled and interior-extremum probes that expose whole sub-step shadow phases."""

    probes = [left + (right - left) * index / 4.0 for index in range(5)]
    def metric(time_s: float) -> float:
        return _transition_metric(problem, history, time_s, boundary)

    values = [metric(time_s) for time_s in probes]
    extrema: list[float] = []
    scale = max(1.0, *(abs(value) for value in values))
    equality_tolerance = 1.0e-12 * scale
    for target, maximize in ((min(values), False), (max(values), True)):
        indices = [
            index for index, value in enumerate(values)
            if abs(value - target) <= equality_tolerance
        ]
        first, last = indices[0], indices[-1]
        if first > 0 and last < len(probes) - 1:
            extrema.append(
                _bounded_extreme(
                    metric,
                    probes[first - 1],
                    probes[last + 1],
                    maximize=maximize,
                )
            )
    return sorted({*probes, *extrema})


def _grid_with_refined_transitions(
    problem: SpacecraftPowerProblem,
    history: AnalysisHistory,
) -> tuple[list[float], list[_RefinedTransition]]:
    base = _regular_times(problem, history)
    if problem.shadow_model == "none":
        return base, []
    refined: list[_RefinedTransition] = []
    boundaries = (("partial_shadow", lambda shadow: shadow < 1.0 - _EPS),)
    if problem.shadow_model == "conical":
        boundaries += (("umbra", lambda shadow: shadow <= _EPS),)
    for left, right in zip(base, base[1:]):
        for boundary, predicate in boundaries:
            probes = _transition_probes(problem, history, left, right, boundary)
            memberships = [predicate(_instantaneous(problem, history, time_s)[0]) for time_s in probes]
            for bracket_left, bracket_right, left_value, right_value in zip(
                probes, probes[1:], memberships, memberships[1:]
            ):
                if left_value != right_value:
                    refined.append(
                        _refine_membership_change(
                            problem,
                            history,
                            bracket_left,
                            bracket_right,
                            predicate,
                            original_left=left,
                            original_right=right,
                        )
                    )
    times = sorted({*base, *(item.time_s for item in refined)})
    if len(times) > MAX_POWER_SAMPLES:
        raise SpacecraftPowerError(f"The refined power-analysis grid exceeds {MAX_POWER_SAMPLES} samples.")
    return times, sorted(refined, key=lambda item: item.time_s)


def _illumination_intervals(
    problem: SpacecraftPowerProblem,
    history: AnalysisHistory,
    times: Sequence[float],
) -> tuple[PowerInterval, ...]:
    raw: list[tuple[float, float, str]] = []
    for left, right in zip(times, times[1:]):
        midpoint = 0.5 * (left + right)
        raw.append((left, right, _instantaneous(problem, history, midpoint)[4]))
    grouped: list[tuple[float, float, str]] = []
    for left, right, kind in raw:
        if grouped and grouped[-1][2] == kind and abs(grouped[-1][1] - left) <= 1.0e-9:
            grouped[-1] = (grouped[-1][0], right, kind)
        else:
            grouped.append((left, right, kind))
    return tuple(
        PowerInterval(index, left, right, right - left, kind)
        for index, (left, right, kind) in enumerate(grouped)
    )


def assess_spacecraft_power(
    problem: SpacecraftPowerProblem | Mapping[str, Any],
    history: AnalysisHistory | Mapping[str, Any],
) -> SpacecraftPowerResult:
    """Assess deterministic solar generation, loads, and battery feasibility."""

    parsed, normalized_history = validate_spacecraft_power_inputs(problem, history)

    times, refined = _grid_with_refined_transitions(parsed, normalized_history)
    intervals = _illumination_intervals(parsed, normalized_history, times)
    battery = parsed.battery
    minimum_energy = battery.minimum_soc_fraction * battery.capacity_wh
    maximum_energy = battery.maximum_soc_fraction * battery.capacity_wh
    stored = battery.initial_soc_fraction * battery.capacity_wh
    initial_energy = stored
    cumulative_generated = 0.0
    cumulative_load = 0.0
    cumulative_served = 0.0
    cumulative_unmet = 0.0
    cumulative_curtailed = 0.0
    charged_stored = 0.0
    discharged_stored = 0.0
    minimum_soc = stored / battery.capacity_wh
    maximum_soc = minimum_soc
    events: list[PowerEvent] = []

    initial = _instantaneous(parsed, normalized_history, times[0])
    samples = [
        PowerSample(
            time_s=times[0],
            illumination_class=initial[4],
            shadow_factor=initial[0],
            incidence_cosine=initial[1],
            generation_power_w=initial[2],
            load_power_w=initial[3],
            battery_energy_wh=stored,
            battery_soc_fraction=stored / battery.capacity_wh,
            cumulative_generated_wh=0.0,
            cumulative_load_wh=0.0,
            cumulative_served_load_wh=0.0,
            cumulative_unmet_load_wh=0.0,
            cumulative_curtailed_wh=0.0,
        )
    ]
    for left, right in zip(times, times[1:]):
        midpoint = 0.5 * (left + right)
        dt_hours = (right - left) / 3600.0
        left_values = _instantaneous(parsed, normalized_history, left)
        middle_values = _instantaneous(parsed, normalized_history, midpoint)
        right_values = _instantaneous(parsed, normalized_history, right)
        generation_w = (left_values[2] + 4.0 * middle_values[2] + right_values[2]) / 6.0
        load_w = middle_values[3]
        generated_wh = generation_w * dt_hours
        load_wh = load_w * dt_hours
        cumulative_generated += generated_wh
        cumulative_load += load_wh
        before = stored
        direct_power_w = min(generation_w, load_w)
        direct_wh = direct_power_w * dt_hours
        served_wh = direct_wh
        curtailed_wh = 0.0
        unmet_wh = 0.0
        if generation_w >= load_w:
            surplus_w = generation_w - load_w
            charge_power_w = min(surplus_w, battery.maximum_charge_power_w)
            possible_stored_wh = charge_power_w * battery.charge_efficiency * dt_hours
            actual_stored_wh = min(possible_stored_wh, max(0.0, maximum_energy - stored))
            stored += actual_stored_wh
            charged_stored += actual_stored_wh
            charge_input_wh = actual_stored_wh / battery.charge_efficiency
            curtailed_wh = max(0.0, surplus_w * dt_hours - charge_input_wh)
            if before < maximum_energy - 1.0e-12 and stored >= maximum_energy - 1.0e-12:
                rate = charge_power_w * battery.charge_efficiency
                event_time = right if rate <= 0.0 else min(right, left + 3600.0 * (maximum_energy - before) / rate)
                events.append(
                    PowerEvent(
                        -1, event_time, "battery_maximum_soc", "charging", "saturated",
                        "integrated", left, right, left, right, 0,
                    )
                )
        else:
            deficit_w = load_w - generation_w
            discharge_bus_power_w = min(deficit_w, battery.maximum_discharge_power_w)
            available_bus_wh = max(0.0, stored - minimum_energy) * battery.discharge_efficiency
            delivered_bus_wh = min(discharge_bus_power_w * dt_hours, available_bus_wh)
            removed_stored_wh = delivered_bus_wh / battery.discharge_efficiency
            stored -= removed_stored_wh
            discharged_stored += removed_stored_wh
            served_wh += delivered_bus_wh
            unmet_wh = max(0.0, deficit_w * dt_hours - delivered_bus_wh)
            if before > minimum_energy + 1.0e-12 and stored <= minimum_energy + 1.0e-12:
                rate = discharge_bus_power_w / battery.discharge_efficiency
                event_time = right if rate <= 0.0 else min(right, left + 3600.0 * (before - minimum_energy) / rate)
                events.append(
                    PowerEvent(
                        -1, event_time, "battery_minimum_soc", "discharging",
                        "reserve_reached", "integrated", left, right, left, right, 0,
                    )
                )
            if unmet_wh > 1.0e-12 and cumulative_unmet <= 1.0e-12:
                immediate_shortfall = deficit_w > battery.maximum_discharge_power_w + 1.0e-12
                if immediate_shortfall or before <= minimum_energy + 1.0e-12:
                    unmet_start_s = left
                else:
                    discharge_bus_power_w = min(deficit_w, battery.maximum_discharge_power_w)
                    stored_rate_wh_per_hour = discharge_bus_power_w / battery.discharge_efficiency
                    unmet_start_s = min(
                        right,
                        left + 3600.0 * (before - minimum_energy) / stored_rate_wh_per_hour,
                    )
                events.append(
                    PowerEvent(
                        -1, unmet_start_s, "unmet_load_start", "served", "unmet",
                        "integrated", left, right, left, right, 0,
                    )
                )
        cumulative_served += served_wh
        cumulative_unmet += unmet_wh
        cumulative_curtailed += curtailed_wh
        soc = stored / battery.capacity_wh
        minimum_soc = min(minimum_soc, soc)
        maximum_soc = max(maximum_soc, soc)
        end_values = _instantaneous(parsed, normalized_history, right)
        samples.append(
            PowerSample(
                time_s=right,
                illumination_class=end_values[4],
                shadow_factor=end_values[0],
                incidence_cosine=end_values[1],
                generation_power_w=end_values[2],
                load_power_w=end_values[3],
                battery_energy_wh=stored,
                battery_soc_fraction=soc,
                cumulative_generated_wh=cumulative_generated,
                cumulative_load_wh=cumulative_load,
                cumulative_served_load_wh=cumulative_served,
                cumulative_unmet_load_wh=cumulative_unmet,
                cumulative_curtailed_wh=cumulative_curtailed,
            )
        )

    for previous, following in zip(intervals, intervals[1:]):
        boundary = previous.end_s
        match = min(refined, key=lambda item: abs(item.time_s - boundary), default=None)
        matched = match is not None and abs(match.time_s - boundary) <= parsed.transition_time_tolerance_s
        events.append(
            PowerEvent(
                -1,
                boundary,
                "illumination_transition",
                previous.illumination_class,
                following.illumination_class,
                "provider_refined" if matched else "sample_bounded",
                match.bracket_start_s if matched else boundary,
                match.bracket_end_s if matched else boundary,
                match.original_bracket_start_s if matched else boundary,
                match.original_bracket_end_s if matched else boundary,
                match.iterations if matched else 0,
            )
        )
    ordered_events = tuple(
        replace(item, event_index=index)
        for index, item in enumerate(sorted(events, key=lambda item: (item.time_s, item.event_kind, item.to_state)))
    )
    history_payload = power_history_to_dict(normalized_history)
    history_hash = _digest(history_payload)
    problem_hash = _digest(parsed.to_dict())
    battery_residual = initial_energy + charged_stored - discharged_stored - stored
    bus_residual = cumulative_generated + discharged_stored * battery.discharge_efficiency - cumulative_served
    bus_residual -= charged_stored / battery.charge_efficiency + cumulative_curtailed
    load_residual = cumulative_load - cumulative_served - cumulative_unmet
    feasible = cumulative_unmet <= 1.0e-9
    result_hash = _digest(
        {
            "problem_sha256": problem_hash,
            "history_sha256": history_hash,
            "samples": [asdict(item) for item in samples],
            "intervals": [asdict(item) for item in intervals],
            "events": [asdict(item) for item in ordered_events],
        }
    )
    summary = {
        "schema_version": SPACECRAFT_POWER_EVIDENCE_SCHEMA,
        "analysis_id": parsed.analysis_id,
        "asset_id": parsed.asset_id,
        "status": "completed",
        "feasibility": "feasible" if feasible else "infeasible",
        "model": "deterministic_sampled_solar_array_battery_v1",
        "problem_semantic_sha256": problem_hash,
        "history_semantic_sha256": history_hash,
        "result_semantic_sha256": result_hash,
        "sample_count": len(samples),
        "illumination_interval_count": len(intervals),
        "event_count": len(ordered_events),
        "totals": {
            "generated_energy_wh": cumulative_generated,
            "load_energy_wh": cumulative_load,
            "served_load_energy_wh": cumulative_served,
            "unmet_load_energy_wh": cumulative_unmet,
            "curtailed_energy_wh": cumulative_curtailed,
            "charged_battery_energy_wh": charged_stored,
            "discharged_battery_energy_wh": discharged_stored,
        },
        "battery": {
            "initial_energy_wh": initial_energy,
            "final_energy_wh": stored,
            "initial_soc_fraction": battery.initial_soc_fraction,
            "final_soc_fraction": stored / battery.capacity_wh,
            "minimum_soc_fraction": minimum_soc,
            "maximum_soc_fraction": maximum_soc,
            "minimum_soc_margin_fraction": minimum_soc - battery.minimum_soc_fraction,
        },
        "conservation_residuals_wh": {
            "battery_storage": battery_residual,
            "power_bus": bus_residual,
            "load_service": load_residual,
        },
        "source_product_sha256s": sorted(
            {item.source_product_sha256 for item in parsed.activities if item.source_product_sha256}
        ),
        "claim_limits": [
            "This is deterministic engineering evidence for one supplied state history and declared load timeline.",
            "Ideal Sun tracking is an explicit assumption when selected; body-fixed mode requires retained attitude evidence.",
            "The v1 model excludes thermal state, degradation, self-shadowing, regulator detail, uncertainty, and hardware qualification.",
            "A feasible result is not operational authorization or flight qualification.",
        ],
    }
    return SpacecraftPowerResult(parsed, history_hash, tuple(samples), intervals, ordered_events, summary)


def problem_with_mission_schedule(
    problem: SpacecraftPowerProblem | Mapping[str, Any],
    schedule_dir: str | Path,
    *,
    activity_power_w: Mapping[str, float],
) -> SpacecraftPowerProblem:
    """Add loads from one verified exact mission schedule without changing the schedule."""

    parsed = problem if isinstance(problem, SpacecraftPowerProblem) else SpacecraftPowerProblem.from_mapping(problem)
    root = Path(schedule_dir).expanduser()
    if root.is_symlink():
        raise SpacecraftPowerError("Mission-schedule evidence directory must not be a symbolic link.")
    root = root.resolve()
    try:
        verified = verify_mission_scheduling_artifacts(root)
    except ValueError as exc:
        raise SpacecraftPowerError(f"Mission-schedule evidence did not verify: {exc}") from exc
    rows = verified.get("activities")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise SpacecraftPowerError(
            "Mission-schedule verification did not return authoritative activity records."
        )
    powers = {key: _finite(value, f"activity_power_w.{key}") for key, value in activity_power_w.items()}
    if set(powers) != {"observation", "downlink"} or any(value < 0.0 for value in powers.values()):
        raise SpacecraftPowerError("activity_power_w must define nonnegative observation and downlink loads.")
    additions = []
    source_hash = _valid_digest(verified["schedule_semantic_sha256"], "schedule_semantic_sha256")
    for row in rows:
        if row.get("asset_id") != parsed.asset_id:
            continue
        category = _required_text(row.get("kind"), "schedule kind").lower()
        if category not in powers:
            raise SpacecraftPowerError(f"Verified schedule contains unsupported activity kind {category!r}.")
        additions.append(
            PowerActivity(
                activity_id=f"schedule-{_required_text(row.get('opportunity_id'), 'opportunity_id')}",
                category=category,
                start_s=_finite(row.get("start_s"), "schedule start_s"),
                end_s=_finite(row.get("end_s"), "schedule end_s"),
                load_power_w=powers[category],
                source_product_sha256=source_hash,
            )
        )
    if not additions:
        raise SpacecraftPowerError(f"Verified schedule contains no selected activities for {parsed.asset_id!r}.")
    return SpacecraftPowerProblem.from_mapping(
        {**parsed.to_dict(), "activities": [asdict(item) for item in (*parsed.activities, *additions)]}
    )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(values: Sequence[Any], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for value in values:
        writer.writerow(asdict(value))
    return stream.getvalue().encode("utf-8")


def _receipt(path: Path, root: Path) -> dict[str, Any]:
    content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=MAX_POWER_JSON_BYTES)
    return {"path": path.relative_to(root).as_posix(), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def write_spacecraft_power_artifacts(
    result: SpacecraftPowerResult,
    history: AnalysisHistory,
    output_dir: str | Path,
) -> SpacecraftPowerArtifacts:
    """Atomically write one exact, replayable spacecraft-power evidence directory."""

    supplied_history_hash = _digest(power_history_to_dict(history))
    if supplied_history_hash != result.history_semantic_sha256:
        raise SpacecraftPowerError("Artifact history does not match the assessed spacecraft-power result.")

    destination_input = Path(output_dir).expanduser()
    if destination_input.is_symlink():
        raise SpacecraftPowerError("output_dir must not be a symbolic link.")
    destination = destination_input.resolve()
    if destination.exists():
        raise SpacecraftPowerError(f"output_dir must not already exist: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        files = {
            "normalized_problem.json": _json_bytes(result.problem.to_dict()),
            "normalized_history.json": _json_bytes(power_history_to_dict(history)),
            "spacecraft_power_summary.json": _json_bytes(result.summary),
            "spacecraft_power_timeseries.csv": _csv_bytes(
                result.samples, tuple(PowerSample.__dataclass_fields__)
            ),
            "spacecraft_power_intervals.csv": _csv_bytes(
                result.intervals, tuple(PowerInterval.__dataclass_fields__)
            ),
            "spacecraft_power_events.csv": _csv_bytes(
                result.events, tuple(PowerEvent.__dataclass_fields__)
            ),
        }
        for name, content in files.items():
            (temporary / name).write_bytes(content)
        receipts = [_receipt(temporary / name, temporary) for name in sorted(files)]
        manifest = {
            "schema_version": SPACECRAFT_POWER_MANIFEST_SCHEMA,
            "analysis_id": result.problem.analysis_id,
            "asset_id": result.problem.asset_id,
            "status": result.summary["status"],
            "feasibility": result.summary["feasibility"],
            "problem_semantic_sha256": result.summary["problem_semantic_sha256"],
            "history_semantic_sha256": result.summary["history_semantic_sha256"],
            "result_semantic_sha256": result.summary["result_semantic_sha256"],
            "artifacts": receipts,
            "claim_limits": result.summary["claim_limits"],
        }
        (temporary / "spacecraft_power_manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return SpacecraftPowerArtifacts(
        output_dir=destination,
        manifest_json=destination / "spacecraft_power_manifest.json",
        problem_json=destination / "normalized_problem.json",
        history_json=destination / "normalized_history.json",
        summary_json=destination / "spacecraft_power_summary.json",
        timeseries_csv=destination / "spacecraft_power_timeseries.csv",
        intervals_csv=destination / "spacecraft_power_intervals.csv",
        events_csv=destination / "spacecraft_power_events.csv",
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _read_json(path: Path, field: str) -> dict[str, Any]:
    try:
        content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=MAX_POWER_JSON_BYTES)
        value = json.loads(content.decode("utf-8"), parse_constant=_reject_json_constant)
    except (SafeReadError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SpacecraftPowerError(f"Could not read {field}: {exc}") from exc
    if not isinstance(value, dict):
        raise SpacecraftPowerError(f"{field} must be a JSON object.")
    return value


def verify_spacecraft_power_artifacts(evidence_dir: str | Path) -> dict[str, Any]:
    """Verify receipts and authoritatively recompute one spacecraft-power result."""

    root_input = Path(evidence_dir).expanduser()
    if root_input.is_symlink():
        raise SpacecraftPowerError("Spacecraft-power evidence directory must not be a symbolic link.")
    root = root_input.resolve()
    if not root.is_dir():
        raise SpacecraftPowerError(f"Spacecraft-power evidence directory does not exist: {root}.")
    expected = {*_ARTIFACT_FILES, "spacecraft_power_manifest.json"}
    actual = {item.name for item in root.iterdir()}
    if actual != expected:
        raise SpacecraftPowerError("Spacecraft-power evidence contains an unexpected artifact set.")
    manifest = _read_json(root / "spacecraft_power_manifest.json", "spacecraft-power manifest")
    _exact_fields(
        manifest,
        {
            "schema_version",
            "analysis_id",
            "asset_id",
            "status",
            "feasibility",
            "problem_semantic_sha256",
            "history_semantic_sha256",
            "result_semantic_sha256",
            "artifacts",
            "claim_limits",
        },
        "spacecraft-power manifest",
    )
    if manifest.get("schema_version") != SPACECRAFT_POWER_MANIFEST_SCHEMA:
        raise SpacecraftPowerError("Spacecraft-power manifest has an unsupported schema version.")
    receipts = manifest.get("artifacts")
    if not isinstance(receipts, list) or len(receipts) != len(_ARTIFACT_FILES):
        raise SpacecraftPowerError("Spacecraft-power manifest has an invalid receipt set.")
    received: set[str] = set()
    for receipt_value in receipts:
        receipt = _mapping(receipt_value, "artifact receipt")
        _exact_fields(receipt, {"path", "bytes", "sha256"}, "artifact receipt")
        name = _required_text(receipt["path"], "artifact receipt path")
        if name not in _ARTIFACT_FILES or name in received or Path(name).name != name:
            raise SpacecraftPowerError("Spacecraft-power manifest has an invalid artifact path.")
        received.add(name)
        path = root / name
        try:
            content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=MAX_POWER_JSON_BYTES)
        except SafeReadError as exc:
            raise SpacecraftPowerError(f"Could not safely read retained artifact {name}: {exc}") from exc
        if len(content) != receipt.get("bytes") or hashlib.sha256(content).hexdigest() != receipt.get("sha256"):
            raise SpacecraftPowerError(f"Artifact receipt mismatch for {name}.")
    if received != _ARTIFACT_FILES:
        raise SpacecraftPowerError("Spacecraft-power manifest does not bind the exact artifact set.")
    retained_problem = _read_json(root / "normalized_problem.json", "problem")
    problem = SpacecraftPowerProblem.from_mapping(retained_problem)
    if retained_problem != problem.to_dict():
        raise SpacecraftPowerError("Retained spacecraft-power problem is not canonically normalized.")
    retained_history = _read_json(root / "normalized_history.json", "history")
    history = power_history_from_mapping(retained_history)
    if retained_history != power_history_to_dict(history):
        raise SpacecraftPowerError("Retained spacecraft-power history is not canonically normalized.")
    result = assess_spacecraft_power(problem, history)
    retained_summary = _read_json(root / "spacecraft_power_summary.json", "summary")
    if retained_summary != result.summary:
        raise SpacecraftPowerError("Retained spacecraft-power summary differs from authoritative replay.")
    expected_csv = {
        "spacecraft_power_timeseries.csv": _csv_bytes(
            result.samples, tuple(PowerSample.__dataclass_fields__)
        ),
        "spacecraft_power_intervals.csv": _csv_bytes(
            result.intervals, tuple(PowerInterval.__dataclass_fields__)
        ),
        "spacecraft_power_events.csv": _csv_bytes(
            result.events, tuple(PowerEvent.__dataclass_fields__)
        ),
    }
    for name, expected_content in expected_csv.items():
        if read_regular_file_nofollow(root / name, min_bytes=1, max_bytes=MAX_POWER_JSON_BYTES) != expected_content:
            raise SpacecraftPowerError(f"Retained {name} differs from authoritative replay.")
    for field in (
        "analysis_id",
        "asset_id",
        "status",
        "feasibility",
        "problem_semantic_sha256",
        "history_semantic_sha256",
        "result_semantic_sha256",
        "claim_limits",
    ):
        if manifest.get(field) != result.summary.get(field):
            raise SpacecraftPowerError(f"Spacecraft-power manifest field {field!r} differs from replay.")
    return {
        "schema_version": SPACECRAFT_POWER_EVIDENCE_SCHEMA,
        "status": "verified",
        "analysis_id": problem.analysis_id,
        "asset_id": problem.asset_id,
        "feasibility": result.summary["feasibility"],
        "problem_semantic_sha256": result.summary["problem_semantic_sha256"],
        "history_semantic_sha256": result.summary["history_semantic_sha256"],
        "result_semantic_sha256": result.summary["result_semantic_sha256"],
    }


__all__ = [
    "MAX_INTEGRATION_STEP_S",
    "MAX_POWER_ACTIVITIES",
    "MAX_POWER_DURATION_S",
    "MAX_POWER_JSON_BYTES",
    "MAX_POWER_SAMPLES",
    "SPACECRAFT_POWER_EVIDENCE_SCHEMA",
    "SPACECRAFT_POWER_HISTORY_SCHEMA",
    "SPACECRAFT_POWER_MANIFEST_SCHEMA",
    "SPACECRAFT_POWER_PROBLEM_SCHEMA",
    "BatteryConfig",
    "PowerActivity",
    "PowerEvent",
    "PowerInterval",
    "PowerSample",
    "SolarArrayConfig",
    "SpacecraftPowerArtifacts",
    "SpacecraftPowerError",
    "SpacecraftPowerProblem",
    "SpacecraftPowerResult",
    "assess_spacecraft_power",
    "power_history_from_mapping",
    "power_history_to_dict",
    "problem_with_mission_schedule",
    "validate_spacecraft_power_inputs",
    "verify_spacecraft_power_artifacts",
    "write_spacecraft_power_artifacts",
]
