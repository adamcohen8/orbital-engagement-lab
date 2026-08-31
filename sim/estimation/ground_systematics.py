from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sim.estimation.parameters import EstimatedParameter, ParameterSet

SYSTEMATIC_PARAMETER_DEFAULTS: dict[str, dict[str, Any]] = {
    "range_bias_km": {
        "unit": "km",
        "scale": 0.01,
        "lower": -10.0,
        "upper": 10.0,
        "prior_sigma": 0.1,
        "components": ("range_km",),
    },
    "range_rate_bias_km_s": {
        "unit": "km/s",
        "scale": 1.0e-5,
        "lower": -0.1,
        "upper": 0.1,
        "prior_sigma": 1.0e-4,
        "components": ("range_rate_km_s",),
    },
    "azimuth_bias_deg": {
        "unit": "deg",
        "scale": 0.01,
        "lower": -5.0,
        "upper": 5.0,
        "prior_sigma": 0.1,
        "components": ("azimuth_deg",),
    },
    "elevation_bias_deg": {
        "unit": "deg",
        "scale": 0.01,
        "lower": -5.0,
        "upper": 5.0,
        "prior_sigma": 0.1,
        "components": ("elevation_deg",),
    },
    "clock_offset_s": {
        "unit": "s",
        "scale": 0.01,
        "lower": -2.0,
        "upper": 2.0,
        "prior_sigma": 0.1,
        "components": (),
    },
    "clock_drift_s_per_s": {
        "unit": "s/s",
        "scale": 1.0e-6,
        "lower": -1.0e-3,
        "upper": 1.0e-3,
        "prior_sigma": 1.0e-5,
        "components": (),
    },
}


def normalize_ground_systematics(
    value: Mapping[str, Any] | None,
    *,
    station_ids: Sequence[str],
) -> dict[str, Any]:
    raw = dict(value or {})
    schema_version = int(raw.get("schema_version", 1) or 1)
    if schema_version != 1:
        raise ValueError("ground systematic_error_model schema_version must be 1.")
    known_stations = sorted({str(station_id) for station_id in station_ids if str(station_id)})

    estimate = [str(item) for item in list(raw.get("estimate", []) or [])]
    parameters_raw = dict(raw.get("parameters", {}) or {})
    for name in estimate:
        parameters_raw.setdefault(name, {})
    unknown_parameters = sorted(set(parameters_raw) - set(SYSTEMATIC_PARAMETER_DEFAULTS))
    if unknown_parameters:
        raise ValueError(f"Unsupported ground systematic parameters: {unknown_parameters}.")
    parameters: dict[str, dict[str, Any]] = {}
    for name in sorted(parameters_raw):
        supplied = dict(parameters_raw[name] or {})
        defaults = dict(SYSTEMATIC_PARAMETER_DEFAULTS[name])
        scope = str(supplied.get("scope", "per_station") or "per_station").strip().lower()
        if scope not in {"shared", "per_station"}:
            raise ValueError(f"systematic parameter {name!r} scope must be 'shared' or 'per_station'.")
        stations = [str(item) for item in list(supplied.get("stations", known_stations) or [])]
        unknown_stations = sorted(set(stations) - set(known_stations))
        if unknown_stations:
            raise ValueError(f"systematic parameter {name!r} references unknown stations: {unknown_stations}.")
        initial = _finite_float(supplied.get("initial", supplied.get("prior_mean", 0.0)), f"{name}.initial")
        prior_mean = _finite_float(supplied.get("prior_mean", initial), f"{name}.prior_mean")
        prior_sigma = _positive_float(
            supplied.get("prior_sigma", defaults["prior_sigma"]),
            f"{name}.prior_sigma",
        )
        scale = _positive_float(supplied.get("scale", defaults["scale"]), f"{name}.scale")
        lower = _finite_float(supplied.get("lower", defaults["lower"]), f"{name}.lower")
        upper = _finite_float(supplied.get("upper", defaults["upper"]), f"{name}.upper")
        if lower >= upper or not lower <= initial <= upper:
            raise ValueError(f"systematic parameter {name!r} requires lower < initial < upper.")
        parameters[name] = {
            "name": name,
            "scope": scope,
            "stations": stations,
            "initial": initial,
            "prior_mean": prior_mean,
            "prior_sigma": prior_sigma,
            "scale": scale,
            "lower": lower,
            "upper": upper,
            "unit": str(defaults["unit"]),
            "components": list(defaults["components"]),
        }

    fixed_shared = _normalize_fixed_systematics(raw.get("fixed_shared", {}), field_name="fixed_shared")
    fixed_by_station: dict[str, dict[str, float]] = {}
    for station_id, station_values in dict(raw.get("fixed_by_station", {}) or {}).items():
        station_key = str(station_id)
        if station_key not in known_stations:
            raise ValueError(f"fixed_by_station references unknown station {station_key!r}.")
        fixed_by_station[station_key] = _normalize_fixed_systematics(
            station_values,
            field_name=f"fixed_by_station.{station_key}",
        )

    elevation_raw = dict(raw.get("elevation_weighting", {}) or {})
    elevation_enabled = bool(elevation_raw.get("enabled", False))
    elevation_model = str(elevation_raw.get("model", "sine") or "sine").strip().lower()
    if elevation_model != "sine":
        raise ValueError("elevation_weighting.model must be 'sine'.")
    min_sine = _positive_float(elevation_raw.get("minimum_sine", 0.15), "elevation_weighting.minimum_sine")
    if min_sine > 1.0:
        raise ValueError("elevation_weighting.minimum_sine must be <= 1.")
    exponent = _positive_float(elevation_raw.get("exponent", 1.0), "elevation_weighting.exponent")
    weight_components = [
        str(item)
        for item in list(
            elevation_raw.get(
                "components",
                ("azimuth_deg", "elevation_deg", "range_km", "range_rate_km_s"),
            )
            or []
        )
    ]

    refraction_raw = dict(raw.get("refraction", {}) or {})
    refraction_enabled = bool(refraction_raw.get("enabled", False))
    refraction_model = str(refraction_raw.get("model", "bennett_1982") or "bennett_1982").strip().lower()
    if refraction_model != "bennett_1982":
        raise ValueError("refraction.model must be 'bennett_1982'.")
    pressure_hpa = _positive_float(refraction_raw.get("pressure_hpa", 1010.0), "refraction.pressure_hpa")
    temperature_c = _finite_float(refraction_raw.get("temperature_c", 10.0), "refraction.temperature_c")
    minimum_elevation_deg = _finite_float(
        refraction_raw.get("minimum_elevation_deg", -0.5),
        "refraction.minimum_elevation_deg",
    )

    light_time_raw = dict(raw.get("light_time", {}) or {})
    if bool(light_time_raw.get("enabled", False)):
        raise ValueError(
            "light_time.enabled is not supported by the Phase 3 ground model; enable it only after a named "
            "one-way/two-way observable contract is implemented."
        )

    clock_raw = dict(raw.get("clock_linearization", {}) or {})
    clock_fd_step_s = _positive_float(clock_raw.get("finite_difference_step_s", 0.25), "clock finite difference")
    clock_reference_time_s = _finite_float(clock_raw.get("reference_time_s", 0.0), "clock reference time")
    return {
        "schema_version": 1,
        "parameters": parameters,
        "fixed_shared": fixed_shared,
        "fixed_by_station": fixed_by_station,
        "elevation_weighting": {
            "enabled": elevation_enabled,
            "model": elevation_model,
            "minimum_sine": min_sine,
            "exponent": exponent,
            "components": weight_components,
        },
        "refraction": {
            "enabled": refraction_enabled,
            "model": refraction_model,
            "pressure_hpa": pressure_hpa,
            "temperature_c": temperature_c,
            "minimum_elevation_deg": minimum_elevation_deg,
            "observable": "apparent_elevation_deg",
        },
        "clock_linearization": {
            "enabled": any(name.startswith("clock_") for name in parameters),
            "model": "engine_derived_first_order_measurement_time_shift",
            "finite_difference_step_s": clock_fd_step_s,
            "reference_time_s": clock_reference_time_s,
        },
        "light_time": {
            "enabled": False,
            "status": "not_implemented_no_named_one_way_or_two_way_observable_contract",
        },
    }


def extend_parameter_set_for_ground_systematics(
    base: ParameterSet,
    *,
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> tuple[ParameterSet, list[dict[str, Any]], list[str], np.ndarray, np.ndarray]:
    records: list[dict[str, Any]] = []
    new_parameters: list[EstimatedParameter] = []
    row_components_by_station: dict[str, set[str]] = {}
    for row in rows:
        row_components_by_station.setdefault(str(row.get("station_id", "")), set()).update(
            str(component) for component in list(row.get("components", []) or [])
        )
    for systematic_name, raw_spec in dict(model.get("parameters", {}) or {}).items():
        spec = dict(raw_spec)
        scope = str(spec["scope"])
        targets: list[str | None] = [None] if scope == "shared" else list(spec["stations"])
        required_components = set(str(item) for item in list(spec.get("components", []) or []))
        for station_id in targets:
            if station_id is not None and required_components:
                if not (required_components & row_components_by_station.get(str(station_id), set())):
                    continue
            parameter_name = _parameter_name(systematic_name, scope=scope, station_id=station_id)
            new_parameters.append(
                EstimatedParameter(
                    parameter_name,
                    float(spec["initial"]),
                    scale=float(spec["scale"]),
                    lower=float(spec["lower"]),
                    upper=float(spec["upper"]),
                    unit=str(spec["unit"]),
                    description=f"{scope} ground-measurement {systematic_name}",
                )
            )
            records.append(
                {
                    "parameter": parameter_name,
                    "systematic": systematic_name,
                    "scope": scope,
                    "station_id": station_id,
                    "prior_mean": float(spec["prior_mean"]),
                    "prior_sigma": float(spec["prior_sigma"]),
                    "unit": str(spec["unit"]),
                }
            )
    combined = ParameterSet([*base.parameters, *new_parameters])
    prior_names = [str(record["parameter"]) for record in records]
    prior_mean = np.array([float(record["prior_mean"]) for record in records], dtype=float)
    prior_covariance = np.diag([float(record["prior_sigma"]) ** 2 for record in records])
    return combined, records, prior_names, prior_mean, prior_covariance


def systematic_prediction(
    geometric: Mapping[str, float],
    *,
    row: Mapping[str, Any],
    parameter_values: Mapping[str, float],
    model: Mapping[str, Any],
    time_derivative: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    station_id = str(row.get("station_id", ""))
    time_s = float(row.get("time_s", 0.0))
    output = {str(key): float(value) for key, value in geometric.items()}
    clock_offset = _systematic_value(
        "clock_offset_s",
        station_id=station_id,
        parameter_values=parameter_values,
        model=model,
    )
    clock_drift = _systematic_value(
        "clock_drift_s_per_s",
        station_id=station_id,
        parameter_values=parameter_values,
        model=model,
    )
    clock_reference = float(dict(model.get("clock_linearization", {}) or {}).get("reference_time_s", 0.0))
    clock_shift = clock_offset + clock_drift * (time_s - clock_reference)
    derivative = dict(time_derivative or {})
    if abs(clock_shift) > 0.0 and not derivative:
        raise ValueError("clock parameters require engine-derived measurement time derivatives.")
    for component in output:
        output[component] += float(derivative.get(component, 0.0)) * clock_shift

    bias_map = {
        "range_km": "range_bias_km",
        "range_rate_km_s": "range_rate_bias_km_s",
        "azimuth_deg": "azimuth_bias_deg",
        "elevation_deg": "elevation_bias_deg",
    }
    applied_biases: dict[str, float] = {}
    for component, systematic_name in bias_map.items():
        bias = _systematic_value(
            systematic_name,
            station_id=station_id,
            parameter_values=parameter_values,
            model=model,
        )
        output[component] += bias
        applied_biases[systematic_name] = bias

    refraction = dict(model.get("refraction", {}) or {})
    refraction_correction_deg = 0.0
    if bool(refraction.get("enabled", False)):
        refraction_correction_deg = bennett_refraction_correction_deg(
            output["elevation_deg"],
            pressure_hpa=float(refraction["pressure_hpa"]),
            temperature_c=float(refraction["temperature_c"]),
            minimum_elevation_deg=float(refraction["minimum_elevation_deg"]),
        )
        output["elevation_deg"] += refraction_correction_deg
    output["azimuth_deg"] = output["azimuth_deg"] % 360.0
    return output, {
        "station_id": station_id,
        "clock_shift_s": clock_shift,
        "clock_model": dict(model.get("clock_linearization", {}) or {}),
        "biases": applied_biases,
        "refraction_model": refraction,
        "refraction_correction_deg": refraction_correction_deg,
        "light_time": dict(model.get("light_time", {}) or {}),
    }


def elevation_weighted_covariance(
    covariance: np.ndarray,
    *,
    row: Mapping[str, Any],
    model: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    weighting = dict(model.get("elevation_weighting", {}) or {})
    components = [str(item) for item in list(row.get("components", []) or [])]
    if not bool(weighting.get("enabled", False)):
        return np.asarray(covariance, dtype=float), {"enabled": False, "sigma_scale": 1.0, "source": "disabled"}
    observed_values = dict(row.get("values", {}) or {})
    if "elevation_deg" not in observed_values:
        return np.asarray(covariance, dtype=float), {
            "enabled": True,
            "sigma_scale": 1.0,
            "source": "not_applied_no_observed_elevation",
        }
    elevation_deg = float(observed_values["elevation_deg"])
    sine = abs(float(np.sin(np.deg2rad(elevation_deg))))
    minimum_sine = float(weighting.get("minimum_sine", 0.15))
    exponent = float(weighting.get("exponent", 1.0))
    sigma_scale = 1.0 / max(sine, minimum_sine) ** exponent
    selected = set(str(item) for item in list(weighting.get("components", []) or []))
    scales = np.array([sigma_scale if component in selected else 1.0 for component in components], dtype=float)
    return np.asarray(covariance, dtype=float) * np.outer(scales, scales), {
        "enabled": True,
        "model": "sine",
        "observed_elevation_deg": elevation_deg,
        "sigma_scale": sigma_scale,
        "minimum_sine": minimum_sine,
        "exponent": exponent,
        "components": sorted(selected),
        "source": "observed_elevation",
    }


def bennett_refraction_correction_deg(
    elevation_deg: float,
    *,
    pressure_hpa: float,
    temperature_c: float,
    minimum_elevation_deg: float = -0.5,
) -> float:
    elevation = float(elevation_deg)
    if elevation < float(minimum_elevation_deg) or elevation >= 90.0:
        return 0.0
    denominator_angle_deg = elevation + 10.3 / (elevation + 5.11)
    tangent = float(np.tan(np.deg2rad(denominator_angle_deg)))
    if abs(tangent) <= 1.0e-12:
        return 0.0
    arcminutes = 1.02 / tangent
    arcminutes *= float(pressure_hpa) / 1010.0
    arcminutes *= 283.0 / (273.0 + float(temperature_c))
    return float(arcminutes / 60.0)


def _systematic_value(
    systematic_name: str,
    *,
    station_id: str,
    parameter_values: Mapping[str, float],
    model: Mapping[str, Any],
) -> float:
    value = float(dict(model.get("fixed_shared", {}) or {}).get(systematic_name, 0.0) or 0.0)
    value += float(
        dict(dict(model.get("fixed_by_station", {}) or {}).get(station_id, {}) or {}).get(systematic_name, 0.0) or 0.0
    )
    parameters = dict(model.get("parameters", {}) or {})
    if systematic_name not in parameters:
        return value
    spec = dict(parameters[systematic_name])
    if spec.get("scope") == "shared":
        value += float(parameter_values.get(_parameter_name(systematic_name, scope="shared", station_id=None), 0.0))
    elif station_id in list(spec.get("stations", []) or []):
        value += float(
            parameter_values.get(
                _parameter_name(systematic_name, scope="per_station", station_id=station_id),
                0.0,
            )
        )
    return value


def _parameter_name(systematic_name: str, *, scope: str, station_id: str | None) -> str:
    if scope == "shared":
        return f"systematic__{systematic_name}__shared"
    station_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(station_id or "station"))
    return f"systematic__{systematic_name}__{station_token}"


def _normalize_fixed_systematics(value: Any, *, field_name: str) -> dict[str, float]:
    raw = dict(value or {})
    unknown = sorted(set(raw) - set(SYSTEMATIC_PARAMETER_DEFAULTS))
    if unknown:
        raise ValueError(f"{field_name} contains unsupported systematic fields: {unknown}.")
    return {str(name): _finite_float(item, f"{field_name}.{name}") for name, item in raw.items()}


def _finite_float(value: Any, field_name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return parsed
