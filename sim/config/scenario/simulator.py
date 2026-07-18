from __future__ import annotations

from typing import Any

from sim.config.scenario.models import (
    SimulatorSection,
)
from sim.config.scenario.primitives import (
    _REENTRY_TERMINATION_LIMIT_FIELDS,
    _SIMULATOR_UNSUPPORTED_ALIASES,
    _as_dict,
    _parse_bool,
    _parse_float,
    _parse_optional_float,
    _reject_unknown_fields,
    _reject_unsupported_aliases,
    _validate_sim_timing,
)

__all__ = [
    '_normalize_reentry_termination_block',
    '_normalize_reentry_section',
    '_normalize_simulator_termination_block',
    '_parse_simulator_section',
    '_normalize_dynamics_section',
    '_parse_simulator_frames_section',
    '_parse_resource_profile',
    '_parse_acceleration_section',
    '_parse_simulator_execution_section',
]

def _normalize_reentry_termination_block(
    value: Any,
    path: str,
    *,
    fill_bool_defaults: bool,
) -> dict[str, Any]:
    termination = _as_dict(value, path)
    if fill_bool_defaults or "enabled" in termination:
        termination["enabled"] = _parse_bool(
            termination.get("enabled", False),
            f"{path}.enabled",
        )
    if fill_bool_defaults or "terminate_on_entry" in termination:
        termination["terminate_on_entry"] = _parse_bool(
            termination.get("terminate_on_entry", False),
            f"{path}.terminate_on_entry",
        )
    for key in _REENTRY_TERMINATION_LIMIT_FIELDS:
        if termination.get(key) is None:
            continue
        termination[key] = _parse_float(termination.get(key), f"{path}.{key}")
        if float(termination[key]) < 0.0:
            raise ValueError(f"{path}.{key} must be >= 0.")
    return termination


def _normalize_reentry_section(dynamics: dict[str, Any]) -> dict[str, Any]:
    out = dict(dynamics or {})
    if "reentry" not in out:
        return out
    raw = _as_dict(out.get("reentry"), "simulator.dynamics.reentry")
    normalized = dict(raw)
    normalized["enabled"] = _parse_bool(
        normalized.get("enabled", False),
        "simulator.dynamics.reentry.enabled",
    )
    normalized["begin_altitude_km"] = _parse_float(
        normalized.get("begin_altitude_km", 300.0),
        "simulator.dynamics.reentry.begin_altitude_km",
    )
    if normalized["begin_altitude_km"] < 0.0:
        raise ValueError("simulator.dynamics.reentry.begin_altitude_km must be >= 0.")
    if normalized.get("nose_radius_m") is not None:
        normalized["nose_radius_m"] = _parse_float(
            normalized.get("nose_radius_m"),
            "simulator.dynamics.reentry.nose_radius_m",
        )
        if normalized["nose_radius_m"] <= 0.0:
            raise ValueError("simulator.dynamics.reentry.nose_radius_m must be positive.")
    normalized["heat_rate_coefficient"] = _parse_float(
        normalized.get("heat_rate_coefficient", 1.83e-4),
        "simulator.dynamics.reentry.heat_rate_coefficient",
    )
    if normalized["heat_rate_coefficient"] < 0.0:
        raise ValueError("simulator.dynamics.reentry.heat_rate_coefficient must be >= 0.")
    object_ids = normalized.get("object_ids", [])
    if isinstance(object_ids, str):
        normalized["object_ids"] = [object_ids]
    elif isinstance(object_ids, list):
        normalized["object_ids"] = [str(item) for item in object_ids]
    else:
        raise ValueError("simulator.dynamics.reentry.object_ids must be a string or list of strings.")
    atmosphere_model = normalized.get("atmosphere_model")
    if atmosphere_model is not None:
        atmosphere_model = str(atmosphere_model).strip().lower()
        if atmosphere_model not in {
            "",
            "exponential",
            "ussa1976",
            "msis86",
            "msis-86",
            "hpop_msis86",
            "nrlmsise00",
            "jacchia70",
            "jacchia-70",
            "hpop_jacchia70",
            "jb2006",
            "jb2008",
            "harris_priester",
            "harris-priester",
            "hp",
            "hpop_harris_priester",
        }:
            raise ValueError(
                "simulator.dynamics.reentry.atmosphere_model must be one of: "
                "exponential, ussa1976, msis86, nrlmsise00, jacchia70, jb2006, jb2008, harris_priester."
            )
        normalized["atmosphere_model"] = atmosphere_model
    termination = dict(normalized.get("termination", {}) or {})
    if termination:
        by_object_raw = termination.pop("by_object", {})
        termination = _normalize_reentry_termination_block(
            termination,
            "simulator.dynamics.reentry.termination",
            fill_bool_defaults=True,
        )
        if by_object_raw is not None:
            if not isinstance(by_object_raw, dict):
                raise ValueError("simulator.dynamics.reentry.termination.by_object must be a mapping.")
            by_object: dict[str, Any] = {}
            for object_id, object_termination in by_object_raw.items():
                object_path = f"simulator.dynamics.reentry.termination.by_object.{object_id}"
                by_object[str(object_id)] = _normalize_reentry_termination_block(
                    object_termination,
                    object_path,
                    fill_bool_defaults=False,
                )
            termination["by_object"] = by_object
    normalized["termination"] = termination
    out["reentry"] = normalized
    return out
def _normalize_simulator_termination_block(
    value: Any,
    path: str,
    *,
    fill_defaults: bool,
) -> dict[str, Any]:
    termination = _as_dict(value, path)
    _reject_unknown_fields(termination, path, {"enabled", "earth_impact_enabled", "earth_radius_km"})
    if "enabled" in termination and "earth_impact_enabled" not in termination:
        termination["earth_impact_enabled"] = termination.pop("enabled")
    elif "enabled" in termination:
        termination.pop("enabled")
    if fill_defaults or "earth_impact_enabled" in termination:
        termination["earth_impact_enabled"] = _parse_bool(
            termination.get("earth_impact_enabled", True),
            f"{path}.earth_impact_enabled",
        )
    if fill_defaults or "earth_radius_km" in termination:
        termination["earth_radius_km"] = _parse_float(
            termination.get("earth_radius_km", 6378.137),
            f"{path}.earth_radius_km",
        )
        if float(termination["earth_radius_km"]) <= 0.0:
            raise ValueError(f"{path}.earth_radius_km must be positive.")
    return termination


def _parse_simulator_section(value: Any) -> SimulatorSection:
    d = _as_dict(value, "simulator")
    _reject_unsupported_aliases(d, "simulator", _SIMULATOR_UNSUPPORTED_ALIASES)
    _reject_unknown_fields(
        d,
        "simulator",
        {
            "duration_s",
            "dt_s",
            "initial_jd_utc",
            "resource_profile",
            "acceleration",
            "execution",
            "frames",
            "dynamics",
            "environment",
            "plugin_validation",
            "termination",
        },
    )
    plugin_validation = {"strict": True}
    plugin_validation.update(dict(d.get("plugin_validation", {}) or {}))
    _reject_unknown_fields(plugin_validation, "simulator.plugin_validation", {"strict", "strict_runtime"})
    termination_raw = dict(d.get("termination", {}) or {})
    termination_by_object_raw = termination_raw.pop("by_object", {})
    termination = {"earth_impact_enabled": True, "earth_radius_km": 6378.137}
    termination.update(termination_raw)
    plugin_validation["strict"] = _parse_bool(
        plugin_validation.get("strict", True), "simulator.plugin_validation.strict"
    )
    plugin_validation["strict_runtime"] = _parse_bool(
        plugin_validation.get("strict_runtime", False), "simulator.plugin_validation.strict_runtime"
    )
    termination = _normalize_simulator_termination_block(
        termination,
        "simulator.termination",
        fill_defaults=True,
    )
    if termination_by_object_raw is not None:
        if not isinstance(termination_by_object_raw, dict):
            raise ValueError("simulator.termination.by_object must be a mapping.")
        by_object: dict[str, Any] = {}
        for object_id, object_termination in termination_by_object_raw.items():
            object_path = f"simulator.termination.by_object.{object_id}"
            by_object[str(object_id)] = _normalize_simulator_termination_block(
                object_termination,
                object_path,
                fill_defaults=False,
            )
        termination["by_object"] = by_object
    dynamics = _normalize_dynamics_section(dict(d.get("dynamics", {}) or {}))
    out = SimulatorSection(
        duration_s=_parse_float(d.get("duration_s", 3600.0), "simulator.duration_s"),
        dt_s=_parse_float(d.get("dt_s", 1.0), "simulator.dt_s"),
        initial_jd_utc=_parse_optional_float(d.get("initial_jd_utc"), "simulator.initial_jd_utc"),
        resource_profile=_parse_resource_profile(d.get("resource_profile"), "simulator.resource_profile"),
        acceleration=_parse_acceleration_section(d.get("acceleration")),
        execution=_parse_simulator_execution_section(d.get("execution")),
        frames=_parse_simulator_frames_section(d.get("frames")),
        dynamics=dynamics,
        environment=dict(d.get("environment", {}) or {}),
        plugin_validation=plugin_validation,
        termination=termination,
    )
    if str(out.frames.model).strip().lower() in {"iau76_80_eop", "iau76_fk5_iau80_eop", "hpop_like", "hpop"}:
        has_manual_eop = any(out.frames.get(key) is not None for key in ("dut1_s", "xp_arcsec", "yp_arcsec", "dat_s"))
        has_manual_eop = has_manual_eop or any(
            float(out.frames.get(key, 0.0) or 0.0) != 0.0 for key in ("ddpsi_rad", "ddeps_rad")
        )
        if (out.frames.eop_path is not None or has_manual_eop) and out.initial_jd_utc is None:
            raise ValueError("simulator.frames EOP settings require simulator.initial_jd_utc for frame rotation.")
    _validate_sim_timing(out)
    return out


def _normalize_dynamics_section(value: dict[str, Any]) -> dict[str, Any]:
    dynamics = dict(value or {})
    _reject_unknown_fields(dynamics, "simulator.dynamics", {"orbit", "attitude", "rocket", "reentry"})
    orbit = _as_dict(dynamics.get("orbit"), "simulator.dynamics.orbit")
    _reject_unknown_fields(
        orbit,
        "simulator.dynamics.orbit",
        {
            "model",
            "cr3bp_system",
            "propagation_method",
            "integrator",
            "adaptive_atol",
            "adaptive_rtol",
            "orbit_substep_s",
            "j2",
            "j3",
            "j4",
            "drag",
            "lift",
            "srp",
            "third_body_sun",
            "third_body_moon",
            "atmosphere_model",
            "drag_frame_model",
            "drag_eop_path",
            "drag_earth_rotation_rad_s",
            "de440_coeff_path",
            "de440_eop_path",
            "spherical_harmonics",
        },
    )
    attitude = _as_dict(dynamics.get("attitude"), "simulator.dynamics.attitude")
    _reject_unknown_fields(
        attitude,
        "simulator.dynamics.attitude",
        {"enabled", "attitude_substep_s", "disturbance_torques", "guardrail_policy"},
    )
    dynamics["orbit"] = orbit
    dynamics["attitude"] = attitude
    return _normalize_reentry_section(dynamics)


def _parse_simulator_frames_section(value: Any) -> dict[str, Any]:
    raw = _as_dict(value, "simulator.frames")
    _reject_unknown_fields(
        raw,
        "simulator.frames",
        {
            "model",
            "frame_model",
            "eop_path",
            "eop_extrapolation",
            "time_scale_model",
            "tt_minus_utc_s",
            "dut1_s",
            "xp_arcsec",
            "yp_arcsec",
            "dat_s",
            "ddpsi_rad",
            "ddeps_rad",
        },
    )
    out = dict(raw)
    model = str(out.get("model", out.get("frame_model", "simple_gmst")) or "simple_gmst").strip().lower()
    allowed = {
        "simple",
        "simple_gmst",
        "simple_earth_rotation",
        "gmst",
        "hpop_like",
        "hpop",
        "iau76_80_eop",
        "iau76_fk5_iau80_eop",
    }
    if model not in allowed:
        raise ValueError(
            "simulator.frames.model must be one of: simple_gmst, simple_earth_rotation, "
            "hpop_like, iau76_80_eop."
        )
    out["model"] = model
    for key in ("tt_minus_utc_s", "dut1_s", "xp_arcsec", "yp_arcsec", "dat_s", "ddpsi_rad", "ddeps_rad"):
        if out.get(key) is not None:
            out[key] = _parse_float(out.get(key), f"simulator.frames.{key}")
    if out.get("eop_path") in ("",):
        out["eop_path"] = None
    eop_extrapolation = str(out.get("eop_extrapolation", "error") or "error").strip().lower()
    if eop_extrapolation not in {"error", "hold"}:
        raise ValueError("simulator.frames.eop_extrapolation must be 'error' or 'hold'.")
    out["eop_extrapolation"] = eop_extrapolation
    if out.get("time_scale_model") is not None:
        out["time_scale_model"] = str(out.get("time_scale_model"))
    return out


def _parse_resource_profile(value: Any, field_name: str) -> str | None:
    if value in (None, ""):
        return None
    profile = str(value).strip().lower()
    if profile not in {"config", "laptop-safe", "standard", "aggressive", "off"}:
        raise ValueError(f"{field_name} must be one of: config, laptop-safe, standard, aggressive, off.")
    return profile


def _parse_acceleration_section(value: Any) -> dict[str, Any]:
    d = _as_dict(value, "simulator.acceleration")
    _reject_unknown_fields(d, "simulator.acceleration", {"mode", "warmup", "env_override"})
    mode = str(d.get("mode", "off") or "off").strip().lower()
    if mode not in {"off", "auto", "numba"}:
        raise ValueError("simulator.acceleration.mode must be one of: off, auto, numba.")
    return {
        **d,
        "mode": mode,
        "warmup": _parse_bool(d.get("warmup", False), "simulator.acceleration.warmup"),
        "env_override": _parse_bool(d.get("env_override", True), "simulator.acceleration.env_override"),
    }


def _parse_simulator_execution_section(value: Any) -> dict[str, Any]:
    d = _as_dict(value, "simulator.execution")
    _reject_unknown_fields(
        d,
        "simulator.execution",
        {"policy", "object_parallelism", "runtime_profiler", "controller"},
    )
    object_parallelism = _as_dict(d.get("object_parallelism"), "simulator.execution.object_parallelism")
    runtime_profiler = _as_dict(d.get("runtime_profiler"), "simulator.execution.runtime_profiler")
    controller = _as_dict(d.get("controller"), "simulator.execution.controller")
    backend = str(object_parallelism.get("backend", "serial") or "serial").strip().lower()
    if backend not in {"serial", "process_pool"}:
        raise ValueError("simulator.execution.object_parallelism.backend must be one of: serial, process_pool.")
    workers = int(object_parallelism.get("workers", 0) or 0)
    max_workers = int(object_parallelism.get("max_workers", 0) or 0)
    reserve_workers = int(object_parallelism.get("reserve_workers", 1) or 0)
    min_objects = int(object_parallelism.get("min_objects", 3) or 0)
    if workers < 0:
        raise ValueError("simulator.execution.object_parallelism.workers must be >= 0.")
    if max_workers < 0:
        raise ValueError("simulator.execution.object_parallelism.max_workers must be >= 0.")
    if reserve_workers < 0:
        raise ValueError("simulator.execution.object_parallelism.reserve_workers must be >= 0.")
    if min_objects < 1:
        raise ValueError("simulator.execution.object_parallelism.min_objects must be >= 1.")
    policy = str(d.get("policy", "configured") or "configured").strip().lower()
    raw_enabled = object_parallelism.get("enabled", False)
    if isinstance(raw_enabled, str) and raw_enabled.strip().lower() == "auto":
        if policy not in {"configured", "auto"}:
            raise ValueError(
                "simulator.execution.object_parallelism.enabled=auto conflicts with "
                f"simulator.execution.policy={policy!r}."
            )
        policy = "auto"
        enabled = True
    else:
        enabled = _parse_bool(
            raw_enabled,
            "simulator.execution.object_parallelism.enabled",
        )
    if policy not in {"configured", "serial", "parallel", "auto"}:
        raise ValueError(
            "simulator.execution.policy must be one of: auto, configured, parallel, serial."
        )
    if policy == "serial":
        enabled = False
        backend = "serial"
    elif policy in {"parallel", "auto"}:
        enabled = True
        backend = "process_pool"
    orbit_budget_ms = _parse_float(
        controller.get("orbit_budget_ms", 2.0),
        "simulator.execution.controller.orbit_budget_ms",
    )
    attitude_budget_ms = _parse_float(
        controller.get("attitude_budget_ms", 2.0),
        "simulator.execution.controller.attitude_budget_ms",
    )
    if orbit_budget_ms <= 0.0 or attitude_budget_ms <= 0.0:
        raise ValueError("simulator.execution.controller budgets must be positive.")
    deadline_policy = str(controller.get("deadline_policy", "record") or "record").strip().lower()
    if deadline_policy not in {"record", "zero_command", "error"}:
        raise ValueError(
            "simulator.execution.controller.deadline_policy must be one of: error, record, zero_command."
        )
    return {
        **d,
        "policy": policy,
        "object_parallelism": {
            **object_parallelism,
            "enabled": enabled,
            "backend": backend,
            "workers": workers,
            "max_workers": max_workers,
            "reserve_workers": reserve_workers,
            "min_objects": min_objects,
        },
        "runtime_profiler": {
            **runtime_profiler,
            "enabled": _parse_bool(
                runtime_profiler.get("enabled", True),
                "simulator.execution.runtime_profiler.enabled",
            ),
        },
        "controller": {
            **controller,
            "orbit_budget_ms": orbit_budget_ms,
            "attitude_budget_ms": attitude_budget_ms,
            "deadline_policy": deadline_policy,
        },
    }
