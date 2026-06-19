from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.aero import (
    atmosphere_relative_velocity_eci_km_s,
    compute_aero_load_scalars,
    dynamic_pressure_pa,
    sutton_graves_heat_rate_w_m2,
)
from sim.dynamics.orbit.atmosphere import atmosphere_state_from_model
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM, EARTH_ROT_RATE_RAD_S

REENTRY_METRIC_KEYS = (
    "active",
    "altitude_km",
    "density_kg_m3",
    "relative_speed_m_s",
    "dynamic_pressure_pa",
    "drag_decel_m_s2",
    "lift_accel_m_s2",
    "lift_to_drag",
    "g_load",
    "heat_rate_w_m2",
    "heat_load_j_m2",
)

G0_M_S2 = 9.80665
SUTTON_GRAVES_COEFFICIENT_SI = 1.83e-4


@dataclass(frozen=True)
class ReentryTerminationConfig:
    enabled: bool = False
    terminate_on_entry: bool = False
    min_altitude_km: float | None = None
    max_dynamic_pressure_pa: float | None = None
    max_drag_decel_m_s2: float | None = None
    max_g_load: float | None = None
    max_heat_rate_w_m2: float | None = None
    max_heat_load_j_m2: float | None = None


@dataclass(frozen=True)
class ReentryConfig:
    enabled: bool = False
    begin_altitude_km: float = 300.0
    object_ids: tuple[str, ...] = ()
    atmosphere_model: str | None = None
    default_nose_radius_m: float = 0.5
    heat_rate_coefficient: float = SUTTON_GRAVES_COEFFICIENT_SI
    termination: ReentryTerminationConfig = field(default_factory=ReentryTerminationConfig)
    termination_by_object: dict[str, ReentryTerminationConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class ReentryObjectProperties:
    mass_kg: float
    drag_area_m2: float
    cd: float
    nose_radius_m: float
    lift_area_m2: float | None = None
    cl: float = 0.0


def reentry_config_from_dynamics(dynamics: dict[str, Any]) -> ReentryConfig:
    raw = dict(dict(dynamics or {}).get("reentry", {}) or {})
    termination_raw = dict(raw.get("termination", {}) or {})
    termination = _reentry_termination_from_raw(termination_raw)
    termination_by_object_raw = termination_raw.get("by_object", {})
    termination_by_object: dict[str, ReentryTerminationConfig] = {}
    if isinstance(termination_by_object_raw, dict):
        termination_by_object = {
            str(object_id): _reentry_termination_from_raw(raw_override, base=termination)
            for object_id, raw_override in termination_by_object_raw.items()
            if isinstance(raw_override, dict)
        }
    object_ids_raw = raw.get("object_ids", ())
    if isinstance(object_ids_raw, str):
        object_ids = (object_ids_raw,)
    else:
        object_ids = tuple(str(item) for item in list(object_ids_raw or ()))
    atmosphere_model_raw = raw.get("atmosphere_model")
    return ReentryConfig(
        enabled=bool(raw.get("enabled", False)),
        begin_altitude_km=float(raw.get("begin_altitude_km", 300.0)),
        object_ids=object_ids,
        atmosphere_model=None if atmosphere_model_raw in (None, "") else str(atmosphere_model_raw),
        default_nose_radius_m=float(raw.get("nose_radius_m", 0.5)),
        heat_rate_coefficient=float(raw.get("heat_rate_coefficient", SUTTON_GRAVES_COEFFICIENT_SI)),
        termination=termination,
        termination_by_object=termination_by_object,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _reentry_termination_from_raw(
    raw: dict[str, Any],
    *,
    base: ReentryTerminationConfig | None = None,
) -> ReentryTerminationConfig:
    base_cfg = base or ReentryTerminationConfig()
    return ReentryTerminationConfig(
        enabled=bool(raw.get("enabled", base_cfg.enabled)),
        terminate_on_entry=bool(raw.get("terminate_on_entry", base_cfg.terminate_on_entry)),
        min_altitude_km=_optional_float(raw.get("min_altitude_km", base_cfg.min_altitude_km)),
        max_dynamic_pressure_pa=_optional_float(raw.get("max_dynamic_pressure_pa", base_cfg.max_dynamic_pressure_pa)),
        max_drag_decel_m_s2=_optional_float(raw.get("max_drag_decel_m_s2", base_cfg.max_drag_decel_m_s2)),
        max_g_load=_optional_float(raw.get("max_g_load", base_cfg.max_g_load)),
        max_heat_rate_w_m2=_optional_float(raw.get("max_heat_rate_w_m2", base_cfg.max_heat_rate_w_m2)),
        max_heat_load_j_m2=_optional_float(raw.get("max_heat_load_j_m2", base_cfg.max_heat_load_j_m2)),
    )


def reentry_termination_for_object(cfg: ReentryConfig, object_id: str | None) -> ReentryTerminationConfig:
    if object_id is None:
        return cfg.termination
    return cfg.termination_by_object.get(str(object_id), cfg.termination)


def evaluate_reentry_termination(
    metrics: dict[str, float],
    cfg: ReentryConfig,
    object_id: str | None = None,
) -> str | None:
    term = reentry_termination_for_object(cfg, object_id)
    if not bool(term.enabled):
        return None
    active = float(metrics.get("active", 0.0)) > 0.5
    if not active:
        return None
    if bool(term.terminate_on_entry):
        return "reentry_entry"

    threshold_checks = (
        ("altitude_km", term.min_altitude_km, "reentry_min_altitude", "le"),
        ("dynamic_pressure_pa", term.max_dynamic_pressure_pa, "reentry_dynamic_pressure", "ge"),
        ("drag_decel_m_s2", term.max_drag_decel_m_s2, "reentry_drag_decel", "ge"),
        ("g_load", term.max_g_load, "reentry_g_load", "ge"),
        ("heat_rate_w_m2", term.max_heat_rate_w_m2, "reentry_heat_rate", "ge"),
        ("heat_load_j_m2", term.max_heat_load_j_m2, "reentry_heat_load", "ge"),
    )
    for metric_key, threshold, reason, comparison in threshold_checks:
        if threshold is None:
            continue
        value = float(metrics.get(metric_key, float("nan")))
        if not np.isfinite(value):
            continue
        if comparison == "le" and value <= float(threshold):
            return reason
        if comparison == "ge" and value >= float(threshold):
            return reason
    return None


def radial_altitude_km(r_eci_km: np.ndarray) -> float:
    r = float(np.linalg.norm(np.array(r_eci_km, dtype=float).reshape(3)))
    return float(r - EARTH_RADIUS_KM)


def reentry_metrics_for_state(
    *,
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    t_s: float,
    dt_s: float,
    cfg: ReentryConfig,
    props: ReentryObjectProperties,
    env: dict[str, Any],
    active: bool,
    previous_heat_load_j_m2: float = 0.0,
) -> dict[str, float]:
    altitude_km = radial_altitude_km(r_eci_km)
    out = {key: float("nan") for key in REENTRY_METRIC_KEYS}
    out["active"] = 1.0 if active else 0.0
    out["altitude_km"] = float(altitude_km)
    prev_heat = 0.0 if not np.isfinite(previous_heat_load_j_m2) else float(previous_heat_load_j_m2)
    out["heat_load_j_m2"] = prev_heat
    if not active:
        return out

    density_override = env.get("density_kg_m3")
    if density_override is None:
        model = str(cfg.atmosphere_model or env.get("atmosphere_model", "ussa1976")).lower()
        atmosphere = atmosphere_state_from_model(model, np.array(r_eci_km, dtype=float), float(t_s), env=env)
        rho = float(max(atmosphere.get("density_kg_m3", 0.0), 0.0))
    else:
        rho = float(max(float(density_override), 0.0))
    omega_raw = env.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
    v_rel = atmosphere_relative_velocity_eci_km_s(
        r_eci_km,
        v_eci_km_s,
        t_s=float(t_s),
        earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
        frame_model=str(env.get("drag_frame_model", "inertial_z")),
        jd_utc_start=env.get("jd_utc_start"),
        eop_path=env.get("drag_eop_path"),
    )
    v_rel_m_s = v_rel * 1000.0
    speed_m_s = float(np.linalg.norm(v_rel_m_s))
    q_dyn_pa = dynamic_pressure_pa(rho, speed_m_s)

    mass_kg = float(max(props.mass_kg, 1e-12))
    drag_area_m2 = float(max(props.drag_area_m2, 0.0))
    cd = float(max(props.cd, 0.0))
    lift_area_m2 = drag_area_m2 if props.lift_area_m2 is None else float(max(props.lift_area_m2, 0.0))
    loads = compute_aero_load_scalars(
        density_kg_m3=rho,
        speed_m_s=speed_m_s,
        mass_kg=mass_kg,
        drag_area_m2=drag_area_m2,
        cd=cd,
        lift_area_m2=lift_area_m2,
        cl=float(props.cl),
    )
    drag_decel_m_s2 = loads.drag_accel_m_s2
    lift_accel_m_s2 = loads.lift_accel_m_s2
    lift_to_drag = loads.lift_to_drag
    nose_radius_m = float(max(props.nose_radius_m, 1e-9))
    heat_rate_w_m2 = sutton_graves_heat_rate_w_m2(
        density_kg_m3=rho,
        speed_m_s=speed_m_s,
        nose_radius_m=nose_radius_m,
        coefficient=float(cfg.heat_rate_coefficient),
    )
    heat_load_j_m2 = prev_heat + max(float(dt_s), 0.0) * max(heat_rate_w_m2, 0.0)

    out.update(
        {
            "density_kg_m3": rho,
            "relative_speed_m_s": speed_m_s,
            "dynamic_pressure_pa": q_dyn_pa,
            "drag_decel_m_s2": drag_decel_m_s2,
            "lift_accel_m_s2": lift_accel_m_s2,
            "lift_to_drag": lift_to_drag,
            "g_load": drag_decel_m_s2 / G0_M_S2,
            "heat_rate_w_m2": heat_rate_w_m2,
            "heat_load_j_m2": heat_load_j_m2,
        }
    )
    return out
