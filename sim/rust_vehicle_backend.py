"""Rust kernels for rocket, re-entry, and spherical collision math.

Python remains the owner of lifecycle,
configuration, atmospheric, frame, and event semantics.  This module is an
adapter for the primary numerical kernels exported by the
``oel_rust_orbit`` wheel.  It intentionally does not hide a missing wheel or
silently change the selected backend.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from math import isfinite

import numpy as np


@lru_cache(maxsize=1)
def _extension():
    try:
        return import_module("oel_rust_orbit")
    except ImportError as exc:
        raise RuntimeError(
            "Rust vehicle backend requested but oel_rust_orbit is unavailable; "
            "install the optional oel_rust_orbit wheel first"
        ) from exc


def _vector(value, name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    values = array.tolist()
    if not all(isfinite(item) for item in values):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _row(value, size: int, name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values")
    values = array.tolist()
    if not all(isfinite(item) for item in values):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _require(name: str):
    function = getattr(_extension(), name, None)
    if function is None:
        raise RuntimeError(
            f"Rust vehicle backend requires an oel_rust_orbit wheel exporting {name}"
        )
    return function


def rocket_aero_state(
    *,
    rho_kg_m3: float,
    pressure_pa: float,
    temperature_k: float,
    sound_speed_m_s: float,
    v_rel_body_m_s: np.ndarray,
    alpha_limit_deg: float,
    beta_limit_deg: float,
) -> np.ndarray:
    result = _require("rocket_aero_state")(
        float(rho_kg_m3),
        float(pressure_pa),
        float(temperature_k),
        float(sound_speed_m_s),
        _vector(v_rel_body_m_s, "v_rel_body_m_s"),
        float(alpha_limit_deg),
        float(beta_limit_deg),
    )
    return np.asarray(result, dtype=np.float64).reshape(9)


def rocket_aero_loads(
    *,
    v_rel_body_m_s: np.ndarray,
    atmosphere: np.ndarray,
    enabled: bool,
    reference_area_m2: float,
    reference_length_m: float,
    cp_offset_body_m: np.ndarray,
    cd_base: float,
    cd_alpha2: float,
    cd_supersonic: float,
    transonic_peak_cd: float,
    transonic_mach: float,
    transonic_width: float,
    cl_alpha_per_rad: float,
    cy_beta_per_rad: float,
    cm_alpha_per_rad: float,
    cn_beta_per_rad: float,
    cl_roll_per_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    result = _require("rocket_aero_loads")(
        _vector(v_rel_body_m_s, "v_rel_body_m_s"),
        _row(atmosphere, 9, "atmosphere"),
        bool(enabled),
        float(reference_area_m2),
        float(reference_length_m),
        _vector(cp_offset_body_m, "cp_offset_body_m"),
        float(cd_base),
        float(cd_alpha2),
        float(cd_supersonic),
        float(transonic_peak_cd),
        float(transonic_mach),
        float(transonic_width),
        float(cl_alpha_per_rad),
        float(cy_beta_per_rad),
        float(cm_alpha_per_rad),
        float(cn_beta_per_rad),
        float(cl_roll_per_rad),
    )
    values = np.asarray(result, dtype=np.float64).reshape(13)
    return values[0:3], values[3:6], values[6:9], values[9:12], float(values[12])


def rocket_stage_engine_perf(
    *,
    pressure_pa: float,
    sea_level_thrust_n: float,
    vacuum_thrust_n: float,
    sea_level_isp_s: float,
    vacuum_isp_s: float,
) -> tuple[float, float]:
    values = _require("rocket_stage_engine_perf")(
        float(pressure_pa),
        float(sea_level_thrust_n),
        float(vacuum_thrust_n),
        float(sea_level_isp_s),
        float(vacuum_isp_s),
    )
    return float(values[0]), float(values[1])


def rocket_propellant_step(
    *,
    propellant_left_kg: float,
    throttle: float,
    pressure_pa: float,
    dt_s: float,
    mass_start_kg: float,
    sea_level_thrust_n: float,
    vacuum_thrust_n: float,
    sea_level_isp_s: float,
    vacuum_isp_s: float,
) -> np.ndarray:
    values = np.asarray(
        _require("rocket_propellant_step")(
            float(propellant_left_kg),
            float(throttle),
            float(pressure_pa),
            float(dt_s),
            float(mass_start_kg),
            float(sea_level_thrust_n),
            float(vacuum_thrust_n),
            float(sea_level_isp_s),
            float(vacuum_isp_s),
        ),
        dtype=np.float64,
    )
    return values


def reentry_metrics(
    *,
    density_kg_m3: float,
    speed_m_s: float,
    mass_kg: float,
    drag_area_m2: float,
    cd: float,
    lift_area_m2: float,
    cl: float,
    nose_radius_m: float,
    coefficient: float,
    dt_s: float,
    previous_heat_load_j_m2: float,
    previous_heat_rate_w_m2: float | None,
) -> np.ndarray:
    previous_rate = float("nan") if previous_heat_rate_w_m2 is None else float(previous_heat_rate_w_m2)
    result = _require("reentry_metrics")(
        float(density_kg_m3),
        float(speed_m_s),
        float(mass_kg),
        float(drag_area_m2),
        float(cd),
        float(lift_area_m2),
        float(cl),
        float(nose_radius_m),
        float(coefficient),
        float(dt_s),
        float(previous_heat_load_j_m2),
        previous_rate,
    )
    return np.asarray(result, dtype=np.float64)


def collision_chord_geometry(start_rel: np.ndarray, end_rel: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(
        _require("collision_chord_geometry")(
            _vector(start_rel, "start_rel"), _vector(end_rel, "end_rel")
        ),
        dtype=np.float64,
    ).reshape(3)
    return float(values[0]), float(values[1]), float(values[2])


def collision_linear_contact_fraction(
    start_rel: np.ndarray,
    end_rel: np.ndarray,
    radius_km: float,
) -> float | None:
    result = _require("collision_linear_contact_fraction")(
        _vector(start_rel, "start_rel"), _vector(end_rel, "end_rel"), float(radius_km)
    )
    return None if result is None else float(result)


def collision_elastic_impact(
    *,
    position_a: np.ndarray,
    position_b: np.ndarray,
    velocity_a: np.ndarray,
    velocity_b: np.ndarray,
    mass_a_kg: float,
    mass_b_kg: float,
    restitution: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    values = np.asarray(
        _require("collision_elastic_impact")(
            _vector(position_a, "position_a"),
            _vector(position_b, "position_b"),
            _vector(velocity_a, "velocity_a"),
            _vector(velocity_b, "velocity_b"),
            float(mass_a_kg),
            float(mass_b_kg),
            float(restitution),
        ),
        dtype=np.float64,
    ).reshape(10)
    return values[0:3], values[3:6], values[6:9], float(values[9])


__all__ = [
    "collision_chord_geometry",
    "collision_elastic_impact",
    "collision_linear_contact_fraction",
    "reentry_metrics",
    "rocket_aero_loads",
    "rocket_aero_state",
    "rocket_propellant_step",
    "rocket_stage_engine_perf",
]
