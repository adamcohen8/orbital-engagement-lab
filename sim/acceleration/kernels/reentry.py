from __future__ import annotations

import numpy as np

from sim.acceleration.optional import njit_or_identity

EARTH_RADIUS_KM = 6378.137
EARTH_ROT_RATE_RAD_S = 7.2921159e-5
G0_M_S2 = 9.80665


@njit_or_identity(cache=True)
def radial_altitude_km_kernel(r_eci_km: np.ndarray) -> float:
    return float(np.sqrt(np.dot(r_eci_km, r_eci_km)) - EARTH_RADIUS_KM)


@njit_or_identity(cache=True)
def atmosphere_relative_velocity_eci_km_s_kernel(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    earth_rotation_rad_s: float = EARTH_ROT_RATE_RAD_S,
) -> np.ndarray:
    out = np.empty(3, dtype=np.float64)
    out[0] = v_eci_km_s[0] + earth_rotation_rad_s * r_eci_km[1]
    out[1] = v_eci_km_s[1] - earth_rotation_rad_s * r_eci_km[0]
    out[2] = v_eci_km_s[2]
    return out


@njit_or_identity(cache=True)
def reentry_scalar_metrics_kernel(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    rho_kg_m3: float,
    mass_kg: float,
    drag_area_m2: float,
    cd: float,
    nose_radius_m: float,
    heat_rate_coefficient: float,
    dt_s: float,
    previous_heat_load_j_m2: float,
    earth_rotation_rad_s: float = EARTH_ROT_RATE_RAD_S,
) -> np.ndarray:
    v_rel = atmosphere_relative_velocity_eci_km_s_kernel(r_eci_km, v_eci_km_s, earth_rotation_rad_s)
    speed_m_s = 1000.0 * np.sqrt(np.dot(v_rel, v_rel))
    rho = max(rho_kg_m3, 0.0)
    q_dyn_pa = 0.5 * rho * speed_m_s * speed_m_s
    drag_decel_m_s2 = 0.5 * rho * max(cd, 0.0) * max(drag_area_m2, 0.0) / max(mass_kg, 1e-12)
    drag_decel_m_s2 *= speed_m_s * speed_m_s
    heat_rate_w_m2 = heat_rate_coefficient * np.sqrt(rho / max(nose_radius_m, 1e-9)) * speed_m_s**3
    prev_heat = previous_heat_load_j_m2 if np.isfinite(previous_heat_load_j_m2) else 0.0
    heat_load_j_m2 = prev_heat + max(dt_s, 0.0) * max(heat_rate_w_m2, 0.0)
    out = np.empty(6, dtype=np.float64)
    out[0] = speed_m_s
    out[1] = q_dyn_pa
    out[2] = drag_decel_m_s2
    out[3] = drag_decel_m_s2 / G0_M_S2
    out[4] = heat_rate_w_m2
    out[5] = heat_load_j_m2
    return out
