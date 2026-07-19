"""Optional compiled cannonball solar-radiation-pressure kernel."""

from __future__ import annotations

import math

import numpy as np

from sim.acceleration.optional import njit_or_identity


@njit_or_identity(cache=True, fastmath=False)
def srp_acceleration_kernel(
    r_sc_eci_km: np.ndarray,
    sun_pos_eci_km: np.ndarray,
    mass_kg: float,
    area_m2: float,
    cr: float,
    pressure_n_m2: float,
    au_km: float,
    earth_radius_km: float,
    sun_radius_km: float,
    shadow_model: int,
) -> np.ndarray:
    """Evaluate SRP geometry, eclipse fraction, and acceleration together.

    ``shadow_model`` is 0 for none, 1 for cylindrical, and 2 for conical.
    """

    r_norm2 = float(np.dot(r_sc_eci_km, r_sc_eci_km))
    r_norm = float(math.sqrt(r_norm2)) if r_norm2 > 0.0 else 0.0
    sun_norm2 = float(np.dot(sun_pos_eci_km, sun_pos_eci_km))
    sun_norm = float(math.sqrt(sun_norm2)) if sun_norm2 > 0.0 else 0.0

    rho = sun_pos_eci_km - r_sc_eci_km
    rho_norm2 = float(np.dot(rho, rho))
    rho_norm = float(math.sqrt(rho_norm2)) if rho_norm2 > 0.0 else 0.0
    if rho_norm > 0.0:
        sun_dir_sc_eci = rho / rho_norm
        distance_scale = float((au_km / rho_norm) ** 2)
    else:
        sun_dir_sc_eci = np.zeros(3, dtype=np.float64)
        distance_scale = 1.0

    shadow = 1.0
    if shadow_model != 0:
        if r_norm <= earth_radius_km:
            shadow = 0.0
        elif rho_norm > 0.0:
            has_sun_direction = sun_norm > 0.0
            s_hat = np.zeros(3, dtype=np.float64)
            sunward = False
            if has_sun_direction:
                s_hat = sun_pos_eci_km / sun_norm
                sunward = float(np.dot(r_sc_eci_km, s_hat)) >= 0.0
            if not sunward:
                if shadow_model == 1:
                    if not has_sun_direction:
                        s_hat = sun_pos_eci_km / max(sun_norm, 1.0e-12)
                    r_sc_along_sun = float(np.dot(r_sc_eci_km, s_hat))
                    if r_sc_along_sun >= 0.0:
                        shadow = 1.0
                    else:
                        cross_track2 = max(
                            0.0,
                            float(np.dot(r_sc_eci_km, r_sc_eci_km))
                            - r_sc_along_sun * r_sc_along_sun,
                        )
                        shadow = (
                            0.0
                            if cross_track2 < earth_radius_km * earth_radius_km
                            else 1.0
                        )
                else:
                    earth_ratio = max(-1.0, min(1.0, earth_radius_km / r_norm))
                    sun_ratio = max(-1.0, min(1.0, sun_radius_km / rho_norm))
                    alpha = float(math.asin(earth_ratio))
                    beta = float(math.asin(sun_ratio))
                    u_earth = -r_sc_eci_km / r_norm
                    cosine_separation = max(
                        -1.0,
                        min(1.0, float(np.dot(u_earth, sun_dir_sc_eci))),
                    )
                    gamma = float(math.acos(cosine_separation))
                    if gamma >= alpha + beta:
                        shadow = 1.0
                    elif alpha > beta and gamma <= alpha - beta:
                        shadow = 0.0
                    elif beta > alpha and gamma <= beta - alpha:
                        shadow = max(0.0, 1.0 - (alpha * alpha) / (beta * beta))
                    else:
                        lo = abs(alpha - beta)
                        hi = alpha + beta
                        if hi <= lo:
                            shadow = 1.0
                        else:
                            fraction = (gamma - lo) / (hi - lo)
                            fraction = max(0.0, min(1.0, fraction))
                            if beta > alpha:
                                min_illumination = max(
                                    0.0,
                                    1.0 - (alpha * alpha) / (beta * beta),
                                )
                                shadow = max(
                                    0.0,
                                    min(
                                        1.0,
                                        min_illumination
                                        + (1.0 - min_illumination) * fraction,
                                    ),
                                )
                            else:
                                shadow = fraction

    if mass_kg <= 0.0 or area_m2 <= 0.0 or shadow <= 0.0:
        return np.zeros(3, dtype=np.float64)

    direction_norm2 = float(np.dot(sun_dir_sc_eci, sun_dir_sc_eci))
    if direction_norm2 <= 0.0:
        return np.zeros(3, dtype=np.float64)
    if abs(direction_norm2 - 1.0) > 1.0e-12:
        sun_dir_sc_eci = sun_dir_sc_eci / float(math.sqrt(direction_norm2))

    force_n = pressure_n_m2 * distance_scale * cr * area_m2
    acceleration_m_s2 = force_n / mass_kg
    return -(acceleration_m_s2 / 1.0e3) * shadow * sun_dir_sc_eci
