from __future__ import annotations

import numpy as np

from sim.acceleration.optional import njit_or_identity

EARTH_RADIUS_KM = 6378.137
EARTH_J2 = 1.08262668e-3
EARTH_J3 = -2.53215306e-6
EARTH_J4 = -1.61098761e-6


@njit_or_identity(cache=True)
def two_body_accel_eci(r_eci_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    r2 = r_eci_km[0] * r_eci_km[0] + r_eci_km[1] * r_eci_km[1] + r_eci_km[2] * r_eci_km[2]
    out = np.zeros(3, dtype=np.float64)
    if r2 <= 0.0:
        return out
    r = np.sqrt(r2)
    scale = -mu_km3_s2 / (r * r2)
    out[0] = scale * r_eci_km[0]
    out[1] = scale * r_eci_km[1]
    out[2] = scale * r_eci_km[2]
    return out


@njit_or_identity(cache=True)
def j2_accel_eci(
    r_eci_km: np.ndarray,
    mu_km3_s2: float,
    j2: float = EARTH_J2,
    re_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    x = r_eci_km[0]
    y = r_eci_km[1]
    z = r_eci_km[2]
    r2 = x * x + y * y + z * z
    out = np.zeros(3, dtype=np.float64)
    if r2 <= 0.0:
        return out
    r = np.sqrt(r2)
    z2 = z * z
    f = 1.5 * j2 * mu_km3_s2 * (re_km * re_km) / (r**5)
    g = 5.0 * z2 / r2
    out[0] = f * x * (g - 1.0)
    out[1] = f * y * (g - 1.0)
    out[2] = f * z * (g - 3.0)
    return out


@njit_or_identity(cache=True)
def j3_accel_eci(
    r_eci_km: np.ndarray,
    mu_km3_s2: float,
    j3: float = EARTH_J3,
    re_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    x = r_eci_km[0]
    y = r_eci_km[1]
    z = r_eci_km[2]
    r2 = x * x + y * y + z * z
    out = np.zeros(3, dtype=np.float64)
    if r2 <= 0.0:
        return out
    r = np.sqrt(r2)
    s = z / r
    s2 = s * s
    s4 = s2 * s2
    axy_scale = mu_km3_s2 * j3 * (re_km**3) / (r**6)
    axy_factor = 2.5 * s * (7.0 * s2 - 3.0)
    az_scale = mu_km3_s2 * j3 * (re_km**3) / (r**5)
    az_factor = 0.5 * (35.0 * s4 - 30.0 * s2 + 3.0)
    out[0] = axy_scale * x * axy_factor
    out[1] = axy_scale * y * axy_factor
    out[2] = az_scale * az_factor
    return out


@njit_or_identity(cache=True)
def j4_accel_eci(
    r_eci_km: np.ndarray,
    mu_km3_s2: float,
    j4: float = EARTH_J4,
    re_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    x = r_eci_km[0]
    y = r_eci_km[1]
    z = r_eci_km[2]
    r2 = x * x + y * y + z * z
    out = np.zeros(3, dtype=np.float64)
    if r2 <= 0.0:
        return out
    r = np.sqrt(r2)
    s = z / r
    s2 = s * s
    s4 = s2 * s2
    axy_scale = mu_km3_s2 * j4 * (re_km**4) / (r**7)
    axy_factor = 0.625 * (63.0 * s4 - 42.0 * s2 + 3.0)
    az_scale = mu_km3_s2 * j4 * (re_km**4) / (r**6)
    az_factor = 0.625 * s * (63.0 * s4 - 70.0 * s2 + 15.0)
    out[0] = axy_scale * x * axy_factor
    out[1] = axy_scale * y * axy_factor
    out[2] = az_scale * az_factor
    return out


@njit_or_identity(cache=True)
def zonal_accel_eci(
    r_eci_km: np.ndarray,
    mu_km3_s2: float,
    include_j2: bool,
    include_j3: bool,
    include_j4: bool,
) -> np.ndarray:
    """Evaluate two-body plus selected zonals while sharing radial terms.

    Keep each component's arithmetic and the J2 -> J3 -> J4 accumulation
    order identical to the standalone kernels.  The fused implementation only
    removes repeated radius reductions, square roots, and temporary arrays.
    """

    x = r_eci_km[0]
    y = r_eci_km[1]
    z = r_eci_km[2]
    r2 = x * x + y * y + z * z
    out = np.zeros(3, dtype=np.float64)
    if r2 <= 0.0:
        return out
    r = np.sqrt(r2)
    scale = -mu_km3_s2 / (r * r2)
    out[0] = scale * x
    out[1] = scale * y
    out[2] = scale * z

    if include_j2:
        z2 = z * z
        f = 1.5 * EARTH_J2 * mu_km3_s2 * (EARTH_RADIUS_KM * EARTH_RADIUS_KM) / (r**5)
        g = 5.0 * z2 / r2
        out[0] += f * x * (g - 1.0)
        out[1] += f * y * (g - 1.0)
        out[2] += f * z * (g - 3.0)
    if include_j3:
        s = z / r
        s2 = s * s
        s4 = s2 * s2
        axy_scale = mu_km3_s2 * EARTH_J3 * (EARTH_RADIUS_KM**3) / (r**6)
        axy_factor = 2.5 * s * (7.0 * s2 - 3.0)
        az_scale = mu_km3_s2 * EARTH_J3 * (EARTH_RADIUS_KM**3) / (r**5)
        az_factor = 0.5 * (35.0 * s4 - 30.0 * s2 + 3.0)
        out[0] += axy_scale * x * axy_factor
        out[1] += axy_scale * y * axy_factor
        out[2] += az_scale * az_factor
    if include_j4:
        s = z / r
        s2 = s * s
        s4 = s2 * s2
        axy_scale = mu_km3_s2 * EARTH_J4 * (EARTH_RADIUS_KM**4) / (r**7)
        axy_factor = 0.625 * (63.0 * s4 - 42.0 * s2 + 3.0)
        az_scale = mu_km3_s2 * EARTH_J4 * (EARTH_RADIUS_KM**4) / (r**6)
        az_factor = 0.625 * s * (63.0 * s4 - 70.0 * s2 + 15.0)
        out[0] += axy_scale * x * axy_factor
        out[1] += axy_scale * y * axy_factor
        out[2] += az_scale * az_factor
    return out


@njit_or_identity(cache=True)
def rk4_zonal_step_state(
    x_eci: np.ndarray,
    dt_s: float,
    command_accel_eci_km_s2: np.ndarray,
    mu_km3_s2: float,
    include_j2: bool,
    include_j3: bool,
    include_j4: bool,
) -> np.ndarray:
    k1 = np.empty(6, dtype=np.float64)
    k2 = np.empty(6, dtype=np.float64)
    k3 = np.empty(6, dtype=np.float64)
    k4 = np.empty(6, dtype=np.float64)
    x_stage = np.empty(6, dtype=np.float64)
    _zonal_derivative_into(
        x_eci,
        command_accel_eci_km_s2,
        mu_km3_s2,
        include_j2,
        include_j3,
        include_j4,
        k1,
    )
    half_step = 0.5 * dt_s
    for index in range(6):
        x_stage[index] = half_step * k1[index]
        x_stage[index] += x_eci[index]
    _zonal_derivative_into(
        x_stage,
        command_accel_eci_km_s2,
        mu_km3_s2,
        include_j2,
        include_j3,
        include_j4,
        k2,
    )
    for index in range(6):
        x_stage[index] = half_step * k2[index]
        x_stage[index] += x_eci[index]
    _zonal_derivative_into(
        x_stage,
        command_accel_eci_km_s2,
        mu_km3_s2,
        include_j2,
        include_j3,
        include_j4,
        k3,
    )
    for index in range(6):
        x_stage[index] = dt_s * k3[index]
        x_stage[index] += x_eci[index]
    _zonal_derivative_into(
        x_stage,
        command_accel_eci_km_s2,
        mu_km3_s2,
        include_j2,
        include_j3,
        include_j4,
        k4,
    )
    result = np.empty(6, dtype=np.float64)
    step_scale = dt_s / 6.0
    for index in range(6):
        weighted = k1[index] + 2.0 * k2[index]
        weighted += 2.0 * k3[index]
        weighted += k4[index]
        result[index] = x_eci[index] + step_scale * weighted
    return result


@njit_or_identity(cache=True)
def _zonal_derivative(
    x_eci: np.ndarray,
    command_accel_eci_km_s2: np.ndarray,
    mu_km3_s2: float,
    include_j2: bool,
    include_j3: bool,
    include_j4: bool,
) -> np.ndarray:
    dx = np.empty(6, dtype=np.float64)
    _zonal_derivative_into(
        x_eci,
        command_accel_eci_km_s2,
        mu_km3_s2,
        include_j2,
        include_j3,
        include_j4,
        dx,
    )
    return dx


@njit_or_identity(cache=True)
def _zonal_derivative_into(
    x_eci: np.ndarray,
    command_accel_eci_km_s2: np.ndarray,
    mu_km3_s2: float,
    include_j2: bool,
    include_j3: bool,
    include_j4: bool,
    dx: np.ndarray,
) -> None:
    dx[0] = x_eci[3]
    dx[1] = x_eci[4]
    dx[2] = x_eci[5]
    acc = zonal_accel_eci(x_eci[:3], mu_km3_s2, include_j2, include_j3, include_j4)
    dx[3] = acc[0] + command_accel_eci_km_s2[0]
    dx[4] = acc[1] + command_accel_eci_km_s2[1]
    dx[5] = acc[2] + command_accel_eci_km_s2[2]
