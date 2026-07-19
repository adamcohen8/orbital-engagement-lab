"""Optional compiled kernels for normalized spherical-harmonic gravity."""

from __future__ import annotations

import math

import numpy as np

from sim.acceleration.optional import njit_or_identity

_FLOAT64_EPS = 2.220446049250313e-16


@njit_or_identity(cache=True, fastmath=False)
def normalized_spherical_harmonic_accel_eci_kernel(
    r_eci_km: np.ndarray,
    eci_to_body_fixed: np.ndarray,
    mu_km3_s2: float,
    reference_radius_km: float,
    c_nm: np.ndarray,
    s_nm: np.ndarray,
    legendre_diag_scale: np.ndarray,
    legendre_subdiag_scale: np.ndarray,
    legendre_recur_a: np.ndarray,
    legendre_recur_b: np.ndarray,
    legendre_recur_c: np.ndarray,
    n_max: int,
    m_max: int,
) -> np.ndarray:
    """Evaluate normalized harmonic perturbation acceleration in ECI.

    The rotation is prepared by the authoritative frame implementation before
    entering this numeric kernel.  This keeps EOP and frame-model behavior in
    one owner while compiling the degree/order recurrence and summation loops.
    """

    r_eci = np.asarray(r_eci_km, dtype=np.float64)
    rotation = np.asarray(eci_to_body_fixed, dtype=np.float64)
    r_body_fixed = rotation @ r_eci
    x_bf = r_body_fixed[0]
    y_bf = r_body_fixed[1]
    z_bf = r_body_fixed[2]
    distance_squared = x_bf * x_bf + y_bf * y_bf + z_bf * z_bf
    if distance_squared <= 0.0:
        return np.zeros(3, dtype=np.float64)
    distance = math.sqrt(distance_squared)
    latitude_gc = math.asin(z_bf / distance)
    sin_latitude = math.sin(latitude_gc)
    cos_latitude = math.cos(latitude_gc)

    p_nm = np.zeros((n_max + 1, m_max + 1), dtype=np.float64)
    dp_nm = np.zeros((n_max + 1, m_max + 1), dtype=np.float64)
    p_nm[0, 0] = 1.0
    if n_max >= 1 and m_max >= 1:
        sqrt_three = math.sqrt(3.0)
        p_nm[1, 1] = sqrt_three * cos_latitude
        dp_nm[1, 1] = -sqrt_three * sin_latitude

    for degree in range(2, n_max + 1):
        if degree <= m_max:
            scale = legendre_diag_scale[degree]
            previous = p_nm[degree - 1, degree - 1]
            p_nm[degree, degree] = scale * cos_latitude * previous
            dp_nm[degree, degree] = scale * (
                cos_latitude * dp_nm[degree - 1, degree - 1] - sin_latitude * previous
            )

    for degree in range(1, n_max + 1):
        order = degree - 1
        if order <= m_max:
            scale = legendre_subdiag_scale[degree]
            p_nm[degree, order] = scale * sin_latitude * p_nm[degree - 1, order]
            dp_nm[degree, order] = scale * (
                cos_latitude * p_nm[degree - 1, order]
                + sin_latitude * dp_nm[degree - 1, order]
            )

    for order in range(m_max + 1):
        for degree in range(order + 2, n_max + 1):
            a_scale = legendre_recur_a[degree, order]
            b_scale = legendre_recur_b[degree, order]
            c_scale = legendre_recur_c[degree, order]
            b_value = b_scale * sin_latitude * p_nm[degree - 1, order]
            c_value = c_scale * p_nm[degree - 2, order]
            p_nm[degree, order] = a_scale * (b_value - c_value)
            db_value = b_scale * sin_latitude * dp_nm[degree - 1, order]
            dc_value = b_scale * cos_latitude * p_nm[degree - 1, order]
            dd_value = c_scale * dp_nm[degree - 2, order]
            dp_nm[degree, order] = a_scale * (db_value + dc_value - dd_value)

    longitude = math.atan2(y_bf, x_bf)
    cos_order_longitude = np.empty(m_max + 1, dtype=np.float64)
    sin_order_longitude = np.empty(m_max + 1, dtype=np.float64)
    for order in range(m_max + 1):
        cos_order_longitude[order] = math.cos(order * longitude)
        sin_order_longitude[order] = math.sin(order * longitude)

    radial_derivative = 0.0
    latitude_derivative = 0.0
    longitude_derivative = 0.0
    for degree in range(n_max + 1):
        radius_power = (reference_radius_km / distance) ** degree
        radial_scale = (-mu_km3_s2 / distance_squared) * radius_power * (degree + 1.0)
        angular_scale = (mu_km3_s2 / distance) * radius_power
        radial_sum = 0.0
        latitude_sum = 0.0
        longitude_sum = 0.0
        upper_order = min(m_max, degree)
        for order in range(upper_order + 1):
            cos_ml = cos_order_longitude[order]
            sin_ml = sin_order_longitude[order]
            cosine_coefficient = c_nm[degree, order]
            sine_coefficient = s_nm[degree, order]
            amplitude = cosine_coefficient * cos_ml + sine_coefficient * sin_ml
            radial_sum += p_nm[degree, order] * amplitude
            latitude_sum += dp_nm[degree, order] * amplitude
            longitude_sum += order * p_nm[degree, order] * (
                sine_coefficient * cos_ml - cosine_coefficient * sin_ml
            )
        radial_derivative += radial_sum * radial_scale
        latitude_derivative += latitude_sum * angular_scale
        longitude_derivative += longitude_sum * angular_scale

    xy_radius_squared = x_bf * x_bf + y_bf * y_bf
    if xy_radius_squared <= 0.0:
        xy_radius_squared = _FLOAT64_EPS
    xy_radius = math.sqrt(xy_radius_squared)
    common = (
        radial_derivative / distance
        - z_bf * latitude_derivative / (distance_squared * xy_radius)
    )
    acceleration_body_fixed = np.empty(3, dtype=np.float64)
    acceleration_body_fixed[0] = (
        common * x_bf - longitude_derivative * y_bf / xy_radius_squared
    )
    acceleration_body_fixed[1] = (
        common * y_bf + longitude_derivative * x_bf / xy_radius_squared
    )
    acceleration_body_fixed[2] = (
        radial_derivative * z_bf / distance
        + xy_radius * latitude_derivative / distance_squared
    )
    acceleration_eci = rotation.T @ acceleration_body_fixed

    r2_eci = r_eci[0] * r_eci[0] + r_eci[1] * r_eci[1] + r_eci[2] * r_eci[2]
    if r2_eci <= 0.0:
        return np.zeros(3, dtype=np.float64)
    r_eci_norm = math.sqrt(r2_eci)
    two_body_scale = -mu_km3_s2 / (r_eci_norm * r2_eci)
    for axis in range(3):
        acceleration_eci[axis] -= two_body_scale * r_eci[axis]
    return acceleration_eci
