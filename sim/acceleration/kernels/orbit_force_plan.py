"""Compiled orbit integration and component kernels for numeric force plans."""

from __future__ import annotations

import math

import numpy as np

from sim.acceleration.kernels.geodesy import ecef_to_geodetic_deg_km_kernel
from sim.acceleration.kernels.nrlmsise00 import quiet_thermosphere_density_kernel
from sim.acceleration.kernels.spherical_harmonics import (
    normalized_spherical_harmonic_accel_eci_kernel,
)
from sim.acceleration.kernels.srp import srp_acceleration_kernel
from sim.acceleration.optional import njit_or_identity

FORCE_SPHERICAL_HARMONICS = 1
FORCE_DRAG = 2
FORCE_SRP = 3
FORCE_THIRD_BODY_SUN = 4
FORCE_THIRD_BODY_MOON = 5
FORCE_J2 = 6
FORCE_J3 = 7
FORCE_J4 = 8
FORCE_LIFT = 9
FORCE_THIRD_BODY_PLANETS = 10

DENSITY_NONE = 0
DENSITY_CONSTANT = 1
DENSITY_NRLMSISE00_QUIET_THERMOSPHERE = 2


@njit_or_identity(cache=True, fastmath=False)
def _dot3_numpy_accelerate(a: np.ndarray, b: np.ndarray) -> float:
    """Match the fused length-three reduction used by NumPy/Accelerate."""

    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


@njit_or_identity(cache=True, fastmath=False)
def _third_body_acceleration(
    r_eci_km: np.ndarray,
    body_pos_eci_km: np.ndarray,
    body_mu_km3_s2: float,
) -> np.ndarray:
    rb = body_pos_eci_km - r_eci_km
    rb_norm2 = _dot3_numpy_accelerate(rb, rb)
    b_norm2 = _dot3_numpy_accelerate(body_pos_eci_km, body_pos_eci_km)
    rb_norm = float(np.sqrt(rb_norm2)) if rb_norm2 > 0.0 else 0.0
    b_norm = float(np.sqrt(b_norm2)) if b_norm2 > 0.0 else 0.0
    if rb_norm == 0.0 or b_norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    return body_mu_km3_s2 * (rb / (rb_norm**3) - body_pos_eci_km / (b_norm**3))


@njit_or_identity(cache=True, fastmath=False)
def _drag_acceleration(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    rotation: np.ndarray,
    density_kg_m3: float,
    mass_kg: float,
    cd: float,
    area_m2: float,
    earth_rotation_rad_s: float,
) -> np.ndarray:
    if density_kg_m3 <= 0.0 or mass_kg <= 0.0 or area_m2 <= 0.0:
        return np.zeros(3, dtype=np.float64)
    r_frame = rotation @ r_eci_km
    v_frame = rotation @ v_eci_km_s
    v_atm_frame = np.empty(3, dtype=np.float64)
    v_atm_frame[0] = -earth_rotation_rad_s * r_frame[1]
    v_atm_frame[1] = earth_rotation_rad_s * r_frame[0]
    v_atm_frame[2] = 0.0
    v_rel_m_s = (rotation.T @ (v_frame - v_atm_frame)) * 1.0e3
    v_norm2 = _dot3_numpy_accelerate(v_rel_m_s, v_rel_m_s)
    if v_norm2 == 0.0:
        return np.zeros(3, dtype=np.float64)
    v_norm = float(np.sqrt(v_norm2))
    a_m_s2 = -0.5 * density_kg_m3 * cd * area_m2 / mass_kg * v_norm * v_rel_m_s
    return a_m_s2 / 1.0e3


@njit_or_identity(cache=True, fastmath=False)
def _zonal_acceleration_from_radius_into(
    r_eci_km: np.ndarray,
    mu_km3_s2: float,
    force_code: int,
    r2: float,
    r: float,
    out: np.ndarray,
) -> None:
    """Evaluate one zonal component into caller-owned storage."""

    x = r_eci_km[0]
    y = r_eci_km[1]
    z = r_eci_km[2]
    if r2 <= 0.0:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        return
    re_km = 6378.137
    if force_code == FORCE_J2:
        z2 = z * z
        f = 1.5 * 1.08262668e-3 * mu_km3_s2 * re_km * re_km / (r**5)
        g = 5.0 * z2 / r2
        out[0] = f * x * (g - 1.0)
        out[1] = f * y * (g - 1.0)
        out[2] = f * z * (g - 3.0)
    elif force_code == FORCE_J3:
        s = z / r
        s2 = s * s
        s4 = s2 * s2
        axy_scale = mu_km3_s2 * -2.53215306e-6 * (re_km**3) / (r**6)
        axy_factor = 2.5 * s * (7.0 * s2 - 3.0)
        az_scale = mu_km3_s2 * -2.53215306e-6 * (re_km**3) / (r**5)
        az_factor = 0.5 * (35.0 * s4 - 30.0 * s2 + 3.0)
        out[0] = axy_scale * x * axy_factor
        out[1] = axy_scale * y * axy_factor
        out[2] = az_scale * az_factor
    else:
        s = z / r
        s2 = s * s
        s4 = s2 * s2
        axy_scale = mu_km3_s2 * -1.61098761e-6 * (re_km**4) / (r**7)
        axy_factor = 0.625 * (63.0 * s4 - 42.0 * s2 + 3.0)
        az_scale = mu_km3_s2 * -1.61098761e-6 * (re_km**4) / (r**6)
        az_factor = 0.625 * s * (63.0 * s4 - 70.0 * s2 + 15.0)
        out[0] = axy_scale * x * axy_factor
        out[1] = axy_scale * y * axy_factor
        out[2] = az_scale * az_factor


@njit_or_identity(cache=True, fastmath=False)
def _zonal_acceleration_from_radius(
    r_eci_km: np.ndarray,
    mu_km3_s2: float,
    force_code: int,
    r2: float,
    r: float,
) -> np.ndarray:
    """Evaluate one zonal component from shared radial terms."""

    out = np.empty(3, dtype=np.float64)
    _zonal_acceleration_from_radius_into(r_eci_km, mu_km3_s2, force_code, r2, r, out)
    return out


@njit_or_identity(cache=True, fastmath=False)
def _zonal_acceleration(r_eci_km: np.ndarray, mu_km3_s2: float, force_code: int) -> np.ndarray:
    """Evaluate one zonal component when no shared radius is available."""

    r2 = _dot3_numpy_accelerate(r_eci_km, r_eci_km)
    r = np.sqrt(r2) if r2 > 0.0 else 0.0
    return _zonal_acceleration_from_radius(r_eci_km, mu_km3_s2, force_code, r2, r)


@njit_or_identity(cache=True, fastmath=False)
def _lift_acceleration(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    rotation: np.ndarray,
    density_kg_m3: float,
    mass_kg: float,
    cl: float,
    area_m2: float,
    earth_rotation_rad_s: float,
    lift_direction_eci: np.ndarray,
) -> np.ndarray:
    if density_kg_m3 <= 0.0 or mass_kg <= 0.0 or area_m2 <= 0.0 or cl == 0.0:
        return np.zeros(3, dtype=np.float64)
    r_frame = rotation @ r_eci_km
    v_frame = rotation @ v_eci_km_s
    v_atm_frame = np.empty(3, dtype=np.float64)
    v_atm_frame[0] = -earth_rotation_rad_s * r_frame[1]
    v_atm_frame[1] = earth_rotation_rad_s * r_frame[0]
    v_atm_frame[2] = 0.0
    v_rel_m_s = (rotation.T @ (v_frame - v_atm_frame)) * 1.0e3
    speed_m_s = np.sqrt(_dot3_numpy_accelerate(v_rel_m_s, v_rel_m_s))
    if speed_m_s <= 0.0:
        return np.zeros(3, dtype=np.float64)
    v_hat = v_rel_m_s / speed_m_s
    desired_norm = np.sqrt(_dot3_numpy_accelerate(lift_direction_eci, lift_direction_eci))
    if desired_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)
    desired = lift_direction_eci / desired_norm
    lift_dir = desired - _dot3_numpy_accelerate(desired, v_hat) * v_hat
    lift_norm = np.sqrt(_dot3_numpy_accelerate(lift_dir, lift_dir))
    if lift_norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    q_dyn_pa = 0.5 * density_kg_m3 * speed_m_s * speed_m_s
    return (q_dyn_pa * area_m2 * cl / mass_kg / 1.0e3) * (lift_dir / lift_norm)


@njit_or_identity(cache=True, fastmath=False)
def builtin_force_components_kernel(
    x_eci: np.ndarray,
    force_codes: np.ndarray,
    plugin_densities_kg_m3: np.ndarray,
    harmonic_rotation: np.ndarray,
    drag_rotation: np.ndarray,
    sun_position_eci_km: np.ndarray,
    moon_position_eci_km: np.ndarray,
    planet_positions_eci_km: np.ndarray,
    planet_mu_km3_s2: np.ndarray,
    planet_count: int,
    scalar_parameters: np.ndarray,
    shadow_model: int,
    lift_direction_eci: np.ndarray,
    lift_coefficient: float,
    lift_area_m2: float,
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
    """Evaluate per-plugin built-in force components without changing sum order."""

    components = np.zeros((force_codes.size, 3), dtype=np.float64)
    r_eci = x_eci[:3]
    zonal_r2 = -1.0
    zonal_r = 0.0
    for plugin_index in range(force_codes.size):
        force_code = force_codes[plugin_index]
        if force_code == FORCE_SPHERICAL_HARMONICS:
            components[plugin_index] = normalized_spherical_harmonic_accel_eci_kernel(
                r_eci,
                harmonic_rotation,
                scalar_parameters[14],
                scalar_parameters[1],
                c_nm,
                s_nm,
                legendre_diag_scale,
                legendre_subdiag_scale,
                legendre_recur_a,
                legendre_recur_b,
                legendre_recur_c,
                n_max,
                m_max,
            )
        elif force_code == FORCE_DRAG:
            components[plugin_index] = _drag_acceleration(
                r_eci,
                x_eci[3:],
                drag_rotation,
                plugin_densities_kg_m3[plugin_index],
                scalar_parameters[2],
                scalar_parameters[3],
                scalar_parameters[4],
                scalar_parameters[5],
            )
        elif force_code == FORCE_LIFT:
            components[plugin_index] = _lift_acceleration(
                r_eci,
                x_eci[3:],
                drag_rotation,
                plugin_densities_kg_m3[plugin_index],
                scalar_parameters[2],
                lift_coefficient,
                lift_area_m2,
                scalar_parameters[5],
                lift_direction_eci,
            )
        elif force_code == FORCE_SRP:
            components[plugin_index] = srp_acceleration_kernel(
                r_eci,
                sun_position_eci_km,
                scalar_parameters[2],
                scalar_parameters[6],
                scalar_parameters[7],
                scalar_parameters[8],
                scalar_parameters[9],
                scalar_parameters[10],
                scalar_parameters[11],
                shadow_model,
            )
        elif force_code == FORCE_THIRD_BODY_SUN:
            components[plugin_index] = _third_body_acceleration(r_eci, sun_position_eci_km, scalar_parameters[12])
        elif force_code == FORCE_THIRD_BODY_MOON:
            components[plugin_index] = _third_body_acceleration(r_eci, moon_position_eci_km, scalar_parameters[13])
        elif force_code in (FORCE_J2, FORCE_J3, FORCE_J4):
            if zonal_r2 < 0.0:
                zonal_r2 = _dot3_numpy_accelerate(r_eci, r_eci)
                zonal_r = np.sqrt(zonal_r2) if zonal_r2 > 0.0 else 0.0
            _zonal_acceleration_from_radius_into(
                r_eci,
                scalar_parameters[14],
                force_code,
                zonal_r2,
                zonal_r,
                components[plugin_index],
            )
        elif force_code == FORCE_THIRD_BODY_PLANETS:
            total = np.zeros(3, dtype=np.float64)
            for planet_index in range(planet_count):
                total += _third_body_acceleration(
                    r_eci,
                    planet_positions_eci_km[planet_index],
                    planet_mu_km3_s2[planet_index],
                )
            components[plugin_index] = total
    return components


@njit_or_identity(cache=True, fastmath=False)
def _stage_derivative(
    x_eci: np.ndarray,
    command_accel_eci_km_s2: np.ndarray,
    stage_index: int,
    force_codes: np.ndarray,
    harmonic_rotations: np.ndarray,
    density_rotations: np.ndarray,
    drag_rotations: np.ndarray,
    sun_positions_eci_km: np.ndarray,
    moon_positions_eci_km: np.ndarray,
    atmosphere_inputs: np.ndarray,
    scalar_parameters: np.ndarray,
    density_mode: int,
    constant_density_kg_m3: float,
    shadow_model: int,
    c_nm: np.ndarray,
    s_nm: np.ndarray,
    legendre_diag_scale: np.ndarray,
    legendre_subdiag_scale: np.ndarray,
    legendre_recur_a: np.ndarray,
    legendre_recur_b: np.ndarray,
    legendre_recur_c: np.ndarray,
    n_max: int,
    m_max: int,
    pt1: np.ndarray,
    ps1: np.ndarray,
    pd1: np.ndarray,
    pdl1: np.ndarray,
    ptm1: np.ndarray,
    pdm1: np.ndarray,
    ptl1: np.ndarray,
    pma1: np.ndarray,
    zn1: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, bool]:
    mu_km3_s2 = scalar_parameters[0]
    r_eci = x_eci[:3]
    r2 = _dot3_numpy_accelerate(r_eci, r_eci)
    if r2 == 0.0:
        acceleration = np.zeros(3, dtype=np.float64)
    else:
        r_norm = float(np.sqrt(r2))
        acceleration = (-mu_km3_s2 / (r_norm * r2)) * r_eci
    acceleration += command_accel_eci_km_s2

    for force_code in force_codes:
        if force_code == FORCE_SPHERICAL_HARMONICS:
            acceleration += normalized_spherical_harmonic_accel_eci_kernel(
                r_eci,
                harmonic_rotations[stage_index],
                scalar_parameters[14],
                scalar_parameters[1],
                c_nm,
                s_nm,
                legendre_diag_scale,
                legendre_subdiag_scale,
                legendre_recur_a,
                legendre_recur_b,
                legendre_recur_c,
                n_max,
                m_max,
            )
        elif force_code == FORCE_DRAG:
            density = constant_density_kg_m3
            if density_mode == DENSITY_NRLMSISE00_QUIET_THERMOSPHERE:
                r_ecef = density_rotations[stage_index] @ r_eci
                lat_deg, lon_deg, alt_km = ecef_to_geodetic_deg_km_kernel(r_ecef)
                alt_km = max(alt_km, 0.0)
                if alt_km < 300.0:
                    return np.zeros(6, dtype=np.float64), False
                hour_angle = (
                    atmosphere_inputs[stage_index, 4]
                    + math.radians(lon_deg)
                    - atmosphere_inputs[stage_index, 5]
                    + math.pi
                ) % (2.0 * math.pi) - math.pi
                lst_hr = (12.0 + hour_angle * 12.0 / math.pi) % 24.0
                density = quiet_thermosphere_density_kernel(
                    int(atmosphere_inputs[stage_index, 0]),
                    atmosphere_inputs[stage_index, 1],
                    alt_km,
                    lat_deg,
                    lon_deg,
                    lst_hr,
                    atmosphere_inputs[stage_index, 2],
                    atmosphere_inputs[stage_index, 3],
                    pt1,
                    ps1,
                    pd1,
                    pdl1,
                    ptm1,
                    pdm1,
                    ptl1,
                    pma1,
                    zn1,
                    alpha,
                )
            acceleration += _drag_acceleration(
                r_eci,
                x_eci[3:],
                drag_rotations[stage_index],
                density,
                scalar_parameters[2],
                scalar_parameters[3],
                scalar_parameters[4],
                scalar_parameters[5],
            )
        elif force_code == FORCE_SRP:
            acceleration += srp_acceleration_kernel(
                r_eci,
                sun_positions_eci_km[stage_index],
                scalar_parameters[2],
                scalar_parameters[6],
                scalar_parameters[7],
                scalar_parameters[8],
                scalar_parameters[9],
                scalar_parameters[10],
                scalar_parameters[11],
                shadow_model,
            )
        elif force_code == FORCE_THIRD_BODY_SUN:
            acceleration += _third_body_acceleration(
                r_eci,
                sun_positions_eci_km[stage_index],
                scalar_parameters[12],
            )
        elif force_code == FORCE_THIRD_BODY_MOON:
            acceleration += _third_body_acceleration(
                r_eci,
                moon_positions_eci_km[stage_index],
                scalar_parameters[13],
            )

    derivative = np.empty(6, dtype=np.float64)
    derivative[:3] = x_eci[3:]
    derivative[3:] = acceleration
    return derivative, True


@njit_or_identity(cache=True, fastmath=False)
def rk4_builtin_force_plan_step_kernel(
    x_eci: np.ndarray,
    dt_s: float,
    command_accel_eci_km_s2: np.ndarray,
    force_codes: np.ndarray,
    harmonic_rotations: np.ndarray,
    density_rotations: np.ndarray,
    drag_rotations: np.ndarray,
    sun_positions_eci_km: np.ndarray,
    moon_positions_eci_km: np.ndarray,
    atmosphere_inputs: np.ndarray,
    scalar_parameters: np.ndarray,
    density_mode: int,
    constant_density_kg_m3: float,
    shadow_model: int,
    c_nm: np.ndarray,
    s_nm: np.ndarray,
    legendre_diag_scale: np.ndarray,
    legendre_subdiag_scale: np.ndarray,
    legendre_recur_a: np.ndarray,
    legendre_recur_b: np.ndarray,
    legendre_recur_c: np.ndarray,
    n_max: int,
    m_max: int,
    pt1: np.ndarray,
    ps1: np.ndarray,
    pd1: np.ndarray,
    pdl1: np.ndarray,
    ptm1: np.ndarray,
    pdm1: np.ndarray,
    ptl1: np.ndarray,
    pma1: np.ndarray,
    zn1: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Advance one RK4 step, returning ``False`` when a safe fallback is needed."""

    k1, valid = _stage_derivative(
        x_eci,
        command_accel_eci_km_s2,
        0,
        force_codes,
        harmonic_rotations,
        density_rotations,
        drag_rotations,
        sun_positions_eci_km,
        moon_positions_eci_km,
        atmosphere_inputs,
        scalar_parameters,
        density_mode,
        constant_density_kg_m3,
        shadow_model,
        c_nm,
        s_nm,
        legendre_diag_scale,
        legendre_subdiag_scale,
        legendre_recur_a,
        legendre_recur_b,
        legendre_recur_c,
        n_max,
        m_max,
        pt1,
        ps1,
        pd1,
        pdl1,
        ptm1,
        pdm1,
        ptl1,
        pma1,
        zn1,
        alpha,
    )
    if not valid:
        return x_eci.copy(), False
    half_step = 0.5 * dt_s
    x_stage = np.empty(6, dtype=np.float64)
    for index in range(6):
        x_stage[index] = k1[index] * half_step
        x_stage[index] += x_eci[index]
    k2, valid = _stage_derivative(
        x_stage,
        command_accel_eci_km_s2,
        1,
        force_codes,
        harmonic_rotations,
        density_rotations,
        drag_rotations,
        sun_positions_eci_km,
        moon_positions_eci_km,
        atmosphere_inputs,
        scalar_parameters,
        density_mode,
        constant_density_kg_m3,
        shadow_model,
        c_nm,
        s_nm,
        legendre_diag_scale,
        legendre_subdiag_scale,
        legendre_recur_a,
        legendre_recur_b,
        legendre_recur_c,
        n_max,
        m_max,
        pt1,
        ps1,
        pd1,
        pdl1,
        ptm1,
        pdm1,
        ptl1,
        pma1,
        zn1,
        alpha,
    )
    if not valid:
        return x_eci.copy(), False
    for index in range(6):
        x_stage[index] = k2[index] * half_step
        x_stage[index] += x_eci[index]
    k3, valid = _stage_derivative(
        x_stage,
        command_accel_eci_km_s2,
        1,
        force_codes,
        harmonic_rotations,
        density_rotations,
        drag_rotations,
        sun_positions_eci_km,
        moon_positions_eci_km,
        atmosphere_inputs,
        scalar_parameters,
        density_mode,
        constant_density_kg_m3,
        shadow_model,
        c_nm,
        s_nm,
        legendre_diag_scale,
        legendre_subdiag_scale,
        legendre_recur_a,
        legendre_recur_b,
        legendre_recur_c,
        n_max,
        m_max,
        pt1,
        ps1,
        pd1,
        pdl1,
        ptm1,
        pdm1,
        ptl1,
        pma1,
        zn1,
        alpha,
    )
    if not valid:
        return x_eci.copy(), False
    for index in range(6):
        x_stage[index] = k3[index] * dt_s
        x_stage[index] += x_eci[index]
    k4, valid = _stage_derivative(
        x_stage,
        command_accel_eci_km_s2,
        2,
        force_codes,
        harmonic_rotations,
        density_rotations,
        drag_rotations,
        sun_positions_eci_km,
        moon_positions_eci_km,
        atmosphere_inputs,
        scalar_parameters,
        density_mode,
        constant_density_kg_m3,
        shadow_model,
        c_nm,
        s_nm,
        legendre_diag_scale,
        legendre_subdiag_scale,
        legendre_recur_a,
        legendre_recur_b,
        legendre_recur_c,
        n_max,
        m_max,
        pt1,
        ps1,
        pd1,
        pdl1,
        ptm1,
        pdm1,
        ptl1,
        pma1,
        zn1,
        alpha,
    )
    if not valid:
        return x_eci.copy(), False
    result = np.empty(6, dtype=np.float64)
    step_scale = dt_s / 6.0
    for index in range(6):
        weighted = k1[index] + 2.0 * k2[index]
        weighted += 2.0 * k3[index]
        weighted += k4[index]
        result[index] = x_eci[index] + step_scale * weighted
    return result, True
