from __future__ import annotations

import numpy as np

from sim.acceleration.optional import njit_or_identity

MAX_ABS_RATE_RAD_S = 1e6
MAX_ABS_TORQUE_NM = 1e12

STAT_NON_FINITE_INPUT = 0
STAT_RATE_CLAMP = 1
STAT_TORQUE_CLAMP = 2
STAT_NON_FINITE_CORIOLIS = 3
STAT_SINGULAR_INERTIA = 4
STAT_NON_FINITE_OUTPUT = 5

DISTURBANCE_GRAVITY_GRADIENT = 0
DISTURBANCE_MAGNETIC = 1
DISTURBANCE_DRAG = 2
DISTURBANCE_SRP = 3

FACET_MODE_NONE = 0
FACET_MODE_SCALAR = 1
FACET_MODE_FACETS = 2


@njit_or_identity(cache=True)
def normalize_quaternion_kernel(q: np.ndarray) -> np.ndarray:
    out = np.empty(4, dtype=np.float64)
    if q.size != 4:
        out[0] = 1.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        return out
    n2 = 0.0
    for i in range(4):
        if not np.isfinite(q[i]):
            out[0] = 1.0
            out[1] = 0.0
            out[2] = 0.0
            out[3] = 0.0
            return out
        n2 += q[i] * q[i]
    if n2 <= 0.0 or not np.isfinite(n2):
        out[0] = 1.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        return out
    n = np.sqrt(n2)
    if n <= 0.0 or not np.isfinite(n):
        out[0] = 1.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        return out
    for i in range(4):
        out[i] = q[i] / n
    return out


@njit_or_identity(cache=True)
def quaternion_multiply_kernel(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a = normalize_quaternion_kernel(q1)
    b = normalize_quaternion_kernel(q2)
    out = np.empty(4, dtype=np.float64)
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return out


@njit_or_identity(cache=True)
def quaternion_delta_from_body_rate_kernel(omega_body_rad_s: np.ndarray, dt_s: float) -> np.ndarray:
    out = np.empty(4, dtype=np.float64)
    if not np.isfinite(dt_s):
        out[0] = 1.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        return out
    w2 = 0.0
    for i in range(3):
        if not np.isfinite(omega_body_rad_s[i]):
            out[0] = 1.0
            out[1] = 0.0
            out[2] = 0.0
            out[3] = 0.0
            return out
        w2 += omega_body_rad_s[i] * omega_body_rad_s[i]
    w_norm = np.sqrt(w2)
    if w_norm <= 1e-15 or dt_s == 0.0:
        out[0] = 1.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        return out
    half_theta = 0.5 * w_norm * dt_s
    if not np.isfinite(half_theta):
        out[0] = 1.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        return out
    half_theta = np.remainder(half_theta, 2.0 * np.pi)
    s = np.sin(half_theta)
    c = np.cos(half_theta)
    out[0] = c
    out[1] = omega_body_rad_s[0] / w_norm * s
    out[2] = omega_body_rad_s[1] / w_norm * s
    out[3] = omega_body_rad_s[2] / w_norm * s
    return normalize_quaternion_kernel(out)


@njit_or_identity(cache=True)
def propagate_attitude_exponential_map_kernel(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats = np.zeros(6, dtype=np.int64)
    quaternion_bad = quat_bn.size != 4
    quaternion_norm_sq = 0.0
    if not quaternion_bad:
        for i in range(4):
            if not np.isfinite(quat_bn[i]):
                quaternion_bad = True
            quaternion_norm_sq += quat_bn[i] * quat_bn[i]
        if quaternion_norm_sq <= 0.0 or not np.isfinite(quaternion_norm_sq):
            quaternion_bad = True
    if quaternion_bad:
        stats[STAT_NON_FINITE_INPUT] += 1
    q = normalize_quaternion_kernel(quat_bn)
    w = np.empty(3, dtype=np.float64)
    tau = np.empty(3, dtype=np.float64)
    inertia = np.empty((3, 3), dtype=np.float64)

    input_bad = False
    for i in range(3):
        if not np.isfinite(omega_body_rad_s[i]):
            input_bad = True
            stats[STAT_RATE_CLAMP] += 1
        if not np.isfinite(torque_body_nm[i]):
            input_bad = True
            stats[STAT_TORQUE_CLAMP] += 1
        w[i] = _nan_to_num_clamped(omega_body_rad_s[i], MAX_ABS_RATE_RAD_S)
        tau[i] = _nan_to_num_clamped(torque_body_nm[i], MAX_ABS_TORQUE_NM)
    for i in range(3):
        for j in range(3):
            if not np.isfinite(inertia_kg_m2[i, j]):
                input_bad = True
            inertia[i, j] = inertia_kg_m2[i, j]
    if input_bad:
        stats[STAT_NON_FINITE_INPUT] += 1

    for i in range(3):
        w_clipped = min(max(w[i], -MAX_ABS_RATE_RAD_S), MAX_ABS_RATE_RAD_S)
        tau_clipped = min(max(tau[i], -MAX_ABS_TORQUE_NM), MAX_ABS_TORQUE_NM)
        if w_clipped != w[i]:
            stats[STAT_RATE_CLAMP] += 1
        if tau_clipped != tau[i]:
            stats[STAT_TORQUE_CLAMP] += 1
        w[i] = w_clipped
        tau[i] = tau_clipped

    iw = inertia @ w
    coriolis = np.empty(3, dtype=np.float64)
    coriolis[0] = w[1] * iw[2] - w[2] * iw[1]
    coriolis[1] = w[2] * iw[0] - w[0] * iw[2]
    coriolis[2] = w[0] * iw[1] - w[1] * iw[0]
    if not (np.isfinite(coriolis[0]) and np.isfinite(coriolis[1]) and np.isfinite(coriolis[2])):
        stats[STAT_NON_FINITE_CORIOLIS] += 1
    rhs = np.empty(3, dtype=np.float64)
    for i in range(3):
        rhs[i] = tau[i] - _nan_to_num_clamped(coriolis[i], MAX_ABS_TORQUE_NM)

    omega_dot = np.zeros(3, dtype=np.float64)
    det = _det3(inertia)
    if not np.isfinite(det) or det == 0.0:
        stats[STAT_SINGULAR_INERTIA] += 1
    else:
        omega_dot = np.linalg.solve(inertia, rhs)
    if not (np.isfinite(omega_dot[0]) and np.isfinite(omega_dot[1]) and np.isfinite(omega_dot[2])):
        stats[STAT_NON_FINITE_OUTPUT] += 1
    for i in range(3):
        omega_dot[i] = _nan_to_num_clamped(omega_dot[i], MAX_ABS_RATE_RAD_S)

    dt = max(dt_s, 0.0)
    omega_next = np.empty(3, dtype=np.float64)
    for i in range(3):
        omega_next[i] = omega_body_rad_s[i] + dt * omega_dot[i]
    if not (np.isfinite(omega_next[0]) and np.isfinite(omega_next[1]) and np.isfinite(omega_next[2])):
        stats[STAT_NON_FINITE_OUTPUT] += 1
    for i in range(3):
        omega_next[i] = min(max(_nan_to_num_clamped(omega_next[i], MAX_ABS_RATE_RAD_S), -MAX_ABS_RATE_RAD_S), MAX_ABS_RATE_RAD_S)

    omega_mid = np.empty(3, dtype=np.float64)
    for i in range(3):
        omega_mid[i] = omega_body_rad_s[i] + 0.5 * dt * omega_dot[i]
    if not (np.isfinite(omega_mid[0]) and np.isfinite(omega_mid[1]) and np.isfinite(omega_mid[2])):
        stats[STAT_NON_FINITE_OUTPUT] += 1
    for i in range(3):
        omega_mid[i] = min(max(_nan_to_num_clamped(omega_mid[i], MAX_ABS_RATE_RAD_S), -MAX_ABS_RATE_RAD_S), MAX_ABS_RATE_RAD_S)

    dq = quaternion_delta_from_body_rate_kernel(omega_mid, dt)
    q_next = normalize_quaternion_kernel(quaternion_multiply_kernel(q, dq))
    if not (np.isfinite(q_next[0]) and np.isfinite(q_next[1]) and np.isfinite(q_next[2]) and np.isfinite(q_next[3])):
        stats[STAT_NON_FINITE_OUTPUT] += 1
    return q_next, omega_next, stats


@njit_or_identity(cache=True)
def propagate_attitude_builtin_disturbances_kernel(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    command_torque_body_nm: np.ndarray,
    substeps_s: np.ndarray,
    position_eci_km: np.ndarray,
    mu_km3_s2: float,
    enabled: np.ndarray,
    magnetic_dipole_body_a_m2: np.ndarray,
    magnetic_field_eci_t: np.ndarray,
    magnetic_field_provided: bool,
    density_kg_m3: float,
    drag_v_rel_eci_m_s: np.ndarray,
    drag_v_rel_norm_m_s: float,
    drag_mode: int,
    drag_area_m2: float,
    drag_cd: float,
    drag_cp_offset_body_m: np.ndarray,
    drag_facet_normals_body: np.ndarray,
    drag_facet_areas_m2: np.ndarray,
    drag_facet_cd: np.ndarray,
    drag_facet_cp_offsets_body_m: np.ndarray,
    sun_dir_eci_unit: np.ndarray,
    srp_pressure_scaled_n_m2: float,
    srp_mode: int,
    srp_area_m2: float,
    srp_cp_offset_body_m: np.ndarray,
    srp_facet_normals_body: np.ndarray,
    srp_facet_areas_m2: np.ndarray,
    srp_facet_cp_offsets_body_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate built-in attitude disturbances across one outer dynamics step."""

    q = quat_bn.copy()
    omega = omega_body_rad_s.copy()
    aggregate_stats = np.zeros(6, dtype=np.int64)
    for index in range(substeps_s.size):
        disturbance_torque = builtin_disturbance_torque_kernel(
            q,
            position_eci_km,
            inertia_kg_m2,
            mu_km3_s2,
            enabled,
            magnetic_dipole_body_a_m2,
            magnetic_field_eci_t,
            magnetic_field_provided,
            density_kg_m3,
            drag_v_rel_eci_m_s,
            drag_v_rel_norm_m_s,
            drag_mode,
            drag_area_m2,
            drag_cd,
            drag_cp_offset_body_m,
            drag_facet_normals_body,
            drag_facet_areas_m2,
            drag_facet_cd,
            drag_facet_cp_offsets_body_m,
            sun_dir_eci_unit,
            srp_pressure_scaled_n_m2,
            srp_mode,
            srp_area_m2,
            srp_cp_offset_body_m,
            srp_facet_normals_body,
            srp_facet_areas_m2,
            srp_facet_cp_offsets_body_m,
        )
        total_torque = np.empty(3, dtype=np.float64)
        for axis in range(3):
            total_torque[axis] = command_torque_body_nm[axis] + disturbance_torque[axis]
        q, omega, stats = propagate_attitude_exponential_map_kernel(
            q,
            omega,
            inertia_kg_m2,
            total_torque,
            substeps_s[index],
        )
        for stat_index in range(6):
            aggregate_stats[stat_index] += stats[stat_index]
    return q, omega, aggregate_stats


@njit_or_identity(cache=True)
def builtin_disturbance_torque_kernel(
    quat_bn: np.ndarray,
    position_eci_km: np.ndarray,
    inertia_kg_m2: np.ndarray,
    mu_km3_s2: float,
    enabled: np.ndarray,
    magnetic_dipole_body_a_m2: np.ndarray,
    magnetic_field_eci_t: np.ndarray,
    magnetic_field_provided: bool,
    density_kg_m3: float,
    drag_v_rel_eci_m_s: np.ndarray,
    drag_v_rel_norm_m_s: float,
    drag_mode: int,
    drag_area_m2: float,
    drag_cd: float,
    drag_cp_offset_body_m: np.ndarray,
    drag_facet_normals_body: np.ndarray,
    drag_facet_areas_m2: np.ndarray,
    drag_facet_cd: np.ndarray,
    drag_facet_cp_offsets_body_m: np.ndarray,
    sun_dir_eci_unit: np.ndarray,
    srp_pressure_scaled_n_m2: float,
    srp_mode: int,
    srp_area_m2: float,
    srp_cp_offset_body_m: np.ndarray,
    srp_facet_normals_body: np.ndarray,
    srp_facet_areas_m2: np.ndarray,
    srp_facet_cp_offsets_body_m: np.ndarray,
) -> np.ndarray:
    """Evaluate the numeric built-in disturbance plan in public model order."""

    torque = np.zeros(3, dtype=np.float64)
    c_bn = _quaternion_to_dcm_bn_kernel(quat_bn)
    if enabled[DISTURBANCE_GRAVITY_GRADIENT] != 0:
        component = _gravity_gradient_torque_kernel(position_eci_km, c_bn, inertia_kg_m2, mu_km3_s2)
        _add3_in_place(torque, component)
    if enabled[DISTURBANCE_MAGNETIC] != 0:
        component = _magnetic_torque_kernel(
            position_eci_km,
            c_bn,
            magnetic_dipole_body_a_m2,
            magnetic_field_eci_t,
            magnetic_field_provided,
        )
        _add3_in_place(torque, component)
    if enabled[DISTURBANCE_DRAG] != 0:
        component = _drag_torque_kernel(
            c_bn,
            density_kg_m3,
            drag_v_rel_eci_m_s,
            drag_v_rel_norm_m_s,
            drag_mode,
            drag_area_m2,
            drag_cd,
            drag_cp_offset_body_m,
            drag_facet_normals_body,
            drag_facet_areas_m2,
            drag_facet_cd,
            drag_facet_cp_offsets_body_m,
        )
        _add3_in_place(torque, component)
    if enabled[DISTURBANCE_SRP] != 0:
        component = _srp_torque_kernel(
            c_bn,
            sun_dir_eci_unit,
            srp_pressure_scaled_n_m2,
            srp_mode,
            srp_area_m2,
            srp_cp_offset_body_m,
            srp_facet_normals_body,
            srp_facet_areas_m2,
            srp_facet_cp_offsets_body_m,
        )
        _add3_in_place(torque, component)
    return torque


@njit_or_identity(cache=True)
def _quaternion_to_dcm_bn_kernel(quat_bn: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_kernel(quat_bn)
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    out = np.empty((3, 3), dtype=np.float64)
    out[0, 0] = 1.0 - 2.0 * (q2**2 + q3**2)
    out[0, 1] = 2.0 * (q1 * q2 + q0 * q3)
    out[0, 2] = 2.0 * (q1 * q3 - q0 * q2)
    out[1, 0] = 2.0 * (q1 * q2 - q0 * q3)
    out[1, 1] = 1.0 - 2.0 * (q1**2 + q3**2)
    out[1, 2] = 2.0 * (q2 * q3 + q0 * q1)
    out[2, 0] = 2.0 * (q1 * q3 + q0 * q2)
    out[2, 1] = 2.0 * (q2 * q3 - q0 * q1)
    out[2, 2] = 1.0 - 2.0 * (q1**2 + q2**2)
    return out


@njit_or_identity(cache=True)
def _gravity_gradient_torque_kernel(
    position_eci_km: np.ndarray,
    c_bn: np.ndarray,
    inertia_kg_m2: np.ndarray,
    mu_km3_s2: float,
) -> np.ndarray:
    r_i_m = np.empty(3, dtype=np.float64)
    for axis in range(3):
        r_i_m[axis] = position_eci_km[axis] * 1.0e3
    r_norm_m = _norm3_kernel(r_i_m)
    if r_norm_m == 0.0:
        return np.zeros(3, dtype=np.float64)
    r_hat_i = np.empty(3, dtype=np.float64)
    for axis in range(3):
        r_hat_i[axis] = r_i_m[axis] / r_norm_m
    r_hat_b = c_bn @ r_hat_i
    inertia_r_hat = inertia_kg_m2 @ r_hat_b
    cross = _cross3_kernel(r_hat_b, inertia_r_hat)
    scale = 3.0 * (mu_km3_s2 * 1.0e9) / (r_norm_m**3)
    for axis in range(3):
        cross[axis] *= scale
    return cross


@njit_or_identity(cache=True)
def _magnetic_torque_kernel(
    position_eci_km: np.ndarray,
    c_bn: np.ndarray,
    magnetic_dipole_body_a_m2: np.ndarray,
    magnetic_field_eci_t: np.ndarray,
    magnetic_field_provided: bool,
) -> np.ndarray:
    if magnetic_field_provided:
        b_eci = magnetic_field_eci_t.copy()
    else:
        r_i_m = np.empty(3, dtype=np.float64)
        for axis in range(3):
            r_i_m[axis] = position_eci_km[axis] * 1.0e3
        r_norm_m = _norm3_kernel(r_i_m)
        if r_norm_m == 0.0:
            return np.zeros(3, dtype=np.float64)
        r_hat = np.empty(3, dtype=np.float64)
        for axis in range(3):
            r_hat[axis] = r_i_m[axis] / r_norm_m
        dipole_dot_r = 7.94e15 * r_hat[2]
        denominator = r_norm_m**3
        b_eci = np.empty(3, dtype=np.float64)
        b_eci[0] = 3.0 * r_hat[0] * dipole_dot_r / denominator
        b_eci[1] = 3.0 * r_hat[1] * dipole_dot_r / denominator
        b_eci[2] = (3.0 * r_hat[2] * dipole_dot_r - 7.94e15) / denominator
    b_body = c_bn @ b_eci
    return _cross3_kernel(magnetic_dipole_body_a_m2, b_body)


@njit_or_identity(cache=True)
def _drag_torque_kernel(
    c_bn: np.ndarray,
    density_kg_m3: float,
    v_rel_eci_m_s: np.ndarray,
    v_norm_m_s: float,
    mode: int,
    area_m2: float,
    drag_cd: float,
    cp_offset_body_m: np.ndarray,
    facet_normals_body: np.ndarray,
    facet_areas_m2: np.ndarray,
    facet_cd: np.ndarray,
    facet_cp_offsets_body_m: np.ndarray,
) -> np.ndarray:
    if v_norm_m_s == 0.0 or density_kg_m3 <= 0.0 or mode == FACET_MODE_NONE:
        return np.zeros(3, dtype=np.float64)
    v_rel_body = c_bn @ v_rel_eci_m_s
    if mode == FACET_MODE_SCALAR:
        force_mag = 0.5 * density_kg_m3 * (v_norm_m_s**2) * drag_cd * area_m2
        force_body = np.empty(3, dtype=np.float64)
        for axis in range(3):
            force_body[axis] = -force_mag * (v_rel_body[axis] / v_norm_m_s)
        return _cross3_kernel(cp_offset_body_m, force_body)

    torque = np.zeros(3, dtype=np.float64)
    v_hat_body = np.empty(3, dtype=np.float64)
    for axis in range(3):
        v_hat_body[axis] = v_rel_body[axis] / v_norm_m_s
    for facet_index in range(facet_areas_m2.size):
        normal = facet_normals_body[facet_index]
        normal_norm = _norm3_kernel(normal)
        if normal_norm <= 0.0:
            continue
        projected_cosine = 0.0
        for axis in range(3):
            projected_cosine += (normal[axis] / normal_norm) * v_hat_body[axis]
        projected_area = facet_areas_m2[facet_index] * max(0.0, projected_cosine)
        if projected_area <= 0.0:
            continue
        scale = -0.5 * density_kg_m3 * projected_area * facet_cd[facet_index] * (v_norm_m_s**2)
        force_body = np.empty(3, dtype=np.float64)
        for axis in range(3):
            force_body[axis] = scale * v_hat_body[axis]
        component = _cross3_kernel(facet_cp_offsets_body_m[facet_index], force_body)
        _add3_in_place(torque, component)
    return torque


@njit_or_identity(cache=True)
def _srp_torque_kernel(
    c_bn: np.ndarray,
    sun_dir_eci_unit: np.ndarray,
    pressure_scaled_n_m2: float,
    mode: int,
    area_m2: float,
    cp_offset_body_m: np.ndarray,
    facet_normals_body: np.ndarray,
    facet_areas_m2: np.ndarray,
    facet_cp_offsets_body_m: np.ndarray,
) -> np.ndarray:
    if pressure_scaled_n_m2 <= 0.0 or mode == FACET_MODE_NONE:
        return np.zeros(3, dtype=np.float64)
    sun_dir_body = c_bn @ sun_dir_eci_unit
    if mode == FACET_MODE_SCALAR:
        force_body = np.empty(3, dtype=np.float64)
        force_mag = pressure_scaled_n_m2 * area_m2
        for axis in range(3):
            force_body[axis] = -force_mag * sun_dir_body[axis]
        return _cross3_kernel(cp_offset_body_m, force_body)

    torque = np.zeros(3, dtype=np.float64)
    for facet_index in range(facet_areas_m2.size):
        normal = facet_normals_body[facet_index]
        normal_norm = _norm3_kernel(normal)
        if normal_norm <= 0.0:
            continue
        illumination = 0.0
        for axis in range(3):
            illumination += (normal[axis] / normal_norm) * sun_dir_body[axis]
        illumination = max(0.0, illumination)
        if illumination <= 0.0:
            continue
        force_body = np.empty(3, dtype=np.float64)
        scale = -pressure_scaled_n_m2 * facet_areas_m2[facet_index] * illumination
        for axis in range(3):
            force_body[axis] = scale * sun_dir_body[axis]
        component = _cross3_kernel(facet_cp_offsets_body_m[facet_index], force_body)
        _add3_in_place(torque, component)
    return torque


@njit_or_identity(cache=True)
def _cross3_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(3, dtype=np.float64)
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]
    return out


@njit_or_identity(cache=True)
def _norm3_kernel(vector: np.ndarray) -> float:
    return np.sqrt(np.dot(vector, vector))


@njit_or_identity(cache=True)
def _add3_in_place(target: np.ndarray, value: np.ndarray) -> None:
    target[0] += value[0]
    target[1] += value[1]
    target[2] += value[2]


@njit_or_identity(cache=True)
def _nan_to_num_clamped(value: float, limit: float) -> float:
    if np.isnan(value):
        return 0.0
    if value == np.inf:
        return limit
    if value == -np.inf:
        return -limit
    return value


@njit_or_identity(cache=True)
def _det3(matrix: np.ndarray) -> float:
    return (
        matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )
