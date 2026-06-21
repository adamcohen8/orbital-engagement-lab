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
