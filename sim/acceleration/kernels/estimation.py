from __future__ import annotations

import numpy as np

from sim.acceleration.kernels.attitude import normalize_quaternion_kernel
from sim.acceleration.kernels.orbit import rk4_zonal_step_state
from sim.acceleration.optional import njit_or_identity


@njit_or_identity(cache=True)
def propagate_two_body_rk4_kernel(x_eci: np.ndarray, dt_s: float, mu_km3_s2: float) -> np.ndarray:
    zero = np.zeros(3, dtype=np.float64)
    return rk4_zonal_step_state(x_eci, dt_s, zero, mu_km3_s2, False, False, False)


@njit_or_identity(cache=True)
def orbit_ekf_numerical_jacobian_kernel(
    x_eci: np.ndarray,
    base_eval: np.ndarray,
    dt_s: float,
    mu_km3_s2: float,
    eps: float = 1e-6,
) -> np.ndarray:
    j = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        xp = x_eci.copy()
        xp[i] += eps
        yp = propagate_two_body_rk4_kernel(xp, dt_s, mu_km3_s2)
        for row in range(6):
            j[row, i] = (yp[row] - base_eval[row]) / eps
    return j


@njit_or_identity(cache=True)
def attitude_ekf_propagate_state_kernel(x: np.ndarray, dt_s: float, inertia_kg_m2: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_kernel(x[:4])
    w = x[4:7]
    q_dot = np.empty(4, dtype=np.float64)
    q_dot[0] = -0.5 * (w[0] * q[1] + w[1] * q[2] + w[2] * q[3])
    q_dot[1] = 0.5 * (w[0] * q[0] + w[2] * q[2] - w[1] * q[3])
    q_dot[2] = 0.5 * (w[1] * q[0] - w[2] * q[1] + w[0] * q[3])
    q_dot[3] = 0.5 * (w[2] * q[0] + w[1] * q[1] - w[0] * q[2])
    q_next_raw = q + dt_s * q_dot
    q_next = normalize_quaternion_kernel(q_next_raw)

    iw = inertia_kg_m2 @ w
    coriolis = np.empty(3, dtype=np.float64)
    coriolis[0] = w[1] * iw[2] - w[2] * iw[1]
    coriolis[1] = w[2] * iw[0] - w[0] * iw[2]
    coriolis[2] = w[0] * iw[1] - w[1] * iw[0]
    w_dot = np.linalg.solve(inertia_kg_m2, -coriolis)
    w_next = w + dt_s * w_dot

    out = np.empty(7, dtype=np.float64)
    out[:4] = q_next
    out[4:7] = w_next
    return out


@njit_or_identity(cache=True)
def attitude_ekf_numerical_jacobian_kernel(
    x: np.ndarray,
    base_eval: np.ndarray,
    dt_s: float,
    inertia_kg_m2: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    j = np.zeros((7, 7), dtype=np.float64)
    for i in range(7):
        xp = x.copy()
        xp[i] += eps
        yp = attitude_ekf_propagate_state_kernel(xp, dt_s, inertia_kg_m2)
        for row in range(7):
            j[row, i] = (yp[row] - base_eval[row]) / eps
    return j
