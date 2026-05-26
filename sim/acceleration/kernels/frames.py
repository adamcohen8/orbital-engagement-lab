from __future__ import annotations

import numpy as np

from sim.acceleration.optional import njit_or_identity


@njit_or_identity(cache=True)
def ric_dcm_ir_from_rv_kernel(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    r_norm = max(np.sqrt(np.dot(r_eci_km, r_eci_km)), 1e-12)
    r_hat = r_eci_km / r_norm
    h = np.empty(3, dtype=np.float64)
    h[0] = r_eci_km[1] * v_eci_km_s[2] - r_eci_km[2] * v_eci_km_s[1]
    h[1] = r_eci_km[2] * v_eci_km_s[0] - r_eci_km[0] * v_eci_km_s[2]
    h[2] = r_eci_km[0] * v_eci_km_s[1] - r_eci_km[1] * v_eci_km_s[0]
    h_norm = max(np.sqrt(np.dot(h, h)), 1e-12)
    c_hat = h / h_norm
    i_hat = np.empty(3, dtype=np.float64)
    i_hat[0] = c_hat[1] * r_hat[2] - c_hat[2] * r_hat[1]
    i_hat[1] = c_hat[2] * r_hat[0] - c_hat[0] * r_hat[2]
    i_hat[2] = c_hat[0] * r_hat[1] - c_hat[1] * r_hat[0]
    i_norm = max(np.sqrt(np.dot(i_hat, i_hat)), 1e-12)
    i_hat = i_hat / i_norm
    out = np.empty((3, 3), dtype=np.float64)
    out[:, 0] = r_hat
    out[:, 1] = i_hat
    out[:, 2] = c_hat
    return out


@njit_or_identity(cache=True)
def ric_angular_rate_eci_from_rv_kernel(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    r2 = np.dot(r_eci_km, r_eci_km)
    out = np.zeros(3, dtype=np.float64)
    if r2 <= 1e-12:
        return out
    out[0] = r_eci_km[1] * v_eci_km_s[2] - r_eci_km[2] * v_eci_km_s[1]
    out[1] = r_eci_km[2] * v_eci_km_s[0] - r_eci_km[0] * v_eci_km_s[2]
    out[2] = r_eci_km[0] * v_eci_km_s[1] - r_eci_km[1] * v_eci_km_s[0]
    return out / r2


@njit_or_identity(cache=True)
def ric_rect_state_to_eci_kernel(
    x_rel_ric_rect: np.ndarray,
    r_chief_eci_km: np.ndarray,
    v_chief_eci_km_s: np.ndarray,
) -> np.ndarray:
    c_ir = ric_dcm_ir_from_rv_kernel(r_chief_eci_km, v_chief_eci_km_s)
    omega = ric_angular_rate_eci_from_rv_kernel(r_chief_eci_km, v_chief_eci_km_s)
    dr_eci = c_ir @ x_rel_ric_rect[:3]
    omega_cross_dr = np.empty(3, dtype=np.float64)
    omega_cross_dr[0] = omega[1] * dr_eci[2] - omega[2] * dr_eci[1]
    omega_cross_dr[1] = omega[2] * dr_eci[0] - omega[0] * dr_eci[2]
    omega_cross_dr[2] = omega[0] * dr_eci[1] - omega[1] * dr_eci[0]
    dv_eci = c_ir @ x_rel_ric_rect[3:] + omega_cross_dr
    out = np.empty(6, dtype=np.float64)
    out[:3] = r_chief_eci_km + dr_eci
    out[3:] = v_chief_eci_km_s + dv_eci
    return out


@njit_or_identity(cache=True)
def eci_relative_to_ric_rect_kernel(x_dep_eci: np.ndarray, x_chief_eci: np.ndarray) -> np.ndarray:
    r_chief = x_chief_eci[:3]
    v_chief = x_chief_eci[3:]
    c_ir = ric_dcm_ir_from_rv_kernel(r_chief, v_chief)
    omega = ric_angular_rate_eci_from_rv_kernel(r_chief, v_chief)
    dr_eci = x_dep_eci[:3] - r_chief
    dv_eci = x_dep_eci[3:] - v_chief
    dr_ric = c_ir.T @ dr_eci
    omega_cross_dr = np.empty(3, dtype=np.float64)
    omega_cross_dr[0] = omega[1] * dr_eci[2] - omega[2] * dr_eci[1]
    omega_cross_dr[1] = omega[2] * dr_eci[0] - omega[0] * dr_eci[2]
    omega_cross_dr[2] = omega[0] * dr_eci[1] - omega[1] * dr_eci[0]
    dv_ric = c_ir.T @ (dv_eci - omega_cross_dr)
    out = np.empty(6, dtype=np.float64)
    out[:3] = dr_ric
    out[3:] = dv_ric
    return out


@njit_or_identity(cache=True)
def ric_curv_to_rect_kernel(x_ric_curv: np.ndarray, r0_km: float, eps: float = 1e-12) -> np.ndarray:
    x_r_curv = x_ric_curv[0]
    x_i_curv = x_ric_curv[1]
    x_c_curv = x_ric_curv[2]
    x_r_curv_dot = x_ric_curv[3]
    x_i_curv_dot = x_ric_curv[4]
    x_c_curv_dot = x_ric_curv[5]
    r0 = max(r0_km, eps)
    r = max(r0 + x_r_curv, eps)
    theta_i = x_i_curv / r0
    theta_c = x_c_curv / r0
    c_i = np.cos(theta_i)
    s_i = np.sin(theta_i)
    c_c = np.cos(theta_c)
    s_c = np.sin(theta_c)
    x = r * c_c * c_i
    y = r * c_c * s_i
    z = r * s_c
    r_dot = x_r_curv_dot
    theta_i_dot = x_i_curv_dot / r0
    theta_c_dot = x_c_curv_dot / r0
    out = np.empty(6, dtype=np.float64)
    out[0] = x - r0
    out[1] = y
    out[2] = z
    out[3] = r_dot * c_c * c_i - r * s_c * theta_c_dot * c_i - r * c_c * s_i * theta_i_dot
    out[4] = r_dot * c_c * s_i - r * s_c * theta_c_dot * s_i + r * c_c * c_i * theta_i_dot
    out[5] = r_dot * s_c + r * c_c * theta_c_dot
    return out


@njit_or_identity(cache=True)
def ric_rect_to_curv_kernel(x_ric_rect: np.ndarray, r0_km: float, eps: float = 1e-12) -> np.ndarray:
    x_r = x_ric_rect[0]
    x_i = x_ric_rect[1]
    x_c = x_ric_rect[2]
    x_rdot = x_ric_rect[3]
    x_idot = x_ric_rect[4]
    x_cdot = x_ric_rect[5]
    r0 = max(r0_km, eps)
    x = r0 + x_r
    y = x_i
    z = x_c
    r = max(np.sqrt(x * x + y * y + z * z), eps)
    p2 = x * x + y * y
    p = np.sqrt(max(p2, eps))
    theta_i = np.arctan2(y, x)
    theta_c = np.arctan2(z, p)
    r_dot = (x * x_rdot + y * x_idot + z * x_cdot) / r
    theta_i_dot = (x * x_idot - y * x_rdot) / max(p2, eps)
    p_dot = (x * x_rdot + y * x_idot) / p
    theta_c_dot = (p * x_cdot - z * p_dot) / (r * r)
    out = np.empty(6, dtype=np.float64)
    out[0] = r - r0
    out[1] = r0 * theta_i
    out[2] = r0 * theta_c
    out[3] = r_dot
    out[4] = r0 * theta_i_dot
    out[5] = r0 * theta_c_dot
    return out
