from __future__ import annotations

from typing import Literal

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.utils.quaternion import quaternion_to_dcm_bn

ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]
OrbitalElementSeriesCache = dict[
    str,
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
]
RICSummaryFrame = Literal["rectangular", "curvilinear"]

ORBITAL_ELEMENT_SPECS: dict[str, tuple[str, str]] = {
    "a": ("Semi-Major Axis", "km"),
    "ecc": ("Eccentricity", ""),
    "inc": ("Inclination", "deg"),
    "raan": ("RAAN", "deg"),
    "argp": ("Argument of Perigee", "deg"),
    "true_anomaly": ("True Anomaly", "deg"),
}


def _cumulative_delta_v_m_s(t_s: np.ndarray, accel_km_s2: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(np.nan_to_num(accel_km_s2, nan=0.0), axis=1)
    if mag.size == 0:
        return mag
    dt = np.diff(t_s[: mag.size], prepend=t_s[0] if t_s.size else 0.0)
    dt = np.clip(dt, 0.0, None)
    return np.cumsum(mag * dt) * 1000.0


def _quat_error_angle_deg(q_des: np.ndarray, q_cur: np.ndarray) -> float:
    qd = np.array(q_des, dtype=float).reshape(-1)
    qc = np.array(q_cur, dtype=float).reshape(-1)
    if qd.size != 4 or qc.size != 4:
        return float("nan")
    nd = float(np.linalg.norm(qd))
    nc = float(np.linalg.norm(qc))
    if nd <= 0.0 or nc <= 0.0:
        return float("nan")
    qd = qd / nd
    qc = qc / nc
    dot = abs(float(np.dot(qd, qc)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _quat_error_series_deg(
    *,
    truth_hist: np.ndarray,
    desired_attitude_hist: np.ndarray | None,
    n_s: int,
) -> np.ndarray:
    err_deg = np.full(n_s, np.nan, dtype=float)
    if desired_attitude_hist is None or desired_attitude_hist.size == 0:
        return err_deg
    qd = np.array(desired_attitude_hist[:n_s, :], dtype=float)
    qc = np.array(truth_hist[:n_s, 6:10], dtype=float)
    for k in range(1, n_s):
        if not np.all(np.isfinite(qd[k, :])) and np.all(np.isfinite(qd[k - 1, :])):
            qd[k, :] = qd[k - 1, :]
    for k in range(n_s):
        if not (np.all(np.isfinite(qd[k, :])) and np.all(np.isfinite(qc[k, :]))):
            continue
        err_deg[k] = _quat_error_angle_deg(qd[k, :], qc[k, :])
    return err_deg


def _thrust_alignment_error_deg_series(
    *,
    truth_hist: np.ndarray,
    thrust_hist: np.ndarray,
    thrust_axis_body: np.ndarray,
    n_s: int,
) -> np.ndarray:
    axis_body = np.array(thrust_axis_body, dtype=float).reshape(-1)
    if axis_body.size != 3:
        axis_body = np.array([1.0, 0.0, 0.0], dtype=float)
    norm_axis = float(np.linalg.norm(axis_body))
    if norm_axis <= 0.0:
        axis_body = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis_body = axis_body / norm_axis
    err_deg = np.full(n_s, np.nan, dtype=float)
    for k in range(n_s):
        a_cmd = np.array(thrust_hist[k, :3], dtype=float)
        a_norm = float(np.linalg.norm(a_cmd))
        if a_norm <= 1e-15 or not np.all(np.isfinite(a_cmd)):
            continue
        q_bn = np.array(truth_hist[k, 6:10], dtype=float)
        if not np.all(np.isfinite(q_bn)):
            continue
        c_bn = quaternion_to_dcm_bn(q_bn)
        thrust_axis_eci = c_bn.T @ axis_body
        burn_dir_eci = -a_cmd / a_norm
        cosang = float(np.clip(np.dot(thrust_axis_eci, burn_dir_eci), -1.0, 1.0))
        err_deg[k] = float(np.degrees(np.arccos(cosang)))
    return err_deg


def _safe_angle_deg(cos_value: float, *, flip: bool = False) -> float:
    angle = float(np.degrees(np.arccos(float(np.clip(cos_value, -1.0, 1.0)))))
    return 360.0 - angle if flip and angle > 0.0 else angle


def _classical_orbital_elements_series(
    truth_hist: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> dict[str, np.ndarray]:
    arr = np.array(truth_hist, dtype=float)
    n = arr.shape[0] if arr.ndim == 2 else 0
    out = {key: np.full(n, np.nan, dtype=float) for key in ORBITAL_ELEMENT_SPECS}
    if n == 0 or arr.shape[1] < 6 or not np.isfinite(mu_km3_s2) or mu_km3_s2 <= 0.0:
        return out

    h_tol = 1e-10
    n_tol = 1e-10
    e_tol = 1e-8
    k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    for idx in range(n):
        r_vec = np.array(arr[idx, 0:3], dtype=float)
        v_vec = np.array(arr[idx, 3:6], dtype=float)
        if not (np.all(np.isfinite(r_vec)) and np.all(np.isfinite(v_vec))):
            continue
        r = float(np.linalg.norm(r_vec))
        if r <= 0.0:
            continue
        h_vec = np.cross(r_vec, v_vec)
        h = float(np.linalg.norm(h_vec))
        if h <= h_tol:
            continue
        v2 = float(np.dot(v_vec, v_vec))
        eps = 0.5 * v2 - mu_km3_s2 / r
        if abs(eps) > 1e-14:
            out["a"][idx] = float(-mu_km3_s2 / (2.0 * eps))
        e_vec = np.cross(v_vec, h_vec) / mu_km3_s2 - r_vec / r
        ecc = float(np.linalg.norm(e_vec))
        out["ecc"][idx] = ecc
        out["inc"][idx] = _safe_angle_deg(h_vec[2] / h)

        n_vec = np.cross(k_hat, h_vec)
        n_norm = float(np.linalg.norm(n_vec))
        if n_norm > n_tol:
            out["raan"][idx] = _safe_angle_deg(n_vec[0] / n_norm, flip=n_vec[1] < 0.0)
        if n_norm > n_tol and ecc > e_tol:
            out["argp"][idx] = _safe_angle_deg(
                float(np.dot(n_vec, e_vec)) / (n_norm * ecc),
                flip=e_vec[2] < 0.0,
            )
        if ecc > e_tol:
            out["true_anomaly"][idx] = _safe_angle_deg(
                float(np.dot(e_vec, r_vec)) / (ecc * r),
                flip=float(np.dot(r_vec, v_vec)) < 0.0,
            )
    return out
