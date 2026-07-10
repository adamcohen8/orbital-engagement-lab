from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2


@dataclass(frozen=True)
class LambertSolution:
    v1_km_s: np.ndarray
    v2_km_s: np.ndarray
    time_of_flight_s: float
    short_way: bool
    revolutions: int
    converged: bool
    iterations: int
    residual_s: float


def solve_lambert_universal_variable(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    time_of_flight_s: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    short_way: bool = True,
    revolutions: int = 0,
    max_iterations: int = 100,
    tolerance_s: float = 1.0e-7,
) -> LambertSolution:
    """Solve the single-revolution two-body Lambert boundary-value problem.

    The implementation uses the classical universal-variable formulation with
    a bracketed bisection search on the Stumpff parameter.  Multi-revolution
    Lambert transfers are intentionally rejected for this first deterministic
    planner surface.
    """

    if int(revolutions) != 0:
        raise ValueError("Only zero-revolution Lambert transfers are currently supported.")
    tof = float(time_of_flight_s)
    if tof <= 0.0:
        raise ValueError("Lambert time_of_flight_s must be positive.")
    mu = float(mu_km3_s2)
    if mu <= 0.0:
        raise ValueError("Lambert mu_km3_s2 must be positive.")

    r1 = np.asarray(r1_km, dtype=float).reshape(3)
    r2 = np.asarray(r2_km, dtype=float).reshape(3)
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    if r1_norm <= 0.0 or r2_norm <= 0.0:
        raise ValueError("Lambert endpoint position norms must be positive.")

    cos_dtheta = float(np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0))
    sin_dtheta = float(np.linalg.norm(np.cross(r1, r2)) / (r1_norm * r2_norm))
    if not bool(short_way):
        sin_dtheta = -sin_dtheta
    one_minus_cos = 1.0 - cos_dtheta
    if one_minus_cos <= 1.0e-14 or abs(sin_dtheta) <= 1.0e-14:
        raise ValueError("Lambert transfer angle is singular for the current solver.")
    a_lambert = sin_dtheta * float(np.sqrt(r1_norm * r2_norm / one_minus_cos))
    if abs(a_lambert) <= 1.0e-14:
        raise ValueError("Lambert geometry produced a singular transfer parameter.")

    def time_for_z(z: float) -> tuple[float, float, float, float]:
        c = _stumpff_c(z)
        s = _stumpff_s(z)
        if c <= 0.0:
            return float("nan"), float("nan"), c, s
        y = r1_norm + r2_norm + a_lambert * ((z * s - 1.0) / float(np.sqrt(c)))
        if y < 0.0:
            return float("nan"), y, c, s
        x = float(np.sqrt(y / c))
        dt = (x * x * x * s + a_lambert * float(np.sqrt(y))) / float(np.sqrt(mu))
        return float(dt), float(y), c, s

    lower = -4.0 * np.pi * np.pi
    upper = 4.0 * np.pi * np.pi
    dt_upper, _, _, _ = time_for_z(float(upper))
    expand_count = 0
    while (not np.isfinite(dt_upper) or dt_upper < tof) and expand_count < 25:
        upper *= 2.0
        dt_upper, _, _, _ = time_for_z(float(upper))
        expand_count += 1
    if not np.isfinite(dt_upper) or dt_upper < tof:
        raise ValueError("Lambert solver could not bracket the requested time of flight.")

    z_mid = 0.0
    dt_mid = float("nan")
    y_mid = float("nan")
    c_mid = float("nan")
    s_mid = float("nan")
    residual = float("inf")
    iterations = 0
    for iteration_index in range(1, int(max_iterations) + 1):
        iterations = int(iteration_index)
        z_mid = 0.5 * (float(lower) + float(upper))
        dt_mid, y_mid, c_mid, s_mid = time_for_z(z_mid)
        if not np.isfinite(dt_mid) or y_mid < 0.0:
            lower = z_mid
            continue
        residual = dt_mid - tof
        if abs(residual) <= float(tolerance_s):
            break
        if residual <= 0.0:
            lower = z_mid
        else:
            upper = z_mid

    if not np.isfinite(dt_mid) or not np.isfinite(y_mid) or y_mid < 0.0:
        raise ValueError("Lambert solver did not converge to a finite transfer.")
    f = 1.0 - y_mid / r1_norm
    g = a_lambert * float(np.sqrt(y_mid / mu))
    gdot = 1.0 - y_mid / r2_norm
    if abs(g) <= 1.0e-14:
        raise ValueError("Lambert solver produced a singular Lagrange g coefficient.")
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    converged = bool(abs(residual) <= float(tolerance_s))
    return LambertSolution(
        v1_km_s=np.asarray(v1, dtype=float),
        v2_km_s=np.asarray(v2, dtype=float),
        time_of_flight_s=tof,
        short_way=bool(short_way),
        revolutions=0,
        converged=converged,
        iterations=int(iterations),
        residual_s=float(residual),
    )


def _stumpff_c(z: float) -> float:
    zf = float(z)
    if zf > 1.0e-8:
        root = float(np.sqrt(zf))
        return float((1.0 - np.cos(root)) / zf)
    if zf < -1.0e-8:
        root = float(np.sqrt(-zf))
        return float((np.cosh(root) - 1.0) / (-zf))
    return 0.5 - zf / 24.0 + zf * zf / 720.0 - zf * zf * zf / 40320.0


def _stumpff_s(z: float) -> float:
    zf = float(z)
    if zf > 1.0e-8:
        root = float(np.sqrt(zf))
        return float((root - np.sin(root)) / (root * root * root))
    if zf < -1.0e-8:
        root = float(np.sqrt(-zf))
        return float((np.sinh(root) - root) / (root * root * root))
    return 1.0 / 6.0 - zf / 120.0 + zf * zf / 5040.0 - zf * zf * zf / 362880.0
