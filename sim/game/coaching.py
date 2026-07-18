# ruff: noqa: F401,I001
from __future__ import annotations

import numpy as np

from sim.game.formatting import format_distance_km, format_speed_km_s, format_speed_m_s

def _format_optional_time(value: float | None) -> str:
    if value is None:
        return "Not Achieved"
    return f"{float(value):.1f} s"


def _format_distance_text(value_km: float) -> str:
    return format_distance_km(value_km)


def _format_speed_text(value_km_s: float) -> str:
    return format_speed_km_s(value_km_s)


def _format_signed_distance_text(value_km: float) -> str:
    sign = "+" if float(value_km) >= 0.0 else "-"
    return sign + format_distance_km(abs(float(value_km)))

def _sampled_dwell_time_s(mask: np.ndarray, time_s: np.ndarray) -> float:
    inside = np.array(mask, dtype=bool).reshape(-1)
    t = np.array(time_s, dtype=float).reshape(-1)
    n = min(inside.size, t.size)
    if n < 2:
        return 0.0
    dt = np.diff(t[:n])
    valid = np.isfinite(dt) & (dt > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(dt[valid] * inside[: n - 1][valid]))


def _integrated_delta_v_m_s(thrust_km_s2: np.ndarray, time_s: np.ndarray) -> float:
    thrust = np.array(thrust_km_s2, dtype=float)
    t = np.array(time_s, dtype=float).reshape(-1)
    n = min(thrust.shape[0], t.size)
    if n < 2:
        return 0.0
    # Snapshot i reports the command applied during the interval ending at t[i].
    accel = np.linalg.norm(thrust[1:n, :], axis=1)
    dt = np.diff(t[:n])
    valid = np.isfinite(accel) & np.isfinite(dt) & (dt > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(accel[valid] * dt[valid]) * 1.0e3)

__all__ = [name for name in globals() if not name.startswith("__")]
