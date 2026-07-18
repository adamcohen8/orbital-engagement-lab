# ruff: noqa: F401,I001
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from sim.dynamics.orbit.cr3bp import cr3bp_moon_state_km_s, cr3bp_relative_state
from sim.utils.frames import eci_relative_to_ric_rect

if TYPE_CHECKING:
    from .training_models import ForbiddenRegionConfig

EARTH_MU_KM3_S2 = 398600.4418

def relative_ric_state_from_arrays(target_truth: np.ndarray, chaser_truth: np.ndarray) -> np.ndarray:
    target = np.array(target_truth, dtype=float).reshape(-1)
    chaser = np.array(chaser_truth, dtype=float).reshape(-1)
    if target.size < 6 or chaser.size < 6:
        return np.full(6, np.nan, dtype=float)
    return eci_relative_to_ric_rect(chaser[:6], target[:6])


def relative_moon_ric_state_from_arrays(target_truth: np.ndarray, chaser_truth: np.ndarray) -> np.ndarray:
    target = np.array(target_truth, dtype=float).reshape(-1)
    chaser = np.array(chaser_truth, dtype=float).reshape(-1)
    if target.size < 6 or chaser.size < 6:
        return np.full(6, np.nan, dtype=float)
    moon = cr3bp_moon_state_km_s()
    return eci_relative_to_ric_rect(chaser[:6] - moon, target[:6] - moon)


def relative_state_from_arrays(target_truth: np.ndarray, chaser_truth: np.ndarray, *, frame: str = "ric") -> np.ndarray:
    frame_key = _relative_frame_key(frame)
    if frame_key == "cislunar":
        target = np.array(target_truth, dtype=float).reshape(-1)
        chaser = np.array(chaser_truth, dtype=float).reshape(-1)
        if target.size < 6 or chaser.size < 6:
            return np.full(6, np.nan, dtype=float)
        return cr3bp_relative_state(chaser[:6], target[:6])
    if frame_key == "moon_ric":
        return relative_moon_ric_state_from_arrays(target_truth, chaser_truth)
    return relative_ric_state_from_arrays(target_truth, chaser_truth)


def _relative_frame_key(frame: str) -> str:
    key = str(frame or "ric").strip().lower().replace("-", "_")
    if key in {"cislunar", "cislunar_l1", "earth_moon_rotating", "cr3bp", "cr3bp_rotating"}:
        return "cislunar"
    if key in {"moon_ric", "lunar_ric", "target_moon_ric", "target_lunar_ric"}:
        return "moon_ric"
    return "ric"


def nmt_position_error_km(
    relative_ric_km: np.ndarray,
    *,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float = 0.0,
    cross_track_phase_deg: float = 0.0,
    center_ric_km: np.ndarray,
) -> np.ndarray:
    pos = np.array(relative_ric_km, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    if pos.shape[1] < 3:
        raise ValueError("relative_ric_km must contain R, I, and C components.")
    center = np.array(center_ric_km, dtype=float).reshape(3)
    curve = nmt_curve_points_km(
        radial_amplitude_km=radial_amplitude_km,
        cross_track_amplitude_km=cross_track_amplitude_km,
        cross_track_phase_deg=cross_track_phase_deg,
        center_ric_km=center,
    )
    if curve.size == 0:
        return np.linalg.norm(pos[:, :3] - center.reshape(1, 3), axis=1)
    delta = pos[:, None, :3] - curve[None, :, :]
    return np.min(np.linalg.norm(delta, axis=2), axis=1)


def nmt_curve_points_km(
    *,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float = 0.0,
    cross_track_phase_deg: float = 0.0,
    center_ric_km: np.ndarray,
    samples: int = 721,
) -> np.ndarray:
    a_r = float(radial_amplitude_km)
    if not np.isfinite(a_r) or a_r <= 0.0:
        return np.empty((0, 3), dtype=float)
    a_c = float(cross_track_amplitude_km)
    if not np.isfinite(a_c):
        a_c = 0.0
    phase = np.deg2rad(float(cross_track_phase_deg))
    center = np.array(center_ric_km, dtype=float).reshape(3)
    return _cached_nmt_curve_points_km(
        float(a_r),
        float(a_c),
        float(phase),
        (float(center[0]), float(center[1]), float(center[2])),
        int(max(int(samples), 8)),
    ).copy()


@lru_cache(maxsize=64)
def _cached_nmt_curve_points_km(
    radial_amplitude_km: float,
    cross_track_amplitude_km: float,
    phase_rad: float,
    center_ric_km: tuple[float, float, float],
    samples: int,
) -> np.ndarray:
    a_r = float(radial_amplitude_km)
    a_c = float(cross_track_amplitude_km)
    phase = float(phase_rad)
    center = np.array(center_ric_km, dtype=float).reshape(3)
    theta = np.linspace(0.0, 2.0 * np.pi, max(int(samples), 8), endpoint=True)
    pts = np.zeros((theta.size, 3), dtype=float)
    pts[:, 0] = center[0] + a_r * np.cos(theta)
    pts[:, 1] = center[1] - 2.0 * a_r * np.sin(theta)
    pts[:, 2] = center[2] + a_c * np.cos(theta + phase)
    pts.setflags(write=False)
    return pts


def nmt_velocity_error_km_s(
    relative_ric_state: np.ndarray,
    *,
    mean_motion_rad_s: float,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float = 0.0,
    cross_track_phase_deg: float = 0.0,
    center_ric_km: np.ndarray,
) -> float:
    rel = np.array(relative_ric_state, dtype=float).reshape(-1)
    if rel.size < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    center = np.array(center_ric_km, dtype=float).reshape(3)
    n = float(mean_motion_rad_s)
    curve = nmt_curve_points_km(
        radial_amplitude_km=radial_amplitude_km,
        cross_track_amplitude_km=cross_track_amplitude_km,
        cross_track_phase_deg=cross_track_phase_deg,
        center_ric_km=center,
    )
    if curve.size == 0 or not np.isfinite(n):
        return float(np.linalg.norm(rel[3:6]))
    idx = int(np.argmin(np.linalg.norm(curve - rel[:3].reshape(1, 3), axis=1)))
    theta = 2.0 * np.pi * idx / max(curve.shape[0] - 1, 1)
    a_r = float(radial_amplitude_km)
    a_c = float(cross_track_amplitude_km)
    phase = np.deg2rad(float(cross_track_phase_deg))
    expected = np.array(
        [
            -a_r * n * np.sin(theta),
            -2.0 * a_r * n * np.cos(theta),
            -a_c * n * np.sin(theta + phase),
        ],
        dtype=float,
    )
    return float(np.linalg.norm(rel[3:6] - expected))


def nmt_element_errors(
    relative_ric_state: np.ndarray,
    *,
    mean_motion_rad_s: np.ndarray | float,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float,
    center_ric_km: np.ndarray,
) -> dict[str, np.ndarray]:
    rel = np.array(relative_ric_state, dtype=float)
    if rel.ndim == 1:
        rel = rel.reshape(1, -1)
    if rel.shape[1] < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    n_raw = np.array(mean_motion_rad_s, dtype=float).reshape(-1)
    if n_raw.size == 1:
        n = np.full(rel.shape[0], float(n_raw[0]), dtype=float)
    else:
        n = n_raw[: rel.shape[0]]
        if n.size < rel.shape[0]:
            n = np.pad(n, (0, rel.shape[0] - n.size), constant_values=np.nan)
    center = np.array(center_ric_km, dtype=float).reshape(3)
    pos = rel[:, :3] - center.reshape(1, 3)
    vel = rel[:, 3:6]
    valid_n = np.isfinite(n) & (np.abs(n) > 1.0e-12)
    radial_amp = np.full(rel.shape[0], np.nan, dtype=float)
    cross_amp = np.full(rel.shape[0], np.nan, dtype=float)
    drift_vel_err = np.full(rel.shape[0], np.nan, dtype=float)
    radial_amp[valid_n] = np.sqrt(pos[valid_n, 0] ** 2 + (vel[valid_n, 0] / n[valid_n]) ** 2)
    cross_amp[valid_n] = np.sqrt(pos[valid_n, 2] ** 2 + (vel[valid_n, 2] / n[valid_n]) ** 2)
    drift_vel_err[valid_n] = np.abs(vel[valid_n, 1] + 2.0 * n[valid_n] * pos[valid_n, 0])
    return {
        "radial_amplitude_km": radial_amp,
        "cross_track_amplitude_km": cross_amp,
        "radial_amplitude_error_km": np.abs(radial_amp - float(radial_amplitude_km)),
        "cross_track_amplitude_error_km": np.abs(cross_amp - float(cross_track_amplitude_km)),
        "drift_velocity_error_km_s": drift_vel_err,
    }


def _nmt_element_error_values(
    relative_ric_state: np.ndarray,
    *,
    mean_motion_rad_s: float,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float,
    center_ric_km: np.ndarray,
    drift_velocity_error_km_s: float | None = None,
) -> dict[str, float]:
    rel = np.array(relative_ric_state, dtype=float).reshape(-1)
    center = np.array(center_ric_km, dtype=float).reshape(3)
    if rel.size < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    n = float(mean_motion_rad_s)
    if not np.isfinite(n) or abs(n) <= 1.0e-12:
        return {
            "radial_amplitude_km": float("nan"),
            "cross_track_amplitude_km": float("nan"),
            "radial_amplitude_error_km": float("nan"),
            "cross_track_amplitude_error_km": float("nan"),
            "drift_velocity_error_km_s": float("nan"),
        }
    pos = rel[:3] - center
    vel = rel[3:6]
    radial_amp = float(np.sqrt(pos[0] ** 2 + (vel[0] / n) ** 2))
    cross_amp = float(np.sqrt(pos[2] ** 2 + (vel[2] / n) ** 2))
    drift_error = (
        float(drift_velocity_error_km_s)
        if drift_velocity_error_km_s is not None and np.isfinite(float(drift_velocity_error_km_s))
        else abs(float(vel[1]) + 2.0 * n * float(pos[0]))
    )
    return {
        "radial_amplitude_km": radial_amp,
        "cross_track_amplitude_km": cross_amp,
        "radial_amplitude_error_km": abs(radial_amp - float(radial_amplitude_km)),
        "cross_track_amplitude_error_km": abs(cross_amp - float(cross_track_amplitude_km)),
        "drift_velocity_error_km_s": abs(drift_error),
    }


def _semimajor_axis_drift_velocity_error_km_s(
    target_state_eci: np.ndarray | None,
    chaser_state_eci: np.ndarray | None,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> float | None:
    if target_state_eci is None or chaser_state_eci is None:
        return None
    target_a = _semimajor_axis_km(target_state_eci, mu_km3_s2=mu_km3_s2)
    chaser_a = _semimajor_axis_km(chaser_state_eci, mu_km3_s2=mu_km3_s2)
    if target_a is None or chaser_a is None or target_a <= 0.0:
        return None
    n = float(np.sqrt(float(mu_km3_s2) / (float(target_a) ** 3)))
    return float(abs(0.5 * n * (float(chaser_a) - float(target_a))))


def _semimajor_axis_km(state_eci: np.ndarray, *, mu_km3_s2: float = EARTH_MU_KM3_S2) -> float | None:
    state = np.array(state_eci, dtype=float).reshape(-1)
    if state.size < 6:
        return None
    r_norm = float(np.linalg.norm(state[:3]))
    v_norm = float(np.linalg.norm(state[3:6]))
    if not np.isfinite(r_norm) or not np.isfinite(v_norm) or r_norm <= 0.0:
        return None
    specific_energy = 0.5 * v_norm * v_norm - float(mu_km3_s2) / r_norm
    if not np.isfinite(specific_energy) or abs(specific_energy) <= 1.0e-12:
        return None
    return float(-float(mu_km3_s2) / (2.0 * specific_energy))


def _nmt_element_goal_error_km(
    *,
    radial_amplitude_error_km: float,
    cross_track_amplitude_error_km: float,
    include_radial: bool,
    include_cross_track: bool,
) -> float:
    values = []
    if include_radial:
        values.append(float(radial_amplitude_error_km))
    if include_cross_track:
        values.append(float(cross_track_amplitude_error_km))
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else float("nan")


def _nmt_element_goal_error_array(
    element_errors: dict[str, np.ndarray],
    *,
    include_radial: bool,
    include_cross_track: bool,
) -> np.ndarray:
    values: list[np.ndarray] = []
    if include_radial:
        values.append(np.array(element_errors["radial_amplitude_error_km"], dtype=float).reshape(-1))
    if include_cross_track:
        values.append(np.array(element_errors["cross_track_amplitude_error_km"], dtype=float).reshape(-1))
    if not values:
        first = next(iter(element_errors.values()), np.zeros(0, dtype=float))
        return np.zeros(np.array(first, dtype=float).reshape(-1).shape, dtype=float)
    return np.nanmax(np.vstack(values), axis=0)


def _final_nmt_element_values(element_errors: dict[str, np.ndarray] | None) -> dict[str, float]:
    keys = (
        "radial_amplitude_km",
        "cross_track_amplitude_km",
        "radial_amplitude_error_km",
        "cross_track_amplitude_error_km",
        "drift_velocity_error_km_s",
    )
    if element_errors is None:
        return {k: float("nan") for k in keys}
    return {k: float(np.array(element_errors[k], dtype=float).reshape(-1)[-1]) for k in keys}

def _position_segment_intersects_box(
    start_ric_km: np.ndarray, end_ric_km: np.ndarray, *, center: np.ndarray, half_width: np.ndarray
) -> bool:
    start = np.array(start_ric_km, dtype=float).reshape(3)
    end = np.array(end_ric_km, dtype=float).reshape(3)
    lo = np.array(center, dtype=float).reshape(3) - np.array(half_width, dtype=float).reshape(3)
    hi = np.array(center, dtype=float).reshape(3) + np.array(half_width, dtype=float).reshape(3)
    delta = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(float(delta[axis])) <= 1.0e-12:
            if start[axis] < lo[axis] or start[axis] > hi[axis]:
                return False
            continue
        inv_delta = 1.0 / float(delta[axis])
        t1 = float((lo[axis] - start[axis]) * inv_delta)
        t2 = float((hi[axis] - start[axis]) * inv_delta)
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return False
    return True


def _position_segment_intersects_bounds(
    start_ric_km: np.ndarray,
    end_ric_km: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    start = np.asarray(start_ric_km, dtype=float).reshape(3)
    end = np.asarray(end_ric_km, dtype=float).reshape(3)
    lo = np.asarray(lower, dtype=float).reshape(3)
    hi = np.asarray(upper, dtype=float).reshape(3)
    delta = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(float(delta[axis])) <= 1.0e-12:
            if start[axis] < lo[axis] or start[axis] > hi[axis]:
                return False
            continue
        if np.isfinite(lo[axis]):
            t_min = max(t_min, float((lo[axis] - start[axis]) / delta[axis])) if delta[axis] > 0 else t_min
            t_max = min(t_max, float((lo[axis] - start[axis]) / delta[axis])) if delta[axis] < 0 else t_max
        if np.isfinite(hi[axis]):
            t_max = min(t_max, float((hi[axis] - start[axis]) / delta[axis])) if delta[axis] > 0 else t_max
            t_min = max(t_min, float((hi[axis] - start[axis]) / delta[axis])) if delta[axis] < 0 else t_min
        if t_min > t_max:
            return False
    return t_max >= 0.0 and t_min <= 1.0


def _position_segment_intersects_cylinder(
    start_ric_km: np.ndarray,
    end_ric_km: np.ndarray,
    *,
    center: np.ndarray,
    axis: int,
    radius_km: float | None,
    height_km: float | None,
) -> bool:
    if radius_km is None or height_km is None:
        return False
    start = np.asarray(start_ric_km, dtype=float).reshape(3) - np.asarray(center, dtype=float).reshape(3)
    end = np.asarray(end_ric_km, dtype=float).reshape(3) - np.asarray(center, dtype=float).reshape(3)
    delta = end - start
    half_height = max(float(height_km), 0.0) / 2.0
    axial = _linear_interval_in_bounds(float(start[axis]), float(delta[axis]), -half_height, half_height)
    if axial is None:
        return False
    cross_axes = tuple(idx for idx in range(3) if idx != int(axis))
    p = start[list(cross_axes)]
    d = delta[list(cross_axes)]
    radial = _quadratic_radius_interval(p, d, max(float(radius_km), 0.0))
    if radial is None:
        return False
    return max(axial[0], radial[0], 0.0) <= min(axial[1], radial[1], 1.0)


def _position_segment_intersects_annular_sector(
    start_ric_km: np.ndarray,
    end_ric_km: np.ndarray,
    *,
    region: ForbiddenRegionConfig,
) -> bool:
    if region.inner_radius_km is None or region.outer_radius_km is None:
        return False
    start = np.asarray(start_ric_km, dtype=float).reshape(3)
    end = np.asarray(end_ric_km, dtype=float).reshape(3)
    delta = end - start
    x_axis, y_axis, out_axis = _plane_axes(region.plane)
    center = np.asarray(region.center_ric_km, dtype=float).reshape(3)
    p = (start - center)[[x_axis, y_axis]]
    d = delta[[x_axis, y_axis]]
    candidates = {0.0, 1.0}
    for radius in (float(region.inner_radius_km), float(region.outer_radius_km)):
        candidates.update(_quadratic_boundary_roots(p, d, max(radius, 0.0)))
    if region.max_abs_out_of_plane_km is not None:
        limit = max(float(region.max_abs_out_of_plane_km), 0.0)
        p_out = float(start[out_axis] - center[out_axis])
        d_out = float(delta[out_axis])
        if abs(d_out) > 1.0e-12:
            candidates.update(((limit - p_out) / d_out, (-limit - p_out) / d_out))
    if region.angle_min_deg is not None or region.angle_max_deg is not None:
        start_deg = 0.0 if region.angle_min_deg is None else float(region.angle_min_deg)
        end_deg = 360.0 if region.angle_max_deg is None else float(region.angle_max_deg)
        for angle_deg in (start_deg, end_deg):
            ray = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))], dtype=float)
            denom = float(d[0] * ray[1] - d[1] * ray[0])
            if abs(denom) > 1.0e-12:
                candidates.add(float((p[1] * ray[0] - p[0] * ray[1]) / denom))
    ordered = sorted(float(value) for value in candidates if np.isfinite(value) and -1.0e-12 <= value <= 1.0 + 1.0e-12)
    probes = ordered + [(left + right) / 2.0 for left, right in zip(ordered, ordered[1:], strict=False)]
    points = np.vstack([start + np.clip(value, 0.0, 1.0) * delta for value in probes])
    return bool(np.any(region.contains_positions(points)))


def _linear_interval_in_bounds(value: float, delta: float, lower: float, upper: float) -> tuple[float, float] | None:
    if abs(delta) <= 1.0e-12:
        return (0.0, 1.0) if lower <= value <= upper else None
    t0 = (lower - value) / delta
    t1 = (upper - value) / delta
    return (min(t0, t1), max(t0, t1))


def _quadratic_boundary_roots(position: np.ndarray, delta: np.ndarray, radius: float) -> tuple[float, ...]:
    a = float(np.dot(delta, delta))
    b = 2.0 * float(np.dot(position, delta))
    c = float(np.dot(position, position) - radius * radius)
    if a <= 1.0e-18:
        return ()
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return ()
    root = float(np.sqrt(max(discriminant, 0.0)))
    return ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))


def _quadratic_radius_interval(
    position: np.ndarray,
    delta: np.ndarray,
    radius: float,
) -> tuple[float, float] | None:
    roots = _quadratic_boundary_roots(position, delta, radius)
    if roots:
        return (min(roots), max(roots))
    return (0.0, 1.0) if float(np.linalg.norm(position)) <= radius else None


def _hard_speed_limit_violated(relative_ric_state: np.ndarray, *, radius_km: float, speed_limit_km_s: float) -> bool:
    rel = np.array(relative_ric_state, dtype=float)
    if rel.ndim == 1:
        rel = rel.reshape(1, -1)
    if rel.shape[1] < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    positions = rel[:, :3]
    velocities = rel[:, 3:6]
    ranges = np.linalg.norm(positions, axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    if bool(np.any((ranges <= float(radius_km)) & (speeds > float(speed_limit_km_s)))):
        return True
    if rel.shape[0] < 2:
        return False
    for idx in range(1, rel.shape[0]):
        interval = _position_segment_sphere_interval(
            positions[idx - 1],
            positions[idx],
            radius_km=float(radius_km),
        )
        if interval is None:
            continue
        u0, u1 = interval
        v0 = velocities[idx - 1]
        dv = velocities[idx] - velocities[idx - 1]
        entry_speed = float(np.linalg.norm(v0 + dv * u0))
        exit_speed = float(np.linalg.norm(v0 + dv * u1))
        if max(entry_speed, exit_speed) > float(speed_limit_km_s):
            return True
    return False


def _hard_speed_limit_sample_violated(
    previous_relative_ric_state: np.ndarray | None,
    current_relative_ric_state: np.ndarray,
    *,
    radius_km: float,
    speed_limit_km_s: float,
) -> bool:
    current = np.array(current_relative_ric_state, dtype=float).reshape(6)
    radius = float(radius_km)
    speed_limit = float(speed_limit_km_s)
    if float(np.linalg.norm(current[:3])) <= radius and float(np.linalg.norm(current[3:6])) > speed_limit:
        return True
    if previous_relative_ric_state is None:
        return False
    previous = np.array(previous_relative_ric_state, dtype=float).reshape(6)
    interval = _position_segment_sphere_interval(previous[:3], current[:3], radius_km=radius)
    if interval is None:
        return False
    u0, u1 = interval
    v0 = previous[3:6]
    dv = current[3:6] - previous[3:6]
    entry_speed = float(np.linalg.norm(v0 + dv * u0))
    exit_speed = float(np.linalg.norm(v0 + dv * u1))
    return bool(max(entry_speed, exit_speed) > speed_limit)


def _position_segment_sphere_interval(
    start_ric_km: np.ndarray,
    end_ric_km: np.ndarray,
    *,
    radius_km: float,
) -> tuple[float, float] | None:
    start = np.array(start_ric_km, dtype=float).reshape(3)
    end = np.array(end_ric_km, dtype=float).reshape(3)
    radius = float(radius_km)
    if radius < 0.0:
        return None
    inside_start = float(np.linalg.norm(start)) <= radius
    inside_end = float(np.linalg.norm(end)) <= radius
    if inside_start and inside_end:
        return (0.0, 1.0)
    delta = end - start
    a = float(np.dot(delta, delta))
    if a <= 1.0e-18:
        return (0.0, 1.0) if inside_start else None
    b = 2.0 * float(np.dot(start, delta))
    c = float(np.dot(start, start) - radius * radius)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = float(np.sqrt(max(disc, 0.0)))
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    if t0 > t1:
        t0, t1 = t1, t0
    entry = max(0.0, t0)
    exit_ = min(1.0, t1)
    if inside_start:
        entry = 0.0
    if inside_end:
        exit_ = 1.0
    if entry <= exit_ and t1 >= 0.0 and t0 <= 1.0:
        return (float(entry), float(exit_))
    return None


def _ric_bound_array(value: Any, *, default: float, field_name: str) -> np.ndarray:
    if value is None:
        return np.full(3, float(default), dtype=float)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Forbidden region {field_name} must be a length-3 list.")
    vals = [float(default) if item is None else float(item) for item in value]
    return np.array(vals, dtype=float).reshape(3)


def _unit_ric_array(value: Any, *, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Sun angle constraint {field_name} must be a length-3 list.")
    vec = np.array(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Sun angle constraint {field_name} must be nonzero.")
    return vec / norm


def _unit_direction_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array(value, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    dirs = np.zeros((pos.shape[0], 3), dtype=float)
    if pos.shape[1] < 3:
        return dirs, np.zeros(pos.shape[0], dtype=bool)
    norm = np.linalg.norm(pos[:, :3], axis=1)
    valid = np.isfinite(norm) & (norm > 0.0)
    dirs[valid, :] = pos[valid, :3] / norm[valid].reshape(-1, 1)
    return dirs, valid


def _validate_annular_sector_region(region: ForbiddenRegionConfig) -> None:
    _plane_axes(region.plane)
    if region.inner_radius_km is None or region.outer_radius_km is None:
        raise ValueError(f"Forbidden region '{region.name}' annular_sector requires inner_radius_km and outer_radius_km.")
    if float(region.inner_radius_km) < 0.0:
        raise ValueError(f"Forbidden region '{region.name}' inner_radius_km must be nonnegative.")
    if float(region.outer_radius_km) <= float(region.inner_radius_km):
        raise ValueError(f"Forbidden region '{region.name}' outer_radius_km must be greater than inner_radius_km.")
    if region.max_abs_out_of_plane_km is not None and float(region.max_abs_out_of_plane_km) < 0.0:
        raise ValueError(f"Forbidden region '{region.name}' max_abs_out_of_plane_km must be nonnegative.")


def _validate_cylinder_region(region: ForbiddenRegionConfig) -> None:
    _axis_index(region.axis)
    if region.radius_km is None or region.height_km is None:
        raise ValueError(f"Forbidden region '{region.name}' cylinder requires radius_km and height_km.")
    if float(region.radius_km) <= 0.0:
        raise ValueError(f"Forbidden region '{region.name}' radius_km must be positive.")
    if float(region.height_km) <= 0.0:
        raise ValueError(f"Forbidden region '{region.name}' height_km must be positive.")


def _validate_sphere_region(region: ForbiddenRegionConfig) -> None:
    if region.radius_km is None:
        raise ValueError(f"Forbidden region '{region.name}' sphere requires radius_km.")
    if float(region.radius_km) <= 0.0:
        raise ValueError(f"Forbidden region '{region.name}' radius_km must be positive.")


def _axis_index(axis: str) -> int:
    key = str(axis or "").strip().upper()
    if key == "R":
        return 0
    if key == "I":
        return 1
    if key == "C":
        return 2
    raise ValueError(f"Forbidden region axis must be one of R, I, or C; got '{axis}'.")


def _plane_axes(plane: str) -> tuple[int, int, int]:
    key = str(plane or "").strip().upper()
    if key == "RI":
        return 1, 0, 2
    if key == "RC":
        return 2, 0, 1
    if key == "IC":
        return 1, 2, 0
    raise ValueError(f"Forbidden region plane must be one of RI, RC, or IC; got '{plane}'.")


def _angles_in_range_deg(angles_deg: np.ndarray, start_deg: float, end_deg: float) -> np.ndarray:
    span = float(end_deg) - float(start_deg)
    if span >= 360.0:
        return np.ones_like(np.array(angles_deg, dtype=float), dtype=bool)
    while span < 0.0:
        span += 360.0
    relative = (np.array(angles_deg, dtype=float) - float(start_deg)) % 360.0
    return relative <= span

__all__ = [name for name in globals() if not name.startswith("__")]
