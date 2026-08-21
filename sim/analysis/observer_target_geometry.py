"""Shared vectorized observer/target geometry for OEL analysis products."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceTargetGeometry:
    range_km: np.ndarray
    cosine_off_axis: np.ndarray
    horizon_clearance_km: np.ndarray
    visible: np.ndarray
    inside_pattern: np.ndarray
    inside_range: np.ndarray
    available: np.ndarray


def evaluate_surface_targets_ecef(
    *,
    observer_ecef_km: np.ndarray,
    target_ecef_km: np.ndarray,
    target_outward_normal_ecef: np.ndarray,
    boresight_ecef: np.ndarray,
    half_angle_rad: float,
    max_range_km: float | None = None,
    angular_tolerance_rad: float = 1.0e-12,
    range_tolerance_km: float = 1.0e-9,
) -> SurfaceTargetGeometry:
    """Evaluate clear-line-of-sight hard-cone geometry to convex Earth targets.

    Targets must lie on the WGS84 ellipsoid and normals must be unit outward
    normals. Positive normal clearance is equivalent to an unobstructed open
    segment for a convex ellipsoid; exact tangency is blocked.
    """

    observer = np.asarray(observer_ecef_km, dtype=float).reshape(3)
    targets = np.asarray(target_ecef_km, dtype=float)
    normals = np.asarray(target_outward_normal_ecef, dtype=float)
    boresight = np.asarray(boresight_ecef, dtype=float).reshape(3)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.all(np.isfinite(targets)):
        raise ValueError("target_ecef_km must be finite with shape (targets, 3).")
    if normals.shape != targets.shape or not np.all(np.isfinite(normals)):
        raise ValueError("target_outward_normal_ecef must be finite and match target_ecef_km.")
    if not np.all(np.isfinite(observer)) or not np.all(np.isfinite(boresight)):
        raise ValueError("Observer and boresight vectors must be finite.")
    boresight_norm = float(np.linalg.norm(boresight))
    if abs(boresight_norm - 1.0) > 1.0e-10:
        raise ValueError("boresight_ecef must be a unit vector within 1e-10.")
    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(np.abs(normal_norms - 1.0) > 1.0e-10):
        raise ValueError("Target outward normals must be unit vectors within 1e-10.")
    if not np.isfinite(float(half_angle_rad)) or not 0.0 < float(half_angle_rad) < 0.5 * np.pi:
        raise ValueError("half_angle_rad must be finite and strictly within (0, pi/2).")
    if max_range_km is not None and (
        not np.isfinite(float(max_range_km)) or float(max_range_km) <= 0.0
    ):
        raise ValueError("max_range_km must be positive and finite when provided.")

    delta = targets - observer
    ranges = np.linalg.norm(delta, axis=1)
    if np.any(~np.isfinite(ranges)) or np.any(ranges <= 0.0):
        raise ValueError("Observer/target range must be positive and finite.")
    horizon_clearance = np.einsum("ij,ij->i", normals, observer - targets)
    cosine_off_axis = (delta @ boresight) / ranges
    visible = horizon_clearance > float(range_tolerance_km)
    inside_pattern = cosine_off_axis >= float(
        np.cos(float(half_angle_rad) + float(angular_tolerance_rad))
    )
    if max_range_km is None:
        inside_range = np.ones(ranges.shape, dtype=bool)
    else:
        inside_range = ranges <= float(max_range_km) + float(range_tolerance_km)
    return SurfaceTargetGeometry(
        range_km=ranges,
        cosine_off_axis=cosine_off_axis,
        horizon_clearance_km=horizon_clearance,
        visible=visible,
        inside_pattern=inside_pattern,
        inside_range=inside_range,
        available=visible & inside_pattern & inside_range,
    )


__all__ = ["SurfaceTargetGeometry", "evaluate_surface_targets_ecef"]
