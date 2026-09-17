"""Spherical forbidden shell with an open approach cone; shared scoring/display geometry."""

from dataclasses import replace
from functools import lru_cache

import numpy as np


def validate(region):
    values = (region.inner_radius_km, region.outer_radius_km, region.cone_half_angle_deg)
    if any(value is None or not np.isfinite(value) for value in values):
        raise ValueError("Spherical corridor requires finite inner/outer radii and cone_half_angle_deg.")
    if not 0 < region.inner_radius_km < region.outer_radius_km or not 0 < region.cone_half_angle_deg < 90:
        raise ValueError("Spherical corridor requires 0 < inner < outer and 0 < cone half angle < 90 degrees.")
    if not np.all(np.isfinite(region.center_ric_km)):
        raise ValueError("Spherical corridor center must be finite.")


def contains(region, positions):
    delta = np.atleast_2d(positions)[:, :3] - region.center_ric_km
    radius = np.linalg.norm(delta, axis=1)
    # The permitted cone points down -R. Its wall belongs to the forbidden shell.
    return (
        (radius >= region.inner_radius_km)
        & (radius <= region.outer_radius_km)
        & (-delta[:, 0] <= radius * np.cos(np.deg2rad(region.cone_half_angle_deg)))
    )


def intersects(region, start, end):
    """Partition a segment at all sphere/cone crossings, then classify each interval."""
    origin = np.asarray(start) - region.center_ric_km
    direction = np.asarray(end) - np.asarray(start)
    cuts = [0.0, 1.0]

    def roots(a, b, c):
        if abs(a) < 1e-15:
            if abs(b) > 1e-15:
                cuts.append(-c / b)
            return
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            root = np.sqrt(discriminant)
            cuts.extend(((-b - root) / (2 * a), (-b + root) / (2 * a)))

    for radius in (region.inner_radius_km, region.outer_radius_km):
        roots(direction @ direction, 2 * (origin @ direction), origin @ origin - radius * radius)
    cos2 = np.cos(np.deg2rad(region.cone_half_angle_deg)) ** 2
    roots(
        direction[0] ** 2 - cos2 * (direction @ direction),
        2 * (origin[0] * direction[0] - cos2 * (origin @ direction)),
        origin[0] ** 2 - cos2 * (origin @ origin),
    )
    cuts = np.unique([t for t in cuts if 0 <= t <= 1])
    samples = np.r_[cuts, (cuts[:-1] + cuts[1:]) / 2]
    return bool(np.any(contains(region, np.asarray(start) + samples[:, None] * direction)))


def section(region, x_axis, y_axis):
    """True central planar cut, not an opaque silhouette hiding the corridor."""
    plane = {(1, 0): "RI", (2, 0): "RC", (1, 2): "IC"}.get((x_axis, y_axis))
    if plane is None:
        return np.empty((0, 3))
    if 0 in (x_axis, y_axis):
        cone_angle = np.degrees(np.arctan2(-1 if y_axis == 0 else 0, -1 if x_axis == 0 else 0))
        start, end = cone_angle + region.cone_half_angle_deg, cone_angle + 360 - region.cone_half_angle_deg
    else:
        start, end = 0.0, 360.0
    return replace(
        region, kind="annular_sector", plane=plane, angle_min_deg=start, angle_max_deg=end
    ).sector_polygon_ric(samples=96)


def surface(region):
    """Display mesh of outer/inner spheres and the conical wall (no corridor cap)."""
    return _surface(
        region.inner_radius_km, region.outer_radius_km, region.cone_half_angle_deg, tuple(region.center_ric_km)
    )


@lru_cache(maxsize=16)
def _surface(inner_radius, outer_radius, half_angle, center):
    center = np.asarray(center)
    theta = np.linspace(np.deg2rad(half_angle), np.pi, 25)
    phi = np.linspace(0, 2 * np.pi, 49)
    directions = np.stack(
        np.broadcast_arrays(
            -np.cos(theta[:, None]), np.sin(theta[:, None]) * np.cos(phi), np.sin(theta[:, None]) * np.sin(phi)
        ),
        axis=-1,
    )
    inner = center + inner_radius * directions
    outer = center + outer_radius * directions
    faces, lines = [], []
    for sphere in (inner, outer):
        for row in range(len(theta) - 1):
            for col in range(len(phi) - 1):
                faces.append(sphere[[row, row, row + 1, row + 1], [col, col + 1, col + 1, col]])
        lines.extend(sphere[row] for row in range(0, len(theta), 4))
        lines.extend(sphere[:, col] for col in range(0, len(phi) - 1, 8))
    for col in range(len(phi) - 1):
        faces.append(np.array([inner[0, col], outer[0, col], outer[0, col + 1], inner[0, col + 1]]))
    lines.extend(np.array([inner[0, col], outer[0, col]]) for col in range(0, len(phi) - 1, 8))
    return faces, lines
