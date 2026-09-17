"""Finite display meshes for existing trainer constraints; no scoring logic."""

from itertools import product

import numpy as np

from sim.game import spherical_corridor
from sim.game.training_geometry import _plane_axes


def box(lower, upper):
    corners = np.array(list(product(*zip(lower, upper))), dtype=float)
    indices = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    faces = [corners[list(index)] for index in indices]
    return faces, [np.vstack((face, face[0])) for face in faces]


def sphere(center, radius):
    theta = np.linspace(0, np.pi, 17)
    phi = np.linspace(0, 2 * np.pi, 33)
    directions = np.stack(
        np.broadcast_arrays(
            np.cos(theta[:, None]), np.sin(theta[:, None]) * np.cos(phi), np.sin(theta[:, None]) * np.sin(phi)
        ),
        axis=-1,
    )
    rows = np.asarray(center) + radius * directions
    faces = [rows[[i, i, i + 1, i + 1], [j, j + 1, j + 1, j]] for i in range(16) for j in range(32)]
    lines = [rows[i] for i in range(0, 17, 4)] + [rows[:, j] for j in range(0, 32, 8)]
    return faces, lines


def extrusion(lower, upper):
    faces = [lower, upper]
    faces.extend(
        np.array([lower[i], lower[(i + 1) % len(lower)], upper[(i + 1) % len(lower)], upper[i]])
        for i in range(len(lower))
    )
    lines = [np.vstack((cap, cap[0])) for cap in (lower, upper)]
    lines.extend(np.array([lower[i], upper[i]]) for i in range(0, len(lower), max(1, len(lower) // 8)))
    return faces, lines


def region_mesh(region):
    if region.kind == "spherical_corridor":
        return spherical_corridor.surface(region)
    if region.kind == "sphere":
        return sphere(region.center_ric_km, region.radius_km)
    if region.kind == "box":
        if not np.all(np.isfinite([region.min_ric_km, region.max_ric_km])):
            raise ValueError("3D constraint boxes require finite boundaries.")
        return box(region.min_ric_km, region.max_ric_km)
    if region.kind == "annular_sector":
        polygon = region.sector_polygon_ric(samples=48)
        axis = _plane_axes(region.plane)[2]
        half = region.max_abs_out_of_plane_km
    elif region.kind == "cylinder":
        axis = {"R": 0, "I": 1, "C": 2}[region.axis]
        transverse = [i for i in range(3) if i != axis]
        angle = np.linspace(0, 2 * np.pi, 49)[:-1]
        polygon = np.tile(region.center_ric_km, (len(angle), 1))
        polygon[:, transverse[0]] += region.radius_km * np.cos(angle)
        polygon[:, transverse[1]] += region.radius_km * np.sin(angle)
        half = region.height_km / 2
    else:
        raise ValueError(f"Unsupported 3D constraint: {region.kind}")
    if half is None:
        raise ValueError("3D constraint extrusion requires a finite thickness.")
    lower, upper = polygon.copy(), polygon.copy()
    lower[:, axis] -= half
    upper[:, axis] += half
    return extrusion(lower, upper)
