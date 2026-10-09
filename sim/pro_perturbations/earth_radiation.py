"""Spherical-Earth Lambertian albedo/IR with Knocke seasonal climatology.

Integrates radiance over the actual apparent Earth disk, not the full Earth
hemisphere. Returns separate pressure vectors, without direct solar pressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from sim.dynamics.orbit.epoch import resolve_sun_moon_positions
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context

EARTH_RADIUS_KM = 6378.137
AU_KM = 149597870.691  # Orekit JPL_SSD astronomical unit, for comparison.
SOLAR_PRESSURE_PA = 4.5606e-6
REFERENCE_JD_UTC = 2444960.5  # 1981-12-22 00:00 UTC, TAI-UTC = 20 s.


def knocke_coefficients(sin_latitude, elapsed_s):
    """Knocke latitude/annual albedo and IR emissivity coefficients."""
    latitude = np.asarray(sin_latitude, dtype=float)
    seasonal = math.cos(2 * math.pi * float(elapsed_s) / (365.25 * 86400))
    p2 = 0.5 * (3 * latitude**2 - 1)
    return 0.34 + 0.10 * seasonal * latitude + 0.29 * p2, 0.68 - 0.07 * seasonal * latitude - 0.18 * p2


@lru_cache(maxsize=8)
def _quadrature(order):
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 8 <= order <= 128:
        raise ValueError("earth_radiation.quadrature_order must be an integer from 8 to 128.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    azimuth = (np.arange(4 * order) + 0.5) * (2 * math.pi / (4 * order))
    result = (nodes, weights, np.cos(azimuth), np.sin(azimuth))
    for array in result:
        array.setflags(write=False)
    return result


def radiation_pressure_components(
    position_km,
    sun_position_km,
    elapsed_s,
    *,
    order=32,
    radius_km=EARTH_RADIUS_KM,
    uniform_coefficients=None,
    use_acceleration=False,
):
    """Return (albedo, infrared) vectors in Pa in the input Earth-equatorial frame.

    Rays intersect the near surface of the sphere. Integrating radiance dOmega
    incorporates the surface emission cosine and inverse-square dilution exactly
    in the solid-angle measure. Uniform coefficients are an analytical-test hook.
    """
    r = np.asarray(position_km, dtype=float)
    sun = np.asarray(sun_position_km, dtype=float)
    if r.shape != (3,) or sun.shape != (3,) or not np.all(np.isfinite([r, sun])):
        raise ValueError("Earth radiation positions must be finite 3-vectors.")
    distance, sun_distance = float(np.linalg.norm(r)), float(np.linalg.norm(sun))
    if not math.isfinite(radius_km) or radius_km <= 0 or distance <= radius_km or sun_distance <= radius_km:
        raise ValueError("Earth radiation requires spacecraft and Sun outside the positive-radius Earth sphere.")
    if not math.isfinite(elapsed_s):
        raise ValueError("Earth radiation epoch must be finite.")
    nodes, weights, cp, sp = _quadrature(order)
    if use_acceleration and uniform_coefficients is None:
        from sim.pro_perturbations._kernels import radiation_kernel

        return radiation_kernel(r, sun, elapsed_s, radius_km, nodes, weights, cp, sp)
    radial = r / distance
    east = np.cross([0.0, 0.0, 1.0], radial)
    if np.linalg.norm(east) < 1e-12:
        east = np.array([0.0, 1.0, 0.0])
    else:
        east /= np.linalg.norm(east)
    north = np.cross(radial, east)
    cos_limb = math.sqrt(max(0.0, 1 - (radius_km / distance) ** 2))
    width = (radius_km / distance) ** 2 / (1 + cos_limb)
    cosine = 1 - width * (1 - nodes) / 2
    sine = np.sqrt(np.maximum(0, 1 - cosine * cosine))
    direction = cosine[:, None, None] * radial + sine[:, None, None] * (
        cp[None, :, None] * east + sp[None, :, None] * north
    )
    root = np.sqrt(np.maximum(0, radius_km**2 - distance**2 * (1 - cosine * cosine)))
    ray_length = (distance**2 - radius_km**2) / (distance * cosine + root)
    normals = (r - ray_length[:, None, None] * direction) / radius_km
    albedo, emissivity = knocke_coefficients(normals[:, :, 2], elapsed_s)
    if uniform_coefficients is not None:
        if len(uniform_coefficients) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in uniform_coefficients):
            raise ValueError("Uniform albedo/emissivity must lie in [0,1].")
        albedo, emissivity = uniform_coefficients
    incidence = np.maximum(0, np.sum(normals * (sun / sun_distance), axis=2))
    pressure = SOLAR_PRESSURE_PA * (AU_KM / sun_distance) ** 2
    weights = weights[:, None, None] * (width / 2) * (2 * math.pi / (4 * order))
    weighted_direction = weights * direction
    albedo_pressure = np.sum(weighted_direction * (albedo * incidence)[:, :, None], axis=(0, 1)) * pressure / math.pi
    infrared_pressure = (
        np.sum(weighted_direction * np.broadcast_to(emissivity, incidence.shape)[:, :, None], axis=(0, 1))
        * pressure
        / (4 * math.pi)
    )
    return albedo_pressure, infrared_pressure


@dataclass(frozen=True)
class EarthRadiationPressure:
    frames: FrameContext
    albedo: bool = True
    infrared: bool = True
    quadrature_order: int = 32
    area_m2: float | None = None
    acceleration_mode: str = "off"
    _accelerated: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(self):
        from sim.acceleration.settings import acceleration_settings_from_mode

        object.__setattr__(self, "_accelerated", acceleration_settings_from_mode(self.acceleration_mode).enabled)
        _quadrature(self.quadrature_order)
        if self.frames.jd_utc_start is None or not math.isfinite(self.frames.jd_utc_start):
            raise ValueError("earth_radiation requires an absolute initial epoch.")
        if not self.albedo and not self.infrared:
            raise ValueError("earth_radiation requires albedo and/or infrared.")
        if self.area_m2 is not None and (not math.isfinite(self.area_m2) or self.area_m2 < 0):
            raise ValueError("Earth radiation area must be finite and nonnegative.")

    def components(self, t_s, state, env, ctx):
        area = ctx.area_m2 if self.area_m2 is None else self.area_m2
        if not all(math.isfinite(v) for v in (area, ctx.cr, ctx.mass_kg)) or area < 0 or ctx.cr < 0 or ctx.mass_kg <= 0:
            raise ValueError("Earth radiation requires positive mass and nonnegative finite area/Cr.")
        if area == 0 or ctx.cr == 0:
            return np.zeros(3), np.zeros(3)
        frame = self.frames.at(t_s)
        rotation = eci_to_ecef_rotation_context(t_s, frame)
        sun, _ = resolve_sun_moon_positions(dict(env, jd_utc_start=frame.jd_utc_start), t_s)
        elapsed = (frame.jd_utc_start - REFERENCE_JD_UTC) * 86400 + float(t_s) + frame.tt_minus_utc_s - 52.184
        if env.get("_rust_numeric_backend") == "rust":
            from sim.rust_environment_backend import try_earth_radiation_components

            native = try_earth_radiation_components(
                rotation @ state[:3],
                rotation @ sun,
                elapsed,
                order=self.quadrature_order,
                include_albedo=self.albedo,
                include_infrared=self.infrared,
            )
            if native is not None:
                a, ir = native
                scale = area * ctx.cr / ctx.mass_kg / 1000
                return rotation.T @ a * scale, rotation.T @ ir * scale
        a, ir = radiation_pressure_components(
            rotation @ state[:3],
            rotation @ sun,
            elapsed,
            order=self.quadrature_order,
            use_acceleration=self._accelerated,
        )
        scale = area * ctx.cr / ctx.mass_kg / 1000
        return rotation.T @ a * scale, rotation.T @ ir * scale

    def __call__(self, t_s, state, env, ctx):
        a, ir = self.components(t_s, state, env, ctx)
        return (a if self.albedo else np.zeros(3)) + (ir if self.infrared else np.zeros(3))

    def components_batch(self, times_s, states, env, ctx):
        """Evaluate each epoch with its own frame and Sun geometry."""
        times = np.asarray(times_s, dtype=float)
        values = np.asarray(states, dtype=float)
        if times.ndim != 1 or values.shape != (times.size, 6):
            raise ValueError("Earth radiation batch requires times (N,) and states (N, 6).")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise ValueError("Earth radiation batch times and states must be finite.")
        area = ctx.area_m2 if self.area_m2 is None else self.area_m2
        if not all(math.isfinite(v) for v in (area, ctx.cr, ctx.mass_kg)) or area < 0 or ctx.cr < 0 or ctx.mass_kg <= 0:
            raise ValueError("Earth radiation requires positive mass and nonnegative finite area/Cr.")
        if area == 0 or ctx.cr == 0 or times.size == 0:
            return np.zeros((times.size, 3)), np.zeros((times.size, 3))
        rotations = np.empty((times.size, 3, 3))
        positions = np.empty((times.size, 3))
        suns = np.empty_like(positions)
        elapsed_times = np.empty_like(times)
        for index, (time, state) in enumerate(zip(times, values, strict=True)):
            frame = self.frames.at(time)
            rotation = eci_to_ecef_rotation_context(time, frame)
            sun, _ = resolve_sun_moon_positions(dict(env, jd_utc_start=frame.jd_utc_start), time)
            rotations[index] = rotation
            positions[index] = rotation @ state[:3]
            suns[index] = rotation @ sun
            elapsed_times[index] = (frame.jd_utc_start - REFERENCE_JD_UTC) * 86400 + float(time) + frame.tt_minus_utc_s - 52.184
        components = None
        if env.get("_rust_numeric_backend") == "rust":
            from sim.rust_environment_backend import try_earth_radiation_components_batch

            components = try_earth_radiation_components_batch(
                positions, suns, elapsed_times, order=self.quadrature_order,
                include_albedo=self.albedo, include_infrared=self.infrared,
            )
        if components is None:
            rows = [radiation_pressure_components(position, sun, elapsed,
                    order=self.quadrature_order, use_acceleration=self._accelerated)
                    for position, sun, elapsed in zip(positions, suns, elapsed_times, strict=True)]
            components = (
                np.asarray([pair[0] for pair in rows]) if self.albedo else np.zeros((times.size, 3)),
                np.asarray([pair[1] for pair in rows]) if self.infrared else np.zeros((times.size, 3)),
            )
        scale = area * ctx.cr / ctx.mass_kg / 1000
        return tuple(np.asarray([rotation.T @ row * scale for rotation, row in zip(rotations, rows, strict=True)])
                     for rows in components)
