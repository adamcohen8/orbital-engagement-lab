"""Precision force descriptors and exact stage metadata for the native ONP plan.

Frame/resource policy stays with the existing Python owners. Rust evaluates
spacecraft-dependent acceleration using the same authoritative force kernels.
"""
from __future__ import annotations

import math
import os
from dataclasses import fields

import numpy as np

from sim.dynamics.orbit.environment import MOON_MU_KM3_S2, SUN_MU_KM3_S2
from sim.dynamics.orbit.epoch import resolve_sun_moon_positions
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.pro_perturbations.earth_radiation import REFERENCE_JD_UTC, EarthRadiationPressure, _quadrature
from sim.pro_perturbations.ocean_tides import OceanTides
from sim.pro_perturbations.schwarzschild import SchwarzschildAcceleration
from sim.pro_perturbations.solid_earth_tides import SolidEarthTides

_CLASSES = {EarthRadiationPressure: 9, SchwarzschildAcceleration: 10, SolidEarthTides: 11, OceanTides: 12}


def precision_code(plugin):
    """Exact types only: subclasses retain their Python call contract."""
    code = _CLASSES.get(type(plugin))
    if code is not None and (code == 10 or type(plugin.frames) is FrameContext):
        return code
    return None


def precision_signature(plugin, ctx):
    code = precision_code(plugin)
    settings = tuple((f.name, getattr(plugin, f.name)) for f in fields(plugin) if f.init)
    tables = () if code != 12 else (id(plugin._factors), id(plugin._coefficients), plugin.coefficient_sha256)
    return code, settings, tables, float(ctx.area_m2)


def specifications(plugins, ctx):
    rows = []
    for plugin in plugins:
        code = precision_code(plugin)
        if code == 9:
            area = ctx.area_m2 if plugin.area_m2 is None else plugin.area_m2
            rows.append((code, [float(area), float(plugin.albedo), float(plugin.infrared)],
                         [array.ravel().tolist() for array in _quadrature(plugin.quadrature_order)]))
        elif code == 10:
            rows.append((code, [], []))
        elif code == 11:
            rows.append((code, [float(plugin.tide_system == "zero_tide"), float(plugin.pole_tide), SUN_MU_KM3_S2, MOON_MU_KM3_S2], []))
        elif code == 12:
            rows.append((code, [float(plugin.degree), float(plugin.pole_tide)],
                         [plugin._factors.ravel().tolist(), plugin._coefficients.ravel().tolist()]))
    return rows


def wrap_stages(base, plugins, env, ctx):
    selected = [plugin for plugin in plugins if precision_code(plugin) is not None]
    empty = np.eye(3).ravel().tolist() + [0.0] * 11 + [6378.137]
    # Cache only exact time-dependent frames. Resource signatures are checked
    # on each stage; spacecraft state, density and force outputs are never cached.
    cache = {}

    def stage(time, state):
        result = list(base(time, state))
        for plugin in selected:
            code = precision_code(plugin)
            if code == 10:
                result.extend(empty)
                continue
            if code == 9:
                area = ctx.area_m2 if plugin.area_m2 is None else plugin.area_m2
                if not all(math.isfinite(v) for v in (area, ctx.cr, ctx.mass_kg)) or area < 0 or ctx.cr < 0 or ctx.mass_kg <= 0:
                    raise ValueError("Earth radiation requires positive mass and nonnegative finite area/Cr.")
                if area == 0 or ctx.cr == 0:
                    result.extend(empty)
                    continue
            resource = None
            if plugin.frames.eop_path:
                path = os.path.expanduser(plugin.frames.eop_path)
                try:
                    stat = os.stat(path)
                    resource = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                except OSError:
                    resource = (path, None)
            key = (plugin.frames, float(time), resource)
            prepared = cache.get(key)
            if prepared is None:
                frame = plugin.frames.at(time)
                rotation = eci_to_ecef_rotation_context(time, frame)
                prepared = frame, rotation
                cache[key] = prepared
                while len(cache) > 8:
                    cache.pop(next(iter(cache)))
            frame, rotation = prepared
            sun = moon = np.zeros(3)
            if code in (9, 11):
                sun, moon = resolve_sun_moon_positions(dict(env, jd_utc_start=frame.jd_utc_start, _rust_numeric_backend="rust"), time)
                sun, moon = rotation @ sun, rotation @ moon
            jd = frame.jd_utc_start + float(time) / 86400
            elapsed = (frame.jd_utc_start - REFERENCE_JD_UTC) * 86400 + float(time) + frame.tt_minus_utc_s - 52.184
            result.extend(rotation.ravel().tolist() + sun.tolist() + moon.tolist() + [
                jd + frame.tt_minus_utc_s / 86400, jd + (frame.dut1_s or 0.0) / 86400,
                float(frame.xp_arcsec or 0.0), float(frame.yp_arcsec or 0.0), elapsed,
                float(env.get("spherical_harmonics_reference_radius_km", 6378.137)),
            ])
        return result

    return stage
