"""Resource-bound native assembly for built-in ONP integration stages.

Configuration and resource loading use the existing Python owners. Native
contexts assemble numeric stages; Python atmosphere/custom ephemeris owners
remain callbacks. A changed resource returns to the reference stage adapter.
"""
from __future__ import annotations

import os
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    FrameContext,
    _load_eop_table,
)


def _frame_spec(context: FrameContext | None, *, guard=False):
    if context is None:
        context = FrameContext()
    path = context.eop_path
    mode = 0
    if context.model == FRAME_MODEL_IAU76_80_EOP and context.eop_rotation_available:
        mode = 2 if context.eop_extrapolation == "hold" else 1
    table = [] if not path or (mode == 0 and not guard) else [row.tolist() for row in _load_eop_table(str(path))]
    values = [
        0.0 if context.xp_arcsec is None else float(context.xp_arcsec),
        0.0 if context.yp_arcsec is None else float(context.yp_arcsec),
        0.0 if context.dut1_s is None else float(context.dut1_s),
        context.tt_minus_utc_s - 32.184 if context.dat_s is None else float(context.dat_s),
        context.ddpsi_rad, context.ddeps_rad, 0.0,
    ]
    return mode, context.jd_utc_start, values, table, "" if not path else os.path.expanduser(str(path))


def try_native_stage_context(propagator, codes, scalars, env, native_env, native,
                             harmonic_frame, drag_frame, density_frame, de440_path,
                             fallback):
    from sim.dynamics.orbit import atmosphere as atmosphere_owner
    density_from_model = atmosphere_owner.density_from_model
    from sim.dynamics.orbit.epoch import resolve_sun_moon_positions, resolve_time_dependent_env
    from sim.dynamics.orbit.rust_force_plan import _resource_signature
    from sim.rust_environment_backend import _extension

    constructor = getattr(_extension(), "ONPStageContext", None)
    if constructor is None or env.get("_rust_stage_preparation_disabled", False):
        return None
    # Keep arbitrary hooks on the original adapter, including their environment
    # dictionaries and observable invocation order.
    if any(callable(value) for key, value in env.items() if not str(key).startswith("_")):
        return None
    paths = {os.path.expanduser(str(value)) for key, value in env.items()
             if str(key).endswith("eop_path") and value not in (None, "")}
    if de440_path is not None:
        paths.add(str(de440_path))
    # Relative EOP paths follow the current working directory on each call.
    # The reference adapter owns that uncommon dynamic-path policy.
    if (paths and os.name != "posix") or any(not os.path.isabs(path) for path in paths):
        return None
    signatures = {path: _resource_signature(path) for path in paths}
    def frozen(value):
        if value is None or isinstance(value, (str, bool, int, float, datetime)):
            return value
        if isinstance(value, Path):
            return str(value)
        if type(value) is FrameContext:
            return tuple((field.name, frozen(getattr(value, field.name))) for field in fields(value))
        if isinstance(value, np.ndarray):
            return value.shape, str(value.dtype), value.tobytes()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (tuple, list)):
            return tuple(frozen(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((str(key), frozen(item)) for key, item in value.items()))
        raise TypeError("unsupported native stage environment value")
    try:
        key = (tuple(codes), float(scalars[5]), id(native), tuple(sorted(signatures.items())),
               tuple(sorted((str(name), frozen(value)) for name, value in env.items()
                            if not str(name).startswith("_") and name not in {"world_truth", "sim_t_s", "object_id", "spherical_harmonics_terms"})))
    except TypeError:
        return None
    cache = getattr(propagator, "_rust_stage_context_cache", None)
    if cache is None:
        cache = propagator._rust_stage_context_cache = {}
    cached = cache.get(key)
    if cached is not None and not cached.using_fallback():
        return cached
    frames = []
    for prepared in (harmonic_frame, drag_frame, density_frame):
        context = None if prepared is None else prepared.context
        if context is not None and context.model == FRAME_MODEL_IAU76_80_EOP and context.jd_utc_start is None:
            return None
        frames.append(_frame_spec(context))
    jd = env.get("jd_utc_start")
    mode = str(env.get("ephemeris_mode", "analytic_enhanced")).lower()
    de440 = bool(de440_path is not None and de440_path.suffix.lower() == ".npz"
                 and native is not None and mode in {"de440", "hpop_de440", "de440_hpop"}
                 and (jd is not None or "jd_utc" in env)
                 and not any(key in env for key in (
                     "sun_pos_eci_km", "moon_pos_eci_km", "sun_ephemeris_time_s", "moon_ephemeris_time_s"
                 )))

    explicit_bodies = any(key in env for key in (
        "sun_pos_eci_km", "moon_pos_eci_km", "sun_ephemeris_time_s", "moon_ephemeris_time_s"))
    body_mode = 1 if de440 else 0
    if not explicit_bodies:
        if jd is None and "jd_utc" not in env:
            body_mode = 4
        elif mode in {"analytic_simple", "simple"}:
            body_mode = 2
        elif mode in {"analytic_enhanced", "enhanced", ""}:
            body_mode = 3

    if not any(code in codes for code in (3, 4, 5)):
        body_mode = 0
    body_path = env.get("de440_eop_path") or env.get("spherical_harmonics_eop_path") or env.get("drag_eop_path")
    body_context = FrameContext(model=FRAME_MODEL_IAU76_80_EOP, jd_utc_start=jd,
                               eop_path=body_path if body_mode == 1 else None, dat_s=37.0)
    # Ephemeris interpolation always has error-on-coverage policy, independent
    # of the frame's configured hold policy.
    frames.append(_frame_spec(body_context if body_mode == 1 else None))
    epoch_path = next((env.get(key) for key in (
        "eop_path", "de440_eop_path", "spherical_harmonics_eop_path", "drag_eop_path", "density_eop_path"
    ) if env.get(key) not in (None, "")), None)
    frames.append(_frame_spec(FrameContext(jd_utc_start=jd, eop_path=epoch_path if body_mode in (1, 2, 3) else None), guard=True))
    def bodies(time):
        resolved = resolve_time_dependent_env(native_env, float(time),
                                              cache_override=propagator._builtin_time_env_cache)
        result = tuple(np.asarray(value, dtype=float).tolist()
                       for value in resolve_sun_moon_positions(resolved, float(time)))
        while len(propagator._builtin_time_env_cache) > 8:
            propagator._builtin_time_env_cache.pop(next(iter(propagator._builtin_time_env_cache)))
        return result

    density = env.get("density_kg_m3")
    density_mode = 0 if density is not None else 2
    model = str(env.get("atmosphere_model", "")).lower()
    if 2 in codes and density is None and not model:
        return None
    if density is None and model == "exponential":
        density_mode = 1
    density_values = [
        0.0 if density is None else float(density),
        float(env.get("exponential_reference_density_kg_m3", 1.225)),
        float(env.get("exponential_reference_altitude_km", 0.0)),
        float(env.get("exponential_scale_height_km", 8.5)),
        float(env.get("exponential_ceiling_altitude_km", 1000.0)), float(scalars[5]),
    ]
    # DE440 vectors already have the same resource/epoch guards as the
    # reference resolver. Keep other ephemeris modes on their original route.
    density_body_inputs = bool(
        model == "nrlmsise00" and body_mode == 1
        and callable(getattr(atmosphere_owner, "_nrlmsise00_sun_longitude_rad", None))
        and env.get("nrlmsise00_lst_hr") is None
        and hasattr(constructor, "supports_density_body_inputs")
        and constructor.supports_density_body_inputs()
    )

    density_frame_inputs = bool(
        density_body_inputs and density_frame is not None
        and hasattr(constructor, "supports_density_frame_inputs")
        and constructor.supports_density_frame_inputs()
    )

    def atmosphere(time, raw_state, coordinates, prepared_bodies=None, prepared_rotation=None):
        # Per-call scratch: no state-derived coordinates survive the callback.
        stage_env = dict(native_env)
        stage_env["_native_density_coordinates"] = coordinates
        if prepared_rotation is not None:
            stage_env["_native_density_rotation"] = np.asarray(prepared_rotation, dtype=float).reshape(3, 3)
        if prepared_bodies is not None:
            stage_env["sun_pos_eci_km"], stage_env["moon_pos_eci_km"] = prepared_bodies
        return float(density_from_model(model, np.asarray(raw_state, dtype=float)[:3],
                                       float(time), env=stage_env))

    body_options = {"density_body_inputs": True} if density_body_inputs else {}
    if density_frame_inputs:
        body_options["density_frame_inputs"] = True
    result = constructor(native, frames, codes, density_mode, density_values,
                         str(env.get("geodetic_model", "")).lower() == "wgs84",
                         atmosphere, bodies, body_mode, jd, env.get("jd_utc"),
                         env.get("de440_tai_utc_s"), sorted(paths), fallback, **body_options)
    # Reject a snapshot assembled across a concurrent rewrite.
    if any(_resource_signature(path) != signature for path, signature in signatures.items()):
        return None
    cache[key] = result
    while len(cache) > 8:
        cache.pop(next(iter(cache)))
    return result
