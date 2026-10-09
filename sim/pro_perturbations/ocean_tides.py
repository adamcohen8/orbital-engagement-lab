"""FES normalized ocean tidal gravity and IERS 2010 degree-2 ocean pole tide.

The coefficient input is FES Cnm-Snm (10^-11), not height/amplitude/phase data.
Doodson phase conventions follow IERS 2010 6.15 and Orekit OceanTidesWave.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.pro_perturbations.solid_earth_tides import _ARCSEC, _DELAUNAY, _poly, tidal_acceleration_fixed


def load_fes_coefficients(path, degree=6, order=6):
    """Read a local FES normalized C/S file once; reject malformed/duplicate data."""
    if (
        any(isinstance(v, bool) or not isinstance(v, int) for v in (degree, order))
        or not 2 <= degree <= 20
        or not 0 <= order <= degree
    ):
        raise ValueError("Ocean tides require integer 2 <= degree <= 20 and 0 <= order <= degree.")
    data = Path(path).read_bytes()
    text = data.decode("utf-8")
    if (
        not re.search(r"\bnormalized\b", text[:600], re.IGNORECASE)
        or "10^-11" not in text[:600]
        or "DelC+" not in text[:600]
    ):
        raise ValueError("Expected normalized FES Cnm-Snm coefficients with units 10^-11.")
    waves, seen = {}, set()
    started = False
    maximum_degree = maximum_order = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or not re.fullmatch(r"\d{2,3}\.\d{3}", fields[0]):
            if line.strip() and (started or fields[0][0].isdigit()):
                raise ValueError(f"Malformed FES row {line_number}.")
            continue
        started = True
        try:
            if len(fields) != 8:
                raise ValueError()
            doodson = int(fields[0].replace(".", ""))
            n, m = int(fields[2]), int(fields[3])
            values = np.array([float(v) for v in fields[4:]]) * 1e-11
            if not 0 <= m <= n <= 100 or not np.all(np.isfinite(values)) or (doodson, n, m) in seen:
                raise ValueError()
        except ValueError as exc:
            raise ValueError(f"Invalid or duplicate FES row {line_number}.") from exc
        seen.add((doodson, n, m))
        maximum_degree, maximum_order = max(maximum_degree, n), max(maximum_order, m)
        if 2 <= n <= degree and m <= order:
            wave = waves.setdefault(doodson, np.zeros((degree + 1, degree + 1, 4)))
            wave[n, m] = values
    if not waves or maximum_degree < degree or maximum_order < order:
        raise ValueError("FES file does not cover the requested degree/order.")
    factors, coefficients = [], []
    for doodson, wave in sorted(waves.items()):
        tau = doodson // 100000 % 10
        s, h, p, node, ps = [(doodson // power % 10) - 5 for power in (10000, 1000, 100, 10, 1)]
        factors.append([tau, -p, -ps, -tau + s + h + p + ps, -h - ps, -tau + s + h + p - node + ps])
        coefficients.append(wave)
    factors, coefficients = np.asarray(factors), np.asarray(coefficients)
    factors.setflags(write=False)
    coefficients.setflags(write=False)
    return factors, coefficients, hashlib.sha256(data).hexdigest()


def ocean_pole_coefficients(jd_tt, xp_arcsec, yp_arcsec):
    """IERS 2010 6.5 C21/S21, original piecewise conventional mean pole."""
    if not all(math.isfinite(v) for v in (jd_tt, xp_arcsec, yp_arcsec)):
        raise ValueError("Ocean pole tide requires finite TT epoch and pole coordinates.")
    years = (jd_tt - 2451545.0) / 365.25
    if jd_tt <= 2455197.5:
        mean_x = _poly((55.974, 1.8243, 0.18413, 0.007024), years) / 1000
        mean_y = _poly((346.346, 1.7896, -0.10729, -0.000908), years) / 1000
    else:
        mean_x = (23.513 + 7.6141 * years) / 1000
        mean_y = (358.891 - 0.6287 * years) / 1000
    m1, m2 = xp_arcsec - mean_x, mean_y - yp_arcsec
    return -2.1778e-10 * (m1 - 0.01724 * m2), -1.7232e-10 * (m2 - 0.03365 * m1)


def ocean_coefficients(factors, coefficients, jd_tt, jd_ut1, pole_xy_arcsec=None):
    """Fully normalized additive Stokes coefficients, with no central term."""
    if not all(math.isfinite(v) for v in (jd_tt, jd_ut1)):
        raise ValueError("Ocean tides require finite TT and UT1 epochs.")
    t = (jd_tt - 2451545.0) / 36525
    delaunay = [_poly(row, t) * _ARCSEC for row in _DELAUNAY]
    gamma = 2 * math.pi * (0.7790572732640 + 1.00273781191135448 * (jd_ut1 - 2451545.0))
    gamma += _poly((0.014506, 4612.156534, 1.3915817, -0.00000044, -0.000029956, -0.0000000368), t) * _ARCSEC + math.pi
    phase = factors @ np.array([gamma, *delaunay])
    cosine, sine = np.cos(phase)[:, None, None], np.sin(phase)[:, None, None]
    cp, sp, cm, sm = np.moveaxis(coefficients, -1, 0)
    c = np.sum((cp + cm) * cosine + (sp + sm) * sine, axis=0)
    s = np.sum((sp - sm) * cosine - (cp - cm) * sine, axis=0)
    if pole_xy_arcsec is not None:
        pc, ps = ocean_pole_coefficients(jd_tt, *pole_xy_arcsec)
        c[2, 1] += pc
        s[2, 1] += ps
    return c, s


@dataclass(frozen=True)
class OceanTides:
    frames: FrameContext
    coeff_path: str
    degree: int = 6
    order: int = 6
    pole_tide: bool = True
    acceleration_mode: str = "off"
    _accelerated: bool = field(init=False, repr=False, compare=False, default=False)
    coefficient_sha256: str = field(init=False)
    _factors: np.ndarray = field(init=False, repr=False, compare=False, hash=False)
    _coefficients: np.ndarray = field(init=False, repr=False, compare=False, hash=False)
    _native_context: object = field(init=False, repr=False, compare=False, hash=False, default=None)
    _native_context_ready: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(self):
        from sim.acceleration.settings import acceleration_settings_from_mode

        object.__setattr__(self, "_accelerated", acceleration_settings_from_mode(self.acceleration_mode).enabled)
        frame = self.frames
        if frame.jd_utc_start is None or not frame.eop_rotation_available:
            raise ValueError("ocean_tides requires an absolute epoch and EOP-backed scenario frames.")
        if not frame.eop_path and any(getattr(frame, k) is None for k in ("dut1_s", "dat_s", "xp_arcsec", "yp_arcsec")):
            raise ValueError("ocean_tides requires EOP data or explicit dut1_s, dat_s, xp_arcsec, yp_arcsec.")
        if self.pole_tide and self.order < 1:
            raise ValueError("Ocean pole tide requires order >= 1.")
        factors, coefficients, digest = load_fes_coefficients(self.coeff_path, self.degree, self.order)
        # An immutable backing binds the prepared native context to exactly the
        # same loaded values; even setflags(write=True) cannot change the table.
        object.__setattr__(self, "_factors", np.frombuffer(factors.tobytes(), dtype=factors.dtype).reshape(factors.shape))
        object.__setattr__(self, "_coefficients", np.frombuffer(coefficients.tobytes(), dtype=coefficients.dtype).reshape(coefficients.shape))
        object.__setattr__(self, "coefficient_sha256", digest)

    def __getstate__(self):
        # Native contexts are a disposable cache, not persisted model evidence.
        state = self.__dict__.copy()
        state["_native_context"] = None
        state["_native_context_ready"] = False
        return state

    def __setstate__(self, state):
        for name, value in state.items():
            if name in {"_factors", "_coefficients"}:
                value = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_native_context", None)
        object.__setattr__(self, "_native_context_ready", False)

    def __call__(self, t_s, state, env, ctx):
        frame = self.frames.at(t_s)
        rotation = eci_to_ecef_rotation_context(t_s, frame)
        jd = frame.jd_utc_start + float(t_s) / 86400
        jd_tt = jd + frame.tt_minus_utc_s / 86400
        jd_ut1 = jd + frame.dut1_s / 86400
        pole_xy = (frame.xp_arcsec, frame.yp_arcsec) if self.pole_tide else None
        radius = float(env.get("spherical_harmonics_reference_radius_km", EARTH_RADIUS_KM))
        position_fixed = rotation @ state[:3]
        if env.get("_rust_numeric_backend") == "rust":
            from sim.rust_environment_backend import (
                try_context_ocean_tides_acceleration,
                try_create_ocean_tides_context,
            )

            if not self._native_context_ready:
                object.__setattr__(self, "_native_context", try_create_ocean_tides_context(self._factors, self._coefficients))
                object.__setattr__(self, "_native_context_ready", True)
            if self._native_context is not None:
                native = try_context_ocean_tides_acceleration(
                    self._native_context, position_fixed, jd_tt=jd_tt, jd_ut1=jd_ut1,
                    pole_xy_arcsec=pole_xy, mu_km3_s2=ctx.mu_km3_s2, radius_km=radius,
                )
                if native is not None:
                    return rotation.T @ native
        c, s = ocean_coefficients(self._factors, self._coefficients, jd_tt, jd_ut1, pole_xy)
        if env.get("_rust_numeric_backend") == "rust":
            from sim.rust_environment_backend import try_tidal_acceleration

            native = try_tidal_acceleration(
                position_fixed, c, s, mu_km3_s2=ctx.mu_km3_s2, radius_km=radius,
            )
            if native is not None:
                return rotation.T @ native
        return rotation.T @ tidal_acceleration_fixed(
            position_fixed, c, s, mu_km3_s2=ctx.mu_km3_s2, radius_km=radius, use_acceleration=self._accelerated
        )
