"""IERS 2010 solid Earth tidal gravity (Sun/Moon), in OEL km/s units.

Degree 2/3 anelastic response, induced degree 4, frequency corrections,
permanent-tide convention, and optional solid pole tide. No ocean tides.
Equations: IERS Conventions 2010, chapter 6, sections 6.2 and 6.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM, MOON_MU_KM3_S2, SUN_MU_KM3_S2
from sim.dynamics.orbit.epoch import resolve_sun_moon_positions
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.pro_perturbations._iers2010_tables import K20, K21, K22

_ARCSEC = math.pi / 648000.0
_LOVE = (
    (2, 0, 0.30190, 0.0, -0.00089),
    (2, 1, 0.29830, -0.00144, -0.00080),
    (2, 2, 0.30102, -0.00130, -0.00057),
    (3, 0, 0.093, 0.0, 0.0),
    (3, 1, 0.093, 0.0, 0.0),
    (3, 2, 0.093, 0.0, 0.0),
    (3, 3, 0.094, 0.0, 0.0),
)
_DELAUNAY = (
    (134.96340251 * 3600, 1717915923.2178, 31.8792, 0.051635, -0.00024470),
    (357.52910918 * 3600, 129596581.0481, -0.5532, 0.000136, -0.00001149),
    (93.27209062 * 3600, 1739527262.8478, -12.7512, -0.001037, 0.00000417),
    (297.85019547 * 3600, 1602961601.2090, -6.3706, 0.006593, -0.00003169),
    (125.04455501 * 3600, -6962890.5431, 7.4722, 0.007702, -0.00005939),
)


def _poly(coefficients, x):
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * x + coefficient
    return result


def _vector(value, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.linalg.norm(result) == 0:
        raise ValueError(f"{label} must be a finite nonzero 3-vector.")
    return result


@lru_cache(maxsize=20)
def _harmonic_normalization(degree):
    """State-independent normalization; retain the scalar arithmetic order."""
    return tuple(
        tuple(
            math.sqrt((1 if m == 0 else 2) * (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m))
            for m in range(n + 1)
        )
        for n in range(degree + 1)
    )


@lru_cache(maxsize=20)
def _normalization_array(degree):
    result = np.zeros((degree + 1, degree + 1))
    for n, row in enumerate(_harmonic_normalization(degree)):
        result[n, : n + 1] = row
    result.setflags(write=False)
    return result


def _solid_harmonics(position, degree=4, *, gradients=True, use_acceleration=False):
    """Unnormalized regular solid harmonics and Cartesian gradients, no CS phase.

    Polynomial recurrence avoids longitude/pole singularities and subtraction
    of the much larger central gravity when evaluating tiny tidal forces.
    """
    if use_acceleration:
        from sim.pro_perturbations._kernels import solid_harmonics_kernel

        return solid_harmonics_kernel(position, _normalization_array(degree), gradients)
    from sim.pro_perturbations._numeric import solid_harmonics_numeric

    return solid_harmonics_numeric(position, _normalization_array(degree), gradients)


def tidal_coefficients(
    *,
    sun_fixed_km,
    moon_fixed_km,
    mu_km3_s2,
    radius_km,
    jd_tt,
    jd_ut1,
    tide_system,
    pole_xy_arcsec=None,
    sun_mu=SUN_MU_KM3_S2,
    moon_mu=MOON_MU_KM3_S2,
    use_acceleration=False,
):
    """Return fully normalized additive C/S arrays, not a complete gravity field."""
    if tide_system not in {"tide_free", "zero_tide"}:
        raise ValueError("solid_earth_tides.tide_system must be tide_free or zero_tide.")
    if not all(math.isfinite(v) and v > 0 for v in (mu_km3_s2, radius_km, sun_mu, moon_mu)):
        raise ValueError("Tidal radii and gravitational parameters must be positive and finite.")
    if not all(math.isfinite(v) for v in (jd_tt, jd_ut1)):
        raise ValueError("Tidal epochs must be finite.")
    c, s = np.zeros((5, 5)), np.zeros((5, 5))
    for label, body, gm in (("Sun", sun_fixed_km, sun_mu), ("Moon", moon_fixed_km, moon_mu)):
        body = _vector(body, label)
        distance = float(np.linalg.norm(body))
        if distance <= radius_km:
            raise ValueError(f"{label} must lie outside Earth for solid tides.")
        h, _ = _solid_harmonics(body / distance, gradients=False, use_acceleration=use_acceleration)
        for n, m, kr, ki, kp in _LOVE:
            q = gm / mu_km3_s2 * (radius_km / distance) ** (n + 1) / (2 * n + 1) * h[n, m]
            c[n, m] += kr * q.real + ki * q.imag
            s[n, m] += kr * q.imag - ki * q.real
            if n == 2:
                c[4, m] += kp * q.real
                s[4, m] += kp * q.imag
    t = (jd_tt - 2451545.0) / 36525.0
    delaunay = np.array([_poly(row, t) * _ARCSEC for row in _DELAUNAY])
    # IERS 2010 GMST = ERA + polynomial (Table 5.2e), gamma = GMST + pi.
    era = 2 * math.pi * (0.7790572732640 + 1.00273781191135448 * (jd_ut1 - 2451545.0))
    gamma = (
        era + _poly((0.014506, 4612.156534, 1.3915817, -0.00000044, -0.000029956, -0.0000000368), t) * _ARCSEC + math.pi
    )
    for order, table in ((0, K20), (1, K21), (2, K22)):
        for multiplier, arguments, ip, op in table:
            phase = multiplier * gamma - float(np.dot(arguments, delaunay))
            sn, cs = math.sin(phase), math.cos(phase)
            if order == 0:
                c[2, 0] += 1e-12 * (ip * cs - op * sn)
            elif order == 1:
                c[2, 1] += 1e-12 * (ip * sn + op * cs)
                s[2, 1] += 1e-12 * (ip * cs - op * sn)
            else:
                c[2, 2] += 1e-12 * ip * cs
                s[2, 2] -= 1e-12 * ip * sn
    if tide_system == "zero_tide":
        c[2, 0] -= 4.4228e-8 * -0.31460 * 0.30190
    if pole_xy_arcsec is not None:
        xp, yp = pole_xy_arcsec
        if not math.isfinite(xp) or not math.isfinite(yp):
            raise ValueError("Pole coordinates must be finite arcseconds.")
        years = (jd_tt - 2451545.0) / 365.25
        if jd_tt <= 2455197.5:  # 2010-01-01 TT; original IERS 2010 mean pole.
            mean_x = _poly((55.974, 1.8243, 0.18413, 0.007024), years) / 1000
            mean_y = _poly((346.346, 1.7896, -0.10729, -0.000908), years) / 1000
        else:
            mean_x = (23.513 + 7.6141 * years) / 1000
            mean_y = (358.891 - 0.6287 * years) / 1000
        m1, m2 = xp - mean_x, mean_y - yp
        c[2, 1] -= 1.333e-9 * (m1 + 0.0115 * m2)
        s[2, 1] -= 1.333e-9 * (m2 - 0.0115 * m1)
    return c, s


def tidal_acceleration_fixed(position_km, c, s, *, mu_km3_s2, radius_km, use_acceleration=False):
    """Analytic gradient of tidal potential only, including at either pole."""
    r = _vector(position_km, "Spacecraft position")
    distance = float(np.linalg.norm(r))
    degree = len(c) - 1
    if use_acceleration:
        from sim.pro_perturbations._kernels import tidal_acceleration_kernel

        return tidal_acceleration_kernel(r, c, s, mu_km3_s2, radius_km, _normalization_array(degree))
    h, g = _solid_harmonics(r / distance, degree)
    acceleration = np.zeros(3)
    for n in range(2, degree + 1):
        scale = mu_km3_s2 / distance**2 * (radius_km / distance) ** n
        for m in range(n + 1):
            gradient = g[n, m] - (2 * n + 1) * h[n, m] * r / distance
            acceleration += scale * (c[n, m] * gradient.real + s[n, m] * gradient.imag)
    return acceleration


@dataclass(frozen=True)
class SolidEarthTides:
    """Callable OEL acceleration plugin; uses the shared scenario frame context."""

    frames: FrameContext
    tide_system: str
    pole_tide: bool = True
    acceleration_mode: str = "off"
    _accelerated: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(self):
        from sim.acceleration.settings import acceleration_settings_from_mode

        object.__setattr__(self, "_accelerated", acceleration_settings_from_mode(self.acceleration_mode).enabled)
        if self.tide_system not in {"tide_free", "zero_tide"}:
            raise ValueError("solid_earth_tides requires an explicit supported tide_system.")
        frame = self.frames
        if frame.jd_utc_start is None or not frame.eop_rotation_available:
            raise ValueError("solid_earth_tides requires an absolute epoch and EOP-backed scenario frames.")
        if not frame.eop_path and any(getattr(frame, k) is None for k in ("dut1_s", "dat_s", "xp_arcsec", "yp_arcsec")):
            raise ValueError("solid_earth_tides requires EOP data or explicit dut1_s, dat_s, xp_arcsec, yp_arcsec.")

    def __call__(self, t_s, state, env, ctx):
        frame = self.frames.at(t_s)
        rotation = eci_to_ecef_rotation_context(t_s, frame)
        ephemeris_env = dict(env, jd_utc_start=frame.jd_utc_start)
        sun, moon = resolve_sun_moon_positions(ephemeris_env, t_s)
        jd_utc = frame.jd_utc_start + float(t_s) / 86400
        radius = float(env.get("spherical_harmonics_reference_radius_km", EARTH_RADIUS_KM))
        if env.get("_rust_numeric_backend") == "rust":
            from sim.rust_environment_backend import try_solid_tides_acceleration

            native = try_solid_tides_acceleration(
                rotation @ state[:3],
                rotation @ sun,
                rotation @ moon,
                mu_km3_s2=ctx.mu_km3_s2,
                radius_km=radius,
                jd_tt=jd_utc + frame.tt_minus_utc_s / 86400,
                jd_ut1=jd_utc + frame.dut1_s / 86400,
                tide_system=self.tide_system,
                pole_xy_arcsec=(frame.xp_arcsec, frame.yp_arcsec) if self.pole_tide else None,
                sun_mu=SUN_MU_KM3_S2,
                moon_mu=MOON_MU_KM3_S2,
            )
            if native is not None:
                return rotation.T @ native
        c, s = tidal_coefficients(
            sun_fixed_km=rotation @ sun,
            moon_fixed_km=rotation @ moon,
            mu_km3_s2=ctx.mu_km3_s2,
            radius_km=radius,
            jd_tt=jd_utc + frame.tt_minus_utc_s / 86400,
            jd_ut1=jd_utc + frame.dut1_s / 86400,
            tide_system=self.tide_system,
            use_acceleration=self._accelerated,
            pole_xy_arcsec=(frame.xp_arcsec, frame.yp_arcsec) if self.pole_tide else None,
        )
        if env.get("_rust_numeric_backend") == "rust":
            from sim.rust_environment_backend import try_tidal_acceleration

            native = try_tidal_acceleration(
                rotation @ state[:3],
                c,
                s,
                mu_km3_s2=ctx.mu_km3_s2,
                radius_km=radius,
            )
            if native is not None:
                return rotation.T @ native
        return rotation.T @ tidal_acceleration_fixed(
            rotation @ state[:3], c, s, mu_km3_s2=ctx.mu_km3_s2, radius_km=radius, use_acceleration=self._accelerated
        )
