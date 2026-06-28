from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit import _sdp4_equations
from sim.dynamics.orbit.sgp4 import SGP4State
from sim.dynamics.orbit.tle import TLEElements

_MINUTES_PER_DAY = 1440.0
_TWO_PI = 2.0 * math.pi
_VALLADO_EPOCH_JD_OFFSET = 2433281.5
_WGS72_CONSTANTS = _sdp4_equations.getgravconst("wgs72")


@dataclass
class _SDP4Record:
    """Mutable propagation record for the OEL SDP4 equation kernel."""


@dataclass
class SDP4Context:
    """Initialized scalar OGP-SDP4 propagation context.

    The underlying analytical deep-space equations carry secular and periodic
    state between calls, so reuse a context within one sequential propagation
    loop rather than sharing it across worker threads.
    """

    elements: TLEElements
    record: _SDP4Record
    period_min: float


def _sdp4_error_message(code: int, detail: str | None = None) -> str:
    prefix = f"OEL OGP-SDP4 propagation failed with error code {int(code)}"
    return f"{prefix}: {detail}" if detail else prefix


def _sdp4_validation_error(elements: TLEElements) -> str | None:
    mean_motion = float(elements.mean_motion_rev_per_day)
    if mean_motion <= 0.0:
        return "SDP4 mean motion must be positive."
    period_min = _MINUTES_PER_DAY / mean_motion
    if period_min < 225.0:
        return f"OEL OGP-SDP4 only supports deep-space TLEs with period >= 225 min; got {period_min:.3f} min."
    if float(elements.eccentricity) < 0.0 or float(elements.eccentricity) >= 1.0:
        return "SDP4 eccentricity must be in [0, 1)."
    return None


def _build_record(elements: TLEElements) -> _SDP4Record:
    record = _SDP4Record()
    xpdotp = _MINUTES_PER_DAY / _TWO_PI
    epoch = float(elements.epoch_jd_utc) - _VALLADO_EPOCH_JD_OFFSET
    no_kozai = float(elements.mean_motion_rev_per_day) / xpdotp
    ndot = float(elements.mean_motion_derivative_rev_per_day2) / (xpdotp * _MINUTES_PER_DAY)
    nddot = float(elements.mean_motion_second_derivative_rev_per_day3) / (
        xpdotp * _MINUTES_PER_DAY * _MINUTES_PER_DAY
    )
    _sdp4_equations.sgp4init(
        _WGS72_CONSTANTS,
        "i",
        str(elements.norad_number or "00000").zfill(5)[-5:],
        epoch,
        float(elements.bstar),
        ndot,
        nddot,
        float(elements.eccentricity),
        math.radians(float(elements.argp_deg)),
        math.radians(float(elements.inclination_deg)),
        math.radians(float(elements.mean_anomaly_deg)),
        no_kozai,
        math.radians(float(elements.raan_deg)),
        record,
    )
    return record


def sdp4_initialize(elements: TLEElements) -> SDP4Context:
    """Initialize a reusable scalar OGP-SDP4 context for one TLE."""

    error = _sdp4_validation_error(elements)
    if error:
        raise ValueError(error)
    record = _build_record(elements)
    if getattr(record, "method", "") != "d":
        raise ValueError("OEL OGP-SDP4 initialization did not select the deep-space SDP4 method.")
    return SDP4Context(
        elements=elements,
        record=record,
        period_min=_MINUTES_PER_DAY / float(elements.mean_motion_rev_per_day),
    )


def sdp4_propagate_teme_from_context(context: SDP4Context, tsince_min: float) -> SGP4State:
    """Propagate using a previously initialized OGP-SDP4 context."""

    if not math.isfinite(float(tsince_min)):
        return SGP4State(np.zeros(3), np.zeros(3), "SDP4 time offset must be finite.")
    try:
        position, velocity = _sdp4_equations.sgp4(context.record, float(tsince_min), _WGS72_CONSTANTS)
    except Exception as exc:
        return SGP4State(np.zeros(3), np.zeros(3), f"OEL OGP-SDP4 propagation raised {type(exc).__name__}: {exc}")

    error = int(getattr(context.record, "error", 0) or 0)
    if error:
        return SGP4State(
            np.zeros(3),
            np.zeros(3),
            _sdp4_error_message(error, getattr(context.record, "error_message", None)),
        )
    return SGP4State(
        position_teme_km=np.array(position, dtype=float),
        velocity_teme_km_s=np.array(velocity, dtype=float),
    )


def sdp4_propagate_teme(elements: TLEElements, tsince_min: float) -> SGP4State:
    """Propagate a deep-space TLE/mean-element product with OGP-SDP4.

    Runtime propagation is self-contained in OEL; external SDP4 packages are
    used only for validation fixture generation and parity checks.
    """

    try:
        context = sdp4_initialize(elements)
    except Exception as exc:
        return SGP4State(np.zeros(3), np.zeros(3), str(exc))
    return sdp4_propagate_teme_from_context(context, tsince_min)
