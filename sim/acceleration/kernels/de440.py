"""Optional compiled kernels for OEL's compact DE440 ephemeris."""

from __future__ import annotations

import math

import numpy as np

from sim.acceleration.optional import njit_or_identity


@njit_or_identity(cache=True, fastmath=False)
def _covering_row_kernel(starts: np.ndarray, ends: np.ndarray, jd_tdb: float) -> int:
    for index in range(starts.size):
        if starts[index] <= jd_tdb <= ends[index]:
            return index
    return -1


@njit_or_identity(cache=True, fastmath=False)
def _eval_light_body_kernel(
    jd_tdb: float,
    coeff_count: int,
    segments: int,
    span_days: float,
    starts: np.ndarray,
    ends: np.ndarray,
    x_coefficients: np.ndarray,
    y_coefficients: np.ndarray,
    z_coefficients: np.ndarray,
) -> tuple[np.ndarray, int]:
    row_index = _covering_row_kernel(starts, ends, jd_tdb)
    out = np.zeros(3, dtype=np.float64)
    if row_index < 0:
        return out, row_index

    t1 = float(starts[row_index])
    dt = float(jd_tdb - t1)
    segment_length = float(span_days) / float(segments)
    segment_index = int(math.floor(dt / segment_length))
    segment_index = max(0, min(int(segments) - 1, segment_index))
    ta = float(t1 + segment_length * segment_index)
    tb = ta + segment_length
    tau = (2.0 * jd_tdb - ta - tb) / (tb - ta)
    coefficient_start = segment_index * coeff_count

    f1x = 0.0
    f1y = 0.0
    f1z = 0.0
    f2x = 0.0
    f2y = 0.0
    f2z = 0.0
    for coefficient_index in range(int(coeff_count) - 1, 0, -1):
        old_f1x = f1x
        old_f1y = f1y
        old_f1z = f1z
        array_index = coefficient_start + coefficient_index
        f1x = 2.0 * tau * f1x - f2x + float(x_coefficients[row_index, array_index])
        f1y = 2.0 * tau * f1y - f2y + float(y_coefficients[row_index, array_index])
        f1z = 2.0 * tau * f1z - f2z + float(z_coefficients[row_index, array_index])
        f2x = old_f1x
        f2y = old_f1y
        f2z = old_f1z

    out[0] = 1.0e3 * (
        tau * f1x - f2x + float(x_coefficients[row_index, coefficient_start])
    )
    out[1] = 1.0e3 * (
        tau * f1y - f2y + float(y_coefficients[row_index, coefficient_start])
    )
    out[2] = 1.0e3 * (
        tau * f1z - f2z + float(z_coefficients[row_index, coefficient_start])
    )
    return out, row_index


@njit_or_identity(cache=True, fastmath=False)
def de440_light_core_kernel(
    jd_tdb: float,
    earthmoon_coeff_count: int,
    earthmoon_segments: int,
    earthmoon_span_days: float,
    earthmoon_starts: np.ndarray,
    earthmoon_ends: np.ndarray,
    earthmoon_x: np.ndarray,
    earthmoon_y: np.ndarray,
    earthmoon_z: np.ndarray,
    moon_coeff_count: int,
    moon_segments: int,
    moon_span_days: float,
    moon_starts: np.ndarray,
    moon_ends: np.ndarray,
    moon_x: np.ndarray,
    moon_y: np.ndarray,
    moon_z: np.ndarray,
    sun_coeff_count: int,
    sun_segments: int,
    sun_span_days: float,
    sun_starts: np.ndarray,
    sun_ends: np.ndarray,
    sun_x: np.ndarray,
    sun_y: np.ndarray,
    sun_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Evaluate the mandatory DE440 light bodies in one compiled boundary."""

    earthmoon, earthmoon_row = _eval_light_body_kernel(
        jd_tdb,
        earthmoon_coeff_count,
        earthmoon_segments,
        earthmoon_span_days,
        earthmoon_starts,
        earthmoon_ends,
        earthmoon_x,
        earthmoon_y,
        earthmoon_z,
    )
    moon, moon_row = _eval_light_body_kernel(
        jd_tdb,
        moon_coeff_count,
        moon_segments,
        moon_span_days,
        moon_starts,
        moon_ends,
        moon_x,
        moon_y,
        moon_z,
    )
    sun, sun_row = _eval_light_body_kernel(
        jd_tdb,
        sun_coeff_count,
        sun_segments,
        sun_span_days,
        sun_starts,
        sun_ends,
        sun_x,
        sun_y,
        sun_z,
    )
    return earthmoon, moon, sun, earthmoon_row, moon_row, sun_row


@njit_or_identity(cache=True, fastmath=False)
def de440_sun_moon_from_utc_kernel(
    jd_utc: float,
    tai_utc_s: float,
    earth_moon_mass_ratio: float,
    earthmoon_coeff_count: int,
    earthmoon_segments: int,
    earthmoon_span_days: float,
    earthmoon_starts: np.ndarray,
    earthmoon_ends: np.ndarray,
    earthmoon_x: np.ndarray,
    earthmoon_y: np.ndarray,
    earthmoon_z: np.ndarray,
    moon_coeff_count: int,
    moon_segments: int,
    moon_span_days: float,
    moon_starts: np.ndarray,
    moon_ends: np.ndarray,
    moon_x: np.ndarray,
    moon_y: np.ndarray,
    moon_z: np.ndarray,
    sun_coeff_count: int,
    sun_segments: int,
    sun_span_days: float,
    sun_starts: np.ndarray,
    sun_ends: np.ndarray,
    sun_x: np.ndarray,
    sun_y: np.ndarray,
    sun_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Convert UTC to TDB and evaluate the geocentric Sun/Moon pair."""

    mjd_utc = jd_utc - 2400000.5
    mjd_tt = mjd_utc + (32.184 + tai_utc_s) / 86400.0
    t_tt = (mjd_tt - 51544.5) / 36525.0
    mjd_tdb = mjd_tt + (
        0.001657 * np.sin(628.3076 * t_tt + 6.2401)
        + 0.000022 * np.sin(575.3385 * t_tt + 4.2970)
        + 0.000014 * np.sin(1256.6152 * t_tt + 6.1969)
        + 0.000005 * np.sin(606.9777 * t_tt + 4.0212)
        + 0.000005 * np.sin(52.9691 * t_tt + 0.4444)
        + 0.000002 * np.sin(21.3299 * t_tt + 5.5431)
        + 0.000010 * np.sin(628.3076 * t_tt + 4.2490)
    ) / 86400.0
    jd_tdb = mjd_tdb + 2400000.5
    earthmoon, moon, sun, earthmoon_row, moon_row, sun_row = de440_light_core_kernel(
        jd_tdb,
        earthmoon_coeff_count,
        earthmoon_segments,
        earthmoon_span_days,
        earthmoon_starts,
        earthmoon_ends,
        earthmoon_x,
        earthmoon_y,
        earthmoon_z,
        moon_coeff_count,
        moon_segments,
        moon_span_days,
        moon_starts,
        moon_ends,
        moon_x,
        moon_y,
        moon_z,
        sun_coeff_count,
        sun_segments,
        sun_span_days,
        sun_starts,
        sun_ends,
        sun_x,
        sun_y,
        sun_z,
    )
    earth = earthmoon - (1.0 / (1.0 + earth_moon_mass_ratio)) * moon
    return (-earth + sun) / 1.0e3, moon / 1.0e3, earthmoon_row, moon_row, sun_row
