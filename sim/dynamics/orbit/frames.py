from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.epoch import gmst_angle_rad_from_jd

_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)
_MJD0 = 2400000.5
_J2000 = 2451545.0
_DAYSEC = 86400.0
_JULIAN_CENTURY_DAYS = 36525.0
_DEFAULT_TT_MINUS_UTC_S = 69.184


def eci_to_ecef_rotation(t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    if jd_utc_start is None:
        theta = EARTH_ROT_RATE_RAD_S * t_s
    else:
        theta = gmst_angle_rad_from_jd(float(jd_utc_start) + float(t_s) / 86400.0)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef(r_eci_km: np.ndarray, t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start) @ r_eci_km


def ecef_to_eci(r_ecef_km: np.ndarray, t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start).T @ r_ecef_km


def _rx(angle_rad: float) -> np.ndarray:
    s = math.sin(float(angle_rad))
    c = math.cos(float(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=float)


def _ry(angle_rad: float) -> np.ndarray:
    s = math.sin(float(angle_rad))
    c = math.cos(float(angle_rad))
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=float)


def _rz(angle_rad: float) -> np.ndarray:
    s = math.sin(float(angle_rad))
    c = math.cos(float(angle_rad))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


@lru_cache(maxsize=4)
def _load_eop_table(eop_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mjd = []
    xp_arcsec = []
    yp_arcsec = []
    dut1_s = []
    dat_s = []
    with Path(eop_path).expanduser().resolve().open("r", encoding="utf-8", errors="ignore") as f:
        in_data = False
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("NUM_OBSERVED_POINTS") or line.startswith("NUM_PREDICTED_POINTS"):
                in_data = True
                continue
            if not in_data or line.startswith("#") or line.startswith("VERSION") or line.startswith("UPDATED"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                mjd.append(float(parts[3]))
                xp_arcsec.append(float(parts[4]))
                yp_arcsec.append(float(parts[5]))
                dut1_s.append(float(parts[6]))
                dat_s.append(float(parts[12]))
            except ValueError:
                continue
    if not mjd:
        raise ValueError(f"No EOP records parsed from {eop_path}")
    return (
        np.array(mjd, dtype=float),
        np.array(xp_arcsec, dtype=float),
        np.array(yp_arcsec, dtype=float),
        np.array(dut1_s, dtype=float),
        np.array(dat_s, dtype=float),
    )


def _interp_eop(mjd_utc: float, eop_path: str) -> tuple[float, float, float, float]:
    mjd, xp_arcsec, yp_arcsec, dut1_s, dat_s = _load_eop_table(eop_path)
    x = float(mjd_utc)
    xp = float(np.interp(x, mjd, xp_arcsec))
    yp = float(np.interp(x, mjd, yp_arcsec))
    dut1 = float(np.interp(x, mjd, dut1_s))
    dat = float(np.interp(x, mjd, dat_s))
    return xp, yp, dut1, dat


def _mean_obliquity_iau1980_rad(jd_tt: float) -> float:
    t = (float(jd_tt) - _J2000) / _JULIAN_CENTURY_DAYS
    eps_arcsec = 84381.448 - 46.8150 * t - 0.00059 * (t**2) + 0.001813 * (t**3)
    return float(eps_arcsec * _ARCSEC_TO_RAD)


def _precession_iau1976_matrix(jd_tt: float) -> np.ndarray:
    t = (float(jd_tt) - _J2000) / _JULIAN_CENTURY_DAYS
    zeta = (2306.2181 * t + 0.30188 * (t**2) + 0.017998 * (t**3)) * _ARCSEC_TO_RAD
    theta = (2004.3109 * t - 0.42665 * (t**2) - 0.041833 * (t**3)) * _ARCSEC_TO_RAD
    z = (2306.2181 * t + 1.09468 * (t**2) + 0.018203 * (t**3)) * _ARCSEC_TO_RAD
    return _rz(-z) @ _ry(theta) @ _rz(-zeta)


@lru_cache(maxsize=1)
def _load_nut80_table() -> tuple[np.ndarray, np.ndarray]:
    path = Path(__file__).with_name("data") / "nut80.dat"
    raw = np.loadtxt(path, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 9:
        raise ValueError(f"Invalid IAU-80 nutation table at {path}")
    coeffs = raw[:, :5].astype(int)
    terms = raw[:, 5:9] * (0.0001 * _ARCSEC_TO_RAD)
    return coeffs, terms


def _fundamental_args_iau1980(ttt: float) -> tuple[float, float, float, float, float]:
    t = float(ttt)
    args_deg = (
        (((0.064 * t + 31.310) * t + 1717915922.6330) * t) / 3600.0 + 134.96298139,
        (((-0.012 * t - 0.577) * t + 129596581.2240) * t) / 3600.0 + 357.52772333,
        (((0.011 * t - 13.257) * t + 1739527263.1370) * t) / 3600.0 + 93.27191028,
        (((0.019 * t - 6.891) * t + 1602961601.3280) * t) / 3600.0 + 297.85036306,
        (((0.008 * t + 7.455) * t - 6962890.5390) * t) / 3600.0 + 125.04452222,
    )
    return tuple(math.radians(arg % 360.0) for arg in args_deg)


def _precession_iau1976_vallado_matrix(jd_tt: float) -> np.ndarray:
    ttt = (float(jd_tt) - _J2000) / _JULIAN_CENTURY_DAYS
    ttt2 = ttt * ttt
    ttt3 = ttt2 * ttt
    zeta = (2306.2181 * ttt + 0.30188 * ttt2 + 0.017998 * ttt3) * _ARCSEC_TO_RAD
    theta = (2004.3109 * ttt - 0.42665 * ttt2 - 0.041833 * ttt3) * _ARCSEC_TO_RAD
    z = (2306.2181 * ttt + 1.09468 * ttt2 + 0.018203 * ttt3) * _ARCSEC_TO_RAD
    coszeta = math.cos(zeta)
    sinzeta = math.sin(zeta)
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    cosz = math.cos(z)
    sinz = math.sin(z)
    return np.array(
        [
            [coszeta * costheta * cosz - sinzeta * sinz, coszeta * costheta * sinz + sinzeta * cosz, coszeta * sintheta],
            [-sinzeta * costheta * cosz - coszeta * sinz, -sinzeta * costheta * sinz + coszeta * cosz, -sinzeta * sintheta],
            [-sintheta * cosz, -sintheta * sinz, costheta],
        ],
        dtype=float,
    )


def _nutation_iau1980_vallado_matrix(
    jd_tt: float,
    *,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> tuple[float, float, float, float, np.ndarray]:
    ttt = (float(jd_tt) - _J2000) / _JULIAN_CENTURY_DAYS
    ttt2 = ttt * ttt
    ttt3 = ttt2 * ttt
    meaneps = ((-46.8150 * ttt - 0.00059 * ttt2 + 0.001813 * ttt3 + 84381.448) / 3600.0) % 360.0
    meaneps_rad = math.radians(meaneps)
    args = np.array(_fundamental_args_iau1980(ttt), dtype=float)
    coeffs, terms = _load_nut80_table()
    phase = coeffs @ args
    deltapsi = float(np.sum((terms[:, 0] + terms[:, 1] * ttt) * np.sin(phase)))
    deltaeps = float(np.sum((terms[:, 2] + terms[:, 3] * ttt) * np.cos(phase)))
    deltapsi = float(math.fmod(deltapsi + float(ddpsi_rad), 2.0 * math.pi))
    deltaeps = float(math.fmod(deltaeps + float(ddeps_rad), 2.0 * math.pi))
    trueeps = meaneps_rad + deltaeps
    omega = float(args[4])

    cospsi = math.cos(deltapsi)
    sinpsi = math.sin(deltapsi)
    coseps = math.cos(meaneps_rad)
    sineps = math.sin(meaneps_rad)
    costrueeps = math.cos(trueeps)
    sintrueeps = math.sin(trueeps)
    nut = np.array(
        [
            [cospsi, costrueeps * sinpsi, sintrueeps * sinpsi],
            [-coseps * sinpsi, costrueeps * coseps * cospsi + sintrueeps * sineps, sintrueeps * coseps * cospsi - sineps * costrueeps],
            [-sineps * sinpsi, costrueeps * sineps * cospsi - sintrueeps * coseps, sintrueeps * sineps * cospsi + costrueeps * coseps],
        ],
        dtype=float,
    )
    return deltapsi, trueeps, meaneps_rad, omega, nut


def teme_to_eci_matrix_vallado_iau80(
    jd_utc: float,
    *,
    tt_minus_utc_s: float = _DEFAULT_TT_MINUS_UTC_S,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> np.ndarray:
    """Return Vallado IAU-76/FK5 + IAU-80 TEME-to-ECI rotation.

    The implementation mirrors the MATLAB SGP4 package's ``teme2eci.m`` path.
    The default TT-UTC value matches 2024-era TLE validation cases; callers with
    EOP data can pass a different value and nutation corrections explicitly.
    """
    jd_tt = float(jd_utc) + float(tt_minus_utc_s) / _DAYSEC
    prec = _precession_iau1976_vallado_matrix(jd_tt)
    deltapsi, _trueeps, meaneps, _omega, nut = _nutation_iau1980_vallado_matrix(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    eqeg = math.fmod(deltapsi * math.cos(meaneps), 2.0 * math.pi)
    eqe = np.array(
        [
            [math.cos(eqeg), math.sin(eqeg), 0.0],
            [-math.sin(eqeg), math.cos(eqeg), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return prec @ nut @ eqe.T


def teme_to_eci_vallado_iau80(
    position_teme_km: np.ndarray,
    velocity_teme_km_s: np.ndarray,
    *,
    jd_utc: float,
    tt_minus_utc_s: float = _DEFAULT_TT_MINUS_UTC_S,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rot = teme_to_eci_matrix_vallado_iau80(
        jd_utc,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    return rot @ np.array(position_teme_km, dtype=float), rot @ np.array(velocity_teme_km_s, dtype=float)


def _short_nutation_1980_rad(jd_tt: float) -> tuple[float, float]:
    t = (float(jd_tt) - _J2000) / _JULIAN_CENTURY_DAYS
    mean_sun_long = math.radians((280.4665 + 36000.7698 * t) % 360.0)
    mean_moon_long = math.radians((218.3165 + 481267.8813 * t) % 360.0)
    omega = math.radians((125.04452 - 1934.136261 * t + 0.0020708 * (t**2) + (t**3) / 450000.0) % 360.0)
    dpsi_arcsec = (
        -17.20 * math.sin(omega)
        - 1.32 * math.sin(2.0 * mean_sun_long)
        - 0.23 * math.sin(2.0 * mean_moon_long)
        + 0.21 * math.sin(2.0 * omega)
    )
    deps_arcsec = (
        9.20 * math.cos(omega)
        + 0.57 * math.cos(2.0 * mean_sun_long)
        + 0.10 * math.cos(2.0 * mean_moon_long)
        - 0.09 * math.cos(2.0 * omega)
    )
    return float(dpsi_arcsec * _ARCSEC_TO_RAD), float(deps_arcsec * _ARCSEC_TO_RAD)


def _precession_nutation_matrix_approx(jd_tt: float) -> tuple[np.ndarray, float, float]:
    eps = _mean_obliquity_iau1980_rad(jd_tt)
    dpsi, deps = _short_nutation_1980_rad(jd_tt)
    precession = _precession_iau1976_matrix(jd_tt)
    nutation = _rx(-(eps + deps)) @ _rz(-dpsi) @ _rx(eps)
    return nutation @ precession, dpsi, eps + deps


def apparent_sidereal_time_hpop_like(jd_utc: float, eop_path: str | None = None) -> float:
    if not eop_path:
        return gmst_angle_rad_from_jd(float(jd_utc))
    mjd_utc = float(jd_utc) - _MJD0
    _xp_arcsec, _yp_arcsec, dut1_s, dat_s = _interp_eop(mjd_utc, eop_path)
    jd_ut1 = float(jd_utc) + dut1_s / _DAYSEC
    jd_tt = float(jd_utc) + (dat_s + 32.184) / _DAYSEC
    _rbpn, dpsi, true_obliquity = _precession_nutation_matrix_approx(jd_tt)
    return float((gmst_angle_rad_from_jd(jd_ut1) + dpsi * math.cos(true_obliquity)) % (2.0 * math.pi))


def precession_nutation_rotation_hpop_like(
    t_s: float,
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
) -> np.ndarray:
    if jd_utc_start is None or not eop_path:
        return np.eye(3, dtype=float)
    jd_utc = float(jd_utc_start) + float(t_s) / _DAYSEC
    mjd_utc = jd_utc - _MJD0
    _xp_arcsec, _yp_arcsec, _dut1_s, dat_s = _interp_eop(mjd_utc, eop_path)
    jd_tt = jd_utc + (dat_s + 32.184) / _DAYSEC
    rbpn, _dpsi, _true_obliquity = _precession_nutation_matrix_approx(jd_tt)
    return rbpn


def _polar_motion_matrix(xp_rad: float, yp_rad: float, sp_rad: float) -> np.ndarray:
    return _rz(sp_rad) @ _ry(-xp_rad) @ _rx(-yp_rad)


def eci_to_ecef_rotation_hpop_like(
    t_s: float,
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
) -> np.ndarray:
    if jd_utc_start is None or not eop_path:
        return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start)

    jd_utc = float(jd_utc_start) + float(t_s) / 86400.0
    mjd_utc = jd_utc - _MJD0
    xp_arcsec, yp_arcsec, dut1_s, dat_s = _interp_eop(mjd_utc, eop_path)
    jd_tt = jd_utc + (dat_s + 32.184) / _DAYSEC
    rbpn, _dpsi, _true_obliquity = _precession_nutation_matrix_approx(jd_tt)
    gast = apparent_sidereal_time_hpop_like(jd_utc, eop_path)
    sp = -47.0e-6 * ((jd_tt - _J2000) / _JULIAN_CENTURY_DAYS) * _ARCSEC_TO_RAD
    return _polar_motion_matrix(xp_arcsec * _ARCSEC_TO_RAD, yp_arcsec * _ARCSEC_TO_RAD, sp) @ _rz(gast) @ rbpn


def eci_to_ecef_harmonic(
    r_eci_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_model: str = "simple",
    eop_path: str | None = None,
) -> np.ndarray:
    model = str(frame_model).strip().lower()
    if model == "hpop_like":
        rot = eci_to_ecef_rotation_hpop_like(t_s, jd_utc_start=jd_utc_start, eop_path=eop_path)
        return rot @ np.array(r_eci_km, dtype=float)
    return eci_to_ecef(np.array(r_eci_km, dtype=float), t_s, jd_utc_start=jd_utc_start)


def ecef_to_eci_harmonic(
    r_ecef_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_model: str = "simple",
    eop_path: str | None = None,
) -> np.ndarray:
    model = str(frame_model).strip().lower()
    if model == "hpop_like":
        rot = eci_to_ecef_rotation_hpop_like(t_s, jd_utc_start=jd_utc_start, eop_path=eop_path)
        return rot.T @ np.array(r_ecef_km, dtype=float)
    return ecef_to_eci(np.array(r_ecef_km, dtype=float), t_s, jd_utc_start=jd_utc_start)
