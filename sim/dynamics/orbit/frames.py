from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.epoch import gmst_angle_rad_from_jd

_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)
_MJD0 = 2400000.5
_J2000 = 2451545.0
_DAYSEC = 86400.0
_JULIAN_CENTURY_DAYS = 36525.0
_DEFAULT_TT_MINUS_UTC_S = 69.184

FRAME_MODEL_SIMPLE_GMST = "simple_gmst"
FRAME_MODEL_IAU76_80_EOP = "iau76_80_eop"

_FRAME_MODEL_ALIASES = {
    "": FRAME_MODEL_SIMPLE_GMST,
    "simple": FRAME_MODEL_SIMPLE_GMST,
    "simple_gmst": FRAME_MODEL_SIMPLE_GMST,
    "simple_earth_rotation": FRAME_MODEL_SIMPLE_GMST,
    "gmst": FRAME_MODEL_SIMPLE_GMST,
    "inertial_z": FRAME_MODEL_SIMPLE_GMST,
    "hpop_like": FRAME_MODEL_IAU76_80_EOP,
    "hpop": FRAME_MODEL_IAU76_80_EOP,
    "iau76_80_eop": FRAME_MODEL_IAU76_80_EOP,
    "iau76_fk5_iau80_eop": FRAME_MODEL_IAU76_80_EOP,
}


@dataclass(frozen=True)
class FrameContext:
    """Scenario-level frame and time-scale provenance for Earth-fixed transforms."""

    model: str = FRAME_MODEL_SIMPLE_GMST
    jd_utc_start: float | None = None
    eop_path: str | None = None
    eop_extrapolation: str = "error"
    time_scale_model: str = "utc_only"
    tt_minus_utc_s: float = _DEFAULT_TT_MINUS_UTC_S
    dut1_s: float | None = None
    xp_arcsec: float | None = None
    yp_arcsec: float | None = None
    dat_s: float | None = None
    ddpsi_rad: float = 0.0
    ddeps_rad: float = 0.0
    source: str = "scenario"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", normalize_frame_model(self.model))
        if self.eop_path in ("",):
            object.__setattr__(self, "eop_path", None)
        policy = str(self.eop_extrapolation or "error").strip().lower()
        if policy not in {"error", "hold"}:
            raise ValueError("eop_extrapolation must be 'error' or 'hold'.")
        object.__setattr__(self, "eop_extrapolation", policy)

    @property
    def legacy_frame_model(self) -> str:
        return "hpop_like" if self.model == FRAME_MODEL_IAU76_80_EOP else "simple"

    @property
    def eop_rotation_available(self) -> bool:
        return bool(
            self.model == FRAME_MODEL_IAU76_80_EOP
            and self.jd_utc_start is not None
            and (self.eop_path or self.has_manual_eop)
        )

    @property
    def has_manual_eop(self) -> bool:
        return bool(
            self.dut1_s is not None
            or self.xp_arcsec is not None
            or self.yp_arcsec is not None
            or self.dat_s is not None
            or float(self.ddpsi_rad) != 0.0
            or float(self.ddeps_rad) != 0.0
        )

    def at(self, t_s: float) -> FrameContext:
        if self.jd_utc_start is None or self.model != FRAME_MODEL_IAU76_80_EOP or not self.eop_path:
            return self
        jd_utc = float(self.jd_utc_start) + float(t_s) / _DAYSEC
        xp_arcsec, yp_arcsec, dut1_s, dat_s = _interp_eop(
            jd_utc - _MJD0,
            self.eop_path,
            extrapolation=self.eop_extrapolation,
        )
        return FrameContext(
            model=self.model,
            jd_utc_start=self.jd_utc_start,
            eop_path=self.eop_path,
            eop_extrapolation=self.eop_extrapolation,
            time_scale_model="eop_utc_ut1_tt",
            tt_minus_utc_s=float(dat_s) + 32.184,
            dut1_s=float(dut1_s),
            xp_arcsec=float(xp_arcsec),
            yp_arcsec=float(yp_arcsec),
            dat_s=float(dat_s),
            ddpsi_rad=self.ddpsi_rad,
            ddeps_rad=self.ddeps_rad,
            source=self.source,
        )

    def metadata(self, *, sample_t_s: float = 0.0) -> dict[str, Any]:
        sampled = self.at(sample_t_s)
        data = asdict(sampled)
        data["legacy_frame_model"] = sampled.legacy_frame_model
        data["sample_t_s"] = float(sample_t_s)
        data["polar_motion_applied"] = sampled.eop_rotation_available
        data["nutation_corrections_applied"] = bool(
            sampled.eop_rotation_available and (float(sampled.ddpsi_rad) != 0.0 or float(sampled.ddeps_rad) != 0.0)
        )
        return data


def normalize_frame_model(model: Any) -> str:
    key = str(model or "").strip().lower().replace("-", "_")
    if key in _FRAME_MODEL_ALIASES:
        return _FRAME_MODEL_ALIASES[key]
    choices = ", ".join(sorted(set(_FRAME_MODEL_ALIASES) - {""}))
    raise ValueError(f"Unsupported frame model {model!r}; expected one of: {choices}.")


def _resolve_frame_path(raw_path: Any) -> str | None:
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return str(path.resolve())


def frame_context_from_mapping(
    frames: dict[str, Any] | None,
    *,
    jd_utc_start: float | None = None,
    source: str = "scenario",
) -> FrameContext:
    data = dict(frames or {})
    if jd_utc_start is None and data.get("jd_utc_start") is not None:
        jd_utc_start = float(data["jd_utc_start"])
    model = normalize_frame_model(data.get("model", data.get("frame_model", FRAME_MODEL_SIMPLE_GMST)))
    eop_path = _resolve_frame_path(data.get("eop_path"))
    tt_minus_raw = data.get("tt_minus_utc_s", _DEFAULT_TT_MINUS_UTC_S)
    tt_minus_utc_s = _DEFAULT_TT_MINUS_UTC_S if tt_minus_raw is None else float(tt_minus_raw)
    if data.get("dat_s") is not None:
        tt_minus_utc_s = float(data["dat_s"]) + 32.184
    return FrameContext(
        model=model,
        jd_utc_start=jd_utc_start,
        eop_path=eop_path,
        eop_extrapolation=str(data.get("eop_extrapolation", "error") or "error"),
        time_scale_model=str(
            data.get(
                "time_scale_model",
                "eop_utc_ut1_tt" if model == FRAME_MODEL_IAU76_80_EOP and eop_path else "utc_only",
            )
        ),
        tt_minus_utc_s=tt_minus_utc_s,
        dut1_s=None if data.get("dut1_s") is None else float(data["dut1_s"]),
        xp_arcsec=None if data.get("xp_arcsec") is None else float(data["xp_arcsec"]),
        yp_arcsec=None if data.get("yp_arcsec") is None else float(data["yp_arcsec"]),
        dat_s=None if data.get("dat_s") is None else float(data["dat_s"]),
        ddpsi_rad=float(data.get("ddpsi_rad", 0.0) or 0.0),
        ddeps_rad=float(data.get("ddeps_rad", 0.0) or 0.0),
        source=source,
    )


def frame_context_from_environment(env: dict[str, Any] | None) -> FrameContext:
    data = dict(env or {})
    model = data.get("frame_model", data.get("spherical_harmonics_frame_model", data.get("drag_frame_model", "simple")))
    eop_path = data.get("eop_path", data.get("spherical_harmonics_eop_path", data.get("drag_eop_path")))
    frames = {
        "model": model,
        "eop_path": eop_path,
        "eop_extrapolation": data.get("eop_extrapolation", "error"),
        "tt_minus_utc_s": data.get("tt_minus_utc_s", _DEFAULT_TT_MINUS_UTC_S),
        "dut1_s": data.get("dut1_s"),
        "xp_arcsec": data.get("xp_arcsec"),
        "yp_arcsec": data.get("yp_arcsec"),
        "dat_s": data.get("dat_s"),
        "ddpsi_rad": data.get("ddpsi_rad", 0.0),
        "ddeps_rad": data.get("ddeps_rad", 0.0),
    }
    return frame_context_from_mapping(
        frames,
        jd_utc_start=None if data.get("jd_utc_start") is None else float(data["jd_utc_start"]),
        source="environment",
    )


@lru_cache(maxsize=8192)
def _eci_to_ecef_rotation_components(
    t_s: float,
    jd_utc_start: float | None,
) -> tuple[np.floating, np.floating]:
    if jd_utc_start is None:
        theta = EARTH_ROT_RATE_RAD_S * t_s
    else:
        theta = gmst_angle_rad_from_jd(float(jd_utc_start) + float(t_s) / 86400.0)
    return np.cos(theta), np.sin(theta)


def eci_to_ecef_rotation(t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    c, s = _eci_to_ecef_rotation_components(
        float(t_s),
        None if jd_utc_start is None else float(jd_utc_start),
    )
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


@lru_cache(maxsize=8192)
def _interp_eop(
    mjd_utc: float,
    eop_path: str,
    *,
    extrapolation: str = "error",
) -> tuple[float, float, float, float]:
    mjd, xp_arcsec, yp_arcsec, dut1_s, dat_s = _load_eop_table(eop_path)
    x = float(mjd_utc)
    policy = str(extrapolation or "error").strip().lower()
    if policy not in {"error", "hold"}:
        raise ValueError("EOP extrapolation policy must be 'error' or 'hold'.")
    if (x < float(mjd[0]) or x > float(mjd[-1])) and policy == "error":
        raise ValueError(
            f"Requested MJD {x:.9f} is outside EOP coverage "
            f"[{float(mjd[0]):.9f}, {float(mjd[-1]):.9f}] for {eop_path}. "
            "Set eop_extrapolation='hold' only when endpoint holding is intentional."
        )
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
            [
                coszeta * costheta * cosz - sinzeta * sinz,
                coszeta * costheta * sinz + sinzeta * cosz,
                coszeta * sintheta,
            ],
            [
                -sinzeta * costheta * cosz - coszeta * sinz,
                -sinzeta * costheta * sinz + coszeta * cosz,
                -sinzeta * sintheta,
            ],
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
            [
                -coseps * sinpsi,
                costrueeps * coseps * cospsi + sintrueeps * sineps,
                sintrueeps * coseps * cospsi - sineps * costrueeps,
            ],
            [
                -sineps * sinpsi,
                costrueeps * sineps * cospsi - sintrueeps * coseps,
                sintrueeps * sineps * cospsi + costrueeps * coseps,
            ],
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


def _precession_nutation_matrix_iau76_80(
    jd_tt: float,
    *,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    dpsi, true_eps, mean_eps, _omega, _nut_vallado = _nutation_iau1980_vallado_matrix(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    precession = _precession_iau1976_matrix(jd_tt)
    nutation = _rx(-true_eps) @ _rz(-dpsi) @ _rx(mean_eps)
    return nutation @ precession, dpsi, true_eps


@lru_cache(maxsize=8192)
def _precession_nutation_matrix_approx(
    jd_tt: float,
    *,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Compatibility alias for the full IAU-76/FK5 + IAU-80 reduction.

    The returned matrix is treated as immutable by the internal frame consumers.
    Caching lets density/local-solar-time and body-fixed gravity share the exact
    same epoch reduction when both are active.
    """

    return _precession_nutation_matrix_iau76_80(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )


def apparent_sidereal_time_hpop_like(
    jd_utc: float,
    eop_path: str | None = None,
    *,
    dut1_s: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> float:
    has_manual_eop = any(value is not None for value in (dut1_s, dat_s, tt_minus_utc_s)) or (
        float(ddpsi_rad) != 0.0 or float(ddeps_rad) != 0.0
    )
    if not eop_path and not has_manual_eop:
        return gmst_angle_rad_from_jd(float(jd_utc))
    if eop_path:
        mjd_utc = float(jd_utc) - _MJD0
        _xp_arcsec, _yp_arcsec, dut1_s, dat_s = _interp_eop(mjd_utc, eop_path)
    else:
        dut1_s = 0.0 if dut1_s is None else float(dut1_s)
        if dat_s is None:
            dat_s = (float(tt_minus_utc_s) if tt_minus_utc_s is not None else _DEFAULT_TT_MINUS_UTC_S) - 32.184
        else:
            dat_s = float(dat_s)
    jd_ut1 = float(jd_utc) + float(dut1_s) / _DAYSEC
    jd_tt = float(jd_utc) + (float(dat_s) + 32.184) / _DAYSEC
    _rbpn, dpsi, true_obliquity = _precession_nutation_matrix_approx(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    return float((gmst_angle_rad_from_jd(jd_ut1) + dpsi * math.cos(true_obliquity)) % (2.0 * math.pi))


def precession_nutation_rotation_hpop_like(
    t_s: float,
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
    *,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> np.ndarray:
    if jd_utc_start is None or not eop_path:
        return np.eye(3, dtype=float)
    jd_utc = float(jd_utc_start) + float(t_s) / _DAYSEC
    mjd_utc = jd_utc - _MJD0
    _xp_arcsec, _yp_arcsec, _dut1_s, dat_s = _interp_eop(mjd_utc, eop_path)
    jd_tt = jd_utc + (dat_s + 32.184) / _DAYSEC
    rbpn, _dpsi, _true_obliquity = _precession_nutation_matrix_approx(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    return rbpn


def _polar_motion_matrix(xp_rad: float, yp_rad: float, sp_rad: float) -> np.ndarray:
    return _rz(sp_rad) @ _ry(-xp_rad) @ _rx(-yp_rad)


def _eci_to_ecef_rotation_hpop_like_uncached(
    t_s: float,
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
    *,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> np.ndarray:
    has_manual_eop = any(value is not None for value in (dut1_s, xp_arcsec, yp_arcsec, dat_s)) or (
        float(ddpsi_rad) != 0.0 or float(ddeps_rad) != 0.0
    )
    if jd_utc_start is None or (not eop_path and not has_manual_eop):
        return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start)

    jd_utc = float(jd_utc_start) + float(t_s) / 86400.0
    if eop_path:
        mjd_utc = jd_utc - _MJD0
        xp_arcsec, yp_arcsec, dut1_s, dat_s = _interp_eop(
            mjd_utc,
            eop_path,
            extrapolation=eop_extrapolation,
        )
    else:
        xp_arcsec = 0.0 if xp_arcsec is None else float(xp_arcsec)
        yp_arcsec = 0.0 if yp_arcsec is None else float(yp_arcsec)
        dut1_s = 0.0 if dut1_s is None else float(dut1_s)
        if dat_s is None:
            dat_s = (float(tt_minus_utc_s) if tt_minus_utc_s is not None else _DEFAULT_TT_MINUS_UTC_S) - 32.184
        else:
            dat_s = float(dat_s)
    jd_tt = jd_utc + (float(dat_s) + 32.184) / _DAYSEC
    rbpn, dpsi, true_obliquity = _precession_nutation_matrix_approx(
        jd_tt,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
    )
    jd_ut1 = jd_utc + float(dut1_s) / _DAYSEC
    gast = float((gmst_angle_rad_from_jd(jd_ut1) + dpsi * math.cos(true_obliquity)) % (2.0 * math.pi))
    sp = -47.0e-6 * ((jd_tt - _J2000) / _JULIAN_CENTURY_DAYS) * _ARCSEC_TO_RAD
    return _polar_motion_matrix(float(xp_arcsec) * _ARCSEC_TO_RAD, float(yp_arcsec) * _ARCSEC_TO_RAD, sp) @ _rz(gast) @ rbpn


@lru_cache(maxsize=8192)
def _cached_eci_to_ecef_rotation_hpop_like(
    t_s: float,
    jd_utc_start: float | None,
    eop_path: str | None,
    dut1_s: float | None,
    xp_arcsec: float | None,
    yp_arcsec: float | None,
    dat_s: float | None,
    tt_minus_utc_s: float | None,
    ddpsi_rad: float,
    ddeps_rad: float,
    eop_extrapolation: str,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    rotation = _eci_to_ecef_rotation_hpop_like_uncached(
        t_s,
        jd_utc_start=jd_utc_start,
        eop_path=eop_path,
        dut1_s=dut1_s,
        xp_arcsec=xp_arcsec,
        yp_arcsec=yp_arcsec,
        dat_s=dat_s,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
        eop_extrapolation=eop_extrapolation,
    )
    flattened = rotation.ravel()
    return (
        float(flattened[0]),
        float(flattened[1]),
        float(flattened[2]),
        float(flattened[3]),
        float(flattened[4]),
        float(flattened[5]),
        float(flattened[6]),
        float(flattened[7]),
        float(flattened[8]),
    )


def eci_to_ecef_rotation_hpop_like(
    t_s: float,
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
    *,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> np.ndarray:
    """Return a fresh ECI-to-ECEF rotation, reusing exact frame inputs."""

    values = _cached_eci_to_ecef_rotation_hpop_like(
        float(t_s),
        None if jd_utc_start is None else float(jd_utc_start),
        None if eop_path is None else str(eop_path),
        None if dut1_s is None else float(dut1_s),
        None if xp_arcsec is None else float(xp_arcsec),
        None if yp_arcsec is None else float(yp_arcsec),
        None if dat_s is None else float(dat_s),
        None if tt_minus_utc_s is None else float(tt_minus_utc_s),
        float(ddpsi_rad),
        float(ddeps_rad),
        str(eop_extrapolation),
    )
    return np.array(values, dtype=float).reshape(3, 3)


def eci_to_ecef_rotation_context(t_s: float, context: FrameContext) -> np.ndarray:
    ctx = context.at(t_s)
    if ctx.model == FRAME_MODEL_IAU76_80_EOP:
        if ctx.eop_path and ctx.jd_utc_start is None:
            raise ValueError("IAU76/80 EOP frame rotation requires simulator.initial_jd_utc when eop_path is set.")
        if ctx.has_manual_eop and ctx.jd_utc_start is None:
            raise ValueError("IAU76/80 manual EOP frame rotation requires simulator.initial_jd_utc.")
        return eci_to_ecef_rotation_hpop_like(
            t_s,
            jd_utc_start=ctx.jd_utc_start,
            eop_path=ctx.eop_path,
            dut1_s=ctx.dut1_s,
            xp_arcsec=ctx.xp_arcsec,
            yp_arcsec=ctx.yp_arcsec,
            dat_s=ctx.dat_s,
            tt_minus_utc_s=ctx.tt_minus_utc_s,
            ddpsi_rad=ctx.ddpsi_rad,
            ddeps_rad=ctx.ddeps_rad,
            eop_extrapolation=ctx.eop_extrapolation,
        )
    return eci_to_ecef_rotation(t_s, jd_utc_start=ctx.jd_utc_start)


def rotation_between(
    source_frame: str,
    target_frame: str,
    *,
    t_s: float,
    context: FrameContext,
) -> np.ndarray:
    source = str(source_frame or "").strip().lower()
    target = str(target_frame or "").strip().lower()
    if source == target:
        return np.eye(3, dtype=float)
    if source == "eci" and target in {"ecef", "itrf"}:
        return eci_to_ecef_rotation_context(t_s, context)
    if source in {"ecef", "itrf"} and target == "eci":
        return eci_to_ecef_rotation_context(t_s, context).T
    raise ValueError(f"Unsupported frame rotation {source_frame!r} -> {target_frame!r}.")


def transform_position(
    position_km: np.ndarray,
    source_frame: str,
    target_frame: str,
    *,
    t_s: float,
    context: FrameContext,
) -> np.ndarray:
    return rotation_between(source_frame, target_frame, t_s=t_s, context=context) @ np.array(position_km, dtype=float)


def transform_state(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    source_frame: str,
    target_frame: str,
    *,
    t_s: float,
    context: FrameContext,
) -> tuple[np.ndarray, np.ndarray]:
    source = str(source_frame or "").strip().lower()
    target = str(target_frame or "").strip().lower()
    pos = np.array(position_km, dtype=float).reshape(3)
    vel = np.array(velocity_km_s, dtype=float).reshape(3)
    if source == target:
        return pos.copy(), vel.copy()
    if source == "eci" and target in {"ecef", "itrf"}:
        rot = eci_to_ecef_rotation_context(t_s, context)
        rot_dot = _eci_to_ecef_rotation_derivative_context(t_s, context)
        return rot @ pos, rot @ vel + rot_dot @ pos
    if source in {"ecef", "itrf"} and target == "eci":
        rot = eci_to_ecef_rotation_context(t_s, context)
        rot_dot = _eci_to_ecef_rotation_derivative_context(t_s, context)
        pos_eci = rot.T @ pos
        vel_eci = rot.T @ (vel - rot_dot @ pos_eci)
        return pos_eci, vel_eci
    raise ValueError(f"Unsupported frame state transform {source_frame!r} -> {target_frame!r}.")


def _eci_to_ecef_rotation_derivative_context(t_s: float, context: FrameContext) -> np.ndarray:
    # Julian dates near the present epoch have a floating-point spacing of
    # roughly tens of microseconds.  A 0.01 s two-point difference therefore
    # amplified epoch quantization into mm/s-to-m/s station-velocity errors.
    # A symmetric five-point stencil over 30 s suppresses that roundoff while
    # retaining fourth-order accuracy for Earth rotation and the much slower
    # precession/nutation/EOP terms.
    step_s = 30.0
    t = float(t_s)
    return (
        -eci_to_ecef_rotation_context(t + 2.0 * step_s, context)
        + 8.0 * eci_to_ecef_rotation_context(t + step_s, context)
        - 8.0 * eci_to_ecef_rotation_context(t - step_s, context)
        + eci_to_ecef_rotation_context(t - 2.0 * step_s, context)
    ) / (
        12.0 * step_s
    )


def eci_to_ecef_harmonic(
    r_eci_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_model: str = "simple",
    eop_path: str | None = None,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> np.ndarray:
    r_eci = (
        r_eci_km
        if isinstance(r_eci_km, np.ndarray) and r_eci_km.dtype == np.float64
        else np.asarray(r_eci_km, dtype=float)
    )
    model = normalize_frame_model(frame_model)
    if model == FRAME_MODEL_IAU76_80_EOP:
        rot = eci_to_ecef_rotation_hpop_like(
            t_s,
            jd_utc_start=jd_utc_start,
            eop_path=eop_path,
            dut1_s=dut1_s,
            xp_arcsec=xp_arcsec,
            yp_arcsec=yp_arcsec,
            dat_s=dat_s,
            tt_minus_utc_s=tt_minus_utc_s,
            ddpsi_rad=ddpsi_rad,
            ddeps_rad=ddeps_rad,
        )
        return rot @ r_eci
    return eci_to_ecef(r_eci, t_s, jd_utc_start=jd_utc_start)


def ecef_to_eci_harmonic(
    r_ecef_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_model: str = "simple",
    eop_path: str | None = None,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
) -> np.ndarray:
    model = normalize_frame_model(frame_model)
    if model == FRAME_MODEL_IAU76_80_EOP:
        rot = eci_to_ecef_rotation_hpop_like(
            t_s,
            jd_utc_start=jd_utc_start,
            eop_path=eop_path,
            dut1_s=dut1_s,
            xp_arcsec=xp_arcsec,
            yp_arcsec=yp_arcsec,
            dat_s=dat_s,
            tt_minus_utc_s=tt_minus_utc_s,
            ddpsi_rad=ddpsi_rad,
            ddeps_rad=ddeps_rad,
        )
        return rot.T @ np.array(r_ecef_km, dtype=float)
    return ecef_to_eci(np.array(r_ecef_km, dtype=float), t_s, jd_utc_start=jd_utc_start)
