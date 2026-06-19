from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.epoch import datetime_to_julian_date, julian_date_to_datetime
from sim.dynamics.orbit.frames import apparent_sidereal_time_hpop_like, eci_to_ecef_harmonic
from sim.dynamics.orbit.harris_priester_backend import harris_priester_density
from sim.dynamics.orbit.jacchia70_backend import jacchia70_density
from sim.dynamics.orbit.jb2008_backend import jb2006_density, jb2008_density
from sim.dynamics.orbit.msis86_backend import msis86_density as msis86_local_density
from sim.dynamics.orbit.nrlmsise00_backend import nrlmsise00_density as nrlmsise00_local_density
from sim.utils.geodesy import ecef_to_geodetic_deg_km

AtmosphereModelName = Literal[
    "exponential", "ussa1976", "msis86", "nrlmsise00", "jacchia70", "jb2006", "jb2008", "harris_priester"
]


def _radial_altitude_km_from_eci(r_eci_km: np.ndarray) -> float:
    r_vec = np.asarray(r_eci_km, dtype=float).reshape(3)
    r2 = float(np.dot(r_vec, r_vec))
    if r2 <= 0.0:
        return 0.0
    return float(max(0.0, np.sqrt(r2) - EARTH_RADIUS_KM))


def _ecef_from_eci_for_atmosphere(r_eci_km: np.ndarray, t_s: float, env: dict) -> np.ndarray:
    frame_model = str(env.get("density_frame_model", env.get("drag_frame_model", "simple"))).strip().lower()
    eop_path = env.get("density_eop_path", env.get("drag_eop_path"))
    return eci_to_ecef_harmonic(
        np.array(r_eci_km, dtype=float),
        float(t_s),
        jd_utc_start=env.get("jd_utc_start"),
        frame_model=frame_model,
        eop_path=None if eop_path is None else str(eop_path),
    )


def _altitude_km_from_eci(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    env = {} if env is None else env
    r_ecef_km = _ecef_from_eci_for_atmosphere(r_eci_km, t_s, env)
    if str(env.get("geodetic_model", "")).lower() == "wgs84":
        _, _, alt_km = ecef_to_geodetic_deg_km(r_ecef_km)
        return float(max(alt_km, 0.0))
    return float(max(0.0, np.linalg.norm(r_ecef_km) - EARTH_RADIUS_KM))


def _spherical_lat_lon_deg_from_eci(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> tuple[float, float]:
    env = {} if env is None else env
    r_ecef_km = _ecef_from_eci_for_atmosphere(r_eci_km, t_s, env)
    if str(env.get("geodetic_model", "")).lower() == "wgs84":
        lat, lon, _ = ecef_to_geodetic_deg_km(r_ecef_km)
        return float(lat), float(lon)
    r = float(np.linalg.norm(r_ecef_km))
    if r <= 0.0:
        return 0.0, 0.0
    x, y, z = r_ecef_km
    lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return float(lat), float(lon)


def density_exponential(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    alt_km = _radial_altitude_km_from_eci(r_eci_km)
    if alt_km > 1000.0:
        return 0.0
    rho0 = 1.225
    h = 8.5
    return float(rho0 * np.exp(-alt_km / h))


def density_ussa1976(r_eci_km: np.ndarray, t_s: float) -> float:
    """
    Approximate US Standard Atmosphere 1976 density profile.

    - 0..86 km: standard lapse-rate layers via geopotential-altitude equations.
    - 86..1000 km: log-linear interpolation on tabulated USSA-1976 reference densities.
    """
    alt_km = _altitude_km_from_eci(r_eci_km, t_s)

    if alt_km <= 86.0:
        # 1976 standard atmosphere layers (0-86 km).
        g0 = 9.80665
        r_air = 287.05287
        hb = np.array([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 86.0], dtype=float) * 1e3
        lb = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020], dtype=float)
        tb = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65], dtype=float)
        pb = np.array([101325.0], dtype=float)
        # Build base pressures recursively.
        for i in range(lb.size):
            h0 = hb[i]
            h1 = hb[i + 1]
            lapse = lb[i]
            t0 = tb[i]
            p0 = pb[-1]
            if abs(lapse) < 1e-12:
                p1 = p0 * np.exp(-g0 * (h1 - h0) / (r_air * t0))
            else:
                t1 = t0 + lapse * (h1 - h0)
                p1 = p0 * (t1 / t0) ** (-g0 / (r_air * lapse))
            pb = np.append(pb, p1)

        h = alt_km * 1e3
        i = int(np.searchsorted(hb, h, side="right") - 1)
        i = max(0, min(i, lb.size - 1))
        h0 = hb[i]
        lapse = lb[i]
        t0 = tb[i]
        p0 = pb[i]
        if abs(lapse) < 1e-12:
            t = t0
            p = p0 * np.exp(-g0 * (h - h0) / (r_air * t))
        else:
            t = t0 + lapse * (h - h0)
            p = p0 * (t / t0) ** (-g0 / (r_air * lapse))
        rho = p / (r_air * t)
        return float(max(rho, 0.0))

    if alt_km > 1000.0:
        return 0.0

    # USSA-1976 high-altitude reference density table (kg/m^3), log-space interpolated.
    table_alt_km = np.array(
        [
            86.0,
            100.0,
            110.0,
            120.0,
            130.0,
            140.0,
            150.0,
            180.0,
            200.0,
            250.0,
            300.0,
            350.0,
            400.0,
            450.0,
            500.0,
            600.0,
            700.0,
            800.0,
            900.0,
            1000.0,
        ],
        dtype=float,
    )
    table_rho = np.array(
        [
            6.958e-6,
            5.604e-7,
            9.708e-8,
            2.222e-8,
            8.152e-9,
            3.831e-9,
            2.076e-9,
            5.464e-10,
            2.789e-10,
            7.248e-11,
            2.418e-11,
            9.518e-12,
            3.725e-12,
            1.585e-12,
            6.967e-13,
            1.454e-13,
            3.614e-14,
            1.170e-14,
            5.245e-15,
            3.019e-15,
        ],
        dtype=float,
    )
    lrho = np.interp(float(alt_km), table_alt_km, np.log(table_rho))
    return float(np.exp(lrho))


def _temperature_from_altitude_k_approx(alt_km: float) -> float:
    h = float(max(alt_km, 0.0))
    if h < 11.0:
        return 288.15 - 6.5 * h
    if h < 20.0:
        return 216.65
    if h < 32.0:
        return 216.65 + (h - 20.0) * 1.0
    if h < 47.0:
        return 228.65 + (h - 32.0) * 2.8
    if h < 51.0:
        return 270.65
    if h < 71.0:
        return 270.65 - (h - 51.0) * 2.8
    if h < 86.0:
        return 214.65 - (h - 71.0) * 2.0
    return 186.87


def density_nrlmsise00(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    """
    Local NRLMSISE-00 density model copied from the MATLAB HPOP implementation.

    Supported backends:
    - env-provided callable: env["nrlmsise00_density_callable"](alt_km, lat_deg, lon_deg, dt_utc, env) -> kg/m^3
    - source-local OEL backend with direct env f107/f107a/ap inputs, or optional HPOP-style SW-All table input.
    """
    env = {} if env is None else dict(env)
    alt_km = _altitude_km_from_eci(r_eci_km, t_s, env=env)
    lat_deg, lon_deg = _spherical_lat_lon_deg_from_eci(r_eci_km, t_s, env=env)

    jd_utc = env.get("jd_utc")
    if jd_utc is not None:
        dt_utc = julian_date_to_datetime(float(jd_utc))
    else:
        base_epoch = env.get("atmo_epoch_utc", datetime(2020, 1, 1, tzinfo=timezone.utc))
        if isinstance(base_epoch, datetime):
            if base_epoch.tzinfo is None:
                base_epoch = base_epoch.replace(tzinfo=timezone.utc)
            dt_utc = base_epoch + timedelta(seconds=float(t_s))
        else:
            dt_utc = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(t_s))

    custom_fn = env.get("nrlmsise00_density_callable", None)
    if callable(custom_fn):
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env)))
    if env.get("nrlmsise00_lst_hr") is None:
        frame_model = str(env.get("density_frame_model", env.get("drag_frame_model", ""))).strip().lower()
        if frame_model == "hpop_like":
            eop_path = env.get("density_eop_path", env.get("drag_eop_path"))
            sidereal = apparent_sidereal_time_hpop_like(datetime_to_julian_date(dt_utc), None if eop_path is None else str(eop_path))
            env["nrlmsise00_lst_hr"] = ((math.radians(float(lon_deg)) + sidereal) % (2.0 * math.pi)) * 24.0 / (
                2.0 * math.pi
            )

    return float(max(0.0, nrlmsise00_local_density(alt_km, lat_deg, lon_deg, dt_utc, env)))


def density_msis86(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    env = {} if env is None else dict(env)
    alt_km = _altitude_km_from_eci(r_eci_km, t_s, env=env)
    lat_deg, lon_deg = _spherical_lat_lon_deg_from_eci(r_eci_km, t_s, env=env)

    jd_utc = env.get("jd_utc")
    if jd_utc is not None:
        dt_utc = julian_date_to_datetime(float(jd_utc))
    else:
        base_epoch = env.get("atmo_epoch_utc", datetime(2020, 1, 1, tzinfo=timezone.utc))
        if isinstance(base_epoch, datetime):
            if base_epoch.tzinfo is None:
                base_epoch = base_epoch.replace(tzinfo=timezone.utc)
            dt_utc = base_epoch + timedelta(seconds=float(t_s))
        else:
            dt_utc = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(t_s))

    custom_fn = env.get("msis86_density_callable", None)
    if callable(custom_fn):
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env)))
    if bool(env.get("msis86_hpop_angle_compat", True)):
        if env.get("msis86_lst_hr") is None:
            frame_model = str(env.get("density_frame_model", env.get("drag_frame_model", ""))).strip().lower()
            eop_path = env.get("density_eop_path", env.get("drag_eop_path")) if frame_model == "hpop_like" else None
            sidereal = apparent_sidereal_time_hpop_like(datetime_to_julian_date(dt_utc), None if eop_path is None else str(eop_path))
            env["msis86_lst_hr"] = (
                ((math.radians(float(lon_deg)) + sidereal) % (2.0 * math.pi))
                * 24.0
                / (2.0 * math.pi)
            )
        lat_deg = math.radians(lat_deg)
        lon_deg = math.radians(lon_deg)
    return float(max(0.0, msis86_local_density(alt_km, lat_deg, lon_deg, dt_utc, env)))


def density_jb2008(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    """
    JB2008 density model via externally supplied backend callable.

    Required:
    - env["jb2008_density_callable"](alt_km, lat_deg, lon_deg, dt_utc, env) -> kg/m^3
    """
    env = {} if env is None else dict(env)
    alt_km = _altitude_km_from_eci(r_eci_km, t_s, env=env)
    lat_deg, lon_deg = _spherical_lat_lon_deg_from_eci(r_eci_km, t_s, env=env)

    jd_utc = env.get("jd_utc")
    if jd_utc is not None:
        dt_utc = julian_date_to_datetime(float(jd_utc))
    else:
        base_epoch = env.get("atmo_epoch_utc", datetime(2020, 1, 1, tzinfo=timezone.utc))
        if isinstance(base_epoch, datetime):
            if base_epoch.tzinfo is None:
                base_epoch = base_epoch.replace(tzinfo=timezone.utc)
            dt_utc = base_epoch + timedelta(seconds=float(t_s))
        else:
            dt_utc = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(t_s))

    custom_fn = env.get("jb2008_density_callable", None)
    if callable(custom_fn):
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env)))
    return float(max(0.0, jb2008_density(alt_km, lat_deg, lon_deg, dt_utc, env)))


def density_jb2006(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    env = {} if env is None else dict(env)
    alt_km = _altitude_km_from_eci(r_eci_km, t_s, env=env)
    lat_deg, lon_deg = _spherical_lat_lon_deg_from_eci(r_eci_km, t_s, env=env)

    jd_utc = env.get("jd_utc")
    if jd_utc is not None:
        dt_utc = julian_date_to_datetime(float(jd_utc))
    else:
        base_epoch = env.get("atmo_epoch_utc", datetime(2020, 1, 1, tzinfo=timezone.utc))
        if isinstance(base_epoch, datetime):
            if base_epoch.tzinfo is None:
                base_epoch = base_epoch.replace(tzinfo=timezone.utc)
            dt_utc = base_epoch + timedelta(seconds=float(t_s))
        else:
            dt_utc = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(t_s))

    custom_fn = env.get("jb2006_density_callable", None)
    if callable(custom_fn):
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env)))
    return float(max(0.0, jb2006_density(alt_km, lat_deg, lon_deg, dt_utc, env)))


def density_jacchia70(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    env = {} if env is None else dict(env)
    alt_km = _altitude_km_from_eci(r_eci_km, t_s, env=env)
    lat_deg, lon_deg = _spherical_lat_lon_deg_from_eci(r_eci_km, t_s, env=env)

    jd_utc = env.get("jd_utc")
    if jd_utc is not None:
        dt_utc = julian_date_to_datetime(float(jd_utc))
    else:
        base_epoch = env.get("atmo_epoch_utc", datetime(2020, 1, 1, tzinfo=timezone.utc))
        if isinstance(base_epoch, datetime):
            if base_epoch.tzinfo is None:
                base_epoch = base_epoch.replace(tzinfo=timezone.utc)
            dt_utc = base_epoch + timedelta(seconds=float(t_s))
        else:
            dt_utc = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(t_s))

    custom_fn = env.get("jacchia70_density_callable", None)
    if callable(custom_fn):
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env)))
    return float(max(0.0, jacchia70_density(alt_km, lat_deg, lon_deg, dt_utc, env)))


def density_harris_priester(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    return harris_priester_density(r_eci_km, t_s, env=env)


def atmosphere_state_from_model(
    model: AtmosphereModelName,
    r_eci_km: np.ndarray,
    t_s: float,
    env: dict | None = None,
) -> dict[str, float]:
    """
    Return atmosphere state dictionary with at least:
    - density_kg_m3
    - temperature_k
    - pressure_pa
    - sound_speed_m_s
    """
    env_local = {} if env is None else dict(env)
    alt_km = _altitude_km_from_eci(r_eci_km, t_s, env=env_local)
    lat_deg, lon_deg = _spherical_lat_lon_deg_from_eci(r_eci_km, t_s, env=env_local)
    r_air = float(env_local.get("air_gas_constant_j_kg_k", 287.05287))
    gamma = float(env_local.get("air_gamma", 1.4))

    jd_utc = env_local.get("jd_utc")
    if jd_utc is not None:
        dt_utc = julian_date_to_datetime(float(jd_utc))
    else:
        base_epoch = env_local.get("atmo_epoch_utc", datetime(2020, 1, 1, tzinfo=timezone.utc))
        if isinstance(base_epoch, datetime):
            if base_epoch.tzinfo is None:
                base_epoch = base_epoch.replace(tzinfo=timezone.utc)
            dt_utc = base_epoch + timedelta(seconds=float(t_s))
        else:
            dt_utc = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(t_s))

    m = str(model).lower()
    cb = None
    if m == "nrlmsise00":
        cb = env_local.get("nrlmsise00_density_callable", None)
    elif m in {"msis86", "msis-86", "hpop_msis86"}:
        cb = env_local.get("msis86_density_callable", None)
    elif m in {"jacchia70", "jacchia-70", "hpop_jacchia70"}:
        cb = env_local.get("jacchia70_density_callable", None)
    elif m == "jb2006":
        cb = env_local.get("jb2006_density_callable", None)
    elif m == "jb2008":
        cb = env_local.get("jb2008_density_callable", None)

    if callable(cb):
        out = cb(alt_km, lat_deg, lon_deg, dt_utc, env_local)
        if isinstance(out, dict):
            rho = float(out.get("density_kg_m3", out.get("rho_kg_m3", 0.0)))
            t_k = float(out.get("temperature_k", _temperature_from_altitude_k_approx(alt_km)))
            p_pa = float(out.get("pressure_pa", max(rho, 0.0) * r_air * max(t_k, 1.0)))
            a_m_s = float(out.get("sound_speed_m_s", np.sqrt(max(gamma * r_air * max(t_k, 1.0), 1e-9))))
            return {
                "density_kg_m3": max(rho, 0.0),
                "temperature_k": max(t_k, 1.0),
                "pressure_pa": max(p_pa, 0.0),
                "sound_speed_m_s": max(a_m_s, 1e-3),
            }
        rho = float(out)
        t_k = _temperature_from_altitude_k_approx(alt_km)
        p_pa = max(rho, 0.0) * r_air * t_k
        a_m_s = float(np.sqrt(max(gamma * r_air * t_k, 1e-9)))
        return {
            "density_kg_m3": max(rho, 0.0),
            "temperature_k": t_k,
            "pressure_pa": max(p_pa, 0.0),
            "sound_speed_m_s": max(a_m_s, 1e-3),
        }

    rho = density_from_model(model, r_eci_km, t_s, env=env_local)
    t_k = _temperature_from_altitude_k_approx(alt_km)
    p_pa = max(rho, 0.0) * r_air * t_k
    a_m_s = float(np.sqrt(max(gamma * r_air * t_k, 1e-9)))
    return {
        "density_kg_m3": max(float(rho), 0.0),
        "temperature_k": max(t_k, 1.0),
        "pressure_pa": max(float(p_pa), 0.0),
        "sound_speed_m_s": max(a_m_s, 1e-3),
    }


def density_from_model(
    model: AtmosphereModelName,
    r_eci_km: np.ndarray,
    t_s: float,
    env: dict | None = None,
) -> float:
    m = str(model).lower()
    if m == "exponential":
        return density_exponential(r_eci_km, t_s, env=env)
    if m == "ussa1976":
        return density_ussa1976(r_eci_km, t_s)
    if m == "nrlmsise00":
        return density_nrlmsise00(r_eci_km, t_s, env=env)
    if m in {"msis86", "msis-86", "hpop_msis86"}:
        return density_msis86(r_eci_km, t_s, env=env)
    if m in {"jacchia70", "jacchia-70", "hpop_jacchia70"}:
        return density_jacchia70(r_eci_km, t_s, env=env)
    if m == "jb2006":
        return density_jb2006(r_eci_km, t_s, env=env)
    if m == "jb2008":
        return density_jb2008(r_eci_km, t_s, env=env)
    if m in {"harris_priester", "harris-priester", "hp", "hpop_harris_priester"}:
        return density_harris_priester(r_eci_km, t_s, env=env)
    raise ValueError(f"Unknown atmosphere model '{model}'.")
