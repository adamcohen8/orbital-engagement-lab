from __future__ import annotations

import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.epoch import datetime_to_julian_date, sun_position_eci_km_enhanced, sun_position_eci_km_simple
from sim.dynamics.orbit.frames import apparent_sidereal_time_hpop_like
from sim.dynamics.orbit.jb2008_backend import _position_eci_from_geodetic

_BDATA = np.array([28.15204, -0.085586, 0.0001284, -0.000010056, -0.00001021, 0.0000015044, 0.000000099826])
_ALPHA = np.array([0.0, 0.0, 0.0, 0.0, -0.38, 0.0])
_EI = np.array([28.0134, 31.9988, 15.9994, 39.948, 4.0026, 1.00797])
_ALTMIN = np.array([90.0, 105.0, 125.0, 160.0, 200.0, 300.0, 500.0, 1500.0, 2500.0])
_NG = np.array([4, 5, 6, 6, 6, 6, 6, 6])
_CZ = np.array([1.0, 0.9045085, 0.6545085, 0.3454915, 0.0954915, 0.0])
_GAUSS_W = {
    3: np.array([0.5555556, 0.8888889, 0.5555556]),
    4: np.array([0.3478548, 0.6521452, 0.6521452, 0.3478548]),
    5: np.array([0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269]),
    6: np.array([0.1713245, 0.3607616, 0.4679139, 0.4679139, 0.3607616, 0.1713245]),
    7: np.array([0.1294850, 0.2797054, 0.3818301, 0.4179592, 0.3818301, 0.2797054, 0.1294850]),
    8: np.array([0.1012285, 0.2223810, 0.3137067, 0.3626838, 0.3626838, 0.3137067, 0.2223810, 0.1012285]),
}
_GAUSS_X = {
    3: np.array([-0.7745967, 0.0, 0.7745967]),
    4: np.array([-0.8611363, -0.3399810, 0.3399810, 0.8611363]),
    5: np.array([-0.9061798, -0.5384693, 0.0, 0.5384693, 0.9061798]),
    6: np.array([-0.9324695, -0.6612094, -0.2386192, 0.2386192, 0.6612094, 0.9324695]),
    7: np.array([-0.9491079, -0.7415312, -0.4058452, 0.0, 0.4058452, 0.7415312, 0.9491079]),
    8: np.array([-0.9602899, -0.7966665, -0.5255324, -0.1834346, 0.1834346, 0.5255324, 0.7966665, 0.9602899]),
}


@lru_cache(maxsize=4)
def _load_swdata(path: str) -> np.ndarray:
    arr = np.loadtxt(str(Path(path).expanduser().resolve()), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 30:
        raise ValueError(f"Unexpected SW-All shape {arr.shape} from {path}")
    return arr


def _resolve_table_path(raw: str) -> str:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / p
    return str(p.resolve())


def _day_of_year(dt_utc: datetime) -> float:
    start = datetime(dt_utc.year, 1, 1, tzinfo=timezone.utc)
    return 1.0 + (dt_utc - start).total_seconds() / 86400.0


def _solar_geomagnetic_inputs(dt_utc: datetime, env: dict) -> tuple[float, float, float, int]:
    i1 = int(env.get("jacchia70_geomagnetic_index_type", 2))
    f10_direct = env.get("jacchia70_f10", env.get("f107"))
    f10b_direct = env.get("jacchia70_f10b", env.get("f107a"))
    gi_direct = env.get("jacchia70_ap" if i1 == 2 else "jacchia70_kp", env.get("ap" if i1 == 2 else "kp"))

    sw_path = env.get("jacchia70_sw_path")
    if sw_path not in (None, "") and (f10_direct is None or f10b_direct is None or gi_direct is None):
        sw = _load_swdata(_resolve_table_path(str(sw_path)))
        matches = np.where((sw[:, 0] == dt_utc.year) & (sw[:, 1] == dt_utc.month) & (sw[:, 2] == dt_utc.day))[0]
        if matches.size == 0:
            raise RuntimeError(f"Jacchia-70 SW data missing day {dt_utc.year}-{dt_utc.month}-{dt_utc.day}")
        i = int(matches[0])
        slot = max(0, min(int(math.floor(dt_utc.hour / 3.0)), 7))
        if gi_direct is None:
            gi_direct = sw[i, 14 + slot] if i1 == 2 else sw[i, 5 + slot]
        if f10_direct is None:
            f10_direct = sw[max(i - 1, 0), 29]
        if f10b_direct is None:
            f10b_direct = 0.5 * (sw[i, 27] + sw[max(i - 81, 0), 27])

    f10 = float(150.0 if f10_direct is None else f10_direct)
    f10b = float(f10 if f10b_direct is None else f10b_direct)
    gi = float(4.0 if gi_direct is None else gi_direct)
    return f10, f10b, gi, i1


def _jgrav(alt_km: float) -> float:
    return 9.80665 / (1.0 + float(alt_km) / 6356.766) ** 2


def _jmweight(alt_km: float) -> float:
    if alt_km > 105.0:
        return 1.0
    u = float(alt_km) - 100.0
    return float(sum(float(coeff) * u**idx for idx, coeff in enumerate(_BDATA)))


def _jtemp(alt_km: float, tx: float, t1: float, t3: float, t4: float, a2: float) -> float:
    u = float(alt_km) - 125.0
    if u > 0.0:
        return float(tx + a2 * math.atan(t1 * u * (1.0 + 4.5e-6 * u**2.5) / a2))
    return float(tx + t1 * u + t3 * u**3 + t4 * u**4)


def _jgauss(z1: float, z2: float, nmin: int, tx: float, t1: float, t3: float, t4: float, a2: float) -> float:
    if z2 <= z1:
        return 0.0
    total = 0.0
    for k in range(max(0, int(nmin) - 1), 8):
        ngauss = int(_NG[k])
        a = float(_ALTMIN[k])
        d = min(float(z2), float(_ALTMIN[k + 1]))
        if d <= a:
            continue
        half_width = 0.5 * (d - a)
        rr = 0.0
        for weight, abscissa in zip(_GAUSS_W[ngauss], _GAUSS_X[ngauss]):
            z = half_width * (float(abscissa) + 1.0) + a
            rr += float(weight) * _jmweight(z) * _jgrav(z) / _jtemp(z, tx, t1, t3, t4, a2)
        total += half_width * rr
        if d == z2:
            break
    return float(total)


def _jacchia(z_km: float, tinf_k: float) -> tuple[float, float, float, float, np.ndarray]:
    av = 6.02257e23
    qn = 0.7811
    qo2 = 0.20955
    qa = 9.343e-3
    qhe = 0.00001289
    rgas = 8.31432
    t0 = 183.0

    tx = 444.3807 + 0.02385 * tinf_k - 392.8292 * math.exp(-0.0021357 * tinf_k)
    a2 = 2.0 * (tinf_k - tx) / math.pi
    txt0 = tx - t0
    t1 = 1.9 * txt0 / 35.0
    t3 = -1.7 * txt0 / 35.0**3
    t4 = -0.8 * txt0 / 35.0**4
    tz = _jtemp(z_km, tx, t1, t3, t4, a2)

    d = min(z_km, 105.0)
    r = _jgauss(90.0, d, 1, tx, t1, t3, t4, a2)
    em = _jmweight(d)
    td = _jtemp(d, tx, t1, t3, t4, a2)
    dens = 2.1926e-8 * em * math.exp(-r / rgas) / td
    factor = av * dens
    par = factor / em
    factor = factor / 28.96

    if z_km <= 105.0:
        a = np.zeros(6, dtype=float)
        a[0] = math.log10(max(qn * factor, 1e-300))
        a[3] = math.log10(max(qa * factor, 1e-300))
        a[4] = math.log10(max(qhe * factor, 1e-300))
        a[2] = math.log10(max(2.0 * par * (1.0 - em / 28.96), 1e-300))
        a[1] = math.log10(max(par * (em * (1.0 + qo2) / 28.96 - 1.0), 1e-300))
        a[5] = 0.0
        return float(dens), math.log10(max(dens, 1e-300)), float(em), float(tz), a

    di = np.zeros(6, dtype=float)
    di[0] = qn * factor
    di[1] = par * (em * (1.0 + qo2) / 28.96 - 1.0)
    di[2] = 2.0 * par * (1.0 - em / 28.96)
    di[3] = qa * factor
    di[4] = qhe * factor

    r = _jgauss(d, z_km, 2, tx, t1, t3, t4, a2)
    dit = np.ones(6, dtype=float)
    for idx in range(5):
        dit[idx] = di[idx] * (td / tz) ** (1.0 + _ALPHA[idx]) * math.exp(-_EI[idx] * r / rgas)
        if dit[idx] <= 0.0:
            dit[idx] = 1e-6

    if z_km > 500.0:
        s = _jtemp(500.0, tx, t1, t3, t4, a2)
        di[5] = 10.0 ** (73.13 - 39.4 * math.log10(s) + 5.5 * math.log10(s) ** 2)
        r = _jgauss(500.0, z_km, 7, tx, t1, t3, t4, a2)
        dit[5] = di[5] * (s / tz) * math.exp(-_EI[5] * r / rgas)

    dens = float(np.sum(_EI * dit) / av)
    em = float(dens * av / np.sum(dit))
    a = np.log10(np.maximum(dit, 1e-300))
    return dens, math.log10(max(dens, 1e-300)), em, float(tz), a


def _jtinf(f10: float, f10b: float, gi: float, xlat: float, sda: float, sha: float, dy: float, i1: int) -> float:
    tc = 383.0 + 3.32 * f10b + 1.8 * (f10 - f10b)
    eta = 0.5 * abs(xlat - sda)
    theta = 0.5 * abs(xlat + sda)
    tau = sha - 0.6457718 + 0.1047198 * math.sin(sha + 0.7504916)
    if tau > math.pi:
        tau -= 2.0 * math.pi
    if tau < -math.pi:
        tau += 2.0 * math.pi

    a1 = math.sin(theta) ** 2.5
    a2 = math.cos(eta) ** 2.5
    a3 = math.cos(tau / 2.0) ** 3
    b1 = 1.0 + 0.31 * a1
    tv = b1 * (1.0 + 0.31 * ((a2 - a1) / b1) * a3)
    tl = tc * tv

    tg = 28.0 * gi + 0.03 * math.exp(gi) if i1 == 1 else gi + 100.0 * (1.0 - math.exp(-0.08 * gi))
    g3 = 0.5 * (1.0 + math.sin(2.0 * math.pi * dy + 5.974262))
    tau1 = dy + 0.1145 * (g3**2.16 - 0.5)
    g1 = 0.349 + 0.206 * math.sin(2.0 * math.pi * tau1 + 3.9531708)
    g2 = math.sin(4.0 * math.pi * tau1 + 4.3214352)
    ts = 2.41 + f10b * g1 * g2
    return float(tl + tg + ts)


def _jslv(alt_km: float, xlat_rad: float, day: float) -> float:
    if alt_km > 170.0:
        return 0.0
    z = alt_km - 90.0
    d = 0.014 * z * math.exp(-0.0013 * z * z) * math.sin(0.0172 * day + 1.72) * math.sin(xlat_rad) ** 2
    return float(-d if xlat_rad < 0.0 else d)


def _jslvh(dl: float, denhe: float, xlat_rad: float, sda: float) -> float:
    ezero = 10.0**denhe
    a = abs(0.65 * (sda / 0.40909079))
    b = 0.5 * xlat_rad
    if sda < 0.0:
        b = -b
    denhe = denhe + a * (math.sin(0.7854 - b) ** 3 - 0.35356)
    rho = 10.0**dl + 6.646e-24 * (10.0**denhe - ezero)
    return math.log10(max(rho, 1e-300))


def _jfair(dhel1: float, dhel2: float, dlg1: float, dlg2: float, alt_km: float) -> tuple[float, float]:
    idx = max(0, min(int(math.trunc((alt_km - 440.0) / 10.0)), _CZ.size - 1))
    czi = float(_CZ[idx])
    szi = 1.0 - czi
    return dhel1 * czi + dhel2 * szi, dlg1 * czi + dlg2 * szi


def _sun_position_from_env(jd_utc: float, env: dict) -> np.ndarray:
    r_sun = env.get("sun_pos_eci_km")
    if r_sun is not None:
        return np.asarray(r_sun, dtype=float).reshape(3)
    sun_model = str(env.get("atmosphere_sun_model", env.get("sun_model", ""))).strip().lower()
    if sun_model in {"hpop_simple", "hpop_validation_simple", "validation_simple", "simple"}:
        return sun_position_eci_km_simple(float(jd_utc))
    return sun_position_eci_km_enhanced(float(jd_utc))


def jacchia70_density(
    alt_km: float, lat_deg: float, lon_deg: float, dt_utc: datetime, env: dict | None = None
) -> float:
    env = {} if env is None else dict(env)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)

    z = max(90.0, min(float(alt_km), 2500.0))
    jd_utc = datetime_to_julian_date(dt_utc)
    r_sun = _sun_position_from_env(jd_utc, env)
    ra_sun = math.atan2(float(r_sun[1]), float(r_sun[0])) % (2.0 * math.pi)
    sda = math.atan2(float(r_sun[2]), math.hypot(float(r_sun[0]), float(r_sun[1])))
    frame_model = str(env.get("density_frame_model", env.get("drag_frame_model", ""))).strip().lower()
    if frame_model == "hpop_like":
        eop_path = env.get("density_eop_path", env.get("drag_eop_path"))
        gast = env.get("jacchia70_gast_rad")
        if gast is None:
            gast = apparent_sidereal_time_hpop_like(jd_utc, None if eop_path is None else str(eop_path))
        ra_sat = (float(gast) + math.radians(float(lon_deg))) % (2.0 * math.pi)
    else:
        r_sat_eci_km = _position_eci_from_geodetic(float(lat_deg), float(lon_deg), z, jd_utc, env)
        ra_sat = math.atan2(float(r_sat_eci_km[1]), float(r_sat_eci_km[0])) % (2.0 * math.pi)
    sha = (ra_sat - ra_sun) % (2.0 * math.pi)
    if sha > math.pi:
        sha -= 2.0 * math.pi
    if sha < -math.pi:
        sha += 2.0 * math.pi

    f10, f10b, gi, i1 = _solar_geomagnetic_inputs(dt_utc, env)
    xlat = math.radians(float(lat_deg))
    day = _day_of_year(dt_utc)
    dy = day / 365.2422
    te = _jtinf(f10, f10b, gi, xlat, sda, sha, dy, i1)
    _, dl, _, _, species_log10 = _jacchia(z, te)

    denlg = _jslv(z, xlat, day) if z <= 170.0 else 0.0
    if z >= 500.0:
        dl = _jslvh(dl, float(species_log10[4]), xlat, sda)
    elif z > 440.0:
        dlg2 = _jslvh(dl, float(species_log10[4]), xlat, sda)
        species_log10[4], dl = _jfair(float(species_log10[4]), float(species_log10[4]), dl, dlg2, z)

    return float(max(0.0, 1000.0 * 10.0 ** (dl + denlg)))
