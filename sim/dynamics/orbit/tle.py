from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.epoch import datetime_to_julian_date


@dataclass(frozen=True)
class TLEElements:
    line1: str
    line2: str
    epoch_jd_utc: float
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    argp_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_per_day: float


def tle_epoch_to_julian_date(epoch_text: str) -> float:
    text = str(epoch_text or "").strip()
    if len(text) < 5:
        raise ValueError("TLE epoch must use YYDDD.DDDDDDDD format.")
    year_two = int(text[:2])
    year = 2000 + year_two if year_two < 57 else 1900 + year_two
    day_of_year = float(text[2:])
    if day_of_year < 1.0:
        raise ValueError("TLE epoch day-of-year must be >= 1.")
    day_index = int(math.floor(day_of_year))
    frac_day = day_of_year - day_index
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_index - 1, seconds=frac_day * 86400.0)
    return datetime_to_julian_date(dt)


def _checksum_ok(line: str) -> bool:
    if len(line) < 69 or not line[68].isdigit():
        return False
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10 == int(line[68])


def parse_tle_lines(line1: str, line2: str, *, require_checksum: bool = False) -> TLEElements:
    l1 = str(line1 or "").rstrip("\n")
    l2 = str(line2 or "").rstrip("\n")
    if not l1.startswith("1 ") or not l2.startswith("2 "):
        raise ValueError("TLE line1 must start with '1 ' and line2 must start with '2 '.")
    if len(l1) < 63 or len(l2) < 63:
        raise ValueError("TLE lines are too short.")
    if require_checksum and (not _checksum_ok(l1) or not _checksum_ok(l2)):
        raise ValueError("TLE checksum validation failed.")

    ecc_text = l2[26:33].strip()
    if not ecc_text or not ecc_text.isdigit():
        raise ValueError("TLE eccentricity field is invalid.")

    return TLEElements(
        line1=l1,
        line2=l2,
        epoch_jd_utc=tle_epoch_to_julian_date(l1[18:32]),
        inclination_deg=float(l2[8:16]),
        raan_deg=float(l2[17:25]),
        eccentricity=float(f"0.{ecc_text}"),
        argp_deg=float(l2[34:42]),
        mean_anomaly_deg=float(l2[43:51]),
        mean_motion_rev_per_day=float(l2[52:63]),
    )


def _solve_eccentric_anomaly(mean_anomaly_rad: float, eccentricity: float) -> float:
    mean_anomaly_rad = float(np.mod(mean_anomaly_rad, 2.0 * np.pi))
    e = float(eccentricity)
    ecc_anomaly = mean_anomaly_rad if e < 0.8 else np.pi
    for _ in range(30):
        f = ecc_anomaly - e * np.sin(ecc_anomaly) - mean_anomaly_rad
        fp = 1.0 - e * np.cos(ecc_anomaly)
        step = f / fp
        ecc_anomaly -= step
        if abs(step) < 1e-13:
            break
    return float(ecc_anomaly)


def _coe_to_rv_eci(
    *,
    a_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    a = float(a_km)
    e = float(ecc)
    inc = np.deg2rad(float(inc_deg))
    raan = np.deg2rad(float(raan_deg))
    argp = np.deg2rad(float(argp_deg))
    nu = np.deg2rad(float(true_anomaly_deg))
    p = a * (1.0 - e * e)
    if a <= 0.0 or p <= 0.0:
        raise ValueError("TLE-derived orbit is invalid.")

    cnu, snu = np.cos(nu), np.sin(nu)
    r_pf = np.array([p * cnu / (1.0 + e * cnu), p * snu / (1.0 + e * cnu), 0.0], dtype=float)
    v_pf = np.sqrt(mu_km3_s2 / p) * np.array([-snu, e + cnu, 0.0], dtype=float)

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    q_pf_to_eci = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ],
        dtype=float,
    )
    return q_pf_to_eci @ r_pf, q_pf_to_eci @ v_pf


def tle_to_rv_eci(
    elements: TLEElements,
    *,
    target_jd_utc: float | None = None,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    mean_motion_rad_s = float(elements.mean_motion_rev_per_day) * 2.0 * np.pi / 86400.0
    if mean_motion_rad_s <= 0.0:
        raise ValueError("TLE mean motion must be positive.")
    a_km = (float(mu_km3_s2) / (mean_motion_rad_s**2)) ** (1.0 / 3.0)
    dt_s = 0.0 if target_jd_utc is None else (float(target_jd_utc) - float(elements.epoch_jd_utc)) * 86400.0
    mean_anomaly_rad = np.deg2rad(float(elements.mean_anomaly_deg)) + mean_motion_rad_s * dt_s
    ecc_anomaly = _solve_eccentric_anomaly(mean_anomaly_rad, float(elements.eccentricity))
    e = float(elements.eccentricity)
    true_anomaly_rad = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(ecc_anomaly / 2.0),
        np.sqrt(1.0 - e) * np.cos(ecc_anomaly / 2.0),
    )
    return _coe_to_rv_eci(
        a_km=a_km,
        ecc=e,
        inc_deg=float(elements.inclination_deg),
        raan_deg=float(elements.raan_deg),
        argp_deg=float(elements.argp_deg),
        true_anomaly_deg=float(np.rad2deg(true_anomaly_rad)),
        mu_km3_s2=mu_km3_s2,
    )


def tle_block_to_rv_eci(tle_block: dict[str, Any], *, target_jd_utc: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    block = dict(tle_block or {})
    lines = block.get("lines")
    if isinstance(lines, (list, tuple)) and len(lines) >= 2:
        line1 = str(lines[0])
        line2 = str(lines[1])
    else:
        line1 = str(block.get("line1", "") or "")
        line2 = str(block.get("line2", "") or "")
    elements = parse_tle_lines(line1, line2, require_checksum=bool(block.get("require_checksum", False)))
    propagate = bool(block.get("propagate_to_initial_epoch", True))
    return tle_to_rv_eci(elements, target_jd_utc=target_jd_utc if propagate else None)
