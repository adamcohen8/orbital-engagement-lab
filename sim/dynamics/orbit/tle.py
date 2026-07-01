from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.epoch import datetime_to_julian_date

TLE_INITIALIZATION_MODEL_OGP = "ogp"
TLE_INITIALIZATION_MODEL_KEPLERIAN = "keplerian_mean_elements"

_OGP_INITIALIZATION_ALIASES = {
    "ogp",
    "general",
    "general_perturbations",
    "gp",
    "sgp4",
    "sdp4",
    "ogp_sgp4",
    "ogp_sdp4",
}
_KEPLERIAN_INITIALIZATION_ALIASES = {
    "keplerian",
    "keplerian_mean_elements",
    "mean_elements",
    "two_body",
    "legacy",
}


@dataclass(frozen=True)
class TLEElements:
    line1: str
    line2: str
    norad_number: str
    classification: str
    international_designator: str
    epoch_text: str
    epoch_jd_utc: float
    mean_motion_derivative_rev_per_day2: float
    mean_motion_second_derivative_rev_per_day3: float
    bstar: float
    ephemeris_type: str
    element_number: int
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    argp_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_per_day: float
    revolution_number: int


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


def _parse_tle_float(text: str, *, default: float = 0.0) -> float:
    raw = str(text or "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _parse_tle_int(text: str, *, default: int = 0) -> int:
    raw = str(text or "").strip()
    if not raw:
        return int(default)
    return int(raw)


def _parse_tle_compact_exponential(text: str) -> float:
    raw = str(text or "").strip()
    if not raw:
        return 0.0
    raw = raw.replace(" ", "")
    if "e" in raw.lower():
        return float(raw)
    if len(raw) < 3:
        return float(raw)
    mantissa_text = raw[:-2]
    exponent_text = raw[-2:]
    if mantissa_text in {"", "+", "-"}:
        return 0.0
    sign = -1.0 if mantissa_text.startswith("-") else 1.0
    mantissa_digits = mantissa_text.lstrip("+-")
    if "." in mantissa_digits:
        mantissa = float(mantissa_text)
    else:
        mantissa = sign * float(f"0.{mantissa_digits}")
    return float(mantissa * (10.0 ** int(exponent_text)))


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
        norad_number=l1[2:7].strip(),
        classification=l1[7:8].strip(),
        international_designator=l1[9:17].strip(),
        epoch_text=l1[18:32].strip(),
        epoch_jd_utc=tle_epoch_to_julian_date(l1[18:32]),
        mean_motion_derivative_rev_per_day2=_parse_tle_float(l1[33:43]),
        mean_motion_second_derivative_rev_per_day3=_parse_tle_compact_exponential(l1[44:52]),
        bstar=_parse_tle_compact_exponential(l1[53:61]),
        ephemeris_type=l1[62:63].strip(),
        element_number=_parse_tle_int(l1[64:68]),
        inclination_deg=float(l2[8:16]),
        raan_deg=float(l2[17:25]),
        eccentricity=float(f"0.{ecc_text}"),
        argp_deg=float(l2[34:42]),
        mean_anomaly_deg=float(l2[43:51]),
        mean_motion_rev_per_day=float(l2[52:63]),
        revolution_number=_parse_tle_int(l2[63:68]),
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


def _normalise_initialization_model(raw: Any) -> str:
    key = str(raw or TLE_INITIALIZATION_MODEL_OGP).strip().lower().replace("-", "_")
    if key in _OGP_INITIALIZATION_ALIASES:
        return TLE_INITIALIZATION_MODEL_OGP
    if key in _KEPLERIAN_INITIALIZATION_ALIASES:
        return TLE_INITIALIZATION_MODEL_KEPLERIAN
    aliases = sorted(_OGP_INITIALIZATION_ALIASES | _KEPLERIAN_INITIALIZATION_ALIASES)
    raise ValueError(f"Unsupported TLE initialization_model {raw!r}; expected one of: {', '.join(aliases)}.")


def _tle_lines_from_block(tle_block: dict[str, Any]) -> tuple[str, str]:
    block = dict(tle_block or {})
    lines = block.get("lines")
    if isinstance(lines, (list, tuple)) and len(lines) >= 2:
        return str(lines[0]), str(lines[1])
    return str(block.get("line1", "") or ""), str(block.get("line2", "") or "")


def tle_to_rv_eci_ogp(
    elements: TLEElements,
    *,
    target_jd_utc: float | None = None,
    frame_transform: str = "teme_as_eci",
) -> tuple[np.ndarray, np.ndarray]:
    transform = str(frame_transform or "teme_as_eci").strip().lower()
    if transform not in {"teme_as_eci", "teme_to_eci_iau80"}:
        raise ValueError(
            "TLE OGP initialization supports initialization_frame_transform values "
            "'teme_as_eci' and 'teme_to_eci_iau80'."
        )
    jd_utc = float(elements.epoch_jd_utc if target_jd_utc is None else target_jd_utc)
    tsince_min = (jd_utc - float(elements.epoch_jd_utc)) * 1440.0
    from sim.dynamics.orbit.ogp import ogp_propagate_teme
    from sim.dynamics.orbit.sgp4 import transform_teme_to_output_frame

    state = ogp_propagate_teme(elements, tsince_min)
    if state.error:
        raise ValueError(state.error)
    return transform_teme_to_output_frame(
        state.position_teme_km,
        state.velocity_teme_km_s,
        jd_utc=jd_utc,
        output_frame="eci",
        frame_transform=transform,
    )


def tle_block_to_rv_eci(
    tle_block: dict[str, Any], *, target_jd_utc: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    block = dict(tle_block or {})
    line1, line2 = _tle_lines_from_block(block)
    elements = parse_tle_lines(line1, line2, require_checksum=bool(block.get("require_checksum", False)))
    propagate = bool(block.get("propagate_to_initial_epoch", True))
    effective_jd = target_jd_utc if propagate else None
    model = _normalise_initialization_model(block.get("initialization_model"))
    if model == TLE_INITIALIZATION_MODEL_KEPLERIAN:
        return tle_to_rv_eci(elements, target_jd_utc=effective_jd)
    return tle_to_rv_eci_ogp(
        elements,
        target_jd_utc=effective_jd,
        frame_transform=str(block.get("initialization_frame_transform", "teme_as_eci") or "teme_as_eci"),
    )


def tle_block_initialization_metadata(
    tle_block: dict[str, Any],
    *,
    target_jd_utc: float | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    block = dict(tle_block or {})
    line1, line2 = _tle_lines_from_block(block)
    elements = parse_tle_lines(line1, line2, require_checksum=bool(block.get("require_checksum", False)))
    propagate = bool(block.get("propagate_to_initial_epoch", True))
    effective_jd = float(target_jd_utc if propagate and target_jd_utc is not None else elements.epoch_jd_utc)
    model = _normalise_initialization_model(block.get("initialization_model"))
    if model == TLE_INITIALIZATION_MODEL_OGP:
        from sim.dynamics.orbit.ogp import ogp_propagator_name_for_elements

        native_frame = "teme"
        frame_transform = (
            str(block.get("initialization_frame_transform", "teme_as_eci") or "teme_as_eci").strip().lower()
        )
        propagator_family = "OGP"
        propagator_name = ogp_propagator_name_for_elements(elements)
    else:
        native_frame = "eci"
        frame_transform = "keplerian_mean_elements"
        propagator_family = "two_body"
        propagator_name = "Keplerian mean-elements conversion"
    age_days = float(effective_jd - float(elements.epoch_jd_utc))
    return {
        "source": "tle",
        "initialization_model": model,
        "initialization_propagator_family": propagator_family,
        "initialization_propagator_name": propagator_name,
        "handoff_propagation_method": "special",
        "native_frame": native_frame,
        "output_frame": "eci",
        "frame_transform": frame_transform,
        "tle_epoch_jd_utc": float(elements.epoch_jd_utc),
        "initial_jd_utc": float(effective_jd),
        "tle_age_initialization_days": age_days,
        "propagate_to_initial_epoch": bool(propagate),
        "simulation_duration_s": float(duration_s),
        "note": "TLE recovered the initial state only; subsequent trajectory uses configured ONP dynamics.",
    }
