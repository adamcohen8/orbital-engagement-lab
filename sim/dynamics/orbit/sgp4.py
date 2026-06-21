from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.core.models import StateTruth
from sim.dynamics.orbit.tle import TLEElements, parse_tle_lines


@dataclass(frozen=True)
class SGP4State:
    position_teme_km: np.ndarray
    velocity_teme_km_s: np.ndarray
    error: str | None = None


@dataclass(frozen=True)
class SGP4PropagationMetadata:
    propagation_method: str
    general_model: str
    native_frame: str
    output_frame: str
    frame_transform: str
    tle_epoch_jd_utc: float
    tle_age_start_days: float
    tle_age_end_days: float


@dataclass(frozen=True)
class SGP4EphemerisProvider:
    elements: TLEElements
    mass_kg: float
    start_jd_utc: float
    duration_s: float
    output_frame: str = "eci"
    frame_transform: str = "teme_as_eci"
    attitude_quat_bn: np.ndarray | None = None
    angular_rate_body_rad_s: np.ndarray | None = None

    @classmethod
    def from_tle_block(
        cls,
        tle_block: dict,
        *,
        mass_kg: float,
        start_jd_utc: float | None,
        duration_s: float,
        output_frame: str = "eci",
        frame_transform: str = "teme_as_eci",
        attitude_quat_bn: np.ndarray | None = None,
        angular_rate_body_rad_s: np.ndarray | None = None,
    ) -> SGP4EphemerisProvider:
        block = dict(tle_block or {})
        lines = block.get("lines")
        if isinstance(lines, (list, tuple)) and len(lines) >= 2:
            line1 = str(lines[0])
            line2 = str(lines[1])
        else:
            line1 = str(block.get("line1", "") or "")
            line2 = str(block.get("line2", "") or "")
        elements = parse_tle_lines(line1, line2, require_checksum=bool(block.get("require_checksum", False)))
        resolved_start = float(elements.epoch_jd_utc if start_jd_utc is None else start_jd_utc)
        return cls(
            elements=elements,
            mass_kg=float(mass_kg),
            start_jd_utc=resolved_start,
            duration_s=float(duration_s),
            output_frame=str(output_frame or "eci").strip().lower(),
            frame_transform=str(frame_transform or "teme_as_eci").strip().lower(),
            attitude_quat_bn=attitude_quat_bn,
            angular_rate_body_rad_s=angular_rate_body_rad_s,
        )

    def state_at(self, t_s: float) -> StateTruth:
        t_s = float(t_s)
        jd_utc = float(self.start_jd_utc) + t_s / 86400.0
        tsince_min = (float(self.start_jd_utc) - float(self.elements.epoch_jd_utc)) * 1440.0 + t_s / 60.0
        native = sgp4_propagate_teme(self.elements, tsince_min)
        if native.error:
            raise ValueError(native.error)
        pos, vel = transform_teme_to_output_frame(
            native.position_teme_km,
            native.velocity_teme_km_s,
            jd_utc=jd_utc,
            output_frame=self.output_frame,
            frame_transform=self.frame_transform,
        )
        return StateTruth(
            position_eci_km=np.array(pos, dtype=float),
            velocity_eci_km_s=np.array(vel, dtype=float),
            attitude_quat_bn=(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                if self.attitude_quat_bn is None
                else np.array(self.attitude_quat_bn, dtype=float)
            ),
            angular_rate_body_rad_s=(
                np.zeros(3, dtype=float)
                if self.angular_rate_body_rad_s is None
                else np.array(self.angular_rate_body_rad_s, dtype=float)
            ),
            mass_kg=float(self.mass_kg),
            t_s=t_s,
        )

    def metadata(self) -> SGP4PropagationMetadata:
        start_age = float(self.start_jd_utc) - float(self.elements.epoch_jd_utc)
        end_age = start_age + float(self.duration_s) / 86400.0
        return SGP4PropagationMetadata(
            propagation_method="general",
            general_model="sgp4",
            native_frame="teme",
            output_frame=self.output_frame,
            frame_transform=self.frame_transform,
            tle_epoch_jd_utc=float(self.elements.epoch_jd_utc),
            tle_age_start_days=float(start_age),
            tle_age_end_days=float(end_age),
        )


def _actan(y: float, x: float) -> float:
    return float(math.atan2(y, x) % (2.0 * math.pi))


def _fmod2p(x: float) -> float:
    return float(x % (2.0 * math.pi))


def _sgp4_satdata(elements: TLEElements) -> dict[str, float]:
    minutes_per_day = 1440.0
    return {
        "xmo": math.radians(float(elements.mean_anomaly_deg)),
        "xnodeo": math.radians(float(elements.raan_deg)),
        "omegao": math.radians(float(elements.argp_deg)),
        "xincl": math.radians(float(elements.inclination_deg)),
        "eo": float(elements.eccentricity),
        "xno": float(elements.mean_motion_rev_per_day) * 2.0 * math.pi / minutes_per_day,
        "xndt2o": float(elements.mean_motion_derivative_rev_per_day2)
        * 2.0
        * math.pi
        / (minutes_per_day * minutes_per_day),
        "xndd6o": float(elements.mean_motion_second_derivative_rev_per_day3)
        * 2.0
        * math.pi
        / (minutes_per_day * minutes_per_day * minutes_per_day),
        "bstar": float(elements.bstar),
    }


def sgp4_propagate_teme(elements: TLEElements, tsince_min: float) -> SGP4State:
    satdata = _sgp4_satdata(elements)
    ae = 1.0
    tothrd = 2.0 / 3.0
    xj3 = -2.53881e-6
    e6a = 1.0e-6
    xkmper = 6378.135
    ge = 398600.8
    ck2 = 1.0826158e-3 / 2.0
    ck4 = -3.0 * -1.65597e-6 / 8.0

    if satdata["xno"] <= 0.0:
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 mean motion must be positive.")
    if satdata["eo"] < 0.0 or satdata["eo"] >= 1.0:
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 eccentricity must be in [0, 1).")

    s = ae + 78.0 / xkmper
    qo = ae + 120.0 / xkmper
    xke = math.sqrt((3600.0 * ge) / (xkmper**3))
    qoms2t = (qo - s) ** 4
    a1 = (xke / satdata["xno"]) ** tothrd
    cosio = math.cos(satdata["xincl"])
    theta2 = cosio * cosio
    x3thm1 = 3.0 * theta2 - 1.0
    eosq = satdata["eo"] * satdata["eo"]
    betao2 = 1.0 - eosq
    betao = math.sqrt(betao2)
    del1 = 1.5 * ck2 * x3thm1 / ((a1 * a1) * betao * betao2)
    ao = a1 * (1.0 - del1 * (1.0 / 3.0 + del1 * (1.0 + 134.0 / 81.0 * del1)))
    delo = 1.5 * ck2 * x3thm1 / ((ao * ao) * betao * betao2)
    xnodp = satdata["xno"] / (1.0 + delo)
    aodp = ao / (1.0 - delo)

    isimp = (aodp * (1.0 - satdata["eo"]) / ae) < (220.0 / xkmper + ae)
    s4 = s
    qoms24 = qoms2t
    perige = (aodp * (1.0 - satdata["eo"]) - ae) * xkmper
    if perige < 156.0:
        s4 = perige - 78.0
        if perige <= 98.0:
            s4 = 20.0
        qoms24 = ((120.0 - s4) * ae / xkmper) ** 4
        s4 = s4 / xkmper + ae

    pinvsq = 1.0 / ((aodp * aodp) * (betao2 * betao2))
    tsi = 1.0 / (aodp - s4)
    eta = aodp * satdata["eo"] * tsi
    etasq = eta * eta
    eeta = satdata["eo"] * eta
    psisq = abs(1.0 - etasq)
    coef = qoms24 * (tsi**4)
    coef1 = coef / (psisq**3.5)
    c2 = coef1 * xnodp * (
        aodp * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
        + 0.75 * ck2 * tsi / psisq * x3thm1 * (8.0 + 3.0 * etasq * (8.0 + etasq))
    )
    c1 = satdata["bstar"] * c2
    sinio = math.sin(satdata["xincl"])
    a3ovk2 = -xj3 / ck2 * (ae**3)
    c3 = 0.0 if abs(satdata["eo"]) < 1.0e-12 else coef * tsi * a3ovk2 * xnodp * ae * sinio / satdata["eo"]
    x1mth2 = 1.0 - theta2
    c4 = 2.0 * xnodp * coef1 * aodp * betao2 * (
        eta * (2.0 + 0.5 * etasq)
        + satdata["eo"] * (0.5 + 2.0 * etasq)
        - 2.0
        * ck2
        * tsi
        / (aodp * psisq)
        * (
            -3.0 * x3thm1 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
            + 0.75
            * x1mth2
            * (2.0 * etasq - eeta * (1.0 + etasq))
            * math.cos(2.0 * satdata["omegao"])
        )
    )
    c5 = 2.0 * coef1 * aodp * betao2 * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
    theta4 = theta2 * theta2
    temp1 = 3.0 * ck2 * pinvsq * xnodp
    temp2 = temp1 * ck2 * pinvsq
    temp3 = 1.25 * ck4 * pinvsq * pinvsq * xnodp
    xmdot = xnodp + 0.5 * temp1 * betao * x3thm1 + 0.0625 * temp2 * betao * (13.0 - 78.0 * theta2 + 137.0 * theta4)
    x1m5th = 1.0 - 5.0 * theta2
    omgdot = -0.5 * temp1 * x1m5th + 0.0625 * temp2 * (7.0 - 114.0 * theta2 + 395.0 * theta4) + temp3 * (3.0 - 36.0 * theta2 + 49.0 * theta4)
    xhdot1 = -temp1 * cosio
    xnodot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * theta2) + 2.0 * temp3 * (3.0 - 7.0 * theta2)) * cosio
    omgcof = satdata["bstar"] * c3 * math.cos(satdata["omegao"])
    xmcof = 0.0 if abs(eeta) < 1.0e-12 else -(2.0 / 3.0) * coef * satdata["bstar"] * ae / eeta
    xnodcf = 3.5 * betao2 * xhdot1 * c1
    t2cof = 1.5 * c1
    xlcof = 0.125 * a3ovk2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio)
    aycof = 0.25 * a3ovk2 * sinio
    delmo = (1.0 + eta * math.cos(satdata["xmo"])) ** 3
    sinmo = math.sin(satdata["xmo"])
    x7thm1 = 7.0 * theta2 - 1.0
    d2 = d3 = d4 = t3cof = t4cof = t5cof = 0.0
    if not isimp:
        c1sq = c1 * c1
        d2 = 4.0 * aodp * tsi * c1sq
        temp = d2 * tsi * c1 / 3.0
        d3 = (17.0 * aodp + s4) * temp
        d4 = 0.5 * temp * aodp * tsi * (221.0 * aodp + 31.0 * s4) * c1
        t3cof = d2 + 2.0 * c1sq
        t4cof = 0.25 * (3.0 * d3 + c1 * (12.0 * d2 + 10.0 * c1sq))
        t5cof = 0.2 * (3.0 * d4 + 12.0 * c1 * d3 + 6.0 * d2 * d2 + 15.0 * c1sq * (2.0 * d2 + c1sq))

    tsince = float(tsince_min)
    xmdf = satdata["xmo"] + xmdot * tsince
    omgadf = satdata["omegao"] + omgdot * tsince
    xnoddf = satdata["xnodeo"] + xnodot * tsince
    omega = omgadf
    xmp = xmdf
    tsq = tsince * tsince
    xnode = xnoddf + xnodcf * tsq
    tempa = 1.0 - c1 * tsince
    tempe = satdata["bstar"] * c4 * tsince
    templ = t2cof * tsq
    if not isimp:
        delomg = omgcof * tsince
        delm = xmcof * ((1.0 + eta * math.cos(xmdf)) ** 3 - delmo)
        temp = delomg + delm
        xmp = xmdf + temp
        omega = omgadf - temp
        tcube = tsq * tsince
        tfour = tsince * tcube
        tempa = tempa - d2 * tsq - d3 * tcube - d4 * tfour
        tempe = tempe + satdata["bstar"] * c5 * (math.sin(xmp) - sinmo)
        templ = templ + t3cof * tcube + tfour * (t4cof + tsince * t5cof)

    a = aodp * (tempa * tempa)
    e = satdata["eo"] - tempe
    if a <= 0.0 or e < 0.0 or e >= 1.0:
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 propagated orbit became invalid.")
    xl = xmp + omega + xnode + xnodp * templ
    beta = math.sqrt(1.0 - e * e)
    xn = xke / (a**1.5)
    axn = e * math.cos(omega)
    temp = 1.0 / (a * beta * beta)
    xll = temp * xlcof * axn
    aynl = temp * aycof
    xlt = xl + xll
    ayn = e * math.sin(omega) + aynl

    capu = _fmod2p(xlt - xnode)
    epw = capu
    sinepw = cosepw = temp3 = temp4 = temp5 = temp6 = 0.0
    for _ in range(10):
        sinepw = math.sin(epw)
        cosepw = math.cos(epw)
        temp3 = axn * sinepw
        temp4 = ayn * cosepw
        temp5 = axn * cosepw
        temp6 = ayn * sinepw
        next_epw = (capu - temp4 + temp3 - epw) / (1.0 - temp5 - temp6) + epw
        prev_epw = epw
        epw = next_epw
        if abs(next_epw - prev_epw) <= e6a:
            break
    ecose = temp5 + temp6
    esine = temp3 - temp4
    elsq = axn * axn + ayn * ayn
    temp = 1.0 - elsq
    pl = a * temp
    if pl <= 0.0:
        return SGP4State(np.zeros(3), np.zeros(3), "SGP4 semi-latus rectum became invalid.")
    r = a * (1.0 - ecose)
    temp1 = 1.0 / r
    rdot = xke * math.sqrt(a) * esine * temp1
    rfdot = xke * math.sqrt(pl) * temp1
    temp2 = a * temp1
    betal = math.sqrt(temp)
    temp3 = 1.0 / (1.0 + betal)
    cosu = temp2 * (cosepw - axn + ayn * esine * temp3)
    sinu = temp2 * (sinepw - ayn - axn * esine * temp3)
    u = _actan(sinu, cosu)
    sin2u = 2.0 * sinu * cosu
    cos2u = 2.0 * cosu * cosu - 1.0
    temp = 1.0 / pl
    temp1 = ck2 * temp
    temp2 = temp1 * temp
    rk = r * (1.0 - 1.5 * temp2 * betal * x3thm1) + 0.5 * temp1 * x1mth2 * cos2u
    uk = u - 0.25 * temp2 * x7thm1 * sin2u
    xnodek = xnode + 1.5 * temp2 * cosio * sin2u
    xinck = satdata["xincl"] + 1.5 * temp2 * cosio * sinio * cos2u
    rdotk = rdot - xn * temp1 * x1mth2 * sin2u
    rfdotk = rfdot + xn * temp1 * (x1mth2 * cos2u + 1.5 * x3thm1)

    mv = np.array([-math.sin(xnodek) * math.cos(xinck), math.cos(xnodek) * math.cos(xinck), math.sin(xinck)])
    nv = np.array([math.cos(xnodek), math.sin(xnodek), 0.0])
    uv = mv * math.sin(uk) + nv * math.cos(uk)
    vv = mv * math.cos(uk) - nv * math.sin(uk)
    pos_er = rk * uv
    vel_er_per_min = rdotk * uv + rfdotk * vv
    return SGP4State(
        position_teme_km=np.array(pos_er * xkmper, dtype=float),
        velocity_teme_km_s=np.array(vel_er_per_min * xkmper / 60.0, dtype=float),
    )


def transform_teme_to_output_frame(
    position_teme_km: np.ndarray,
    velocity_teme_km_s: np.ndarray,
    *,
    jd_utc: float,
    output_frame: str = "eci",
    frame_transform: str = "teme_as_eci",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the v1 OEL output-frame view of a propagated TEME state.

    v1 intentionally exposes only a transparent TEME-as-ECI approximation. A
    full TEME-to-ECI reduction needs EOP/nutation handling and is outside the
    initial public SGP4 scope.
    """
    frame = str(output_frame or "eci").strip().lower()
    transform = str(frame_transform or "teme_as_eci").strip().lower()
    if frame != "eci":
        raise ValueError("SGP4 v1 only supports output_frame='eci'.")
    if transform != "teme_as_eci":
        raise ValueError("SGP4 frame_transform must be 'teme_as_eci' for v1.")
    _ = float(jd_utc)
    return np.array(position_teme_km, dtype=float), np.array(velocity_teme_km_s, dtype=float)
