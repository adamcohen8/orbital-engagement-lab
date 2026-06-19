from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.epoch import datetime_to_julian_date, gmst_angle_rad_from_jd
from sim.dynamics.orbit.msis86_coeff import PD, PDL_0, PDL_1, PDM, PS, PT, PTM, SW, SWC


def _pad1(a: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.asarray(a, dtype=float).reshape(-1)))


def _pad2(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    out = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=float)
    out[1:, 1:] = arr
    return out


def _slice1(row: np.ndarray, start: int, stop: int) -> np.ndarray:
    return np.concatenate(([0.0], np.asarray(row[start : stop + 1], dtype=float)))


PT1 = _pad1(PT)
PS1 = _pad1(PS)
PD1 = _pad2(PD)
PDL01 = _pad1(PDL_0)
PDL11 = _pad1(PDL_1)
PTM1 = _pad1(PTM)
PDM1 = _pad2(PDM)
SW1 = _pad1(SW)
SWC1 = _pad1(SWC)


@dataclass
class _Input:
    doy: float
    sec: float
    alt: float
    glat: float
    glong: float
    stl: float
    f107a: float
    f107: float
    ap: np.ndarray


@dataclass
class _Output:
    d: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=float))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


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


def _solar_geomagnetic_inputs(dt_utc: datetime, env: dict) -> tuple[float, float, np.ndarray]:
    f107a = env.get("msis86_f107a", env.get("f107a"))
    f107 = env.get("msis86_f107", env.get("f107"))
    ap = env.get("msis86_ap", env.get("ap"))
    ap_a = env.get("msis86_ap_a")
    sw_path = env.get("msis86_sw_path")
    if sw_path not in (None, "") and (f107a is None or f107 is None or ap is None or ap_a is None):
        sw = _load_swdata(_resolve_table_path(str(sw_path)))
        matches = np.where((sw[:, 0] == dt_utc.year) & (sw[:, 1] == dt_utc.month) & (sw[:, 2] == dt_utc.day))[0]
        if matches.size == 0:
            raise RuntimeError(f"MSIS-86 SW data missing day {dt_utc.year}-{dt_utc.month}-{dt_utc.day}")
        i = int(matches[0])
        if i < 3:
            raise RuntimeError("MSIS-86 SW data needs at least three prior days for Ap history")
        sw0, sw1, sw2, sw3 = sw[i], sw[i - 1], sw[i - 2], sw[i - 3]
        if ap is None:
            ap = sw0[22]
        if ap_a is None:
            ap_a = [
                sw0[22],
                sw0[14],
                sw1[21],
                sw1[20],
                sw1[19],
                np.sum([sw1[18], sw1[17], sw1[16], sw1[15], sw1[14], sw2[21], sw2[20], sw2[19]]) / 8.0,
                np.sum([sw2[18], sw2[17], sw2[16], sw2[15], sw2[14], sw3[21], sw3[20], sw3[19]]) / 8.0,
            ]
        if f107 is None:
            f107 = sw1[29]
        if f107a is None:
            f107a = sw0[27]

    f107 = float(150.0 if f107 is None else f107)
    f107a = float(f107 if f107a is None else f107a)
    ap_scalar = float(4.0 if ap is None else ap)
    if ap_a is None:
        ap_arr = np.concatenate(([0.0], np.full(7, ap_scalar, dtype=float)))
    else:
        vals = list(ap_a)
        if len(vals) != 7:
            raise ValueError("env['msis86_ap_a'] must contain seven Ap values.")
        ap_arr = np.concatenate(([0.0], np.asarray(vals, dtype=float)))
    return f107a, f107, ap_arr


class _MSIS86:
    def __init__(self) -> None:
        self.gsurf = 980.665
        self.re = 6356.77
        self.sw = np.array(SW1, copy=True)
        self.swc = np.array(SWC1, copy=True)
        self.plg = np.zeros((9, 5), dtype=float)
        self.dfa = 0.0
        self.ctloc = self.stloc = self.c2tloc = self.s2tloc = self.c3tloc = self.s3tloc = 0.0
        self.apdf = 0.0
        self.day = 0.0
        self.apt = np.zeros(4, dtype=float)

    def _zeta(self, zz: float, zl: float) -> float:
        return (zz - zl) * (self.re + zl) / (self.re + zz)

    def _denss(
        self,
        alt: float,
        dlb: float,
        tinf: float,
        tlb: float,
        xm: float,
        alpha: float,
        zlb: float,
        s2: float,
        t0: float,
        za: float,
        z0: float,
        tr12: float,
    ) -> tuple[float, float]:
        z = max(float(alt), float(za))
        zg2 = self._zeta(z, zlb)
        tt = tinf - (tinf - tlb) * math.exp(-s2 * zg2)
        ta = tt
        tz = tt
        denss = tz
        if alt < za:
            zg0 = self._zeta(z0, za)
            dta = (tinf - ta) * s2 * ((self.re + zlb) / (self.re + za)) ** 2
            t12 = t0 + tr12 * (ta - t0)
            zg1 = self._zeta(alt, za)
            dd = 0.666666 * zg0 * dta / ta / ta - 3.11111 * (1.0 / ta - 1.0 / t0) + 7.11111 * (1.0 / t12 - 1.0 / t0)
            cc = zg0 * dta / (2.0 * ta * ta) - (1.0 / ta - 1.0 / t0) - 2.0 * dd
            bb = (1.0 / ta - 1.0 / t0) - cc - dd
            x = (zg0 - zg1) / zg0
            x2 = x * x
            tz = 1.0 / (1.0 / t0 + bb * x2 + cc * x2 * x2 + dd * x2 * x2 * x2)
            denss = tz
        else:
            zg0 = x = x2 = bb = cc = dd = 0.0
        if xm != 0.0:
            if ta <= 0.0 or tz <= 0.0:
                tt = tlb
                ta = tlb
                tz = tlb
            glb = self.gsurf / (1.0 + zlb / self.re) ** 2
            gamma = xm * glb / (s2 * 831.4 * tinf)
            densa = dlb * (tlb / tt) ** (1.0 + alpha + gamma) * math.exp(-s2 * gamma * zg2)
            denss = densa
            if alt < za:
                glb = self.gsurf / (1.0 + za / self.re) ** 2
                gamm = xm * glb * zg0 / 831.4
                denss = (
                    densa
                    * (ta / tz) ** (1.0 + alpha)
                    * math.exp(
                        gamm
                        * (
                            (x - 1.0) / t0
                            + bb * (x * x2 - 1.0) / 3.0
                            + cc * (x2 * x2 * x - 1.0) / 5.0
                            + dd * (x2 * x2 * x2 * x - 1.0) / 7.0
                        )
                    )
                )
        return float(tz), float(denss)

    @staticmethod
    def _g0(a: float, p25: float, p26: float) -> float:
        p25a = abs(p25)
        return a - 4.0 + (p26 - 1.0) * (a - 4.0 + (math.exp(-p25a * (a - 4.0)) - 1.0) / p25a)

    @staticmethod
    def _sumex(ex: float) -> float:
        return 1.0 + (1.0 - ex**19.0) / (1.0 - ex) * math.sqrt(ex)

    def _sg0(self, ex: float, ap: np.ndarray, p: np.ndarray) -> float:
        return (
            self._g0(ap[1], p[24], p[25])
            + (
                self._g0(ap[2], p[24], p[25]) * ex
                + self._g0(ap[3], p[24], p[25]) * ex * ex
                + self._g0(ap[4], p[24], p[25]) * ex**3
                + (self._g0(ap[5], p[24], p[25]) * ex**4 + self._g0(ap[6], p[24], p[25]) * ex**12)
                * (1.0 - ex**8)
                / (1.0 - ex)
            )
        ) / self._sumex(ex)

    def _legendre_local_time(self, lat: float, tloc: float) -> None:
        dgtr = 1.74533e-2
        hr = 0.2618
        c = math.sin(lat * dgtr)
        s = math.cos(lat * dgtr)
        c2 = c * c
        c4 = c2 * c2
        s2 = s * s
        p = self.plg
        p[2, 1] = c
        p[3, 1] = 0.5 * (3.0 * c2 - 1.0)
        p[4, 1] = 0.5 * (5.0 * c * c2 - 3.0 * c)
        p[5, 1] = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0
        p[6, 1] = (63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0
        p[7, 1] = (11.0 * c * p[6, 1] - 5.0 * p[5, 1]) / 6.0
        p[2, 2] = s
        p[3, 2] = 3.0 * c * s
        p[4, 2] = 1.5 * (5.0 * c2 - 1.0) * s
        p[5, 2] = 2.5 * (7.0 * c2 * c - 3.0 * c) * s
        p[6, 2] = 1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s
        p[7, 2] = (11.0 * c * p[6, 2] - 6.0 * p[5, 2]) / 5.0
        p[3, 3] = 3.0 * s2
        p[4, 3] = 15.0 * s2 * c
        p[5, 3] = 7.5 * (7.0 * c2 - 1.0) * s2
        p[6, 3] = 3.0 * c * p[5, 3] - 2.0 * p[4, 3]
        p[7, 3] = (11.0 * c * p[6, 3] - 7.0 * p[5, 3]) / 4.0
        p[8, 3] = (13.0 * c * p[7, 3] - 8.0 * p[6, 3]) / 5.0
        p[4, 4] = 15.0 * s2 * s
        p[5, 4] = 105.0 * s2 * s * c
        p[6, 4] = (9.0 * c * p[5, 4] - 7.0 * p[4, 4]) / 2.0
        p[7, 4] = (11.0 * c * p[6, 4] - 8.0 * p[5, 4]) / 3.0
        self.stloc = math.sin(hr * tloc)
        self.ctloc = math.cos(hr * tloc)
        self.s2tloc = math.sin(2.0 * hr * tloc)
        self.c2tloc = math.cos(2.0 * hr * tloc)
        self.s3tloc = math.sin(3.0 * hr * tloc)
        self.c3tloc = math.cos(3.0 * hr * tloc)

    def _globe5(
        self,
        yrd: float,
        sec: float,
        lat: float,
        along: float,
        tloc: float,
        f107a: float,
        f107: float,
        ap: np.ndarray,
        p_in: np.ndarray,
    ) -> float:
        p = np.array(p_in, dtype=float, copy=True)
        t = np.zeros(15, dtype=float)
        dgtr = 1.74533e-2
        dr = 1.72142e-2
        hr = 0.2618
        sr = 7.2722e-5
        self.day = yrd - math.trunc(yrd / 1000.0) * 1000.0
        self._legendre_local_time(lat, tloc)
        plg = self.plg
        cd14 = math.cos(dr * (self.day - p[14]))
        cd18 = math.cos(2.0 * dr * (self.day - p[18]))
        cd32 = math.cos(dr * (self.day - p[32]))
        cd39 = math.cos(2.0 * dr * (self.day - p[39]))

        df = f107 - f107a
        self.dfa = f107a - 150.0
        t[1] = p[20] * df + p[21] * df * df + p[22] * self.dfa + p[30] * self.dfa * self.dfa
        f1 = 1.0 + (p[48] * self.dfa + p[20] * df + p[21] * df * df) * self.swc[1]
        f2 = 1.0 + (p[50] * self.dfa + p[20] * df + p[21] * df * df) * self.swc[1]
        t[2] = (
            p[2] * plg[3, 1]
            + p[3] * plg[5, 1]
            + p[23] * plg[7, 1]
            + p[15] * plg[3, 1] * self.dfa * self.swc[1]
            + p[27] * plg[2, 1]
        )
        t[3] = p[19] * cd32
        t[4] = (p[16] + p[17] * plg[3, 1]) * cd18
        t[5] = f1 * (p[10] * plg[2, 1] + p[11] * plg[4, 1]) * cd14
        t[6] = p[38] * plg[2, 1] * cd39
        t71 = (p[12] * plg[3, 2] + p[36] * plg[2, 2]) * cd14 * self.swc[5]
        t72 = (p[13] * plg[3, 2] + p[37] * plg[2, 2]) * cd14 * self.swc[5]
        t[7] = f2 * (
            (p[4] * plg[2, 2] + p[5] * plg[4, 2] + p[28] * plg[6, 2] + t71) * self.ctloc
            + (p[7] * plg[2, 2] + p[8] * plg[4, 2] + p[29] * plg[6, 2] + t72) * self.stloc
        )
        t81 = p[24] * plg[4, 3] * cd14 * self.swc[5]
        t82 = p[34] * plg[4, 3] * cd14 * self.swc[5]
        t[8] = f2 * (
            (p[6] * plg[3, 3] + p[42] * plg[5, 3] + t81) * self.c2tloc
            + (p[9] * plg[3, 3] + p[43] * plg[5, 3] + t82) * self.s2tloc
        )
        t[14] = f2 * (
            (p[40] * plg[3, 3] + (p[94] * plg[5, 4] + p[47] * plg[7, 4]) * cd14 * self.swc[5]) * self.s3tloc
            + (p[41] * plg[4, 4] + (p[95] * plg[5, 4] + p[49] * plg[7, 4]) * cd14 * self.swc[5]) * self.c3tloc
        )
        if self.sw[9] == -1.0 and p[52] != 0.0:
            exp1 = min(math.exp(-10800.0 * abs(p[52]) / (1.0 + p[139] * (45.0 - abs(lat)))), 0.99999)
            exp2 = min(math.exp(-10800.0 * abs(p[54])), 0.99999)
            if p[25] < 1.0e-4:
                p[25] = 1.0e-4
            self.apt[1] = self._sg0(exp1, ap, p)
            self.apt[3] = self._sg0(exp2, ap, p)
            t[9] = self.apt[1] * (
                p[51]
                + p[97] * plg[3, 1]
                + p[55] * plg[5, 1]
                + (p[126] * plg[2, 1] + p[127] * plg[4, 1] + p[128] * plg[6, 1]) * cd14 * self.swc[5]
                + (p[129] * plg[2, 2] + p[130] * plg[4, 2] + p[131] * plg[6, 2])
                * self.swc[7]
                * math.cos(hr * (tloc - p[132]))
            )
        else:
            apd = ap[1] - 4.0
            p44 = max(p[44], 1.0e-5)
            p45 = p[45]
            self.apdf = apd + (p45 - 1.0) * (apd + (math.exp(-p44 * apd) - 1.0) / p44)
            t[9] = self.apdf * (
                p[33]
                + p[46] * plg[3, 1]
                + p[35] * plg[5, 1]
                + (p[101] * plg[2, 1] + p[102] * plg[4, 1] + p[103] * plg[6, 1]) * cd14 * self.swc[5]
                + (p[122] * plg[2, 2] + p[123] * plg[4, 2] + p[124] * plg[6, 2])
                * self.swc[7]
                * math.cos(hr * (tloc - p[125]))
            )
        if self.sw[10] != 0.0 and along > -1000.0:
            t[11] = (
                (1.0 + p[90] * plg[2, 1])
                * (1.0 + p[81] * self.dfa * self.swc[1])
                * (
                    (
                        p[65] * plg[3, 2]
                        + p[66] * plg[5, 2]
                        + p[67] * plg[7, 2]
                        + p[104] * plg[2, 2]
                        + p[105] * plg[4, 2]
                        + p[106] * plg[6, 2]
                        + self.swc[5] * (p[110] * plg[2, 2] + p[111] * plg[4, 2] + p[112] * plg[6, 2]) * cd14
                    )
                    * math.cos(dgtr * along)
                    + (
                        p[91] * plg[3, 2]
                        + p[92] * plg[5, 2]
                        + p[93] * plg[7, 2]
                        + p[107] * plg[2, 2]
                        + p[108] * plg[4, 2]
                        + p[109] * plg[6, 2]
                        + self.swc[5] * (p[113] * plg[2, 2] + p[114] * plg[4, 2] + p[115] * plg[6, 2]) * cd14
                    )
                    * math.sin(dgtr * along)
                )
            )
            t[12] = (
                (1.0 + p[96] * plg[2, 1])
                * (1.0 + p[82] * self.dfa * self.swc[1])
                * (1.0 + p[120] * plg[2, 1] * self.swc[5] * cd14)
                * (p[69] * plg[2, 1] + p[70] * plg[4, 1] + p[71] * plg[6, 1])
                * math.cos(sr * (sec - p[72]))
            )
            t[12] += (
                self.swc[11]
                * (p[77] * plg[4, 3] + p[78] * plg[6, 3] + p[79] * plg[8, 3])
                * math.cos(sr * (sec - p[80]) + 2.0 * dgtr * along)
                * (1.0 + p[138] * self.dfa * self.swc[1])
            )
            if self.sw[9] == -1.0 and p[52] != 0.0:
                t[13] = (
                    self.apt[1]
                    * self.swc[11]
                    * (1.0 + p[133] * plg[2, 1])
                    * (p[53] * plg[3, 2] + p[99] * plg[5, 2] + p[68] * plg[7, 2])
                    * math.cos(dgtr * (along - p[98]))
                    + self.apt[1]
                    * self.swc[11]
                    * self.swc[5]
                    * (p[134] * plg[2, 2] + p[135] * plg[4, 2] + p[136] * plg[6, 2])
                    * cd14
                    * math.cos(dgtr * (along - p[137]))
                    + self.apt[1]
                    * self.swc[12]
                    * (p[56] * plg[2, 1] + p[57] * plg[4, 1] + p[58] * plg[6, 1])
                    * math.cos(sr * (sec - p[59]))
                )
            else:
                t[13] = (
                    self.apdf
                    * self.swc[11]
                    * (1.0 + p[121] * plg[2, 1])
                    * (p[61] * plg[3, 2] + p[62] * plg[5, 2] + p[63] * plg[7, 2])
                    * math.cos(dgtr * (along - p[64]))
                    + self.apdf
                    * self.swc[11]
                    * self.swc[5]
                    * (p[116] * plg[2, 2] + p[117] * plg[4, 2] + p[118] * plg[6, 2])
                    * cd14
                    * math.cos(dgtr * (along - p[119]))
                    + self.apdf
                    * self.swc[12]
                    * (p[84] * plg[2, 1] + p[85] * plg[4, 1] + p[86] * plg[6, 1])
                    * math.cos(sr * (sec - p[76]))
                )
        tinf = p[31] if self.sw[9] == -1.0 else 0.0
        for i in range(1, 15):
            tinf += abs(self.sw[i]) * t[i]
        return float(tinf)

    def _glob5l(self, p: np.ndarray) -> float:
        dr = 1.72142e-2
        t = np.zeros(15, dtype=float)
        cd7 = math.cos(dr * (self.day - p[7]))
        cd9 = math.cos(2.0 * dr * (self.day - p[9]))
        cd11 = math.cos(dr * (self.day - p[11]))
        plg = self.plg
        t[1] = p[2] * self.dfa
        t[2] = p[4] * plg[3, 1]
        t[3] = p[6] * cd7
        t[4] = p[8] * cd9
        t[5] = (p[10] * plg[2, 1] + p[22] * plg[4, 1]) * cd11
        t[7] = p[14] * plg[2, 2] * self.ctloc + p[15] * plg[2, 2] * self.stloc
        t[8] = (p[16] * plg[3, 3] + p[18] * plg[5, 3] + p[20] * plg[6, 3] * cd11 * self.swc[5]) * self.c2tloc + (
            p[17] * plg[3, 3] + p[19] * plg[5, 3] + p[21] * plg[6, 3] * cd11 * self.swc[5]
        ) * self.s2tloc
        t[14] = p[12] * plg[4, 4] * self.c3tloc + p[25] * plg[4, 4] * self.s3tloc
        if self.sw[9] == 1.0:
            t[9] = self.apdf * (p[23] + p[24] * plg[3, 1] * self.swc[2])
        if self.sw[9] == -1.0:
            t[9] = p[3] * self.apt[3] + p[5] * plg[3, 1] * self.apt[3] * self.swc[2]
        return float(sum(abs(self.sw[i]) * t[i] for i in range(1, 15)))

    @staticmethod
    def _dnet(dd: float, dm: float, zhm: float, xmm: float, xm: float) -> float:
        if dd <= 0.0 or dm <= 0.0:
            return max(dd, dm, 0.0)
        a = zhm / (xmm - xm)
        ylog = a * math.log(dm / dd)
        if ylog < -10.0:
            return dd
        if ylog > 10.0:
            return dm
        return dd * (1.0 + math.exp(ylog)) ** (1.0 / a)

    @staticmethod
    def _ccor(alt: float, r: float, h1: float, zh: float) -> float:
        e = (alt - zh) / h1
        if e > 70.0:
            ccor = 0.0
        elif e < -70.0:
            ccor = r
        else:
            ccor = r / (1.0 + math.exp(e))
        return math.exp(ccor)

    def density(self, input_: _Input) -> float:
        d = np.zeros(9, dtype=float)
        t = np.zeros(3, dtype=float)
        alt = input_.alt
        altl = np.array([0.0, 200.0, 400.0, 150.0, 200.0, 240.0, 450.0, 320.0, 450.0], dtype=float)

        gggg = self._globe5(
            input_.doy, input_.sec, input_.glat, input_.glong, input_.stl, input_.f107a, input_.f107, input_.ap, PT1
        )
        tinf = PTM1[1] * (1.0 + self.sw[16] * gggg) * PT1[1]
        za = PTM1[5] * PDL11[16]
        t0 = PTM1[3] * PD1[3, 76] * (1.0 + self.sw[18] * self._glob5l(_slice1(PD1[3, :], 76, 100)))
        tlb = PTM1[2] * (1.0 + self.sw[17] * self._glob5l(_slice1(PD1[3, :], 26, 50))) * PD1[3, 26]
        z0 = PTM1[7] * (1.0 + self.sw[20] * self._glob5l(_slice1(PD1[3, :], 51, 75))) * PD1[3, 51]
        g0 = (
            PTM1[4]
            * PS1[1]
            * (
                1.0
                + self.sw[19]
                * self._globe5(
                    input_.doy,
                    input_.sec,
                    input_.glat,
                    input_.glong,
                    input_.stl,
                    input_.f107a,
                    input_.f107,
                    input_.ap,
                    PS1,
                )
            )
        )
        s = g0 / (tinf - tlb)
        tr12 = PD1[3, 101] * (1.0 + self.sw[22] * self._glob5l(_slice1(PD1[3, :], 101, 125)))
        t[1] = tinf
        xmm = PDM1[5, 3]

        def dens(alt_i: float, dlb: float, xm: float, alpha: float) -> tuple[float, float]:
            return self._denss(alt_i, dlb, tinf, tlb, xm, alpha, PTM1[6], s, t0, za, z0, tr12)

        g28 = self.sw[21] * self._glob5l(_slice1(PD1[3, :], 1, 25))
        db28 = PDM1[1, 3] * math.exp(g28) * PD1[3, 1]
        t[2], d[3] = dens(alt, db28, 28.0, 0.0)
        zh28 = PDM1[3, 3]
        zhm28 = PDM1[4, 3] * PDL11[6]
        _, b28 = dens(zh28, db28, 28.0 - xmm, -1.0)
        if alt < altl[3] and self.sw[15] != 0.0:
            _, dm28 = dens(alt, b28, xmm, 0.0)
            d[3] = self._dnet(d[3], dm28, zhm28, xmm, 28.0)

        g4 = self.sw[21] * self._globe5(
            input_.doy,
            input_.sec,
            input_.glat,
            input_.glong,
            input_.stl,
            input_.f107a,
            input_.f107,
            input_.ap,
            PD1[1, :],
        )
        db04 = PDM1[1, 1] * math.exp(g4) * PD1[1, 1]
        t[2], d[1] = dens(alt, db04, 4.0, -0.40)
        if alt < altl[1] and self.sw[15] != 0.0:
            t[2], b04 = dens(PDM1[3, 1], db04, 4.0 - xmm, -1.40)
            t[2], dm04 = dens(alt, b04, xmm, 0.0)
            d[1] = self._dnet(d[1], dm04, zhm28, xmm, 4.0)
            d[1] *= self._ccor(alt, math.log(b28 * PDM1[2, 1] / b04), PDM1[6, 2] * PDL11[2], PDM1[5, 1] * PDL11[1])

        g16 = self.sw[21] * self._globe5(
            input_.doy,
            input_.sec,
            input_.glat,
            input_.glong,
            input_.stl,
            input_.f107a,
            input_.f107,
            input_.ap,
            PD1[2, :],
        )
        db16 = PDM1[1, 2] * math.exp(g16) * PD1[2, 1]
        t[2], d[2] = dens(alt, db16, 16.0, 0.0)
        if alt <= altl[2] and self.sw[15] != 0.0:
            t[2], b16 = dens(PDM1[3, 2], db16, 16.0 - xmm, -1.0)
            t[2], dm16 = dens(alt, b16, xmm, 0.0)
            d[2] = self._dnet(d[2], dm16, zhm28, xmm, 16.0)
            d[2] *= self._ccor(
                alt, math.log(b28 * PDM1[2, 2] * abs(PDL11[17]) / b16), PDM1[6, 2] * PDL11[4], PDM1[5, 2] * PDL11[3]
            )
            d[2] *= self._ccor(alt, PDM1[4, 2] * PDL11[15], PDM1[8, 2] * PDL11[14], PDM1[7, 2] * PDL11[13])

        g32 = self.sw[21] * self._globe5(
            input_.doy,
            input_.sec,
            input_.glat,
            input_.glong,
            input_.stl,
            input_.f107a,
            input_.f107,
            input_.ap,
            PD1[4, :],
        )
        db32 = PDM1[1, 4] * math.exp(g32) * PD1[4, 1]
        t[2], d[4] = dens(alt, db32, 32.0, 0.0)
        if alt <= altl[4] and self.sw[15] != 0.0:
            t[2], b32 = dens(PDM1[3, 4], db32, 32.0 - xmm, -1.0)
            t[2], dm32 = dens(alt, b32, xmm, 0.0)
            d[4] = self._dnet(d[4], dm32, zhm28, xmm, 32.0)
            d[4] *= self._ccor(alt, math.log(b28 * PDM1[2, 4] / b32), PDM1[6, 4] * PDL11[8], PDM1[5, 4] * PDL11[7])

        g40 = self.sw[21] * self._globe5(
            input_.doy,
            input_.sec,
            input_.glat,
            input_.glong,
            input_.stl,
            input_.f107a,
            input_.f107,
            input_.ap,
            PD1[5, :],
        )
        db40 = PDM1[1, 5] * math.exp(g40) * PD1[5, 1]
        t[2], d[5] = dens(alt, db40, 40.0, 0.0)
        if alt <= altl[5] and self.sw[15] != 0.0:
            t[2], b40 = dens(PDM1[3, 5], db40, 40.0 - xmm, -1.0)
            t[2], dm40 = dens(alt, b40, xmm, 0.0)
            d[5] = self._dnet(d[5], dm40, zhm28, xmm, 40.0)
            d[5] *= self._ccor(alt, math.log(b28 * PDM1[2, 5] / b40), PDM1[6, 5] * PDL11[10], PDM1[5, 5] * PDL11[9])

        g1 = self.sw[21] * self._globe5(
            input_.doy,
            input_.sec,
            input_.glat,
            input_.glong,
            input_.stl,
            input_.f107a,
            input_.f107,
            input_.ap,
            PD1[6, :],
        )
        db01 = PDM1[1, 6] * math.exp(g1) * PD1[6, 1]
        t[2], d[7] = dens(alt, db01, 1.0, -0.40)
        if alt <= altl[7] and self.sw[15] != 0.0:
            t[2], b01 = dens(PDM1[3, 6], db01, 1.0 - xmm, -1.40)
            t[2], dm01 = dens(alt, b01, xmm, 0.0)
            d[7] = self._dnet(d[7], dm01, zhm28, xmm, 1.0)
            d[7] *= self._ccor(
                alt, math.log(b28 * PDM1[2, 6] * abs(PDL11[18]) / b01), PDM1[6, 6] * PDL11[12], PDM1[5, 6] * PDL11[11]
            )
            d[7] *= self._ccor(alt, PDM1[4, 6] * PDL11[21], PDM1[8, 6] * PDL11[20], PDM1[7, 6] * PDL11[19])

        g14 = self.sw[21] * self._globe5(
            input_.doy,
            input_.sec,
            input_.glat,
            input_.glong,
            input_.stl,
            input_.f107a,
            input_.f107,
            input_.ap,
            PD1[7, :],
        )
        db14 = PDM1[1, 7] * math.exp(g14) * PD1[7, 1]
        t[2], d[8] = dens(alt, db14, 14.0, 0.0)
        if alt <= altl[8] and self.sw[15] != 0.0:
            t[2], b14 = dens(PDM1[3, 7], db14, 14.0 - xmm, -1.0)
            t[2], dm14 = dens(alt, b14, xmm, 0.0)
            d[8] = self._dnet(d[8], dm14, zhm28, xmm, 14.0)
            d[8] *= self._ccor(
                alt, math.log(b28 * PDM1[2, 7] * abs(PDL01[3]) / b14), PDM1[6, 7] * PDL01[2], PDM1[5, 7] * PDL01[1]
            )
            d[8] *= self._ccor(alt, PDM1[4, 7] * PDL01[6], PDM1[8, 7] * PDL01[5], PDM1[7, 7] * PDL01[4])

        d[6] = 1.66e-24 * (4.0 * d[1] + 16.0 * d[2] + 28.0 * d[3] + 32.0 * d[4] + 40.0 * d[5] + d[7] + 14.0 * d[8])
        return float(max(0.0, 1000.0 * d[6]))


def msis86_density(alt_km: float, lat_deg: float, lon_deg: float, dt_utc: datetime, env: dict | None = None) -> float:
    env = {} if env is None else dict(env)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)
    f107a, f107, ap = _solar_geomagnetic_inputs(dt_utc, env)
    jd_utc = datetime_to_julian_date(dt_utc)
    lst_hr = env.get("msis86_lst_hr")
    if lst_hr is None:
        lst_hr = (
            ((math.radians(float(lon_deg)) + gmst_angle_rad_from_jd(jd_utc)) % (2.0 * math.pi)) * 24.0 / (2.0 * math.pi)
        )
    sec = dt_utc.hour * 3600.0 + dt_utc.minute * 60.0 + dt_utc.second + dt_utc.microsecond * 1e-6
    input_ = _Input(
        doy=_day_of_year(dt_utc),
        sec=sec,
        alt=float(max(85.0, alt_km)),
        glat=float(lat_deg),
        glong=float(lon_deg),
        stl=float(lst_hr),
        f107a=f107a,
        f107=f107,
        ap=ap,
    )
    return _MSIS86().density(input_)
