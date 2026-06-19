from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.epoch import datetime_to_julian_date, gmst_angle_rad_from_jd
from sim.dynamics.orbit.nrlmsise00_coeff import PAVGM, PD, PDL, PDM, PMA, PS, PT, PTL, PTM


def _pad1(a: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.asarray(a, dtype=float).reshape(-1)))


def _pad2(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    out = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=float)
    out[1:, 1:] = arr
    return out


PT1 = _pad1(PT)
PS1 = _pad1(PS)
PD1 = _pad2(PD)
PDL1 = _pad2(PDL)
PTM1 = _pad1(PTM)
PDM1 = _pad2(PDM)
PTL1 = _pad2(PTL)
PMA1 = _pad2(PMA)
PAVGM1 = _pad1(PAVGM)


@dataclass
class _Flags:
    switches: np.ndarray = field(default_factory=lambda: np.zeros(25, dtype=float))
    sw: np.ndarray = field(default_factory=lambda: np.zeros(25, dtype=float))
    swc: np.ndarray = field(default_factory=lambda: np.zeros(25, dtype=float))


@dataclass
class _Input:
    doy: int
    sec: float
    alt: float
    g_lat: float
    g_long: float
    lst: float
    f107a: float
    f107: float
    ap: float
    ap_a: np.ndarray


@dataclass
class _Output:
    d: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=float))
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


def _solar_geomagnetic_inputs(dt_utc: datetime, env: dict) -> tuple[float, float, float, np.ndarray]:
    f107a = env.get("nrlmsise00_f107a", env.get("f107a"))
    f107 = env.get("nrlmsise00_f107", env.get("f107"))
    ap = env.get("nrlmsise00_ap", env.get("ap"))
    ap_a = env.get("nrlmsise00_ap_a")
    sw_path = env.get("nrlmsise00_sw_path", env.get("msis_sw_path"))
    if sw_path not in (None, "") and (f107a is None or f107 is None or ap is None or ap_a is None):
        sw = _load_swdata(_resolve_table_path(str(sw_path)))
        matches = np.where((sw[:, 0] == dt_utc.year) & (sw[:, 1] == dt_utc.month) & (sw[:, 2] == dt_utc.day))[0]
        if matches.size == 0:
            raise RuntimeError(f"NRLMSISE-00 SW data missing day {dt_utc.year}-{dt_utc.month}-{dt_utc.day}")
        i = int(matches[0])
        if i < 3:
            raise RuntimeError("NRLMSISE-00 SW data needs at least three prior days for ap_a history")
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
    ap = float(4.0 if ap is None else ap)
    if ap_a is None:
        ap_a_arr = np.array([0.0, ap, ap, ap, ap, ap, ap, ap], dtype=float)
    else:
        vals = list(ap_a)
        if len(vals) != 7:
            raise ValueError("env['nrlmsise00_ap_a'] must contain seven Ap values.")
        ap_a_arr = np.concatenate(([0.0], np.asarray(vals, dtype=float)))
    return f107a, f107, ap, ap_a_arr


class _NRLMSISE00:
    def __init__(self) -> None:
        self.gsurf = 0.0
        self.re = 6367.088132098377
        self.dd = 0.0
        self.dm04 = self.dm16 = self.dm28 = self.dm32 = self.dm40 = self.dm01 = self.dm14 = 0.0
        self.meso_tn1 = np.zeros(6, dtype=float)
        self.meso_tn2 = np.zeros(5, dtype=float)
        self.meso_tn3 = np.zeros(6, dtype=float)
        self.meso_tgn1 = np.zeros(3, dtype=float)
        self.meso_tgn2 = np.zeros(3, dtype=float)
        self.meso_tgn3 = np.zeros(3, dtype=float)
        self.dfa = 0.0
        self.plg = np.zeros((5, 10), dtype=float)
        self.ctloc = self.stloc = self.c2tloc = self.s2tloc = self.s3tloc = self.c3tloc = 0.0
        self.apdf = 0.0
        self.apt = np.zeros(5, dtype=float)

    @staticmethod
    def _tselec(flags: _Flags) -> _Flags:
        for i in range(1, 25):
            if i != 10:
                flags.sw[i] = 1.0 if flags.switches[i] == 1 else 0.0
                flags.swc[i] = 1.0 if flags.switches[i] > 0 else 0.0
            else:
                flags.sw[i] = flags.switches[i]
                flags.swc[i] = flags.switches[i]
        return flags

    @staticmethod
    def _glatf(lat: float) -> tuple[float, float]:
        dgtr = 1.74533e-2
        c2 = math.cos(2.0 * dgtr * lat)
        gv = 980.616 * (1.0 - 0.0026373 * c2)
        reff = 2.0 * gv / (3.085462e-6 + 2.27e-9 * c2) * 1.0e-5
        return gv, reff

    @staticmethod
    def _ccor(alt: float, r: float, h1: float, zh: float) -> float:
        e = (alt - zh) / h1
        if e > 70.0:
            return 1.0
        if e < -70.0:
            return math.exp(r)
        return math.exp(r / (1.0 + math.exp(e)))

    @staticmethod
    def _ccor2(alt: float, r: float, h1: float, zh: float, h2: float) -> float:
        e1 = (alt - zh) / h1
        e2 = (alt - zh) / h2
        if e1 > 70.0 or e2 > 70.0:
            return 1.0
        if e1 < -70.0 and e2 < -70.0:
            return math.exp(r)
        return math.exp(r / (1.0 + 0.5 * (math.exp(e1) + math.exp(e2))))

    def _scalh(self, alt: float, xm: float, temp: float) -> float:
        g = self.gsurf / ((1.0 + alt / self.re) ** 2.0)
        return 831.4 * temp / (g * xm)

    @staticmethod
    def _dnet(dd: float, dm: float, zhm: float, xmm: float, xm: float) -> float:
        a = zhm / (xmm - xm)
        if not (dm > 0.0 and dd > 0.0):
            if dd == 0.0 and dm == 0.0:
                dd = 1.0
            if dm == 0.0:
                return dd
            if dd == 0.0:
                return dm
        ylog = a * math.log(dm / dd)
        if ylog < -10.0:
            return dd
        if ylog > 10.0:
            return dm
        return dd * (1.0 + math.exp(ylog)) ** (1.0 / a)

    @staticmethod
    def _spline(x: np.ndarray, y: np.ndarray, n: int, yp1: float, ypn: float) -> np.ndarray:
        u = np.zeros(max(6, n + 1), dtype=float)
        y2 = np.zeros(max(6, n + 1), dtype=float)
        if yp1 > 0.99e30:
            y2[1] = 0.0
            u[1] = 0.0
        else:
            y2[1] = -0.5
            u[1] = (3.0 / (x[2] - x[1])) * ((y[2] - y[1]) / (x[2] - x[1]) - yp1)
        for i in range(2, n):
            sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
            p = sig * y2[i - 1] + 2.0
            y2[i] = (sig - 1.0) / p
            u[i] = (
                6.0
                * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
                / (x[i + 1] - x[i - 1])
                - sig * u[i - 1]
            ) / p
        if ypn > 0.99e30:
            qn = 0.0
            un = 0.0
        else:
            qn = 0.5
            un = (3.0 / (x[n] - x[n - 1])) * (ypn - (y[n] - y[n - 1]) / (x[n] - x[n - 1]))
        y2[n] = (un - qn * u[n - 1]) / (qn * y2[n - 1] + 1.0)
        for k in range(n - 1, 0, -1):
            y2[k] = y2[k] * y2[k + 1] + u[k]
        return y2

    @staticmethod
    def _splint(xa: np.ndarray, ya: np.ndarray, y2a: np.ndarray, n: int, x: float) -> float:
        klo = 1
        khi = n
        while khi - klo > 1:
            k = int((khi + klo) / 2)
            if xa[k] > x:
                khi = k
            else:
                klo = k
        h = xa[khi] - xa[klo]
        if h == 0.0:
            return float(ya[klo])
        a = (xa[khi] - x) / h
        b = (x - xa[klo]) / h
        return float(a * ya[klo] + b * ya[khi] + ((a**3 - a) * y2a[klo] + (b**3 - b) * y2a[khi]) * h * h / 6.0)

    @staticmethod
    def _splini(xa: np.ndarray, ya: np.ndarray, y2a: np.ndarray, n: int, x: float) -> float:
        yi = 0.0
        klo = 0
        khi = 1
        while x > xa[klo + 1] and khi < n:
            xx = x
            if khi < n - 1:
                xx = x if x < xa[khi + 1] else xa[khi + 1]
            h = xa[khi + 1] - xa[klo + 1]
            a = (xa[khi + 1] - xx) / h
            b = (xx - xa[klo + 1]) / h
            a2 = a * a
            b2 = b * b
            yi += (
                (1.0 - a2) * ya[klo + 1] / 2.0
                + b2 * ya[khi + 1] / 2.0
                + ((-(1.0 + a2 * a2) / 4.0 + a2 / 2.0) * y2a[klo + 1] + (b2 * b2 / 4.0 - b2 / 2.0) * y2a[khi + 1])
                * h
                * h
                / 6.0
            ) * h
            klo += 1
            khi += 1
        return float(yi)

    def _zeta(self, zz: float, zl: float) -> float:
        return (zz - zl) * (self.re + zl) / (self.re + zz)

    def _densm(
        self,
        alt: float,
        d0: float,
        xm: float,
        tz: float,
        mn3: int,
        zn3: np.ndarray,
        tn3: np.ndarray,
        tgn3: np.ndarray,
        mn2: int,
        zn2: np.ndarray,
        tn2: np.ndarray,
        tgn2: np.ndarray,
    ) -> tuple[float, float]:
        xs = np.zeros(11, dtype=float)
        ys = np.zeros(11, dtype=float)
        densm_tmp = d0
        if alt > zn2[1]:
            return (tz if xm == 0.0 else d0), tz

        z = alt if alt > zn2[mn2] else zn2[mn2]
        z1, z2 = zn2[1], zn2[mn2]
        t1, t2 = tn2[1], tn2[mn2]
        zg = self._zeta(z, z1)
        zgdif = self._zeta(z2, z1)
        for k in range(1, mn2 + 1):
            xs[k] = self._zeta(zn2[k], z1) / zgdif
            ys[k] = 1.0 / tn2[k]
        yd1 = -tgn2[1] / (t1 * t1) * zgdif
        yd2 = -tgn2[2] / (t2 * t2) * zgdif * (((self.re + z2) / (self.re + z1)) ** 2.0)
        y2out = self._spline(xs, ys, mn2, yd1, yd2)
        x = zg / zgdif
        tz = 1.0 / self._splint(xs, ys, y2out, mn2, x)
        if xm != 0.0:
            glb = self.gsurf / ((1.0 + z1 / self.re) ** 2.0)
            gamm = xm * glb * zgdif / 831.4
            expl = min(50.0, gamm * self._splini(xs, ys, y2out, mn2, x))
            densm_tmp = densm_tmp * (t1 / tz) * math.exp(-expl)
        if alt > zn3[1]:
            return (tz if xm == 0.0 else densm_tmp), tz

        z = alt
        z1, z2 = zn3[1], zn3[mn3]
        t1, t2 = tn3[1], tn3[mn3]
        zg = self._zeta(z, z1)
        zgdif = self._zeta(z2, z1)
        for k in range(1, mn3 + 1):
            xs[k] = self._zeta(zn3[k], z1) / zgdif
            ys[k] = 1.0 / tn3[k]
        yd1 = -tgn3[1] / (t1 * t1) * zgdif
        yd2 = -tgn3[2] / (t2 * t2) * zgdif * (((self.re + z2) / (self.re + z1)) ** 2.0)
        y2out = self._spline(xs, ys, mn3, yd1, yd2)
        x = zg / zgdif
        tz = 1.0 / self._splint(xs, ys, y2out, mn3, x)
        if xm != 0.0:
            glb = self.gsurf / ((1.0 + z1 / self.re) ** 2.0)
            gamm = xm * glb * zgdif / 831.4
            expl = min(50.0, gamm * self._splini(xs, ys, y2out, mn3, x))
            densm_tmp = densm_tmp * (t1 / tz) * math.exp(-expl)
        return (tz if xm == 0.0 else densm_tmp), tz

    def _densu(
        self,
        alt: float,
        dlb: float,
        tinf: float,
        tlb: float,
        xm: float,
        alpha: float,
        tz: float,
        zlb: float,
        s2: float,
        mn1: int,
        zn1: np.ndarray,
        tn1: np.ndarray,
        tgn1: np.ndarray,
    ) -> tuple[float, float]:
        xs = np.zeros(6, dtype=float)
        ys = np.zeros(6, dtype=float)
        za = zn1[1]
        z = alt if alt > za else za
        zg2 = self._zeta(z, zlb)
        tt = tinf - (tinf - tlb) * math.exp(-s2 * zg2)
        ta = tt
        tz = tt
        densu_temp = tz
        x = 0.0
        y2out = np.zeros(6, dtype=float)
        if alt < za:
            dta = (tinf - ta) * s2 * ((self.re + zlb) / (self.re + za)) ** 2.0
            tgn1[1] = dta
            tn1[1] = ta
            z = alt if alt > zn1[mn1] else zn1[mn1]
            z1, z2 = zn1[1], zn1[mn1]
            t1, t2 = tn1[1], tn1[mn1]
            zg = self._zeta(z, z1)
            zgdif = self._zeta(z2, z1)
            for k in range(1, mn1 + 1):
                xs[k] = self._zeta(zn1[k], z1) / zgdif
                ys[k] = 1.0 / tn1[k]
            yd1 = -tgn1[1] / (t1 * t1) * zgdif
            yd2 = -tgn1[2] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)) ** 2.0
            y2out = self._spline(xs, ys, mn1, yd1, yd2)
            x = zg / zgdif
            tz = 1.0 / self._splint(xs, ys, y2out, mn1, x)
            densu_temp = tz
        if xm == 0.0:
            return densu_temp, tz
        glb = self.gsurf / (1.0 + zlb / self.re) ** 2.0
        gamma = xm * glb / (s2 * 831.4 * tinf)
        expl = math.exp(-s2 * gamma * zg2)
        if expl > 50.0 or tt <= 0.0:
            expl = 50.0
        densa = dlb * (tlb / tt) ** (1.0 + alpha + gamma) * expl
        if alt >= za:
            return densa, tz
        glb = self.gsurf / (1.0 + zn1[1] / self.re) ** 2.0
        gamm = xm * glb * self._zeta(zn1[mn1], zn1[1]) / 831.4
        expl2 = min(50.0, gamm * self._splini(xs, ys, y2out, mn1, x))
        if tz <= 0.0:
            expl2 = 50.0
        return densa * (tn1[1] / tz) ** (1.0 + alpha) * math.exp(-expl2), tz

    @staticmethod
    def _g0(a: float, p: np.ndarray) -> float:
        p25 = abs(p[25])
        return a - 4.0 + (p[26] - 1.0) * (a - 4.0 + (math.exp(-p25 * (a - 4.0)) - 1.0) / p25)

    @staticmethod
    def _sumex(ex: float) -> float:
        return 1.0 + (1.0 - ex**19.0) / (1.0 - ex) * ex**0.5

    def _sg0(self, ex: float, p: np.ndarray, ap: np.ndarray) -> float:
        return (
            self._g0(ap[2], p)
            + (
                self._g0(ap[3], p) * ex
                + self._g0(ap[4], p) * ex * ex
                + self._g0(ap[5], p) * ex**3.0
                + (self._g0(ap[6], p) * ex**4.0 + self._g0(ap[7], p) * ex**12.0) * (1.0 - ex**8.0) / (1.0 - ex)
            )
        ) / self._sumex(ex)

    def _legendre_and_local_time(self, input_: _Input, flags: _Flags) -> None:
        dgtr = 1.74533e-2
        hr = 0.2618
        c = math.sin(input_.g_lat * dgtr)
        s = math.cos(input_.g_lat * dgtr)
        c2 = c * c
        c4 = c2 * c2
        s2 = s * s
        plg = self.plg
        plg[1, 2] = c
        plg[1, 3] = 0.5 * (3.0 * c2 - 1.0)
        plg[1, 4] = 0.5 * (5.0 * c * c2 - 3.0 * c)
        plg[1, 5] = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0
        plg[1, 6] = (63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0
        plg[1, 7] = (11.0 * c * plg[1, 6] - 5.0 * plg[1, 5]) / 6.0
        plg[2, 2] = s
        plg[2, 3] = 3.0 * c * s
        plg[2, 4] = 1.5 * (5.0 * c2 - 1.0) * s
        plg[2, 5] = 2.5 * (7.0 * c2 * c - 3.0 * c) * s
        plg[2, 6] = 1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s
        plg[2, 7] = (11.0 * c * plg[2, 6] - 6.0 * plg[2, 5]) / 5.0
        plg[3, 3] = 3.0 * s2
        plg[3, 4] = 15.0 * s2 * c
        plg[3, 5] = 7.5 * (7.0 * c2 - 1.0) * s2
        plg[3, 6] = 3.0 * c * plg[3, 5] - 2.0 * plg[3, 4]
        plg[3, 7] = (11.0 * c * plg[3, 6] - 7.0 * plg[3, 5]) / 4.0
        plg[3, 8] = (13.0 * c * plg[3, 7] - 8.0 * plg[3, 6]) / 5.0
        plg[4, 4] = 15.0 * s2 * s
        plg[4, 5] = 105.0 * s2 * s * c
        plg[4, 6] = (9.0 * c * plg[4, 5] - 7.0 * plg[4, 4]) / 2.0
        plg[4, 7] = (11.0 * c * plg[4, 6] - 8.0 * plg[4, 5]) / 3.0
        if not (((flags.sw[8] == 0.0) and (flags.sw[9] == 0.0)) and (flags.sw[15] == 0.0)):
            self.stloc = math.sin(hr * input_.lst)
            self.ctloc = math.cos(hr * input_.lst)
            self.s2tloc = math.sin(2.0 * hr * input_.lst)
            self.c2tloc = math.cos(2.0 * hr * input_.lst)
            self.s3tloc = math.sin(3.0 * hr * input_.lst)
            self.c3tloc = math.cos(3.0 * hr * input_.lst)

    def _globe7(self, p_in: np.ndarray, input_: _Input, flags: _Flags) -> float:
        p = np.array(p_in, dtype=float, copy=True)
        t = np.zeros(15, dtype=float)
        sr = 7.2722e-5
        dgtr = 1.74533e-2
        dr = 1.72142e-2
        hr = 0.2618
        self._legendre_and_local_time(input_, flags)
        plg = self.plg
        cd32 = math.cos(dr * (input_.doy - p[32]))
        cd18 = math.cos(2.0 * dr * (input_.doy - p[18]))
        cd14 = math.cos(dr * (input_.doy - p[14]))
        cd39 = math.cos(2.0 * dr * (input_.doy - p[39]))
        df = input_.f107 - input_.f107a
        self.dfa = input_.f107a - 150.0
        t[1] = p[20] * df * (1.0 + p[60] * self.dfa) + p[21] * df * df + p[22] * self.dfa + p[30] * self.dfa**2.0
        f1 = 1.0 + (p[48] * self.dfa + p[20] * df + p[21] * df * df) * flags.swc[2]
        f2 = 1.0 + (p[50] * self.dfa + p[20] * df + p[21] * df * df) * flags.swc[2]
        t[2] = (
            p[2] * plg[1, 3]
            + p[3] * plg[1, 5]
            + p[23] * plg[1, 7]
            + p[15] * plg[1, 3] * self.dfa * flags.swc[2]
            + p[27] * plg[1, 2]
        )
        t[3] = p[19] * cd32
        t[4] = (p[16] + p[17] * plg[1, 3]) * cd18
        t[5] = f1 * (p[10] * plg[1, 2] + p[11] * plg[1, 4]) * cd14
        t[6] = p[38] * plg[1, 2] * cd39
        if flags.sw[8]:
            t71 = p[12] * plg[2, 3] * cd14 * flags.swc[6]
            t72 = p[13] * plg[2, 3] * cd14 * flags.swc[6]
            t[7] = f2 * (
                (p[4] * plg[2, 2] + p[5] * plg[2, 4] + p[28] * plg[2, 6] + t71) * self.ctloc
                + (p[7] * plg[2, 2] + p[8] * plg[2, 4] + p[29] * plg[2, 6] + t72) * self.stloc
            )
        if flags.sw[9]:
            t81 = (p[24] * plg[3, 4] + p[36] * plg[3, 6]) * cd14 * flags.swc[6]
            t82 = (p[34] * plg[3, 4] + p[37] * plg[3, 6]) * cd14 * flags.swc[6]
            t[8] = f2 * (
                (p[6] * plg[3, 3] + p[42] * plg[3, 5] + t81) * self.c2tloc
                + (p[9] * plg[3, 3] + p[43] * plg[3, 5] + t82) * self.s2tloc
            )
        if flags.sw[15]:
            t[14] = f2 * (
                (p[40] * plg[4, 4] + (p[94] * plg[4, 5] + p[47] * plg[4, 7]) * cd14 * flags.swc[6]) * self.s3tloc
                + (p[41] * plg[4, 4] + (p[95] * plg[4, 5] + p[49] * plg[4, 7]) * cd14 * flags.swc[6]) * self.c3tloc
            )
        if flags.sw[10] == -1:
            if p[52] != 0.0:
                exp1 = math.exp(-10800.0 * abs(p[52]) / (1.0 + p[139] * (45.0 - abs(input_.g_lat))))
                exp1 = min(exp1, 0.99999)
                if p[25] < 1.0e-4:
                    p[25] = 1.0e-4
                self.apt[1] = self._sg0(exp1, p, input_.ap_a)
                if flags.sw[10]:
                    t[9] = self.apt[1] * (
                        p[51]
                        + p[97] * plg[1, 3]
                        + p[55] * plg[1, 5]
                        + (p[126] * plg[1, 2] + p[127] * plg[1, 4] + p[128] * plg[1, 6]) * cd14 * flags.swc[6]
                        + (p[129] * plg[2, 2] + p[130] * plg[2, 4] + p[131] * plg[2, 6])
                        * flags.swc[8]
                        * math.cos(hr * (input_.lst - p[132]))
                    )
        else:
            apd = input_.ap - 4.0
            p44 = max(p[44], 1.0e-5)
            self.apdf = apd + (p[45] - 1.0) * (apd + (math.exp(-p44 * apd) - 1.0) / p44)
            if flags.sw[10]:
                t[9] = self.apdf * (
                    p[33]
                    + p[46] * plg[1, 3]
                    + p[35] * plg[1, 5]
                    + (p[101] * plg[1, 2] + p[102] * plg[1, 4] + p[103] * plg[1, 6]) * cd14 * flags.swc[6]
                    + (p[122] * plg[2, 2] + p[123] * plg[2, 4] + p[124] * plg[2, 6])
                    * flags.swc[8]
                    * math.cos(hr * (input_.lst - p[125]))
                )
        if flags.sw[11] and input_.g_long > -1000.0:
            if flags.sw[12]:
                t[11] = (1.0 + p[81] * self.dfa * flags.swc[2]) * (
                    (
                        p[65] * plg[2, 3]
                        + p[66] * plg[2, 5]
                        + p[67] * plg[2, 7]
                        + p[104] * plg[2, 2]
                        + p[105] * plg[2, 4]
                        + p[106] * plg[2, 6]
                        + flags.swc[6] * (p[110] * plg[2, 2] + p[111] * plg[2, 4] + p[112] * plg[2, 6]) * cd14
                    )
                    * math.cos(dgtr * input_.g_long)
                    + (
                        p[91] * plg[2, 3]
                        + p[92] * plg[2, 5]
                        + p[93] * plg[2, 7]
                        + p[107] * plg[2, 2]
                        + p[108] * plg[2, 4]
                        + p[109] * plg[2, 6]
                        + flags.swc[6] * (p[113] * plg[2, 2] + p[114] * plg[2, 4] + p[115] * plg[2, 6]) * cd14
                    )
                    * math.sin(dgtr * input_.g_long)
                )
            if flags.sw[13]:
                t[12] = (
                    (1.0 + p[96] * plg[1, 2])
                    * (1.0 + p[82] * self.dfa * flags.swc[2])
                    * (1.0 + p[120] * plg[1, 2] * flags.swc[6] * cd14)
                    * (p[69] * plg[1, 2] + p[70] * plg[1, 4] + p[71] * plg[1, 6])
                    * math.cos(sr * (input_.sec - p[72]))
                )
                t[12] += (
                    flags.swc[12]
                    * (p[77] * plg[3, 4] + p[78] * plg[3, 6] + p[79] * plg[3, 8])
                    * math.cos(sr * (input_.sec - p[80]) + 2.0 * dgtr * input_.g_long)
                    * (1.0 + p[138] * self.dfa * flags.swc[2])
                )
            if flags.sw[14]:
                if flags.sw[10] == -1 and p[52]:
                    t[13] = (
                        self.apt[1]
                        * flags.swc[12]
                        * (1.0 + p[133] * plg[1, 2])
                        * (p[53] * plg[2, 3] + p[99] * plg[2, 5] + p[68] * plg[2, 7])
                        * math.cos(dgtr * (input_.g_long - p[98]))
                        + self.apt[1]
                        * flags.swc[12]
                        * flags.swc[6]
                        * (p[134] * plg[2, 2] + p[135] * plg[2, 4] + p[136] * plg[2, 6])
                        * cd14
                        * math.cos(dgtr * (input_.g_long - p[137]))
                        + self.apt[1]
                        * flags.swc[13]
                        * (p[56] * plg[1, 2] + p[57] * plg[1, 4] + p[58] * plg[1, 6])
                        * math.cos(sr * (input_.sec - p[59]))
                    )
                elif flags.sw[10] != -1:
                    t[13] = (
                        self.apdf
                        * flags.swc[12]
                        * (1.0 + p[121] * plg[1, 2])
                        * (p[61] * plg[2, 3] + p[62] * plg[2, 5] + p[63] * plg[2, 7])
                        * math.cos(dgtr * (input_.g_long - p[64]))
                        + self.apdf
                        * flags.swc[12]
                        * flags.swc[6]
                        * (p[116] * plg[2, 2] + p[117] * plg[2, 4] + p[118] * plg[2, 6])
                        * cd14
                        * math.cos(dgtr * (input_.g_long - p[119]))
                        + self.apdf
                        * flags.swc[13]
                        * (p[84] * plg[1, 2] + p[85] * plg[1, 4] + p[86] * plg[1, 6])
                        * math.cos(sr * (input_.sec - p[76]))
                    )
        tinf = p[31]
        for i in range(1, 15):
            tinf += abs(flags.sw[i + 1]) * t[i]
        return float(tinf)

    def _glob7s(self, p_in: np.ndarray, input_: _Input, flags: _Flags) -> float:
        p = np.array(p_in, dtype=float, copy=True)
        if p[100] == 0.0:
            p[100] = 2.0
        t = np.zeros(15, dtype=float)
        dr = 1.72142e-2
        dgtr = 1.74533e-2
        plg = self.plg
        cd32 = math.cos(dr * (input_.doy - p[32]))
        cd18 = math.cos(2.0 * dr * (input_.doy - p[18]))
        cd14 = math.cos(dr * (input_.doy - p[14]))
        cd39 = math.cos(2.0 * dr * (input_.doy - p[39]))
        t[1] = p[22] * self.dfa
        t[2] = (
            p[2] * plg[1, 3]
            + p[3] * plg[1, 5]
            + p[23] * plg[1, 7]
            + p[27] * plg[1, 2]
            + p[15] * plg[1, 4]
            + p[60] * plg[1, 6]
        )
        t[3] = (p[19] + p[48] * plg[1, 3] + p[30] * plg[1, 5]) * cd32
        t[4] = (p[16] + p[17] * plg[1, 3] + p[31] * plg[1, 5]) * cd18
        t[5] = (p[10] * plg[1, 2] + p[11] * plg[1, 4] + p[21] * plg[1, 6]) * cd14
        t[6] = p[38] * plg[1, 2] * cd39
        if flags.sw[8]:
            t71 = p[12] * plg[2, 3] * cd14 * flags.swc[6]
            t72 = p[13] * plg[2, 3] * cd14 * flags.swc[6]
            t[7] = (p[4] * plg[2, 2] + p[5] * plg[2, 4] + t71) * self.ctloc + (
                p[7] * plg[2, 2] + p[8] * plg[2, 4] + t72
            ) * self.stloc
        if flags.sw[9]:
            t81 = (p[24] * plg[3, 4] + p[36] * plg[3, 6]) * cd14 * flags.swc[6]
            t82 = (p[34] * plg[3, 4] + p[37] * plg[3, 6]) * cd14 * flags.swc[6]
            t[8] = (p[6] * plg[3, 3] + p[42] * plg[3, 5] + t81) * self.c2tloc + (
                p[9] * plg[3, 3] + p[43] * plg[3, 5] + t82
            ) * self.s2tloc
        if flags.sw[15]:
            t[14] = p[40] * plg[4, 4] * self.s3tloc + p[41] * plg[4, 4] * self.c3tloc
        if flags.sw[10]:
            if flags.sw[10] == 1:
                t[9] = self.apdf * (p[33] + p[46] * plg[1, 3] * flags.swc[3])
            if flags.sw[10] == -1:
                t[9] = p[51] * self.apt[1] + p[97] * plg[1, 3] * self.apt[1] * flags.swc[3]
        if not ((flags.sw[11] == 0.0) or (flags.sw[12] == 0.0) or (input_.g_long <= -1000.0)):
            t[11] = (
                1.0
                + plg[1, 2]
                * (
                    p[81] * flags.swc[6] * math.cos(dr * (input_.doy - p[82]))
                    + p[86] * flags.swc[7] * math.cos(2.0 * dr * (input_.doy - p[87]))
                )
                + p[84] * flags.swc[4] * math.cos(dr * (input_.doy - p[85]))
                + p[88] * flags.swc[5] * math.cos(2.0 * dr * (input_.doy - p[89]))
            ) * (
                (
                    p[65] * plg[2, 3]
                    + p[66] * plg[2, 5]
                    + p[67] * plg[2, 7]
                    + p[75] * plg[2, 2]
                    + p[76] * plg[2, 4]
                    + p[77] * plg[2, 6]
                )
                * math.cos(dgtr * input_.g_long)
                + (
                    p[91] * plg[2, 3]
                    + p[92] * plg[2, 5]
                    + p[93] * plg[2, 7]
                    + p[78] * plg[2, 2]
                    + p[79] * plg[2, 4]
                    + p[80] * plg[2, 6]
                )
                * math.sin(dgtr * input_.g_long)
            )
        return float(sum(abs(flags.sw[i + 1]) * t[i] for i in range(1, 15)))

    def _gts7(self, input_: _Input, flags: _Flags) -> _Output:
        tz = 0.0
        output = _Output()
        zn1 = np.array([0.0, 120.0, 110.0, 100.0, 90.0, 72.5], dtype=float)
        mn1 = 5
        dgtr = 1.74533e-2
        dr = 1.72142e-2
        alpha = np.array([0.0, -0.38, 0.0, 0.0, 0.0, 0.17, 0.0, -0.38, 0.0, 0.0], dtype=float)
        altl = np.array([0.0, 200.0, 300.0, 160.0, 250.0, 240.0, 450.0, 320.0, 450.0], dtype=float)
        zn1[1] = PDL1[2, 16]

        if input_.alt > zn1[1]:
            tinf = PTM1[1] * PT1[1] * (1.0 + flags.sw[17] * self._globe7(PT1, input_, flags))
        else:
            tinf = PTM1[1] * PT1[1]
        output.t[1] = tinf

        if input_.alt > zn1[5]:
            g0 = PTM1[4] * PS1[1] * (1.0 + flags.sw[20] * self._globe7(PS1, input_, flags))
        else:
            g0 = PTM1[4] * PS1[1]
        tlb = PTM1[2] * (1.0 + flags.sw[18] * self._globe7(PD1[4, :], input_, flags)) * PD1[4, 1]
        s = g0 / (tinf - tlb)

        if input_.alt < 300.0:
            self.meso_tn1[2] = PTM1[7] * PTL1[1, 1] / (1.0 - flags.sw[19] * self._glob7s(PTL1[1, :], input_, flags))
            self.meso_tn1[3] = PTM1[3] * PTL1[2, 1] / (1.0 - flags.sw[19] * self._glob7s(PTL1[2, :], input_, flags))
            self.meso_tn1[4] = PTM1[8] * PTL1[3, 1] / (1.0 - flags.sw[19] * self._glob7s(PTL1[3, :], input_, flags))
            self.meso_tn1[5] = (
                PTM1[5] * PTL1[4, 1] / (1.0 - flags.sw[19] * flags.sw[21] * self._glob7s(PTL1[4, :], input_, flags))
            )
            self.meso_tgn1[2] = (
                PTM1[9]
                * PMA1[9, 1]
                * (1.0 + flags.sw[19] * flags.sw[21] * self._glob7s(PMA1[9, :], input_, flags))
                * self.meso_tn1[5]
                * self.meso_tn1[5]
                / ((PTM1[5] * PTL1[4, 1]) ** 2.0)
            )
        else:
            self.meso_tn1[2] = PTM1[7] * PTL1[1, 1]
            self.meso_tn1[3] = PTM1[3] * PTL1[2, 1]
            self.meso_tn1[4] = PTM1[8] * PTL1[3, 1]
            self.meso_tn1[5] = PTM1[5] * PTL1[4, 1]
            self.meso_tgn1[2] = (
                PTM1[9] * PMA1[9, 1] * self.meso_tn1[5] * self.meso_tn1[5] / ((PTM1[5] * PTL1[4, 1]) ** 2.0)
            )

        g28 = flags.sw[22] * self._globe7(PD1[3, :], input_, flags)
        zhf = PDL1[2, 25] * (
            1.0 + flags.sw[6] * PDL1[1, 25] * math.sin(dgtr * input_.g_lat) * math.cos(dr * (input_.doy - PT1[14]))
        )
        xmm = PDM1[3, 5]
        z = input_.alt

        db28 = PDM1[3, 1] * math.exp(g28) * PD1[3, 1]
        output.d[3], output.t[2] = self._densu(
            z, db28, tinf, tlb, 28.0, alpha[3], output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        zh28 = PDM1[3, 3] * zhf
        zhm28 = PDM1[3, 4] * PDL1[2, 6]
        b28, tz = self._densu(
            zh28, db28, tinf, tlb, 28.0 - xmm, alpha[3] - 1.0, tz, PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        if flags.sw[16] and z <= altl[3]:
            self.dm28, tz = self._densu(
                z, b28, tinf, tlb, xmm, alpha[3], tz, PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
            )
            output.d[3] = self._dnet(output.d[3], self.dm28, zhm28, xmm, 28.0)

        g4 = flags.sw[22] * self._globe7(PD1[1, :], input_, flags)
        db04 = PDM1[1, 1] * math.exp(g4) * PD1[1, 1]
        output.d[1], output.t[2] = self._densu(
            z, db04, tinf, tlb, 4.0, alpha[1], output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        if flags.sw[16] and z < altl[1]:
            b04, output.t[2] = self._densu(
                PDM1[1, 3],
                db04,
                tinf,
                tlb,
                4.0 - xmm,
                alpha[1] - 1.0,
                output.t[2],
                PTM1[6],
                s,
                mn1,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm04, output.t[2] = self._densu(
                z, b04, tinf, tlb, xmm, 0.0, output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
            )
            output.d[1] = self._dnet(output.d[1], self.dm04, zhm28, xmm, 4.0)
            output.d[1] *= self._ccor(
                z, math.log(b28 * PDM1[1, 2] / b04), PDM1[1, 6] * PDL1[2, 2], PDM1[1, 5] * PDL1[2, 1]
            )

        g16 = flags.sw[22] * self._globe7(PD1[2, :], input_, flags)
        db16 = PDM1[2, 1] * math.exp(g16) * PD1[2, 1]
        output.d[2], output.t[2] = self._densu(
            z, db16, tinf, tlb, 16.0, alpha[2], output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        if flags.sw[16] and z <= altl[2]:
            b16, output.t[2] = self._densu(
                PDM1[2, 3],
                db16,
                tinf,
                tlb,
                16.0 - xmm,
                alpha[2] - 1.0,
                output.t[2],
                PTM1[6],
                s,
                mn1,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm16, output.t[2] = self._densu(
                z, b16, tinf, tlb, xmm, 0.0, output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
            )
            output.d[2] = self._dnet(output.d[2], self.dm16, zhm28, xmm, 16.0)
            rl = PDM1[2, 2] * PDL1[2, 17] * (1.0 + flags.sw[2] * PDL1[1, 24] * (input_.f107a - 150.0))
            output.d[2] *= self._ccor2(z, rl, PDM1[2, 6] * PDL1[2, 4], PDM1[2, 5] * PDL1[2, 3], PDM1[2, 6] * PDL1[2, 5])
            output.d[2] *= self._ccor(z, PDM1[2, 4] * PDL1[2, 15], PDM1[2, 8] * PDL1[2, 14], PDM1[2, 7] * PDL1[2, 13])

        g32 = flags.sw[22] * self._globe7(PD1[5, :], input_, flags)
        db32 = PDM1[4, 1] * math.exp(g32) * PD1[5, 1]
        output.d[4], output.t[2] = self._densu(
            z, db32, tinf, tlb, 32.0, alpha[4], output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        if flags.sw[16]:
            if z <= altl[4]:
                b32, output.t[2] = self._densu(
                    PDM1[4, 3],
                    db32,
                    tinf,
                    tlb,
                    32.0 - xmm,
                    alpha[4] - 1.0,
                    output.t[2],
                    PTM1[6],
                    s,
                    mn1,
                    zn1,
                    self.meso_tn1,
                    self.meso_tgn1,
                )
                self.dm32, output.t[2] = self._densu(
                    z, b32, tinf, tlb, xmm, 0.0, output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
                )
                output.d[4] = self._dnet(output.d[4], self.dm32, zhm28, xmm, 32.0)
                output.d[4] *= self._ccor(
                    z, math.log(b28 * PDM1[4, 2] / b32), PDM1[4, 6] * PDL1[2, 8], PDM1[4, 5] * PDL1[2, 7]
                )
            rc32 = PDM1[4, 4] * PDL1[2, 24] * (1.0 + flags.sw[2] * PDL1[1, 24] * (input_.f107a - 150.0))
            output.d[4] *= self._ccor2(
                z, rc32, PDM1[4, 8] * PDL1[2, 23], PDM1[4, 7] * PDL1[2, 22], PDM1[4, 8] * PDL1[1, 23]
            )

        g40 = flags.sw[22] * self._globe7(PD1[6, :], input_, flags)
        db40 = PDM1[5, 1] * math.exp(g40) * PD1[6, 1]
        output.d[5], output.t[2] = self._densu(
            z, db40, tinf, tlb, 40.0, alpha[5], output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        if flags.sw[16] and z <= altl[5]:
            b40, output.t[2] = self._densu(
                PDM1[5, 3],
                db40,
                tinf,
                tlb,
                40.0 - xmm,
                alpha[5] - 1.0,
                output.t[2],
                PTM1[6],
                s,
                mn1,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm40, output.t[2] = self._densu(
                z, b40, tinf, tlb, xmm, 0.0, output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
            )
            output.d[5] = self._dnet(output.d[5], self.dm40, zhm28, xmm, 40.0)
            output.d[5] *= self._ccor(
                z, math.log(b28 * PDM1[5, 2] / b40), PDM1[5, 6] * PDL1[2, 10], PDM1[5, 5] * PDL1[2, 9]
            )

        g1 = flags.sw[22] * self._globe7(PD1[7, :], input_, flags)
        db01 = PDM1[6, 1] * math.exp(g1) * PD1[7, 1]
        output.d[7], output.t[2] = self._densu(
            z, db01, tinf, tlb, 1.0, alpha[7], output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        if flags.sw[16] and z <= altl[7]:
            b01, output.t[2] = self._densu(
                PDM1[6, 3],
                db01,
                tinf,
                tlb,
                1.0 - xmm,
                alpha[7] - 1.0,
                output.t[2],
                PTM1[6],
                s,
                mn1,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm01, output.t[2] = self._densu(
                z, b01, tinf, tlb, xmm, 0.0, output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
            )
            output.d[7] = self._dnet(output.d[7], self.dm01, zhm28, xmm, 1.0)
            output.d[7] *= self._ccor(
                z,
                math.log(b28 * PDM1[6, 2] * abs(PDL1[2, 18]) / b01),
                PDM1[6, 6] * PDL1[2, 12],
                PDM1[6, 5] * PDL1[2, 11],
            )
            output.d[7] *= self._ccor(z, PDM1[6, 4] * PDL1[2, 21], PDM1[6, 8] * PDL1[2, 20], PDM1[6, 7] * PDL1[2, 19])

        g14 = flags.sw[22] * self._globe7(PD1[8, :], input_, flags)
        db14 = PDM1[7, 1] * math.exp(g14) * PD1[8, 1]
        output.d[8], output.t[2] = self._densu(
            z, db14, tinf, tlb, 14.0, alpha[8], output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        if flags.sw[16] and z <= altl[8]:
            b14, output.t[2] = self._densu(
                PDM1[7, 3],
                db14,
                tinf,
                tlb,
                14.0 - xmm,
                alpha[8] - 1.0,
                output.t[2],
                PTM1[6],
                s,
                mn1,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm14, output.t[2] = self._densu(
                z, b14, tinf, tlb, xmm, 0.0, output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
            )
            output.d[8] = self._dnet(output.d[8], self.dm14, zhm28, xmm, 14.0)
            output.d[8] *= self._ccor(
                z, math.log(b28 * PDM1[7, 2] * abs(PDL1[1, 3]) / b14), PDM1[7, 6] * PDL1[1, 2], PDM1[7, 5] * PDL1[1, 1]
            )
            output.d[8] *= self._ccor(z, PDM1[7, 4] * PDL1[1, 6], PDM1[7, 8] * PDL1[1, 5], PDM1[7, 7] * PDL1[1, 4])

        g16h = flags.sw[22] * self._globe7(PD1[9, :], input_, flags)
        db16h = PDM1[8, 1] * math.exp(g16h) * PD1[9, 1]
        tho = PDM1[8, 10] * PDL1[1, 7]
        dd, output.t[2] = self._densu(
            z, db16h, tho, tho, 16.0, alpha[9], output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        zsho = self._scalh(PDM1[8, 5], 16.0, tho)
        output.d[9] = dd * math.exp(-PDM1[8, 6] / zsho * (math.exp(-(z - PDM1[8, 5]) / PDM1[8, 6]) - 1.0))

        output.d[6] = 1.66e-24 * (
            4.0 * output.d[1]
            + 16.0 * output.d[2]
            + 28.0 * output.d[3]
            + 32.0 * output.d[4]
            + 40.0 * output.d[5]
            + output.d[7]
            + 14.0 * output.d[8]
        )
        _, output.t[2] = self._densu(
            abs(input_.alt), 1.0, tinf, tlb, 0.0, 0.0, output.t[2], PTM1[6], s, mn1, zn1, self.meso_tn1, self.meso_tgn1
        )
        if flags.sw[1]:
            for i in range(1, 10):
                output.d[i] *= 1.0e6
            output.d[6] /= 1000.0
        return output

    def _gtd7(self, input_: _Input, flags: _Flags) -> _Output:
        tz = 0.0
        output = _Output()
        mn3 = 5
        zn3 = np.array([0.0, 32.5, 20.0, 15.0, 10.0, 0.0], dtype=float)
        mn2 = 4
        zn2 = np.array([0.0, 72.5, 55.0, 45.0, 32.5], dtype=float)
        zmix = 62.5
        flags = self._tselec(flags)
        xlat = input_.g_lat if flags.sw[3] != 0.0 else 45.0
        self.gsurf, self.re = self._glatf(xlat)
        xmm = PDM1[3, 5]
        altt = input_.alt if input_.alt > zn2[1] else zn2[1]
        tmp = input_.alt
        input_.alt = altt
        soutput = self._gts7(input_, flags)
        input_.alt = tmp
        dm28m = self.dm28 * 1.0e6 if flags.sw[1] else self.dm28
        output.t[1] = soutput.t[1]
        output.t[2] = soutput.t[2]
        if input_.alt >= zn2[1]:
            output.d[:] = soutput.d
            return output

        self.meso_tgn2[1] = self.meso_tgn1[2]
        self.meso_tn2[1] = self.meso_tn1[5]
        self.meso_tn2[2] = PMA1[1, 1] * PAVGM1[1] / (1.0 - flags.sw[21] * self._glob7s(PMA1[1, :], input_, flags))
        self.meso_tn2[3] = PMA1[2, 1] * PAVGM1[2] / (1.0 - flags.sw[21] * self._glob7s(PMA1[2, :], input_, flags))
        self.meso_tn2[4] = (
            PMA1[3, 1] * PAVGM1[3] / (1.0 - flags.sw[21] * flags.sw[23] * self._glob7s(PMA1[3, :], input_, flags))
        )
        self.meso_tgn2[2] = (
            PAVGM1[9]
            * PMA1[10, 1]
            * (1.0 + flags.sw[21] * flags.sw[23] * self._glob7s(PMA1[10, :], input_, flags))
            * self.meso_tn2[4]
            * self.meso_tn2[4]
            / ((PMA1[3, 1] * PAVGM1[3]) ** 2.0)
        )
        self.meso_tn3[1] = self.meso_tn2[4]
        if input_.alt <= zn3[1]:
            self.meso_tgn3[1] = self.meso_tgn2[2]
            self.meso_tn3[2] = PMA1[4, 1] * PAVGM1[4] / (1.0 - flags.sw[23] * self._glob7s(PMA1[4, :], input_, flags))
            self.meso_tn3[3] = PMA1[5, 1] * PAVGM1[5] / (1.0 - flags.sw[23] * self._glob7s(PMA1[5, :], input_, flags))
            self.meso_tn3[4] = PMA1[6, 1] * PAVGM1[6] / (1.0 - flags.sw[23] * self._glob7s(PMA1[6, :], input_, flags))
            self.meso_tn3[5] = PMA1[7, 1] * PAVGM1[7] / (1.0 - flags.sw[23] * self._glob7s(PMA1[7, :], input_, flags))
            self.meso_tgn3[2] = (
                PMA1[8, 1]
                * PAVGM1[8]
                * (1.0 + flags.sw[23] * self._glob7s(PMA1[8, :], input_, flags))
                * self.meso_tn3[5]
                * self.meso_tn3[5]
                / ((PMA1[7, 1] * PAVGM1[7]) ** 2.0)
            )
        dmc = 1.0 - (zn2[1] - input_.alt) / (zn2[1] - zmix) if input_.alt > zmix else 0.0
        dz28 = soutput.d[3]
        dmr = soutput.d[3] / dm28m - 1.0 if dm28m != 0.0 else 0.0
        output.d[3], tz = self._densm(
            input_.alt, dm28m, xmm, tz, mn3, zn3, self.meso_tn3, self.meso_tgn3, mn2, zn2, self.meso_tn2, self.meso_tgn2
        )
        output.d[3] *= 1.0 + dmr * dmc
        dmr = soutput.d[1] / (dz28 * PDM1[1, 2]) - 1.0 if dz28 != 0.0 else 0.0
        output.d[1] = output.d[3] * PDM1[1, 2] * (1.0 + dmr * dmc)
        output.d[2] = 0.0
        output.d[9] = 0.0
        dmr = soutput.d[4] / (dz28 * PDM1[4, 2]) - 1.0 if dz28 != 0.0 else 0.0
        output.d[4] = output.d[3] * PDM1[4, 2] * (1.0 + dmr * dmc)
        dmr = soutput.d[5] / (dz28 * PDM1[5, 2]) - 1.0 if dz28 != 0.0 else 0.0
        output.d[5] = output.d[3] * PDM1[5, 2] * (1.0 + dmr * dmc)
        output.d[7] = 0.0
        output.d[8] = 0.0
        output.d[6] = 1.66e-24 * (
            4.0 * output.d[1]
            + 16.0 * output.d[2]
            + 28.0 * output.d[3]
            + 32.0 * output.d[4]
            + 40.0 * output.d[5]
            + output.d[7]
            + 14.0 * output.d[8]
        )
        if flags.sw[1]:
            output.d[6] /= 1000.0
        _, tz = self._densm(
            input_.alt, 1.0, 0.0, tz, mn3, zn3, self.meso_tn3, self.meso_tgn3, mn2, zn2, self.meso_tn2, self.meso_tgn2
        )
        output.t[2] = tz
        return output

    def _gtd7d(self, input_: _Input, flags: _Flags) -> _Output:
        output = self._gtd7(input_, flags)
        output.d[6] = 1.66e-24 * (
            4.0 * output.d[1]
            + 16.0 * output.d[2]
            + 28.0 * output.d[3]
            + 32.0 * output.d[4]
            + 40.0 * output.d[5]
            + output.d[7]
            + 14.0 * output.d[8]
            + 16.0 * output.d[9]
        )
        if flags.sw[1]:
            output.d[6] /= 1000.0
        return output

    def density(self, input_: _Input) -> float:
        flags = _Flags()
        flags.switches[1] = 0.0
        flags.switches[2:25] = 1.0
        flags.switches[10] = -1.0
        output = self._gtd7d(input_, flags)
        return float(max(0.0, 1000.0 * output.d[6]))


def nrlmsise00_density(
    alt_km: float, lat_deg: float, lon_deg: float, dt_utc: datetime, env: dict | None = None
) -> float:
    env = {} if env is None else dict(env)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)
    f107a, f107, ap, ap_a = _solar_geomagnetic_inputs(dt_utc, env)
    jd_utc = datetime_to_julian_date(dt_utc)
    lon_rad = math.radians(float(lon_deg))
    lst_hr = env.get("nrlmsise00_lst_hr")
    if lst_hr is None:
        lst_hr = ((lon_rad + gmst_angle_rad_from_jd(jd_utc)) % (2.0 * math.pi)) * 24.0 / (2.0 * math.pi)
    sec = dt_utc.hour * 3600.0 + dt_utc.minute * 60.0 + dt_utc.second + dt_utc.microsecond * 1e-6
    input_ = _Input(
        doy=int(math.floor(_day_of_year(dt_utc))),
        sec=sec,
        alt=float(max(0.0, alt_km)),
        g_lat=float(lat_deg),
        g_long=float(lon_deg),
        lst=float(lst_hr),
        f107a=f107a,
        f107=f107,
        ap=ap,
        ap_a=ap_a,
    )
    return _NRLMSISE00().density(input_)
