"""Optional compiled scalar kernels for the NRLMSISE-00 atmosphere model."""

from __future__ import annotations

import math

import numpy as np

from sim.acceleration.optional import njit_or_identity
from sim.dynamics.orbit.nrlmsise00_backend import _globe7_quiet_python

globe7_quiet_kernel = njit_or_identity(cache=True, fastmath=False)(_globe7_quiet_python)


@njit_or_identity(cache=True, fastmath=False)
def _zeta_kernel(zz: float, zl: float, re_km: float) -> float:
    return (zz - zl) * (re_km + zl) / (re_km + zz)


@njit_or_identity(cache=True, fastmath=False)
def _spline_kernel(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    yp1: float,
    ypn: float,
) -> np.ndarray:
    u = np.zeros(max(6, n + 1), dtype=np.float64)
    y2 = np.zeros(max(6, n + 1), dtype=np.float64)
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


@njit_or_identity(cache=True, fastmath=False)
def _splint_kernel(
    xa: np.ndarray,
    ya: np.ndarray,
    y2a: np.ndarray,
    n: int,
    x: float,
) -> float:
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


@njit_or_identity(cache=True, fastmath=False)
def _splini_kernel(
    xa: np.ndarray,
    ya: np.ndarray,
    y2a: np.ndarray,
    n: int,
    x: float,
) -> float:
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
            + (
                (-(1.0 + a2 * a2) / 4.0 + a2 / 2.0) * y2a[klo + 1]
                + (b2 * b2 / 4.0 - b2 / 2.0) * y2a[khi + 1]
            )
            * h
            * h
            / 6.0
        ) * h
        klo += 1
        khi += 1
    return float(yi)


@njit_or_identity(cache=True, fastmath=False)
def densu_kernel(
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
    gsurf: float,
    re_km: float,
) -> tuple[float, float]:
    """Evaluate the complete NRLMSISE ``densu`` branch without fast math."""

    za = zn1[1]
    z = alt if alt > za else za
    zg2 = _zeta_kernel(z, zlb, re_km)
    tt = tinf - (tinf - tlb) * math.exp(-s2 * zg2)
    ta = tt
    tz = tt
    densu_temp = tz
    x = 0.0
    xs = np.empty(0, dtype=np.float64)
    ys = np.empty(0, dtype=np.float64)
    y2out = np.empty(0, dtype=np.float64)
    if alt < za:
        xs = np.zeros(6, dtype=np.float64)
        ys = np.zeros(6, dtype=np.float64)
        dta = (tinf - ta) * s2 * ((re_km + zlb) / (re_km + za)) ** 2.0
        tgn1[1] = dta
        tn1[1] = ta
        z = alt if alt > zn1[mn1] else zn1[mn1]
        z1, z2 = zn1[1], zn1[mn1]
        t1, t2 = tn1[1], tn1[mn1]
        zg = _zeta_kernel(z, z1, re_km)
        zgdif = _zeta_kernel(z2, z1, re_km)
        for k in range(1, mn1 + 1):
            xs[k] = _zeta_kernel(zn1[k], z1, re_km) / zgdif
            ys[k] = 1.0 / tn1[k]
        yd1 = -tgn1[1] / (t1 * t1) * zgdif
        yd2 = -tgn1[2] / (t2 * t2) * zgdif * ((re_km + z2) / (re_km + z1)) ** 2.0
        y2out = _spline_kernel(xs, ys, mn1, yd1, yd2)
        x = zg / zgdif
        tz = 1.0 / _splint_kernel(xs, ys, y2out, mn1, x)
        densu_temp = tz
    if xm == 0.0:
        return densu_temp, tz
    glb = gsurf / (1.0 + zlb / re_km) ** 2.0
    gamma = xm * glb / (s2 * 831.4 * tinf)
    expl = math.exp(-s2 * gamma * zg2)
    if expl > 50.0 or tt <= 0.0:
        expl = 50.0
    densa = dlb * (tlb / tt) ** (1.0 + alpha + gamma) * expl
    if alt >= za:
        return densa, tz
    glb = gsurf / (1.0 + zn1[1] / re_km) ** 2.0
    gamm = xm * glb * _zeta_kernel(zn1[mn1], zn1[1], re_km) / 831.4
    expl2 = min(50.0, gamm * _splini_kernel(xs, ys, y2out, mn1, x))
    if tz <= 0.0:
        expl2 = 50.0
    return densa * (tn1[1] / tz) ** (1.0 + alpha) * math.exp(-expl2), tz


@njit_or_identity(cache=True, fastmath=False)
def _ccor_kernel(alt: float, r: float, h1: float, zh: float) -> float:
    e = (alt - zh) / h1
    if e > 70.0:
        return 1.0
    if e < -70.0:
        return math.exp(r)
    return math.exp(r / (1.0 + math.exp(e)))


@njit_or_identity(cache=True, fastmath=False)
def _ccor2_kernel(alt: float, r: float, h1: float, zh: float, h2: float) -> float:
    e1 = (alt - zh) / h1
    e2 = (alt - zh) / h2
    if e1 > 70.0 or e2 > 70.0:
        return 1.0
    if e1 < -70.0 and e2 < -70.0:
        return math.exp(r)
    return math.exp(r / (1.0 + 0.5 * (math.exp(e1) + math.exp(e2))))


@njit_or_identity(cache=True, fastmath=False)
def _dnet_kernel(dd: float, dm: float, zhm: float, xmm: float, xm: float) -> float:
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


@njit_or_identity(cache=True, fastmath=False)
def quiet_thermosphere_density_kernel(
    doy: int,
    sec: float,
    alt: float,
    g_lat: float,
    g_long: float,
    lst: float,
    f107a: float,
    f107: float,
    pt1: np.ndarray,
    ps1: np.ndarray,
    pd1: np.ndarray,
    pdl1: np.ndarray,
    ptm1: np.ndarray,
    pdm1: np.ndarray,
    ptl1: np.ndarray,
    pma1: np.ndarray,
    zn1: np.ndarray,
    alpha: np.ndarray,
) -> float:
    """Return exact NRLMSISE-00 density for quiet-Ap thermosphere inputs.

    This is the standard-switch ``gtd7d`` branch for altitudes at or above
    300 km when all six historical Ap inputs used by the model equal 4.
    Lower and disturbed-atmosphere inputs remain on the authoritative Python
    implementation.
    """

    dgtr = 1.74533e-2
    hr = 0.2618
    c = math.sin(g_lat * dgtr)
    s_lat = math.cos(g_lat * dgtr)
    c2 = c * c
    c4 = c2 * c2
    s2 = s_lat * s_lat
    context = np.zeros(31, dtype=np.float64)
    context[0] = f107a - 150.0
    context[1] = f107 - f107a
    context[2] = c
    context[3] = 0.5 * (3.0 * c2 - 1.0)
    context[4] = 0.5 * (5.0 * c * c2 - 3.0 * c)
    context[5] = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0
    context[6] = (63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0
    context[7] = (11.0 * c * context[6] - 5.0 * context[5]) / 6.0
    context[8] = s_lat
    context[9] = 3.0 * c * s_lat
    context[10] = 1.5 * (5.0 * c2 - 1.0) * s_lat
    context[11] = 2.5 * (7.0 * c2 * c - 3.0 * c) * s_lat
    context[12] = 1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s_lat
    context[13] = (11.0 * c * context[12] - 6.0 * context[11]) / 5.0
    context[14] = 3.0 * s2
    context[15] = 15.0 * s2 * c
    context[16] = 7.5 * (7.0 * c2 - 1.0) * s2
    context[17] = 3.0 * c * context[16] - 2.0 * context[15]
    plg37 = (11.0 * c * context[17] - 7.0 * context[16]) / 4.0
    context[18] = (13.0 * c * plg37 - 8.0 * context[17]) / 5.0
    context[19] = 15.0 * s2 * s_lat
    context[20] = 105.0 * s2 * s_lat * c
    plg46 = (9.0 * c * context[20] - 7.0 * context[19]) / 2.0
    context[21] = (11.0 * c * plg46 - 8.0 * context[20]) / 3.0
    context[22] = math.cos(hr * lst)
    context[23] = math.sin(hr * lst)
    context[24] = math.cos(2.0 * hr * lst)
    context[25] = math.sin(2.0 * hr * lst)
    context[26] = math.sin(3.0 * hr * lst)
    context[27] = math.cos(3.0 * hr * lst)
    context[28] = math.cos(dgtr * g_long)
    context[29] = math.sin(dgtr * g_long)
    context[30] = context[0] ** 2.0

    c2_gravity = math.cos(2.0 * dgtr * g_lat)
    gsurf = 980.616 * (1.0 - 0.0026373 * c2_gravity)
    re_km = 2.0 * gsurf / (3.085462e-6 + 2.27e-9 * c2_gravity) * 1.0e-5
    meso_tn1 = np.zeros(6, dtype=np.float64)
    meso_tgn1 = np.zeros(3, dtype=np.float64)

    tinf = ptm1[1] * pt1[1] * (
        1.0 + globe7_quiet_kernel(pt1, doy, sec, g_long, context)
    )
    g0 = ptm1[4] * ps1[1] * (
        1.0 + globe7_quiet_kernel(ps1, doy, sec, g_long, context)
    )
    tlb = (
        ptm1[2]
        * (1.0 + globe7_quiet_kernel(pd1[4], doy, sec, g_long, context))
        * pd1[4, 1]
    )
    slope = g0 / (tinf - tlb)
    meso_tn1[2] = ptm1[7] * ptl1[1, 1]
    meso_tn1[3] = ptm1[3] * ptl1[2, 1]
    meso_tn1[4] = ptm1[8] * ptl1[3, 1]
    meso_tn1[5] = ptm1[5] * ptl1[4, 1]
    meso_tgn1[2] = (
        ptm1[9]
        * pma1[9, 1]
        * meso_tn1[5]
        * meso_tn1[5]
        / ((ptm1[5] * ptl1[4, 1]) ** 2.0)
    )

    dr = 1.72142e-2
    z = alt
    zhf = pdl1[2, 25] * (
        1.0 + pdl1[1, 25] * c * math.cos(dr * (doy - pt1[14]))
    )
    xmm = pdm1[3, 5]
    densities = np.zeros(10, dtype=np.float64)
    temperature = 0.0
    tz = 0.0

    g28 = globe7_quiet_kernel(pd1[3], doy, sec, g_long, context)
    db28 = pdm1[3, 1] * math.exp(g28) * pd1[3, 1]
    densities[3], temperature = densu_kernel(
        z, db28, tinf, tlb, 28.0, alpha[3], temperature,
        ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
    )
    zh28 = pdm1[3, 3] * zhf
    zhm28 = pdm1[3, 4] * pdl1[2, 6]
    b28, tz = densu_kernel(
        zh28, db28, tinf, tlb, 28.0 - xmm, alpha[3] - 1.0, tz,
        ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
    )

    g4 = globe7_quiet_kernel(pd1[1], doy, sec, g_long, context)
    db04 = pdm1[1, 1] * math.exp(g4) * pd1[1, 1]
    densities[1], temperature = densu_kernel(
        z, db04, tinf, tlb, 4.0, alpha[1], temperature,
        ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
    )

    g16 = globe7_quiet_kernel(pd1[2], doy, sec, g_long, context)
    db16 = pdm1[2, 1] * math.exp(g16) * pd1[2, 1]
    densities[2], temperature = densu_kernel(
        z, db16, tinf, tlb, 16.0, alpha[2], temperature,
        ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
    )
    if z <= 300.0:
        b16, temperature = densu_kernel(
            pdm1[2, 3], db16, tinf, tlb, 16.0 - xmm, alpha[2] - 1.0,
            temperature, ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
        )
        dm16, temperature = densu_kernel(
            z, b16, tinf, tlb, xmm, 0.0, temperature,
            ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
        )
        densities[2] = _dnet_kernel(densities[2], dm16, zhm28, xmm, 16.0)
        rl = pdm1[2, 2] * pdl1[2, 17] * (1.0 + pdl1[1, 24] * (f107a - 150.0))
        densities[2] *= _ccor2_kernel(
            z, rl, pdm1[2, 6] * pdl1[2, 4], pdm1[2, 5] * pdl1[2, 3],
            pdm1[2, 6] * pdl1[2, 5],
        )
        densities[2] *= _ccor_kernel(
            z, pdm1[2, 4] * pdl1[2, 15], pdm1[2, 8] * pdl1[2, 14],
            pdm1[2, 7] * pdl1[2, 13],
        )

    g32 = globe7_quiet_kernel(pd1[5], doy, sec, g_long, context)
    db32 = pdm1[4, 1] * math.exp(g32) * pd1[5, 1]
    densities[4], temperature = densu_kernel(
        z, db32, tinf, tlb, 32.0, alpha[4], temperature,
        ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
    )
    rc32 = pdm1[4, 4] * pdl1[2, 24] * (
        1.0 + pdl1[1, 24] * (f107a - 150.0)
    )
    densities[4] *= _ccor2_kernel(
        z, rc32, pdm1[4, 8] * pdl1[2, 23], pdm1[4, 7] * pdl1[2, 22],
        pdm1[4, 8] * pdl1[1, 23],
    )

    g40 = globe7_quiet_kernel(pd1[6], doy, sec, g_long, context)
    db40 = pdm1[5, 1] * math.exp(g40) * pd1[6, 1]
    densities[5], temperature = densu_kernel(
        z, db40, tinf, tlb, 40.0, alpha[5], temperature,
        ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
    )

    g1 = globe7_quiet_kernel(pd1[7], doy, sec, g_long, context)
    db01 = pdm1[6, 1] * math.exp(g1) * pd1[7, 1]
    densities[7], temperature = densu_kernel(
        z, db01, tinf, tlb, 1.0, alpha[7], temperature,
        ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
    )
    if z <= 320.0:
        b01, temperature = densu_kernel(
            pdm1[6, 3], db01, tinf, tlb, 1.0 - xmm, alpha[7] - 1.0,
            temperature, ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
        )
        dm01, temperature = densu_kernel(
            z, b01, tinf, tlb, xmm, 0.0, temperature,
            ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
        )
        densities[7] = _dnet_kernel(densities[7], dm01, zhm28, xmm, 1.0)
        densities[7] *= _ccor_kernel(
            z, math.log(b28 * pdm1[6, 2] * abs(pdl1[2, 18]) / b01),
            pdm1[6, 6] * pdl1[2, 12], pdm1[6, 5] * pdl1[2, 11],
        )
        densities[7] *= _ccor_kernel(
            z, pdm1[6, 4] * pdl1[2, 21], pdm1[6, 8] * pdl1[2, 20],
            pdm1[6, 7] * pdl1[2, 19],
        )

    g14 = globe7_quiet_kernel(pd1[8], doy, sec, g_long, context)
    db14 = pdm1[7, 1] * math.exp(g14) * pd1[8, 1]
    densities[8], temperature = densu_kernel(
        z, db14, tinf, tlb, 14.0, alpha[8], temperature,
        ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
    )
    if z <= 450.0:
        b14, temperature = densu_kernel(
            pdm1[7, 3], db14, tinf, tlb, 14.0 - xmm, alpha[8] - 1.0,
            temperature, ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
        )
        dm14, temperature = densu_kernel(
            z, b14, tinf, tlb, xmm, 0.0, temperature,
            ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
        )
        densities[8] = _dnet_kernel(densities[8], dm14, zhm28, xmm, 14.0)
        densities[8] *= _ccor_kernel(
            z, math.log(b28 * pdm1[7, 2] * abs(pdl1[1, 3]) / b14),
            pdm1[7, 6] * pdl1[1, 2], pdm1[7, 5] * pdl1[1, 1],
        )
        densities[8] *= _ccor_kernel(
            z, pdm1[7, 4] * pdl1[1, 6], pdm1[7, 8] * pdl1[1, 5],
            pdm1[7, 7] * pdl1[1, 4],
        )

    g16h = globe7_quiet_kernel(pd1[9], doy, sec, g_long, context)
    db16h = pdm1[8, 1] * math.exp(g16h) * pd1[9, 1]
    tho = pdm1[8, 10] * pdl1[1, 7]
    dd, temperature = densu_kernel(
        z, db16h, tho, tho, 16.0, alpha[9], temperature,
        ptm1[6], slope, 5, zn1, meso_tn1, meso_tgn1, gsurf, re_km,
    )
    gravity_at_reference = gsurf / ((1.0 + pdm1[8, 5] / re_km) ** 2.0)
    zsho = 831.4 * tho / (gravity_at_reference * 16.0)
    densities[9] = dd * math.exp(
        -pdm1[8, 6] / zsho
        * (math.exp(-(z - pdm1[8, 5]) / pdm1[8, 6]) - 1.0)
    )

    total = 1.66e-24 * (
        4.0 * densities[1]
        + 16.0 * densities[2]
        + 28.0 * densities[3]
        + 32.0 * densities[4]
        + 40.0 * densities[5]
        + densities[7]
        + 14.0 * densities[8]
        + 16.0 * densities[9]
    )
    return max(0.0, 1000.0 * total)
