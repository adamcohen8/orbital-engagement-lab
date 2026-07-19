"""Optional compiled scalar kernels for the MSIS-86 atmosphere model."""

from __future__ import annotations

import math

from sim.acceleration.optional import njit_or_identity


@njit_or_identity(cache=True, fastmath=False)
def _zeta_kernel(zz: float, zl: float, re_km: float) -> float:
    return (zz - zl) * (re_km + zl) / (re_km + zz)


@njit_or_identity(cache=True, fastmath=False)
def denss_kernel(
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
    gsurf: float,
    re_km: float,
) -> tuple[float, float]:
    """Evaluate the complete MSIS-86 ``denss`` branch without fast math."""

    z = alt if alt > za else za
    zg2 = _zeta_kernel(z, zlb, re_km)
    tt = tinf - (tinf - tlb) * math.exp(-s2 * zg2)
    ta = tt
    tz = tt
    denss = tz
    if alt < za:
        zg0 = _zeta_kernel(z0, za, re_km)
        dta = (tinf - ta) * s2 * ((re_km + zlb) / (re_km + za)) ** 2.0
        t12 = t0 + tr12 * (ta - t0)
        zg1 = _zeta_kernel(alt, za, re_km)
        dd = 0.666666 * zg0 * dta / ta / ta - 3.11111 * (1.0 / ta - 1.0 / t0) + 7.11111 * (
            1.0 / t12 - 1.0 / t0
        )
        cc = zg0 * dta / (2.0 * ta * ta) - (1.0 / ta - 1.0 / t0) - 2.0 * dd
        bb = (1.0 / ta - 1.0 / t0) - cc - dd
        x = (zg0 - zg1) / zg0
        x2 = x * x
        tz = 1.0 / (1.0 / t0 + bb * x2 + cc * x2 * x2 + dd * x2 * x2 * x2)
        denss = tz
    else:
        zg0 = 0.0
        x = 0.0
        x2 = 0.0
        bb = 0.0
        cc = 0.0
        dd = 0.0
    if xm != 0.0:
        if ta <= 0.0 or tz <= 0.0:
            tt = tlb
            ta = tlb
            tz = tlb
        glb = gsurf / (1.0 + zlb / re_km) ** 2.0
        gamma = xm * glb / (s2 * 831.4 * tinf)
        densa = dlb * (tlb / tt) ** (1.0 + alpha + gamma) * math.exp(-s2 * gamma * zg2)
        denss = densa
        if alt < za:
            glb = gsurf / (1.0 + za / re_km) ** 2.0
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
    return tz, denss
