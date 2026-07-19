"""Optional compiled WGS-84 geodetic-coordinate kernels."""

from __future__ import annotations

import math

import numpy as np

from sim.acceleration.optional import njit_or_identity

WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@njit_or_identity(cache=True, fastmath=False)
def ecef_to_geodetic_deg_km_kernel(r_ecef_km: np.ndarray) -> tuple[float, float, float]:
    """Match OEL's iterative WGS-84 latitude/longitude/altitude conversion."""

    x = float(r_ecef_km[0])
    y = float(r_ecef_km[1])
    z = float(r_ecef_km[2])
    lon = math.atan2(y, x)
    p = float(np.hypot(x, y))
    if p <= 1.0e-12:
        lat = math.pi / 2.0 if z >= 0.0 else -math.pi / 2.0
        alt = abs(z) - WGS84_B_KM
        return float(np.rad2deg(lat)), float(np.rad2deg(lon)), alt

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        prime_vertical = WGS84_A_KM / math.sqrt(
            max(1.0 - WGS84_E2 * sin_lat * sin_lat, 1.0e-15)
        )
        alt = p / max(math.cos(lat), 1.0e-15) - prime_vertical
        lat_next = math.atan2(
            z,
            p
            * (
                1.0
                - WGS84_E2
                * prime_vertical
                / max(prime_vertical + alt, 1.0e-15)
            ),
        )
        if abs(lat_next - lat) <= 1.0e-13:
            lat = lat_next
            break
        lat = lat_next
    sin_lat = math.sin(lat)
    prime_vertical = WGS84_A_KM / math.sqrt(
        max(1.0 - WGS84_E2 * sin_lat * sin_lat, 1.0e-15)
    )
    alt = p / max(math.cos(lat), 1.0e-15) - prime_vertical
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), alt
