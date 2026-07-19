from __future__ import annotations

import math

import numpy as np

WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef_km(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    alt = float(alt_km)
    s = float(np.sin(lat))
    c = float(np.cos(lat))
    n = WGS84_A_KM / np.sqrt(max(1.0 - WGS84_E2 * s * s, 1e-15))
    x = (n + alt) * c * np.cos(lon)
    y = (n + alt) * c * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * s
    return np.array([x, y, z], dtype=float)


def _ecef_to_geodetic_lat_alt_rad_km(x: float, y: float, z: float) -> tuple[float, float]:
    p = float(np.hypot(x, y))
    if p <= 1e-12:
        lat = math.pi / 2.0 if z >= 0.0 else -math.pi / 2.0
        return lat, abs(z) - WGS84_B_KM

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        s = math.sin(lat)
        n = WGS84_A_KM / math.sqrt(max(1.0 - WGS84_E2 * s * s, 1e-15))
        alt = p / max(math.cos(lat), 1e-15) - n
        lat_next = math.atan2(z, p * (1.0 - WGS84_E2 * n / max(n + alt, 1e-15)))
        if abs(lat_next - lat) <= 1e-13:
            lat = lat_next
            break
        lat = lat_next
    s = math.sin(lat)
    n = WGS84_A_KM / math.sqrt(max(1.0 - WGS84_E2 * s * s, 1e-15))
    alt = p / max(math.cos(lat), 1e-15) - n
    return lat, alt


def ecef_to_geodetic_altitude_km(r_ecef_km: np.ndarray) -> float:
    """Return only WGS-84 altitude, avoiding unused angle calculations."""
    x, y, z = np.asarray(r_ecef_km, dtype=float).reshape(3)
    _, alt = _ecef_to_geodetic_lat_alt_rad_km(float(x), float(y), float(z))
    return float(alt)


def ecef_to_geodetic_deg_km(r_ecef_km: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.asarray(r_ecef_km, dtype=float).reshape(3)
    lon = math.atan2(float(y), float(x))
    lat, alt = _ecef_to_geodetic_lat_alt_rad_km(float(x), float(y), float(z))
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


def ecef_to_enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    slat = float(np.sin(lat))
    clat = float(np.cos(lat))
    slon = float(np.sin(lon))
    clon = float(np.cos(lon))
    return np.array(
        [
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat],
        ],
        dtype=float,
    )


def enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    return ecef_to_enu_rotation(lat_deg, lon_deg).T
