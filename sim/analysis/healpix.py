"""Dependency-free HEALPix NESTED centers on the WGS84 authalic sphere.

The indexing equations follow the published HEALPix NESTED ``pix2loc``
definition. OEL maps the equal-area sphere to WGS84 with Snyder's authalic
latitude transform; physical coverage geometry continues to use the WGS84
ellipsoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM, WGS84_E2

HEALPIX_GRID_ID = "healpix_nest_wgs84_authalic_v1"
WGS84_AUTHALIC_INVERSE_TOLERANCE_RAD = 1.0e-13

_WGS84_E = float(np.sqrt(WGS84_E2))
_HALF_PI = 0.5 * np.pi
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)


def _authalic_q(geodetic_latitude_rad: np.ndarray) -> np.ndarray:
    phi = np.asarray(geodetic_latitude_rad, dtype=float)
    sin_phi = np.sin(phi)
    e_sin_phi = _WGS84_E * sin_phi
    denominator = 1.0 - WGS84_E2 * sin_phi * sin_phi
    logarithm = np.log((1.0 - e_sin_phi) / (1.0 + e_sin_phi))
    return (1.0 - WGS84_E2) * (
        sin_phi / denominator - logarithm / (2.0 * _WGS84_E)
    )


WGS84_AUTHALIC_QP = float(_authalic_q(np.array(_HALF_PI)))
WGS84_AUTHALIC_RADIUS_KM = float(WGS84_A_KM * np.sqrt(0.5 * WGS84_AUTHALIC_QP))
WGS84_SURFACE_AREA_KM2 = float(4.0 * np.pi * WGS84_AUTHALIC_RADIUS_KM**2)


@dataclass(frozen=True)
class HealpixWGS84Centers:
    """Canonical centers for a sorted set of NESTED HEALPix cell indices."""

    order: int
    cell_index: np.ndarray
    authalic_latitude_rad: np.ndarray
    longitude_rad: np.ndarray
    geodetic_latitude_rad: np.ndarray
    ecef_km: np.ndarray
    outward_normal_ecef: np.ndarray

    @property
    def cell_area_km2(self) -> float:
        return WGS84_SURFACE_AREA_KM2 / healpix_npix(self.order)


def healpix_nside(order: int) -> int:
    order_int = int(order)
    if isinstance(order, (bool, np.bool_)) or order_int != order:
        raise ValueError("HEALPix order must be an integer.")
    if not 0 <= order_int <= 29:
        raise ValueError("HEALPix order must be within [0, 29].")
    return 1 << order_int


def healpix_npix(order: int) -> int:
    nside = healpix_nside(order)
    return 12 * nside * nside


def authalic_latitude_rad(geodetic_latitude_rad: np.ndarray | float) -> np.ndarray:
    """Map WGS84 geodetic latitude to equal-area authalic latitude."""

    phi = np.asarray(geodetic_latitude_rad, dtype=float)
    if not np.all(np.isfinite(phi)):
        raise ValueError("Geodetic latitude must be finite.")
    if np.any(np.abs(phi) > _HALF_PI + WGS84_AUTHALIC_INVERSE_TOLERANCE_RAD):
        raise ValueError("Geodetic latitude must be within [-pi/2, pi/2].")
    phi = np.clip(phi, -_HALF_PI, _HALF_PI)
    return np.arcsin(np.clip(_authalic_q(phi) / WGS84_AUTHALIC_QP, -1.0, 1.0))


def geodetic_latitude_from_authalic_rad(
    authalic_latitude: np.ndarray | float,
    *,
    tolerance_rad: float = WGS84_AUTHALIC_INVERSE_TOLERANCE_RAD,
    max_iterations: int = 64,
) -> np.ndarray:
    """Invert WGS84 authalic latitude with a deterministic bracketed solve."""

    beta = np.asarray(authalic_latitude, dtype=float)
    if not np.all(np.isfinite(beta)):
        raise ValueError("Authalic latitude must be finite.")
    if np.any(np.abs(beta) > _HALF_PI + tolerance_rad):
        raise ValueError("Authalic latitude must be within [-pi/2, pi/2].")
    if not np.isfinite(float(tolerance_rad)) or float(tolerance_rad) <= 0.0:
        raise ValueError("Authalic inverse tolerance must be positive and finite.")
    if int(max_iterations) <= 0:
        raise ValueError("Authalic inverse max_iterations must be positive.")

    clipped = np.clip(beta, -_HALF_PI, _HALF_PI)
    flat = clipped.reshape(-1)
    result = np.empty_like(flat)
    north = flat == _HALF_PI
    south = flat == -_HALF_PI
    polar = north | south
    result[north] = _HALF_PI
    result[south] = -_HALF_PI

    active = ~polar
    if np.any(active):
        target = WGS84_AUTHALIC_QP * np.sin(flat[active])
        lower = np.full(target.shape, -_HALF_PI, dtype=float)
        upper = np.full(target.shape, _HALF_PI, dtype=float)
        phi = flat[active].copy()
        converged = False
        for _ in range(int(max_iterations)):
            residual = _authalic_q(phi) - target
            lower = np.where(residual < 0.0, phi, lower)
            upper = np.where(residual >= 0.0, phi, upper)
            sin_phi = np.sin(phi)
            derivative = (
                2.0
                * (1.0 - WGS84_E2)
                * np.cos(phi)
                / (1.0 - WGS84_E2 * sin_phi * sin_phi) ** 2
            )
            midpoint = 0.5 * (lower + upper)
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = phi - residual / derivative
            use_newton = np.isfinite(newton) & (newton > lower) & (newton < upper)
            next_phi = np.where(use_newton, newton, midpoint)
            if float(np.max(np.abs(next_phi - phi))) <= 0.25 * float(tolerance_rad):
                phi = next_phi
                converged = True
                break
            phi = next_phi
        if not converged and float(np.max(upper - lower)) > 2.0 * float(tolerance_rad):
            raise RuntimeError("WGS84 authalic inverse did not converge.")
        result[active] = phi
    return result.reshape(clipped.shape)


def _validate_cell_indices(order: int, cell_indices: np.ndarray | None) -> np.ndarray:
    npix = healpix_npix(order)
    if cell_indices is None:
        return np.arange(npix, dtype=np.int64)
    raw = np.asarray(cell_indices)
    if raw.ndim != 1:
        raise ValueError("HEALPix cell indices must be a one-dimensional array.")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("HEALPix cell indices must be integers.")
    indices = raw.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= npix):
        raise ValueError(f"HEALPix cell indices must be within [0, {npix}).")
    if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
        raise ValueError("HEALPix cell indices must be unique and strictly increasing.")
    return indices


def healpix_nested_centers_authalic(
    order: int,
    cell_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical NESTED center authalic latitude and longitude."""

    nside = healpix_nside(order)
    order_int = int(order)
    indices = _validate_cell_indices(order_int, cell_indices)
    npface = nside * nside
    face = indices // npface
    intra_face = indices % npface

    ix = np.zeros(indices.shape, dtype=np.int64)
    iy = np.zeros(indices.shape, dtype=np.int64)
    for bit in range(order_int):
        ix |= ((intra_face >> (2 * bit)) & 1) << bit
        iy |= ((intra_face >> (2 * bit + 1)) & 1) << bit

    jr = _JRLL[face] * nside - ix - iy - 1
    north = jr < nside
    south = jr > 3 * nside
    nr = np.where(north, jr, np.where(south, 4 * nside - jr, nside))
    fact2 = 1.0 / (3.0 * nside * nside)
    fact1 = 2.0 / (3.0 * nside)
    z = np.where(
        north,
        1.0 - nr * nr * fact2,
        np.where(south, nr * nr * fact2 - 1.0, (2 * nside - jr) * fact1),
    )
    tmp = _JPLL[face] * nr + ix - iy
    tmp = np.mod(tmp, 8 * nr)
    longitude = np.where(
        nr == nside,
        np.pi * tmp / (4.0 * nside),
        np.pi * tmp / (4.0 * nr),
    )
    authalic_latitude = np.arcsin(np.clip(z, -1.0, 1.0))
    longitude = (longitude + np.pi) % (2.0 * np.pi) - np.pi
    return authalic_latitude.astype(float), longitude.astype(float)


def healpix_nested_cell_indices(
    order: int,
    authalic_latitude: np.ndarray | float,
    longitude: np.ndarray | float,
) -> np.ndarray:
    """Map authalic-sphere coordinates to canonical HEALPix NESTED cells.

    Inputs broadcast using NumPy rules. Longitudes are periodic and normalized
    internally; latitude must be within the closed spherical range.
    """

    nside = healpix_nside(order)
    order_int = int(order)
    beta, phi = np.broadcast_arrays(
        np.asarray(authalic_latitude, dtype=float),
        np.asarray(longitude, dtype=float),
    )
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(phi)):
        raise ValueError("HEALPix latitude and longitude must be finite.")
    if np.any(np.abs(beta) > _HALF_PI + WGS84_AUTHALIC_INVERSE_TOLERANCE_RAD):
        raise ValueError("HEALPix latitude must be within [-pi/2, pi/2].")

    shape = beta.shape
    z = np.sin(np.clip(beta, -_HALF_PI, _HALF_PI)).reshape(-1)
    za = np.abs(z)
    tt = (np.mod(phi, 2.0 * np.pi) / _HALF_PI).reshape(-1)
    face = np.empty(z.shape, dtype=np.int64)
    ix = np.empty(z.shape, dtype=np.int64)
    iy = np.empty(z.shape, dtype=np.int64)

    equatorial = za <= (2.0 / 3.0)
    if np.any(equatorial):
        z_eq = z[equatorial]
        tt_eq = tt[equatorial]
        jp_boundary = nside * (0.5 + tt_eq - 0.75 * z_eq)
        # HEALPix assigns an exact ascending-edge tie to the adjacent cell on
        # the lower-jp side.  Express the tie explicitly instead of depending
        # on platform-specific transcendental rounding at cardinal meridians.
        jp = np.floor(np.nextafter(jp_boundary, -np.inf)).astype(np.int64)
        jm = np.floor(nside * (0.5 + tt_eq + 0.75 * z_eq)).astype(np.int64)
        ifp = jp >> order_int
        ifm = jm >> order_int
        face_eq = np.where(
            ifp == ifm,
            ifp | 4,
            np.where(ifp < ifm, ifp, ifm + 8),
        )
        face[equatorial] = face_eq
        ix[equatorial] = jm & (nside - 1)
        iy[equatorial] = nside - (jp & (nside - 1)) - 1

    polar = ~equatorial
    if np.any(polar):
        z_pol = z[polar]
        za_pol = za[polar]
        tt_pol = tt[polar]
        ntt = np.minimum(3, np.floor(tt_pol).astype(np.int64))
        tp = tt_pol - ntt
        radial = nside * np.sqrt(3.0 * (1.0 - za_pol))
        jp = np.minimum(nside - 1, np.floor(tp * radial).astype(np.int64))
        jm = np.minimum(nside - 1, np.floor((1.0 - tp) * radial).astype(np.int64))
        north = z_pol >= 0.0
        face[polar] = np.where(north, ntt, ntt + 8)
        ix[polar] = np.where(north, nside - jm - 1, jp)
        iy[polar] = np.where(north, nside - jp - 1, jm)

    intra_face = np.zeros(z.shape, dtype=np.int64)
    for bit in range(order_int):
        intra_face |= ((ix >> bit) & 1) << (2 * bit)
        intra_face |= ((iy >> bit) & 1) << (2 * bit + 1)
    result = (face << (2 * order_int)) + intra_face
    return result.reshape(shape)


def wgs84_points_to_healpix_nested(
    order: int,
    geodetic_latitude_deg: np.ndarray | float,
    longitude_deg: np.ndarray | float,
) -> np.ndarray:
    """Map zero-height WGS84 points to canonical HEALPix NESTED cells."""

    latitude = np.asarray(geodetic_latitude_deg, dtype=float)
    longitude = np.asarray(longitude_deg, dtype=float)
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ValueError("WGS84 point latitude and longitude must be finite.")
    if np.any(np.abs(latitude) > 90.0):
        raise ValueError("WGS84 geodetic latitude must be within [-90, 90] degrees.")
    latitude, longitude = np.broadcast_arrays(latitude, longitude)
    beta = authalic_latitude_rad(np.deg2rad(latitude))
    return healpix_nested_cell_indices(order, beta, np.deg2rad(longitude))


def healpix_wgs84_centers(
    order: int,
    cell_indices: np.ndarray | None = None,
) -> HealpixWGS84Centers:
    """Return WGS84 surface centers and unit ellipsoid normals."""

    indices = _validate_cell_indices(order, cell_indices)
    beta, longitude = healpix_nested_centers_authalic(order, indices)
    latitude = geodetic_latitude_from_authalic_rad(beta)
    sin_lat = np.sin(latitude)
    cos_lat = np.cos(latitude)
    prime_vertical_radius = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    ecef = np.column_stack(
        (
            prime_vertical_radius * cos_lat * np.cos(longitude),
            prime_vertical_radius * cos_lat * np.sin(longitude),
            prime_vertical_radius * (1.0 - WGS84_E2) * sin_lat,
        )
    )
    normal = np.column_stack(
        (
            ecef[:, 0] / (WGS84_A_KM**2),
            ecef[:, 1] / (WGS84_A_KM**2),
            ecef[:, 2] / (WGS84_B_KM**2),
        )
    )
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    return HealpixWGS84Centers(
        order=int(order),
        cell_index=indices,
        authalic_latitude_rad=beta,
        longitude_rad=longitude,
        geodetic_latitude_rad=latitude,
        ecef_km=ecef,
        outward_normal_ecef=normal,
    )


__all__ = [
    "HEALPIX_GRID_ID",
    "HealpixWGS84Centers",
    "WGS84_AUTHALIC_INVERSE_TOLERANCE_RAD",
    "WGS84_AUTHALIC_QP",
    "WGS84_AUTHALIC_RADIUS_KM",
    "WGS84_SURFACE_AREA_KM2",
    "authalic_latitude_rad",
    "geodetic_latitude_from_authalic_rad",
    "healpix_nested_cell_indices",
    "healpix_nested_centers_authalic",
    "healpix_npix",
    "healpix_nside",
    "healpix_wgs84_centers",
    "wgs84_points_to_healpix_nested",
]
