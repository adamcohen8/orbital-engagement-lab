from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.utils.geodesy import geodetic_to_ecef_km


def _cadence_due(t_s: float, last_update_t_s: float, cadence_s: float) -> bool:
    """Allow roundoff at a due boundary without admitting extra samples."""
    if not math.isfinite(t_s):
        raise ValueError("sensor time_s must be finite.")
    elapsed = t_s - last_update_t_s
    if elapsed >= cadence_s:
        return True
    if elapsed <= 0.0 or not math.isfinite(elapsed):
        return False
    tolerance = min(
        cadence_s * 1.0e-9,
        4.0 * max(math.ulp(t_s), math.ulp(last_update_t_s), math.ulp(cadence_s)),
    )
    return cadence_s - elapsed <= tolerance


@dataclass(frozen=True)
class GroundSite:
    lat_rad: float
    lon_rad: float
    min_elevation_rad: float = 0.0
    alt_km: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(float(value))
            for value in (self.lat_rad, self.lon_rad, self.alt_km, self.min_elevation_rad)
        ):
            raise ValueError("ground-site coordinates and elevation mask must be finite")
        if not -0.5 * np.pi <= self.lat_rad <= 0.5 * np.pi:
            raise ValueError("ground-site latitude must be within [-pi/2, pi/2]")
        if not 0.0 <= self.min_elevation_rad <= 0.5 * np.pi:
            raise ValueError("ground-site minimum elevation must be within [0, pi/2]")


@dataclass(frozen=True)
class AccessConfig:
    update_cadence_s: float = 1.0
    max_range_km: float | None = None
    fov_half_angle_rad: float | None = None
    solid_angle_sr: float | None = None
    require_ground_visibility: bool = False
    ground_site: GroundSite | None = None
    frame_context: FrameContext | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.update_cadence_s)) or float(self.update_cadence_s) <= 0.0:
            raise ValueError("update_cadence_s must be positive and finite.")
        if self.max_range_km is not None and (
            not np.isfinite(float(self.max_range_km)) or float(self.max_range_km) <= 0.0
        ):
            raise ValueError("max_range_km must be positive and finite when provided.")
        if self.fov_half_angle_rad is not None and (
            not np.isfinite(float(self.fov_half_angle_rad))
            or not 0.0 <= float(self.fov_half_angle_rad) <= np.pi
        ):
            raise ValueError("fov_half_angle_rad must be finite and within [0, pi].")
        if self.solid_angle_sr is not None and (
            not np.isfinite(float(self.solid_angle_sr))
            or not 0.0 <= float(self.solid_angle_sr) <= 4.0 * np.pi
        ):
            raise ValueError("solid_angle_sr must be finite and within [0, 4*pi].")
        if self.require_ground_visibility and (
            self.frame_context is None or self.frame_context.jd_utc_start is None
        ):
            raise ValueError(
                "ground visibility requires a FrameContext with an absolute jd_utc_start."
            )


class AccessModel:
    def __init__(self, cfg: AccessConfig):
        self.cfg = cfg
        self._last_update_t_s = -np.inf

    def evaluate(
        self,
        observer_eci_km: np.ndarray,
        target_eci_km: np.ndarray,
        t_s: float,
        *,
        boresight_eci: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        if not _cadence_due(t_s, self._last_update_t_s, self.cfg.update_cadence_s):
            return False, "cadence"

        observer = np.asarray(observer_eci_km, dtype=float)
        target = np.asarray(target_eci_km, dtype=float)
        boresight = None if boresight_eci is None else np.asarray(boresight_eci, dtype=float)
        for name, vector in (("observer", observer), ("target", target), ("boresight", boresight)):
            if vector is not None and (vector.shape != (3,) or not np.all(np.isfinite(vector))):
                raise ValueError(f"access {name} must contain three finite coordinates.")
        ground_zenith: np.ndarray | None = None
        if self.cfg.require_ground_visibility:
            if self.cfg.ground_site is None:
                return False, "ground_site_missing"
            assert self.cfg.frame_context is not None
            observer, ground_zenith = _ground_site_geometry_eci(
                self.cfg.ground_site,
                t_s,
                self.cfg.frame_context,
            )
        los = target - observer
        rng = np.linalg.norm(los)
        if self.cfg.max_range_km is not None and rng > self.cfg.max_range_km:
            return False, "range"

        fov_half_angle_rad = self.cfg.fov_half_angle_rad
        if fov_half_angle_rad is None:
            fov_half_angle_rad = _solid_angle_to_half_angle_rad(self.cfg.solid_angle_sr)
        if fov_half_angle_rad is not None and rng > 0.0:
            if boresight_eci is None:
                boresight = observer / max(np.linalg.norm(observer), 1e-12)
            else:
                boresight = np.array(boresight_eci, dtype=float).reshape(3)
                bn = float(np.linalg.norm(boresight))
                if bn <= 0.0:
                    return False, "boresight"
                boresight = boresight / bn
            cosang = np.clip(np.dot(boresight, los / rng), -1.0, 1.0)
            if np.arccos(cosang) > fov_half_angle_rad:
                return False, "solid_angle"

        if self.cfg.require_ground_visibility:
            assert self.cfg.ground_site is not None
            assert ground_zenith is not None
            zenith = ground_zenith
            elevation = float(np.arcsin(np.clip(np.dot(zenith, los / max(rng, 1.0e-12)), -1.0, 1.0)))
            if elevation < self.cfg.ground_site.min_elevation_rad:
                return False, "ground_elevation"

        return True, "ok"

    def can_update(
        self,
        observer_eci_km: np.ndarray,
        target_eci_km: np.ndarray,
        t_s: float,
        *,
        boresight_eci: np.ndarray | None = None,
    ) -> bool:
        allowed, _ = self.evaluate(observer_eci_km, target_eci_km, t_s, boresight_eci=boresight_eci)
        if not allowed:
            return False
        self._last_update_t_s = t_s
        return True


def _ground_site_geometry_eci(
    site: GroundSite,
    t_s: float,
    frame_context: FrameContext,
) -> tuple[np.ndarray, np.ndarray]:
    lat = float(site.lat_rad)
    lon = float(site.lon_rad)
    position_ecef = geodetic_to_ecef_km(
        np.rad2deg(lat),
        np.rad2deg(lon),
        float(site.alt_km),
    )
    up_ecef = np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=float,
    )
    ecef_from_eci = eci_to_ecef_rotation_context(float(t_s), frame_context)
    return ecef_from_eci.T @ position_ecef, ecef_from_eci.T @ up_ecef


def _solid_angle_to_half_angle_rad(solid_angle_sr: float | None) -> float | None:
    if solid_angle_sr is None:
        return None
    omega = float(solid_angle_sr)
    if not np.isfinite(omega) or omega <= 0.0:
        return 0.0
    if omega >= (4.0 * np.pi - 1e-12):
        return None
    cos_half = float(np.clip(1.0 - omega / (2.0 * np.pi), -1.0, 1.0))
    return float(np.arccos(cos_half))
