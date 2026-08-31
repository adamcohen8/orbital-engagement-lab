"""Optical collection geometry and transparent first-order quality metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from sim.analysis.sensor_footprint_geometry import (
    PRIMARY_REASON_NAMES,
    HardFOVPattern,
    SurfaceServiceConstraints,
    evaluate_rich_surface_targets_ecef,
    fov_boundary_rays_sensor,
    intersect_rays_wgs84,
)
from sim.utils.geodesy import ecef_to_enu_rotation, ecef_to_geodetic_deg_km, geodetic_to_ecef_km

OPTICAL_COLLECTION_MODEL = "oel.transparent-optical-collection.v1"
COLLECTION_REASON_NAMES = (
    "available",
    "earth_blocked",
    "outside_pattern",
    "outside_range",
    "off_nadir_exceeded",
    "incidence_exceeded",
    "illumination_rejected",
    "gimbal_limit_exceeded",
    "slew_rate_exceeded",
    "spacecraft_eclipsed",
    "resolution_exceeded",
)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object.")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(unknown)}.")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer.")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True)
class GroundTarget:
    target_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_km: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GroundTarget:
        raw = _mapping(data, "target")
        _reject_unknown(raw, {"target_id", "latitude_deg", "longitude_deg", "altitude_km"}, "target")
        item = cls(
            target_id=_required_text(raw.get("target_id"), "target.target_id"),
            latitude_deg=_finite_number(raw.get("latitude_deg"), "target.latitude_deg"),
            longitude_deg=_finite_number(raw.get("longitude_deg"), "target.longitude_deg"),
            altitude_km=_finite_number(raw.get("altitude_km", 0.0), "target.altitude_km"),
        )
        if not -90.0 <= item.latitude_deg <= 90.0:
            raise ValueError("target.latitude_deg must lie within [-90, 90].")
        if not -180.0 <= item.longitude_deg <= 180.0:
            raise ValueError("target.longitude_deg must lie within [-180, 180].")
        if abs(item.altitude_km) > 1.0e-12:
            raise ValueError("The v1 optical collection workflow supports WGS84 surface targets at altitude_km = 0 only.")
        return item

    @property
    def ecef_km(self) -> np.ndarray:
        return geodetic_to_ecef_km(self.latitude_deg, self.longitude_deg, self.altitude_km)

    @property
    def outward_normal_ecef(self) -> np.ndarray:
        latitude = math.radians(self.latitude_deg)
        longitude = math.radians(self.longitude_deg)
        return np.array(
            [math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude)],
            dtype=float,
        )


@dataclass(frozen=True)
class OpticalPayload:
    sensor_id: str
    pattern: HardFOVPattern
    pointing_mode: str
    maximum_gimbal_off_nadir_rad: float
    maximum_slew_rate_rad_s: float
    settling_time_s: float
    minimum_collection_duration_s: float
    aperture_m: float
    wavelength_m: float
    focal_length_m: float
    detector_pitch_m: float
    data_generation_rate_bps: float
    boundary_samples_per_edge: int = 16

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> OpticalPayload:
        raw = _mapping(data, "sensor")
        _reject_unknown(
            raw,
            {
                "sensor_id",
                "pattern",
                "pointing_mode",
                "maximum_gimbal_off_nadir_deg",
                "maximum_slew_rate_deg_s",
                "settling_time_s",
                "minimum_collection_duration_s",
                "aperture_m",
                "wavelength_nm",
                "focal_length_m",
                "detector_pitch_um",
                "data_generation_rate_mbps",
                "boundary_samples_per_edge",
            },
            "sensor",
        )
        pattern_raw = _mapping(raw.get("pattern"), "sensor.pattern")
        _reject_unknown(pattern_raw, {"kind", "x_half_angle_deg", "y_half_angle_deg"}, "sensor.pattern")
        kind = _required_text(pattern_raw.get("kind", "rectangular_hard_fov"), "sensor.pattern.kind").lower()
        x = math.radians(_finite_number(pattern_raw.get("x_half_angle_deg"), "sensor.pattern.x_half_angle_deg"))
        y = math.radians(_finite_number(pattern_raw.get("y_half_angle_deg"), "sensor.pattern.y_half_angle_deg"))
        pattern = HardFOVPattern(kind=kind, x_half_angle_rad=x, y_half_angle_rad=y)
        item = cls(
            sensor_id=_required_text(raw.get("sensor_id"), "sensor.sensor_id"),
            pattern=pattern,
            pointing_mode=_required_text(
                raw.get("pointing_mode", "target_track_gimbal"), "sensor.pointing_mode"
            ).lower(),
            maximum_gimbal_off_nadir_rad=math.radians(
                _finite_number(raw.get("maximum_gimbal_off_nadir_deg", 0.0), "sensor.maximum_gimbal_off_nadir_deg")
            ),
            maximum_slew_rate_rad_s=math.radians(
                _finite_number(raw.get("maximum_slew_rate_deg_s", 0.0), "sensor.maximum_slew_rate_deg_s")
            ),
            settling_time_s=_finite_number(raw.get("settling_time_s", 0.0), "sensor.settling_time_s"),
            minimum_collection_duration_s=_finite_number(
                raw.get("minimum_collection_duration_s", 0.0), "sensor.minimum_collection_duration_s"
            ),
            aperture_m=_finite_number(raw.get("aperture_m", 0.0), "sensor.aperture_m"),
            wavelength_m=_finite_number(raw.get("wavelength_nm", 0.0), "sensor.wavelength_nm") * 1.0e-9,
            focal_length_m=_finite_number(raw.get("focal_length_m", 0.0), "sensor.focal_length_m"),
            detector_pitch_m=_finite_number(raw.get("detector_pitch_um", 0.0), "sensor.detector_pitch_um") * 1.0e-6,
            data_generation_rate_bps=_finite_number(
                raw.get("data_generation_rate_mbps", 0.0), "sensor.data_generation_rate_mbps"
            ) * 1.0e6,
            boundary_samples_per_edge=_integer(
                raw.get("boundary_samples_per_edge", 16), "sensor.boundary_samples_per_edge"
            ),
        )
        if item.pointing_mode not in {"nadir_fixed", "target_track_gimbal"}:
            raise ValueError("sensor.pointing_mode must be nadir_fixed or target_track_gimbal.")
        for field_name in (
            "maximum_gimbal_off_nadir_rad",
            "maximum_slew_rate_rad_s",
            "settling_time_s",
            "minimum_collection_duration_s",
        ):
            value = float(getattr(item, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"sensor.{field_name} must be finite and nonnegative.")
        if item.maximum_gimbal_off_nadir_rad > 0.5 * math.pi:
            raise ValueError("maximum_gimbal_off_nadir_deg must not exceed 90 degrees.")
        for field_name in ("aperture_m", "wavelength_m", "focal_length_m", "detector_pitch_m", "data_generation_rate_bps"):
            value = float(getattr(item, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"sensor.{field_name} must be positive and finite.")
        if not 2 <= item.boundary_samples_per_edge <= 4096:
            raise ValueError("sensor.boundary_samples_per_edge must lie within [2, 4096].")
        if item.pointing_mode == "nadir_fixed" and item.maximum_gimbal_off_nadir_rad != 0.0:
            raise ValueError("A nadir_fixed sensor must declare maximum_gimbal_off_nadir_deg = 0.")
        return item

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "pattern": {
                "kind": self.pattern.kind,
                "x_half_angle_deg": math.degrees(self.pattern.x_half_angle_rad),
                "y_half_angle_deg": math.degrees(self.pattern.y_half_angle_rad),
            },
            "pointing_mode": self.pointing_mode,
            "maximum_gimbal_off_nadir_deg": math.degrees(self.maximum_gimbal_off_nadir_rad),
            "maximum_slew_rate_deg_s": math.degrees(self.maximum_slew_rate_rad_s),
            "settling_time_s": self.settling_time_s,
            "minimum_collection_duration_s": self.minimum_collection_duration_s,
            "aperture_m": self.aperture_m,
            "wavelength_nm": self.wavelength_m * 1.0e9,
            "focal_length_m": self.focal_length_m,
            "detector_pitch_um": self.detector_pitch_m * 1.0e6,
            "data_generation_rate_mbps": self.data_generation_rate_bps / 1.0e6,
            "boundary_samples_per_edge": self.boundary_samples_per_edge,
        }


@dataclass(frozen=True)
class CollectionConstraints:
    surface: SurfaceServiceConstraints
    maximum_range_km: float | None
    minimum_spacecraft_illumination_fraction: float
    maximum_effective_resolution_m: float | None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> CollectionConstraints:
        raw = {} if data is None else _mapping(data, "constraints")
        _reject_unknown(
            raw,
            {
                "maximum_target_off_nadir_deg",
                "maximum_incidence_deg",
                "minimum_sun_elevation_deg",
                "maximum_sun_elevation_deg",
                "maximum_range_km",
                "minimum_spacecraft_illumination_fraction",
                "maximum_effective_resolution_m",
            },
            "constraints",
        )
        surface = SurfaceServiceConstraints(
            maximum_target_off_nadir_rad=_optional_angle(raw, "maximum_target_off_nadir_deg"),
            maximum_incidence_rad=_optional_angle(raw, "maximum_incidence_deg"),
            minimum_sun_elevation_rad=_optional_angle(raw, "minimum_sun_elevation_deg"),
            maximum_sun_elevation_rad=_optional_angle(raw, "maximum_sun_elevation_deg"),
        )
        maximum_range = _optional_positive(raw, "maximum_range_km")
        maximum_resolution = _optional_positive(raw, "maximum_effective_resolution_m")
        illumination = _finite_number(
            raw.get("minimum_spacecraft_illumination_fraction", 0.0),
            "constraints.minimum_spacecraft_illumination_fraction",
        )
        if not 0.0 <= illumination <= 1.0:
            raise ValueError("minimum_spacecraft_illumination_fraction must lie within [0, 1].")
        return cls(surface, maximum_range, illumination, maximum_resolution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_target_off_nadir_deg": _degrees_or_none(self.surface.maximum_target_off_nadir_rad),
            "maximum_incidence_deg": _degrees_or_none(self.surface.maximum_incidence_rad),
            "minimum_sun_elevation_deg": _degrees_or_none(self.surface.minimum_sun_elevation_rad),
            "maximum_sun_elevation_deg": _degrees_or_none(self.surface.maximum_sun_elevation_rad),
            "maximum_range_km": self.maximum_range_km,
            "minimum_spacecraft_illumination_fraction": self.minimum_spacecraft_illumination_fraction,
            "maximum_effective_resolution_m": self.maximum_effective_resolution_m,
        }


def _optional_angle(raw: Mapping[str, Any], key: str) -> float | None:
    return None if raw.get(key) is None else math.radians(_finite_number(raw[key], f"constraints.{key}"))


def _optional_positive(raw: Mapping[str, Any], key: str) -> float | None:
    if raw.get(key) is None:
        return None
    value = _finite_number(raw[key], f"constraints.{key}")
    if value <= 0.0:
        raise ValueError(f"{key} must be positive and finite when supplied.")
    return value


def _degrees_or_none(value: float | None) -> float | None:
    return None if value is None else math.degrees(value)


def local_nadir_frame_sensor_from_eci(state_eci_km_km_s: Sequence[float]) -> np.ndarray:
    state = np.asarray(state_eci_km_km_s, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("A local nadir frame requires one finite Cartesian state.")
    position_norm = float(np.linalg.norm(state[:3]))
    if position_norm <= 0.0:
        raise ValueError("A local nadir frame requires nonzero position.")
    z_axis = -state[:3] / position_norm
    along_track = state[3:] - float(state[3:] @ z_axis) * z_axis
    along_track_norm = float(np.linalg.norm(along_track))
    if along_track_norm <= 1.0e-12:
        raise ValueError("A local nadir frame requires velocity transverse to position.")
    along_track /= along_track_norm
    # Public hard-FOV mounting semantics are +X cross-track, +Y along-track,
    # and +Z boresight.  Preserve those axes for rectangular and pushbroom FOVs.
    x_axis = np.cross(along_track, z_axis)
    x_axis /= float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    result = np.vstack((x_axis, y_axis, z_axis))
    if float(np.linalg.det(result)) <= 0.0:
        raise RuntimeError("Local nadir-frame construction produced an improper rotation.")
    return result


def sensor_frame_and_gimbal_vector(
    state_eci_km_km_s: Sequence[float], target_eci_km: Sequence[float], *, pointing_mode: str
) -> tuple[np.ndarray, np.ndarray, float]:
    state = np.asarray(state_eci_km_km_s, dtype=float)
    target = np.asarray(target_eci_km, dtype=float)
    local = local_nadir_frame_sensor_from_eci(state)
    line_of_sight = target - state[:3]
    line_of_sight /= float(np.linalg.norm(line_of_sight))
    gimbal_vector = local @ line_of_sight
    gimbal_angle = math.acos(float(np.clip(gimbal_vector[2], -1.0, 1.0)))
    if pointing_mode == "nadir_fixed":
        # The target may be off boresight while still inside a fixed sensor's
        # hard FOV.  A fixed mounting commands no gimbal motion.
        return local, np.array([0.0, 0.0, 1.0], dtype=float), 0.0
    z_axis = line_of_sight
    x_axis = local[0] - float(local[0] @ z_axis) * z_axis
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1.0e-12:
        x_axis = local[1] - float(local[1] @ z_axis) * z_axis
        x_norm = float(np.linalg.norm(x_axis))
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack((x_axis, y_axis, z_axis)), gimbal_vector, gimbal_angle


def optical_quality_metrics(
    *, slant_range_km: float, incidence_angle_rad: float, payload: OpticalPayload
) -> dict[str, float]:
    range_m = float(slant_range_km) * 1000.0
    projection = max(math.cos(float(incidence_angle_rad)), 1.0e-6)
    diffraction_m = 1.22 * payload.wavelength_m * range_m / payload.aperture_m / projection
    ground_sample_distance_m = payload.detector_pitch_m * range_m / payload.focal_length_m / projection
    effective_resolution_m = max(diffraction_m, 2.0 * ground_sample_distance_m)
    return {
        "diffraction_limited_resolution_m": diffraction_m,
        "ground_sample_distance_m": ground_sample_distance_m,
        "sampling_limited_resolution_m": 2.0 * ground_sample_distance_m,
        "effective_resolution_m": effective_resolution_m,
        "projection_cosine": projection,
    }


def footprint_boundary_evidence(
    *, observer_ecef_km: Sequence[float], dcm_sensor_from_ecef: Sequence[Sequence[float]], payload: OpticalPayload,
    target: GroundTarget,
) -> dict[str, Any]:
    observer = np.asarray(observer_ecef_km, dtype=float)
    rotation = np.asarray(dcm_sensor_from_ecef, dtype=float)
    if observer.shape != (3,) or not np.all(np.isfinite(observer)):
        raise ValueError("observer_ecef_km must contain three finite values.")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("dcm_sensor_from_ecef must be a finite 3x3 matrix.")
    if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1.0e-10) or np.linalg.det(rotation) <= 0.0:
        raise ValueError("dcm_sensor_from_ecef must be a proper orthonormal rotation.")
    rays_sensor = fov_boundary_rays_sensor(payload.pattern, samples_per_edge=payload.boundary_samples_per_edge)
    rays_ecef = rays_sensor @ rotation
    intersection = intersect_rays_wgs84(observer, rays_ecef)
    latitude: list[float | None] = []
    longitude: list[float | None] = []
    for hit, point in zip(intersection.hit, intersection.point_ecef_km, strict=True):
        if not hit:
            latitude.append(None)
            longitude.append(None)
            continue
        lat, lon, _alt = ecef_to_geodetic_deg_km(point)
        latitude.append(lat)
        longitude.append(lon)
    complete = bool(np.all(intersection.hit))
    area_km2: float | None = None
    width_km: float | None = None
    height_km: float | None = None
    if complete:
        enu = ecef_to_enu_rotation(target.latitude_deg, target.longitude_deg)
        relative_enu = (intersection.point_ecef_km - target.ecef_km) @ enu.T
        east = relative_enu[:, 0]
        north = relative_enu[:, 1]
        area_km2 = 0.5 * abs(float(np.dot(east, np.roll(north, -1)) - np.dot(north, np.roll(east, -1))))
        count = payload.boundary_samples_per_edge
        corners = relative_enu[[0, count, 2 * count, 3 * count], :2]
        width_km = 0.5 * (float(np.linalg.norm(corners[1] - corners[0])) + float(np.linalg.norm(corners[2] - corners[3])))
        height_km = 0.5 * (float(np.linalg.norm(corners[3] - corners[0])) + float(np.linalg.norm(corners[2] - corners[1])))
    return {
        "disposition": "complete" if complete else "partial" if bool(np.any(intersection.hit)) else "no_intersection",
        "boundary_hit": intersection.hit.tolist(),
        "boundary_latitude_deg": latitude,
        "boundary_longitude_deg": longitude,
        "tangent_plane_area_km2": area_km2,
        "corner_chord_width_km": width_km,
        "corner_chord_height_km": height_km,
        "area_method": "target_centered_enu_shoelace" if complete else None,
    }


def evaluate_collection_sample(
    *, observer_ecef_km: Sequence[float], target: GroundTarget, dcm_sensor_from_ecef: Sequence[Sequence[float]],
    payload: OpticalPayload, constraints: CollectionConstraints, sun_ecef_km: Sequence[float],
    gimbal_off_nadir_rad: float, required_slew_rate_rad_s: float, spacecraft_illumination_fraction: float,
) -> dict[str, Any]:
    surface_constraints = constraints.surface
    if not surface_constraints.illumination_enabled:
        surface_constraints = SurfaceServiceConstraints(
            maximum_target_off_nadir_rad=surface_constraints.maximum_target_off_nadir_rad,
            maximum_incidence_rad=surface_constraints.maximum_incidence_rad,
            minimum_sun_elevation_rad=-0.5 * math.pi,
            maximum_sun_elevation_rad=0.5 * math.pi,
        )
    geometry = evaluate_rich_surface_targets_ecef(
        observer_ecef_km=np.asarray(observer_ecef_km, dtype=float),
        target_ecef_km=target.ecef_km[None, :],
        target_outward_normal_ecef=target.outward_normal_ecef[None, :],
        dcm_sensor_from_ecef=np.asarray(dcm_sensor_from_ecef, dtype=float),
        pattern=payload.pattern,
        constraints=surface_constraints,
        max_range_km=constraints.maximum_range_km,
        sun_ecef_km=np.asarray(sun_ecef_km, dtype=float),
    )
    quality = optical_quality_metrics(
        slant_range_km=float(geometry.range_km[0]),
        incidence_angle_rad=float(geometry.incidence_angle_rad[0]),
        payload=payload,
    )
    base_reason = PRIMARY_REASON_NAMES[int(geometry.primary_reason_code[0])]
    reason = base_reason
    if reason == "available" and gimbal_off_nadir_rad > payload.maximum_gimbal_off_nadir_rad + 1.0e-12:
        reason = "gimbal_limit_exceeded"
    if reason == "available" and required_slew_rate_rad_s > payload.maximum_slew_rate_rad_s + 1.0e-12:
        reason = "slew_rate_exceeded"
    if (
        reason == "available"
        and spacecraft_illumination_fraction + 1.0e-12 < constraints.minimum_spacecraft_illumination_fraction
    ):
        reason = "spacecraft_eclipsed"
    if (
        reason == "available"
        and constraints.maximum_effective_resolution_m is not None
        and quality["effective_resolution_m"] > constraints.maximum_effective_resolution_m + 1.0e-12
    ):
        reason = "resolution_exceeded"
    return {
        "available": reason == "available",
        "reason": reason,
        "range_km": float(geometry.range_km[0]),
        "sensor_horizontal_angle_deg": math.degrees(float(geometry.sensor_horizontal_angle_rad[0])),
        "sensor_vertical_angle_deg": math.degrees(float(geometry.sensor_vertical_angle_rad[0])),
        "sensor_off_axis_angle_deg": math.degrees(float(geometry.sensor_off_axis_angle_rad[0])),
        "off_nadir_angle_deg": math.degrees(float(geometry.target_off_nadir_angle_rad[0])),
        "incidence_angle_deg": math.degrees(float(geometry.incidence_angle_rad[0])),
        "sun_elevation_deg": math.degrees(float(geometry.sun_elevation_rad[0])),
        "gimbal_off_nadir_deg": math.degrees(float(gimbal_off_nadir_rad)),
        "required_slew_rate_deg_s": math.degrees(float(required_slew_rate_rad_s)),
        "spacecraft_illumination_fraction": float(spacecraft_illumination_fraction),
        **quality,
    }


__all__ = [
    "COLLECTION_REASON_NAMES", "OPTICAL_COLLECTION_MODEL", "CollectionConstraints", "GroundTarget",
    "OpticalPayload", "evaluate_collection_sample", "footprint_boundary_evidence", "local_nadir_frame_sensor_from_eci",
    "optical_quality_metrics", "sensor_frame_and_gimbal_vector",
]
