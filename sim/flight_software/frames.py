"""Versioned frame registry for flight-software boundary values."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from .contracts import FrameId, Quaternion


@dataclass(frozen=True, slots=True)
class FrameDefinition:
    frame_id: FrameId
    parent_frame_id: FrameId | None
    handedness: str = "right"
    transform_direction: str = "parent_to_child"
    static_quat_child_from_parent: Quaternion | None = None
    transform_model_id: str | None = None

    def __post_init__(self) -> None:
        if self.handedness != "right":
            raise ValueError("v1 boundary frames must be right-handed")
        if self.transform_direction != "parent_to_child":
            raise ValueError("v1 frame transforms must use parent_to_child semantics")
        if self.parent_frame_id is None:
            if self.static_quat_child_from_parent is not None or self.transform_model_id is not None:
                raise ValueError("root frames cannot declare a parent transform")
            return
        if self.parent_frame_id.registry_version != self.frame_id.registry_version:
            raise ValueError("parent and child frames must use the same registry version")
        declared = int(self.static_quat_child_from_parent is not None) + int(self.transform_model_id is not None)
        if declared != 1:
            raise ValueError("custom frame must declare exactly one static transform or transform model")
        if self.static_quat_child_from_parent is not None:
            values = tuple(float(value) for value in self.static_quat_child_from_parent)
            if len(values) != 4 or not all(isfinite(value) for value in values):
                raise ValueError("static frame quaternion must contain four finite values")
            if abs(sqrt(sum(value * value for value in values)) - 1.0) > 1.0e-10:
                raise ValueError("static frame quaternion must be normalized within 1e-10")
        if self.transform_model_id is not None and not self.transform_model_id.strip():
            raise ValueError("transform_model_id must be non-empty")


class FrameRegistry:
    def __init__(self, version: str) -> None:
        if not str(version).strip():
            raise ValueError("frame registry version must be non-empty")
        self._version = str(version)
        self._definitions: dict[str, FrameDefinition] = {}

    @property
    def version(self) -> str:
        return self._version

    def register(self, definition: FrameDefinition) -> None:
        if definition.frame_id.registry_version != self._version:
            raise ValueError("frame definition registry version does not match registry")
        name = definition.frame_id.name
        if name in self._definitions:
            raise ValueError(f"frame {name!r} is already registered")
        if definition.parent_frame_id is not None and definition.parent_frame_id.name not in self._definitions:
            raise ValueError(f"parent frame {definition.parent_frame_id.name!r} must be registered first")
        self._definitions[name] = definition

    def resolve(self, frame_id: FrameId) -> FrameDefinition:
        if frame_id.registry_version != self._version:
            raise KeyError(f"frame registry version {frame_id.registry_version!r} is not active")
        try:
            return self._definitions[frame_id.name]
        except KeyError as exc:
            raise KeyError(f"unknown frame {frame_id.name!r}") from exc

    def contains(self, frame_id: FrameId) -> bool:
        try:
            self.resolve(frame_id)
        except KeyError:
            return False
        return True

    def definitions(self) -> tuple[FrameDefinition, ...]:
        return tuple(self._definitions.values())


def build_satellite_frame_registry(
    *,
    version: str,
    inertial_model: str,
    earth_fixed_model: str,
    satellite_id: str,
    sensor_ids: tuple[str, ...] = (),
    actuator_ids: tuple[str, ...] = (),
) -> FrameRegistry:
    """Build the required inertial/Earth/body/device frame names for one satellite."""

    for name, value in (
        ("inertial_model", inertial_model),
        ("earth_fixed_model", earth_fixed_model),
        ("satellite_id", satellite_id),
    ):
        if not str(value).strip():
            raise ValueError(f"{name} must be non-empty")
    if any(not str(value).strip() for value in sensor_ids):
        raise ValueError("sensor_ids must contain only non-empty values")
    if any(not str(value).strip() for value in actuator_ids):
        raise ValueError("actuator_ids must contain only non-empty values")
    if len(sensor_ids) != len(set(sensor_ids)) or len(actuator_ids) != len(set(actuator_ids)):
        raise ValueError("sensor and actuator identifiers must be unique within their namespaces")
    registry = FrameRegistry(version)
    eci = FrameId(f"OEL/ECI/{inertial_model}", version)
    ecef = FrameId(f"OEL/ECEF/{earth_fixed_model}", version)
    ric = FrameId(f"OEL/RIC/{satellite_id}", version)
    lvlh = FrameId(f"OEL/LVLH/{satellite_id}", version)
    body = FrameId(f"OEL/BODY/{satellite_id}", version)
    registry.register(FrameDefinition(frame_id=eci, parent_frame_id=None))
    registry.register(
        FrameDefinition(frame_id=ecef, parent_frame_id=eci, transform_model_id=f"earth_rotation/{earth_fixed_model}")
    )
    registry.register(FrameDefinition(frame_id=ric, parent_frame_id=eci, transform_model_id="relative_orbit_ric"))
    registry.register(
        FrameDefinition(frame_id=lvlh, parent_frame_id=eci, transform_model_id="local_vertical_local_horizontal")
    )
    registry.register(FrameDefinition(frame_id=body, parent_frame_id=eci, transform_model_id="satellite_attitude"))
    identity = (1.0, 0.0, 0.0, 0.0)
    for sensor_id in sensor_ids:
        frame = FrameId(f"OEL/SENSOR/{satellite_id}/{sensor_id}", version)
        registry.register(FrameDefinition(frame_id=frame, parent_frame_id=body, static_quat_child_from_parent=identity))
    for actuator_id in actuator_ids:
        frame = FrameId(f"OEL/ACTUATOR/{satellite_id}/{actuator_id}", version)
        registry.register(FrameDefinition(frame_id=frame, parent_frame_id=body, static_quat_child_from_parent=identity))
    return registry
