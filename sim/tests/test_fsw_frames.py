from __future__ import annotations

import pytest

from sim.flight_software.contracts import FrameId
from sim.flight_software.frames import FrameDefinition, FrameRegistry, build_satellite_frame_registry


def test_required_satellite_frames_are_registered() -> None:
    registry = build_satellite_frame_registry(
        version="frames-v1",
        inertial_model="J2000",
        earth_fixed_model="ITRF",
        satellite_id="chaser",
        sensor_ids=("gyro",),
        actuator_ids=("wheel",),
    )
    expected = {
        "OEL/ECI/J2000",
        "OEL/ECEF/ITRF",
        "OEL/RIC/chaser",
        "OEL/LVLH/chaser",
        "OEL/BODY/chaser",
        "OEL/SENSOR/chaser/gyro",
        "OEL/ACTUATOR/chaser/wheel",
    }
    assert {definition.frame_id.name for definition in registry.definitions()} == expected


def test_frame_registry_rejects_unknown_version_and_missing_parent() -> None:
    registry = FrameRegistry("v1")
    root = FrameId("root", "v1")
    registry.register(FrameDefinition(root, None))
    with pytest.raises(ValueError, match="parent frame"):
        registry.register(FrameDefinition(FrameId("child", "v1"), FrameId("missing", "v1"), transform_model_id="model"))
    with pytest.raises(KeyError, match="not active"):
        registry.resolve(FrameId("root", "v2"))


def test_frame_declarations_are_right_handed_parent_to_child_and_normalized() -> None:
    root = FrameId("root", "v1")
    child = FrameId("child", "v1")
    with pytest.raises(ValueError, match="right-handed"):
        FrameDefinition(child, root, handedness="left", transform_model_id="model")
    with pytest.raises(ValueError, match="parent_to_child"):
        FrameDefinition(child, root, transform_direction="child_to_parent", transform_model_id="model")
    with pytest.raises(ValueError, match="normalized"):
        FrameDefinition(child, root, static_quat_child_from_parent=(2.0, 0.0, 0.0, 0.0))
