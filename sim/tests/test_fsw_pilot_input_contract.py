from __future__ import annotations

import pytest

from sim.flight_software import (
    ClockScale,
    ClockTag,
    ControlAxisSample,
    InputEvent,
    InputKind,
    PacketId,
    PilotInputPayload,
    Quality,
    canonical_json_bytes,
    canonical_loads,
)


def test_pilot_input_is_dimensionless_sequenced_input_not_physics_command() -> None:
    time = ClockTag("pilot-clock", 4, 1_000_000, ClockScale.ONBOARD)
    payload = PilotInputPayload(
        "oel.rpo_pilot.v1",
        axes=(ControlAxisSample("translation_x", 0.5), ControlAxisSample("yaw", -0.25)),
        pressed_actions=("enable_translation",),
    )
    event = InputEvent(PacketId("pilot", "boot", 9), InputKind.PILOT_INPUT, time, time, Quality(), payload)
    restored = canonical_loads(canonical_json_bytes(event))
    assert restored == event
    assert not hasattr(payload, "force_n")
    assert not hasattr(payload, "desired_acceleration")


def test_pilot_input_rejects_out_of_range_duplicate_or_mismatched_values() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        ControlAxisSample("translation_x", 1.01)
    axis = ControlAxisSample("translation_x", 0.5)
    with pytest.raises(ValueError, match="unique"):
        PilotInputPayload("profile", axes=(axis, axis))
    with pytest.raises(ValueError, match="duplicates"):
        PilotInputPayload("profile", pressed_actions=("fire", "fire"))
    time = ClockTag("pilot-clock", 4, 1_000_000, ClockScale.ONBOARD)
    with pytest.raises(TypeError, match="payload"):
        InputEvent(PacketId("pilot", "boot", 9), InputKind.PILOT_INPUT, time, time, Quality(), object())
