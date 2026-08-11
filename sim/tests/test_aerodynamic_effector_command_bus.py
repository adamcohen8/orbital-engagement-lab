from __future__ import annotations

from sim.actuators.command_bus import ActuatorCommandBus, ActuatorDeviceDefinition, DemandMode, ExpiryBehavior
from sim.actuators.physical import AerodynamicEffectorHardware
from sim.flight_software import (
    ActuatorCommand,
    AerodynamicEffectorPositionCommand,
    ClockScale,
    ClockTag,
    CommandDisposition,
    FrameId,
    PacketId,
    ValidityInterval,
)


def _time(ticks: int) -> ClockTag:
    return ClockTag("clock", ticks, 1_000_000_000, ClockScale.ONBOARD)


def _command(sequence: int, position: float, *, coordinate: str = "flap", unit: str = "rad") -> ActuatorCommand:
    frame = FrameId("OEL/ACTUATOR/sat/flap", "v1")
    return ActuatorCommand(
        PacketId("fsw", "boot", sequence),
        "sat",
        "flap",
        _time(0),
        ValidityInterval(_time(0), _time(3)),
        frame,
        AerodynamicEffectorPositionCommand(coordinate, position, unit),  # type: ignore[arg-type]
    )


def test_effector_bus_validates_coordinate_unit_and_physical_range() -> None:
    hardware = AerodynamicEffectorHardware(
        "flap", "flap", unit="rad", minimum=-0.5, maximum=0.5, neutral=0.0, rate_limit_per_s=0.1
    )
    device = ActuatorDeviceDefinition(
        "sat",
        "flap",
        FrameId("OEL/ACTUATOR/sat/flap", "v1"),
        (AerodynamicEffectorPositionCommand,),
        ExpiryBehavior.ZERO,
        validator=hardware.validate,
    )
    bus = ActuatorCommandBus((device,))
    assert bus.publish(_command(1, 0.4), received_at=_time(0)).disposition is CommandDisposition.ACCEPTED  # type: ignore[union-attr]
    assert bus.publish(_command(2, 0.6), received_at=_time(0)).disposition is CommandDisposition.REJECTED_VALUE  # type: ignore[union-attr]
    assert (
        bus.publish(_command(3, 0.2, coordinate="other"), received_at=_time(0)).disposition
        is CommandDisposition.REJECTED_VALUE
    )  # type: ignore[union-attr]
    assert (
        bus.publish(_command(4, 0.2, unit="m"), received_at=_time(0)).disposition is CommandDisposition.REJECTED_VALUE
    )  # type: ignore[union-attr]


def test_effector_state_rate_limits_in_physics_and_returns_to_neutral_on_zero_expiry() -> None:
    hardware = AerodynamicEffectorHardware(
        "flap", "flap", unit="rad", minimum=-0.5, maximum=0.5, neutral=0.0, rate_limit_per_s=0.1
    )
    device = ActuatorDeviceDefinition(
        "sat",
        "flap",
        FrameId("OEL/ACTUATOR/sat/flap", "v1"),
        (AerodynamicEffectorPositionCommand,),
        ExpiryBehavior.ZERO,
        validator=hardware.validate,
    )
    bus = ActuatorCommandBus((device,))
    bus.publish(_command(1, 0.4), received_at=_time(0))
    commanded = bus.demand(satellite_id="sat", actuator_id="flap", at=_time(0))
    first = hardware.advance(commanded, start_time_ns=0, end_time_ns=1_000_000_000)
    assert first.device_state[0].value == 0.1
    assert first.saturated is True

    expired = bus.demand(satellite_id="sat", actuator_id="flap", at=_time(3))
    assert expired.mode is DemandMode.ZERO and expired.payload is None
    second = hardware.advance(expired, start_time_ns=3_000_000_000, end_time_ns=4_000_000_000)
    assert second.device_state[0].value == 0.0
