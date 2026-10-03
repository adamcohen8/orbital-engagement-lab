from __future__ import annotations

from dataclasses import replace

from sim.actuators.command_bus import ActuatorCommandBus, ActuatorDeviceDefinition, DemandMode, ExpiryBehavior
from sim.flight_software import (
    ActuatorCommand,
    ClockScale,
    ClockTag,
    CommandDisposition,
    FrameId,
    IdealWrenchCommand,
    PacketId,
    ThrusterPulseCommand,
    TimeValidity,
    ValidityInterval,
)


def _time(ticks: int, clock_id: str = "clock") -> ClockTag:
    return ClockTag(clock_id, ticks, 1_000_000_000, ClockScale.ONBOARD)


def _command(actuator_id: str, sequence: int) -> ActuatorCommand:
    return ActuatorCommand(
        PacketId("fsw", "boot", sequence),
        "sat",
        actuator_id,
        _time(0),
        ValidityInterval(_time(0), _time(2)),
        FrameId(f"OEL/ACTUATOR/sat/{actuator_id}", "v1"),
        IdealWrenchCommand((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )


def test_expiry_behavior_is_declared_per_device_without_fallback_commands() -> None:
    devices = tuple(
        ActuatorDeviceDefinition(
            "sat",
            name,
            FrameId(f"OEL/ACTUATOR/sat/{name}", "v1"),
            (IdealWrenchCommand,),
            behavior,
        )
        for name, behavior in (
            ("latch", ExpiryBehavior.LATCH),
            ("zero", ExpiryBehavior.ZERO),
            ("idle", ExpiryBehavior.IDLE),
        )
    )
    bus = ActuatorCommandBus(devices)
    for sequence, name in enumerate(("latch", "zero", "idle")):
        bus.publish(_command(name, sequence), received_at=_time(0))
    assert bus.demand(satellite_id="sat", actuator_id="latch", at=_time(2)).mode is DemandMode.LATCHED
    zero = bus.demand(satellite_id="sat", actuator_id="zero", at=_time(2))
    idle = bus.demand(satellite_id="sat", actuator_id="idle", at=_time(2))
    assert zero.mode is DemandMode.ZERO and zero.payload is None
    assert idle.mode is DemandMode.IDLE and idle.payload is None
    assert bus.hard_event_times_ns() == (0, 2_000_000_000)


def test_expired_and_incomparable_commands_are_rejected_without_state_change() -> None:
    device = ActuatorDeviceDefinition(
        "sat", "zero", FrameId("OEL/ACTUATOR/sat/zero", "v1"), (IdealWrenchCommand,), ExpiryBehavior.ZERO
    )
    bus = ActuatorCommandBus((device,))
    expired = bus.publish(_command("zero", 1), received_at=_time(2))
    assert expired is not None and expired.disposition is CommandDisposition.REJECTED_TIME
    wrong_clock = bus.publish(_command("zero", 2), received_at=_time(0, "other"))
    assert wrong_clock is not None and wrong_clock.disposition is CommandDisposition.REJECTED_TIME
    assert bus.demand(satellite_id="sat", actuator_id="zero", at=_time(3)).mode is DemandMode.UNCOMMANDED


def test_thruster_pulse_start_and_stop_are_hard_device_events() -> None:
    frame = FrameId("OEL/ACTUATOR/sat/rcs", "v1")
    bus = ActuatorCommandBus(
        (ActuatorDeviceDefinition("sat", "jet-1", frame, (ThrusterPulseCommand,), ExpiryBehavior.IDLE),)
    )
    command = ActuatorCommand(
        PacketId("fsw", "boot", 1),
        "sat",
        "jet-1",
        _time(0),
        ValidityInterval(_time(0), _time(10)),
        frame,
        ThrusterPulseCommand("jet-1", _time(3), 2.0),
    )
    bus.publish(command, received_at=_time(0))
    assert bus.demand(satellite_id="sat", actuator_id="jet-1", at=_time(2)).mode is DemandMode.UNCOMMANDED
    assert bus.demand(satellite_id="sat", actuator_id="jet-1", at=_time(3)).mode is DemandMode.COMMANDED
    assert bus.demand(satellite_id="sat", actuator_id="jet-1", at=_time(5)).mode is DemandMode.IDLE
    assert bus.hard_event_times_ns() == (3_000_000_000, 5_000_000_000)


def test_latched_device_does_not_keep_a_thruster_pulse_enabled_after_its_end() -> None:
    frame = FrameId("OEL/ACTUATOR/sat/jet-1", "v1")
    bus = ActuatorCommandBus(
        (ActuatorDeviceDefinition("sat", "jet-1", frame, (ThrusterPulseCommand,), ExpiryBehavior.LATCH),)
    )
    command = ActuatorCommand(
        PacketId("fsw", "boot", 1),
        "sat",
        "jet-1",
        _time(0),
        ValidityInterval(_time(0), _time(10)),
        frame,
        ThrusterPulseCommand("jet-1", _time(0), 2.0),
    )
    bus.publish(command, received_at=_time(0))
    assert bus.demand(satellite_id="sat", actuator_id="jet-1", at=_time(1)).mode is DemandMode.COMMANDED
    expired = bus.demand(satellite_id="sat", actuator_id="jet-1", at=_time(2))
    assert expired.mode is DemandMode.ZERO and expired.payload is None


def test_invalid_time_validity_is_rejected_for_commands_and_pulses() -> None:
    device = ActuatorDeviceDefinition(
        "sat", "zero", FrameId("OEL/ACTUATOR/sat/zero", "v1"), (IdealWrenchCommand,), ExpiryBehavior.ZERO
    )
    bus = ActuatorCommandBus((device,))
    base = _command("zero", 1)
    invalid_commands = (
        replace(base, issued_at=replace(base.issued_at, validity=TimeValidity.INVALID)),
        replace(
            base,
            validity=ValidityInterval(
                replace(base.validity.not_before, validity=TimeValidity.INVALID),
                base.validity.expires_at,
            ),
        ),
        replace(
            base,
            validity=ValidityInterval(
                base.validity.not_before,
                replace(base.validity.expires_at, validity=TimeValidity.INVALID),
            ),
        ),
    )
    for sequence, command in enumerate(invalid_commands, start=1):
        receipt = bus.publish(replace(command, command_id=PacketId("fsw", "invalid", sequence)), received_at=_time(1))
        assert receipt is not None and receipt.disposition is CommandDisposition.REJECTED_TIME

    pulse_frame = FrameId("OEL/ACTUATOR/sat/jet-1", "v1")
    pulse_bus = ActuatorCommandBus(
        (ActuatorDeviceDefinition("sat", "jet-1", pulse_frame, (ThrusterPulseCommand,), ExpiryBehavior.ZERO),)
    )
    pulse = ActuatorCommand(
        PacketId("fsw", "pulse-invalid", 1),
        "sat",
        "jet-1",
        _time(0),
        ValidityInterval(_time(0), _time(10)),
        pulse_frame,
        ThrusterPulseCommand("jet-1", replace(_time(1), validity=TimeValidity.INVALID), 1.0),
    )
    receipt = pulse_bus.publish(pulse, received_at=_time(0))
    assert receipt is not None and receipt.disposition is CommandDisposition.REJECTED_TIME
