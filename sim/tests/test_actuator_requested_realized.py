from __future__ import annotations

from math import exp

import pytest

from sim.actuators.command_bus import ActuatorCommandBus, ActuatorDeviceDefinition, ExpiryBehavior
from sim.actuators.physical import IdealWrenchHardware
from sim.flight_software import (
    ActuatorCommand,
    ClockScale,
    ClockTag,
    FrameId,
    IdealWrenchCommand,
    PacketId,
    ValidityInterval,
)


def _time(ticks: int) -> ClockTag:
    return ClockTag("clock", ticks, 1_000_000_000, ClockScale.ONBOARD)


def test_command_acceptance_is_instantaneous_but_realization_evolves_only_during_physics() -> None:
    frame = FrameId("OEL/ACTUATOR/sat/wrench", "v1")
    bus = ActuatorCommandBus(
        (ActuatorDeviceDefinition("sat", "wrench", frame, (IdealWrenchCommand,), ExpiryBehavior.ZERO),)
    )
    hardware = IdealWrenchHardware("wrench", max_force_n=2.0, response_time_constant_s=1.0)
    command = ActuatorCommand(
        PacketId("fsw", "boot", 1),
        "sat",
        "wrench",
        _time(0),
        ValidityInterval(_time(0), _time(3)),
        frame,
        IdealWrenchCommand((4.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    bus.publish(command, received_at=_time(0))
    assert hardware.realized_force_n == (0.0, 0.0, 0.0)

    demand = bus.demand(satellite_id="sat", actuator_id="wrench", at=_time(0))
    realization = hardware.advance(demand, start_time_ns=0, end_time_ns=1_000_000_000)
    assert realization.requested_force_n == (4.0, 0.0, 0.0)
    assert realization.realized_force_n[0] == pytest.approx(2.0 * (1.0 - exp(-1.0)), rel=1e-8)
    assert realization.realized_force_n != realization.requested_force_n
    assert realization.saturated is True
    assert realization.source_command_id == command.command_id


def test_zero_expiry_changes_physical_demand_without_synthesizing_a_command() -> None:
    frame = FrameId("OEL/ACTUATOR/sat/wrench", "v1")
    bus = ActuatorCommandBus(
        (ActuatorDeviceDefinition("sat", "wrench", frame, (IdealWrenchCommand,), ExpiryBehavior.ZERO),)
    )
    command = ActuatorCommand(
        PacketId("fsw", "boot", 1),
        "sat",
        "wrench",
        _time(0),
        ValidityInterval(_time(0), _time(1)),
        frame,
        IdealWrenchCommand((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    bus.publish(command, received_at=_time(0))
    demand = bus.demand(satellite_id="sat", actuator_id="wrench", at=_time(1))
    assert demand.payload is None
    assert demand.source_command is command
