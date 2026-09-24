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


def test_mounted_rcs_realizes_moment_and_scales_it_with_available_thrust() -> None:
    from sim.actuators.command_bus import ActuatorDemand, DemandMode
    from sim.actuators.physical import RcsThrusterHardware
    from sim.flight_software import ThrusterOnOffCommand
    from sim.runtime.satellites.flight_software_runtime import _scale_translational_realizations

    device = RcsThrusterHardware(
        "jet", direction_body=(1.0, 0.0, 0.0), max_thrust_n=2.0,
        position_body_m=(0.0, 0.8, 0.0), specific_impulse_s=220.0,
    )
    demand = ActuatorDemand("jet", DemandMode.COMMANDED, None, ThrusterOnOffCommand("jet", True))
    realization = device.advance(demand, start_time_ns=0, end_time_ns=1_000_000_000)
    assert realization.realized_force_n == pytest.approx((2.0, 0.0, 0.0))
    assert realization.realized_torque_n_m == pytest.approx((0.0, 0.0, -1.6))
    limited = _scale_translational_realizations([(device, realization)], 0.25, propellant_only=True)[0][1]
    assert limited.realized_force_n == pytest.approx((0.5, 0.0, 0.0))
    assert limited.realized_torque_n_m == pytest.approx((0.0, 0.0, -0.4))
    assert limited.mass_flow_kg_s == pytest.approx(realization.mass_flow_kg_s * 0.25)
    centered = RcsThrusterHardware("jet", direction_body=(1.0, 0.0, 0.0), max_thrust_n=2.0)
    assert centered.advance(demand, start_time_ns=0, end_time_ns=1_000_000_000).realized_torque_n_m == (0.0, 0.0, 0.0)


def test_wheel_rotor_speed_limits_momentum_and_reports_speed() -> None:
    from sim.actuators.command_bus import ActuatorDemand, DemandMode
    from sim.actuators.physical import ReactionWheelHardware
    from sim.flight_software import ReactionWheelTorqueCommand

    wheel = ReactionWheelHardware(
        "wheel", axes_body=((1.0, 0.0, 0.0),), max_torque_n_m=(1.0,),
        max_momentum_n_m_s=(10.0,), inertia_kg_m2=(0.1,), max_speed_rad_s=(2.0,),
    )
    demand = ActuatorDemand("wheel", DemandMode.COMMANDED, None, ReactionWheelTorqueCommand((1.0,)))
    realization = wheel.advance(demand, start_time_ns=0, end_time_ns=1_000_000_000)
    assert realization.realized_torque_n_m == pytest.approx((-0.2, 0.0, 0.0))
    assert wheel.momentum_n_m_s[0] == pytest.approx(0.2)
    assert realization.saturated
    with pytest.raises(ValueError, match="together"):
        ReactionWheelHardware("wheel", axes_body=((1.0, 0.0, 0.0),), max_torque_n_m=(1.0,),
                              max_momentum_n_m_s=(10.0,), inertia_kg_m2=(0.1,))
