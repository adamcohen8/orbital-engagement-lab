from __future__ import annotations

from sim.flight_software import (
    AerodynamicEffectorPositionCommand,
    ControlAxisSample,
    FlightSoftwareInputBatch,
    GamePilotMode,
    GamePilotReferenceFlightSoftwareStack,
    IdealWrenchCommand,
    InputEvent,
    InputKind,
    PacketId,
    PilotInputPayload,
    Quality,
)
from sim.tests.fsw_v2_helpers import BOOT_ID, SATELLITE_ID, boot_event, clock, ideal_event
from sim.tests.game_fsw_v2_helpers import game_stack_config


def _pilot_event(sequence: int, tick: int, *axes: tuple[str, float], pressed: tuple[str, ...] = ()) -> InputEvent:
    time = clock(tick)
    payload = PilotInputPayload(
        "game-profile-v1",
        tuple(ControlAxisSample(name, value) for name, value in axes),
        pressed,
    )
    return InputEvent(PacketId("pilot", BOOT_ID, sequence), InputKind.PILOT_INPUT, time, time, Quality(), payload)


def _batch(invocation: int, event: InputEvent) -> FlightSoftwareInputBatch:
    return FlightSoftwareInputBatch(SATELLITE_ID, invocation, clock(invocation), (ideal_event(invocation, invocation), event))


def test_translation_profile_converts_typed_ric_axis_to_actuator_command() -> None:
    stack = GamePilotReferenceFlightSoftwareStack(game_stack_config(GamePilotMode.TRANSLATION))
    stack.boot(boot_event())
    output = stack.step(_batch(1, _pilot_event(0, 1, ("translate_r", 1.0), ("throttle", 1.0))))
    assert len(output.commands) == 1
    assert isinstance(output.commands[0].payload, IdealWrenchCommand)
    assert output.commands[0].payload.force_n[0] > 0.0


def test_direct_eci_profile_converts_typed_axis_to_inertial_actuator_command() -> None:
    stack = GamePilotReferenceFlightSoftwareStack(game_stack_config(GamePilotMode.DIRECT_ECI))
    stack.boot(boot_event())
    output = stack.step(_batch(1, _pilot_event(0, 1, ("translate_i", 1.0), ("throttle", 1.0))))
    assert len(output.commands) == 1
    assert isinstance(output.commands[0].payload, IdealWrenchCommand)
    assert output.commands[0].payload.force_n[1] > 0.0


def test_attitude_thrust_profile_emits_attitude_then_translation_commands() -> None:
    stack = GamePilotReferenceFlightSoftwareStack(game_stack_config(GamePilotMode.ATTITUDE_THRUST))
    stack.boot(boot_event())
    output = stack.step(
        _batch(1, _pilot_event(0, 1, ("pitch", 0.5), ("throttle", 1.0), pressed=("fire",)))
    )
    assert {command.actuator_id for command in output.commands} == {"attitude", "translation"}


def test_aerodynamic_profile_commands_devices_not_physics_effects() -> None:
    stack = GamePilotReferenceFlightSoftwareStack(game_stack_config(GamePilotMode.AERODYNAMIC))
    stack.boot(boot_event())
    output = stack.step(_batch(1, _pilot_event(0, 1, ("deployment", 1.0), ("bank", -0.5))))
    assert {command.actuator_id for command in output.commands} == {"deployment", "bank"}
    assert all(isinstance(command.payload, AerodynamicEffectorPositionCommand) for command in output.commands)


def test_live_navigation_fast_path_preserves_complete_stack_outputs() -> None:
    config = game_stack_config(GamePilotMode.TRANSLATION)
    optimized = GamePilotReferenceFlightSoftwareStack(config)
    audit_path = GamePilotReferenceFlightSoftwareStack(config, _live_navigation_fast_path=False)
    optimized.boot(boot_event())
    audit_path.boot(boot_event())

    for invocation in range(1, 201):
        batch = _batch(
            invocation,
            _pilot_event(
                invocation - 1,
                invocation,
                ("translate_r", 0.25),
                ("translate_i", -0.5),
                ("throttle", 1.0),
            ),
        )
        assert optimized.step(batch) == audit_path.step(batch)

    assert len(optimized._navigator._own_packets) == 1
    assert len(audit_path._navigator._own_packets) == 200
