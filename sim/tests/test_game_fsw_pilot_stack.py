from __future__ import annotations

import numpy as np

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
    canonical_json_bytes,
)
from sim.flight_software.game_stacks import _cross3, _ric_to_eci
from sim.gnc.attitude_v2 import AttitudeSolution
from sim.gnc.contracts import BeliefState
from sim.gnc.navigation_v2 import OrbitNavigationSolution
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


def test_live_command_fast_path_preserves_all_control_outputs() -> None:
    for mode in (GamePilotMode.TRANSLATION, GamePilotMode.DIRECT_ECI, GamePilotMode.ATTITUDE_THRUST):
        config = game_stack_config(mode)
        optimized = GamePilotReferenceFlightSoftwareStack(config)
        audit_path = GamePilotReferenceFlightSoftwareStack(config, _live_command_fast_path=False)
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
                    ("pitch", 0.2),
                    ("throttle", 1.0),
                    pressed=("fire",),
                ),
            )
            optimized_output = optimized.step(batch)
            audit_output = audit_path.step(batch)
            assert canonical_json_bytes(optimized_output) == canonical_json_bytes(audit_output)


def test_game_ric_cross_product_fast_path_is_bit_exact() -> None:
    rng = np.random.default_rng(8341)
    for _ in range(1_000):
        first = rng.normal(size=3)
        second = rng.normal(size=3)
        np.testing.assert_array_equal(_cross3(first, second), np.cross(first, second))


def test_game_ric_transform_fast_path_is_bit_exact() -> None:
    config = game_stack_config(GamePilotMode.TRANSLATION)
    generated_at = clock(1)
    attitude = AttitudeSolution(
        generated_at,
        config.body_frame,
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        None,
        None,
        None,
        None,
        (),
        BeliefState(generated_at),
    )
    solution = OrbitNavigationSolution(
        generated_at,
        config.inertial_frame,
        config.relative_frame,
        (7_000_000.0, 1_000.0, 2_000.0),
        (-10.0, 7_500.0, 20.0),
        100.0,
        attitude,
        (),
        (),
        BeliefState(generated_at),
    )
    vector = np.array([0.1, -0.2, 0.3])
    state = np.asarray((*solution.position_eci_m, *solution.velocity_eci_m_s))
    radial = state[:3] / np.linalg.norm(state[:3])
    cross_track = np.cross(state[:3], state[3:6])
    cross_track /= np.linalg.norm(cross_track)
    in_track = np.cross(cross_track, radial)
    in_track /= np.linalg.norm(in_track)
    legacy = np.column_stack((radial, in_track, cross_track)) @ vector

    np.testing.assert_array_equal(_ric_to_eci(vector, solution), legacy)
