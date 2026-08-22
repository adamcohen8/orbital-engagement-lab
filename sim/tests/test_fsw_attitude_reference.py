from __future__ import annotations

from dataclasses import replace
from math import cos, sin

import numpy as np
import pytest

import sim.gnc.attitude_v2 as attitude_v2
from sim.actuators.command_bus import ActuatorCommandBus, ActuatorDeviceDefinition, ExpiryBehavior
from sim.dynamics.coupled_satellite import (
    CoupledIntegratorConfig,
    CoupledSatelliteDynamics,
    CoupledSatelliteIntegrator,
    CoupledSatelliteState,
    StageEffects,
    constant_mass_properties,
)
from sim.flight_software import (
    AttitudeReferenceFlightSoftwareStack,
    CmgGimbalRateCommand,
    CommandDisposition,
    GroundCommandKind,
    GroundCommandPayload,
    IdealWrenchCommand,
    InputEvent,
    InputKind,
    MagnetometerMeasurement,
    MagnetorquerDipoleCommand,
    MeasurementEvent,
    ModeledFaultIndicationPayload,
    PacketId,
    Quality,
    ReactionWheelTorqueCommand,
    SunVectorMeasurement,
    TelemetryField,
    canonical_json_bytes,
)
from sim.gnc.attitude_v2 import (
    AttitudeAllocatorKind,
    AttitudeNavigator,
    AttitudeReferenceConfig,
    AttitudeReferenceGenerator,
    AttitudeReferenceMode,
    SmallAngleLqrTorqueController,
)
from sim.gnc.operations_v2 import HealthManagerConfig, MomentumUnloadConfig
from sim.tests.fsw_v2_helpers import (
    ACTUATOR_FRAME,
    BODY_FRAME,
    BOOT_ID,
    INERTIAL_FRAME,
    SATELLITE_ID,
    attitude_config,
    batch,
    boot_event,
    clock,
    ideal_event,
)
from sim.utils.quaternion import quaternion_to_dcm_bn


def _measurement_event(sequence: int, tick: int, sensor_id: str, payload: object) -> InputEvent:
    time = clock(tick)
    measurement = MeasurementEvent(sensor_id, payload.schema, time, BODY_FRAME, payload)  # type: ignore[attr-defined,arg-type]
    return InputEvent(
        PacketId(sensor_id, BOOT_ID, sequence),
        InputKind.MEASUREMENT,
        time,
        time,
        Quality(),
        measurement,
    )


def _fault_event(sequence: int, tick: int, component_id: str, active: bool) -> InputEvent:
    time = clock(tick)
    return InputEvent(
        PacketId("fault-monitor", BOOT_ID, sequence),
        InputKind.MODELED_FAULT_INDICATION,
        time,
        time,
        Quality(),
        ModeledFaultIndicationPayload(component_id, "unavailable", active, "fault-monitor"),
    )


@pytest.mark.parametrize(
    ("config", "extra_event"),
    (
        (AttitudeReferenceConfig(AttitudeReferenceMode.QUATERNION), None),
        (AttitudeReferenceConfig(AttitudeReferenceMode.NADIR), None),
        (AttitudeReferenceConfig(AttitudeReferenceMode.VELOCITY), None),
        (
            AttitudeReferenceConfig(AttitudeReferenceMode.SUN),
            SunVectorMeasurement((0.0, 1.0, 0.0)),
        ),
        (
            AttitudeReferenceConfig(AttitudeReferenceMode.TARGET, target_position_eci_m=(7_000_000.0, 1_000.0, 0.0)),
            None,
        ),
        (AttitudeReferenceConfig(AttitudeReferenceMode.RIC, ric_axis="in_track"), None),
        (
            AttitudeReferenceConfig(AttitudeReferenceMode.THRUST, thrust_direction_eci=(0.0, 0.0, 1.0)),
            None,
        ),
    ),
)
def test_reference_modes_produce_typed_actuator_commands(
    config: AttitudeReferenceConfig, extra_event: object | None
) -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config(reference=config))
    stack.boot(boot_event())
    events = [ideal_event(0, 1)]
    if extra_event is not None:
        events.append(_measurement_event(0, 1, "sun", extra_event))
    output = stack.step(batch(1, *events))
    assert len(output.commands) == 1
    assert isinstance(output.commands[0].payload, IdealWrenchCommand)
    assert output.commands[0].frame == ACTUATOR_FRAME
    assert output.commands[0].satellite_id == SATELLITE_ID


def test_attitude_stack_rejects_future_stale_and_out_of_order_measurements() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config())
    stack.boot(boot_event())
    stack.step(batch(1, ideal_event(1, 1, rate=(0.01, 0.02, 0.03))))

    stack.step(batch(2, ideal_event(2, 5, rate=(9.0, 9.0, 9.0))))
    stack.step(batch(3, ideal_event(0, 0, rate=(8.0, 8.0, 8.0))))
    stack.step(batch(400, ideal_event(3, 1, rate=(7.0, 7.0, 7.0))))

    solution = stack._navigator.solution(clock(400))
    assert solution.angular_rate_body_rad_s == pytest.approx((0.01, 0.02, 0.03))


def test_direct_reaction_wheel_allocator_requires_unit_axes() -> None:
    with pytest.raises(ValueError, match="unit vectors"):
        replace(
            attitude_config().allocator,
            kind=AttitudeAllocatorKind.REACTION_WHEEL,
            axes_body=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )


def test_coarse_sun_acquisition_commands_without_attitude_quaternion() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config())
    stack.boot(boot_event())

    output = stack.step(
        batch(1, _measurement_event(0, 1, "sun", SunVectorMeasurement((0.0, 1.0, 0.0))))
    )
    fields = {field.name: field.value for field in output.telemetry[0].fields}

    assert fields["adcs_operational_mode"] == "coarse_sun"
    assert len(output.commands) == 1
    assert isinstance(output.commands[0].payload, IdealWrenchCommand)
    assert output.commands[0].payload.torque_n_m[2] > 0.0


@pytest.mark.parametrize(
    ("reference", "fault_component", "first_extra", "recovery_extra"),
    (
        (
            AttitudeReferenceConfig(AttitudeReferenceMode.SUN),
            "sun_sensor",
            SunVectorMeasurement((1.0, 0.0, 0.0)),
            SunVectorMeasurement((1.0, 0.0, 0.0)),
        ),
        (
            AttitudeReferenceConfig(
                AttitudeReferenceMode.TARGET,
                target_position_eci_m=(7_000_000.0, 1_000_000.0, 0.0),
            ),
            "target_tracker",
            None,
            None,
        ),
    ),
)
def test_reference_loss_rate_damps_and_reacquires_without_invalid_commands(
    reference: AttitudeReferenceConfig,
    fault_component: str,
    first_extra: object | None,
    recovery_extra: object | None,
) -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config(reference=reference))
    stack.boot(boot_event())

    first_events = [ideal_event(0, 1, rate=(0.02, -0.01, 0.005))]
    if first_extra is not None:
        first_events.append(_measurement_event(0, 1, "sun", first_extra))
    nominal = stack.step(batch(1, *first_events))
    unavailable = stack.step(
        batch(
            2,
            ideal_event(1, 2, rate=(0.02, -0.01, 0.005)),
            _fault_event(0, 2, fault_component, True),
        )
    )
    recovery_events = [
        ideal_event(2, 3, rate=(0.01, -0.005, 0.0025)),
        _fault_event(1, 3, fault_component, False),
    ]
    if recovery_extra is not None:
        recovery_events.append(_measurement_event(1, 3, "sun", recovery_extra))
    recovered = stack.step(batch(3, *recovery_events))

    def fields(output) -> dict[str, object]:
        return {field.name: field.value for field in output.telemetry[0].fields}

    assert fields(nominal)["adcs_operational_mode"] == "nominal"
    assert fields(unavailable)["adcs_operational_mode"] == "degraded"
    assert fields(unavailable)["reference_available"] is False
    assert "attitude_error_rad" not in fields(unavailable)
    assert unavailable.commands
    assert all(np.all(np.isfinite(command.payload.torque_n_m)) for command in unavailable.commands)
    assert fields(recovered)["adcs_operational_mode"] == "nominal"
    assert fields(recovered)["reference_available"] is True


def test_scheduled_target_change_is_exactly_once_and_snapshot_replay_safe() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(
        attitude_config(
            reference=AttitudeReferenceConfig(
                AttitudeReferenceMode.TARGET,
                target_position_eci_m=(7_000_000.0, 1_000_000.0, 0.0),
            )
        )
    )
    stack.boot(boot_event())
    command = GroundCommandPayload(
        "target-update-2",
        GroundCommandKind.STACK_COMMAND,
        (
            TelemetryField("operation", "set_target_eci"),
            TelemetryField("target_x_eci_m", 7_000_000.0, "m"),
            TelemetryField("target_y_eci_m", 0.0, "m"),
            TelemetryField("target_z_eci_m", 1_000_000.0, "m"),
        ),
        execute_at=clock(3),
    )
    event = InputEvent(
        PacketId("ground", BOOT_ID, 0),
        InputKind.GROUND_COMMAND,
        clock(2),
        clock(2),
        Quality(),
        command,
    )

    stack.step(batch(1, ideal_event(0, 1)))
    queued = stack.step(batch(2, ideal_event(1, 2), event))
    assert {field.name: field.value for field in queued.telemetry[0].fields}["target_update_count"] == 0
    snapshot = stack.snapshot()

    expected = stack.step(batch(3, ideal_event(2, 3)))
    stack.restore(snapshot)
    replay = stack.step(batch(3, ideal_event(2, 3)))
    assert canonical_json_bytes(replay) == canonical_json_bytes(expected)
    expected_fields = {field.name: field.value for field in expected.telemetry[0].fields}
    assert expected_fields["target_update_count"] == 1
    assert expected.commands[0].payload != queued.commands[0].payload

    duplicate = InputEvent(
        PacketId("ground", BOOT_ID, 1),
        InputKind.GROUND_COMMAND,
        clock(4),
        clock(4),
        Quality(),
        command,
    )
    after_duplicate = stack.step(batch(4, ideal_event(3, 4), duplicate))
    duplicate_fields = {field.name: field.value for field in after_duplicate.telemetry[0].fields}
    assert duplicate_fields["target_update_count"] == 1
    assert duplicate_fields["target_update_rejection_count"] == 0


def test_reference_quaternion_maps_requested_inertial_axis_to_body_x() -> None:
    navigator = AttitudeNavigator(body_frame=BODY_FRAME, inertial_frame=INERTIAL_FRAME)
    navigator.ingest((ideal_event(0, 1),))
    solution = navigator.solution(clock(1))
    reference = AttitudeReferenceGenerator(
        AttitudeReferenceConfig(AttitudeReferenceMode.VELOCITY),
        inertial_frame=INERTIAL_FRAME,
    ).generate(solution)
    assert reference is not None and reference.attitude_quat_from_frame is not None
    mapped = quaternion_to_dcm_bn(np.asarray(reference.attitude_quat_from_frame)) @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(mapped, (1.0, 0.0, 0.0), atol=1e-12)


def test_ideal_attitude_navigation_skips_unused_sensor_frame_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_unused_rotation(_: np.ndarray) -> np.ndarray:
        raise AssertionError("ideal navigation should not build an unused sensor-frame DCM")

    monkeypatch.setattr(attitude_v2, "quaternion_to_dcm_bn", fail_unused_rotation)
    navigator = AttitudeNavigator(body_frame=BODY_FRAME, inertial_frame=INERTIAL_FRAME)
    navigator.ingest((ideal_event(0, 1),))

    solution = navigator.control_solution(clock(1))

    assert solution.attitude_quat_bn == (1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("kind", "payload_type", "extra_event"),
    (
        (AttitudeAllocatorKind.REACTION_WHEEL, ReactionWheelTorqueCommand, None),
        (
            AttitudeAllocatorKind.MAGNETORQUER,
            MagnetorquerDipoleCommand,
            MagnetometerMeasurement((0.0, 0.0, 2.5e-5)),
        ),
        (AttitudeAllocatorKind.CMG, CmgGimbalRateCommand, None),
    ),
)
def test_attitude_allocators_publish_device_coordinate_commands(
    kind: AttitudeAllocatorKind, payload_type: type[object], extra_event: object | None
) -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config(allocator_kind=kind))
    stack.boot(boot_event())
    events = [ideal_event(0, 1, rate=(0.2, -0.1, 0.05))]
    if extra_event is not None:
        events.append(_measurement_event(0, 1, "mag", extra_event))
    output = stack.step(batch(1, *events))
    assert len(output.commands) == 1
    assert isinstance(output.commands[0].payload, payload_type)


def test_dual_actuator_fault_does_not_command_the_isolated_backup() -> None:
    config = replace(
        attitude_config(allocator_kind=AttitudeAllocatorKind.REACTION_WHEEL),
        health=HealthManagerConfig(
            actuator_fallbacks=(("attitude-actuator", "momentum-dump"),)
        ),
        momentum_unload=MomentumUnloadConfig(
            wheel_actuator_id="attitude-actuator",
            torquer_actuator_id="momentum-dump",
            wheel_max_momentum_n_m_s=(1.0, 1.0, 1.0),
        ),
    )
    stack = AttitudeReferenceFlightSoftwareStack(config)
    stack.boot(boot_event())
    time = clock(1)

    def fault(sequence: int, component_id: str) -> InputEvent:
        return InputEvent(
            PacketId("fault-monitor", BOOT_ID, sequence),
            InputKind.MODELED_FAULT_INDICATION,
            time,
            time,
            Quality(),
            ModeledFaultIndicationPayload(component_id, "modeled_fault", True, "fault-monitor"),
        )

    output = stack.step(
        batch(
            1,
            ideal_event(0, 1, rate=(0.2, 0.0, 0.0)),
            fault(0, "attitude-actuator"),
            fault(1, "momentum-dump"),
        )
    )
    fields = {field.name: field.value for field in output.telemetry[0].fields}

    assert output.commands == ()
    assert fields["health_state"] == "recovery"
    assert fields["adcs_operational_mode"] == "degraded"


def test_gyro_rate_propagates_attitude_between_absolute_updates() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config())
    stack.boot(boot_event())
    initial = stack.step(batch(1, ideal_event(0, 1, rate=(0.0, 0.0, 0.02))))
    propagated = stack.step(batch(2))
    assert initial.commands[0].payload != propagated.commands[0].payload


def test_small_angle_lqr_law_composes_with_ric_reference_and_wheel_allocation() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(
        attitude_config(
            allocator_kind=AttitudeAllocatorKind.REACTION_WHEEL,
            reference=AttitudeReferenceConfig(AttitudeReferenceMode.RIC, ric_axis="in_track"),
            controller=SmallAngleLqrTorqueController(max_torque_n_m=0.2),
        )
    )
    stack.boot(boot_event())
    output = stack.step(batch(1, ideal_event(0, 1, rate=(0.02, -0.01, 0.03))))
    assert isinstance(output.commands[0].payload, ReactionWheelTorqueCommand)
    assert any(abs(value) > 0.0 for value in output.commands[0].payload.torque_n_m)


def test_experimental_detumble_and_pointing_vertical_slice() -> None:
    """Test-only wiring: physical sample -> stack -> bus -> coupled physics."""

    stack = AttitudeReferenceFlightSoftwareStack(attitude_config())
    stack.boot(boot_event())
    bus = ActuatorCommandBus(
        (
            ActuatorDeviceDefinition(
                SATELLITE_ID,
                "attitude-actuator",
                ACTUATOR_FRAME,
                (IdealWrenchCommand,),
                ExpiryBehavior.ZERO,
            ),
        )
    )
    dynamics = CoupledSatelliteDynamics(
        effects_model=lambda _time, _state, control: StageEffects(
            torque_body_n_m=np.asarray(control.torque_n_m if isinstance(control, IdealWrenchCommand) else (0, 0, 0))
        ),
        mass_properties_model=constant_mass_properties(np.diag([1.5, 2.0, 2.5])),
    )
    integrator = CoupledSatelliteIntegrator(CoupledIntegratorConfig(0.1, 0.05), dynamics.derivative)
    half_angle = np.deg2rad(60.0) / 2.0
    state = CoupledSatelliteState(
        np.array([7_000.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([cos(half_angle), 0.0, 0.0, sin(half_angle)]),
        np.array([0.8, -0.25, 0.15]),
        100.0,
        np.zeros(0),
        0.0,
    )
    initial_rate = float(np.linalg.norm(state.angular_rate_body_rad_s))
    initial_error = 2.0 * np.arccos(abs(float(state.attitude_quat_bn[0])))
    operational_modes: set[str] = set()

    # Experimental envelope: recover an 0.85 rad/s tumble and 60 degree
    # pointing error within 60 simulated seconds at a 10 Hz FSW cadence.
    for invocation in range(1, 601):
        event = ideal_event(
            invocation - 1,
            invocation,
            quaternion=tuple(float(value) for value in state.attitude_quat_bn),
            rate=tuple(float(value) for value in state.angular_rate_body_rad_s),
            position_m=tuple(float(value * 1_000.0) for value in state.position_eci_km),
            velocity_m_s=tuple(float(value * 1_000.0) for value in state.velocity_eci_km_s),
        )
        output = stack.step(batch(invocation, event))
        operational_modes.add(
            str(next(field.value for field in output.telemetry[0].fields if field.name == "adcs_operational_mode"))
        )
        receipts = bus.publish_all(output.commands, received_at=clock(invocation))
        assert all(receipt.disposition is CommandDisposition.ACCEPTED for receipt in receipts)
        demand = bus.demand(
            satellite_id=SATELLITE_ID,
            actuator_id="attitude-actuator",
            at=clock(invocation),
        )
        state = integrator.propagate(
            state,
            end_time_s=state.t_s + 0.1,
            control=demand.payload,
        ).final_state

    final_rate = float(np.linalg.norm(state.angular_rate_body_rad_s))
    final_error = 2.0 * np.arccos(abs(float(state.attitude_quat_bn[0])))
    assert final_rate < 0.02
    assert final_error < np.deg2rad(2.0)
    assert final_rate < initial_rate / 20.0
    assert final_error < initial_error / 20.0
    assert "detumble" in operational_modes
    assert "nominal" in operational_modes
