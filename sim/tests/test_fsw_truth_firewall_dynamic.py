from __future__ import annotations

from math import cos, sin

import numpy as np
import pytest

from sim.core.models import StateTruth
from sim.flight_software import (
    AttitudeReferenceFlightSoftwareStack,
    DiagnosticTelemetry,
    FlightSoftwareOutput,
    FrameId,
    GyroMeasurement,
    InputEvent,
    InputKind,
    MeasurementEvent,
    PacketId,
    PassiveFlightSoftwareStack,
    PassiveStackConfig,
    Quality,
    TelemetryField,
    canonical_json_bytes,
)
from sim.gnc.attitude_v2 import SensorCalibration, SensorMounting
from sim.runtime.satellites.flight_software_runtime import SatelliteFlightSoftwareRuntime
from sim.tests.fsw_v2_helpers import (
    BODY_FRAME,
    BOOT_ID,
    attitude_config,
    batch,
    boot_event,
    clock,
    ideal_event,
)
from sim.utils.quaternion import quaternion_to_dcm_bn


def _gyro_event(sensor_rate_rad_s: tuple[float, float, float]) -> InputEvent:
    time = clock(1)
    payload = GyroMeasurement(sensor_rate_rad_s)
    measurement = MeasurementEvent("gyro", payload.schema, time, BODY_FRAME, payload)
    return InputEvent(PacketId("gyro", BOOT_ID, 0), InputKind.MEASUREMENT, time, time, Quality(), measurement)


def _command_for_mounting(
    onboard_quat_body_from_sensor: tuple[float, float, float, float],
    *,
    true_quat_body_from_sensor: tuple[float, float, float, float],
    onboard_calibration: SensorCalibration | None = None,
    true_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bytes:
    # Physics uses the hidden true mounting to create the delivered sensor-axis
    # sample.  The stack receives only that sample and its onboard belief.
    true_body_rate = np.array([0.2, -0.05, 0.0])
    sensor_rate = (quaternion_to_dcm_bn(np.asarray(true_quat_body_from_sensor)).T @ true_body_rate) / np.asarray(
        true_scale
    )
    stack = AttitudeReferenceFlightSoftwareStack(
        attitude_config(
            mountings=(SensorMounting("gyro", onboard_quat_body_from_sensor),),
            calibrations=() if onboard_calibration is None else (onboard_calibration,),
        )
    )
    stack.boot(boot_event())
    output = stack.step(
        batch(
            1,
            ideal_event(0, 1),
            _gyro_event(tuple(float(value) for value in sensor_rate)),
        )
    )
    return canonical_json_bytes(output.commands)


def test_false_onboard_mounting_changes_command_without_disclosing_true_mounting() -> None:
    angle = np.deg2rad(90.0) / 2.0
    true_mounting = (cos(angle), 0.0, 0.0, sin(angle))
    correct = _command_for_mounting(true_mounting, true_quat_body_from_sensor=true_mounting)
    incorrect = _command_for_mounting((1.0, 0.0, 0.0, 0.0), true_quat_body_from_sensor=true_mounting)
    repeat = _command_for_mounting((1.0, 0.0, 0.0, 0.0), true_quat_body_from_sensor=true_mounting)

    assert incorrect != correct
    assert repeat == incorrect


def test_false_onboard_calibration_changes_command_without_disclosing_true_calibration() -> None:
    identity = (1.0, 0.0, 0.0, 0.0)
    true_scale = (2.0, 1.0, 1.0)
    correct = _command_for_mounting(
        identity,
        true_quat_body_from_sensor=identity,
        onboard_calibration=SensorCalibration("gyro", scale=true_scale),
        true_scale=true_scale,
    )
    incorrect = _command_for_mounting(
        identity,
        true_quat_body_from_sensor=identity,
        onboard_calibration=SensorCalibration("gyro"),
        true_scale=true_scale,
    )
    assert incorrect != correct


def test_reference_stack_rejects_truth_even_in_an_otherwise_open_input_kind() -> None:
    truth = StateTruth(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 100.0, 0.0)
    time = clock(1)
    with pytest.raises(TypeError, match="ground_command event payload must be GroundCommandPayload"):
        InputEvent(
            PacketId("ground", BOOT_ID, 0),
            InputKind.GROUND_COMMAND,
            time,
            time,
            Quality(),
            truth,
        )


def test_runtime_keeps_recursive_output_firewall_for_external_stack_subclasses() -> None:
    truth = StateTruth(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 100.0, 0.0)

    class LeakyExternalStack(PassiveFlightSoftwareStack):
        def _step(self, incoming):  # type: ignore[no-untyped-def]
            return FlightSoftwareOutput(
                incoming.satellite_id,
                incoming.invocation_id,
                telemetry=(
                    DiagnosticTelemetry(
                        "external.leak",
                        incoming.invocation_time,
                        (TelemetryField("hidden", truth),),
                    ),
                ),
            )

    inertial = FrameId("OEL/ECI/J2000", "frames-v1")
    body = FrameId("OEL/BODY/satellite", "frames-v1")
    runtime = SatelliteFlightSoftwareRuntime(
        satellite_id="satellite",
        stack=LeakyExternalStack(
            PassiveStackConfig(
                satellite_id="satellite",
                inertial_frame=inertial,
                body_frame=body,
            )
        ),
        devices=(),
        hardware={},
        inertial_frame=inertial,
        body_frame=body,
        task_period_ns=1_000_000_000,
    )
    with pytest.raises(TypeError, match="forbidden simulator-owned"):
        runtime.prepare_interval(truth, start_time_ns=0)


def test_runtime_checks_publicly_enqueued_open_kind_at_point_of_use() -> None:
    truth = StateTruth(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 100.0, 0.0)
    inertial = FrameId("OEL/ECI/J2000", "frames-v1")
    body = FrameId("OEL/BODY/satellite", "frames-v1")
    runtime = SatelliteFlightSoftwareRuntime(
        satellite_id="satellite",
        stack=PassiveFlightSoftwareStack(
            PassiveStackConfig(
                satellite_id="satellite",
                inertial_frame=inertial,
                body_frame=body,
            )
        ),
        devices=(),
        hardware={},
        inertial_frame=inertial,
        body_frame=body,
        task_period_ns=1_000_000_000,
    )
    at = runtime.clock_tag(0)
    runtime.enqueue(
        InputEvent(
            PacketId("external", "boot-0", 0),
            InputKind.MISSION_LOAD,
            at,
            at,
            Quality(),
            truth,
        )
    )
    with pytest.raises(TypeError, match="forbidden simulator-owned"):
        runtime.prepare_interval(truth, start_time_ns=0)
