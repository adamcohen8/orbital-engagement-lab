"""Shared fixtures for the GNC v2 WP4 flight-software tests."""

from __future__ import annotations

from sim.flight_software import (
    AttitudeReferenceStackConfig,
    BootEvent,
    ClockScale,
    ClockTag,
    FlightSoftwareInputBatch,
    FrameId,
    IdealOwnStateMeasurement,
    InputEvent,
    InputKind,
    MeasurementEvent,
    PacketId,
    Quality,
)
from sim.gnc.attitude_v2 import (
    AttitudeAllocatorConfig,
    AttitudeAllocatorKind,
    AttitudeReferenceConfig,
    QuaternionTorqueController,
    SensorCalibration,
    SensorMounting,
    SmallAngleLqrTorqueController,
)

SATELLITE_ID = "sat"
BOOT_ID = "boot-wp4"
BODY_FRAME = FrameId("OEL/BODY/sat", "frames-v1")
INERTIAL_FRAME = FrameId("OEL/ECI/J2000", "frames-v1")
ACTUATOR_FRAME = FrameId("OEL/ACTUATOR/sat/attitude", "frames-v1")


def clock(tick: int, *, period_ns: int = 100_000_000) -> ClockTag:
    return ClockTag("sat/onboard", tick, period_ns, ClockScale.ONBOARD)


def attitude_config(
    *,
    allocator_kind: AttitudeAllocatorKind = AttitudeAllocatorKind.IDEAL_WRENCH,
    reference: AttitudeReferenceConfig | None = None,
    mountings: tuple[SensorMounting, ...] = (),
    calibrations: tuple[SensorCalibration, ...] = (),
    controller: QuaternionTorqueController | SmallAngleLqrTorqueController | None = None,
) -> AttitudeReferenceStackConfig:
    return AttitudeReferenceStackConfig(
        SATELLITE_ID,
        BODY_FRAME,
        INERTIAL_FRAME,
        AttitudeAllocatorConfig(
            SATELLITE_ID,
            allocator_kind,
            "attitude-actuator",
            ACTUATOR_FRAME,
            limits=(0.25, 0.25, 0.25),
        ),
        reference=reference or AttitudeReferenceConfig(),
        controller=controller or QuaternionTorqueController(max_torque_n_m=0.25),
        sensor_mountings=mountings,
        sensor_calibrations=calibrations,
    )


def boot_event() -> BootEvent:
    return BootEvent(SATELLITE_ID, BOOT_ID, clock(0))


def ideal_event(
    sequence: int,
    tick: int,
    *,
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    rate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    position_m: tuple[float, float, float] = (7_000_000.0, 0.0, 0.0),
    velocity_m_s: tuple[float, float, float] = (0.0, 7_500.0, 0.0),
) -> InputEvent:
    time = clock(tick)
    payload = IdealOwnStateMeasurement(position_m, velocity_m_s, quaternion, rate)
    measurement = MeasurementEvent("ideal-own-state", payload.schema, time, BODY_FRAME, payload)
    return InputEvent(
        PacketId("ideal-own-state", BOOT_ID, sequence),
        InputKind.MEASUREMENT,
        time,
        time,
        Quality(),
        measurement,
    )


def batch(invocation: int, *events: InputEvent) -> FlightSoftwareInputBatch:
    return FlightSoftwareInputBatch(SATELLITE_ID, invocation, clock(invocation), tuple(events))
