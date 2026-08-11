"""Shared fixtures for GNC v2 WP5 orbit, executive, and RPO tests."""

from __future__ import annotations

from sim.flight_software import (
    ConstraintDefinition,
    FlightSoftwareInputBatch,
    FrameId,
    GnssOwnStateMeasurement,
    GoalDefinition,
    GoalMode,
    GroundCommandKind,
    GroundCommandPayload,
    InputEvent,
    InputKind,
    MeasurementEvent,
    ModeledFaultIndicationPayload,
    PacketId,
    Quality,
    RelativeObservationMeasurement,
    RpoReferenceStackConfig,
    TelemetryField,
)
from sim.flight_software.loads import ConstraintKind
from sim.gnc.executive_v2 import ReferenceExecutiveConfig
from sim.gnc.navigation_v2 import NavigationInitializationMode
from sim.gnc.orbit_v2 import (
    TranslationAllocatorConfig,
    TranslationAllocatorKind,
    TranslationControlConfig,
    TranslationMode,
)
from sim.tests.fsw_v2_helpers import (
    BODY_FRAME,
    BOOT_ID,
    INERTIAL_FRAME,
    SATELLITE_ID,
    batch,
    clock,
    ideal_event,
)

RELATIVE_FRAME = FrameId("OEL/RIC/sat", "frames-v1")
ENGINE_FRAME = FrameId("OEL/ACTUATOR/sat/engine", "frames-v1")


def goal(
    goal_type: str = "rpo.hold",
    *,
    mode: GoalMode = GoalMode.TERMINAL,
    dwell_s: float = 0.0,
    valid_during=None,
) -> GoalDefinition:
    return GoalDefinition(
        "primary", goal_type, mode, target_frame=RELATIVE_FRAME, dwell_s=dwell_s, valid_during=valid_during
    )


def rpo_config(
    mode: TranslationMode = TranslationMode.RIC_HOLD,
    *,
    initialization: NavigationInitializationMode = NavigationInitializationMode.IDEAL,
    allocator_kind: TranslationAllocatorKind = TranslationAllocatorKind.IDEAL_WRENCH,
    max_acceleration_m_s2: float = 0.02,
    max_force_n: float = 5.0,
    executive: ReferenceExecutiveConfig | None = None,
    waypoints: tuple[tuple[float, ...], ...] = (),
    attitude_allocator=None,
    pointing_tolerance_rad: float = 0.1,
) -> RpoReferenceStackConfig:
    actuator_frame = INERTIAL_FRAME if allocator_kind is TranslationAllocatorKind.IDEAL_WRENCH else ENGINE_FRAME
    control = TranslationControlConfig(
        mode,
        100.0,
        max_acceleration_m_s2,
        target_relative_state_ric=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        waypoints_ric=waypoints,
        position_tolerance_m=5.0,
        velocity_tolerance_m_s=0.01,
        target_id="target",
    )
    allocator = TranslationAllocatorConfig(
        SATELLITE_ID,
        allocator_kind,
        "translation",
        actuator_frame,
        max_force_n,
    )
    return RpoReferenceStackConfig(
        SATELLITE_ID,
        BODY_FRAME,
        INERTIAL_FRAME,
        RELATIVE_FRAME,
        initialization,
        control,
        allocator,
        executive or ReferenceExecutiveConfig(goal(), mode.value),
        attitude_allocator=attitude_allocator,
        pointing_tolerance_rad=pointing_tolerance_rad,
    )


def relative_event(
    sequence: int,
    tick: int,
    *,
    range_m: float = 1_000.0,
    range_rate_m_s: float = 0.0,
    los: tuple[float, float, float] = (1.0, 0.0, 0.0),
    angular_rate_rad_s: tuple[float, float, float] | None = None,
    sensor_id: str = "relative",
    frame: FrameId = RELATIVE_FRAME,
) -> InputEvent:
    time = clock(tick)
    payload = RelativeObservationMeasurement(
        range_m,
        range_rate_m_s,
        los,
        angular_rate_rad_s,
        target_track_id="target",
    )
    measurement = MeasurementEvent(sensor_id, payload.schema, time, frame, payload)
    return InputEvent(
        PacketId(sensor_id, BOOT_ID, sequence),
        InputKind.MEASUREMENT,
        time,
        time,
        Quality(),
        measurement,
    )


def gnss_event(sequence: int, tick: int) -> InputEvent:
    time = clock(tick)
    payload = GnssOwnStateMeasurement((7_000_000.0, 0.0, 0.0), (0.0, 7_500.0, 0.0))
    measurement = MeasurementEvent("gnss", payload.schema, time, INERTIAL_FRAME, payload)
    return InputEvent(PacketId("gnss", BOOT_ID, sequence), InputKind.MEASUREMENT, time, time, Quality(), measurement)


def fault_event(sequence: int, tick: int, component_id: str, *, active: bool = True) -> InputEvent:
    time = clock(tick)
    return InputEvent(
        PacketId("fault-monitor", BOOT_ID, sequence),
        InputKind.MODELED_FAULT_INDICATION,
        time,
        time,
        Quality(),
        ModeledFaultIndicationPayload(component_id, "modeled_fault", active, "fault-monitor"),
    )


def ground_condition(sequence: int, tick: int, condition_id: str) -> InputEvent:
    time = clock(tick)
    return InputEvent(
        PacketId("ground", BOOT_ID, sequence),
        InputKind.GROUND_COMMAND,
        time,
        time,
        Quality(),
        GroundCommandPayload(
            f"condition:{condition_id}",
            GroundCommandKind.ACTION_REQUEST,
            (TelemetryField("condition_id", condition_id),),
        ),
    )


def navigation_batch(invocation: int, *, range_m: float = 1_000.0, rate_m_s: float = 0.0) -> FlightSoftwareInputBatch:
    return batch(
        invocation,
        ideal_event(invocation - 1, invocation),
        relative_event(invocation - 1, invocation, range_m=range_m, range_rate_m_s=rate_m_s),
    )


def telemetry_fields(output) -> dict[str, object]:
    return {field.name: field.value for field in output.telemetry[0].fields}


def safety_constraint(minimum_range_m: float) -> ConstraintDefinition:
    from sim.flight_software import TelemetryField

    return ConstraintDefinition(
        "keep-out",
        ConstraintKind.MISSION_SAFETY_ENVELOPE,
        "minimum_range_m",
        (TelemetryField("minimum_m", minimum_range_m, "m"),),
    )
