from __future__ import annotations

from dataclasses import replace

import pytest

from sim.flight_software import (
    ContinuousEngineCommand,
    GoalDefinition,
    GoalMode,
    LowThrustReferenceFlightSoftwareStack,
    LowThrustReferenceStackConfig,
    MeasurementEvent,
    PacketId,
    Quality,
    VehicleResourceMeasurement,
    canonical_json_bytes,
)
from sim.flight_software.contracts import InputEvent, InputKind
from sim.gnc.executive_v2 import ReferenceExecutiveConfig
from sim.gnc.navigation_v2 import NavigationInitializationMode
from sim.gnc.operations_v2 import ResourceLimits
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
    boot_event,
    clock,
    ideal_event,
)
from sim.tests.fsw_v2_orbit_helpers import (
    ENGINE_FRAME,
    RELATIVE_FRAME,
    fault_event,
    gnss_event,
    navigation_batch,
    relative_event,
    telemetry_fields,
)


def _config(
    mode: TranslationMode = TranslationMode.LOW_THRUST_PHASING,
    *,
    max_force_n: float = 0.2,
    control_options: dict[str, object] | None = None,
) -> LowThrustReferenceStackConfig:
    goal_type = "low_thrust_phasing" if mode is TranslationMode.LOW_THRUST_PHASING else "orbital_elements"
    goal = GoalDefinition("phase", goal_type, GoalMode.TERMINAL, target_frame=RELATIVE_FRAME)
    mode_options: dict[str, object] = (
        {"target_relative_state_ric": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), "approach_speed_m_s": 0.02}
        if mode is TranslationMode.LOW_THRUST_PHASING
        else {"target_semi_major_axis_m": 7_100_000.0, "target_eccentricity": 0.001}
    )
    mode_options.update(control_options or {})
    return LowThrustReferenceStackConfig(
        SATELLITE_ID,
        BODY_FRAME,
        INERTIAL_FRAME,
        RELATIVE_FRAME,
        NavigationInitializationMode.IDEAL,
        TranslationControlConfig(
            mode,
            100.0,
            0.002,
            target_id="target",
            **mode_options,
        ),
        TranslationAllocatorConfig(
            SATELLITE_ID,
            TranslationAllocatorKind.CONTINUOUS_ENGINE,
            "engine",
            ENGINE_FRAME,
            max_force_n,
            gimbal_limit_rad=0.5,
        ),
        ReferenceExecutiveConfig(goal, mode.value),
    )


def test_low_thrust_phasing_publishes_bounded_continuous_engine_command() -> None:
    stack = LowThrustReferenceFlightSoftwareStack(_config())
    stack.boot(boot_event())
    output = stack.step(navigation_batch(1, range_m=2_000.0))
    assert len(output.commands) == 1
    payload = output.commands[0].payload
    assert isinstance(payload, ContinuousEngineCommand)
    assert 0.0 < payload.throttle_0_1 <= 1.0
    assert len(payload.gimbal_angles_rad) == 2
    assert telemetry_fields(output)["selected_mode"] == "low_thrust_phasing"


def test_low_thrust_saturation_is_reported_and_snapshot_replay_is_exact() -> None:
    stack = LowThrustReferenceFlightSoftwareStack(_config(max_force_n=0.001))
    stack.boot(boot_event())
    stack.step(navigation_batch(1))
    snapshot = stack.snapshot()
    expected = stack.step(navigation_batch(2, range_m=1_500.0))
    stack.restore(snapshot)
    replay = stack.step(navigation_batch(2, range_m=1_500.0))
    assert canonical_json_bytes(replay.commands) == canonical_json_bytes(expected.commands)
    assert telemetry_fields(replay)["translation_allocation_status"] == "saturated"


def _window_batch(invocation: int):
    return batch(
        invocation,
        ideal_event(invocation, invocation),
        relative_event(invocation, invocation, range_m=2_000.0),
    )


def test_low_thrust_windows_coast_count_missed_opportunities_and_restart_exactly() -> None:
    stack = LowThrustReferenceFlightSoftwareStack(
        _config(
            control_options={
                "thrust_window_period_s": 0.4,
                "thrust_window_duration_s": 0.2,
            }
        )
    )
    stack.boot(boot_event())

    initial = stack.step(_window_batch(1))
    assert telemetry_fields(initial)["low_thrust_window_open"] is True
    assert initial.commands
    observed = stack.step(_window_batch(5))
    assert telemetry_fields(observed)["low_thrust_missed_window_count"] == 0
    snapshot = stack.snapshot()

    expected = stack.step(_window_batch(13))
    fields = telemetry_fields(expected)
    assert fields["low_thrust_window_open"] is True
    assert fields["low_thrust_missed_window_count"] == 1
    stack.restore(snapshot)
    replay = stack.step(_window_batch(13))
    assert canonical_json_bytes(replay) == canonical_json_bytes(expected)

    coast = stack.step(_window_batch(14))
    assert telemetry_fields(coast)["control_phase"] == "thrust_window_coast"
    payload = coast.commands[0].payload
    assert isinstance(payload, ContinuousEngineCommand)
    assert payload.throttle_0_1 == 0.0


def test_low_thrust_power_inhibition_and_resumption_are_stack_owned() -> None:
    base = _config()
    config = replace(
        base,
        resources=ResourceLimits(minimum_available_power_w=50.0),
        executive=replace(base.executive, recovery_clear_dwell_s=0.0),
    )
    stack = LowThrustReferenceFlightSoftwareStack(config)
    stack.boot(boot_event())

    def resource_event(sequence: int, tick: int, power_w: float) -> InputEvent:
        time = clock(tick)
        payload = VehicleResourceMeasurement(available_power_w=power_w)
        measurement = MeasurementEvent("platform", payload.schema, time, BODY_FRAME, payload)
        return InputEvent(
            PacketId("platform", BOOT_ID, sequence),
            InputKind.MEASUREMENT,
            time,
            time,
            Quality(),
            measurement,
        )

    inhibited = stack.step(
        batch(1, ideal_event(0, 1), relative_event(0, 1), resource_event(0, 1, 10.0))
    )
    inhibited_fields = telemetry_fields(inhibited)
    assert inhibited_fields["resource_command_allowed"] is False
    assert inhibited_fields["resource_violation.available_power_low"] is True
    assert inhibited.commands == ()

    resumed = stack.step(
        batch(2, ideal_event(1, 2), relative_event(1, 2), resource_event(1, 2, 100.0))
    )
    assert telemetry_fields(resumed)["resource_command_allowed"] is True
    assert resumed.commands


@pytest.mark.parametrize("mode", (TranslationMode.LOW_THRUST_PHASING, TranslationMode.ORBITAL_ELEMENTS))
def test_each_advertised_low_thrust_mode_has_nominal_saturation_infeasible_and_fault_evidence(
    mode: TranslationMode,
) -> None:
    nominal = LowThrustReferenceFlightSoftwareStack(_config(mode))
    nominal.boot(boot_event())
    nominal_output = nominal.step(navigation_batch(1))
    assert isinstance(nominal_output.commands[0].payload, ContinuousEngineCommand)
    assert telemetry_fields(nominal_output)["selected_mode"] == mode.value

    saturated = LowThrustReferenceFlightSoftwareStack(_config(mode, max_force_n=1.0e-9))
    saturated.boot(boot_event())
    assert telemetry_fields(saturated.step(navigation_batch(1)))["translation_allocation_status"] == "saturated"

    cold = replace(_config(mode), navigation_initialization=NavigationInitializationMode.COLD)
    infeasible = LowThrustReferenceFlightSoftwareStack(cold)
    infeasible.boot(boot_event())
    events = [gnss_event(0, 1)]
    if mode is TranslationMode.LOW_THRUST_PHASING:
        events.append(relative_event(0, 1))
    infeasible_output = infeasible.step(batch(1, *events))
    assert telemetry_fields(infeasible_output)["translation_allocation_status"] == "infeasible"

    faulted = LowThrustReferenceFlightSoftwareStack(_config(mode))
    faulted.boot(boot_event())
    faulted.step(navigation_batch(1))
    sensor_fault = faulted.step(
        batch(
            2,
            fault_event(0, 2, "ideal-own-state"),
            ideal_event(1, 2),
            relative_event(1, 2),
        )
    )
    assert telemetry_fields(sensor_fault)["executive_phase"] == "recovery"
    actuator_fault = faulted.step(
        batch(
            3,
            fault_event(1, 3, "ideal-own-state", active=False),
            fault_event(2, 3, "engine"),
            ideal_event(2, 3),
            relative_event(2, 3),
        )
    )
    assert actuator_fault.commands == ()
