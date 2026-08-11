from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim.flight_software import (
    GoalDefinition,
    GoalMode,
    IdealWrenchCommand,
    OrbitReferenceFlightSoftwareStack,
    OrbitReferenceStackConfig,
    RpoReferenceFlightSoftwareStack,
)
from sim.gnc.executive_v2 import ReferenceExecutiveConfig
from sim.gnc.navigation_v2 import NavigationInitializationMode
from sim.gnc.orbit_v2 import (
    ScheduledBurn,
    TranslationAllocatorConfig,
    TranslationAllocatorKind,
    TranslationControlConfig,
    TranslationMode,
)
from sim.tests.fsw_v2_helpers import BODY_FRAME, INERTIAL_FRAME, SATELLITE_ID, batch, boot_event, ideal_event
from sim.tests.fsw_v2_orbit_helpers import (
    ENGINE_FRAME,
    RELATIVE_FRAME,
    fault_event,
    gnss_event,
    navigation_batch,
    rpo_config,
    telemetry_fields,
)


def _orbit_config(control: TranslationControlConfig, goal_type: str) -> OrbitReferenceStackConfig:
    goal = GoalDefinition("orbit-goal", goal_type, GoalMode.TERMINAL, target_frame=INERTIAL_FRAME)
    return OrbitReferenceStackConfig(
        SATELLITE_ID,
        BODY_FRAME,
        INERTIAL_FRAME,
        RELATIVE_FRAME,
        NavigationInitializationMode.IDEAL,
        control,
        TranslationAllocatorConfig(
            SATELLITE_ID,
            TranslationAllocatorKind.IDEAL_WRENCH,
            "translation",
            INERTIAL_FRAME,
            5.0,
        ),
        ReferenceExecutiveConfig(goal, control.default_mode.value, recovery_mode=control.default_mode.value),
    )


def _orbit_stack(control: TranslationControlConfig, goal_type: str) -> OrbitReferenceFlightSoftwareStack:
    config = _orbit_config(control, goal_type)
    stack = OrbitReferenceFlightSoftwareStack(config)
    stack.boot(boot_event())
    return stack


def test_stationkeeping_reference_uses_si_state_and_force_command() -> None:
    control = TranslationControlConfig(
        TranslationMode.STATIONKEEPING,
        100.0,
        0.02,
        target_state_eci=(7_000_100.0, 0.0, 0.0, 0.0, 7_500.0, 0.0),
        position_tolerance_m=1.0,
    )
    output = _orbit_stack(control, "orbit.stationkeeping").step(batch(1, ideal_event(0, 1)))
    assert isinstance(output.commands[0].payload, IdealWrenchCommand)
    assert output.commands[0].payload.force_n[0] > 0.0
    assert telemetry_fields(output)["selected_mode"] == "stationkeeping"


def test_ideal_wrench_command_is_expressed_in_actuator_body_frame() -> None:
    control = TranslationControlConfig(
        TranslationMode.STATIONKEEPING,
        100.0,
        0.02,
        target_state_eci=(7_000_100.0, 0.0, 0.0, 0.0, 7_500.0, 0.0),
        position_tolerance_m=1.0,
    )
    half = float(np.sqrt(0.5))
    output = _orbit_stack(control, "orbit.stationkeeping").step(
        batch(1, ideal_event(0, 1, quaternion=(half, 0.0, 0.0, -half)))
    )

    force_body = np.asarray(output.commands[0].payload.force_n, dtype=float)  # type: ignore[union-attr]
    assert abs(force_body[0]) < 1.0e-12
    assert force_body[1] > 0.0


def test_orbital_element_reference_generates_bounded_element_trim() -> None:
    control = TranslationControlConfig(
        TranslationMode.ORBITAL_ELEMENTS,
        100.0,
        0.001,
        target_semi_major_axis_m=7_100_000.0,
        target_eccentricity=0.001,
        position_tolerance_m=100.0,
    )
    output = _orbit_stack(control, "orbital_elements").step(batch(1, ideal_event(0, 1)))
    assert output.commands
    assert 0.0 < telemetry_fields(output)["requested_force_n"] <= 0.1


def test_atmospheric_pass_recovery_is_state_driven_and_restart_safe() -> None:
    control = TranslationControlConfig(
        TranslationMode.ATMOSPHERIC_PASS,
        100.0,
        0.01,
        atmospheric_prograde_acceleration_m_s2=0.01,
        atmospheric_pass_entry_altitude_m=180_000.0,
        atmospheric_pass_exit_altitude_m=190_000.0,
        atmospheric_recovery_delta_v_m_s=0.002,
    )
    stack = _orbit_stack(control, "orbit.atmospheric_pass")
    earth_radius_m = 6_378_137.0

    before = stack.step(batch(1, ideal_event(0, 1, position_m=(earth_radius_m + 200_000.0, 0.0, 0.0))))
    assert telemetry_fields(before)["control_phase"] == "awaiting_pass"
    assert before.commands[0].payload.force_n == pytest.approx((0.0, 0.0, 0.0))  # type: ignore[union-attr]

    inside = stack.step(batch(2, ideal_event(1, 2, position_m=(earth_radius_m + 175_000.0, 0.0, 0.0))))
    assert telemetry_fields(inside)["control_phase"] == "atmospheric_pass"
    assert inside.commands[0].payload.force_n == pytest.approx((0.0, 0.0, 0.0))  # type: ignore[union-attr]

    snapshot = stack.snapshot()
    restored = _orbit_stack(control, "orbit.atmospheric_pass")
    restored.restore(snapshot)
    exited = restored.step(batch(3, ideal_event(2, 3, position_m=(earth_radius_m + 195_000.0, 0.0, 0.0))))
    assert telemetry_fields(exited)["control_phase"] == "post_pass_recovery"
    assert exited.commands[0].payload.force_n[1] > 0.0  # type: ignore[union-attr]

    complete = restored.step(batch(6, ideal_event(3, 6, position_m=(earth_radius_m + 200_000.0, 0.0, 0.0))))
    assert telemetry_fields(complete)["goal_state"] == "achieved"
    assert telemetry_fields(complete)["command_count"] == 0


def test_scheduled_burns_reject_overlapping_intervals() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        TranslationControlConfig(
            TranslationMode.SCHEDULED_BURN,
            100.0,
            0.01,
            scheduled_burns=(
                ScheduledBurn(1_000_000_000, 2_000_000_000, (0.001, 0.0, 0.0)),
                ScheduledBurn(2_000_000_000, 1_000_000_000, (0.001, 0.0, 0.0)),
            ),
        )


def test_rpo_hold_command_has_the_expected_closed_loop_direction() -> None:
    stack = RpoReferenceFlightSoftwareStack(rpo_config())
    stack.boot(boot_event())
    output = stack.step(navigation_batch(1, range_m=1_000.0, rate_m_s=0.0))
    force_x = output.commands[0].payload.force_n[0]  # type: ignore[union-attr]
    # The canonical relative state is deputy minus chief.  A positive radial
    # displacement therefore requires a negative radial correction.
    assert force_x < 0.0
    acceleration = force_x / 100.0
    propagated_relative_x = 1_000.0 + 0.5 * acceleration * 10.0**2
    assert propagated_relative_x < 1_000.0


@pytest.mark.parametrize(
    ("mode", "goal_type", "control_changes"),
    (
        (
            TranslationMode.SCHEDULED_BURN,
            "orbit.scheduled_burn",
            {"scheduled_burns": (ScheduledBurn(0, 1_000_000_000, (0.001, 0.0, 0.0)),)},
        ),
        (
            TranslationMode.STATIONKEEPING,
            "orbit.stationkeeping",
            {"target_state_eci": (7_000_100.0, 0.0, 0.0, 0.0, 7_500.0, 0.0)},
        ),
        (
            TranslationMode.ORBITAL_ELEMENTS,
            "orbital_elements",
            {"target_semi_major_axis_m": 7_100_000.0, "target_eccentricity": 0.001},
        ),
        (
            TranslationMode.ATMOSPHERIC_PASS,
            "orbit.atmospheric_pass",
            {
                "atmospheric_raise_start_ns": 0,
                "atmospheric_raise_end_ns": 1_000_000_000,
                "atmospheric_prograde_acceleration_m_s2": 0.001,
            },
        ),
    ),
)
def test_each_advertised_orbit_mode_has_saturation_infeasible_and_fault_evidence(
    mode: TranslationMode,
    goal_type: str,
    control_changes: dict[str, object],
) -> None:
    control = TranslationControlConfig(mode, 100.0, 1.0e-9, **control_changes)
    config = _orbit_config(control, goal_type)

    saturated = OrbitReferenceFlightSoftwareStack(config)
    saturated.boot(boot_event())
    assert telemetry_fields(saturated.step(batch(1, ideal_event(0, 1))))["control_saturated"] is True

    continuous_allocator = TranslationAllocatorConfig(
        SATELLITE_ID,
        TranslationAllocatorKind.CONTINUOUS_ENGINE,
        "translation",
        ENGINE_FRAME,
        5.0,
    )
    infeasible = OrbitReferenceFlightSoftwareStack(
        replace(
            config,
            navigation_initialization=NavigationInitializationMode.COLD,
            allocator=continuous_allocator,
        )
    )
    infeasible.boot(boot_event())
    infeasible_output = infeasible.step(batch(1, gnss_event(0, 1)))
    assert telemetry_fields(infeasible_output)["translation_allocation_status"] == "infeasible"

    faulted = OrbitReferenceFlightSoftwareStack(config)
    faulted.boot(boot_event())
    faulted.step(batch(1, ideal_event(0, 1)))
    sensor_fault = faulted.step(batch(2, fault_event(0, 2, "ideal-own-state"), ideal_event(1, 2)))
    assert telemetry_fields(sensor_fault)["executive_phase"] == "recovery"
    actuator_fault = faulted.step(
        batch(
            3,
            fault_event(1, 3, "ideal-own-state", active=False),
            fault_event(2, 3, "translation"),
            ideal_event(2, 3),
        )
    )
    assert actuator_fault.commands == ()
