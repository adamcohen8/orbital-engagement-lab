from __future__ import annotations

from dataclasses import replace
from math import cos, sin

import pytest

from sim.flight_software import (
    ActuatorTelemetryPayload,
    IdealWrenchCommand,
    InputEvent,
    InputKind,
    PacketId,
    Quality,
    RpoReferenceFlightSoftwareStack,
    TelemetryField,
    ThrusterOnOffCommand,
    ThrusterPulseCommand,
    canonical_json_bytes,
)
from sim.gnc.attitude_v2 import (
    AttitudeAllocatorConfig,
    AttitudeAllocatorKind,
    AttitudeReferenceConfig,
    AttitudeReferenceMode,
)
from sim.gnc.navigation_v2 import NavigationInitializationMode
from sim.gnc.orbit_v2 import (
    RcsThrusterBelief,
    TranslationAllocatorConfig,
    TranslationAllocatorKind,
    TranslationControlLaw,
    TranslationMode,
)
from sim.tests.fsw_v2_helpers import ACTUATOR_FRAME, BOOT_ID, SATELLITE_ID, batch, boot_event, clock, ideal_event
from sim.tests.fsw_v2_orbit_helpers import (
    ENGINE_FRAME,
    fault_event,
    gnss_event,
    navigation_batch,
    relative_event,
    rpo_config,
    safety_constraint,
    telemetry_fields,
)


@pytest.mark.parametrize(
    "mode",
    (
        TranslationMode.RIC_HOLD,
        TranslationMode.R_BAR_APPROACH,
        TranslationMode.V_BAR_APPROACH,
        TranslationMode.C_BAR_APPROACH,
        TranslationMode.WAYPOINT,
        TranslationMode.RIC_PD_TRANSFER,
        TranslationMode.TERMINAL_BRAKING,
        TranslationMode.PASSIVE_RETREAT,
    ),
)
def test_every_advertised_rpo_mode_has_a_nominal_typed_command(mode: TranslationMode) -> None:
    waypoints = ((500.0, 0.0, 0.0, 0.0, 0.0, 0.0),) if mode is TranslationMode.WAYPOINT else ()
    stack = RpoReferenceFlightSoftwareStack(rpo_config(mode, waypoints=waypoints))
    stack.boot(boot_event())
    output = stack.step(navigation_batch(1))
    assert output.commands
    assert isinstance(output.commands[-1].payload, IdealWrenchCommand)
    fields = telemetry_fields(output)
    assert fields["selected_mode"] == mode.value
    assert fields["translation_allocation_status"] in ("exact", "saturated")


def test_nonzero_hold_commands_back_toward_the_target_relative_state() -> None:
    stack = RpoReferenceFlightSoftwareStack(rpo_config(TranslationMode.RIC_HOLD))
    stack.boot(boot_event())
    output = stack.step(navigation_batch(1, range_m=1_000.0))
    command = output.commands[-1].payload
    assert isinstance(command, IdealWrenchCommand)
    assert command.force_n[0] < 0.0


def test_translation_stack_does_not_command_from_stale_sample_hold_navigation() -> None:
    stack = RpoReferenceFlightSoftwareStack(replace(rpo_config(), measurement_stale_after_s=0.1))
    stack.boot(boot_event())
    assert stack.step(navigation_batch(1)).commands

    stale = stack.step(batch(3))

    assert stale.commands == ()
    assert telemetry_fields(stale)["navigation_ready"] is False


def test_passive_retreat_commands_outward_from_the_chief() -> None:
    stack = RpoReferenceFlightSoftwareStack(rpo_config(TranslationMode.PASSIVE_RETREAT))
    stack.boot(boot_event())
    output = stack.step(navigation_batch(1, range_m=100.0))
    command = output.commands[-1].payload
    assert isinstance(command, IdealWrenchCommand)
    assert command.force_n[0] > 0.0


def test_rcs_allocator_rejects_duplicate_physical_thruster_ids() -> None:
    thruster = RcsThrusterBelief("jet", (1.0, 0.0, 0.0), 1.0)
    with pytest.raises(ValueError, match="thruster IDs must be unique"):
        replace(
            rpo_config().allocator,
            kind=TranslationAllocatorKind.RCS_PULSE,
            rcs_thrusters=(thruster, thruster),
        )


def test_live_game_navigation_fast_path_preserves_rpo_outputs() -> None:
    config = rpo_config(TranslationMode.PASSIVE_RETREAT)
    optimized = RpoReferenceFlightSoftwareStack(config, _live_navigation_fast_path=True)
    audit_path = RpoReferenceFlightSoftwareStack(config)
    optimized.boot(boot_event())
    audit_path.boot(boot_event())

    for invocation in range(1, 201):
        current = navigation_batch(invocation, range_m=100.0 + invocation)
        assert optimized.step(current) == audit_path.step(current)

    assert len(optimized._navigator._own_packets) == 1
    assert len(audit_path._navigator._own_packets) == 200


@pytest.mark.parametrize(
    "law",
    (
        TranslationControlLaw.REFERENCE_PD,
        TranslationControlLaw.HCW_LQR,
        TranslationControlLaw.CURVILINEAR_RIC_PD,
        TranslationControlLaw.RMOE_IF_THEN,
    ),
)
def test_each_cataloged_relative_control_law_is_selectable_in_the_complete_stack(
    law: TranslationControlLaw,
) -> None:
    base = rpo_config(TranslationMode.RIC_HOLD)
    stack = RpoReferenceFlightSoftwareStack(replace(base, control=replace(base.control, control_law=law)))
    stack.boot(boot_event())
    output = stack.step(navigation_batch(1, range_m=1_000.0))
    assert output.commands
    assert telemetry_fields(output)["control_law"] == law.value


@pytest.mark.parametrize(("cross_track_los", "expected_force_sign"), ((1.0, -1.0), (-1.0, 1.0)))
def test_subsecond_hcw_lqr_commands_against_cross_track_error(
    cross_track_los: float,
    expected_force_sign: float,
) -> None:
    base = rpo_config(TranslationMode.RIC_HOLD)
    stack = RpoReferenceFlightSoftwareStack(
        replace(
            base,
            control=replace(
                base.control,
                control_law=TranslationControlLaw.HCW_LQR,
                control_design_dt_s=0.1,
                mean_motion_rad_s=0.0010780076,
            ),
        )
    )
    stack.boot(boot_event())
    output = stack.step(
        batch(
            1,
            ideal_event(0, 1),
            relative_event(0, 1, range_m=100.0, los=(0.0, 0.0, cross_track_los)),
        )
    )
    command = output.commands[-1].payload
    assert isinstance(command, IdealWrenchCommand)
    assert expected_force_sign * command.force_n[2] > 0.0


@pytest.mark.parametrize(
    "mode",
    (
        TranslationMode.RIC_HOLD,
        TranslationMode.R_BAR_APPROACH,
        TranslationMode.V_BAR_APPROACH,
        TranslationMode.C_BAR_APPROACH,
        TranslationMode.WAYPOINT,
        TranslationMode.RIC_PD_TRANSFER,
        TranslationMode.TERMINAL_BRAKING,
        TranslationMode.PASSIVE_RETREAT,
    ),
)
def test_each_rpo_mode_has_saturation_infeasibility_and_fault_evidence(mode: TranslationMode) -> None:
    waypoints = ((500.0, 0.0, 0.0, 0.0, 0.0, 0.0),) if mode is TranslationMode.WAYPOINT else ()
    saturated = RpoReferenceFlightSoftwareStack(rpo_config(mode, max_acceleration_m_s2=1e-7, waypoints=waypoints))
    saturated.boot(boot_event())
    assert telemetry_fields(saturated.step(navigation_batch(1)))["control_saturated"] is True

    continuous = rpo_config(
        mode,
        initialization=NavigationInitializationMode.COLD,
        allocator_kind=TranslationAllocatorKind.CONTINUOUS_ENGINE,
        waypoints=waypoints,
    )
    infeasible = RpoReferenceFlightSoftwareStack(continuous)
    infeasible.boot(boot_event())
    output = infeasible.step(batch(1, gnss_event(0, 1), relative_event(0, 1)))
    assert telemetry_fields(output)["translation_allocation_status"] == "infeasible"

    faulted = RpoReferenceFlightSoftwareStack(rpo_config(mode, waypoints=waypoints))
    faulted.boot(boot_event())
    faulted.step(navigation_batch(1))
    sensor_fault = faulted.step(batch(2, fault_event(0, 2, "relative"), relative_event(1, 2)))
    assert telemetry_fields(sensor_fault)["executive_phase"] == "recovery"
    actuator_fault = faulted.step(batch(3, fault_event(1, 3, "translation"), relative_event(2, 3)))
    assert actuator_fault.commands == ()


def test_rpo_saturation_and_infeasible_allocation_are_explicit_evidence() -> None:
    saturated = RpoReferenceFlightSoftwareStack(rpo_config(max_acceleration_m_s2=1e-5, max_force_n=1e-4))
    saturated.boot(boot_event())
    saturated_output = saturated.step(navigation_batch(1, range_m=10_000.0))
    fields = telemetry_fields(saturated_output)
    assert fields["control_saturated"] is True
    assert fields["translation_allocation_status"] == "saturated"

    config = rpo_config(allocator_kind=TranslationAllocatorKind.CONTINUOUS_ENGINE)
    cold_config = replace(config, navigation_initialization=NavigationInitializationMode.COLD)
    infeasible = RpoReferenceFlightSoftwareStack(cold_config)
    infeasible.boot(boot_event())
    output = infeasible.step(batch(1, gnss_event(0, 1), relative_event(0, 1)))
    assert output.commands == ()
    assert telemetry_fields(output)["navigation_ready"] is True
    assert telemetry_fields(output)["translation_allocation_status"] == "infeasible"


@pytest.mark.parametrize(
    ("kind", "payload_type"),
    (
        (TranslationAllocatorKind.RCS_PULSE, ThrusterPulseCommand),
        (TranslationAllocatorKind.RCS_ON_OFF, ThrusterOnOffCommand),
    ),
)
def test_rcs_pulse_and_on_off_allocation_publish_device_commands(
    kind: TranslationAllocatorKind,
    payload_type: type[object],
) -> None:
    base = rpo_config()
    thrusters = (
        RcsThrusterBelief("plus-x", (1.0, 0.0, 0.0), 2.0),
        RcsThrusterBelief("minus-x", (-1.0, 0.0, 0.0), 2.0),
        RcsThrusterBelief("plus-y", (0.0, 1.0, 0.0), 2.0),
        RcsThrusterBelief("minus-y", (0.0, -1.0, 0.0), 2.0),
        RcsThrusterBelief("plus-z", (0.0, 0.0, 1.0), 2.0),
        RcsThrusterBelief("minus-z", (0.0, 0.0, -1.0), 2.0),
    )
    allocator = TranslationAllocatorConfig(
        SATELLITE_ID,
        kind,
        "translation",
        ENGINE_FRAME,
        12.0,
        rcs_thrusters=thrusters,
        pulse_window_s=0.5,
    )
    stack = RpoReferenceFlightSoftwareStack(replace(base, allocator=allocator))
    stack.boot(boot_event())
    output = stack.step(navigation_batch(1))
    assert output.commands
    assert all(isinstance(command.payload, payload_type) for command in output.commands)
    if kind is TranslationAllocatorKind.RCS_PULSE:
        assert any(field.name.startswith("requested_impulse") for field in output.telemetry[0].fields)
    else:
        # On/off realization uses full rated thrust, not the fractional force
        # used to select the thruster set, so the residual is explicit.
        assert telemetry_fields(output)["translation_allocation_status"] == "saturated"


def test_rcs_allocation_obeys_the_configured_stack_force_limit() -> None:
    base = rpo_config()
    thrusters = (
        RcsThrusterBelief("plus-x", (1.0, 0.0, 0.0), 2.0),
        RcsThrusterBelief("minus-x", (-1.0, 0.0, 0.0), 2.0),
    )
    allocator = TranslationAllocatorConfig(
        SATELLITE_ID,
        TranslationAllocatorKind.RCS_PULSE,
        "translation",
        ENGINE_FRAME,
        0.01,
        rcs_thrusters=thrusters,
        pulse_window_s=0.5,
    )
    stack = RpoReferenceFlightSoftwareStack(replace(base, allocator=allocator))
    stack.boot(boot_event())
    output = stack.step(navigation_batch(1))
    fields = telemetry_fields(output)

    assert fields["allocator_requested_force_n"] > 0.01
    assert fields["translation_allocation_status"] == "saturated"
    requested_impulse = sum(
        float(field.value) for field in output.telemetry[0].fields if field.name.startswith("requested_impulse")
    )
    assert requested_impulse <= 0.01 * allocator.pulse_window_s + 1.0e-12


def test_slew_then_burn_gates_translation_until_pointing_is_compliant() -> None:
    attitude_allocator = AttitudeAllocatorConfig(
        SATELLITE_ID,
        AttitudeAllocatorKind.IDEAL_WRENCH,
        "attitude",
        ACTUATOR_FRAME,
    )
    stack = RpoReferenceFlightSoftwareStack(
        rpo_config(attitude_allocator=attitude_allocator, pointing_tolerance_rad=0.02)
    )
    stack.boot(boot_event())
    half = 0.5 * 1.5707963267948966
    misaligned = stack.step(
        batch(
            1,
            ideal_event(0, 1, quaternion=(cos(half), 0.0, 0.0, sin(half))),
            relative_event(0, 1),
        )
    )
    assert {command.actuator_id for command in misaligned.commands} == {"attitude"}
    assert telemetry_fields(misaligned)["pointing_compliant"] is False

    # The corrected deputy-relative-chief convention commands -R for this
    # state, so body +X is aligned by a 180-degree yaw.
    aligned = stack.step(batch(2, ideal_event(1, 2, quaternion=(0.0, 0.0, 0.0, 1.0)), relative_event(1, 2)))
    assert {command.actuator_id for command in aligned.commands} == {"attitude", "translation"}
    assert telemetry_fields(aligned)["pointing_compliant"] is True


def test_slew_then_burn_uses_the_configured_body_thrust_axis() -> None:
    attitude_allocator = AttitudeAllocatorConfig(
        SATELLITE_ID,
        AttitudeAllocatorKind.IDEAL_WRENCH,
        "attitude",
        ACTUATOR_FRAME,
    )
    base = rpo_config(attitude_allocator=attitude_allocator, pointing_tolerance_rad=0.02)
    stack = RpoReferenceFlightSoftwareStack(
        replace(
            base,
            attitude_reference=AttitudeReferenceConfig(
                AttitudeReferenceMode.QUATERNION,
                boresight_body=(0.0, 0.0, 1.0),
            ),
        )
    )
    stack.boot(boot_event())

    # This quaternion aligns body +X with the requested -R force. The same
    # attitude must remain gated when the configured physical thrust axis is +Z.
    output = stack.step(batch(1, ideal_event(0, 1, quaternion=(0.0, 0.0, 0.0, 1.0)), relative_event(0, 1)))

    assert {command.actuator_id for command in output.commands} == {"attitude"}
    assert telemetry_fields(output)["pointing_compliant"] is False


def test_safety_review_violation_does_not_modify_commands_unless_stack_configures_recovery() -> None:
    nominal_config = rpo_config()
    reviewed_config = replace(
        nominal_config,
        executive=replace(
            nominal_config.executive,
            constraints=(safety_constraint(2_000.0),),
            recovery_constraint_kinds=(),
        ),
    )
    nominal = RpoReferenceFlightSoftwareStack(nominal_config)
    reviewed = RpoReferenceFlightSoftwareStack(reviewed_config)
    nominal.boot(boot_event())
    reviewed.boot(boot_event())
    nominal_output = nominal.step(navigation_batch(1))
    reviewed_output = reviewed.step(navigation_batch(1))
    assert canonical_json_bytes(nominal_output.commands[0].payload) == canonical_json_bytes(
        reviewed_output.commands[0].payload
    )
    assert telemetry_fields(reviewed_output)["safety_review_violation_count"] == 1


def test_sensor_and_actuator_fault_evidence_is_stack_owned() -> None:
    stack = RpoReferenceFlightSoftwareStack(rpo_config())
    stack.boot(boot_event())
    stack.step(navigation_batch(1))
    recovery = stack.step(batch(2, fault_event(0, 2, "relative"), relative_event(1, 2)))
    assert telemetry_fields(recovery)["executive_phase"] == "recovery"
    assert telemetry_fields(recovery)["selected_mode"] == "passive_retreat"

    faulted_actuator = stack.step(
        batch(
            3,
            fault_event(1, 3, "relative", active=False),
            fault_event(2, 3, "translation"),
            ideal_event(2, 3),
            relative_event(2, 3),
        )
    )
    assert faulted_actuator.commands == ()
    assert telemetry_fields(faulted_actuator)["executive_phase"] == "recovery"


def test_requested_and_realized_impulse_are_separate_review_fields() -> None:
    base = rpo_config()
    thrusters = (RcsThrusterBelief("minus-x", (-1.0, 0.0, 0.0), 2.0),)
    allocator = TranslationAllocatorConfig(
        SATELLITE_ID,
        TranslationAllocatorKind.RCS_PULSE,
        "translation",
        ENGINE_FRAME,
        2.0,
        rcs_thrusters=thrusters,
        pulse_window_s=0.5,
    )
    stack = RpoReferenceFlightSoftwareStack(replace(base, allocator=allocator))
    stack.boot(boot_event())
    requested = stack.step(navigation_batch(1))
    assert any(field.name.startswith("requested_impulse") for field in requested.telemetry[0].fields)

    time = clock(2)
    realized = InputEvent(
        PacketId("translation-telemetry", BOOT_ID, 0),
        InputKind.ACTUATOR_TELEMETRY,
        time,
        time,
        Quality(),
        ActuatorTelemetryPayload("translation", (TelemetryField("impulse_n_s", 0.12, "N*s"),)),
    )
    reviewed = stack.step(batch(2, ideal_event(1, 2), relative_event(1, 2), realized))
    assert telemetry_fields(reviewed)["realized.translation.impulse_n_s"] == 0.12
