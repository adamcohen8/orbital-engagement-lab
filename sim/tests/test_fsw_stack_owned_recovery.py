from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim.flight_software import (
    InputEvent,
    InputKind,
    MissionLoadManifest,
    OnboardMissionConfigurationLoad,
    PacketId,
    Quality,
    RpoReferenceFlightSoftwareStack,
    TelemetryField,
    TuningTable,
    canonical_json_bytes,
    with_computed_content_hash,
)
from sim.flight_software.loads import ConstraintKind
from sim.gnc.executive_v2 import ActionDefinition, ActionKind, ReferenceExecutiveConfig
from sim.gnc.orbit_v2 import TranslationMode
from sim.tests.fsw_v2_helpers import BOOT_ID, batch, boot_event, clock, ideal_event
from sim.tests.fsw_v2_orbit_helpers import (
    goal,
    navigation_batch,
    relative_event,
    rpo_config,
    safety_constraint,
    telemetry_fields,
)


def test_configured_keep_out_recovery_preempts_primary_and_commands_retreat() -> None:
    base = rpo_config()
    executive = replace(
        base.executive,
        constraints=(safety_constraint(500.0),),
        recovery_constraint_kinds=(ConstraintKind.MISSION_SAFETY_ENVELOPE,),
        recovery_clear_dwell_s=0.0,
    )
    stack = RpoReferenceFlightSoftwareStack(replace(base, executive=executive))
    stack.boot(boot_event())
    recovery = stack.step(navigation_batch(1, range_m=100.0))
    fields = telemetry_fields(recovery)
    assert fields["executive_phase"] == "recovery"
    assert fields["selected_mode"] == "passive_retreat"
    # Positive deputy-relative-chief radial is outward in the canonical RIC
    # convention, so keep-out recovery commands positive radial force.
    assert recovery.commands[0].payload.force_n[0] > 0.0  # type: ignore[union-attr]

    resumed = stack.step(navigation_batch(2, range_m=1_000.0))
    assert telemetry_fields(resumed)["executive_phase"] == "primary"
    assert telemetry_fields(resumed)["selected_mode"] == "ric_hold"


def test_stack_action_timeout_uses_its_configured_recovery_mode() -> None:
    action = ActionDefinition(
        "condition-burn",
        TranslationMode.V_BAR_APPROACH.value,
        ActionKind.CONDITION,
        timeout_s=0.1,
        condition_id="burn-complete",
    )
    executive = ReferenceExecutiveConfig(
        goal(dwell_s=10.0),
        TranslationMode.RIC_HOLD.value,
        actions=(action,),
    )
    stack = RpoReferenceFlightSoftwareStack(rpo_config(executive=executive))
    stack.boot(boot_event())
    assert telemetry_fields(stack.step(navigation_batch(1)))["selected_mode"] == "v_bar_approach"
    timed_out = stack.step(navigation_batch(2))
    assert telemetry_fields(timed_out)["selected_mode"] == "passive_retreat"


def test_typed_mission_load_atomically_replaces_goal_and_reruns_executive_gates() -> None:
    stack = RpoReferenceFlightSoftwareStack(rpo_config())
    stack.boot(boot_event())
    manifest = MissionLoadManifest(
        "rpo-load",
        1,
        "oel.flight_software.mission_load.v1",
        "fsw.rpo_reference",
        ">=2.0.0,<3.0.0",
        "0" * 64,
        clock(1),
    )
    load = with_computed_content_hash(
        OnboardMissionConfigurationLoad(
            manifest,
            goal("rpo.v_bar_approach", dwell_s=0.2),
        )
    )
    time = clock(1)
    load_event = InputEvent(
        PacketId("mission-loader", BOOT_ID, 0),
        InputKind.MISSION_LOAD,
        time,
        time,
        Quality(),
        load,
    )
    output = stack.step(batch(1, load_event, ideal_event(0, 1), relative_event(0, 1)))
    fields = telemetry_fields(output)
    assert fields["mission_load_disposition"] == "accepted"
    assert fields["selected_mode"] == "v_bar_approach"
    snapshot = stack.snapshot()
    assert snapshot.active_load_id == "rpo-load"
    assert snapshot.active_load_revision == 1
    expected = stack.step(navigation_batch(2))
    stack.restore(snapshot)
    replay = stack.step(navigation_batch(2))
    assert canonical_json_bytes(replay.commands) == canonical_json_bytes(expected.commands)


def test_reference_stack_rejects_unimplemented_load_sections_atomically() -> None:
    stack = RpoReferenceFlightSoftwareStack(rpo_config())
    stack.boot(boot_event())
    accepted_manifest = MissionLoadManifest(
        "accepted-load",
        1,
        "oel.flight_software.mission_load.v1",
        "fsw.rpo_reference",
        ">=2.0.0,<3.0.0",
        "0" * 64,
        clock(1),
    )
    accepted = with_computed_content_hash(
        OnboardMissionConfigurationLoad(accepted_manifest, goal("rpo.hold"))
    )
    accepted_time = clock(1)
    accepted_output = stack.step(
        batch(
            1,
            InputEvent(
                PacketId("mission-loader", BOOT_ID, 0),
                InputKind.MISSION_LOAD,
                accepted_time,
                accepted_time,
                Quality(),
                accepted,
            ),
            ideal_event(0, 1),
            relative_event(0, 1),
        )
    )
    assert telemetry_fields(accepted_output)["mission_load_disposition"] == "accepted"

    unsupported_manifest = MissionLoadManifest(
        "unsupported-load",
        2,
        "oel.flight_software.mission_load.v1",
        "fsw.rpo_reference",
        ">=2.0.0,<3.0.0",
        "0" * 64,
        clock(2),
    )
    unsupported = with_computed_content_hash(
        OnboardMissionConfigurationLoad(
            unsupported_manifest,
            goal("rpo.hold"),
            tuning_tables=(TuningTable("guidance", "1", (TelemetryField("kp", 2.0),)),),
        )
    )
    unsupported_time = clock(2)
    rejected_output = stack.step(
        batch(
            2,
            InputEvent(
                PacketId("mission-loader", BOOT_ID, 1),
                InputKind.MISSION_LOAD,
                unsupported_time,
                unsupported_time,
                Quality(),
                unsupported,
            ),
            ideal_event(1, 2),
            relative_event(1, 2),
        )
    )

    assert telemetry_fields(rejected_output)["mission_load_disposition"] == "rejected_by_stack"
    assert stack.snapshot().active_load_id == "accepted-load"
    assert stack.snapshot().active_load_revision == 1


def test_mission_load_goal_parameters_replace_controller_target_atomically() -> None:
    stack = RpoReferenceFlightSoftwareStack(rpo_config())
    stack.boot(boot_event())
    baseline = stack.step(navigation_batch(1))
    baseline_force = np.linalg.norm(baseline.commands[0].payload.force_n)  # type: ignore[union-attr]

    loaded_goal = replace(
        goal("rpo.hold"),
        parameters=(
            TelemetryField("target_r_m", 900.0, "m"),
            TelemetryField("target_i_m", 0.0, "m"),
            TelemetryField("target_c_m", 0.0, "m"),
            TelemetryField("target_dr_m_s", 0.0, "m/s"),
            TelemetryField("target_di_m_s", 0.0, "m/s"),
            TelemetryField("target_dc_m_s", 0.0, "m/s"),
        ),
    )
    manifest = MissionLoadManifest(
        "target-load",
        1,
        "oel.flight_software.mission_load.v1",
        "fsw.rpo_reference",
        ">=2.0.0,<3.0.0",
        "0" * 64,
        clock(2),
    )
    load = with_computed_content_hash(OnboardMissionConfigurationLoad(manifest, loaded_goal))
    time = clock(2)
    load_event = InputEvent(
        PacketId("mission-loader", BOOT_ID, 0),
        InputKind.MISSION_LOAD,
        time,
        time,
        Quality(),
        load,
    )
    loaded = stack.step(batch(2, load_event, ideal_event(1, 2), relative_event(1, 2)))
    loaded_force = np.linalg.norm(loaded.commands[0].payload.force_n)  # type: ignore[union-attr]

    assert telemetry_fields(loaded)["mission_load_disposition"] == "accepted"
    assert loaded_force < baseline_force
