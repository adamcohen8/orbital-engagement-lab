from __future__ import annotations

from dataclasses import replace

from sim.flight_software import (
    MISSION_LOAD_SCHEMA,
    ClockScale,
    ClockTag,
    ConstraintDefinition,
    ConstraintKind,
    GoalDefinition,
    GoalMode,
    MissionLoadDisposition,
    MissionLoadManager,
    MissionLoadManifest,
    OnboardMissionConfigurationLoad,
    RequirementEvaluation,
    SafetyRequirement,
    TelemetryField,
    canonical_json_bytes,
    canonical_loads,
    version_satisfies,
    with_computed_content_hash,
)


def _load(revision: int, *, capability: str = "rendezvous") -> OnboardMissionConfigurationLoad:
    time = ClockTag("clock", revision, 1_000_000, ClockScale.ONBOARD)
    manifest = MissionLoadManifest(
        load_id="mission",
        revision=revision,
        schema_version=MISSION_LOAD_SCHEMA,
        target_stack_id="fsw.reference",
        compatible_stack_versions=">=2.0,<3.0",
        content_hash_sha256="0" * 64,
        created_at=time,
    )
    load = OnboardMissionConfigurationLoad(
        manifest=manifest,
        primary_goal=GoalDefinition(
            "rendezvous",
            "relative_terminal_state",
            GoalMode.TERMINAL,
            parameters=(TelemetryField("range_m", 5.0, "m"),),
            dwell_s=10.0,
        ),
        constraints=(
            ConstraintDefinition(
                "keep-out",
                ConstraintKind.MISSION_SAFETY_ENVELOPE,
                "minimum_range_m",
                parameters=(TelemetryField("radius_m", 2.0, "m"),),
            ),
        ),
        enabled_capabilities=(capability,),
        safety_requirements=(
            SafetyRequirement(
                "keep-out-review",
                "Record keep-out incursions for review",
                RequirementEvaluation.QUANTITATIVE,
                evidence_topics=("relative_state",),
            ),
        ),
    )
    return with_computed_content_hash(load)


def test_mission_load_round_trips_and_version_constraints_are_explicit() -> None:
    load = _load(1)
    assert canonical_loads(canonical_json_bytes(load)) == load
    assert version_satisfies("2.1.0", ">=2.0,<3.0")
    assert not version_satisfies("3.0.0", ">=2.0,<3.0")


def test_atomic_rejection_preserves_previous_revision() -> None:
    manager = MissionLoadManager(stack_id="fsw.reference", stack_version="2.1.0", capabilities=("rendezvous",))
    first = _load(1)
    assert manager.apply(first).accepted

    rejected_by_stack = manager.apply(_load(2), accept=lambda _load: (False, "operator policy"))
    assert rejected_by_stack.disposition is MissionLoadDisposition.REJECTED_BY_STACK
    assert manager.active_load is first

    corrupt = replace(_load(2), primary_goal=replace(_load(2).primary_goal, dwell_s=20.0))
    assert manager.apply(corrupt).disposition is MissionLoadDisposition.REJECTED_HASH
    assert manager.active_load is first

    unsupported = _load(2, capability="unavailable")
    assert manager.apply(unsupported).disposition is MissionLoadDisposition.REJECTED_CAPABILITY
    assert manager.active_load is first


def test_successful_load_activation_is_whole_revision_and_rejects_stale_revision() -> None:
    manager = MissionLoadManager(stack_id="fsw.reference", stack_version="2.1.0", capabilities=("rendezvous",))
    first = _load(1)
    second = _load(2)
    assert manager.apply(first).accepted
    assert manager.apply(second).accepted
    assert manager.active_load is second
    assert manager.apply(first).disposition is MissionLoadDisposition.REJECTED_REVISION
    assert manager.active_load is second
