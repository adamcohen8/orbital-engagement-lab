from __future__ import annotations

from sim.gnc.executive_v2 import (
    ActionDefinition,
    ActionKind,
    ActionState,
    ExecutiveObservation,
    ExecutivePhase,
    ReferenceExecutiveConfig,
    ReferenceMissionExecutive,
)
from sim.tests.fsw_v2_helpers import clock
from sim.tests.fsw_v2_orbit_helpers import goal, safety_constraint


def _observation(*, fault: bool = False, satisfied: bool = False) -> ExecutiveObservation:
    return ExecutiveObservation(
        True,
        satisfied,
        relative_range_m=100.0,
        active_faults=(("relative", "failed"),) if fault else (),
    )


def test_fault_preempts_primary_action_then_stack_owned_recovery_resumes_it() -> None:
    action = ActionDefinition("acquire", "v_bar_approach", ActionKind.CONDITION, condition_id="acquired")
    executive = ReferenceMissionExecutive(
        ReferenceExecutiveConfig(
            goal(dwell_s=1.0),
            "ric_hold",
            actions=(action,),
            recovery_clear_dwell_s=0.2,
        )
    )
    primary = executive.update(clock(1), _observation())
    assert primary.progress.phase is ExecutivePhase.PRIMARY
    assert primary.progress.action_state is ActionState.ACTIVE

    recovery = executive.update(clock(2), _observation(fault=True))
    assert recovery.progress.phase is ExecutivePhase.RECOVERY
    assert recovery.progress.preempted
    assert recovery.selected_mode == "passive_retreat"
    assert recovery.progress.action_state is ActionState.PREEMPTED

    assert executive.update(clock(3), _observation()).progress.phase is ExecutivePhase.RECOVERY
    resumed = executive.update(clock(5), _observation())
    assert resumed.progress.phase is ExecutivePhase.PRIMARY
    assert resumed.progress.action_state is ActionState.ACTIVE
    assert resumed.selected_mode == "v_bar_approach"


def test_safety_constraint_is_typed_review_evidence_without_implicit_recovery() -> None:
    executive = ReferenceMissionExecutive(
        ReferenceExecutiveConfig(
            goal(),
            "ric_hold",
            constraints=(safety_constraint(1_000.0),),
            recovery_constraint_kinds=(),
        )
    )
    result = executive.update(clock(1), _observation())
    constraint = result.constraints.constraints[0]
    assert not constraint.satisfied
    assert constraint.kind.value == "mission_safety_envelope"
    assert result.progress.phase is ExecutivePhase.PRIMARY
    assert result.selected_mode == "ric_hold"


def test_navigation_loss_suspends_commanding_without_inventing_a_fallback() -> None:
    executive = ReferenceMissionExecutive(ReferenceExecutiveConfig(goal(), "ric_hold"))
    result = executive.update(clock(1), ExecutiveObservation(False, False))
    assert result.progress.phase is ExecutivePhase.WAITING_NAVIGATION
    assert result.selected_mode is None
    assert not result.allow_command
