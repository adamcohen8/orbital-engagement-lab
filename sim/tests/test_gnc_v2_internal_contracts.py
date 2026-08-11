from __future__ import annotations

import pytest

from sim.flight_software import ClockScale, ClockTag, FrameId, TelemetryField, ValidityInterval
from sim.gnc import (
    ActiveObjective,
    AllocationResult,
    AllocationStatus,
    BeliefState,
    ConstraintKind,
    ConstraintSet,
    EstimateValidity,
    EvaluatedConstraint,
    ExecutionPolicy,
    GoalState,
    GuidanceReference,
    MissionDecision,
    ObjectiveRole,
    ObjectiveState,
    RequestedEffort,
    RequestedEffortKind,
    StateEstimate,
)


def _time(ticks: int = 1) -> ClockTag:
    return ClockTag("clock", ticks, 1_000_000, ClockScale.ONBOARD)


def _frame() -> FrameId:
    return FrameId("OEL/ECI/J2000", "frames-v1")


def test_navigation_belief_carries_estimates_freshness_and_provenance_not_truth() -> None:
    own_state = StateEstimate(
        "own-orbit",
        _time(),
        _frame(),
        (TelemetryField("position_x_m", 7.0e6, "m"),),
        validity=EstimateValidity.VALID,
    )
    belief = BeliefState(
        generated_at=_time(),
        own_state=own_state,
        freshness=(TelemetryField("own_state_age_s", 0.1, "s"),),
    )
    assert belief.own_state is own_state
    assert not hasattr(belief, "truth")


def test_mission_decision_preserves_primary_goal_during_recovery_objective() -> None:
    constraint = EvaluatedConstraint(
        "keep-out",
        ConstraintKind.MISSION_SAFETY_ENVELOPE,
        _time(),
        satisfied=False,
        margin=-1.0,
    )
    decision = MissionDecision(
        primary_goal_id="rendezvous",
        primary_goal_state=GoalState.SUSPENDED,
        active_objective=ActiveObjective(
            "detumble",
            ObjectiveRole.RECOVERY,
            ObjectiveState.ACTIVE,
            "angular_rate_reduction",
            _time(),
        ),
        constraints=ConstraintSet(_time(), (constraint,)),
        priority=100,
        execution_policy=ExecutionPolicy("detumble_guidance", "b_dot", "magnetorquer"),
    )
    assert decision.primary_goal_id == "rendezvous"
    assert decision.active_objective.role is ObjectiveRole.RECOVERY


def test_guidance_effort_and_allocation_are_typed_separate_handoffs() -> None:
    validity = ValidityInterval(_time(), _time(10))
    reference = GuidanceReference(
        "hold",
        "attitude",
        _frame(),
        validity,
        attitude_quat_from_frame=(1.0, 0.0, 0.0, 0.0),
    )
    effort = RequestedEffort(
        "torque-1",
        RequestedEffortKind.TORQUE,
        _time(),
        _frame(),
        validity,
        torque_n_m=(0.1, 0.0, 0.0),
    )
    allocation = AllocationResult("torque-1", _time(), AllocationStatus.RESIDUAL, residual_torque_n_m=(0.01, 0.0, 0.0))
    assert reference.attitude_quat_from_frame == (1.0, 0.0, 0.0, 0.0)
    assert effort.kind is RequestedEffortKind.TORQUE
    assert allocation.requested_effort_id == effort.effort_id


def test_internal_contracts_reject_invalid_empty_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="required vector"):
        RequestedEffort(
            "force",
            RequestedEffortKind.FORCE,
            _time(),
            _frame(),
            ValidityInterval(_time()),
        )
    with pytest.raises(ValueError, match="normalized"):
        GuidanceReference(
            "attitude",
            "attitude",
            _frame(),
            ValidityInterval(_time()),
            attitude_quat_from_frame=(2.0, 0.0, 0.0, 0.0),
        )
