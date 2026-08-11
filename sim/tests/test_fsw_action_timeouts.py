from __future__ import annotations

import pytest

from sim.gnc.executive_v2 import (
    ActionDefinition,
    ActionKind,
    ExecutiveObservation,
    ExecutivePhase,
    ReferenceExecutiveConfig,
    ReferenceMissionExecutive,
)
from sim.tests.fsw_v2_helpers import clock
from sim.tests.fsw_v2_orbit_helpers import goal


@pytest.mark.parametrize(
    "action",
    (
        ActionDefinition("timed", "ric_hold", ActionKind.TIMED, duration_s=0.1),
        ActionDefinition("pulsed", "ric_hold", ActionKind.PULSED, pulse_count=2),
        ActionDefinition("condition", "ric_hold", ActionKind.CONDITION, condition_id="ready"),
    ),
)
def test_timed_pulsed_and_condition_actions_complete_deterministically(action: ActionDefinition) -> None:
    executive = ReferenceMissionExecutive(ReferenceExecutiveConfig(goal(dwell_s=10.0), "ric_hold", actions=(action,)))
    executive.update(clock(1), ExecutiveObservation(True, False))
    conditions = ("ready",) if action.kind is ActionKind.CONDITION else ()
    result = executive.update(clock(2), ExecutiveObservation(True, False, action_conditions=conditions))
    assert result.progress.active_action_id is None
    assert result.selected_mode == "ric_hold"


def test_action_timeout_enters_stack_owned_recovery() -> None:
    action = ActionDefinition(
        "wait-forever",
        "ric_hold",
        ActionKind.CONDITION,
        timeout_s=0.2,
        condition_id="never",
    )
    executive = ReferenceMissionExecutive(
        ReferenceExecutiveConfig(
            goal(dwell_s=10.0),
            "ric_hold",
            actions=(action,),
            recovery_clear_dwell_s=0.0,
        )
    )
    executive.update(clock(1), ExecutiveObservation(True, False))
    result = executive.update(clock(3), ExecutiveObservation(True, False))
    assert result.progress.phase is ExecutivePhase.RECOVERY
    assert result.selected_mode == "passive_retreat"

    resumed = executive.update(clock(4), ExecutiveObservation(True, False))
    assert resumed.progress.phase is ExecutivePhase.PRIMARY
    assert resumed.selected_mode == "ric_hold"
    assert executive.update(clock(5), ExecutiveObservation(True, False)).progress.phase is ExecutivePhase.PRIMARY
