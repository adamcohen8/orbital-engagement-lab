from __future__ import annotations

import pytest

from sim.flight_software import GoalMode, ValidityInterval
from sim.gnc.contracts import GoalState
from sim.gnc.executive_v2 import ExecutiveObservation, ReferenceExecutiveConfig, ReferenceMissionExecutive
from sim.tests.fsw_v2_helpers import clock
from sim.tests.fsw_v2_orbit_helpers import goal


def _observation(satisfied: bool) -> ExecutiveObservation:
    return ExecutiveObservation(True, satisfied)


def test_terminal_goal_requires_continuous_configured_dwell() -> None:
    executive = ReferenceMissionExecutive(ReferenceExecutiveConfig(goal(dwell_s=0.2), "ric_hold"))
    assert executive.update(clock(1), _observation(True)).progress.state is GoalState.ACTIVE
    executive.update(clock(2), _observation(False))
    assert executive.update(clock(3), _observation(True)).progress.state is GoalState.ACTIVE
    assert executive.update(clock(5), _observation(True)).progress.state is GoalState.ACHIEVED


def test_terminal_goal_outcome_remains_latched_on_later_ticks() -> None:
    executive = ReferenceMissionExecutive(ReferenceExecutiveConfig(goal(), "ric_hold"))
    achieved = executive.update(clock(1), _observation(True))
    assert achieved.progress.state is GoalState.ACHIEVED
    assert achieved.selected_mode is None

    later = executive.update(clock(2), _observation(False))
    assert later.progress.state is GoalState.ACHIEVED
    assert later.selected_mode is None
    assert not later.allow_command


def test_terminal_goal_outcome_survives_a_later_recovery_procedure() -> None:
    executive = ReferenceMissionExecutive(
        ReferenceExecutiveConfig(goal(), "ric_hold", recovery_clear_dwell_s=0.0)
    )
    assert executive.update(clock(1), _observation(True)).progress.state is GoalState.ACHIEVED
    recovery = executive.update(
        clock(2),
        ExecutiveObservation(True, False, active_faults=(("wheel", "failed"),)),
    )
    assert recovery.progress.phase.value == "recovery"

    restored = executive.update(clock(3), _observation(False))
    assert restored.progress.state is GoalState.ACHIEVED
    assert restored.progress.phase.value == "terminal"
    assert restored.selected_mode is None


def test_maintenance_goal_accumulates_compliance_and_closes_at_interval_end() -> None:
    interval = ValidityInterval(clock(1), clock(4))
    executive = ReferenceMissionExecutive(
        ReferenceExecutiveConfig(goal(mode=GoalMode.MAINTENANCE, valid_during=interval), "ric_hold")
    )
    executive.update(clock(1), _observation(True))
    executive.update(clock(2), _observation(True))
    result = executive.update(clock(4), _observation(True))
    assert result.progress.state is GoalState.ACHIEVED
    assert result.progress.compliant_elapsed_s == pytest.approx(0.3)
    assert result.progress.excursion_elapsed_s == 0.0


def test_maintenance_excursion_is_recorded_and_fails_declared_interval() -> None:
    interval = ValidityInterval(clock(1), clock(3))
    executive = ReferenceMissionExecutive(
        ReferenceExecutiveConfig(goal(mode=GoalMode.MAINTENANCE, valid_during=interval), "ric_hold")
    )
    executive.update(clock(1), _observation(True))
    executive.update(clock(2), _observation(False))
    result = executive.update(clock(3), _observation(True))
    assert result.progress.state is GoalState.FAILED
    assert result.progress.excursion_elapsed_s == 0.1
