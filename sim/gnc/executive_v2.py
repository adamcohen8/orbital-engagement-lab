"""Satellite-owned goal, constraint, action, and recovery executive for GNC v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from sim.flight_software.contracts import ClockScale, ClockTag, TimeValidity
from sim.flight_software.loads import ConstraintDefinition, ConstraintKind, GoalDefinition, GoalMode
from sim.gnc.contracts import (
    ActiveObjective,
    ConstraintSet,
    EvaluatedConstraint,
    ExecutionPolicy,
    GoalState,
    MissionDecision,
    ObjectiveRole,
    ObjectiveState,
)


class ExecutivePhase(str, Enum):
    WAITING_NAVIGATION = "waiting_navigation"
    PRIMARY = "primary"
    RECOVERY = "recovery"
    TERMINAL = "terminal"


class ActionKind(str, Enum):
    TIMED = "timed"
    PULSED = "pulsed"
    CONDITION = "condition"


class ActionState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    PREEMPTED = "preempted"


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    action_id: str
    mode: str
    kind: ActionKind
    timeout_s: float | None = None
    duration_s: float | None = None
    pulse_count: int = 1
    condition_id: str | None = None

    def __post_init__(self) -> None:
        if not self.action_id.strip() or not self.mode.strip():
            raise ValueError("action_id and mode must be non-empty")
        if not isinstance(self.kind, ActionKind):
            raise TypeError("kind must be ActionKind")
        for name, value in (("timeout_s", self.timeout_s), ("duration_s", self.duration_s)):
            if value is not None and (not isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.pulse_count, bool) or not isinstance(self.pulse_count, int) or self.pulse_count < 1:
            raise ValueError("pulse_count must be a positive integer")
        if self.kind is ActionKind.TIMED and self.duration_s is None:
            raise ValueError("timed action requires duration_s")
        if self.kind is ActionKind.CONDITION and (self.condition_id is None or not self.condition_id.strip()):
            raise ValueError("condition action requires condition_id")


@dataclass(frozen=True, slots=True)
class ExecutiveObservation:
    navigation_ready: bool
    goal_satisfied: bool
    relative_range_m: float | None = None
    relative_rate_m_s: float | None = None
    active_faults: tuple[tuple[str, str], ...] = ()
    action_conditions: tuple[str, ...] = ()
    mass_kg: float | None = None
    dry_mass_kg: float = 0.0


@dataclass(frozen=True, slots=True)
class GoalProgress:
    goal_id: str
    state: GoalState
    phase: ExecutivePhase
    dwell_elapsed_s: float
    compliant_elapsed_s: float
    excursion_elapsed_s: float
    selected_mode: str | None
    active_action_id: str | None
    action_state: ActionState | None
    preempted: bool


@dataclass(frozen=True, slots=True)
class ExecutiveResult:
    mission_decision: MissionDecision
    progress: GoalProgress
    selected_mode: str | None
    constraints: ConstraintSet
    allow_command: bool


@dataclass(frozen=True, slots=True)
class ReferenceExecutiveConfig:
    primary_goal: GoalDefinition
    primary_mode: str
    constraints: tuple[ConstraintDefinition, ...] = ()
    actions: tuple[ActionDefinition, ...] = ()
    recovery_mode: str = "passive_retreat"
    recover_on_fault: bool = True
    recovery_clear_dwell_s: float = 1.0
    recover_on_action_timeout: bool = True
    recovery_constraint_kinds: tuple[ConstraintKind, ...] = ()

    def __post_init__(self) -> None:
        if not self.primary_mode.strip() or not self.recovery_mode.strip():
            raise ValueError("primary_mode and recovery_mode must be non-empty")
        if not isfinite(self.recovery_clear_dwell_s) or self.recovery_clear_dwell_s < 0.0:
            raise ValueError("recovery_clear_dwell_s must be finite and nonnegative")
        action_ids = [action.action_id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action IDs must be unique")
        constraint_ids = [constraint.constraint_id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint IDs must be unique")


class ReferenceMissionExecutive:
    def __init__(self, config: ReferenceExecutiveConfig) -> None:
        self.config = config
        self._goal_state = GoalState.PENDING
        self._phase = ExecutivePhase.WAITING_NAVIGATION
        self._activated_at: ClockTag | None = None
        self._last_update: ClockTag | None = None
        self._dwell_started: ClockTag | None = None
        self._recovery_clear_started: ClockTag | None = None
        self._pre_recovery_goal_state: GoalState | None = None
        self._pre_recovery_phase: ExecutivePhase | None = None
        self._compliant_elapsed_s = 0.0
        self._excursion_elapsed_s = 0.0
        self._action_index = 0
        self._action_state = ActionState.PENDING if config.actions else None
        self._action_started: ClockTag | None = None
        self._action_pulses = 0

    def update(self, now: ClockTag, observation: ExecutiveObservation) -> ExecutiveResult:
        delta_s = 0.0 if self._last_update is None else max(_elapsed_seconds(self._last_update, now), 0.0)
        self._last_update = now
        constraints = self._evaluate_constraints(now, observation)
        constraint_recovery = any(
            item.satisfied is False and item.kind in self.config.recovery_constraint_kinds
            for item in constraints.constraints
        )
        recovery_required = (self.config.recover_on_fault and bool(observation.active_faults)) or constraint_recovery

        if recovery_required:
            self._enter_recovery(now)
        elif not observation.navigation_ready:
            self._phase = ExecutivePhase.WAITING_NAVIGATION
            self._goal_state = GoalState.PENDING
            self._preempt_action(now)
        elif self._phase is ExecutivePhase.RECOVERY:
            if self._recovery_clear_started is None:
                self._recovery_clear_started = now
            if _elapsed_seconds(self._recovery_clear_started, now) >= self.config.recovery_clear_dwell_s:
                self._leave_recovery(now)
        elif self._phase is ExecutivePhase.TERMINAL:
            # Terminal goal outcomes are latched.  A new goal may only enter
            # through an accepted mission load, which constructs a new
            # executive rather than silently reopening this one.
            pass
        else:
            self._phase = ExecutivePhase.PRIMARY
            if self._activated_at is None:
                self._activated_at = now
            self._goal_state = GoalState.ACTIVE
            self._update_goal(now, delta_s, observation.goal_satisfied)
            self._update_action(now, observation)

        selected_mode = self._selected_mode()
        objective_role = ObjectiveRole.RECOVERY if self._phase is ExecutivePhase.RECOVERY else ObjectiveRole.PRIMARY
        objective_state = (
            ObjectiveState.SUSPENDED
            if self._phase is ExecutivePhase.WAITING_NAVIGATION
            else ObjectiveState.ACTIVE
            if self._phase is ExecutivePhase.RECOVERY
            else ObjectiveState.ACTIVE
            if self._goal_state is GoalState.ACTIVE
            else ObjectiveState.ACHIEVED
            if self._goal_state is GoalState.ACHIEVED
            else ObjectiveState.FAILED
        )
        active_at = self._activated_at or now
        objective = ActiveObjective(
            f"{self.config.primary_goal.goal_id}.{self._phase.value}",
            objective_role,
            objective_state,
            selected_mode or "wait",
            active_at,
            target_frame=self.config.primary_goal.target_frame,
        )
        decision = MissionDecision(
            self.config.primary_goal.goal_id,
            self._goal_state,
            objective,
            constraints,
            100 if self._phase is ExecutivePhase.RECOVERY else 0,
            ExecutionPolicy("reference", "feedback", "configured-allocator"),
        )
        action = self._active_action()
        progress = GoalProgress(
            self.config.primary_goal.goal_id,
            self._goal_state,
            self._phase,
            0.0 if self._dwell_started is None else max(_elapsed_seconds(self._dwell_started, now), 0.0),
            self._compliant_elapsed_s,
            self._excursion_elapsed_s,
            selected_mode,
            None if action is None else action.action_id,
            self._action_state,
            self._phase is ExecutivePhase.RECOVERY,
        )
        return ExecutiveResult(
            decision,
            progress,
            selected_mode,
            constraints,
            observation.navigation_ready
            and selected_mode is not None
            and (self._goal_state is GoalState.ACTIVE or self._phase is ExecutivePhase.RECOVERY),
        )

    def snapshot_state(self) -> dict[str, object]:
        return {
            "goal_state": self._goal_state.value,
            "phase": self._phase.value,
            "activated_at": _clock_state(self._activated_at),
            "last_update": _clock_state(self._last_update),
            "dwell_started": _clock_state(self._dwell_started),
            "recovery_clear_started": _clock_state(self._recovery_clear_started),
            "pre_recovery_goal_state": (
                None if self._pre_recovery_goal_state is None else self._pre_recovery_goal_state.value
            ),
            "pre_recovery_phase": None if self._pre_recovery_phase is None else self._pre_recovery_phase.value,
            "compliant_elapsed_s": self._compliant_elapsed_s,
            "excursion_elapsed_s": self._excursion_elapsed_s,
            "action_index": self._action_index,
            "action_state": None if self._action_state is None else self._action_state.value,
            "action_started": _clock_state(self._action_started),
            "action_pulses": self._action_pulses,
        }

    def restore_state(self, state: dict[str, object]) -> None:
        goal_state = GoalState(str(state["goal_state"]))
        phase = ExecutivePhase(str(state["phase"]))
        action_index = int(state["action_index"])
        if action_index < 0 or action_index > len(self.config.actions):
            raise ValueError("executive snapshot action_index is invalid")
        action_state_value = state.get("action_state")
        self._goal_state = goal_state
        self._phase = phase
        self._activated_at = _clock_from_state(state.get("activated_at"))
        self._last_update = _clock_from_state(state.get("last_update"))
        self._dwell_started = _clock_from_state(state.get("dwell_started"))
        self._recovery_clear_started = _clock_from_state(state.get("recovery_clear_started"))
        pre_recovery_goal_state = state.get("pre_recovery_goal_state")
        pre_recovery_phase = state.get("pre_recovery_phase")
        self._pre_recovery_goal_state = (
            None if pre_recovery_goal_state is None else GoalState(str(pre_recovery_goal_state))
        )
        self._pre_recovery_phase = None if pre_recovery_phase is None else ExecutivePhase(str(pre_recovery_phase))
        self._compliant_elapsed_s = _nonnegative_float(state["compliant_elapsed_s"])
        self._excursion_elapsed_s = _nonnegative_float(state["excursion_elapsed_s"])
        self._action_index = action_index
        self._action_state = None if action_state_value is None else ActionState(str(action_state_value))
        self._action_started = _clock_from_state(state.get("action_started"))
        self._action_pulses = int(state["action_pulses"])

    def _update_goal(self, now: ClockTag, delta_s: float, satisfied: bool) -> None:
        goal = self.config.primary_goal
        if goal.mode is GoalMode.MAINTENANCE:
            if satisfied:
                self._compliant_elapsed_s += delta_s
            else:
                self._excursion_elapsed_s += delta_s
            if goal.valid_during is not None and goal.valid_during.expires_at is not None:
                if _elapsed_seconds(goal.valid_during.expires_at, now) >= 0.0:
                    self._goal_state = GoalState.ACHIEVED if self._excursion_elapsed_s == 0.0 else GoalState.FAILED
                    self._phase = ExecutivePhase.TERMINAL
            return
        if satisfied:
            if self._dwell_started is None:
                self._dwell_started = now
            if _elapsed_seconds(self._dwell_started, now) >= goal.dwell_s:
                self._goal_state = GoalState.ACHIEVED
                self._phase = ExecutivePhase.TERMINAL
        else:
            self._dwell_started = None

    def _update_action(self, now: ClockTag, observation: ExecutiveObservation) -> None:
        action = self._active_action()
        if action is None or self._goal_state is not GoalState.ACTIVE:
            return
        if self._action_state is ActionState.TIMED_OUT:
            return
        if self._action_state in (ActionState.PENDING, ActionState.PREEMPTED):
            self._action_state = ActionState.ACTIVE
            self._action_started = now
        if self._action_started is None:
            self._action_started = now
        elapsed = _elapsed_seconds(self._action_started, now)
        if action.timeout_s is not None and elapsed >= action.timeout_s:
            self._action_state = ActionState.TIMED_OUT
            if self.config.recover_on_action_timeout:
                self._enter_recovery(now)
            return
        completed = False
        if action.kind is ActionKind.TIMED:
            completed = action.duration_s is not None and elapsed >= action.duration_s
        elif action.kind is ActionKind.PULSED:
            self._action_pulses += 1
            completed = self._action_pulses >= action.pulse_count
        else:
            completed = action.condition_id in observation.action_conditions
        if completed:
            self._action_state = ActionState.COMPLETED
            self._action_index += 1
            self._action_started = None
            self._action_pulses = 0
            self._action_state = ActionState.PENDING if self._active_action() is not None else None

    def _enter_recovery(self, now: ClockTag) -> None:
        if self._phase is not ExecutivePhase.RECOVERY:
            self._pre_recovery_goal_state = self._goal_state
            self._pre_recovery_phase = self._phase
        self._phase = ExecutivePhase.RECOVERY
        self._goal_state = GoalState.SUSPENDED
        self._recovery_clear_started = None
        self._preempt_action(now)

    def _leave_recovery(self, now: ClockTag) -> None:
        if self._pre_recovery_phase is ExecutivePhase.TERMINAL and self._pre_recovery_goal_state in (
            GoalState.ACHIEVED,
            GoalState.FAILED,
        ):
            self._phase = ExecutivePhase.TERMINAL
            self._goal_state = self._pre_recovery_goal_state
        else:
            self._phase = ExecutivePhase.PRIMARY
            self._goal_state = GoalState.ACTIVE
            self._resume_action(now)
        self._recovery_clear_started = None
        self._pre_recovery_goal_state = None
        self._pre_recovery_phase = None

    def _preempt_action(self, _now: ClockTag) -> None:
        if self._active_action() is not None and self._action_state is ActionState.ACTIVE:
            self._action_state = ActionState.PREEMPTED

    def _resume_action(self, now: ClockTag) -> None:
        if self._active_action() is not None and self._action_state is ActionState.PREEMPTED:
            self._action_state = ActionState.ACTIVE
            self._action_started = now

    def _selected_mode(self) -> str | None:
        if self._phase is ExecutivePhase.WAITING_NAVIGATION or self._phase is ExecutivePhase.TERMINAL:
            return None
        if self._phase is ExecutivePhase.RECOVERY:
            return self.config.recovery_mode
        action = self._active_action()
        return self.config.primary_mode if action is None or self._action_state is ActionState.TIMED_OUT else action.mode

    def _active_action(self) -> ActionDefinition | None:
        return self.config.actions[self._action_index] if self._action_index < len(self.config.actions) else None

    def _evaluate_constraints(self, now: ClockTag, observation: ExecutiveObservation) -> ConstraintSet:
        evaluated: list[EvaluatedConstraint] = []
        for constraint in self.config.constraints:
            if not constraint.enabled or (
                constraint.applies_to_goal_ids
                and self.config.primary_goal.goal_id not in constraint.applies_to_goal_ids
            ):
                continue
            parameters = {field.name: field.value for field in constraint.parameters}
            satisfied, margin = _evaluate_constraint(constraint.evaluator_id, parameters, observation)
            evaluated.append(
                EvaluatedConstraint(
                    constraint.constraint_id,
                    constraint.kind,
                    now,
                    satisfied,
                    margin,
                )
            )
        return ConstraintSet(now, tuple(evaluated))


def _evaluate_constraint(
    evaluator_id: str,
    parameters: dict[str, object],
    observation: ExecutiveObservation,
) -> tuple[bool | None, float | None]:
    if evaluator_id == "navigation_available":
        return observation.navigation_ready, 1.0 if observation.navigation_ready else -1.0
    if evaluator_id == "minimum_range_m":
        if observation.relative_range_m is None:
            return None, None
        minimum = float(parameters.get("minimum_m", 0.0))
        margin = observation.relative_range_m - minimum
        return margin >= 0.0, margin
    if evaluator_id == "maximum_range_m":
        if observation.relative_range_m is None:
            return None, None
        maximum = float(parameters.get("maximum_m", 0.0))
        margin = maximum - observation.relative_range_m
        return margin >= 0.0, margin
    if evaluator_id == "maximum_rate_m_s":
        if observation.relative_rate_m_s is None:
            return None, None
        maximum = float(parameters.get("maximum_m_s", 0.0))
        margin = maximum - abs(observation.relative_rate_m_s)
        return margin >= 0.0, margin
    if evaluator_id == "maximum_closing_speed_m_s":
        if observation.relative_rate_m_s is None:
            return None, None
        maximum = float(parameters.get("maximum_m_s", 0.0))
        closing_speed = max(-observation.relative_rate_m_s, 0.0)
        margin = maximum - closing_speed
        return margin >= 0.0, margin
    if evaluator_id == "no_active_faults":
        count = len(observation.active_faults)
        return count == 0, float(-count)
    if evaluator_id == "maximum_active_faults":
        maximum = int(parameters.get("maximum_count", 0))
        margin = maximum - len(observation.active_faults)
        return margin >= 0, float(margin)
    if evaluator_id == "minimum_mass_kg":
        if observation.mass_kg is None:
            return None, None
        minimum = float(parameters.get("minimum_kg", 0.0))
        margin = observation.mass_kg - minimum
        return margin >= 0.0, margin
    if evaluator_id == "minimum_propellant_kg":
        if observation.mass_kg is None:
            return None, None
        minimum = float(parameters.get("minimum_kg", 0.0))
        margin = max(observation.mass_kg - observation.dry_mass_kg, 0.0) - minimum
        return margin >= 0.0, margin
    if evaluator_id == "condition_asserted":
        condition_id = str(parameters.get("condition_id", ""))
        satisfied = bool(condition_id) and condition_id in observation.action_conditions
        return satisfied, 1.0 if satisfied else -1.0
    if evaluator_id == "always":
        return bool(parameters.get("satisfied", True)), None
    return None, None


def _elapsed_seconds(start: ClockTag, end: ClockTag) -> float:
    if (start.clock_id, start.tick_period_ns, start.scale, start.reset_counter) != (
        end.clock_id,
        end.tick_period_ns,
        end.scale,
        end.reset_counter,
    ):
        raise ValueError("executive clocks must share a domain")
    return (end.ticks - start.ticks) * start.tick_period_ns * 1.0e-9


def _clock_state(value: ClockTag | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "clock_id": value.clock_id,
        "ticks": value.ticks,
        "tick_period_ns": value.tick_period_ns,
        "scale": value.scale.value,
        "validity": value.validity.value,
        "reset_counter": value.reset_counter,
    }


def _clock_from_state(value: object) -> ClockTag | None:
    if value is None:
        return None
    mapping = dict(value)  # type: ignore[arg-type]
    return ClockTag(
        str(mapping["clock_id"]),
        int(mapping["ticks"]),
        int(mapping["tick_period_ns"]),
        ClockScale(str(mapping["scale"])),
        TimeValidity(str(mapping["validity"])),
        int(mapping["reset_counter"]),
    )


def _nonnegative_float(value: object) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError("executive snapshot duration is invalid")
    return result
