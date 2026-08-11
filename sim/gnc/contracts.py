"""Typed, public contracts for mission intent and compact GNC decisions.

The simulation engine still exchanges dictionaries at plugin boundaries for
backwards compatibility.  These helpers make the merge order and the standard
review record explicit without changing those established plugin signatures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Mapping

import numpy as np

from sim.flight_software.contracts import (
    ActuatorCommand,
    ClockTag,
    FrameId,
    Matrix,
    PacketId,
    Quaternion,
    TelemetryField,
    ValidityInterval,
    Vector3,
)
from sim.flight_software.loads import ConstraintKind

MISSION_INTENT_PRECEDENCE = (
    "mission_modules",
    "mission_strategy",
    "mission_execution",
)

_CONTRACT_METADATA_KEYS = frozenset(
    {
        "_mission_field_sources",
        "_mission_field_collisions",
        "_mission_precedence",
    }
)


class EstimateValidity(str, Enum):
    VALID = "valid"
    DEGRADED = "degraded"
    INVALID = "invalid"


class GoalState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ACHIEVED = "achieved"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ObjectiveRole(str, Enum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"
    ENABLING = "enabling"
    SAFETY = "safety"
    RECOVERY = "recovery"


class ObjectiveState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ACHIEVED = "achieved"
    FAILED = "failed"
    INFEASIBLE = "infeasible"
    CANCELLED = "cancelled"


class RequestedEffortKind(str, Enum):
    WRENCH = "wrench"
    FORCE = "force"
    TORQUE = "torque"
    ACCELERATION = "acceleration"
    ANGULAR_ACCELERATION = "angular_acceleration"


class AllocationStatus(str, Enum):
    EXACT = "exact"
    RESIDUAL = "residual"
    SATURATED = "saturated"
    INFEASIBLE = "infeasible"
    INVALID = "invalid"


def _finite_tuple(name: str, values: tuple[float, ...], size: int) -> None:
    if len(values) != size or not all(isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain exactly {size} finite values")


@dataclass(frozen=True, slots=True)
class StateEstimate:
    estimate_id: str
    epoch: ClockTag
    frame: FrameId
    values: tuple[TelemetryField, ...]
    covariance_order: tuple[str, ...] = ()
    covariance: Matrix | None = None
    source_packets: tuple[PacketId, ...] = ()
    validity: EstimateValidity = EstimateValidity.VALID

    def __post_init__(self) -> None:
        if not self.estimate_id.strip():
            raise ValueError("estimate_id must be non-empty")
        if not isinstance(self.validity, EstimateValidity):
            raise TypeError("validity must be EstimateValidity")
        if self.covariance is not None:
            if not self.covariance_order or len(self.covariance) != len(self.covariance_order):
                raise ValueError("covariance must match its declared field order")
            if any(len(row) != len(self.covariance_order) for row in self.covariance):
                raise ValueError("covariance must be square")
            if not all(isfinite(float(value)) for row in self.covariance for value in row):
                raise ValueError("covariance must contain finite values")


@dataclass(frozen=True, slots=True)
class BeliefState:
    generated_at: ClockTag
    own_state: StateEstimate | None = None
    attitude_state: StateEstimate | None = None
    tracked_objects: tuple[StateEstimate, ...] = ()
    resource_estimates: tuple[TelemetryField, ...] = ()
    actuator_estimates: tuple[TelemetryField, ...] = ()
    environment_estimates: tuple[TelemetryField, ...] = ()
    health_state: tuple[TelemetryField, ...] = ()
    freshness: tuple[TelemetryField, ...] = ()
    provenance: tuple[PacketId, ...] = ()


@dataclass(frozen=True, slots=True)
class ActiveObjective:
    objective_id: str
    role: ObjectiveRole
    state: ObjectiveState
    objective_type: str
    activated_at: ClockTag
    parameters: tuple[TelemetryField, ...] = ()
    target_frame: FrameId | None = None
    deadline: ClockTag | None = None

    def __post_init__(self) -> None:
        if not self.objective_id.strip() or not self.objective_type.strip():
            raise ValueError("objective_id and objective_type must be non-empty")
        if not isinstance(self.role, ObjectiveRole):
            raise TypeError("role must be ObjectiveRole")
        if not isinstance(self.state, ObjectiveState):
            raise TypeError("state must be ObjectiveState")


@dataclass(frozen=True, slots=True)
class EvaluatedConstraint:
    constraint_id: str
    kind: ConstraintKind
    evaluated_at: ClockTag
    satisfied: bool | None
    margin: float | None = None
    details: tuple[TelemetryField, ...] = ()

    def __post_init__(self) -> None:
        if not self.constraint_id.strip():
            raise ValueError("constraint_id must be non-empty")
        if not isinstance(self.kind, ConstraintKind):
            raise TypeError("kind must be ConstraintKind")
        if self.margin is not None and not isfinite(float(self.margin)):
            raise ValueError("margin must be finite when supplied")


@dataclass(frozen=True, slots=True)
class ConstraintSet:
    evaluated_at: ClockTag
    constraints: tuple[EvaluatedConstraint, ...] = ()

    def __post_init__(self) -> None:
        identities = [constraint.constraint_id for constraint in self.constraints]
        if len(identities) != len(set(identities)):
            raise ValueError("constraint identities must be unique")


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    guidance_id: str
    controller_id: str
    allocator_id: str
    maximum_action_duration_s: float | None = None

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.guidance_id, self.controller_id, self.allocator_id)):
            raise ValueError("execution policy module identifiers must be non-empty")
        if self.maximum_action_duration_s is not None:
            duration = float(self.maximum_action_duration_s)
            if not isfinite(duration) or duration <= 0.0:
                raise ValueError("maximum_action_duration_s must be finite and positive")


@dataclass(frozen=True, slots=True)
class MissionDecision:
    primary_goal_id: str
    primary_goal_state: GoalState
    active_objective: ActiveObjective
    constraints: ConstraintSet
    priority: int
    execution_policy: ExecutionPolicy

    def __post_init__(self) -> None:
        if not self.primary_goal_id.strip():
            raise ValueError("primary_goal_id must be non-empty")
        if not isinstance(self.primary_goal_state, GoalState):
            raise TypeError("primary_goal_state must be GoalState")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")


@dataclass(frozen=True, slots=True)
class GuidanceReference:
    reference_id: str
    reference_type: str
    frame: FrameId
    validity: ValidityInterval
    position_m: Vector3 | None = None
    velocity_m_s: Vector3 | None = None
    attitude_quat_from_frame: Quaternion | None = None
    angular_rate_rad_s: Vector3 | None = None
    parameters: tuple[TelemetryField, ...] = ()

    def __post_init__(self) -> None:
        if not self.reference_id.strip() or not self.reference_type.strip():
            raise ValueError("reference_id and reference_type must be non-empty")
        if (
            all(
                value is None
                for value in (
                    self.position_m,
                    self.velocity_m_s,
                    self.attitude_quat_from_frame,
                    self.angular_rate_rad_s,
                )
            )
            and not self.parameters
        ):
            raise ValueError("guidance reference must contain at least one reference value")
        if self.position_m is not None:
            _finite_tuple("position_m", self.position_m, 3)
        if self.velocity_m_s is not None:
            _finite_tuple("velocity_m_s", self.velocity_m_s, 3)
        if self.angular_rate_rad_s is not None:
            _finite_tuple("angular_rate_rad_s", self.angular_rate_rad_s, 3)
        if self.attitude_quat_from_frame is not None:
            _finite_tuple("attitude_quat_from_frame", self.attitude_quat_from_frame, 4)
            norm = sum(float(value) ** 2 for value in self.attitude_quat_from_frame) ** 0.5
            if abs(norm - 1.0) > 1.0e-10:
                raise ValueError("attitude_quat_from_frame must be normalized within 1e-10")


@dataclass(frozen=True, slots=True)
class RequestedEffort:
    effort_id: str
    kind: RequestedEffortKind
    generated_at: ClockTag
    frame: FrameId
    validity: ValidityInterval
    force_n: Vector3 | None = None
    torque_n_m: Vector3 | None = None
    linear_acceleration_m_s2: Vector3 | None = None
    angular_acceleration_rad_s2: Vector3 | None = None

    def __post_init__(self) -> None:
        if not self.effort_id.strip():
            raise ValueError("effort_id must be non-empty")
        if not isinstance(self.kind, RequestedEffortKind):
            raise TypeError("kind must be RequestedEffortKind")
        expected = {
            RequestedEffortKind.WRENCH: (self.force_n is not None and self.torque_n_m is not None),
            RequestedEffortKind.FORCE: self.force_n is not None,
            RequestedEffortKind.TORQUE: self.torque_n_m is not None,
            RequestedEffortKind.ACCELERATION: self.linear_acceleration_m_s2 is not None,
            RequestedEffortKind.ANGULAR_ACCELERATION: self.angular_acceleration_rad_s2 is not None,
        }[self.kind]
        if not expected:
            raise ValueError(f"{self.kind.value} effort is missing its required vector")
        for name, value in (
            ("force_n", self.force_n),
            ("torque_n_m", self.torque_n_m),
            ("linear_acceleration_m_s2", self.linear_acceleration_m_s2),
            ("angular_acceleration_rad_s2", self.angular_acceleration_rad_s2),
        ):
            if value is not None:
                _finite_tuple(name, value, 3)


@dataclass(frozen=True, slots=True)
class AllocationResult:
    requested_effort_id: str
    generated_at: ClockTag
    status: AllocationStatus
    proposed_commands: tuple[ActuatorCommand, ...] = ()
    residual_force_n: Vector3 = (0.0, 0.0, 0.0)
    residual_torque_n_m: Vector3 = (0.0, 0.0, 0.0)
    status_details: tuple[TelemetryField, ...] = ()

    def __post_init__(self) -> None:
        if not self.requested_effort_id.strip():
            raise ValueError("requested_effort_id must be non-empty")
        if not isinstance(self.status, AllocationStatus):
            raise TypeError("status must be AllocationStatus")
        _finite_tuple("residual_force_n", self.residual_force_n, 3)
        _finite_tuple("residual_torque_n_m", self.residual_torque_n_m, 3)


@dataclass(frozen=True)
class MissionIntentCollision:
    """One top-level intent field replaced by a higher-precedence layer."""

    field: str
    previous_source: str
    winning_source: str


@dataclass(frozen=True)
class MissionIntentEnvelope:
    """Merged mission intent plus auditable field provenance."""

    values: dict[str, Any]
    field_sources: dict[str, str]
    collisions: tuple[MissionIntentCollision, ...] = ()
    precedence: tuple[str, ...] = MISSION_INTENT_PRECEDENCE

    def to_runtime_dict(self) -> dict[str, Any]:
        result = dict(self.values)
        result["_mission_field_sources"] = dict(self.field_sources)
        result["_mission_field_collisions"] = [asdict(item) for item in self.collisions]
        result["_mission_precedence"] = list(self.precedence)
        return result


def merge_mission_intent_layers(
    layers: Iterable[tuple[str, Mapping[str, Any] | None]],
    *,
    precedence: tuple[str, ...] = MISSION_INTENT_PRECEDENCE,
) -> MissionIntentEnvelope:
    """Merge mission output layers in order and record top-level replacements.

    Later layers win, matching the historical ``dict.update`` runtime behavior.
    Reserved contract metadata is regenerated and cannot be injected by a
    plugin layer.
    """

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    collisions: list[MissionIntentCollision] = []
    for source, raw_layer in layers:
        if not raw_layer:
            continue
        for key, value in dict(raw_layer).items():
            if key in _CONTRACT_METADATA_KEYS:
                continue
            if key in values:
                collisions.append(
                    MissionIntentCollision(
                        field=str(key),
                        previous_source=sources[str(key)],
                        winning_source=str(source),
                    )
                )
            values[str(key)] = value
            sources[str(key)] = str(source)
    return MissionIntentEnvelope(
        values=values,
        field_sources=sources,
        collisions=tuple(collisions),
        precedence=tuple(precedence),
    )


@dataclass(frozen=True)
class CommandDecision:
    """Compact, always-on record of one controller/mission command decision."""

    sample_index: int
    time_s: float
    interval_end_time_s: float
    dt_s: float
    object_id: str
    orbit_controller: str | None
    attitude_controller: str | None
    mission_strategy: str | None
    mission_execution: str | None
    mission_phase: str | None
    executive_mode: str | None
    requested_accel_eci_km_s2: list[float]
    applied_accel_eci_km_s2: list[float]
    requested_torque_body_nm: list[float]
    applied_torque_body_nm: list[float]
    requested_accel_norm_km_s2: float
    applied_accel_norm_km_s2: float
    requested_torque_norm_nm: float
    applied_torque_norm_nm: float
    burn_requested: bool
    burn_applied: bool
    alignment_error_rad: float | None
    alignment_ok: bool | None
    saturated: bool
    fuel_depleted: bool
    actuator_limited: bool
    deadline_missed: bool
    gate_reason: str | None
    mission_mode: dict[str, Any] = field(default_factory=dict)
    field_sources: dict[str, str] = field(default_factory=dict)
    collisions: list[dict[str, str]] = field(default_factory=list)
    mode_flags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_command_decision(
    *,
    sample_index: int,
    time_s: float,
    interval_end_time_s: float,
    dt_s: float,
    object_id: str,
    agent: Any,
    mission_intent: Mapping[str, Any] | None,
    command_raw: Any,
    command_applied: Any,
) -> CommandDecision:
    """Build the standard decision row from existing runtime objects."""

    intent = dict(mission_intent or {})
    mission_mode = dict(intent.get("mission_mode", {}) or {})
    flags = dict(getattr(command_applied, "mode_flags", {}) or {})
    raw_accel = _vector3(getattr(command_raw, "thrust_eci_km_s2", None))
    applied_accel = _vector3(getattr(command_applied, "thrust_eci_km_s2", None))
    raw_torque = _vector3(getattr(command_raw, "torque_body_nm", None))
    applied_torque = _vector3(getattr(command_applied, "torque_body_nm", None))
    requested_norm = float(np.linalg.norm(raw_accel))
    applied_norm = float(np.linalg.norm(applied_accel))

    alignment_error = _first_finite(
        mission_mode.get("alignment_angle_rad"),
        mission_mode.get("alignment_error_rad"),
        flags.get("alignment_angle_rad"),
        flags.get("alignment_error_rad"),
    )
    alignment_ok = _first_bool(
        mission_mode.get("alignment_ok"),
        flags.get("alignment_ok"),
        flags.get("attitude_alignment_satisfied"),
    )
    deadline_missed = bool(
        flags.get("orbit_controller_deadline_missed", False) or flags.get("attitude_controller_deadline_missed", False)
    )
    saturation_keys = ("saturated", "allocation_saturated", "rcs_allocation_saturated")
    saturated = any(bool(flags.get(key, False)) for key in saturation_keys)
    limit_keys = (
        "thrust_limited_scale",
        "propellant_limited_scale",
        "allocation_saturated",
        "rcs_allocation_saturated",
    )
    actuator_limited = any(key in flags and flags.get(key) not in (False, None, 1.0) for key in limit_keys)
    fuel_depleted = bool(flags.get("fuel_depleted", False))
    gate_reason = _gate_reason(
        mission_mode=mission_mode,
        flags=flags,
        burn_requested=requested_norm > 1.0e-15,
        burn_applied=applied_norm > 1.0e-15,
        alignment_ok=alignment_ok,
        fuel_depleted=fuel_depleted,
        deadline_missed=deadline_missed,
        actuator_limited=actuator_limited,
    )

    return CommandDecision(
        sample_index=int(sample_index),
        time_s=float(time_s),
        interval_end_time_s=float(interval_end_time_s),
        dt_s=float(dt_s),
        object_id=str(object_id),
        orbit_controller=_component_name(getattr(agent, "orbit_controller", None)),
        attitude_controller=_component_name(getattr(agent, "attitude_controller", None)),
        mission_strategy=_component_name(getattr(agent, "mission_strategy", None)),
        mission_execution=_component_name(getattr(agent, "mission_execution", None)),
        mission_phase=_none_or_str(mission_mode.get("phase")),
        executive_mode=_none_or_str(mission_mode.get("executive_mode")),
        requested_accel_eci_km_s2=raw_accel.tolist(),
        applied_accel_eci_km_s2=applied_accel.tolist(),
        requested_torque_body_nm=raw_torque.tolist(),
        applied_torque_body_nm=applied_torque.tolist(),
        requested_accel_norm_km_s2=requested_norm,
        applied_accel_norm_km_s2=applied_norm,
        requested_torque_norm_nm=float(np.linalg.norm(raw_torque)),
        applied_torque_norm_nm=float(np.linalg.norm(applied_torque)),
        burn_requested=requested_norm > 1.0e-15,
        burn_applied=applied_norm > 1.0e-15,
        alignment_error_rad=alignment_error,
        alignment_ok=alignment_ok,
        saturated=bool(saturated),
        fuel_depleted=fuel_depleted,
        actuator_limited=bool(actuator_limited),
        deadline_missed=deadline_missed,
        gate_reason=gate_reason,
        mission_mode=mission_mode,
        field_sources=dict(intent.get("_mission_field_sources", {}) or {}),
        collisions=[dict(item) for item in list(intent.get("_mission_field_collisions", []) or [])],
        mode_flags=flags,
    )


def _vector3(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=float)
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != 3 or not bool(np.all(np.isfinite(result))):
        return np.zeros(3, dtype=float)
    return result


def _component_name(component: Any) -> str | None:
    return None if component is None else f"{type(component).__module__}.{type(component).__qualname__}"


def _none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        if value is not None:
            return bool(value)
    return None


def _first_finite(*values: Any) -> float | None:
    for value in values:
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return None


def _gate_reason(
    *,
    mission_mode: Mapping[str, Any],
    flags: Mapping[str, Any],
    burn_requested: bool,
    burn_applied: bool,
    alignment_ok: bool | None,
    fuel_depleted: bool,
    deadline_missed: bool,
    actuator_limited: bool,
) -> str | None:
    for source in (mission_mode, flags):
        for key in ("gate_reason", "reason", "burn_reason"):
            value = source.get(key)
            if value:
                return str(value)
    if burn_requested and not burn_applied:
        if fuel_depleted:
            return "fuel_depleted"
        if alignment_ok is False:
            return "attitude_alignment_required"
        if deadline_missed:
            return "controller_deadline_missed"
        return "command_suppressed"
    if actuator_limited:
        return "actuator_limited"
    return None
