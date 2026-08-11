"""Immutable onboard mission-configuration loads and atomic activation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from math import isfinite, sqrt
from typing import Literal, Protocol

from .contracts import ClockTag, FrameId, Matrix, Quaternion, TelemetryField, ValidityInterval, Vector3
from .schemas import canonical_json_bytes, register_record_types

MISSION_LOAD_SCHEMA = "oel.flight_software.mission_load.v1"


def _nonempty(name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must be non-empty")


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unique_nonempty(name: str, values: tuple[str, ...]) -> None:
    for index, value in enumerate(values):
        _nonempty(f"{name}[{index}]", value)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


class GoalMode(str, Enum):
    TERMINAL = "terminal"
    MAINTENANCE = "maintenance"


class ConstraintKind(str, Enum):
    PHYSICAL_CAPABILITY = "physical_capability"
    COMMAND_INTERLOCK = "command_interlock"
    MISSION_SAFETY_ENVELOPE = "mission_safety_envelope"
    PERFORMANCE_REQUIREMENT = "performance_requirement"
    PREFERENCE = "preference"


class RequirementEvaluation(str, Enum):
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"


@dataclass(frozen=True, slots=True)
class MissionLoadManifest:
    load_id: str
    revision: int
    schema_version: str
    target_stack_id: str
    compatible_stack_versions: str
    content_hash_sha256: str
    created_at: ClockTag

    def __post_init__(self) -> None:
        _nonempty("load_id", self.load_id)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer")
        if self.revision < 0:
            raise ValueError("revision must be nonnegative")
        _nonempty("schema_version", self.schema_version)
        _nonempty("target_stack_id", self.target_stack_id)
        _nonempty("compatible_stack_versions", self.compatible_stack_versions)
        if len(self.content_hash_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_hash_sha256
        ):
            raise ValueError("content_hash_sha256 must be a 64-character lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class GoalDefinition:
    goal_id: str
    goal_type: str
    mode: GoalMode
    parameters: tuple[TelemetryField, ...] = ()
    target_frame: FrameId | None = None
    valid_during: ValidityInterval | None = None
    dwell_s: float = 0.0

    def __post_init__(self) -> None:
        _nonempty("goal_id", self.goal_id)
        _nonempty("goal_type", self.goal_type)
        if not isinstance(self.mode, GoalMode):
            raise TypeError("mode must be GoalMode")
        if _finite("dwell_s", self.dwell_s) < 0.0:
            raise ValueError("dwell_s must be nonnegative")


@dataclass(frozen=True, slots=True)
class ConstraintDefinition:
    constraint_id: str
    kind: ConstraintKind
    evaluator_id: str
    parameters: tuple[TelemetryField, ...] = ()
    applies_to_goal_ids: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        _nonempty("constraint_id", self.constraint_id)
        if not isinstance(self.kind, ConstraintKind):
            raise TypeError("kind must be ConstraintKind")
        _nonempty("evaluator_id", self.evaluator_id)
        _unique_nonempty("applies_to_goal_ids", self.applies_to_goal_ids)
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")


@dataclass(frozen=True, slots=True)
class OnboardGeometryRecord:
    component_id: str
    component_frame: FrameId
    parent_frame: FrameId
    translation_parent_m: Vector3
    quat_component_from_parent: Quaternion

    def __post_init__(self) -> None:
        _nonempty("component_id", self.component_id)
        if len(self.translation_parent_m) != 3 or not all(
            isfinite(float(value)) for value in self.translation_parent_m
        ):
            raise ValueError("translation_parent_m must contain three finite values")
        quaternion = tuple(float(value) for value in self.quat_component_from_parent)
        if len(quaternion) != 4 or not all(isfinite(value) for value in quaternion):
            raise ValueError("quat_component_from_parent must contain four finite values")
        if abs(sqrt(sum(value * value for value in quaternion)) - 1.0) > 1.0e-10:
            raise ValueError("quat_component_from_parent must be normalized within 1e-10")


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    component_id: str
    calibration_id: str
    parameters: tuple[TelemetryField, ...]
    covariance: Matrix | None = None

    def __post_init__(self) -> None:
        _nonempty("component_id", self.component_id)
        _nonempty("calibration_id", self.calibration_id)
        if self.covariance is not None:
            size = len(self.covariance)
            if any(len(row) != size for row in self.covariance):
                raise ValueError("calibration covariance must be square")
            if not all(isfinite(float(value)) for row in self.covariance for value in row):
                raise ValueError("calibration covariance must contain finite values")


@dataclass(frozen=True, slots=True)
class TuningTable:
    table_id: str
    table_version: str
    values: tuple[TelemetryField, ...]

    def __post_init__(self) -> None:
        _nonempty("table_id", self.table_id)
        _nonempty("table_version", self.table_version)


@dataclass(frozen=True, slots=True)
class SafetyRequirement:
    requirement_id: str
    statement: str
    evaluation: RequirementEvaluation
    evidence_topics: tuple[str, ...] = ()
    parameters: tuple[TelemetryField, ...] = ()

    def __post_init__(self) -> None:
        _nonempty("requirement_id", self.requirement_id)
        _nonempty("statement", self.statement)
        if not isinstance(self.evaluation, RequirementEvaluation):
            raise TypeError("evaluation must be RequirementEvaluation")
        _unique_nonempty("evidence_topics", self.evidence_topics)


@dataclass(frozen=True, slots=True)
class OnboardMissionConfigurationLoad:
    manifest: MissionLoadManifest
    primary_goal: GoalDefinition
    constraints: tuple[ConstraintDefinition, ...] = ()
    onboard_geometry: tuple[OnboardGeometryRecord, ...] = ()
    calibration: tuple[CalibrationRecord, ...] = ()
    tuning_tables: tuple[TuningTable, ...] = ()
    enabled_capabilities: tuple[str, ...] = ()
    safety_requirements: tuple[SafetyRequirement, ...] = ()
    schema: Literal["oel.flight_software.mission_load.v1"] = MISSION_LOAD_SCHEMA

    def __post_init__(self) -> None:
        if self.manifest.schema_version != self.schema:
            raise ValueError("manifest schema_version must match the mission-load schema")
        _unique_nonempty("enabled_capabilities", self.enabled_capabilities)
        for name, values, attribute in (
            ("constraints", self.constraints, "constraint_id"),
            ("onboard_geometry", self.onboard_geometry, "component_id"),
            ("calibration", self.calibration, "calibration_id"),
            ("tuning_tables", self.tuning_tables, "table_id"),
            ("safety_requirements", self.safety_requirements, "requirement_id"),
        ):
            identities = [getattr(item, attribute) for item in values]
            if len(identities) != len(set(identities)):
                raise ValueError(f"{name} identities must be unique")


class MissionLoadDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED_SCHEMA = "rejected_schema"
    REJECTED_HASH = "rejected_hash"
    REJECTED_TARGET = "rejected_target"
    REJECTED_STACK_VERSION = "rejected_stack_version"
    REJECTED_CAPABILITY = "rejected_capability"
    REJECTED_REVISION = "rejected_revision"
    REJECTED_BY_STACK = "rejected_by_stack"


@dataclass(frozen=True, slots=True)
class MissionLoadResult:
    disposition: MissionLoadDisposition
    load_id: str
    revision: int
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.disposition is MissionLoadDisposition.ACCEPTED


class MissionLoadAcceptance(Protocol):
    def __call__(self, load: OnboardMissionConfigurationLoad) -> bool | tuple[bool, str | None]: ...


class MissionLoadManager:
    """Validate and atomically replace one stack's active mission load."""

    def __init__(self, *, stack_id: str, stack_version: str, capabilities: tuple[str, ...]) -> None:
        _nonempty("stack_id", stack_id)
        _parse_version(stack_version)
        _unique_nonempty("capabilities", capabilities)
        self._stack_id = stack_id
        self._stack_version = stack_version
        self._capabilities = frozenset(capabilities)
        self._active_load: OnboardMissionConfigurationLoad | None = None

    @property
    def active_load(self) -> OnboardMissionConfigurationLoad | None:
        return self._active_load

    def apply(
        self,
        load: OnboardMissionConfigurationLoad,
        *,
        accept: MissionLoadAcceptance | None = None,
    ) -> MissionLoadResult:
        manifest = load.manifest
        rejection = self._preflight(load)
        if rejection is not None:
            return rejection
        if accept is not None:
            response = accept(load)
            accepted, reason = response if isinstance(response, tuple) else (bool(response), None)
            if not accepted:
                return MissionLoadResult(
                    MissionLoadDisposition.REJECTED_BY_STACK,
                    manifest.load_id,
                    manifest.revision,
                    reason or "stack rejected load",
                )
        self._active_load = load
        return MissionLoadResult(MissionLoadDisposition.ACCEPTED, manifest.load_id, manifest.revision)

    def _preflight(self, load: OnboardMissionConfigurationLoad) -> MissionLoadResult | None:
        manifest = load.manifest

        def reject(disposition: MissionLoadDisposition, reason: str) -> MissionLoadResult:
            return MissionLoadResult(disposition, manifest.load_id, manifest.revision, reason)

        if manifest.schema_version != MISSION_LOAD_SCHEMA or load.schema != MISSION_LOAD_SCHEMA:
            return reject(MissionLoadDisposition.REJECTED_SCHEMA, "unsupported mission-load schema")
        if mission_load_content_hash(load) != manifest.content_hash_sha256:
            return reject(MissionLoadDisposition.REJECTED_HASH, "mission-load content hash mismatch")
        if manifest.target_stack_id != self._stack_id:
            return reject(MissionLoadDisposition.REJECTED_TARGET, "mission load targets a different stack")
        if not version_satisfies(self._stack_version, manifest.compatible_stack_versions):
            return reject(MissionLoadDisposition.REJECTED_STACK_VERSION, "stack version is not compatible")
        missing = sorted(set(load.enabled_capabilities) - self._capabilities)
        if missing:
            return reject(MissionLoadDisposition.REJECTED_CAPABILITY, f"unsupported capabilities: {', '.join(missing)}")
        if self._active_load is not None:
            active = self._active_load.manifest
            if manifest.load_id == active.load_id and manifest.revision <= active.revision:
                return reject(MissionLoadDisposition.REJECTED_REVISION, "revision must advance the active load")
        return None


def mission_load_content_hash(load: OnboardMissionConfigurationLoad) -> str:
    """Hash every load field except the self-referential manifest hash."""

    manifest = load.manifest
    payload = {
        "schema": load.schema,
        "manifest": {
            "load_id": manifest.load_id,
            "revision": manifest.revision,
            "schema_version": manifest.schema_version,
            "target_stack_id": manifest.target_stack_id,
            "compatible_stack_versions": manifest.compatible_stack_versions,
            "created_at": manifest.created_at,
        },
        "primary_goal": load.primary_goal,
        "constraints": load.constraints,
        "onboard_geometry": load.onboard_geometry,
        "calibration": load.calibration,
        "tuning_tables": load.tuning_tables,
        "enabled_capabilities": load.enabled_capabilities,
        "safety_requirements": load.safety_requirements,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def with_computed_content_hash(load: OnboardMissionConfigurationLoad) -> OnboardMissionConfigurationLoad:
    digest = mission_load_content_hash(load)
    return replace(load, manifest=replace(load.manifest, content_hash_sha256=digest))


def version_satisfies(version: str, constraint: str) -> bool:
    """Evaluate exact or comma-separated semantic-version comparisons."""

    candidate = _parse_version(version)
    expression = constraint.strip()
    if expression in ("", "*"):
        return True
    for clause in expression.split(","):
        match = re.fullmatch(r"\s*(<=|>=|==|=|<|>)?\s*(\d+(?:\.\d+){0,2})\s*", clause)
        if match is None:
            raise ValueError(f"unsupported stack-version constraint {constraint!r}")
        operator = match.group(1) or "=="
        expected = _parse_version(match.group(2))
        comparison = (candidate > expected) - (candidate < expected)
        if not {
            "<": comparison < 0,
            "<=": comparison <= 0,
            "==": comparison == 0,
            "=": comparison == 0,
            ">=": comparison >= 0,
            ">": comparison > 0,
        }[operator]:
            return False
    return True


def _parse_version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or re.fullmatch(r"\d+(?:\.\d+){0,2}", value.strip()) is None:
        raise ValueError(f"version {value!r} must contain one to three numeric components")
    components = [int(item) for item in value.strip().split(".")]
    return tuple((components + [0, 0])[:3])  # type: ignore[return-value]


MISSION_LOAD_RECORD_TYPES = (OnboardMissionConfigurationLoad,)
register_record_types(MISSION_LOAD_RECORD_TYPES)
