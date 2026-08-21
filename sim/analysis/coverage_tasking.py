"""Phase 6 bounded deterministic task selection over proven opportunities."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

COVERAGE_TASKING_CONTRACT_VERSION = "oel.coverage-tasking.v0.2"


def _required(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


@dataclass(frozen=True)
class TaskOpportunity:
    opportunity_id: str
    source_product_sha256: str
    asset_id: str
    kind: str
    start_s: float
    end_s: float
    objective_value: float
    storage_delta_bytes: float
    energy_cost_wh: float
    pointing_unit_eci: tuple[float, float, float] | None = None
    target_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opportunity_id",
            _required(self.opportunity_id, "opportunity_id"),
        )
        source_hash = _required(self.source_product_sha256, "source_product_sha256").lower()
        if len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash):
            raise ValueError("source_product_sha256 must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "source_product_sha256", source_hash)
        object.__setattr__(self, "asset_id", _required(self.asset_id, "asset_id"))
        kind = str(self.kind or "").strip().lower()
        if kind not in {"observation", "downlink", "other"}:
            raise ValueError("kind must be observation, downlink, or other.")
        object.__setattr__(self, "kind", kind)
        for field_name in (
            "start_s",
            "end_s",
            "objective_value",
            "storage_delta_bytes",
            "energy_cost_wh",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, value)
        if self.end_s <= self.start_s:
            raise ValueError("Opportunity end_s must be greater than start_s.")
        if self.objective_value < 0.0:
            raise ValueError("objective_value must be nonnegative.")
        if self.energy_cost_wh < 0.0:
            raise ValueError("energy_cost_wh must be nonnegative.")
        if kind == "observation" and self.storage_delta_bytes < 0.0:
            raise ValueError("Observation storage_delta_bytes must be nonnegative.")
        if kind == "downlink" and self.storage_delta_bytes > 0.0:
            raise ValueError("Downlink storage_delta_bytes must be nonpositive.")
        if self.pointing_unit_eci is not None:
            vector = np.asarray(self.pointing_unit_eci, dtype=float).reshape(-1)
            if vector.size != 3 or not np.all(np.isfinite(vector)):
                raise ValueError("pointing_unit_eci must contain three finite values.")
            norm = float(np.linalg.norm(vector))
            if abs(norm - 1.0) > 1.0e-10:
                raise ValueError("pointing_unit_eci must be normalized within 1e-10.")
            object.__setattr__(self, "pointing_unit_eci", tuple(float(value) for value in vector))
        target = None if self.target_id is None else str(self.target_id).strip()
        object.__setattr__(self, "target_id", target or None)


@dataclass(frozen=True)
class TaskingConstraints:
    horizon_start_s: float
    horizon_end_s: float
    maximum_slew_rate_rad_s: float | None
    settling_time_s: float
    maximum_payload_duty_cycle: float
    storage_capacity_bytes: float
    initial_storage_bytes: float
    energy_budget_wh: float
    maximum_candidates: int = 20

    def __post_init__(self) -> None:
        for field_name in (
            "horizon_start_s",
            "horizon_end_s",
            "settling_time_s",
            "maximum_payload_duty_cycle",
            "storage_capacity_bytes",
            "initial_storage_bytes",
            "energy_budget_wh",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, value)
        if self.horizon_end_s <= self.horizon_start_s:
            raise ValueError("Tasking horizon end must be after its start.")
        if self.settling_time_s < 0.0:
            raise ValueError("settling_time_s must be nonnegative.")
        if not 0.0 <= self.maximum_payload_duty_cycle <= 1.0:
            raise ValueError("maximum_payload_duty_cycle must be within [0, 1].")
        if self.storage_capacity_bytes < 0.0:
            raise ValueError("storage_capacity_bytes must be nonnegative.")
        if not 0.0 <= self.initial_storage_bytes <= self.storage_capacity_bytes:
            raise ValueError("initial_storage_bytes must be within storage capacity.")
        if self.energy_budget_wh < 0.0:
            raise ValueError("energy_budget_wh must be nonnegative.")
        if self.maximum_slew_rate_rad_s is not None:
            rate = float(self.maximum_slew_rate_rad_s)
            if not np.isfinite(rate) or rate <= 0.0:
                raise ValueError("maximum_slew_rate_rad_s must be positive and finite.")
            object.__setattr__(self, "maximum_slew_rate_rad_s", rate)
        if (
            isinstance(self.maximum_candidates, (bool, np.bool_))
            or int(self.maximum_candidates) != self.maximum_candidates
            or not 1 <= int(self.maximum_candidates) <= 24
        ):
            raise ValueError("maximum_candidates must be an integer within [1, 24].")
        object.__setattr__(self, "maximum_candidates", int(self.maximum_candidates))


@dataclass(frozen=True)
class CoverageTaskingConfig:
    analysis_id: str
    asset_id: str
    constraints: TaskingConstraints

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_id", _required(self.analysis_id, "analysis_id"))
        object.__setattr__(self, "asset_id", _required(self.asset_id, "asset_id"))
        if not isinstance(self.constraints, TaskingConstraints):
            raise ValueError("constraints must be validated TaskingConstraints.")


@dataclass(frozen=True)
class ScheduledTask:
    sequence: int
    opportunity_id: str
    asset_id: str
    kind: str
    start_s: float
    end_s: float
    duration_s: float
    objective_value: float
    storage_after_bytes: float
    cumulative_energy_wh: float
    source_product_sha256: str


@dataclass(frozen=True)
class CoverageTaskingResult:
    config: CoverageTaskingConfig
    selected_tasks: tuple[ScheduledTask, ...]
    selected_opportunity_ids: tuple[str, ...]
    rejected_opportunity_reasons: dict[str, str]
    objective_value: float
    final_storage_bytes: float
    energy_used_wh: float
    payload_duty_cycle: float
    evaluated_subset_count: int
    summary: dict[str, Any]
    input_semantic_sha256: str
    schedule_semantic_sha256: str


@dataclass(frozen=True)
class CoverageTaskingArtifacts:
    output_dir: Path
    manifest_json: Path
    summary_json: Path
    schedule_csv: Path
    rejected_csv: Path


def _transition_required_s(
    previous: TaskOpportunity,
    candidate: TaskOpportunity,
    constraints: TaskingConstraints,
) -> float:
    if constraints.maximum_slew_rate_rad_s is None:
        return constraints.settling_time_s
    if previous.pointing_unit_eci is None or candidate.pointing_unit_eci is None:
        raise ValueError("Slew-constrained tasking requires pointing vectors for every opportunity.")
    previous_vector = np.asarray(previous.pointing_unit_eci)
    candidate_vector = np.asarray(candidate.pointing_unit_eci)
    angle = float(np.arccos(np.clip(np.dot(previous_vector, candidate_vector), -1.0, 1.0)))
    return angle / constraints.maximum_slew_rate_rad_s + constraints.settling_time_s


def _input_hash(config: CoverageTaskingConfig, opportunities: tuple[TaskOpportunity, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "contract_version": COVERAGE_TASKING_CONTRACT_VERSION,
                "config": asdict(config),
                "opportunities": [asdict(value) for value in opportunities],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def optimize_coverage_tasking(
    config: CoverageTaskingConfig,
    opportunities: Iterable[TaskOpportunity],
) -> CoverageTaskingResult:
    """Exactly maximize a bounded opportunity set under declared resources."""

    supplied = tuple(opportunities)
    if any(not isinstance(value, TaskOpportunity) for value in supplied):
        raise ValueError("opportunities must contain validated TaskOpportunity values.")
    candidates = tuple(
        sorted(
            supplied,
            key=lambda value: (value.start_s, value.end_s, value.opportunity_id),
        )
    )
    if not candidates:
        raise ValueError("Tasking requires at least one opportunity.")
    if len(candidates) > config.constraints.maximum_candidates:
        raise ValueError("Tasking opportunity count exceeds maximum_candidates.")
    ids = tuple(value.opportunity_id for value in candidates)
    if len(set(ids)) != len(ids):
        raise ValueError("Task opportunity IDs must be unique.")
    for candidate in candidates:
        if candidate.asset_id != config.asset_id:
            raise ValueError(
                f"Opportunity {candidate.opportunity_id!r} does not belong to asset {config.asset_id!r}."
            )
        if (
            candidate.start_s < config.constraints.horizon_start_s
            or candidate.end_s > config.constraints.horizon_end_s
        ):
            raise ValueError(f"Opportunity {candidate.opportunity_id!r} is outside the tasking horizon.")
    if config.constraints.maximum_slew_rate_rad_s is not None and any(
        value.pointing_unit_eci is None for value in candidates
    ):
        raise ValueError("Slew-constrained tasking requires pointing vectors for every opportunity.")

    suffix_value = np.zeros(len(candidates) + 1)
    for index in range(len(candidates) - 1, -1, -1):
        suffix_value[index] = suffix_value[index + 1] + candidates[index].objective_value
    horizon_duration = config.constraints.horizon_end_s - config.constraints.horizon_start_s
    best_value = -1.0
    best_selection: tuple[int, ...] = ()
    evaluated = 0

    def better(value: float, selection: tuple[int, ...]) -> bool:
        nonlocal best_value, best_selection
        if value > best_value + 1.0e-12:
            return True
        if abs(value - best_value) <= 1.0e-12:
            candidate_ids = tuple(sorted(candidates[index].opportunity_id for index in selection))
            best_ids = tuple(sorted(candidates[index].opportunity_id for index in best_selection))
            return candidate_ids < best_ids
        return False

    def visit(
        index: int,
        selection: tuple[int, ...],
        value: float,
        storage: float,
        energy: float,
        observation_duration: float,
    ) -> None:
        nonlocal best_value, best_selection, evaluated
        if value + suffix_value[index] < best_value - 1.0e-12:
            return
        if index == len(candidates):
            evaluated += 1
            if better(value, selection):
                best_value = value
                best_selection = selection
            return
        visit(index + 1, selection, value, storage, energy, observation_duration)
        candidate = candidates[index]
        if selection:
            previous = candidates[selection[-1]]
            if candidate.start_s < previous.end_s - 1.0e-12:
                return
            if candidate.start_s - previous.end_s + 1.0e-12 < _transition_required_s(
                previous,
                candidate,
                config.constraints,
            ):
                return
        next_storage = storage + candidate.storage_delta_bytes
        if next_storage < -1.0e-9 or next_storage > config.constraints.storage_capacity_bytes + 1.0e-9:
            return
        next_energy = energy + candidate.energy_cost_wh
        if next_energy > config.constraints.energy_budget_wh + 1.0e-12:
            return
        next_observation_duration = observation_duration
        if candidate.kind == "observation":
            next_observation_duration += candidate.end_s - candidate.start_s
        if (
            next_observation_duration / horizon_duration
            > config.constraints.maximum_payload_duty_cycle + 1.0e-12
        ):
            return
        visit(
            index + 1,
            (*selection, index),
            value + candidate.objective_value,
            next_storage,
            next_energy,
            next_observation_duration,
        )

    visit(
        0,
        (),
        0.0,
        config.constraints.initial_storage_bytes,
        0.0,
        0.0,
    )
    selected = tuple(candidates[index] for index in best_selection)
    scheduled: list[ScheduledTask] = []
    storage = config.constraints.initial_storage_bytes
    energy = 0.0
    observation_duration = 0.0
    for sequence, candidate in enumerate(selected):
        storage += candidate.storage_delta_bytes
        energy += candidate.energy_cost_wh
        if candidate.kind == "observation":
            observation_duration += candidate.end_s - candidate.start_s
        scheduled.append(
            ScheduledTask(
                sequence=sequence,
                opportunity_id=candidate.opportunity_id,
                asset_id=candidate.asset_id,
                kind=candidate.kind,
                start_s=candidate.start_s,
                end_s=candidate.end_s,
                duration_s=candidate.end_s - candidate.start_s,
                objective_value=candidate.objective_value,
                storage_after_bytes=storage,
                cumulative_energy_wh=energy,
                source_product_sha256=candidate.source_product_sha256,
            )
        )
    selected_ids = tuple(value.opportunity_id for value in selected)
    rejected = {
        candidate.opportunity_id: _rejection_reason(candidate, selected, config.constraints)
        for candidate in candidates
        if candidate.opportunity_id not in selected_ids
    }
    input_hash = _input_hash(config, candidates)
    schedule_hash = hashlib.sha256(
        json.dumps(
            {
                "contract_version": COVERAGE_TASKING_CONTRACT_VERSION,
                "input_semantic_sha256": input_hash,
                "selected_opportunity_ids": selected_ids,
                "objective_value": best_value,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    duty = observation_duration / horizon_duration
    summary = {
        "contract_version": COVERAGE_TASKING_CONTRACT_VERSION,
        "analysis_id": config.analysis_id,
        "asset_id": config.asset_id,
        "status": "complete",
        "solver": "deterministic_exact_bounded_enumeration",
        "candidate_count": len(candidates),
        "evaluated_feasible_leaf_count": evaluated,
        "selected_count": len(selected),
        "objective_value": best_value,
        "final_storage_bytes": storage,
        "energy_used_wh": energy,
        "payload_duty_cycle": duty,
        "claim_limits": [
            "Opportunities and source-product hashes are inputs; tasking does not invent access or propagation.",
            "Slew time uses a direct angular-rate bound plus settling, not full attitude dynamics.",
            "Storage changes occur at task completion and energy is a horizon budget without generation dynamics.",
            "No routing, packet protocols, uncertain availability, constellation design, or orbit optimization.",
        ],
    }
    return CoverageTaskingResult(
        config=config,
        selected_tasks=tuple(scheduled),
        selected_opportunity_ids=selected_ids,
        rejected_opportunity_reasons=rejected,
        objective_value=best_value,
        final_storage_bytes=storage,
        energy_used_wh=energy,
        payload_duty_cycle=duty,
        evaluated_subset_count=evaluated,
        summary=summary,
        input_semantic_sha256=input_hash,
        schedule_semantic_sha256=schedule_hash,
    )


def _rejection_reason(
    candidate: TaskOpportunity,
    selected: tuple[TaskOpportunity, ...],
    constraints: TaskingConstraints,
) -> str:
    for chosen in selected:
        overlaps = candidate.start_s < chosen.end_s and chosen.start_s < candidate.end_s
        if overlaps:
            return f"conflicts_with:{chosen.opportunity_id}"
    ordered = tuple(sorted((*selected, candidate), key=lambda value: (value.start_s, value.end_s)))
    for previous, following in zip(ordered, ordered[1:]):
        if following.start_s - previous.end_s + 1.0e-12 < _transition_required_s(
            previous,
            following,
            constraints,
        ):
            return f"slew_or_settling_conflict:{previous.opportunity_id}:{following.opportunity_id}"
    return "not_selected_by_global_objective_or_resource_coupling"


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_coverage_tasking_artifacts(
    result: CoverageTaskingResult,
    output_dir: str | Path,
) -> CoverageTaskingArtifacts:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Coverage tasking output already exists: {destination}")
    destination.mkdir(parents=True)
    manifest = destination / "coverage_tasking_manifest.json"
    summary = destination / "coverage_tasking_summary.json"
    schedule = destination / "coverage_task_schedule.csv"
    rejected = destination / "coverage_task_rejections.csv"
    _json_dump(summary, result.summary)
    schedule_fields = tuple(ScheduledTask.__dataclass_fields__)
    with schedule.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=schedule_fields, lineterminator="\n")
        writer.writeheader()
        for task in result.selected_tasks:
            writer.writerow(asdict(task))
    with rejected.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("opportunity_id", "reason"))
        for opportunity_id, reason in sorted(result.rejected_opportunity_reasons.items()):
            writer.writerow((opportunity_id, reason))
    artifacts = {
        path.name: {"sha256": _file_hash(path)} for path in (summary, schedule, rejected)
    }
    _json_dump(
        manifest,
        {
            "contract_version": COVERAGE_TASKING_CONTRACT_VERSION,
            "analysis_id": result.config.analysis_id,
            "status": "complete",
            "normalized_config": asdict(result.config),
            "input_semantic_sha256": result.input_semantic_sha256,
            "schedule_semantic_sha256": result.schedule_semantic_sha256,
            "artifacts": artifacts,
            "claim_limits": result.summary["claim_limits"],
        },
    )
    return CoverageTaskingArtifacts(
        output_dir=destination,
        manifest_json=manifest,
        summary_json=summary,
        schedule_csv=schedule,
        rejected_csv=rejected,
    )


__all__ = [
    "COVERAGE_TASKING_CONTRACT_VERSION",
    "CoverageTaskingArtifacts",
    "CoverageTaskingConfig",
    "CoverageTaskingResult",
    "ScheduledTask",
    "TaskOpportunity",
    "TaskingConstraints",
    "optimize_coverage_tasking",
    "write_coverage_tasking_artifacts",
]
