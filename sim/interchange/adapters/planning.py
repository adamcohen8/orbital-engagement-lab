from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from sim.interchange.scenario_patches import build_scenario_patch_product, write_scenario_patch_product

PLANNING_PATCH_ADAPTER_ID = "oel.mission_recovery.scenario_patch"
PLANNING_PATCH_ADAPTER_VERSION = "1"


class PlanningPatchError(ValueError):
    """Raised when mission-recovery evidence cannot be represented as typed patches."""


def emit_mission_recovery_scenario_patches(
    mission_recovery: Mapping[str, Any],
    *,
    source_scenario: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_scenario).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    planner = dict(mission_recovery.get("planner", {}) or {})
    candidates = [dict(item or {}) for item in list(planner.get("candidates", []) or [])]
    if not candidates:
        raise PlanningPatchError("Mission-recovery planner emitted no candidates.")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PlanningPatchError("Source scenario must contain a YAML mapping.")
    object_id = str(mission_recovery.get("object_id", "") or "").strip()
    if not object_id or object_id not in dict(raw.get("objects", {}) or {}):
        raise PlanningPatchError("Mission-recovery object_id must exist in source scenario objects.")
    assessment_time_s = float(mission_recovery.get("assessment_time_s", 0.0) or 0.0)
    recommendations = dict(planner.get("recommended", {}) or {})
    modes_by_id: dict[str, list[str]] = {}
    for mode, candidate_id in recommendations.items():
        if candidate_id:
            modes_by_id.setdefault(str(candidate_id), []).append(str(mode))
    evidence_path = output / "mission_recovery_patch_evidence.json"
    evidence_payload = {
        "schema_id": "oel-mission-recovery-patch-evidence-v1",
        "object_id": object_id,
        "goal": mission_recovery.get("goal"),
        "assessment_time_s": assessment_time_s,
        "planner": deepcopy(planner),
    }
    evidence_path.write_text(json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    entries = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("candidate_id", "") or "").strip()
        if not candidate_id:
            raise PlanningPatchError("Every mission-recovery candidate requires candidate_id.")
        operations, operations_complete = _candidate_operations(
            raw,
            object_id=object_id,
            assessment_time_s=assessment_time_s,
            candidate=candidate,
        )
        eligible = (
            bool(candidate.get("feasible", False))
            and bool(candidate.get("verified", False))
            and operations_complete
            and all(_burn_is_materializable(dict(item or {})) for item in list(candidate.get("burn_sequence", []) or []))
        )
        warnings = [] if eligible else ["Candidate is not both verified, feasible, and fully duration-bound."]
        target = output / f"mission_recovery_{_slug(candidate_id)}.scenario_patch.json"
        product = build_scenario_patch_product(
            source,
            patch_type="mission_recovery_candidate",
            selection={
                "selection_id": candidate_id,
                "selection_kind": "mission_recovery_candidate",
                "rank": rank,
                "recommended_modes": sorted(modes_by_id.get(candidate_id, [])),
            },
            operations=operations,
            evidence={
                "objective": {
                    "goal": mission_recovery.get("goal"),
                    "source": candidate.get("source"),
                    "source_family": candidate.get("source_family"),
                    "target_basis": candidate.get("target_basis"),
                },
                "constraints": {
                    "max_recovery_time_s": planner.get("max_recovery_time_s"),
                    "max_recovery_delta_v_m_s": planner.get("max_recovery_delta_v_m_s"),
                    "feasible": candidate.get("feasible"),
                    "verified": candidate.get("verified"),
                    "within_tolerances": candidate.get("within_tolerances"),
                },
                "producer_evidence": deepcopy(candidate),
            },
            producer_capability_id="mission_recovery_planner",
            producer_run_id=str(raw.get("scenario_name", source.stem)),
            source_artifacts=[evidence_path],
            disposition="accepted" if eligible else "review_required",
            producer_status="verified_feasible" if eligible else "review_required",
            warnings=warnings,
            non_claims=[
                "This deterministic planner candidate is not an operational flight plan.",
                "Materialization validates configured OEL semantics; it does not establish higher-fidelity maneuver validity.",
            ],
            data_markings=_source_markings(raw),
            output_path=target,
        )
        entry = write_scenario_patch_product(product, target)
        entry["rank"] = rank
        entry["recommended_modes"] = sorted(modes_by_id.get(candidate_id, []))
        entry["product_path"] = target.name
        entries.append(entry)
    index = {
        "schema_id": "oel-scenario-patch-index-v1",
        "schema_version": 1,
        "patch_type": "mission_recovery_candidate",
        "source_scenario": str(source),
        "selection_required": True,
        "patches": entries,
    }
    index_path = output / "mission_recovery_scenario_patches.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "emitted", "index_path": str(index_path), "selection_required": True, "patches": entries}


def _candidate_operations(
    source: Mapping[str, Any], *, object_id: str, assessment_time_s: float, candidate: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    operations: list[dict[str, Any]] = []
    complete = True
    latest = assessment_time_s
    for burn_raw in list(candidate.get("burn_sequence", []) or []):
        burn = dict(burn_raw or {})
        start = assessment_time_s + float(burn.get("start_time_s", 0.0) or 0.0)
        duration = burn.get("duration_s")
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            complete = False
            continue
        frame = str(burn.get("frame", "") or "").strip().lower()
        try:
            vector = _burn_vector(burn, frame=frame)
        except PlanningPatchError:
            complete = False
            continue
        if not np.isfinite(duration_value) or duration_value <= 0.0:
            complete = False
            continue
        operations.append(
            {
                "op": "append",
                "kind": "mission_burn",
                "path": f"objects.{object_id}.mission_objectives",
                "value": {
                    "module": "sim.mission.modules",
                    "class_name": "ScheduledVectorBurnMissionModule",
                    "params": {
                        "target_id": "self",
                        "frame": frame,
                        "delta_v_m_s": vector,
                        "burn_start_s": start,
                        "burn_duration_s": duration_value,
                        "require_finite_reference": True,
                    },
                },
                "reason": f"Materialize candidate {candidate.get('candidate_id')} burn {burn.get('burn_index')}.",
            }
        )
        latest = max(latest, start + float(duration_value or 0.0))
    planned_end = assessment_time_s + float(candidate.get("planned_time_s", 0.0) or 0.0)
    required_duration = max(float(dict(source.get("simulator", {}) or {}).get("duration_s", 0.0) or 0.0), latest, planned_end)
    operations.append(
        {
            "op": "replace",
            "kind": "duration_extension",
            "path": "simulator.duration_s",
            "value": required_duration,
            "reason": "Retain the source run and extend through the selected recovery candidate.",
        }
    )
    return operations, complete


def _burn_vector(burn: Mapping[str, Any], *, frame: str) -> list[float]:
    if frame == "eci":
        value = np.asarray(burn.get("delta_v_eci_m_s"), dtype=float).reshape(-1)
        if value.size != 3 or not np.all(np.isfinite(value)):
            raise PlanningPatchError("ECI candidate burns require three finite delta_v_eci_m_s values.")
        return value.tolist()
    if frame != "ric":
        raise PlanningPatchError("Candidate burn frame must be 'eci' or 'ric'.")
    axis = str(burn.get("axis", "") or "").strip().upper()
    units = {"+R": [1.0, 0.0, 0.0], "-R": [-1.0, 0.0, 0.0], "+I": [0.0, 1.0, 0.0], "-I": [0.0, -1.0, 0.0], "+C": [0.0, 0.0, 1.0], "-C": [0.0, 0.0, -1.0]}
    if axis not in units:
        raise PlanningPatchError(f"Unsupported RIC candidate burn axis {axis!r}.")
    magnitude = float(burn.get("delta_v_m_s", 0.0) or 0.0)
    return (np.asarray(units[axis]) * magnitude).tolist()


def _burn_is_materializable(burn: Mapping[str, Any]) -> bool:
    try:
        duration = float(burn.get("duration_s"))
        _burn_vector(burn, frame=str(burn.get("frame", "") or "").lower())
    except (TypeError, ValueError):
        return False
    return np.isfinite(duration) and duration > 0.0


def _source_markings(source: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(source.get("metadata", {}) or {})
    public = metadata.get("owner") == "public" and bool(metadata.get("public_surface"))
    return {
        "scope": "public" if public else "private_pro",
        "handling": "public_synthetic" if public else "private",
        "approved_for_public_export": bool(public),
        "contains_customer_data": False,
        "contains_hidden_truth": False,
    }


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


__all__ = [
    "PLANNING_PATCH_ADAPTER_ID",
    "PLANNING_PATCH_ADAPTER_VERSION",
    "PlanningPatchError",
    "emit_mission_recovery_scenario_patches",
]
