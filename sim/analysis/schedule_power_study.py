"""Retain and verify one schedule-coupled spacecraft-power study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from sim.utils.io import read_regular_file_nofollow

from .history_adapters import AnalysisHistory
from .mission_scheduling import (
    MissionSchedulingProblem,
    verify_mission_scheduling_artifacts,
)
from .mission_scheduling_sources import (
    MissionSchedulingSourcePlan,
    build_solve_mission_schedule_from_sources,
    verify_source_built_mission_schedule,
)
from .orbit_bound_sources import verify_orbit_bound_collection, verify_orbit_bound_link
from .orbit_history_product import (
    orbit_history_semantic_sha256,
    verify_orbit_history_product,
)
from .spacecraft_power import (
    SpacecraftPowerProblem,
    assess_spacecraft_power,
    power_history_from_mapping,
    power_history_to_dict,
    problem_with_mission_schedule,
    validate_spacecraft_power_inputs,
    verify_spacecraft_power_artifacts,
    write_spacecraft_power_artifacts,
)
from .study_lifecycle import (
    CAPABILITY_CONTRACTS,
    STUDY_CLAIMS_SCHEMA,
    STUDY_PLAN_SCHEMA,
    STUDY_REQUEST_SCHEMA,
    build_study_bundle,
    inspect_study_bundle,
)

SCHEDULE_POWER_STUDY_SCHEMA = "oel.schedule_power_study.v1"
SOURCE_SCHEDULE_POWER_STUDY_SCHEMA = "oel.source_schedule_power_study.v1"
ORBIT_BOUND_SOURCE_STUDY_SCHEMA = "oel.orbit_bound_source_study.v1"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_SCHEDULE_FILES = frozenset(
    {
        "normalized_problem.json",
        "mission_schedule_summary.json",
        "mission_schedule.csv",
        "mission_schedule_rejections.csv",
        "mission_resource_summary.csv",
        "mission_data_delivery.csv",
        "mission_schedule_manifest.json",
    }
)
_LIMITS = [
    "The schedule has elapsed-second times and no absolute epoch; its UTC anchor is the analyst's explicit assertion.",
    "Power feasibility covers one declared orbit, load timeline, and bounded battery model, not flight hardware or uncertainty.",
    "Study identity replay does not replace either domain's authoritative replay or authorize operations.",
]
_SOURCE_LIMITS = [
    "The source plan's UTC epoch binds its opportunities; the power history and problem must independently match it.",
    "Source opportunity geometry is not proven to share the supplied power orbit history.",
    "Opportunity products and schedule replay are deterministic model evidence, not operational access or command authority.",
    *_LIMITS[1:],
]


class SchedulePowerStudyError(ValueError):
    """Raised when a schedule-power study cannot be built or verified."""


def _read_json(path: Path, field: str) -> dict[str, Any]:
    value = json.loads(read_regular_file_nofollow(path, min_bytes=1, max_bytes=_MAX_JSON_BYTES))
    if not isinstance(value, dict):
        raise SchedulePowerStudyError(f"{field} must be a JSON object.")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(read_regular_file_nofollow(path, min_bytes=1, max_bytes=_MAX_JSON_BYTES)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _study_records(
    *, study_id: str, title: str, asset_id: str, epoch_jd_utc: float,
    feasibility: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = {
        "schema_version": STUDY_REQUEST_SCHEMA,
        "study_id": study_id,
        "title": title,
        "question": f"Is the selected schedule power-feasible for {asset_id} under the declared model?",
        "capabilities": ["mission_scheduling", "spacecraft_power"],
        "assumptions": [
            f"The schedule's elapsed seconds are anchored at Julian UTC {epoch_jd_utc:.12f}.",
            "The supplied orbit history and power problem describe the same spacecraft and epoch.",
        ],
        "clarifications": [
            {
                "question": "Does study replay recompute scheduling and power?",
                "resolution": "No. The retained domain packets have separate authoritative replay checks.",
            }
        ],
        "context": {
            "epoch": f"Julian UTC {epoch_jd_utc:.12f}, asserted for the schedule",
            "time_system": "elapsed SI seconds from the asserted UTC epoch",
            "frame": "canonical ECI orbit history",
            "units": "kilometres, seconds, watts, and watt-hours",
        },
        "fidelity": {
            "level": "bounded_public",
            "description": "Exact bounded scheduling and deterministic sampled solar-array/battery analysis.",
        },
        "acceptance_criteria": [
            {
                "criterion_id": "schedule-selected",
                "description": "The verified schedule selects at least one activity.",
            },
            {
                "criterion_id": "power-disposition",
                "description": "The verified power result reports feasible or infeasible for the selected activities.",
            },
        ],
    }
    plan = {
        "schema_version": STUDY_PLAN_SCHEMA,
        "study_id": study_id,
        "request_sha256": "auto",
        "resource_profile": "laptop-safe",
        "steps": [
            {
                "step_id": "schedule",
                "capability": "mission_scheduling",
                "analysis_interface": CAPABILITY_CONTRACTS["mission_scheduling"]["analysis_interface"],
                "expected_evidence_schema": CAPABILITY_CONTRACTS["mission_scheduling"]["evidence_schema"],
                "depends_on": [],
                "acceptance_criterion_ids": ["schedule-selected"],
            },
            {
                "step_id": "power",
                "capability": "spacecraft_power",
                "analysis_interface": CAPABILITY_CONTRACTS["spacecraft_power"]["analysis_interface"],
                "expected_evidence_schema": CAPABILITY_CONTRACTS["spacecraft_power"]["evidence_schema"],
                "depends_on": ["schedule"],
                "acceptance_criterion_ids": ["power-disposition"],
            },
        ],
    }
    claims = {
        "schema_version": STUDY_CLAIMS_SCHEMA,
        "study_id": study_id,
        "plan_sha256": "auto",
        "claims": [
            {
                "claim_id": "schedule-retained",
                "statement": "The retained schedule selects at least one activity.",
                "validation_level": "VC-1",
                "criterion_ids": ["schedule-selected"],
                "evidence": [{"step_id": "schedule", "json_pointer": "/selected_count"}],
            },
            {
                "claim_id": "power-assessed",
                "statement": f"The declared schedule is {feasibility} for {asset_id} under the retained power model.",
                "validation_level": "VC-1",
                "criterion_ids": ["power-disposition"],
                "evidence": [{"step_id": "power", "json_pointer": "/feasibility"}],
            },
        ],
        "non_claims": list(_LIMITS),
    }
    return request, plan, claims


def _manifest(
    root: Path, *, schedule_epoch_jd_utc: float, activity_power_w: Mapping[str, float],
) -> dict[str, Any]:
    schedule = verify_mission_scheduling_artifacts(root / "schedule")
    power = verify_spacecraft_power_artifacts(root / "power")
    study = inspect_study_bundle(root / "study")
    schedule_summary = _read_json(root / "schedule" / "mission_schedule_summary.json", "schedule summary")
    power_summary = _read_json(root / "power" / "spacecraft_power_summary.json", "power summary")
    if _read_json(root / "study" / "evidence" / "schedule.json", "retained schedule") != schedule_summary:
        raise SchedulePowerStudyError("Study schedule evidence differs from the authoritative schedule packet.")
    if _read_json(root / "study" / "evidence" / "power.json", "retained power") != power_summary:
        raise SchedulePowerStudyError("Study power evidence differs from the authoritative power packet.")
    if schedule["schedule_semantic_sha256"] not in power_summary["source_product_sha256s"]:
        raise SchedulePowerStudyError("Power evidence does not consume the retained schedule.")
    if power_summary["asset_id"] != power["asset_id"]:
        raise SchedulePowerStudyError("Power asset identity differs from authoritative replay.")
    problem = SpacecraftPowerProblem.from_mapping(
        _read_json(root / "power" / "normalized_problem.json", "power problem")
    )
    if abs(problem.epoch_jd_utc - schedule_epoch_jd_utc) > 1.0e-12:
        raise SchedulePowerStudyError("Asserted schedule epoch does not match the power problem.")
    schedule_problem = MissionSchedulingProblem.from_mapping(
        _read_json(root / "schedule" / "normalized_problem.json", "schedule problem")
    )
    if (schedule_problem.horizon_start_s < problem.horizon_start_s or
            schedule_problem.horizon_end_s > problem.horizon_end_s):
        raise SchedulePowerStudyError("Schedule horizon lies outside the power-analysis horizon.")
    if set(activity_power_w) != {"observation", "downlink"} or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or
        not math.isfinite(value) or value < 0.0 for value in activity_power_w.values()
    ):
        raise SchedulePowerStudyError("Declared activity power must contain finite nonnegative observation and downlink loads.")
    selected = [item for item in schedule["activities"] if item["asset_id"] == problem.asset_id]
    if not selected:
        raise SchedulePowerStudyError("Selected schedule has no activities for the power asset.")
    digest = schedule["schedule_semantic_sha256"]
    expected_activities = {
        f"schedule-{item['opportunity_id']}": (
            item["kind"], item["start_s"], item["end_s"],
            float(activity_power_w[item["kind"]]),
        )
        for item in selected
    }
    actual_activities = {
        item.activity_id: (item.category, item.start_s, item.end_s, item.load_power_w)
        for item in problem.activities if item.source_product_sha256 == digest
    }
    if actual_activities != expected_activities:
        raise SchedulePowerStudyError("Power activities do not match the exact retained schedule and loads.")
    if study["capabilities"] != ["mission_scheduling", "spacecraft_power"] or study["step_count"] != 2:
        raise SchedulePowerStudyError("Study bundle does not contain the exact schedule-power steps.")
    return {
        "schema_version": SCHEDULE_POWER_STUDY_SCHEMA,
        "status": "verified",
        "study_id": study["study_id"],
        "asset_id": problem.asset_id,
        "schedule_epoch_jd_utc": schedule_epoch_jd_utc,
        "activity_power_w": {key: float(activity_power_w[key]) for key in sorted(activity_power_w)},
        "schedule_semantic_sha256": schedule["schedule_semantic_sha256"],
        "power_result_semantic_sha256": power["result_semantic_sha256"],
        "study_bundle_semantic_sha256": study["bundle_semantic_sha256"],
        "schedule_manifest_sha256": _sha256(root / "schedule" / "mission_schedule_manifest.json"),
        "power_manifest_sha256": _sha256(root / "power" / "spacecraft_power_manifest.json"),
        "study_receipt_sha256": _sha256(root / "study" / "study_receipt.json"),
        "feasibility": power["feasibility"],
        "evidence": {"schedule": "schedule", "power": "power", "study": "study"},
        "non_claims": list(_LIMITS),
    }


def inspect_schedule_power_study(output_dir: str | Path) -> dict[str, Any]:
    """Recompute both domains and verify the content-bound completed study."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise SchedulePowerStudyError("Study output directory must not be a symbolic link.")
    root = requested.resolve()
    if not root.is_dir() or {item.name for item in root.iterdir()} != {
        "schedule", "power", "study", "workflow_manifest.json"
    }:
        raise SchedulePowerStudyError("Schedule-power study has an incomplete or unexpected artifact inventory.")
    retained = _read_json(root / "workflow_manifest.json", "workflow manifest")
    if retained.get("schema_version") != SCHEDULE_POWER_STUDY_SCHEMA:
        raise SchedulePowerStudyError("Unsupported schedule-power study manifest schema.")
    epoch = retained.get("schedule_epoch_jd_utc")
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not math.isfinite(epoch):
        raise SchedulePowerStudyError("Schedule epoch assertion must be finite.")
    loads = retained.get("activity_power_w")
    if not isinstance(loads, dict):
        raise SchedulePowerStudyError("Study manifest is missing declared activity power.")
    expected = _manifest(root, schedule_epoch_jd_utc=float(epoch), activity_power_w=loads)
    if retained != expected:
        raise SchedulePowerStudyError("Schedule-power study manifest differs from verified evidence.")
    return expected


def build_schedule_power_study(
    *, schedule_dir: str | Path,
    problem: SpacecraftPowerProblem | Mapping[str, Any],
    history: AnalysisHistory | Mapping[str, Any],
    observation_load_w: float,
    downlink_load_w: float,
    schedule_epoch_jd_utc: float,
    output_dir: str | Path,
    study_id: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Build one atomic study using the authoritative schedule and power tools."""

    schedule_root_input = Path(schedule_dir).expanduser()
    if schedule_root_input.is_symlink():
        raise SchedulePowerStudyError("Schedule evidence directory must not be a symbolic link.")
    schedule_root = schedule_root_input.resolve()
    schedule = verify_mission_scheduling_artifacts(schedule_root)
    parsed_problem = problem if isinstance(problem, SpacecraftPowerProblem) else SpacecraftPowerProblem.from_mapping(problem)
    parsed_history = history if isinstance(history, AnalysisHistory) else power_history_from_mapping(history)
    epoch = float(schedule_epoch_jd_utc)
    if not math.isfinite(epoch) or abs(epoch - parsed_problem.epoch_jd_utc) > 1.0e-12:
        raise SchedulePowerStudyError("Explicit schedule epoch must match the power problem epoch_jd_utc.")
    if schedule["summary"]["status"] != "complete":
        raise SchedulePowerStudyError("Schedule must have a completed feasible selection.")
    schedule_problem = MissionSchedulingProblem.from_mapping(
        _read_json(schedule_root / "normalized_problem.json", "schedule problem")
    )
    if (schedule_problem.horizon_start_s < parsed_problem.horizon_start_s or
            schedule_problem.horizon_end_s > parsed_problem.horizon_end_s):
        raise SchedulePowerStudyError("Schedule horizon lies outside the power-analysis horizon.")
    if not any(item["asset_id"] == parsed_problem.asset_id for item in schedule["activities"]):
        raise SchedulePowerStudyError("Selected schedule has no activities for the power asset.")
    validate_spacecraft_power_inputs(parsed_problem, parsed_history)
    destination_input = Path(output_dir).expanduser()
    if destination_input.is_symlink():
        raise SchedulePowerStudyError("Study output directory must not be a symbolic link.")
    destination = destination_input.resolve()
    if destination.exists() or schedule_root == destination or schedule_root in destination.parents:
        raise SchedulePowerStudyError("Study output must be absent and outside the schedule evidence directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        retained_schedule = staging / "schedule"
        retained_schedule.mkdir()
        for name in sorted(_SCHEDULE_FILES):
            (retained_schedule / name).write_bytes(
                read_regular_file_nofollow(schedule_root / name, min_bytes=1, max_bytes=_MAX_JSON_BYTES)
            )
        copied = verify_mission_scheduling_artifacts(retained_schedule)
        if copied["schedule_semantic_sha256"] != schedule["schedule_semantic_sha256"]:
            raise SchedulePowerStudyError("Schedule changed while it was being retained.")
        converted = problem_with_mission_schedule(
            parsed_problem,
            retained_schedule,
            activity_power_w={"observation": observation_load_w, "downlink": downlink_load_w},
        )
        result = assess_spacecraft_power(converted, parsed_history)
        power_artifacts = write_spacecraft_power_artifacts(result, parsed_history, staging / "power")
        verify_spacecraft_power_artifacts(power_artifacts.output_dir)
        generated_id = "schedule-power-" + hashlib.sha256(
            (copied["schedule_semantic_sha256"] + result.summary["result_semantic_sha256"]).encode("ascii")
        ).hexdigest()[:20]
        request, plan, claims = _study_records(
            study_id=study_id or generated_id,
            title=title or f"Schedule-coupled power for {converted.asset_id}",
            asset_id=converted.asset_id,
            epoch_jd_utc=epoch,
            feasibility=result.summary["feasibility"],
        )
        build_study_bundle(
            request, plan, claims,
            {
                "schedule": retained_schedule / "mission_schedule_summary.json",
                "power": power_artifacts.summary_json,
            },
            staging / "study",
        )
        manifest = _manifest(
            staging,
            schedule_epoch_jd_utc=epoch,
            activity_power_w={"observation": observation_load_w, "downlink": downlink_load_w},
        )
        _write_json(staging / "workflow_manifest.json", manifest)
        inspect_schedule_power_study(staging)
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return inspect_schedule_power_study(destination)


def _check_source_packet_inventory(root: Path) -> None:
    manifest = _read_json(root / "mission_schedule_source_manifest.json", "source manifest")
    claims = manifest.get("sources")
    if not isinstance(claims, list):
        raise SchedulePowerStudyError("Source manifest has no source claims.")
    expected_files = {
        "mission_schedule_source_manifest.json",
        "normalized_source_plan.json",
        *(f"schedule/{name}" for name in _SCHEDULE_FILES),
    }
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("artifacts"), list):
            raise SchedulePowerStudyError("Source manifest has invalid source artifact claims.")
        for receipt in claim["artifacts"]:
            if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
                raise SchedulePowerStudyError("Source manifest has an invalid artifact path.")
            path = Path(receipt["path"])
            if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
                raise SchedulePowerStudyError("Source artifact path must stay inside the packet.")
            expected_files.add(path.as_posix())
    expected_dirs = {"schedule", "source_products"}
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_dirs.add(parent.as_posix())
            parent = parent.parent
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for item in root.rglob("*"):
        if item.is_symlink():
            raise SchedulePowerStudyError("Source packet must not contain symbolic links.")
        relative = item.relative_to(root).as_posix()
        if item.is_file():
            actual_files.add(relative)
        elif item.is_dir():
            actual_dirs.add(relative)
        else:
            raise SchedulePowerStudyError("Source packet contains a non-regular entry.")
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise SchedulePowerStudyError("Source packet has an incomplete or unexpected artifact inventory.")


def _source_manifest(root: Path) -> dict[str, Any]:
    _check_source_packet_inventory(root / "source_schedule")
    sources = verify_source_built_mission_schedule(root / "source_schedule")
    integrated = inspect_schedule_power_study(root / "schedule_power_study")
    source_plan = MissionSchedulingSourcePlan.from_mapping(
        _read_json(root / "source_schedule" / "normalized_source_plan.json", "source plan")
    )
    if source_plan.epoch_jd_utc != integrated["schedule_epoch_jd_utc"]:
        raise SchedulePowerStudyError("Source-plan epoch differs from the schedule-power study epoch.")
    if sources["schedule_semantic_sha256"] != integrated["schedule_semantic_sha256"]:
        raise SchedulePowerStudyError("Retained source schedule differs from the power-study schedule.")
    for name in _SCHEDULE_FILES:
        source_file = root / "source_schedule" / "schedule" / name
        study_file = root / "schedule_power_study" / "schedule" / name
        if _sha256(source_file) != _sha256(study_file):
            raise SchedulePowerStudyError("Power study did not retain the exact source-built schedule packet.")
    return {
        "schema_version": SOURCE_SCHEDULE_POWER_STUDY_SCHEMA,
        "status": "verified",
        "study_id": integrated["study_id"],
        "asset_id": integrated["asset_id"],
        "schedule_epoch_jd_utc": source_plan.epoch_jd_utc,
        "source_count": sources["source_count"],
        "source_plan_semantic_sha256": sources["source_plan_semantic_sha256"],
        "source_manifest_sha256": _sha256(
            root / "source_schedule" / "mission_schedule_source_manifest.json"
        ),
        "schedule_semantic_sha256": sources["schedule_semantic_sha256"],
        "power_result_semantic_sha256": integrated["power_result_semantic_sha256"],
        "study_bundle_semantic_sha256": integrated["study_bundle_semantic_sha256"],
        "feasibility": integrated["feasibility"],
        "evidence": {
            "source_schedule": "source_schedule",
            "schedule_power_study": "schedule_power_study",
        },
        "non_claims": list(_SOURCE_LIMITS),
    }


def inspect_source_schedule_power_study(output_dir: str | Path) -> dict[str, Any]:
    """Replay the retained collection/link sources, schedule, power, and study."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise SchedulePowerStudyError("Source study output directory must not be a symbolic link.")
    root = requested.resolve()
    if not root.is_dir() or {item.name for item in root.iterdir()} != {
        "source_schedule", "schedule_power_study", "workflow_manifest.json"
    }:
        raise SchedulePowerStudyError("Source study has an incomplete or unexpected artifact inventory.")
    retained = _read_json(root / "workflow_manifest.json", "source study manifest")
    if retained.get("schema_version") != SOURCE_SCHEDULE_POWER_STUDY_SCHEMA:
        raise SchedulePowerStudyError("Unsupported source study manifest schema.")
    expected = _source_manifest(root)
    if retained != expected:
        raise SchedulePowerStudyError("Source study manifest differs from verified evidence.")
    return expected


def build_source_schedule_power_study(
    *, source_plan: MissionSchedulingSourcePlan | Mapping[str, Any],
    base_dir: str | Path,
    problem: SpacecraftPowerProblem | Mapping[str, Any],
    history: AnalysisHistory | Mapping[str, Any],
    observation_load_w: float,
    downlink_load_w: float,
    output_dir: str | Path,
    study_id: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Build one retained source-to-schedule-to-power study from OEL products."""

    parsed_plan = (
        source_plan if isinstance(source_plan, MissionSchedulingSourcePlan)
        else MissionSchedulingSourcePlan.from_mapping(source_plan)
    )
    destination_input = Path(output_dir).expanduser()
    if destination_input.is_symlink():
        raise SchedulePowerStudyError("Source study output directory must not be a symbolic link.")
    destination = destination_input.resolve()
    source_base = Path(base_dir).expanduser().resolve()
    if destination.exists():
        raise SchedulePowerStudyError("Source study output directory must be absent.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        source_artifacts = build_solve_mission_schedule_from_sources(
            parsed_plan, base_dir=source_base, output_dir=staging / "source_schedule"
        )
        verify_source_built_mission_schedule(source_artifacts.output_dir)
        build_schedule_power_study(
            schedule_dir=source_artifacts.schedule_artifacts.output_dir,
            problem=problem,
            history=history,
            observation_load_w=observation_load_w,
            downlink_load_w=downlink_load_w,
            schedule_epoch_jd_utc=parsed_plan.epoch_jd_utc,
            output_dir=staging / "schedule_power_study",
            study_id=study_id,
            title=title,
        )
        manifest = _source_manifest(staging)
        _write_json(staging / "workflow_manifest.json", manifest)
        inspect_source_schedule_power_study(staging)
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return inspect_source_schedule_power_study(destination)


def _orbit_bound_manifest(root: Path) -> dict[str, Any]:
    orbit_manifest, history = verify_orbit_history_product(root / "orbit_history")
    source_study = inspect_source_schedule_power_study(root / "source_study")
    source_root = root / "source_study" / "source_schedule"
    plan = MissionSchedulingSourcePlan.from_mapping(
        _read_json(source_root / "normalized_source_plan.json", "source plan")
    )
    digest = orbit_history_semantic_sha256(history)
    if (plan.orbit_history_semantic_sha256 != digest
            or source_study["asset_id"] != history.object_id
            or any(item.asset_id != history.object_id for item in (*plan.collection_sources, *plan.link_sources))):
        raise SchedulePowerStudyError("Orbit-bound source plan or study differs from the retained orbit asset.")
    power_history = _read_json(
        root / "source_study" / "schedule_power_study" / "power" / "normalized_history.json",
        "power history",
    )
    if power_history != power_history_to_dict(history):
        raise SchedulePowerStudyError("Power analysis did not consume the retained parent orbit history.")
    for source in plan.collection_sources:
        verify_orbit_bound_collection(
            source_root / "source_products" / source.source_id / "collection_evidence.json", history
        )
    for source in plan.link_sources:
        verify_orbit_bound_link(source_root / "source_products" / source.source_id, history)
    return {
        "schema_version": ORBIT_BOUND_SOURCE_STUDY_SCHEMA,
        "status": "verified",
        "asset_id": history.object_id,
        "study_id": source_study["study_id"],
        "history_semantic_sha256": digest,
        "source_review_sha256": orbit_manifest["source_review_sha256"],
        "source_plan_semantic_sha256": source_study["source_plan_semantic_sha256"],
        "schedule_semantic_sha256": source_study["schedule_semantic_sha256"],
        "power_result_semantic_sha256": source_study["power_result_semantic_sha256"],
        "study_bundle_semantic_sha256": source_study["study_bundle_semantic_sha256"],
        "feasibility": source_study["feasibility"],
        "evidence": {"orbit_history": "orbit_history", "source_study": "source_study"},
        "non_claims": [
            "Shared orbit identity proves input continuity, not collection, RF, or power-model accuracy.",
            "Attitude, site, sensor, battery, and load assumptions retain their separate model limits.",
            "The study is engineering evidence, not flight or decision authority.",
        ],
    }


def inspect_orbit_bound_source_study(output_dir: str | Path) -> dict[str, Any]:
    """Replay all domains and prove one parent orbit drove collection, link, and power."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise SchedulePowerStudyError("Orbit-bound study directory must not be a symbolic link.")
    root = requested.resolve()
    if not root.is_dir() or {item.name for item in root.iterdir()} != {
        "orbit_history", "source_study", "workflow_manifest.json"
    }:
        raise SchedulePowerStudyError("Orbit-bound study has an incomplete or unexpected inventory.")
    retained = _read_json(root / "workflow_manifest.json", "orbit-bound study manifest")
    expected = _orbit_bound_manifest(root)
    if retained != expected:
        raise SchedulePowerStudyError("Orbit-bound study manifest differs from verified evidence.")
    return expected


def build_orbit_bound_source_study(
    *, orbit_history_dir: str | Path,
    source_plan: MissionSchedulingSourcePlan | Mapping[str, Any],
    base_dir: str | Path,
    problem: SpacecraftPowerProblem | Mapping[str, Any],
    observation_load_w: float,
    downlink_load_w: float,
    output_dir: str | Path,
    study_id: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Build a single-orbit source-to-schedule-to-power study."""

    orbit_source_input = Path(orbit_history_dir).expanduser()
    if orbit_source_input.is_symlink():
        raise SchedulePowerStudyError("Orbit-history source must not be symbolic.")
    orbit_source = orbit_source_input.resolve()
    orbit_manifest, history = verify_orbit_history_product(orbit_source)
    plan = (
        source_plan if isinstance(source_plan, MissionSchedulingSourcePlan)
        else MissionSchedulingSourcePlan.from_mapping(source_plan)
    )
    if (plan.orbit_history_semantic_sha256 != orbit_manifest["history_semantic_sha256"]
            or any(item.asset_id != history.object_id for item in (*plan.collection_sources, *plan.link_sources))):
        raise SchedulePowerStudyError("Source plan must bind every source to the parent orbit asset and digest.")
    destination_input = Path(output_dir).expanduser()
    if destination_input.is_symlink():
        raise SchedulePowerStudyError("Orbit-bound study destination must not be symbolic.")
    destination = destination_input.resolve()
    if destination.exists() or orbit_source == destination or orbit_source in destination.parents:
        raise SchedulePowerStudyError("Orbit-bound study destination must be absent and outside the history product.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        retained_orbit = staging / "orbit_history"
        retained_orbit.mkdir()
        for name in ("normalized_history.json", "orbit_history_manifest.json"):
            (retained_orbit / name).write_bytes(read_regular_file_nofollow(
                orbit_source / name, min_bytes=1, max_bytes=64 * 1024 * 1024
            ))
        verify_orbit_history_product(retained_orbit)
        build_source_schedule_power_study(
            source_plan=plan, base_dir=base_dir, problem=problem, history=history,
            observation_load_w=observation_load_w, downlink_load_w=downlink_load_w,
            output_dir=staging / "source_study", study_id=study_id, title=title,
        )
        manifest = _orbit_bound_manifest(staging)
        _write_json(staging / "workflow_manifest.json", manifest)
        inspect_orbit_bound_source_study(staging)
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return inspect_orbit_bound_source_study(destination)


__all__ = [
    "SCHEDULE_POWER_STUDY_SCHEMA",
    "SOURCE_SCHEDULE_POWER_STUDY_SCHEMA",
    "ORBIT_BOUND_SOURCE_STUDY_SCHEMA",
    "SchedulePowerStudyError",
    "build_schedule_power_study",
    "build_source_schedule_power_study",
    "build_orbit_bound_source_study",
    "inspect_schedule_power_study",
    "inspect_source_schedule_power_study",
    "inspect_orbit_bound_source_study",
]
