# ruff: noqa: E402
"""Build three real public OEL studies from completed deterministic evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.analysis.conjunction_workflow import assess_conjunction, write_conjunction_evidence
from sim.analysis.mission_scheduling import (
    MissionSchedulingProblem,
    solve_mission_schedule,
    write_mission_scheduling_artifacts,
)
from sim.analysis.study_lifecycle import (
    CAPABILITY_CONTRACTS,
    STUDY_CLAIMS_SCHEMA,
    STUDY_PLAN_SCHEMA,
    STUDY_REQUEST_SCHEMA,
    build_study_bundle,
    inspect_study_bundle,
    replay_study_bundle,
)
from sim.analysis.trajectory_targeting import (
    TrajectoryTargetingProblem,
    solve_trajectory_target,
    write_trajectory_targeting_evidence,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _records(
    *,
    study_id: str,
    capability: str,
    title: str,
    question: str,
    criterion: str,
    claim: str,
    evidence_pointer: str,
    non_claim: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    step_id = capability.replace("_", "-")
    criterion_id = f"criterion-{step_id}"
    request = {
        "schema_version": STUDY_REQUEST_SCHEMA,
        "study_id": study_id,
        "title": title,
        "question": question,
        "capabilities": [capability],
        "assumptions": [
            "The checked-in example input is synthetic and public-safe.",
            "The named OEL domain workflow remains authoritative for its physics and result status.",
        ],
        "clarifications": [
            {
                "question": "Does lifecycle replay recompute the domain analysis?",
                "resolution": "No. It verifies retained identity; domain replay remains separate.",
            }
        ],
        "context": {
            "epoch": "relative elapsed seconds from each example's declared initial state",
            "time_system": "elapsed SI seconds",
            "frame": "EME2000-compatible ECI with domain-declared local frames",
            "units": "OEL domain-contract units",
        },
        "fidelity": {
            "level": "bounded_public",
            "description": "Deterministic example inside the named public capability envelope.",
        },
        "acceptance_criteria": [{"criterion_id": criterion_id, "description": criterion}],
    }
    plan = {
        "schema_version": STUDY_PLAN_SCHEMA,
        "study_id": study_id,
        "request_sha256": "auto",
        "resource_profile": "laptop-safe",
        "steps": [
            {
                "step_id": step_id,
                "capability": capability,
                "analysis_interface": CAPABILITY_CONTRACTS[capability]["analysis_interface"],
                "expected_evidence_schema": CAPABILITY_CONTRACTS[capability]["evidence_schema"],
                "depends_on": [],
                "acceptance_criterion_ids": [criterion_id],
            }
        ],
    }
    claims = {
        "schema_version": STUDY_CLAIMS_SCHEMA,
        "study_id": study_id,
        "plan_sha256": "auto",
        "claims": [
            {
                "claim_id": f"claim-{step_id}",
                "statement": claim,
                "validation_level": "VC-1",
                "criterion_ids": [criterion_id],
                "evidence": [{"step_id": step_id, "json_pointer": evidence_pointer}],
            }
        ],
        "non_claims": [
            non_claim,
            "A verified study receipt is not operational authorization or flight qualification.",
        ],
    }
    return request, plan, claims


def build_examples(output_root: str | Path) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"output_root must not already exist: {destination}.")
    destination.mkdir(parents=True)
    sources = destination / "domain_evidence"
    sources.mkdir()

    trajectory_problem = TrajectoryTargetingProblem.from_mapping(
        _read_json(ROOT / "examples/trajectory_targeting/hohmann_apoapsis.json")
    )
    trajectory_evidence = solve_trajectory_target(trajectory_problem)
    trajectory_path = write_trajectory_targeting_evidence(
        sources / "trajectory_targeting_evidence.json", trajectory_evidence
    )

    conjunction_evidence = assess_conjunction(
        _read_json(ROOT / "examples/conjunction/synthetic_crossing.json")
    )
    conjunction_path = write_conjunction_evidence(
        conjunction_evidence, sources / "conjunction_assessment_evidence.json"
    )

    scheduling_problem = MissionSchedulingProblem.from_mapping(
        _read_json(ROOT / "examples/mission_scheduling/public_two_asset_collection_problem.json")
    )
    scheduling_result = solve_mission_schedule(scheduling_problem)
    scheduling_artifacts = write_mission_scheduling_artifacts(
        scheduling_result, sources / "mission_scheduling"
    )

    specifications = [
        (
            "trajectory-targeting-canonical-v1",
            "trajectory_targeting",
            trajectory_path,
            "Target a synthetic Hohmann-like apoapsis",
            "Did the bounded targeter reach and repropagate the declared apoapsis constraint?",
            "The targeter converges and authoritative repropagation verifies the terminal constraint.",
            "The bounded targeter converged and its authoritative repropagation status is verified.",
            "/authoritative_repropagation/status",
            "This local single-shooting result does not establish global optimality.",
        ),
        (
            "conjunction-assessment-canonical-v1",
            "conjunction_assessment",
            conjunction_path,
            "Assess a synthetic crossing encounter",
            "Did the bounded conjunction workflow complete its declared encounter assessment?",
            "The conjunction record completes and retains closest-approach geometry.",
            "The bounded conjunction assessment completed with retained closest-approach geometry.",
            "/baseline/closest_approach/miss_distance_km",
            "The educational probability and candidate screen are not operational collision authority.",
        ),
        (
            "mission-scheduling-canonical-v1",
            "mission_scheduling",
            scheduling_artifacts.summary_json,
            "Schedule two synthetic collection assets",
            "Did the exact bounded scheduler produce a complete feasible schedule?",
            "The exact scheduler returns complete status for the declared opportunity set.",
            "The exact bounded scheduler completed the declared synthetic opportunity problem.",
            "/objective_value",
            "The result does not establish large-scale, disrupted, or rolling-horizon performance.",
        ),
    ]
    summaries: list[dict[str, Any]] = []
    for (
        study_id,
        capability,
        evidence_path,
        title,
        question,
        criterion,
        claim,
        pointer,
        non_claim,
    ) in specifications:
        request, plan, claims = _records(
            study_id=study_id,
            capability=capability,
            title=title,
            question=question,
            criterion=criterion,
            claim=claim,
            evidence_pointer=pointer,
            non_claim=non_claim,
        )
        step_id = capability.replace("_", "-")
        bundle = build_study_bundle(
            request,
            plan,
            claims,
            {step_id: evidence_path},
            destination / study_id,
        )
        inspection = inspect_study_bundle(bundle.output_dir)
        replay = replay_study_bundle(bundle.output_dir)
        summaries.append(
            {
                "study_id": study_id,
                "capability": capability,
                "status": inspection["status"],
                "replay_status": replay["replay_status"],
                "bundle_semantic_sha256": inspection["bundle_semantic_sha256"],
            }
        )
    result = {
        "schema_version": "oel.study_lifecycle_example.v1",
        "status": "verified",
        "study_count": len(summaries),
        "studies": summaries,
    }
    (destination / "study_lifecycle_example_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_examples(args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
