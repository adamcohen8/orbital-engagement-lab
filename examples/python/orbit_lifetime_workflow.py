# ruff: noqa: E402
"""Run, compare, replay, and retain a public deterministic orbit-lifetime study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.analysis.orbit_lifetime import (
    OrbitLifetimeComparisonProblem,
    OrbitLifetimeProblem,
    assess_orbit_lifetime,
    compare_orbit_lifetime_models,
    verify_orbit_lifetime_artifacts,
    verify_orbit_lifetime_comparison_artifacts,
    write_orbit_lifetime_artifacts,
    write_orbit_lifetime_comparison_artifacts,
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _study_records() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    capability = "orbit_lifetime"
    request = {
        "schema_version": STUDY_REQUEST_SCHEMA,
        "study_id": "orbit-lifetime-canonical-v1",
        "title": "Assess one bounded low-orbit decay case",
        "question": "Does the declared synthetic case reach its reentry threshold within the horizon?",
        "capabilities": [capability],
        "assumptions": [
            "The initial state and spacecraft are synthetic.",
            "The atmosphere inputs are frozen modeling assumptions, not current weather.",
        ],
        "clarifications": [
            {
                "question": "Does lifecycle replay recompute the orbit-lifetime result?",
                "resolution": "No. Lifetime replay is authoritative; lifecycle replay verifies identity.",
            }
        ],
        "context": {
            "epoch": "Julian UTC epoch declared by the lifetime problem",
            "time_system": "elapsed SI seconds from epoch",
            "frame": "EME2000-compatible ECI",
            "units": "kilometres, seconds, kilograms, and square metres",
        },
        "fidelity": {
            "level": "bounded_public",
            "description": "ONP RK4 with declared drag, atmosphere, and refined altitude thresholds.",
        },
        "acceptance_criteria": [
            {
                "criterion_id": "reentry-threshold-reached",
                "description": "The completed analysis reaches the declared reentry threshold.",
            }
        ],
    }
    plan = {
        "schema_version": STUDY_PLAN_SCHEMA,
        "study_id": request["study_id"],
        "request_sha256": "auto",
        "resource_profile": "laptop-safe",
        "steps": [
            {
                "step_id": "orbit-lifetime",
                "capability": capability,
                "analysis_interface": CAPABILITY_CONTRACTS[capability]["analysis_interface"],
                "expected_evidence_schema": CAPABILITY_CONTRACTS[capability]["evidence_schema"],
                "depends_on": [],
                "acceptance_criterion_ids": ["reentry-threshold-reached"],
            }
        ],
    }
    claims = {
        "schema_version": STUDY_CLAIMS_SCHEMA,
        "study_id": request["study_id"],
        "plan_sha256": "auto",
        "claims": [
            {
                "claim_id": "claim-reentry-threshold-reached",
                "statement": "The declared synthetic case reaches its reentry threshold within the horizon.",
                "validation_level": "VC-1",
                "criterion_ids": ["reentry-threshold-reached"],
                "evidence": [{"step_id": "orbit-lifetime", "json_pointer": "/outcome"}],
            }
        ],
        "non_claims": [
            "This result does not predict current space weather or calibrated lifetime uncertainty.",
            "This result does not establish disposal compliance, reentry risk, or operational authority.",
        ],
    }
    return request, plan, claims


def build_example(output_root: str | Path) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"output_root must not already exist: {destination}.")
    destination.mkdir(parents=True)

    problem = OrbitLifetimeProblem.from_mapping(
        _read_json(ROOT / "examples/orbit_lifetime/public_low_orbit_decay_problem.json")
    )
    lifetime = write_orbit_lifetime_artifacts(
        assess_orbit_lifetime(problem), destination / "orbit_lifetime"
    )
    lifetime_replay = verify_orbit_lifetime_artifacts(lifetime.output_dir)

    comparison_problem = OrbitLifetimeComparisonProblem.from_mapping(
        _read_json(ROOT / "examples/orbit_lifetime/public_atmosphere_comparison.json")
    )
    comparison = write_orbit_lifetime_comparison_artifacts(
        compare_orbit_lifetime_models(comparison_problem), destination / "atmosphere_comparison"
    )
    comparison_replay = verify_orbit_lifetime_comparison_artifacts(comparison.output_dir)

    request, plan, claims = _study_records()
    study = build_study_bundle(
        request,
        plan,
        claims,
        {"orbit-lifetime": lifetime.summary_json},
        destination / "study",
    )
    inspection = inspect_study_bundle(study.output_dir)
    lifecycle_replay = replay_study_bundle(study.output_dir)
    result = {
        "schema_version": "oel.orbit_lifetime_example.v1",
        "status": "verified",
        "outcome": lifetime_replay["outcome"],
        "lifetime_replay_status": lifetime_replay["status"],
        "lifetime_result_semantic_sha256": lifetime_replay["result_semantic_sha256"],
        "comparison_case_count": comparison_replay["case_count"],
        "comparison_replay_status": comparison_replay["status"],
        "comparison_result_semantic_sha256": comparison_replay["result_semantic_sha256"],
        "study_status": inspection["status"],
        "study_replay_status": lifecycle_replay["replay_status"],
        "study_bundle_semantic_sha256": inspection["bundle_semantic_sha256"],
    }
    (destination / "orbit_lifetime_example_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_example(args.output_root), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
