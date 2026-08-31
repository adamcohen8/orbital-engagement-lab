# ruff: noqa: E402
"""Build and replay a public schedule-coupled spacecraft-power study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.analysis.conjunction_workflow import propagate_history
from sim.analysis.history_adapters import AnalysisHistory
from sim.analysis.mission_scheduling import (
    MissionSchedulingProblem,
    solve_mission_schedule,
    write_mission_scheduling_artifacts,
)
from sim.analysis.spacecraft_power import (
    SpacecraftPowerProblem,
    assess_spacecraft_power,
    problem_with_mission_schedule,
    verify_spacecraft_power_artifacts,
    write_spacecraft_power_artifacts,
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
from sim.analysis.trajectory_targeting import PropagationSettings
from sim.dynamics.orbit.epoch import resolve_sun_moon_positions


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _orbit_history(problem: SpacecraftPowerProblem) -> AnalysisHistory:
    sun, _ = resolve_sun_moon_positions(
        {"jd_utc_start": problem.epoch_jd_utc, "ephemeris_mode": problem.ephemeris_model},
        problem.horizon_start_s,
    )
    radial = sun / np.linalg.norm(sun)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(radial, trial))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    cross_track = np.cross(radial, trial)
    cross_track /= np.linalg.norm(cross_track)
    in_track = np.cross(cross_track, radial)
    radius_km = 7000.0
    speed_km_s = np.sqrt(398600.4418 / radius_km)
    initial_state = np.hstack((radius_km * radial, speed_km_s * in_track))
    propagated = propagate_history(
        initial_state,
        problem.horizon_end_s - problem.horizon_start_s,
        PropagationSettings(step_s=problem.integration_step_s),
    )
    times_s, states = propagated.arrays()
    return AnalysisHistory(
        object_id=problem.asset_id,
        product_kind="onp_two_body_example",
        state_provider_id="example:onp-two-body",
        frame="eci",
        initial_jd_utc=problem.epoch_jd_utc,
        times_s=times_s + problem.horizon_start_s,
        position_eci_km=states[:, :3],
        velocity_eci_km_s=states[:, 3:],
    )


def _study_records() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    capability = "spacecraft_power"
    step_id = "spacecraft-power"
    criterion_id = "power-feasible"
    request = {
        "schema_version": STUDY_REQUEST_SCHEMA,
        "study_id": "spacecraft-power-schedule-canonical-v1",
        "title": "Assess schedule-coupled spacecraft power feasibility",
        "question": "Can SAT-A serve the selected synthetic schedule without violating its declared battery reserve?",
        "capabilities": [capability],
        "assumptions": [
            "The public schedule and orbit are synthetic.",
            "The selected ideal Sun-tracking orientation is an explicit modeling assumption.",
        ],
        "clarifications": [
            {
                "question": "Does lifecycle replay recompute the power analysis?",
                "resolution": "No. Power replay is authoritative; lifecycle replay verifies retained identity.",
            }
        ],
        "context": {
            "epoch": "Julian UTC epoch declared by the power problem",
            "time_system": "elapsed SI seconds from epoch",
            "frame": "EME2000-compatible ECI",
            "units": "kilometres, seconds, watts, and watt-hours",
        },
        "fidelity": {
            "level": "bounded_public",
            "description": "Two-body sampled orbit, analytic Sun, conical shadow, and deterministic lumped battery.",
        },
        "acceptance_criteria": [
            {
                "criterion_id": criterion_id,
                "description": "The completed analysis reports feasible with zero unmet load.",
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
        "study_id": request["study_id"],
        "plan_sha256": "auto",
        "claims": [
            {
                "claim_id": "claim-power-feasible",
                "statement": "The declared synthetic schedule is power-feasible for this retained orbit and model.",
                "validation_level": "VC-1",
                "criterion_ids": [criterion_id],
                "evidence": [{"step_id": step_id, "json_pointer": "/feasibility"}],
            }
        ],
        "non_claims": [
            "This result does not qualify hardware or establish operational power margin.",
            "This result does not include thermal state, degradation, uncertainty, or self-shadowing.",
        ],
    }
    return request, plan, claims


def build_example(output_root: str | Path) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"output_root must not already exist: {destination}.")
    destination.mkdir(parents=True)

    scheduling_problem = MissionSchedulingProblem.from_mapping(
        _read_json(ROOT / "examples/mission_scheduling/public_two_asset_collection_problem.json")
    )
    schedule = write_mission_scheduling_artifacts(
        solve_mission_schedule(scheduling_problem), destination / "mission_schedule"
    )
    base_problem = SpacecraftPowerProblem.from_mapping(
        _read_json(ROOT / "examples/spacecraft_power/public_schedule_power_problem.json")
    )
    problem = problem_with_mission_schedule(
        base_problem,
        schedule.output_dir,
        activity_power_w={"observation": 180.0, "downlink": 120.0},
    )
    history = _orbit_history(problem)
    power = write_spacecraft_power_artifacts(
        assess_spacecraft_power(problem, history), history, destination / "spacecraft_power"
    )
    power_replay = verify_spacecraft_power_artifacts(power.output_dir)

    request, plan, claims = _study_records()
    bundle = build_study_bundle(
        request,
        plan,
        claims,
        {"spacecraft-power": power.summary_json},
        destination / "study",
    )
    inspection = inspect_study_bundle(bundle.output_dir)
    lifecycle_replay = replay_study_bundle(bundle.output_dir)
    result = {
        "schema_version": "oel.spacecraft_power_example.v1",
        "status": "verified",
        "schedule_semantic_sha256": problem.activities[0].source_product_sha256,
        "power_feasibility": power_replay["feasibility"],
        "power_replay_status": power_replay["status"],
        "power_result_semantic_sha256": power_replay["result_semantic_sha256"],
        "study_status": inspection["status"],
        "study_replay_status": lifecycle_replay["replay_status"],
        "study_bundle_semantic_sha256": inspection["bundle_semantic_sha256"],
    }
    (destination / "spacecraft_power_example_summary.json").write_text(
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
