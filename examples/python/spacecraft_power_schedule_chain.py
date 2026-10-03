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
from sim.analysis.schedule_power_study import build_schedule_power_study
from sim.analysis.spacecraft_power import SpacecraftPowerProblem, verify_spacecraft_power_artifacts
from sim.analysis.study_lifecycle import replay_study_bundle
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
    history = _orbit_history(base_problem)
    integrated = build_schedule_power_study(
        schedule_dir=schedule.output_dir,
        problem=base_problem,
        history=history,
        observation_load_w=180.0,
        downlink_load_w=120.0,
        schedule_epoch_jd_utc=base_problem.epoch_jd_utc,
        output_dir=destination / "schedule_power_study",
        study_id="spacecraft-power-schedule-canonical-v1",
        title="Assess schedule-coupled spacecraft power feasibility",
    )
    power_replay = verify_spacecraft_power_artifacts(destination / "schedule_power_study" / "power")
    lifecycle_replay = replay_study_bundle(destination / "schedule_power_study" / "study")
    result = {
        "schema_version": "oel.spacecraft_power_example.v1",
        "status": "verified",
        "schedule_semantic_sha256": integrated["schedule_semantic_sha256"],
        "power_feasibility": integrated["feasibility"],
        "power_replay_status": power_replay["status"],
        "power_result_semantic_sha256": integrated["power_result_semantic_sha256"],
        "study_status": integrated["status"],
        "study_replay_status": lifecycle_replay["replay_status"],
        "study_bundle_semantic_sha256": integrated["study_bundle_semantic_sha256"],
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
