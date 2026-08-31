"""Stable public API and CLI for bounded multi-asset mission scheduling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.mission_scheduling import (
    MAX_PUBLIC_MISSION_OPPORTUNITIES,
    MISSION_SCHEDULING_EVIDENCE_SCHEMA,
    MISSION_SCHEDULING_PROBLEM_SCHEMA,
    AssetScheduleConstraints,
    MissionOpportunity,
    MissionSchedulingArtifacts,
    MissionSchedulingError,
    MissionSchedulingProblem,
    MissionSchedulingResult,
    ObservationDelivery,
    ScheduledActivity,
    replay_mission_schedule,
    solve_mission_schedule,
    verify_mission_scheduling_artifacts,
    write_mission_scheduling_artifacts,
)
from sim.analysis.mission_scheduling_sources import (
    MISSION_SCHEDULING_SOURCE_EVIDENCE_SCHEMA,
    MISSION_SCHEDULING_SOURCE_PLAN_SCHEMA,
    CollectionEvidenceSource,
    LinkEvidenceSource,
    MissionSchedulingSourcePlan,
    SourceBuiltMissionArtifacts,
    SourceBuiltMissionSchedule,
    VerifiedSourceProduct,
    build_mission_scheduling_problem_from_sources,
    build_solve_mission_schedule_from_sources,
    verify_source_built_mission_schedule,
    write_source_built_mission_schedule,
)

__all__ = [
    "MAX_PUBLIC_MISSION_OPPORTUNITIES",
    "MISSION_SCHEDULING_EVIDENCE_SCHEMA",
    "MISSION_SCHEDULING_PROBLEM_SCHEMA",
    "MISSION_SCHEDULING_SOURCE_EVIDENCE_SCHEMA",
    "MISSION_SCHEDULING_SOURCE_PLAN_SCHEMA",
    "AssetScheduleConstraints",
    "CollectionEvidenceSource",
    "LinkEvidenceSource",
    "MissionOpportunity",
    "MissionSchedulingArtifacts",
    "MissionSchedulingError",
    "MissionSchedulingProblem",
    "MissionSchedulingResult",
    "MissionSchedulingSourcePlan",
    "ObservationDelivery",
    "ScheduledActivity",
    "SourceBuiltMissionArtifacts",
    "SourceBuiltMissionSchedule",
    "VerifiedSourceProduct",
    "build_mission_scheduling_problem_from_sources",
    "build_solve_mission_schedule_from_sources",
    "replay_mission_schedule",
    "solve_mission_schedule",
    "verify_mission_scheduling_artifacts",
    "verify_source_built_mission_schedule",
    "write_mission_scheduling_artifacts",
    "write_source_built_mission_schedule",
]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MissionSchedulingError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MissionSchedulingError(f"Expected a JSON object in {path}.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.mission_scheduling",
        description="Solve or replay one bounded exact multi-asset mission schedule.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve", help="Solve a mission-scheduling problem JSON file.")
    solve.add_argument("problem", type=Path)
    solve.add_argument("--output-dir", type=Path, required=True)
    replay = commands.add_parser("replay", help="Authoritatively replay a generated evidence directory.")
    replay.add_argument("evidence_dir", type=Path)
    build_sources = commands.add_parser(
        "build-solve",
        help="Verify collection/link products, build the normalized problem, and solve it.",
    )
    build_sources.add_argument("source_plan", type=Path)
    build_sources.add_argument("--output-dir", type=Path, required=True)
    replay_sources = commands.add_parser(
        "replay-sources",
        help="Verify retained source products, rebuild the problem, and replay the exact optimum.",
    )
    replay_sources.add_argument("evidence_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "solve":
            problem = MissionSchedulingProblem.from_mapping(_read_json_object(args.problem))
            result = solve_mission_schedule(problem)
            artifacts = write_mission_scheduling_artifacts(result, args.output_dir)
            payload: dict[str, Any] = {**result.summary, "manifest": str(artifacts.manifest_json)}
        elif args.command == "replay":
            payload = verify_mission_scheduling_artifacts(args.evidence_dir)
        elif args.command == "build-solve":
            source_plan_path = args.source_plan.expanduser().resolve()
            source_plan = MissionSchedulingSourcePlan.from_mapping(_read_json_object(source_plan_path))
            artifacts = build_solve_mission_schedule_from_sources(
                source_plan,
                base_dir=source_plan_path.parent,
                output_dir=args.output_dir,
            )
            payload = _read_json_object(artifacts.manifest_json)
            payload["manifest"] = str(artifacts.manifest_json)
        else:
            payload = verify_source_built_mission_schedule(args.evidence_dir)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (MissionSchedulingError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
