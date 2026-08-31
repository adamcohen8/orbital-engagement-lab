"""Stable public API and CLI for bounded constellation-design trades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.constellation_design import (
    CONSTELLATION_DESIGN_EVIDENCE_SCHEMA,
    CONSTELLATION_DESIGN_PROBLEM_SCHEMA,
    MAX_PUBLIC_DESIGNS,
    MAX_PUBLIC_SATELLITES_PER_DESIGN,
    ConstellationCandidate,
    ConstellationDesignArtifacts,
    ConstellationDesignError,
    ConstellationDesignProblem,
    ConstellationDesignResult,
    GroundSite,
    generate_constellation_members,
    solve_constellation_design,
    verify_constellation_design_artifacts,
    write_constellation_design_artifacts,
)

__all__ = [
    "CONSTELLATION_DESIGN_EVIDENCE_SCHEMA",
    "CONSTELLATION_DESIGN_PROBLEM_SCHEMA",
    "MAX_PUBLIC_DESIGNS",
    "MAX_PUBLIC_SATELLITES_PER_DESIGN",
    "ConstellationCandidate",
    "ConstellationDesignArtifacts",
    "ConstellationDesignError",
    "ConstellationDesignProblem",
    "ConstellationDesignResult",
    "GroundSite",
    "generate_constellation_members",
    "solve_constellation_design",
    "verify_constellation_design_artifacts",
    "write_constellation_design_artifacts",
]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstellationDesignError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConstellationDesignError(f"Expected a JSON object in {path}.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.constellation_design",
        description="Validate, solve, or replay one bounded deterministic constellation-design trade.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate and normalize a problem JSON file.")
    validate.add_argument("problem", type=Path)
    solve = commands.add_parser("solve", help="Evaluate and rank the explicit candidate inventory.")
    solve.add_argument("problem", type=Path)
    solve.add_argument("--output-dir", type=Path, required=True)
    replay = commands.add_parser("replay", help="Authoritatively replay a generated evidence directory.")
    replay.add_argument("evidence_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            problem = ConstellationDesignProblem.from_mapping(_read_json_object(args.problem))
            payload: dict[str, Any] = {
                "schema_version": CONSTELLATION_DESIGN_PROBLEM_SCHEMA,
                "analysis_id": problem.analysis_id,
                "status": "valid",
                "candidate_count": len(problem.designs),
            }
        elif args.command == "solve":
            problem = ConstellationDesignProblem.from_mapping(_read_json_object(args.problem))
            result = solve_constellation_design(problem)
            artifacts = write_constellation_design_artifacts(result, args.output_dir)
            payload = {
                "schema_version": CONSTELLATION_DESIGN_EVIDENCE_SCHEMA,
                "analysis_id": problem.analysis_id,
                "status": result.evidence["status"],
                "recommended_design_id": result.evidence["recommended_design_id"],
                "ranking": result.evidence["ranking"],
                "manifest": str(artifacts.manifest_json),
            }
        else:
            payload = verify_constellation_design_artifacts(args.evidence_dir)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ConstellationDesignError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
