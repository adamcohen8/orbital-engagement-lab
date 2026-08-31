"""Stable public API and CLI for deterministic trajectory targeting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.trajectory_targeting import (
    TRAJECTORY_TARGETING_EVIDENCE_SCHEMA,
    TRAJECTORY_TARGETING_PROBLEM_SCHEMA,
    DecisionVariable,
    EventRefinementError,
    MissedEventError,
    PropagationSettings,
    SolverSettings,
    TerminalConstraint,
    TrajectoryTargetingError,
    TrajectoryTargetingProblem,
    evaluate_terminal_constraints,
    execute_trajectory,
    finite_difference_jacobian,
    solve_trajectory_target,
    write_trajectory_targeting_evidence,
)

__all__ = [
    "TRAJECTORY_TARGETING_EVIDENCE_SCHEMA",
    "TRAJECTORY_TARGETING_PROBLEM_SCHEMA",
    "DecisionVariable",
    "EventRefinementError",
    "MissedEventError",
    "PropagationSettings",
    "SolverSettings",
    "TerminalConstraint",
    "TrajectoryTargetingError",
    "TrajectoryTargetingProblem",
    "evaluate_terminal_constraints",
    "execute_trajectory",
    "finite_difference_jacobian",
    "solve_trajectory_target",
    "write_trajectory_targeting_evidence",
]


def _read_problem(path: str | Path) -> TrajectoryTargetingProblem:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryTargetingError(f"Could not read trajectory-targeting problem {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrajectoryTargetingError("A trajectory-targeting problem must be a JSON object.")
    return TrajectoryTargetingProblem.from_mapping(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.trajectory_design",
        description="Propagate or solve one deterministic public OEL trajectory-targeting problem.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve", help="Run transparent single shooting and authoritative repropagation.")
    solve.add_argument("problem", help="Path to an oel.trajectory_targeting_problem.v1 JSON file.")
    solve.add_argument("--output", help="Evidence JSON path. Defaults to stdout only.")
    propagate = commands.add_parser("propagate", help="Execute the initial decision vector without correction.")
    propagate.add_argument("problem", help="Path to an oel.trajectory_targeting_problem.v1 JSON file.")
    propagate.add_argument("--output", help="Execution JSON path. Defaults to stdout only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        problem = _read_problem(args.problem)
        if args.command == "solve":
            result = solve_trajectory_target(problem)
            return_code = 0 if result["converged"] else 2
        else:
            execution = execute_trajectory(problem)
            result = {
                "schema_version": TRAJECTORY_TARGETING_EVIDENCE_SCHEMA,
                "problem_name": problem.name,
                "mode": "propagate",
                "execution": execution,
                "constraint_evaluation": evaluate_terminal_constraints(problem, execution),
            }
            return_code = 0
    except (TrajectoryTargetingError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 2
    if args.output:
        write_trajectory_targeting_evidence(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
