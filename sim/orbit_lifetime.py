"""Stable public facade and CLI for bounded orbit-lifetime analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.orbit_lifetime import (
    HARRIS_PRIESTER_SUPPORTED_F107,
    MAX_LIFETIME_ARTIFACT_BYTES,
    MAX_LIFETIME_COMPARISON_CASES,
    MAX_LIFETIME_DURATION_S,
    MAX_LIFETIME_EPOCH_JD_UTC,
    MAX_LIFETIME_INTEGRATION_STEPS,
    MAX_LIFETIME_JSON_BYTES,
    MAX_LIFETIME_OUTPUT_SAMPLES,
    MIN_LIFETIME_EPOCH_JD_UTC,
    ORBIT_LIFETIME_COMPARISON_EVIDENCE_SCHEMA,
    ORBIT_LIFETIME_COMPARISON_MANIFEST_SCHEMA,
    ORBIT_LIFETIME_COMPARISON_PROBLEM_SCHEMA,
    ORBIT_LIFETIME_EVIDENCE_SCHEMA,
    ORBIT_LIFETIME_MANIFEST_SCHEMA,
    ORBIT_LIFETIME_PROBLEM_SCHEMA,
    LifetimeAtmosphere,
    LifetimeComparisonCase,
    LifetimeEvent,
    LifetimeSample,
    LifetimeThresholds,
    OrbitLifetimeArtifacts,
    OrbitLifetimeComparisonArtifacts,
    OrbitLifetimeComparisonProblem,
    OrbitLifetimeComparisonResult,
    OrbitLifetimeError,
    OrbitLifetimeProblem,
    OrbitLifetimeResult,
    assess_orbit_lifetime,
    compare_orbit_lifetime_models,
    verify_orbit_lifetime_artifacts,
    verify_orbit_lifetime_comparison_artifacts,
    write_orbit_lifetime_artifacts,
    write_orbit_lifetime_comparison_artifacts,
)

__all__ = [
    "HARRIS_PRIESTER_SUPPORTED_F107",
    "MAX_LIFETIME_EPOCH_JD_UTC",
    "MAX_LIFETIME_ARTIFACT_BYTES",
    "MAX_LIFETIME_COMPARISON_CASES",
    "MAX_LIFETIME_DURATION_S",
    "MAX_LIFETIME_INTEGRATION_STEPS",
    "MAX_LIFETIME_JSON_BYTES",
    "MAX_LIFETIME_OUTPUT_SAMPLES",
    "MIN_LIFETIME_EPOCH_JD_UTC",
    "ORBIT_LIFETIME_COMPARISON_EVIDENCE_SCHEMA",
    "ORBIT_LIFETIME_COMPARISON_MANIFEST_SCHEMA",
    "ORBIT_LIFETIME_COMPARISON_PROBLEM_SCHEMA",
    "ORBIT_LIFETIME_EVIDENCE_SCHEMA",
    "ORBIT_LIFETIME_MANIFEST_SCHEMA",
    "ORBIT_LIFETIME_PROBLEM_SCHEMA",
    "LifetimeAtmosphere",
    "LifetimeComparisonCase",
    "LifetimeEvent",
    "LifetimeSample",
    "LifetimeThresholds",
    "OrbitLifetimeArtifacts",
    "OrbitLifetimeComparisonArtifacts",
    "OrbitLifetimeComparisonProblem",
    "OrbitLifetimeComparisonResult",
    "OrbitLifetimeError",
    "OrbitLifetimeProblem",
    "OrbitLifetimeResult",
    "assess_orbit_lifetime",
    "compare_orbit_lifetime_models",
    "verify_orbit_lifetime_artifacts",
    "verify_orbit_lifetime_comparison_artifacts",
    "write_orbit_lifetime_artifacts",
    "write_orbit_lifetime_comparison_artifacts",
]


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is not allowed.")


def _read_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OrbitLifetimeError(f"{field} must be a regular file: {path}.")
    size = path.stat().st_size
    if not 0 < size <= MAX_LIFETIME_JSON_BYTES:
        raise OrbitLifetimeError(f"{field} has an invalid byte size: {path}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OrbitLifetimeError(f"Could not read {field}: {exc}") from exc
    if not isinstance(value, dict):
        raise OrbitLifetimeError(f"{field} must contain a JSON object.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.orbit_lifetime",
        description="Run, compare, retain, and replay bounded ONP orbit-lifetime evidence.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Run one normalized lifetime problem.")
    analyze.add_argument("problem", type=Path)
    analyze.add_argument("--output-dir", type=Path, required=True)
    compare = commands.add_parser("compare", help="Run one normalized atmosphere-model comparison.")
    compare.add_argument("comparison", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)
    replay = commands.add_parser("replay", help="Authoritatively replay one lifetime evidence directory.")
    replay.add_argument("evidence_dir", type=Path)
    replay_comparison = commands.add_parser(
        "replay-comparison", help="Authoritatively replay one comparison evidence directory."
    )
    replay_comparison.add_argument("evidence_dir", type=Path)
    validate = commands.add_parser("validate", help="Validate and normalize one lifetime problem.")
    validate.add_argument("problem", type=Path)
    validate_comparison = commands.add_parser(
        "validate-comparison", help="Validate and normalize one atmosphere-model comparison."
    )
    validate_comparison.add_argument("comparison", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "replay":
            payload: Any = verify_orbit_lifetime_artifacts(args.evidence_dir)
        elif args.command == "replay-comparison":
            payload = verify_orbit_lifetime_comparison_artifacts(args.evidence_dir)
        elif args.command in {"compare", "validate-comparison"}:
            comparison = OrbitLifetimeComparisonProblem.from_mapping(
                _read_json(args.comparison, "orbit-lifetime comparison")
            )
            if args.command == "validate-comparison":
                payload = {"status": "valid", "comparison": comparison.to_dict()}
            else:
                result = compare_orbit_lifetime_models(comparison)
                artifacts = write_orbit_lifetime_comparison_artifacts(result, args.output_dir)
                payload = {
                    **verify_orbit_lifetime_comparison_artifacts(artifacts.output_dir),
                    "evidence_dir": str(artifacts.output_dir),
                    "summary": result.summary,
                }
        else:
            problem = OrbitLifetimeProblem.from_mapping(
                _read_json(args.problem, "orbit-lifetime problem")
            )
            if args.command == "validate":
                payload = {"status": "valid", "problem": problem.to_dict()}
            else:
                result = assess_orbit_lifetime(problem)
                artifacts = write_orbit_lifetime_artifacts(result, args.output_dir)
                payload = {
                    **verify_orbit_lifetime_artifacts(artifacts.output_dir),
                    "evidence_dir": str(artifacts.output_dir),
                    "summary": result.summary,
                }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ArithmeticError, OSError, OrbitLifetimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
