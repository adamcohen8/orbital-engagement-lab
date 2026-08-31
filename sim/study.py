"""Stable public facade and CLI for content-bound OEL study bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.study_lifecycle import (
    CAPABILITY_CONTRACTS,
    MAX_STUDY_EVIDENCE_BYTES,
    MAX_STUDY_STEPS,
    STUDY_CLAIMS_SCHEMA,
    STUDY_COMPARISON_SCHEMA,
    STUDY_EVIDENCE_SCHEMA,
    STUDY_PLAN_SCHEMA,
    STUDY_RECEIPT_SCHEMA,
    STUDY_REQUEST_SCHEMA,
    STUDY_RUN_SCHEMA,
    STUDY_VERIFICATION_SCHEMA,
    StudyBundleArtifacts,
    StudyClaims,
    StudyLifecycleError,
    StudyPlan,
    StudyRequest,
    build_study_bundle,
    compare_study_bundles,
    inspect_study_bundle,
    replay_study_bundle,
    verify_study_bundle,
)

__all__ = [
    "CAPABILITY_CONTRACTS",
    "MAX_STUDY_EVIDENCE_BYTES",
    "MAX_STUDY_STEPS",
    "STUDY_CLAIMS_SCHEMA",
    "STUDY_COMPARISON_SCHEMA",
    "STUDY_EVIDENCE_SCHEMA",
    "STUDY_PLAN_SCHEMA",
    "STUDY_RECEIPT_SCHEMA",
    "STUDY_REQUEST_SCHEMA",
    "STUDY_RUN_SCHEMA",
    "STUDY_VERIFICATION_SCHEMA",
    "StudyBundleArtifacts",
    "StudyClaims",
    "StudyLifecycleError",
    "StudyPlan",
    "StudyRequest",
    "build_study_bundle",
    "compare_study_bundles",
    "inspect_study_bundle",
    "replay_study_bundle",
    "verify_study_bundle",
]


def _read_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyLifecycleError(f"Could not read {field} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudyLifecycleError(f"{field} must contain a JSON object.")
    return value


def _evidence_bindings(values: list[str]) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise StudyLifecycleError("--evidence must use STEP_ID=PATH syntax.")
        step_id, path = value.split("=", 1)
        step_id = step_id.strip()
        if not step_id or not path.strip():
            raise StudyLifecycleError("--evidence must use non-empty STEP_ID=PATH values.")
        if step_id in bindings:
            raise StudyLifecycleError(f"Duplicate --evidence binding for {step_id!r}.")
        bindings[step_id] = Path(path)
    return bindings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.study",
        description="Build, inspect, replay, and compare content-bound OEL study bundles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    request = commands.add_parser("validate-request", help="Validate and normalize one study request.")
    request.add_argument("request", type=Path)
    plan = commands.add_parser("validate-plan", help="Validate and bind one study plan to its request.")
    plan.add_argument("request", type=Path)
    plan.add_argument("plan", type=Path)
    claims = commands.add_parser("validate-claims", help="Validate claims against a bound request and plan.")
    claims.add_argument("request", type=Path)
    claims.add_argument("plan", type=Path)
    claims.add_argument("claims", type=Path)
    build = commands.add_parser("build", help="Build a new study bundle from completed evidence JSON files.")
    build.add_argument("request", type=Path)
    build.add_argument("plan", type=Path)
    build.add_argument("claims", type=Path)
    build.add_argument("--evidence", action="append", default=[], metavar="STEP_ID=PATH", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Verify and summarize one study bundle.")
    inspect.add_argument("bundle", type=Path)
    replay = commands.add_parser("replay", help="Rebuild and verify the study identity graph.")
    replay.add_argument("bundle", type=Path)
    compare = commands.add_parser("compare", help="Compare two verified study bundles.")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-request":
            request = StudyRequest.from_mapping(_read_json_object(args.request, "study request"))
            payload: Any = {
                "status": "valid",
                "request": request.to_dict(),
            }
        elif args.command == "validate-plan":
            request = StudyRequest.from_mapping(_read_json_object(args.request, "study request"))
            plan = StudyPlan.from_mapping(_read_json_object(args.plan, "study plan"), request)
            payload = {"status": "valid", "plan": plan.to_dict()}
        elif args.command == "validate-claims":
            request = StudyRequest.from_mapping(_read_json_object(args.request, "study request"))
            plan = StudyPlan.from_mapping(_read_json_object(args.plan, "study plan"), request)
            claims = StudyClaims.from_mapping(
                _read_json_object(args.claims, "study claims"), request, plan
            )
            payload = {"status": "valid", "claims": claims.to_dict()}
        elif args.command == "build":
            artifacts = build_study_bundle(
                _read_json_object(args.request, "study request"),
                _read_json_object(args.plan, "study plan"),
                _read_json_object(args.claims, "study claims"),
                _evidence_bindings(args.evidence),
                args.output_dir,
            )
            payload = inspect_study_bundle(artifacts.output_dir)
        elif args.command == "inspect":
            payload = inspect_study_bundle(args.bundle)
        elif args.command == "replay":
            payload = replay_study_bundle(args.bundle)
        else:
            payload = compare_study_bundles(args.left, args.right)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, StudyLifecycleError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
