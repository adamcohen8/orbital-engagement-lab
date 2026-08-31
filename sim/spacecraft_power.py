"""Stable public CLI and facade for deterministic spacecraft-power analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.history_adapters import history_from_review_store
from sim.analysis.spacecraft_power import (
    MAX_INTEGRATION_STEP_S,
    MAX_POWER_ACTIVITIES,
    MAX_POWER_DURATION_S,
    MAX_POWER_JSON_BYTES,
    MAX_POWER_SAMPLES,
    SPACECRAFT_POWER_EVIDENCE_SCHEMA,
    SPACECRAFT_POWER_HISTORY_SCHEMA,
    SPACECRAFT_POWER_MANIFEST_SCHEMA,
    SPACECRAFT_POWER_PROBLEM_SCHEMA,
    BatteryConfig,
    PowerActivity,
    PowerEvent,
    PowerInterval,
    PowerSample,
    SolarArrayConfig,
    SpacecraftPowerArtifacts,
    SpacecraftPowerError,
    SpacecraftPowerProblem,
    SpacecraftPowerResult,
    assess_spacecraft_power,
    power_history_from_mapping,
    power_history_to_dict,
    problem_with_mission_schedule,
    validate_spacecraft_power_inputs,
    verify_spacecraft_power_artifacts,
    write_spacecraft_power_artifacts,
)

__all__ = [
    "MAX_INTEGRATION_STEP_S",
    "MAX_POWER_ACTIVITIES",
    "MAX_POWER_DURATION_S",
    "MAX_POWER_JSON_BYTES",
    "MAX_POWER_SAMPLES",
    "SPACECRAFT_POWER_EVIDENCE_SCHEMA",
    "SPACECRAFT_POWER_HISTORY_SCHEMA",
    "SPACECRAFT_POWER_MANIFEST_SCHEMA",
    "SPACECRAFT_POWER_PROBLEM_SCHEMA",
    "BatteryConfig",
    "PowerActivity",
    "PowerEvent",
    "PowerInterval",
    "PowerSample",
    "SolarArrayConfig",
    "SpacecraftPowerArtifacts",
    "SpacecraftPowerError",
    "SpacecraftPowerProblem",
    "SpacecraftPowerResult",
    "assess_spacecraft_power",
    "power_history_from_mapping",
    "power_history_to_dict",
    "problem_with_mission_schedule",
    "validate_spacecraft_power_inputs",
    "verify_spacecraft_power_artifacts",
    "write_spacecraft_power_artifacts",
]


def _read_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SpacecraftPowerError(f"{field} must be a regular file: {path}.")
    size = path.stat().st_size
    if not 0 < size <= MAX_POWER_JSON_BYTES:
        raise SpacecraftPowerError(
            f"{field} must contain between 1 and {MAX_POWER_JSON_BYTES} bytes: {path}."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SpacecraftPowerError(f"Could not read {field} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SpacecraftPowerError(f"{field} must contain a JSON object.")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is not allowed.")


def _write_history(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SpacecraftPowerError(f"History output must not already exist: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.spacecraft_power",
        description="Assess, retain, and replay deterministic spacecraft-power evidence.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Analyze one normalized power problem and ECI history.")
    analyze.add_argument("problem", type=Path)
    analyze.add_argument("history", type=Path)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--mission-schedule", type=Path)
    analyze.add_argument("--observation-load-w", type=float)
    analyze.add_argument("--downlink-load-w", type=float)
    replay = commands.add_parser("replay", help="Verify receipts and authoritatively recompute evidence.")
    replay.add_argument("evidence_dir", type=Path)
    export = commands.add_parser(
        "export-review-history",
        help="Export one completed-run review-store object as a portable power history.",
    )
    export.add_argument("completed_run", type=Path)
    export.add_argument("--object-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="Validate and normalize a power problem and history.")
    validate.add_argument("problem", type=Path)
    validate.add_argument("history", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "replay":
            payload: Any = verify_spacecraft_power_artifacts(args.evidence_dir)
        elif args.command == "export-review-history":
            history = history_from_review_store(args.completed_run, object_id=args.object_id)
            value = power_history_to_dict(history)
            _write_history(args.output, value)
            payload = {
                "schema_version": SPACECRAFT_POWER_HISTORY_SCHEMA,
                "status": "exported",
                "asset_id": history.object_id,
                "sample_count": int(history.times_s.size),
                "output": str(args.output.expanduser().resolve()),
            }
        else:
            problem = SpacecraftPowerProblem.from_mapping(_read_json(args.problem, "power problem"))
            history = power_history_from_mapping(_read_json(args.history, "power history"))
            if args.command == "validate":
                problem, history = validate_spacecraft_power_inputs(problem, history)
                payload = {
                    "status": "valid",
                    "problem": problem.to_dict(),
                    "history": power_history_to_dict(history),
                }
            else:
                schedule_values = (
                    args.mission_schedule,
                    args.observation_load_w,
                    args.downlink_load_w,
                )
                if any(value is not None for value in schedule_values):
                    if any(value is None for value in schedule_values):
                        raise SpacecraftPowerError(
                            "--mission-schedule, --observation-load-w, and --downlink-load-w must be supplied together."
                        )
                    problem = problem_with_mission_schedule(
                        problem,
                        args.mission_schedule,
                        activity_power_w={
                            "observation": args.observation_load_w,
                            "downlink": args.downlink_load_w,
                        },
                    )
                result = assess_spacecraft_power(problem, history)
                artifacts = write_spacecraft_power_artifacts(result, history, args.output_dir)
                payload = {
                    **verify_spacecraft_power_artifacts(artifacts.output_dir),
                    "evidence_dir": str(artifacts.output_dir),
                    "summary": result.summary,
                }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, SpacecraftPowerError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
