"""Stable public facade and CLI for conjunction assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.conjunction_geometry import (
    ConjunctionGeometryError,
    StateHistory,
    encounter_frame,
    interpolate_history,
    refine_time_of_closest_approach,
)
from sim.analysis.conjunction_probability import (
    ConjunctionProbabilityError,
    collision_probability_2d,
    covariance_rtn_si_to_eci_km,
    project_combined_covariance,
    ric_basis,
    validate_covariance,
)
from sim.analysis.conjunction_workflow import (
    CONJUNCTION_ASSESSMENT_EVIDENCE_SCHEMA,
    CONJUNCTION_ASSESSMENT_PROBLEM_SCHEMA,
    AvoidanceCandidate,
    ConjunctionAssessmentError,
    ConjunctionAssessmentProblem,
    ConjunctionObject,
    assess_cdm_message,
    assess_conjunction,
    assess_histories,
    propagate_history,
    write_conjunction_evidence,
)
from sim.interchange.ccsds_cdm import (
    CCSDS_CDM_PROFILE,
    CcsdsCdmError,
    CdmHeader,
    CdmMessage,
    CdmObject,
    CdmRelativeMetadata,
    compare_cdm,
    inspect_cdm,
    parse_cdm_kvn,
    read_cdm_kvn,
    serialize_cdm_kvn,
    validate_cdm,
    write_cdm_kvn,
)

__all__ = [
    "CCSDS_CDM_PROFILE",
    "CONJUNCTION_ASSESSMENT_EVIDENCE_SCHEMA",
    "CONJUNCTION_ASSESSMENT_PROBLEM_SCHEMA",
    "AvoidanceCandidate",
    "CcsdsCdmError",
    "CdmHeader",
    "CdmMessage",
    "CdmObject",
    "CdmRelativeMetadata",
    "ConjunctionAssessmentError",
    "ConjunctionAssessmentProblem",
    "ConjunctionGeometryError",
    "ConjunctionObject",
    "ConjunctionProbabilityError",
    "StateHistory",
    "assess_cdm_message",
    "assess_conjunction",
    "assess_histories",
    "collision_probability_2d",
    "compare_cdm",
    "covariance_rtn_si_to_eci_km",
    "encounter_frame",
    "inspect_cdm",
    "interpolate_history",
    "parse_cdm_kvn",
    "project_combined_covariance",
    "propagate_history",
    "read_cdm_kvn",
    "refine_time_of_closest_approach",
    "ric_basis",
    "serialize_cdm_kvn",
    "validate_cdm",
    "validate_covariance",
    "write_cdm_kvn",
    "write_conjunction_evidence",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.conjunction",
        description="Inspect CDMs and run deterministic public conjunction assessments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect-cdm", help="Validate and inspect one CCSDS CDM 1.0 KVN file.")
    inspect_parser.add_argument("path", type=Path)
    roundtrip = commands.add_parser("roundtrip-cdm", help="Parse and canonically serialize one CDM.")
    roundtrip.add_argument("source", type=Path)
    roundtrip.add_argument("destination", type=Path)
    assess_cdm_parser = commands.add_parser(
        "assess-cdm", help="Recompute instantaneous geometry and educational Pc from one CDM."
    )
    assess_cdm_parser.add_argument("path", type=Path)
    assess_cdm_parser.add_argument("--primary-radius-m", type=float, required=True)
    assess_cdm_parser.add_argument("--secondary-radius-m", type=float, required=True)
    assess = commands.add_parser("assess", help="Run one JSON conjunction-assessment problem.")
    assess.add_argument("problem", type=Path)
    assess.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-cdm":
            payload: Any = inspect_cdm(args.path)
        elif args.command == "roundtrip-cdm":
            message = read_cdm_kvn(args.source)
            target = write_cdm_kvn(message, args.destination)
            payload = {"status": "written", "path": str(target), "comparison": compare_cdm(message, target)}
        elif args.command == "assess-cdm":
            payload = assess_cdm_message(
                args.path, primary_radius_m=args.primary_radius_m, secondary_radius_m=args.secondary_radius_m
            )
        else:
            problem = json.loads(args.problem.read_text(encoding="utf-8"))
            payload = assess_conjunction(problem)
            if args.output:
                write_conjunction_evidence(payload, args.output)
                payload = {**payload, "evidence_path": str(args.output.resolve())}
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
