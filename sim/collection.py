"""Stable public API and CLI for optical collection-opportunity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.collection_opportunity import (
    COLLECTION_OPPORTUNITY_EVIDENCE_SCHEMA,
    COLLECTION_OPPORTUNITY_PROBLEM_SCHEMA,
    CollectionOpportunityError,
    CollectionOpportunityProblem,
    SpacecraftSource,
    assess_collection_opportunities,
    write_collection_evidence,
)
from sim.analysis.collection_opportunity_resources import (
    CollectionResources,
    DownlinkWindowInput,
    screen_collection_resources,
)
from sim.analysis.optical_collection import (
    COLLECTION_REASON_NAMES,
    OPTICAL_COLLECTION_MODEL,
    CollectionConstraints,
    GroundTarget,
    OpticalPayload,
    evaluate_collection_sample,
    footprint_boundary_evidence,
    local_nadir_frame_sensor_from_eci,
    optical_quality_metrics,
    sensor_frame_and_gimbal_vector,
)

__all__ = [
    "COLLECTION_OPPORTUNITY_EVIDENCE_SCHEMA",
    "COLLECTION_OPPORTUNITY_PROBLEM_SCHEMA",
    "COLLECTION_REASON_NAMES",
    "OPTICAL_COLLECTION_MODEL",
    "CollectionConstraints",
    "CollectionOpportunityError",
    "CollectionOpportunityProblem",
    "CollectionResources",
    "DownlinkWindowInput",
    "GroundTarget",
    "OpticalPayload",
    "SpacecraftSource",
    "assess_collection_opportunities",
    "evaluate_collection_sample",
    "footprint_boundary_evidence",
    "local_nadir_frame_sensor_from_eci",
    "optical_quality_metrics",
    "screen_collection_resources",
    "sensor_frame_and_gimbal_vector",
    "write_collection_evidence",
]


def _read_problem(path: str | Path) -> CollectionOpportunityProblem:
    source = Path(path)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CollectionOpportunityError(f"Duplicate JSON field {key!r} is ambiguous.")
            result[key] = value
        return result

    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionOpportunityError(f"Could not read collection problem {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CollectionOpportunityError("A collection-opportunity problem must be a JSON object.")
    return CollectionOpportunityProblem.from_mapping(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.collection",
        description="Evaluate one deterministic public optical collection-opportunity problem.",
    )
    parser.add_argument("problem", help="Path to an oel.collection_opportunity_problem.v1 JSON file.")
    parser.add_argument("--output", help="Evidence JSON path. Defaults to stdout only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = assess_collection_opportunities(_read_problem(args.problem))
        if args.output:
            write_collection_evidence(result, args.output)
    except (CollectionOpportunityError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
