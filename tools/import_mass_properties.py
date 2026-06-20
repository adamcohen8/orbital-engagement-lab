#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.digital_twin.mass_properties import import_mass_properties, mass_property_report_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import CAD/exported mass properties into an OEL specs snippet. "
            "Supports OEL-shaped YAML/JSON and simple generic fields such as mass, center_of_mass, and inertia."
        )
    )
    parser.add_argument("input", help="Input JSON/YAML mass-property export.")
    parser.add_argument("-o", "--output", required=True, help="Output YAML snippet path.")
    parser.add_argument("--report", default="", help="Optional Markdown audit report path.")
    parser.add_argument(
        "--source",
        default="cad_export",
        choices=["user_supplied", "cad_export", "oel_estimate", "mesh_uniform_density_estimate", "preset", "unknown"],
        help="Source label stored in mass_properties.source.",
    )
    parser.add_argument(
        "--confidence",
        default="high",
        choices=["high", "medium", "low", "assumed", "unknown"],
        help="Confidence label stored in mass_properties.confidence.",
    )
    parser.add_argument(
        "--frame",
        default="body",
        choices=["body", "body_frame", "principal_axes", "unknown"],
        help="Frame label stored in mass_properties.frame.",
    )
    parser.add_argument(
        "--inertia-reference-point",
        default="center_of_mass",
        choices=["center_of_mass", "body_origin", "body_frame_origin", "unknown"],
        help="Reference point label stored in mass_properties.inertia_reference_point.",
    )
    parser.add_argument("--summary", action="store_true", help="Print a compact JSON summary.")
    return parser


def _write_yaml(path: str | Path, data: dict) -> Path:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to write mass-property YAML snippets.") from exc
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def main() -> int:
    args = build_parser().parse_args()
    result = import_mass_properties(
        args.input,
        source=args.source,
        confidence=args.confidence,
        frame=args.frame,
        inertia_reference_point=args.inertia_reference_point,
    )
    out = _write_yaml(args.output, result.snippet)
    report_path = None
    if str(args.report).strip():
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            mass_property_report_markdown(result.snippet, source_path=args.input),
            encoding="utf-8",
        )
    if args.summary:
        audit = result.audit
        summary = {
            "output": str(out.resolve()),
            "report": None if report_path is None else str(report_path.resolve()),
            "mass_kg": audit.mass_kg,
            "source": audit.source,
            "confidence": audit.confidence,
            "warnings": audit.validation.warnings,
        }
        if audit.principal_moments_kg_m2 is not None:
            summary["principal_moments_kg_m2"] = [float(v) for v in audit.principal_moments_kg_m2]
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
