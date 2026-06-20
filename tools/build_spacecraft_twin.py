#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.digital_twin.package import SpacecraftTwinPackage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and assemble an OEL spacecraft twin package."
    )
    parser.add_argument("twin", help="Path to twin.yaml.")
    parser.add_argument("--validate", action="store_true", help="Run twin-package validation.")
    parser.add_argument("--report", action="store_true", help="Write the Markdown validation report.")
    parser.add_argument("--report-path", default="", help="Override the report output path.")
    parser.add_argument("--emit-object-yaml", default="", help="Write assembled scenario object YAML.")
    parser.add_argument("--print-summary", action="store_true", help="Print compact JSON summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    package = SpacecraftTwinPackage.load(args.twin)
    validation = package.validate()
    report_path = None
    object_yaml_path = None
    if args.report:
        report_path = package.write_report(args.report_path or None)
    if str(args.emit_object_yaml).strip():
        object_yaml_path = package.write_object_yaml(args.emit_object_yaml)
    if args.print_summary or args.validate:
        summary = {
            "twin": str(Path(args.twin).expanduser().resolve()),
            "object_id": package.object_id,
            "display_name": package.display_name,
            "version": package.version,
            "ok": validation.ok,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "missing_inputs": validation.missing_inputs,
            "report": None if report_path is None else str(report_path.resolve()),
            "object_yaml": None if object_yaml_path is None else str(object_yaml_path.resolve()),
        }
        if validation.geometry_summary is not None:
            geom = validation.geometry_summary
            summary["geometry"] = {
                "path": str(geom.path),
                "sample_count": geom.sample_count,
                "area_min_m2": geom.area_min_m2,
                "area_mean_m2": geom.area_mean_m2,
                "area_max_m2": geom.area_max_m2,
                "confidence": geom.confidence,
            }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
