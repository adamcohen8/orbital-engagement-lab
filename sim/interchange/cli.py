from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .comparison import compare_handoff
from .completed_runs import export_completed_run_state
from .inspection import inspect_path
from .maneuver_detection import export_maneuver_detection_product
from .materialization import materialize_ogp, materialize_onp
from .overlays import emit_scenario_overlay, load_scenario_overlay
from .scenario_patches import materialize_scenario_patch, select_patch_product
from .snapshots import export_completed_run_snapshot, materialize_snapshot_onp
from .validation import load_interchange_document, validate_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.handoff",
        description=(
            "Inspect versioned OEL interchange products or materialize a validated ONP scenario; "
            "handoff commands never execute scenarios."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a product or handoff manifest read-only.")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    inspect_parser.add_argument(
        "--no-verify-sources",
        action="store_true",
        help="Skip source-file fingerprint comparison; schema and semantic validation still run.",
    )

    validate_parser = subparsers.add_parser("validate-product", help="Validate a Product Envelope v1.")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    validate_parser.add_argument(
        "--no-verify-sources",
        action="store_true",
        help="Skip source-file fingerprint comparison; schema and semantic validation still run.",
    )

    compare_parser = subparsers.add_parser(
        "compare-handoff",
        help="Write a read-only semantic-parity packet for a product, materialized scenario, and manifest.",
    )
    compare_parser.add_argument("--product", required=True, type=Path)
    compare_parser.add_argument("--scenario", required=True, type=Path)
    compare_parser.add_argument("--manifest", type=Path, help="Defaults to <scenario-stem>.handoff_manifest.json.")
    compare_parser.add_argument("--run-output-dir", type=Path, help="Optionally compare the first review-store state row.")
    compare_parser.add_argument("--output", required=True, type=Path, help="Comparison packet JSON path.")
    compare_parser.add_argument("--json", action="store_true")

    export_parser = subparsers.add_parser(
        "export-state",
        help="Export one exact ECI state from a completed review store without executing a scenario.",
    )
    export_parser.add_argument("completed_run", type=Path)
    export_parser.add_argument("--output", required=True, type=Path, help="Completed-run state product JSON path.")
    export_parser.add_argument("--object-id", help="Required when the completed run has multiple eligible objects.")
    selection = export_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample", choices=("final",), help="Select the object's final recorded sample.")
    selection.add_argument("--sample-index", type=int, help="Select one exact object sample_index.")
    selection.add_argument("--time-s", type=float, help="Select one exact object-state time_s.")
    selection.add_argument("--event-id", help="Select the sample associated with one exact event_id.")
    export_parser.add_argument(
        "--epoch-jd-utc",
        type=float,
        help="Explicit epoch anchor for a relative-time source run; must match a configured source epoch when present.",
    )
    export_parser.add_argument("--overwrite", action="store_true")
    export_parser.add_argument("--json", action="store_true")

    detection_parser = subparsers.add_parser(
        "export-maneuver-detection",
        help="Export one confirmed maneuver-detection event and its detector evidence.",
    )
    detection_parser.add_argument("completed_run", type=Path)
    detection_parser.add_argument("--output", required=True, type=Path)
    detection_parser.add_argument("--event-id")
    detection_parser.add_argument("--observer-id")
    detection_parser.add_argument("--target-id")
    detection_parser.add_argument("--overwrite", action="store_true")
    detection_parser.add_argument("--json", action="store_true")

    snapshot_parser = subparsers.add_parser(
        "export-snapshot",
        help="Export an atomic multi-object ECI snapshot from one completed-run sample.",
    )
    snapshot_parser.add_argument("completed_run", type=Path)
    snapshot_parser.add_argument("--output", required=True, type=Path)
    snapshot_parser.add_argument("--object-id", action="append", required=True)
    snapshot_selection = snapshot_parser.add_mutually_exclusive_group(required=True)
    snapshot_selection.add_argument("--sample", choices=("final",))
    snapshot_selection.add_argument("--sample-index", type=int)
    snapshot_selection.add_argument("--time-s", type=float)
    snapshot_selection.add_argument("--event-id")
    snapshot_parser.add_argument("--epoch-jd-utc", type=float)
    snapshot_parser.add_argument("--overwrite", action="store_true")
    snapshot_parser.add_argument("--json", action="store_true")

    snapshot_onp_parser = subparsers.add_parser(
        "materialize-snapshot-onp",
        help="Materialize a passive multi-object ONP continuation from an atomic snapshot.",
    )
    snapshot_onp_parser.add_argument("--snapshot-product", required=True, type=Path)
    snapshot_onp_parser.add_argument("--scenario-name", required=True)
    snapshot_onp_parser.add_argument("--output", required=True, type=Path)
    snapshot_onp_parser.add_argument("--run-output-dir", required=True, type=Path)
    snapshot_onp_parser.add_argument("--duration-s", required=True, type=float)
    snapshot_onp_parser.add_argument("--dt-s", required=True, type=float)
    snapshot_onp_parser.add_argument("--trust-plugins", action="store_true")
    snapshot_onp_parser.add_argument("--overwrite", action="store_true")
    snapshot_onp_parser.add_argument("--json", action="store_true")

    overlay_parser = subparsers.add_parser(
        "emit-overlay",
        help="Emit a bounded typed scenario-capability overlay without executing or materializing it.",
    )
    overlay_parser.add_argument("--source-scenario", required=True, type=Path)
    overlay_parser.add_argument("--overlay", required=True, type=Path)
    overlay_parser.add_argument("--overlay-id", required=True)
    overlay_parser.add_argument("--rationale", required=True)
    overlay_parser.add_argument("--output", required=True, type=Path)
    overlay_parser.add_argument("--json", action="store_true")

    onp_parser = subparsers.add_parser(
        "materialize-onp",
        help="Materialize and validate a passive ONP scenario from an accepted state product without executing it.",
    )
    onp_parser.add_argument("--state-product", required=True, type=Path)
    onp_parser.add_argument("--scenario-name", required=True)
    onp_parser.add_argument("--output", required=True, type=Path, help="Generated scenario YAML path.")
    onp_parser.add_argument("--run-output-dir", required=True, type=Path)
    onp_parser.add_argument("--duration-s", required=True, type=float)
    onp_parser.add_argument("--dt-s", required=True, type=float)
    onp_parser.add_argument("--manifest", type=Path, help="Optional exact manifest path.")
    onp_parser.add_argument(
        "--trust-plugins",
        action="store_true",
        help="Run ordinary plugin-importing validation after safe validation.",
    )
    onp_parser.add_argument("--overwrite", action="store_true", help="Explicitly replace a different output file.")
    onp_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    ogp_parser = subparsers.add_parser(
        "materialize-ogp",
        help="Materialize and validate passive OGP propagation from fitted native mean elements without executing it.",
    )
    ogp_parser.add_argument("--mean-element-product", required=True, type=Path)
    ogp_parser.add_argument("--scenario-name", required=True)
    ogp_parser.add_argument("--output", required=True, type=Path)
    ogp_parser.add_argument("--run-output-dir", required=True, type=Path)
    ogp_parser.add_argument("--duration-s", required=True, type=float)
    ogp_parser.add_argument("--dt-s", required=True, type=float)
    ogp_parser.add_argument("--manifest", type=Path)
    ogp_parser.add_argument("--trust-plugins", action="store_true")
    ogp_parser.add_argument("--overwrite", action="store_true")
    ogp_parser.add_argument("--json", action="store_true")

    patch_parser = subparsers.add_parser(
        "materialize-scenario-patch",
        help="Select, apply, and validate a typed scenario patch without executing the scenario.",
    )
    source_group = patch_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--patch-product", type=Path)
    source_group.add_argument("--patch-index", type=Path)
    patch_parser.add_argument("--selection-id", help="Required with --patch-index; selects one exact candidate/variant.")
    patch_parser.add_argument("--source-scenario", required=True, type=Path)
    patch_parser.add_argument("--scenario-name", required=True)
    patch_parser.add_argument("--output", required=True, type=Path)
    patch_parser.add_argument("--run-output-dir", required=True, type=Path)
    patch_parser.add_argument("--manifest", type=Path)
    patch_parser.add_argument("--trust-plugins", action="store_true")
    patch_parser.add_argument("--overwrite", action="store_true")
    patch_parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_path(args.path, verify_sources=not args.no_verify_sources)
            _print_inspection(result, json_mode=args.json)
            return 0 if result["validation"]["valid"] else 2
        if args.command == "validate-product":
            document = load_interchange_document(args.path)
            report = validate_document(
                document,
                source_path=args.path,
                verify_sources=not args.no_verify_sources,
            )
            payload = report.to_dict()
            _print_validation(payload, json_mode=args.json)
            if not report.valid:
                return 2
            return 0 if report.promotable else 3
        if args.command == "compare-handoff":
            payload = compare_handoff(
                args.product,
                args.scenario,
                manifest_path=args.manifest,
                run_output_dir=args.run_output_dir,
                output_path=args.output,
            )
            _print_handoff_comparison(payload, json_mode=args.json)
            return 0 if payload.get("status") == "equivalent" else 2
        if args.command == "export-state":
            if args.sample_index is not None:
                selector = "sample_index"
            elif args.time_s is not None:
                selector = "time_s"
            elif args.event_id is not None:
                selector = "event"
            else:
                selector = "final"
            payload = export_completed_run_state(
                args.completed_run,
                output_path=args.output,
                object_id=args.object_id,
                selector=selector,
                sample_index=args.sample_index,
                time_s=args.time_s,
                event_id=args.event_id,
                epoch_jd_utc=args.epoch_jd_utc,
                overwrite=args.overwrite,
            )
            _print_state_export(payload, json_mode=args.json)
            return 0
        if args.command == "export-maneuver-detection":
            payload = export_maneuver_detection_product(
                args.completed_run,
                output_path=args.output,
                event_id=args.event_id,
                observer_id=args.observer_id,
                target_id=args.target_id,
                overwrite=args.overwrite,
            )
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"status: {payload.get('status', '')}")
                print(f"product_path: {payload.get('product_path', '')}")
                print(f"product_id: {payload.get('product_id', '')}")
                print(f"event_id: {payload.get('event_id', '')}")
                print("execution_occurred: false")
            return 0
        if args.command == "export-snapshot":
            if args.sample_index is not None:
                selector = "sample_index"
            elif args.time_s is not None:
                selector = "time_s"
            elif args.event_id is not None:
                selector = "event"
            else:
                selector = "final"
            payload = export_completed_run_snapshot(
                args.completed_run,
                output_path=args.output,
                object_ids=args.object_id,
                selector=selector,
                sample_index=args.sample_index,
                time_s=args.time_s,
                event_id=args.event_id,
                epoch_jd_utc=args.epoch_jd_utc,
                overwrite=args.overwrite,
            )
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"status: {payload.get('status', '')}")
                print(f"product_path: {payload.get('product_path', '')}")
                print(f"product_id: {payload.get('product_id', '')}")
                print(f"object_ids: {', '.join(payload.get('object_ids', []))}")
                print("execution_occurred: false")
            return 0
        if args.command == "materialize-snapshot-onp":
            payload = materialize_snapshot_onp(
                args.snapshot_product,
                scenario_name=args.scenario_name,
                scenario_path=args.output,
                output_dir=args.run_output_dir,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                trust_plugins=args.trust_plugins,
                overwrite=args.overwrite,
            )
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"status: {payload.get('status', '')}")
                print(f"scenario_path: {payload.get('scenario_path', '')}")
                print(f"object_count: {payload.get('object_count', 0)}")
                print("execution_occurred: false")
            return 0 if payload.get("status") == "materialized" else 2
        if args.command == "emit-overlay":
            payload = emit_scenario_overlay(
                args.source_scenario,
                load_scenario_overlay(args.overlay),
                overlay_id=args.overlay_id,
                output_path=args.output,
                rationale=args.rationale,
            )
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"status: {payload.get('status', '')}")
                print(f"product_path: {payload.get('product_path', '')}")
                print(f"product_id: {payload.get('product_id', '')}")
            return 0
        if args.command == "materialize-onp":
            payload = materialize_onp(
                args.state_product,
                scenario_name=args.scenario_name,
                scenario_path=args.output,
                output_dir=args.run_output_dir,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                manifest_path=args.manifest,
                trust_plugins=args.trust_plugins,
                overwrite=args.overwrite,
            )
            _print_materialization(payload, json_mode=args.json)
            return 0 if payload.get("status") == "materialized" else 2
        if args.command == "materialize-ogp":
            payload = materialize_ogp(
                args.mean_element_product,
                scenario_name=args.scenario_name,
                scenario_path=args.output,
                output_dir=args.run_output_dir,
                duration_s=args.duration_s,
                dt_s=args.dt_s,
                manifest_path=args.manifest,
                trust_plugins=args.trust_plugins,
                overwrite=args.overwrite,
            )
            _print_materialization(payload, json_mode=args.json)
            return 0 if payload.get("status") == "materialized" else 2
        if args.command == "materialize-scenario-patch":
            if args.patch_index is not None:
                if not str(args.selection_id or "").strip():
                    raise ValueError("--selection-id is required with --patch-index.")
                patch_product = select_patch_product(args.patch_index, args.selection_id)
            else:
                if args.selection_id is not None:
                    raise ValueError("--selection-id is only valid with --patch-index.")
                patch_product = args.patch_product
            payload = materialize_scenario_patch(
                patch_product,
                args.source_scenario,
                scenario_name=args.scenario_name,
                scenario_path=args.output,
                output_dir=args.run_output_dir,
                manifest_path=args.manifest,
                trust_plugins=args.trust_plugins,
                overwrite=args.overwrite,
            )
            _print_materialization(payload, json_mode=args.json)
            return 0 if payload.get("status") == "materialized" else 2
    except (OSError, ValueError) as exc:
        print(f"handoff command failed: {exc}", file=sys.stderr)
        return 2
    return 2


def _print_inspection(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"document_type: {payload.get('document_type', '')}")
    print(f"schema: {payload.get('schema_id', '')} v{payload.get('schema_version', '')}")
    identifier = payload.get("product_id") or payload.get("manifest_id") or ""
    print(f"identifier: {identifier}")
    if payload.get("product_kind"):
        print(f"product_kind: {payload.get('product_kind')}")
        print(f"disposition: {dict(payload.get('quality', {}) or {}).get('disposition', '')}")
        freshness = dict(payload.get("freshness", {}) or {})
        print(f"integrity_status: {freshness.get('integrity_status', '')}")
        print(f"age_status: {freshness.get('age_status', '')}")
    _print_validation(dict(payload.get("validation", {}) or {}), json_mode=False)


def _print_validation(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"valid: {str(bool(payload.get('valid'))).lower()}")
    print(f"promotable: {str(bool(payload.get('promotable'))).lower()}")
    issues = list(payload.get("issues", []) or [])
    if not issues:
        print("issues: none")
        return
    print("issues:")
    for issue in issues:
        row = dict(issue or {})
        print(f"- [{row.get('severity')}] {row.get('code')} {row.get('path')}: {row.get('message')}")


def _print_materialization(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"status: {payload.get('status', '')}")
    print(f"scenario_path: {payload.get('scenario_path', '')}")
    print(f"manifest_path: {payload.get('manifest_path', '')}")
    print(f"manifest_id: {payload.get('manifest_id', '')}")
    print("execution_occurred: false")
    if payload.get("failures"):
        print("failures:")
        for failure in payload["failures"]:
            row = dict(failure or {})
            print(f"- {row.get('code')}: {row.get('message')}")
    print(f"next_action: {payload.get('recommended_next_action', '')}")


def _print_state_export(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"status: {payload.get('status', '')}")
    print(f"product_path: {payload.get('product_path', '')}")
    print(f"product_id: {payload.get('product_id', '')}")
    print(f"object_id: {payload.get('object_id', '')}")
    selection = dict(payload.get("selection", {}) or {})
    print(f"sample_index: {selection.get('sample_index', '')}")
    print(f"time_s: {selection.get('time_s', '')}")
    print(f"epoch_jd_utc: {payload.get('epoch_jd_utc', '')}")
    print(f"covariance_present: {str(bool(payload.get('covariance_present'))).lower()}")
    print("execution_occurred: false")


def _print_handoff_comparison(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"status: {payload.get('status', '')}")
    print(f"comparison_id: {payload.get('comparison_id', '')}")
    summary = dict(payload.get("summary", {}) or {})
    print(f"checks: {summary.get('passed_count', 0)}/{summary.get('check_count', 0)} passed")
    failed = list(summary.get("failed_check_ids", []) or [])
    print(f"failed_check_ids: {', '.join(failed) if failed else 'none'}")


if __name__ == "__main__":
    raise SystemExit(main())
