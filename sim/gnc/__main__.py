"""Command-line discovery for OEL's built-in GNC catalog."""

from __future__ import annotations

import argparse
import json
from typing import Any

from sim.gnc.catalog import catalog_entries, catalog_entry, validate_catalog


def _print(payload: Any, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if isinstance(payload, list):
        for row in payload:
            print(f"{row['builtin_id']:<38} {row['maturity']:<14} {row['display_name']}")
        return
    for key, value in payload.items():
        if key == "parameters":
            print("parameters:")
            for name, spec in value.items():
                required = "required" if spec.get("required") else f"default={spec.get('default')!r}"
                print(f"  {name}: {spec.get('annotation')} ({required}, units={spec.get('units')})")
        else:
            print(f"{key}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect OEL's built-in GNC library.")
    sub = parser.add_subparsers(dest="command", required=True)
    list_parser = sub.add_parser("list", help="List built-in GNC surfaces.")
    list_parser.add_argument("--category")
    list_parser.add_argument("--maturity")
    list_parser.add_argument("--include-internal", action="store_true")
    list_parser.add_argument("--json", action="store_true")
    show_parser = sub.add_parser("show", help="Show one built-in and its parameters.")
    show_parser.add_argument("builtin_id")
    show_parser.add_argument("--json", action="store_true")
    validate_parser = sub.add_parser("validate", help="Validate catalog imports and identities.")
    validate_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        rows = [entry.to_dict() for entry in catalog_entries(include_internal=args.include_internal)]
        if args.category:
            rows = [row for row in rows if row["category"] == args.category]
        if args.maturity:
            rows = [row for row in rows if row["maturity"] == args.maturity]
        _print(rows, json_mode=args.json)
        return 0
    if args.command == "show":
        entry = catalog_entry(args.builtin_id)
        if entry is None:
            parser.error(f"unknown built-in GNC id: {args.builtin_id}")
        _print(entry.to_dict(include_parameters=True), json_mode=args.json)
        return 0
    errors = validate_catalog()
    payload = {"valid": not errors, "entries": len(catalog_entries()), "errors": errors}
    _print(payload, json_mode=args.json)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
