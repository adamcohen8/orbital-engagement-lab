from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sim.review import (
    ReviewQueryError,
    ReviewStoreNotFoundError,
    ReviewWorkspace,
    get_saved_review_query,
    list_saved_review_queries,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query an OEL output review store.")
    parser.add_argument("output_dir", nargs="?", help="Output directory or review/run.sqlite path.")
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--query", "-q", help="Read-only SELECT query to run.")
    query_group.add_argument("--saved-query", help="Named built-in review query to run.")
    parser.add_argument("--list-saved-queries", action="store_true", help="List built-in saved review queries and exit.")
    parser.add_argument("--max-rows", type=int, default=50, help="Maximum rows to print.")
    parser.add_argument("--max-vm-steps", type=int, default=250_000, help="SQLite virtual-machine step budget.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.list_saved_queries:
        for item in list_saved_review_queries():
            print(f"{item.name}: {item.description}")
        return 0

    sql = args.query
    if args.saved_query:
        saved_query = get_saved_review_query(args.saved_query)
        if saved_query is None:
            print(f"unknown saved review query: {args.saved_query}", file=sys.stderr)
            return 2
        sql = saved_query.sql
    if not sql:
        parser.error("one of --query, --saved-query, or --list-saved-queries is required")
    if not args.output_dir:
        parser.error("output_dir is required when running a query")

    try:
        workspace = ReviewWorkspace.open(Path(args.output_dir))
        result = workspace.query(
            sql,
            max_rows=max(int(args.max_rows), 1),
            max_vm_steps=max(int(args.max_vm_steps), 1),
        )
    except (ReviewStoreNotFoundError, ReviewQueryError) as exc:
        print(f"review query failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "columns": result.columns,
                    "rows": result.rows,
                    "row_count": result.row_count,
                    "truncated": result.truncated,
                },
                indent=2,
            )
        )
    else:
        _print_table(result.columns, result.rows, truncated=result.truncated)
    return 0


def _print_table(columns: list[str], rows: list[dict[str, Any]], *, truncated: bool) -> None:
    if not columns:
        print("(no columns)")
        return
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = min(max(widths[col], len(_fmt(row.get(col)))), 80)
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(_fmt(row.get(col))[: widths[col]].ljust(widths[col]) for col in columns))
    if truncated:
        print("(truncated)")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
