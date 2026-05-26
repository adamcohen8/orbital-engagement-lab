from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sim.review import ReviewQueryError, ReviewStoreNotFoundError, ReviewWorkspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query an OEL output review store.")
    parser.add_argument("output_dir", help="Output directory or review/run.sqlite path.")
    parser.add_argument("--query", "-q", required=True, help="Read-only SELECT query to run.")
    parser.add_argument("--max-rows", type=int, default=50, help="Maximum rows to print.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    try:
        workspace = ReviewWorkspace.open(Path(args.output_dir))
        result = workspace.query(args.query, max_rows=max(int(args.max_rows), 1))
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
