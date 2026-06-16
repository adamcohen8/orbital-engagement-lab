from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sim.review.plotting import EVIDENCE_PLOT_RECIPES, EvidencePlotter, ReviewPlotSpec, _normalize_style_alias
from sim.review.workspace import ReviewQueryError, ReviewStoreNotFoundError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create OEL-styled custom plots from a completed review store.")
    parser.add_argument("output_dir", nargs="?", help="Output directory or review/run.sqlite path.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--sql", help="Read-only SELECT/WITH query to plot.")
    source_group.add_argument("--table", help="Plot from a review table.")
    source_group.add_argument("--saved-query", help="Plot from a built-in saved review query.")
    source_group.add_argument("--recipe", help="Run a built-in evidence plot recipe.")
    parser.add_argument("--list-recipes", action="store_true", help="List built-in plot recipes and exit.")
    parser.add_argument("--x", help="X-axis column.")
    parser.add_argument("--y", action="append", help="Y/value column. Repeat for multi-series plots.")
    parser.add_argument("--group", default="", help="Grouping column, or heatmap y-axis category.")
    parser.add_argument("--type", default="line", choices=("line", "scatter", "bar", "histogram", "heatmap"))
    parser.add_argument("--style", default="dark", choices=("dark", "light", "oel_dark", "oel_light"))
    parser.add_argument("--title", default="")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--x-label", default="")
    parser.add_argument("--y-label", default="")
    parser.add_argument("--format", default="png", choices=("png", "svg", "pdf"))
    parser.add_argument("--artifact-id", default="")
    parser.add_argument("--output", help="Output path, relative to the run output folder unless absolute.")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plot spec without writing.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.list_recipes:
        payload = {"recipes": [_recipe_payload(item) for item in EVIDENCE_PLOT_RECIPES.values()]}
        _print_payload(payload, json_mode=args.json)
        return 0
    if not args.output_dir:
        parser.error("output_dir is required unless --list-recipes is used")

    try:
        plotter = EvidencePlotter(Path(args.output_dir))
        spec = _spec_from_args(plotter, args)
        if args.dry_run:
            payload = plotter.dry_run(spec)
            _print_payload(payload, json_mode=True if args.json else False)
            return 0
        artifact = plotter.save(spec, output=args.output)
    except (ReviewStoreNotFoundError, ReviewQueryError, ValueError) as exc:
        print(f"review plot failed: {exc}", file=sys.stderr)
        return 2

    payload = {
        "artifact_id": artifact.artifact_id,
        "path": str(artifact.path),
        "relative_path": artifact.relative_path,
        "row_count": artifact.row_count,
        "truncated": artifact.truncated,
        "spec": _spec_payload(artifact.spec),
    }
    _print_payload(payload, json_mode=args.json)
    return 0


def _spec_from_args(plotter: EvidencePlotter, args: argparse.Namespace) -> ReviewPlotSpec:
    style = _normalize_style_alias(args.style)
    common = {
        "style": style,
        "title": args.title,
        "subtitle": args.subtitle,
        "x_label": args.x_label,
        "y_label": args.y_label,
        "artifact_id": args.artifact_id,
        "file_format": args.format,
        "dpi": args.dpi,
        "max_rows": args.max_rows,
        "extra": {"source": "oel_review_plot_cli"},
    }
    if args.recipe:
        recipe = EVIDENCE_PLOT_RECIPES.get(args.recipe)
        if recipe is None:
            raise ValueError(
                f"Unknown evidence plot recipe '{args.recipe}'. "
                f"Available recipes: {', '.join(sorted(EVIDENCE_PLOT_RECIPES))}."
            )
        plotter._require_tables(recipe.required_tables, recipe_id=recipe.recipe_id)
        return ReviewPlotSpec(
            sql=recipe.sql,
            x_column=args.x or recipe.x_column,
            y_columns=args.y or list(recipe.y_columns),
            plot_type=args.type if args.type != "line" else recipe.plot_type,
            group_column=args.group or recipe.group_column,
            style_name=style,
            title=args.title or recipe.title,
            subtitle=args.subtitle,
            x_label=args.x_label or recipe.x_label,
            y_label=args.y_label or recipe.y_label,
            artifact_id=args.artifact_id or recipe.artifact_id,
            file_format=args.format,
            dpi=args.dpi,
            max_rows=args.max_rows,
            extra={"source": "oel_review_plot_cli", "recipe_id": recipe.recipe_id},
        )
    if args.sql:
        sql = args.sql
    elif args.table:
        if not args.x and not args.y:
            return _auto_spec(plotter, table=args.table, args=args)
        sql = f'SELECT * FROM "{args.table.replace(chr(34), chr(34) + chr(34))}" LIMIT {max(int(args.max_rows), 1)}'
    elif args.saved_query:
        return _auto_spec(plotter, saved_query=args.saved_query, args=args)
    else:
        raise ValueError("Choose --sql, --table, --saved-query, or --recipe.")
    if not args.y:
        return _auto_spec(plotter, sql=sql, args=args)
    if args.type != "histogram" and not args.x:
        raise ValueError("--x is required unless using --type histogram or a source that can infer columns.")
    return ReviewPlotSpec(
        sql=sql,
        x_column=args.x or "",
        y_columns=list(args.y or []),
        plot_type=args.type,
        group_column=args.group,
        style_name=style,
        title=args.title,
        subtitle=args.subtitle,
        x_label=args.x_label,
        y_label=args.y_label,
        artifact_id=args.artifact_id,
        file_format=args.format,
        dpi=args.dpi,
        max_rows=args.max_rows,
        extra=common["extra"],
    )


def _auto_spec(
    plotter: EvidencePlotter,
    *,
    args: argparse.Namespace,
    sql: str = "",
    table: str = "",
    saved_query: str = "",
) -> ReviewPlotSpec:
    kwargs = {
        "x": args.x or "",
        "y": args.y,
        "plot_type": args.type,
        "style": args.style,
        "title": args.title,
        "subtitle": args.subtitle,
        "x_label": args.x_label,
        "y_label": args.y_label,
        "artifact_id": args.artifact_id,
        "file_format": args.format,
        "dpi": args.dpi,
        "max_rows": args.max_rows,
        "extra": {"source": "oel_review_plot_cli"},
    }
    if table:
        source_sql = f'SELECT * FROM "{table.replace(chr(34), chr(34) + chr(34))}" LIMIT {max(int(args.max_rows), 1)}'
    elif saved_query:
        from sim.review.queries import get_saved_review_query

        saved = get_saved_review_query(saved_query)
        if saved is None:
            raise ValueError(f"Unknown saved review query '{saved_query}'.")
        source_sql = saved.sql
        if not kwargs["title"]:
            kwargs["title"] = saved.description
    else:
        source_sql = sql
    result = plotter.workspace.query(source_sql, max_rows=max(int(args.max_rows), 1))
    from sim.review.plotting import default_plot_spec

    spec = default_plot_spec(source_sql, result, artifact_id=args.artifact_id)
    updates = {
        "plot_type": args.type,
        "style_name": _normalize_style_alias(args.style),
        "title": kwargs["title"] or spec.title,
        "subtitle": args.subtitle,
        "x_label": args.x_label or spec.x_label,
        "y_label": args.y_label or spec.y_label,
        "file_format": args.format,
        "dpi": args.dpi,
        "max_rows": args.max_rows,
        "extra": {"source": "oel_review_plot_cli"},
    }
    if args.x:
        updates["x_column"] = args.x
    if args.y:
        updates["y_columns"] = list(args.y)
    if args.group:
        updates["group_column"] = args.group
    from dataclasses import replace

    return replace(spec, **updates)


def _recipe_payload(recipe: Any) -> dict[str, Any]:
    return {
        "recipe_id": recipe.recipe_id,
        "title": recipe.title,
        "description": recipe.description,
        "required_tables": list(recipe.required_tables),
    }


def _spec_payload(spec: ReviewPlotSpec) -> dict[str, Any]:
    return {
        "sql": spec.sql,
        "x_column": spec.x_column,
        "y_columns": list(spec.y_columns),
        "plot_type": spec.plot_type,
        "group_column": spec.group_column,
        "style_name": spec.style_name,
        "title": spec.title,
        "x_label": spec.x_label,
        "y_label": spec.y_label,
        "artifact_id": spec.artifact_id,
        "file_format": spec.file_format,
        "extra": dict(spec.extra or {}),
    }


def _print_payload(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2))
        return
    if "recipes" in payload:
        for item in payload["recipes"]:
            print(f"{item['recipe_id']}: {item['title']} - {item['description']}")
        return
    if "spec" in payload and "path" not in payload:
        print(json.dumps(payload, indent=2))
        return
    print(f"{payload['relative_path']} ({payload['row_count']} row(s))")


if __name__ == "__main__":
    raise SystemExit(main())
