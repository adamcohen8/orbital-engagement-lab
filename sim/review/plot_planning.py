from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from sim.review.plotting import ReviewPlotArtifact, ReviewPlotSpec, save_review_plot
from sim.review.workspace import ReviewWorkspace

REVIEW_PLOT_PLAN_SCHEMA_VERSION = 1


def plot_spec_from_mapping(arguments: Mapping[str, Any], *, source: str) -> ReviewPlotSpec:
    return ReviewPlotSpec(
        sql=str(arguments["sql"]),
        x_column=str(arguments.get("x_column", "") or ""),
        y_columns=[str(item) for item in list(arguments.get("y_columns", []) or [])],
        plot_type=str(arguments.get("plot_type", "line") or "line"),
        group_column=str(arguments.get("group_column", "") or ""),
        style_name=str(arguments.get("style", "oel_dark") or "oel_dark"),
        title=str(arguments.get("title", "") or ""),
        subtitle=str(arguments.get("subtitle", "") or ""),
        x_label=str(arguments.get("x_label", "") or ""),
        y_label=str(arguments.get("y_label", "") or ""),
        artifact_id=str(arguments.get("artifact_id", "") or ""),
        file_format=str(arguments.get("format", "png") or "png"),
        dpi=int(arguments.get("dpi", 150)),
        max_rows=int(arguments.get("max_rows", 5000)),
        renderer_id="generic",
        extra={"source": source, "plot_contract": "typed_review_plot_v2"},
    )


def plan_review_plot(output_dir: str | Path, spec: ReviewPlotSpec) -> dict[str, Any]:
    workspace = ReviewWorkspace.open(output_dir)
    dry_run = _dry_run(workspace, spec)
    plan_id = review_plot_plan_id(workspace, spec)
    warnings: list[str] = []
    if dry_run["truncated"]:
        warnings.append("The query reached max_rows; render evidence would be truncated.")
    return {
        "status": "planned",
        "output_dir": str(workspace.output_dir),
        "review_store": "review/run.sqlite",
        "plot_plan_id": plan_id,
        "spec": asdict(spec),
        "columns": dry_run["columns"],
        "numeric_columns": dry_run["numeric_columns"],
        "row_count": dry_run["row_count"],
        "truncated": dry_run["truncated"],
        "warnings": warnings,
        "render_authorized": False,
        "visual_review_required": True,
    }


def render_review_plot(
    output_dir: str | Path,
    spec: ReviewPlotSpec,
    *,
    plot_plan_id: str,
    path: str | Path,
) -> ReviewPlotArtifact:
    workspace = ReviewWorkspace.open(output_dir)
    current_id = review_plot_plan_id(workspace, spec)
    if str(plot_plan_id) != current_id:
        raise ValueError("The plot_plan_id is stale or does not match the review store and plot specification.")
    return save_review_plot(workspace, spec, path=path)


def review_plot_plan_id(workspace: ReviewWorkspace, spec: ReviewPlotSpec) -> str:
    stat = workspace.db_path.stat()
    payload = {
        "schema_version": REVIEW_PLOT_PLAN_SCHEMA_VERSION,
        "review_store": {
            "path": str(workspace.db_path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "spec": asdict(spec),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "oel-review-plot-plan-v1:" + hashlib.sha256(encoded).hexdigest()


def _dry_run(workspace: ReviewWorkspace, spec: ReviewPlotSpec) -> dict[str, Any]:
    from sim.review.plotting import EvidencePlotter

    return EvidencePlotter(workspace).dry_run(spec)


__all__ = [
    "REVIEW_PLOT_PLAN_SCHEMA_VERSION",
    "plan_review_plot",
    "plot_spec_from_mapping",
    "render_review_plot",
    "review_plot_plan_id",
]
