from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.plotting.style import artifact_metadata, oel_plot_context, save_oel_figure
from sim.review.workspace import ReviewQueryResult, ReviewWorkspace

PLOT_TYPES = ("line", "scatter", "bar")
STYLE_NAMES = ("oel_dark", "oel_light")


@dataclass(frozen=True)
class ReviewPlotSpec:
    sql: str
    x_column: str
    y_columns: list[str]
    plot_type: str = "line"
    group_column: str = ""
    style_name: str = "oel_dark"
    title: str = ""
    subtitle: str = ""
    x_label: str = ""
    y_label: str = ""
    artifact_id: str = ""
    file_format: str = "png"
    dpi: int = 150
    max_rows: int = 5000
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewPlotArtifact:
    artifact_id: str
    path: Path
    relative_path: str
    row_count: int
    truncated: bool
    spec: ReviewPlotSpec


def numeric_columns(result: ReviewQueryResult) -> list[str]:
    numeric: list[str] = []
    for column in result.columns:
        saw_value = False
        for row in result.rows:
            value = row.get(column)
            if value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                break
            saw_value = True
        else:
            if saw_value:
                numeric.append(column)
    return numeric


def categorical_columns(result: ReviewQueryResult) -> list[str]:
    numeric = set(numeric_columns(result))
    return [column for column in result.columns if column not in numeric]


def default_plot_spec(sql: str, result: ReviewQueryResult, *, artifact_id: str = "") -> ReviewPlotSpec:
    numeric = numeric_columns(result)
    if len(numeric) >= 2:
        x_column = "time_s" if "time_s" in numeric else numeric[0]
        y_columns = [column for column in numeric if column != x_column][:1]
    elif len(numeric) == 1:
        x_column = result.columns[0] if result.columns else numeric[0]
        y_columns = numeric
    else:
        x_column = result.columns[0] if result.columns else ""
        y_columns = []
    return ReviewPlotSpec(
        sql=sql,
        x_column=x_column,
        y_columns=y_columns,
        title=_default_title(y_columns, x_column),
        artifact_id=artifact_id,
    )


def save_review_plot(
    workspace: ReviewWorkspace,
    spec: ReviewPlotSpec,
    *,
    path: str | Path | None = None,
    record: bool = True,
) -> ReviewPlotArtifact:
    result = workspace.query(spec.sql, max_rows=max(int(spec.max_rows), 1))
    _validate_plot_spec(spec, result)
    artifact_id = _normalize_artifact_id(spec.artifact_id or _default_artifact_id(spec))
    extension = _normalize_extension(spec.file_format)
    if path is None:
        figures_dir = workspace.output_dir / "review" / "figures"
        out_path = _unique_path(figures_dir / f"{artifact_id}.{extension}")
    else:
        out_path = Path(path)
    scenario_name = _scenario_name(workspace)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with oel_plot_context(
        style_name=_normalize_style(spec.style_name),
        metadata=artifact_metadata(scenario_name=scenario_name, artifact_id=artifact_id),
    ):
        fig, ax = plt.subplots(figsize=(9.5, 5.25))
        _draw_plot(ax, result, spec)
        ax.set_title(spec.title or _default_title(spec.y_columns, spec.x_column))
        if spec.subtitle:
            fig.suptitle(spec.subtitle, y=0.975, fontsize=9)
        ax.set_xlabel(spec.x_label or spec.x_column)
        ax.set_ylabel(spec.y_label or _default_y_label(spec.y_columns))
        ax.grid(True, alpha=0.35)
        if _needs_legend(spec):
            ax.legend(loc="best")
        fig.tight_layout(rect=(0, 0.025, 1, 0.96 if spec.subtitle else 1))
        save_oel_figure(
            fig,
            out_path,
            dpi=int(spec.dpi),
            metadata=artifact_metadata(scenario_name=scenario_name, artifact_id=artifact_id),
            artifact_id=artifact_id,
            style_name=_normalize_style(spec.style_name),
            bbox_inches="tight",
        )
        plt.close(fig)

    artifact = ReviewPlotArtifact(
        artifact_id=artifact_id,
        path=out_path,
        relative_path=_relative_to_output(workspace, out_path),
        row_count=result.row_count,
        truncated=result.truncated,
        spec=spec,
    )
    if record:
        record_generated_artifact(workspace, artifact)
    return artifact


def record_generated_artifact(workspace: ReviewWorkspace, artifact: ReviewPlotArtifact) -> None:
    index_path = workspace.output_dir / "review" / "generated_artifacts.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                existing = [dict(item) for item in list(data.get("artifacts", []) or []) if isinstance(item, dict)]
        except Exception:
            existing = []
    spec = artifact.spec
    existing.append(
        {
            "artifact_id": artifact.artifact_id,
            "artifact_type": "figure",
            "path": artifact.relative_path,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "output_review_workbench",
            "source_query": spec.sql,
            "plot_type": spec.plot_type,
            "style_name": _normalize_style(spec.style_name),
            "x_column": spec.x_column,
            "y_columns": list(spec.y_columns),
            "group_column": spec.group_column,
            "title": spec.title,
            "subtitle": spec.subtitle,
            "x_label": spec.x_label,
            "y_label": spec.y_label,
            "row_count": artifact.row_count,
            "truncated": artifact.truncated,
        }
    )
    index_path.write_text(json.dumps({"artifacts": existing}, indent=2) + "\n", encoding="utf-8")


def _draw_plot(ax: Any, result: ReviewQueryResult, spec: ReviewPlotSpec) -> None:
    plot_type = _normalize_plot_type(spec.plot_type)
    rows = result.rows
    if spec.group_column:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(spec.group_column) or "n/a"), []).append(row)
        for group, group_rows in groups.items():
            _draw_series(ax, group_rows, spec.x_column, spec.y_columns[0], plot_type, label=group)
        return
    if plot_type == "bar":
        _draw_bar(ax, rows, spec)
        return
    for y_column in spec.y_columns:
        _draw_series(ax, rows, spec.x_column, y_column, plot_type, label=y_column if len(spec.y_columns) > 1 else "")


def _draw_series(
    ax: Any,
    rows: list[dict[str, Any]],
    x_column: str,
    y_column: str,
    plot_type: str,
    *,
    label: str = "",
) -> None:
    pairs = [
        (row.get(x_column), row.get(y_column))
        for row in rows
        if row.get(x_column) is not None and row.get(y_column) is not None
    ]
    x_values = [item[0] for item in pairs]
    y_values = [float(item[1]) for item in pairs]
    if plot_type == "scatter":
        ax.scatter(x_values, y_values, s=20, label=label or None)
    else:
        ax.plot(x_values, y_values, marker="o", markersize=3, label=label or None)


def _draw_bar(ax: Any, rows: list[dict[str, Any]], spec: ReviewPlotSpec) -> None:
    y_column = spec.y_columns[0]
    labels = [str(row.get(spec.x_column)) for row in rows if row.get(y_column) is not None]
    values = [float(row.get(y_column)) for row in rows if row.get(y_column) is not None]
    ax.bar(labels, values, label=y_column if len(spec.y_columns) == 1 else None)
    if len(labels) > 8:
        ax.tick_params(axis="x", labelrotation=35)


def _validate_plot_spec(spec: ReviewPlotSpec, result: ReviewQueryResult) -> None:
    if result.row_count <= 0:
        raise ValueError("The review query returned no rows to plot.")
    columns = set(result.columns)
    if spec.x_column not in columns:
        raise ValueError(f"x_column '{spec.x_column}' is not in the query result.")
    if not spec.y_columns:
        raise ValueError("At least one y_column is required.")
    for column in spec.y_columns:
        if column not in columns:
            raise ValueError(f"y_column '{column}' is not in the query result.")
    if spec.group_column and spec.group_column not in columns:
        raise ValueError(f"group_column '{spec.group_column}' is not in the query result.")
    numeric = set(numeric_columns(result))
    for column in spec.y_columns:
        if column not in numeric:
            raise ValueError(f"y_column '{column}' must contain numeric values.")
    if spec.group_column and len(spec.y_columns) != 1:
        raise ValueError("Grouped plots support exactly one y_column.")
    _normalize_plot_type(spec.plot_type)
    _normalize_style(spec.style_name)
    _normalize_extension(spec.file_format)


def _scenario_name(workspace: ReviewWorkspace) -> str:
    try:
        result = workspace.query("SELECT scenario_name FROM run_metadata", max_rows=1)
    except Exception:
        return ""
    if not result.rows:
        return ""
    return str(result.rows[0].get("scenario_name") or "")


def _default_title(y_columns: list[str], x_column: str) -> str:
    if not y_columns:
        return "Review Plot"
    if len(y_columns) == 1:
        return f"{y_columns[0]} vs {x_column}"
    return f"{', '.join(y_columns[:3])} vs {x_column}"


def _default_y_label(y_columns: list[str]) -> str:
    if len(y_columns) == 1:
        return y_columns[0]
    return "value"


def _default_artifact_id(spec: ReviewPlotSpec) -> str:
    return f"orw_{spec.plot_type}_{spec.y_columns[0] if spec.y_columns else 'plot'}"


def _normalize_artifact_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return cleaned[:80] or f"orw_plot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _normalize_plot_type(value: str) -> str:
    plot_type = str(value or "line").strip().lower()
    if plot_type not in PLOT_TYPES:
        raise ValueError(f"Unsupported plot type '{value}'. Valid plot types: {', '.join(PLOT_TYPES)}.")
    return plot_type


def _normalize_style(value: str) -> str:
    style = str(value or "oel_dark").strip().lower()
    if style not in STYLE_NAMES:
        raise ValueError(f"Unsupported style '{value}'. Valid styles: {', '.join(STYLE_NAMES)}.")
    return style


def _normalize_extension(value: str) -> str:
    extension = str(value or "png").strip().lower().lstrip(".")
    if extension not in {"png", "svg", "pdf"}:
        raise ValueError("Review plot exports support png, svg, or pdf.")
    return extension


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not find an available artifact filename for {path.name}.")


def _relative_to_output(workspace: ReviewWorkspace, path: Path) -> str:
    try:
        return path.relative_to(workspace.output_dir).as_posix()
    except ValueError:
        return str(path)


def _needs_legend(spec: ReviewPlotSpec) -> bool:
    return bool(spec.group_column) or len(spec.y_columns) > 1
