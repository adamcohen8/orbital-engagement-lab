from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sim.plotting.style import artifact_metadata, oel_plot_context, save_oel_figure
from sim.review.queries import get_saved_review_query
from sim.review.workspace import ReviewQueryResult, ReviewWorkspace
from sim.runtime_environment import configure_headless_runtime

PLOT_TYPES = ("line", "scatter", "bar", "histogram", "heatmap")
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


@dataclass(frozen=True)
class EvidencePlotRecipe:
    recipe_id: str
    title: str
    description: str
    sql: str
    x_column: str
    y_columns: tuple[str, ...]
    plot_type: str = "line"
    group_column: str = ""
    x_label: str = ""
    y_label: str = ""
    artifact_id: str = ""
    required_tables: tuple[str, ...] = ()


EVIDENCE_PLOT_RECIPES: dict[str, EvidencePlotRecipe] = {
    "relative_range": EvidencePlotRecipe(
        recipe_id="relative_range",
        title="Relative range over time",
        description="Deputy-chief range from the relative_state review table.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, range_km "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        x_column="time_s",
        y_columns=("range_km",),
        group_column="pair_id",
        x_label="Time (s)",
        y_label="Range (km)",
        artifact_id="evidence_relative_range",
        required_tables=("relative_state",),
    ),
    "relative_range_rate": EvidencePlotRecipe(
        recipe_id="relative_range_rate",
        title="Relative range rate over time",
        description="Relative range rate from the relative_state review table.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, range_rate_km_s "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        x_column="time_s",
        y_columns=("range_rate_km_s",),
        group_column="pair_id",
        x_label="Time (s)",
        y_label="Range rate (km/s)",
        artifact_id="evidence_relative_range_rate",
        required_tables=("relative_state",),
    ),
    "relative_velocity_components": EvidencePlotRecipe(
        recipe_id="relative_velocity_components",
        title="Relative velocity components over time",
        description="RIC-frame relative velocity components from the relative_state review table.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, "
            "v_radial_km_s, v_intrack_km_s, v_crosstrack_km_s "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        x_column="time_s",
        y_columns=("v_radial_km_s", "v_intrack_km_s", "v_crosstrack_km_s"),
        group_column="pair_id",
        x_label="Time (s)",
        y_label="Relative velocity (km/s)",
        artifact_id="evidence_relative_velocity",
        required_tables=("relative_state",),
    ),
    "burn_activity": EvidencePlotRecipe(
        recipe_id="burn_activity",
        title="Burn activity by object",
        description="Active thrust samples by object.",
        sql="SELECT object_id, SUM(burn_active) AS active_samples FROM thrust GROUP BY object_id ORDER BY object_id",
        x_column="object_id",
        y_columns=("active_samples",),
        plot_type="bar",
        x_label="Object",
        y_label="Active thrust samples",
        artifact_id="evidence_burn_activity",
        required_tables=("thrust",),
    ),
    "ground_access": EvidencePlotRecipe(
        recipe_id="ground_access",
        title="Ground access samples",
        description="Access sample counts by station/object from the ground_access review table.",
        sql=(
            "SELECT station_id || ':' || object_id AS station_object, SUM(access) AS access_samples "
            "FROM ground_access GROUP BY station_id, object_id ORDER BY station_id, object_id"
        ),
        x_column="station_object",
        y_columns=("access_samples",),
        plot_type="bar",
        x_label="Station:Object",
        y_label="Access samples",
        artifact_id="evidence_ground_access",
        required_tables=("ground_access",),
    ),
    "campaign_closest_approach": EvidencePlotRecipe(
        recipe_id="campaign_closest_approach",
        title="Campaign closest approach by iteration",
        description="Monte Carlo closest-approach results by iteration.",
        sql="SELECT iteration, closest_approach_km FROM campaign_runs ORDER BY iteration",
        x_column="iteration",
        y_columns=("closest_approach_km",),
        plot_type="scatter",
        x_label="Iteration",
        y_label="Closest approach (km)",
        artifact_id="evidence_campaign_closest_approach",
        required_tables=("campaign_runs",),
    ),
    "sensitivity_effects": EvidencePlotRecipe(
        recipe_id="sensitivity_effects",
        title="Sensitivity effect sizes",
        description="Ranked sensitivity effect sizes by parameter.",
        sql="SELECT parameter_path, effect_size FROM sensitivity_rankings ORDER BY rank, parameter_path, metric_path",
        x_column="parameter_path",
        y_columns=("effect_size",),
        plot_type="bar",
        x_label="Parameter",
        y_label="Effect size",
        artifact_id="evidence_sensitivity_effects",
        required_tables=("sensitivity_rankings",),
    ),
}


class EvidencePlotter:
    """Agent-friendly OEL-styled plotting API for completed review stores."""

    def __init__(self, workspace: ReviewWorkspace | str | Path) -> None:
        self.workspace = workspace if isinstance(workspace, ReviewWorkspace) else ReviewWorkspace.open(workspace)

    def line(self, *, sql: str, x: str, y: str | Sequence[str], **kwargs: Any) -> ReviewPlotArtifact:
        return self.plot(sql=sql, x=x, y=y, plot_type="line", **kwargs)

    def scatter(self, *, sql: str, x: str, y: str | Sequence[str], **kwargs: Any) -> ReviewPlotArtifact:
        return self.plot(sql=sql, x=x, y=y, plot_type="scatter", **kwargs)

    def bar(self, *, sql: str, x: str, y: str | Sequence[str], **kwargs: Any) -> ReviewPlotArtifact:
        return self.plot(sql=sql, x=x, y=y, plot_type="bar", **kwargs)

    def histogram(self, *, sql: str, y: str | Sequence[str], x: str = "", **kwargs: Any) -> ReviewPlotArtifact:
        return self.plot(sql=sql, x=x, y=y, plot_type="histogram", **kwargs)

    def heatmap(self, *, sql: str, x: str, y: str, value: str, **kwargs: Any) -> ReviewPlotArtifact:
        y_columns = [value]
        return self.plot(sql=sql, x=x, y=y_columns, group=y, plot_type="heatmap", **kwargs)

    def table(
        self,
        table: str,
        *,
        x: str = "",
        y: str | Sequence[str] | None = None,
        limit: int = 1000,
        **kwargs: Any,
    ) -> ReviewPlotArtifact:
        sql = f"SELECT * FROM {_quote_identifier(table)} LIMIT {max(int(limit), 1)}"
        return self.auto(sql=sql, x=x, y=y, **kwargs)

    def saved_query(
        self,
        name: str,
        *,
        x: str = "",
        y: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ReviewPlotArtifact:
        saved = get_saved_review_query(name)
        if saved is None:
            raise ValueError(f"Unknown saved review query '{name}'.")
        return self.auto(sql=saved.sql, x=x, y=y, title=kwargs.pop("title", saved.description), **kwargs)

    def recipe(self, recipe_id: str, **kwargs: Any) -> ReviewPlotArtifact:
        recipe = EVIDENCE_PLOT_RECIPES.get(recipe_id)
        if recipe is None:
            raise ValueError(
                f"Unknown evidence plot recipe '{recipe_id}'. "
                f"Available recipes: {', '.join(sorted(EVIDENCE_PLOT_RECIPES))}."
            )
        self._require_tables(recipe.required_tables, recipe_id=recipe.recipe_id)
        return self.plot(
            sql=recipe.sql,
            x=recipe.x_column,
            y=list(recipe.y_columns),
            group=kwargs.pop("group", recipe.group_column),
            plot_type=kwargs.pop("plot_type", recipe.plot_type),
            title=kwargs.pop("title", recipe.title),
            x_label=kwargs.pop("x_label", recipe.x_label),
            y_label=kwargs.pop("y_label", recipe.y_label),
            artifact_id=kwargs.pop("artifact_id", recipe.artifact_id),
            extra={**{"recipe_id": recipe.recipe_id}, **dict(kwargs.pop("extra", {}) or {})},
            **kwargs,
        )

    def relative_range(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("relative_range", **kwargs)

    def relative_range_rate(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("relative_range_rate", **kwargs)

    def relative_velocity_components(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("relative_velocity_components", **kwargs)

    def burn_activity(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("burn_activity", **kwargs)

    def ground_access(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("ground_access", **kwargs)

    def auto(
        self,
        *,
        sql: str,
        x: str = "",
        y: str | Sequence[str] | None = None,
        plot_type: str = "line",
        **kwargs: Any,
    ) -> ReviewPlotArtifact:
        result = self.workspace.query(sql, max_rows=max(int(kwargs.get("max_rows", 5000)), 1))
        spec = default_plot_spec(sql, result, artifact_id=str(kwargs.pop("artifact_id", "") or ""))
        if x:
            spec = _replace_spec(spec, x_column=x)
        if y is not None:
            spec = _replace_spec(spec, y_columns=_coerce_y_columns(y))
        output = kwargs.pop("output", None)
        return self.save(_replace_spec(spec, plot_type=plot_type, **_spec_kwargs(kwargs)), output=output)

    def plot(
        self,
        *,
        sql: str,
        x: str,
        y: str | Sequence[str],
        plot_type: str = "line",
        group: str = "",
        style: str = "oel_dark",
        title: str = "",
        subtitle: str = "",
        x_label: str = "",
        y_label: str = "",
        artifact_id: str = "",
        file_format: str = "png",
        dpi: int = 150,
        max_rows: int = 5000,
        output: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ReviewPlotArtifact:
        spec = ReviewPlotSpec(
            sql=sql,
            x_column=x,
            y_columns=_coerce_y_columns(y),
            plot_type=plot_type,
            group_column=group,
            style_name=_normalize_style_alias(style),
            title=title,
            subtitle=subtitle,
            x_label=x_label,
            y_label=y_label,
            artifact_id=artifact_id,
            file_format=file_format,
            dpi=dpi,
            max_rows=max_rows,
            extra={"source": "oel_review_plot_api", **dict(extra or {})},
        )
        return self.save(spec, output=output)

    def save(self, spec: ReviewPlotSpec, *, output: str | Path | None = None) -> ReviewPlotArtifact:
        return save_review_plot(self.workspace, spec, path=_resolve_output_path(self.workspace, output))

    def preview(self, spec: ReviewPlotSpec, *, path: str | Path) -> ReviewPlotArtifact:
        return save_review_plot(self.workspace, spec, path=path, record=False)

    def dry_run(self, spec: ReviewPlotSpec) -> dict[str, Any]:
        result = self.workspace.query(spec.sql, max_rows=max(int(spec.max_rows), 1))
        _validate_plot_spec(spec, result)
        return {
            "spec": _spec_to_dict(spec),
            "columns": result.columns,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "numeric_columns": numeric_columns(result),
        }

    def _require_tables(self, tables: Sequence[str], *, recipe_id: str) -> None:
        available = set(self.workspace.tables())
        missing = [table for table in tables if table not in available]
        if missing:
            raise ValueError(
                f"Recipe '{recipe_id}' requires missing review table(s): {', '.join(missing)}. "
                f"Available tables: {', '.join(sorted(available)) or '(none)'}."
            )


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
    spec = _replace_spec(spec, style_name=_normalize_style_alias(spec.style_name))
    artifact_id = _normalize_artifact_id(spec.artifact_id or _default_artifact_id(spec))
    extension = _normalize_extension(spec.file_format)
    if path is None:
        figures_dir = workspace.output_dir / "review" / "figures"
        out_path = _unique_path(figures_dir / f"{artifact_id}.{extension}")
        artifact_id = _normalize_artifact_id(out_path.stem)
        spec = _replace_spec(spec, artifact_id=artifact_id)
    else:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_name = _scenario_name(workspace)

    _ensure_matplotlib_cache_env()
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
            "created_utc": os.environ.get("OEL_GENERATED_UTC", "").strip(),
            "source": str(dict(spec.extra or {}).get("source", "oel_review_plot_api") or "oel_review_plot_api"),
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
            "extra": dict(spec.extra or {}),
        }
    )
    index_path.write_text(json.dumps({"artifacts": existing}, indent=2) + "\n", encoding="utf-8")


def _draw_plot(ax: Any, result: ReviewQueryResult, spec: ReviewPlotSpec) -> None:
    plot_type = _normalize_plot_type(spec.plot_type)
    rows = result.rows
    if plot_type == "heatmap":
        _draw_heatmap(ax, rows, spec)
        return
    if spec.group_column:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(spec.group_column) or "n/a"), []).append(row)
        for group, group_rows in groups.items():
            for y_column in spec.y_columns:
                label = group if len(spec.y_columns) == 1 else f"{group}:{y_column}"
                _draw_series(ax, group_rows, spec.x_column, y_column, plot_type, label=label)
        return
    if plot_type == "bar":
        _draw_bar(ax, rows, spec)
        return
    if plot_type == "histogram":
        _draw_histogram(ax, rows, spec)
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


def _draw_histogram(ax: Any, rows: list[dict[str, Any]], spec: ReviewPlotSpec) -> None:
    for y_column in spec.y_columns:
        values = [float(row.get(y_column)) for row in rows if row.get(y_column) is not None]
        ax.hist(values, bins="auto", alpha=0.72, label=y_column if len(spec.y_columns) > 1 else None)


def _draw_heatmap(ax: Any, rows: list[dict[str, Any]], spec: ReviewPlotSpec) -> None:
    x_column = spec.x_column
    y_column = spec.group_column
    value_column = spec.y_columns[0]
    x_labels = _ordered_unique(str(row.get(x_column)) for row in rows if row.get(x_column) is not None)
    y_labels = _ordered_unique(str(row.get(y_column)) for row in rows if row.get(y_column) is not None)
    x_index = {value: idx for idx, value in enumerate(x_labels)}
    y_index = {value: idx for idx, value in enumerate(y_labels)}
    matrix = [[float("nan") for _ in x_labels] for _ in y_labels]
    for row in rows:
        x_value = row.get(x_column)
        y_value = row.get(y_column)
        z_value = row.get(value_column)
        if x_value is None or y_value is None or z_value is None:
            continue
        matrix[y_index[str(y_value)]][x_index[str(x_value)]] = float(z_value)
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(x_labels)), x_labels)
    ax.set_yticks(range(len(y_labels)), y_labels)
    if len(x_labels) > 6:
        ax.tick_params(axis="x", labelrotation=35)
    ax.figure.colorbar(image, ax=ax, label=spec.y_label or value_column)


def _validate_plot_spec(spec: ReviewPlotSpec, result: ReviewQueryResult) -> None:
    if result.row_count <= 0:
        raise ValueError("The review query returned no rows to plot.")
    columns = set(result.columns)
    if spec.x_column not in columns:
        if _normalize_plot_type(spec.plot_type) != "histogram" or spec.x_column:
            raise ValueError(
                f"x_column '{spec.x_column}' is not in the query result. "
                f"Available columns: {', '.join(result.columns) or '(none)'}."
            )
    if not spec.y_columns:
        raise ValueError("At least one y_column is required.")
    for column in spec.y_columns:
        if column not in columns:
            raise ValueError(
                f"y_column '{column}' is not in the query result. "
                f"Available columns: {', '.join(result.columns) or '(none)'}."
            )
    if spec.group_column and spec.group_column not in columns:
        raise ValueError(
            f"group_column '{spec.group_column}' is not in the query result. "
            f"Available columns: {', '.join(result.columns) or '(none)'}."
        )
    numeric = set(numeric_columns(result))
    for column in spec.y_columns:
        if column not in numeric:
            raise ValueError(
                f"y_column '{column}' must contain numeric values. "
                f"Numeric columns: {', '.join(numeric_columns(result)) or '(none)'}."
            )
    if spec.group_column and len(spec.y_columns) != 1:
        raise ValueError("Grouped plots support exactly one y_column.")
    if _normalize_plot_type(spec.plot_type) == "heatmap" and not spec.group_column:
        raise ValueError("Heatmap plots require a group_column for the y-axis category.")
    _normalize_plot_type(spec.plot_type)
    _normalize_style(spec.style_name)
    _normalize_extension(spec.file_format)


def _scenario_name(workspace: ReviewWorkspace) -> str:
    for table in ("run_metadata", "workflow_metadata"):
        try:
            result = workspace.query(f"SELECT scenario_name FROM {table}", max_rows=1)
        except Exception:
            continue
        if result.rows:
            return str(result.rows[0].get("scenario_name") or "")
    return ""


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
    return f"evidence_{spec.plot_type}_{spec.y_columns[0] if spec.y_columns else 'plot'}"


def _coerce_y_columns(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        out = [value]
    else:
        out = [str(item) for item in value]
    return [item.strip() for item in out if item and item.strip()]


def _replace_spec(spec: ReviewPlotSpec, **kwargs: Any) -> ReviewPlotSpec:
    return replace(spec, **{key: value for key, value in kwargs.items() if value is not None})


def _spec_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "group": "group_column",
        "style": "style_name",
        "title": "title",
        "subtitle": "subtitle",
        "x_label": "x_label",
        "y_label": "y_label",
        "file_format": "file_format",
        "dpi": "dpi",
        "max_rows": "max_rows",
    }
    out: dict[str, Any] = {}
    if "extra" in kwargs:
        out["extra"] = {"source": "oel_review_plot_api", **dict(kwargs["extra"] or {})}
    for source, target in mapping.items():
        if source not in kwargs:
            continue
        value = kwargs[source]
        if target == "style_name":
            value = _normalize_style_alias(str(value))
        out[target] = value
    return out


def _resolve_output_path(workspace: ReviewWorkspace, output: str | Path | None) -> Path | None:
    if output is None or str(output).strip() == "":
        return None
    path = Path(output)
    if path.is_absolute():
        return path
    return workspace.output_dir / path


def _spec_to_dict(spec: ReviewPlotSpec) -> dict[str, Any]:
    return {
        "sql": spec.sql,
        "x_column": spec.x_column,
        "y_columns": list(spec.y_columns),
        "plot_type": spec.plot_type,
        "group_column": spec.group_column,
        "style_name": spec.style_name,
        "title": spec.title,
        "subtitle": spec.subtitle,
        "x_label": spec.x_label,
        "y_label": spec.y_label,
        "artifact_id": spec.artifact_id,
        "file_format": spec.file_format,
        "dpi": spec.dpi,
        "max_rows": spec.max_rows,
        "extra": dict(spec.extra or {}),
    }


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _ordered_unique(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_artifact_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return cleaned[:80] or f"evidence_plot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


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


def _normalize_style_alias(value: str) -> str:
    style = str(value or "oel_dark").strip().lower()
    if style == "dark":
        return "oel_dark"
    if style == "light":
        return "oel_light"
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
    return _normalize_plot_type(spec.plot_type) != "heatmap" and (
        bool(spec.group_column) or len(spec.y_columns) > 1
    )


def _ensure_matplotlib_cache_env() -> None:
    status = configure_headless_runtime(force=True)
    if not status.ok:
        raise RuntimeError("Could not prepare headless plotting caches: " + "; ".join(status.errors))
