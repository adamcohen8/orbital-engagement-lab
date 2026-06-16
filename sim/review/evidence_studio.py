from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sim.review.plotting import (
    PLOT_TYPES,
    STYLE_NAMES,
    ReviewPlotArtifact,
    ReviewPlotSpec,
    default_plot_spec,
    numeric_columns,
    save_review_plot,
)
from sim.review.queries import get_saved_review_query, list_saved_review_queries
from sim.review.workspace import ReviewStoreNotFoundError, ReviewWorkspace


@dataclass(frozen=True)
class _PlotRecipe:
    recipe_id: str
    title: str
    description: str
    sql: str
    x_column: str
    y_columns: tuple[str, ...]
    plot_type: str
    x_label: str
    y_label: str
    artifact_id: str
    supported_tables: tuple[str, ...]


PLOT_RECIPES: dict[str, _PlotRecipe] = {
    "relative_range": _PlotRecipe(
        recipe_id="relative_range",
        title="Relative range over time",
        description="Plots deputy-chief range from the relative_state review table.",
        sql="SELECT time_s, deputy_id, chief_id, range_km FROM relative_state ORDER BY time_s",
        x_column="time_s",
        y_columns=("range_km",),
        plot_type="line",
        x_label="Time (s)",
        y_label="Range (km)",
        artifact_id="evidence_relative_range",
        supported_tables=("relative_state",),
    ),
    "relative_range_rate": _PlotRecipe(
        recipe_id="relative_range_rate",
        title="Relative range rate over time",
        description="Plots relative range rate from the relative_state review table.",
        sql="SELECT time_s, range_rate_km_s FROM relative_state ORDER BY time_s",
        x_column="time_s",
        y_columns=("range_rate_km_s",),
        plot_type="line",
        x_label="Time (s)",
        y_label="Range rate (km/s)",
        artifact_id="evidence_relative_range_rate",
        supported_tables=("relative_state",),
    ),
    "relative_velocity_components": _PlotRecipe(
        recipe_id="relative_velocity_components",
        title="Relative velocity components over time",
        description="Plots RIC-frame relative velocity components from the relative_state review table.",
        sql=(
            "SELECT time_s, v_radial_km_s, v_intrack_km_s, v_crosstrack_km_s "
            "FROM relative_state ORDER BY time_s"
        ),
        x_column="time_s",
        y_columns=("v_radial_km_s", "v_intrack_km_s", "v_crosstrack_km_s"),
        plot_type="line",
        x_label="Time (s)",
        y_label="Relative velocity (km/s)",
        artifact_id="evidence_relative_velocity",
        supported_tables=("relative_state",),
    ),
    "burn_activity": _PlotRecipe(
        recipe_id="burn_activity",
        title="Burn activity by object",
        description="Plots active thrust samples by object.",
        sql="SELECT object_id, SUM(burn_active) AS active_samples FROM thrust GROUP BY object_id ORDER BY object_id",
        x_column="object_id",
        y_columns=("active_samples",),
        plot_type="bar",
        x_label="Object",
        y_label="Active thrust samples",
        artifact_id="evidence_burn_activity",
        supported_tables=("thrust",),
    ),
    "campaign_closest_approach": _PlotRecipe(
        recipe_id="campaign_closest_approach",
        title="Campaign closest approach by iteration",
        description="Plots Monte Carlo closest-approach results by iteration.",
        sql="SELECT iteration, closest_approach_km FROM campaign_runs ORDER BY iteration",
        x_column="iteration",
        y_columns=("closest_approach_km",),
        plot_type="scatter",
        x_label="Iteration",
        y_label="Closest approach (km)",
        artifact_id="evidence_campaign_closest_approach",
        supported_tables=("campaign_runs",),
    ),
    "sensitivity_effects": _PlotRecipe(
        recipe_id="sensitivity_effects",
        title="Sensitivity effect sizes",
        description="Plots ranked sensitivity effect sizes by parameter.",
        sql="SELECT parameter_path, effect_size FROM sensitivity_rankings ORDER BY rank, parameter_path, metric_path",
        x_column="parameter_path",
        y_columns=("effect_size",),
        plot_type="bar",
        x_label="Parameter",
        y_label="Effect size",
        artifact_id="evidence_sensitivity_effects",
        supported_tables=("sensitivity_rankings",),
    ),
}


@dataclass(frozen=True)
class EvidenceSelection:
    kind: str = ""
    label: str = ""
    path: str = ""
    table: str = ""
    saved_query: str = ""
    sql: str = ""
    recipe_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "kind": self.kind,
                "label": self.label,
                "path": self.path,
                "table": self.table,
                "saved_query": self.saved_query,
                "sql": self.sql,
                "recipe_id": self.recipe_id,
            }.items()
            if value
        }


@dataclass(frozen=True)
class EvidenceStudioRequest:
    output_dir: Path
    instruction: str
    style_name: str = "oel_dark"
    file_format: str = "png"
    selection: EvidenceSelection | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class EvidenceStudioResult:
    status: str
    message: str
    artifact: ReviewPlotArtifact | None = None
    sql: str = ""
    recipe_id: str = ""
    plot_spec: ReviewPlotSpec | None = None
    selected_context: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "planned"}

    def to_dict(self) -> dict[str, Any]:
        artifact = None
        if self.artifact is not None:
            artifact = {
                "artifact_id": self.artifact.artifact_id,
                "path": str(self.artifact.path),
                "relative_path": self.artifact.relative_path,
                "row_count": self.artifact.row_count,
                "truncated": self.artifact.truncated,
            }
        return {
            "status": self.status,
            "message": self.message,
            "artifact": artifact,
            "sql": self.sql,
            "recipe_id": self.recipe_id,
            "plot_spec": asdict(self.plot_spec) if self.plot_spec is not None else None,
            "selected_context": dict(self.selected_context),
            "guardrails": evidence_studio_guardrails(),
        }


EVIDENCE_PLAN_SCHEMA_VERSION = "oel_evidence_plan.v1"
EVIDENCE_AGENT_WORKSPACE_DIRNAME = "evidence_studio_workspace"


@dataclass(frozen=True)
class EvidencePlotPlan:
    schema_version: str
    action: str
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
    rationale: str = ""
    recipe_id: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidencePlotPlan:
        return cls(
            schema_version=str(payload.get("schema_version", "") or ""),
            action=str(payload.get("action", "") or ""),
            sql=str(payload.get("sql", "") or ""),
            x_column=str(payload.get("x_column", "") or ""),
            y_columns=[str(item) for item in list(payload.get("y_columns", []) or [])],
            plot_type=str(payload.get("plot_type", "line") or "line"),
            group_column=str(payload.get("group_column", "") or ""),
            style_name=str(payload.get("style_name", "oel_dark") or "oel_dark"),
            title=str(payload.get("title", "") or ""),
            subtitle=str(payload.get("subtitle", "") or ""),
            x_label=str(payload.get("x_label", "") or ""),
            y_label=str(payload.get("y_label", "") or ""),
            artifact_id=str(payload.get("artifact_id", "") or ""),
            file_format=str(payload.get("file_format", "png") or "png"),
            rationale=str(payload.get("rationale", "") or ""),
            recipe_id=str(payload.get("recipe_id", "") or ""),
        )

    def to_spec(self, *, instruction: str, selected_context: dict[str, str]) -> ReviewPlotSpec:
        return _with_agent_provenance(
            ReviewPlotSpec(
                sql=self.sql,
                x_column=self.x_column,
                y_columns=list(self.y_columns),
                plot_type=self.plot_type,
                group_column=self.group_column,
                style_name=self.style_name,
                title=self.title,
                subtitle=self.subtitle,
                x_label=self.x_label,
                y_label=self.y_label,
                artifact_id=self.artifact_id,
                file_format=self.file_format,
                extra={
                    "planner_schema_version": self.schema_version,
                    "planner_rationale": self.rationale,
                    "planner_recipe_id": self.recipe_id,
                },
            ),
            instruction=instruction,
            selected_context=selected_context,
            generated_by="oel_evidence_studio_codex_plan",
        )


@dataclass(frozen=True)
class EvidenceAgentWorkspace:
    output_dir: Path
    workspace_dir: Path
    data_dir: Path
    generated_dir: Path
    manifest_path: Path
    task_packet_path: Path | None
    agents_path: Path
    readme_path: Path
    copied_files: int
    skipped_files: int
    review_db_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "workspace_dir": str(self.workspace_dir),
            "data_dir": str(self.data_dir),
            "generated_dir": str(self.generated_dir),
            "manifest_path": str(self.manifest_path),
            "task_packet_path": str(self.task_packet_path) if self.task_packet_path is not None else "",
            "agents_path": str(self.agents_path),
            "readme_path": str(self.readme_path),
            "copied_files": self.copied_files,
            "skipped_files": self.skipped_files,
            "review_db_path": str(self.review_db_path) if self.review_db_path is not None else "",
        }


def prepare_evidence_agent_workspace(
    output_dir: str | Path,
    *,
    workspace_name: str = EVIDENCE_AGENT_WORKSPACE_DIRNAME,
) -> EvidenceAgentWorkspace:
    """Create the bounded folder used by the Evidence Studio embedded agent."""

    source_output = Path(output_dir).expanduser().resolve()
    if not source_output.exists() or not source_output.is_dir():
        raise FileNotFoundError(f"Output folder does not exist: {source_output}")

    workspace_dir = source_output / workspace_name
    data_dir = workspace_dir / "data"
    generated_dir = workspace_dir / "generated"
    tools_dir = workspace_dir / "tools"
    for path in (data_dir, generated_dir, tools_dir):
        path.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    skipped_files = 0
    for src in sorted(source_output.rglob("*"), key=lambda item: str(item)):
        if not src.is_file():
            continue
        if _is_relative_to(src, workspace_dir):
            skipped_files += 1
            continue
        rel = src.relative_to(source_output)
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns and dst.stat().st_size == src.stat().st_size:
            skipped_files += 1
            continue
        shutil.copy2(src, dst)
        copied_files += 1

    _write_agent_workspace_tools(tools_dir)
    agents_path = workspace_dir / "AGENTS.md"
    readme_path = workspace_dir / "README.md"
    agents_path.write_text(_agent_workspace_agents_md(), encoding="utf-8")
    readme_path.write_text(_agent_workspace_readme_md(), encoding="utf-8")

    review_db_path = data_dir / "review" / "run.sqlite"
    manifest_path = workspace_dir / "evidence_manifest.json"
    task_packet_path: Path | None = None
    manifest = _agent_workspace_manifest(
        source_output=source_output,
        workspace_dir=workspace_dir,
        data_dir=data_dir,
        generated_dir=generated_dir,
        review_db_path=review_db_path if review_db_path.is_file() else None,
        copied_files=copied_files,
        skipped_files=skipped_files,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if review_db_path.is_file():
        task_packet = build_planner_task_packet(
            data_dir,
            instruction=(
                "Use only this Evidence Studio workspace to answer the user's brief-support request. "
                "Create any new artifacts under generated/ and cite the review tables, SQL, or files used."
            ),
        )
        task_packet["workspace_dir"] = str(workspace_dir)
        task_packet["generated_dir"] = str(generated_dir)
        task_packet["review_db"] = str(review_db_path)
        task_packet_path = workspace_dir / "task_packet.json"
        task_packet_path.write_text(json.dumps(task_packet, indent=2), encoding="utf-8")

    return EvidenceAgentWorkspace(
        output_dir=source_output,
        workspace_dir=workspace_dir,
        data_dir=data_dir,
        generated_dir=generated_dir,
        manifest_path=manifest_path,
        task_packet_path=task_packet_path,
        agents_path=agents_path,
        readme_path=readme_path,
        copied_files=copied_files,
        skipped_files=skipped_files,
        review_db_path=review_db_path if review_db_path.is_file() else None,
    )


def handle_evidence_studio_request(request: EvidenceStudioRequest) -> EvidenceStudioResult:
    instruction = str(request.instruction or "").strip()
    if not instruction:
        return EvidenceStudioResult(status="failed", message="Enter a plot request for the selected output.")

    try:
        workspace = ReviewWorkspace.open(request.output_dir)
    except ReviewStoreNotFoundError:
        return EvidenceStudioResult(
            status="failed",
            message=(
                "This output folder does not have review/run.sqlite, so Evidence Studio can preview existing "
                "artifacts but cannot generate a new evidence-backed plot."
            ),
        )
    except Exception as exc:
        return EvidenceStudioResult(status="failed", message=f"Could not open the review store: {exc}")

    selection = request.selection or EvidenceSelection()
    selected_context = selection.to_dict()
    try:
        spec, recipe_id = _request_plot_spec(
            workspace=workspace,
            instruction=instruction,
            selection=selection,
            style_name=request.style_name,
            file_format=request.file_format,
        )
        spec = _with_agent_provenance(spec, instruction=instruction, selected_context=selected_context)
        if request.dry_run:
            return EvidenceStudioResult(
                status="planned",
                message="Planned an Evidence Studio plot without writing an artifact.",
                sql=spec.sql,
                recipe_id=recipe_id,
                plot_spec=spec,
                selected_context=selected_context,
            )
        artifact = save_review_plot(workspace, spec)
    except Exception as exc:
        return EvidenceStudioResult(
            status="failed",
            message=f"Could not generate a plot from that request: {exc}",
            selected_context=selected_context,
        )

    return EvidenceStudioResult(
        status="ok",
        message=f"Generated {artifact.relative_path} from review-store evidence.",
        artifact=artifact,
        sql=spec.sql,
        recipe_id=recipe_id,
        plot_spec=spec,
        selected_context=selected_context,
    )


def evidence_studio_guardrails() -> list[str]:
    return [
        "Uses completed OEL output folders only.",
        "Reads review/run.sqlite through ReviewWorkspace read-only connections.",
        "Accepts only SELECT/WITH review SQL through the review query validator.",
        "Does not execute arbitrary Python or import scenario plugins.",
        "External planners may return only an EvidencePlotPlan JSON object.",
        "EvidencePlotPlan JSON is validated before any artifact is written.",
        "Generates plots through the OEL review plotting service with provenance.",
    ]


def evidence_agent_workspace_guardrails() -> list[str]:
    return [
        "The CLI agent starts inside evidence_studio_workspace, not the OEL source tree.",
        "The workspace contains copied run evidence under data/ and an empty generated/ output folder.",
        "Agent outputs should be written only under generated/.",
        "The agent instructions prohibit new simulations, invented data, and modification of copied source evidence.",
        "Evidence Studio can refresh and preview generated artifacts from the workspace.",
        "CLI sandbox strength depends on the launched agent command; use Codex workspace-write or stricter settings.",
    ]


def list_evidence_plot_recipes() -> list[dict[str, str]]:
    return [
        {
            "recipe_id": recipe.recipe_id,
            "title": recipe.title,
            "description": recipe.description,
        }
        for recipe in PLOT_RECIPES.values()
    ]


def evidence_plan_schema() -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_PLAN_SCHEMA_VERSION,
        "type": "object",
        "required": ["schema_version", "action", "sql", "x_column", "y_columns"],
        "properties": {
            "schema_version": {"const": EVIDENCE_PLAN_SCHEMA_VERSION},
            "action": {"const": "plot"},
            "sql": {"type": "string", "description": "Read-only SELECT/WITH query against review/run.sqlite."},
            "x_column": {"type": "string"},
            "y_columns": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "plot_type": {"enum": list(PLOT_TYPES)},
            "group_column": {"type": "string"},
            "style_name": {"enum": list(STYLE_NAMES)},
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "x_label": {"type": "string"},
            "y_label": {"type": "string"},
            "artifact_id": {"type": "string"},
            "file_format": {"enum": ["png", "svg", "pdf"]},
            "rationale": {"type": "string"},
            "recipe_id": {"type": "string"},
        },
        "additionalProperties": False,
    }


def build_planner_task_packet(
    output_dir: str | Path,
    *,
    instruction: str,
    selection: EvidenceSelection | None = None,
    max_columns_per_table: int = 40,
) -> dict[str, Any]:
    workspace = ReviewWorkspace.open(output_dir)
    schema = workspace.schema()
    columns_by_table = {}
    for table, columns in dict(schema.get("columns", {}) or {}).items():
        columns_by_table[table] = list(columns or [])[: max(int(max_columns_per_table), 1)]
    return {
        "task": "Return exactly one EvidencePlotPlan JSON object. Do not include markdown or prose.",
        "instruction": instruction,
        "output_dir": str(workspace.output_dir),
        "review_db": str(workspace.db_path),
        "selected_context": (selection or EvidenceSelection()).to_dict(),
        "available_tables": workspace.tables(),
        "columns": columns_by_table,
        "saved_queries": [
            {"name": item.name, "description": item.description, "sql": item.sql}
            for item in list_saved_review_queries()
        ],
        "plot_recipes": list_evidence_plot_recipes(),
        "plan_schema": evidence_plan_schema(),
        "guardrails": evidence_studio_guardrails(),
    }


def _agent_workspace_manifest(
    *,
    source_output: Path,
    workspace_dir: Path,
    data_dir: Path,
    generated_dir: Path,
    review_db_path: Path | None,
    copied_files: int,
    skipped_files: int,
) -> dict[str, Any]:
    files = []
    for path in sorted(data_dir.rglob("*"), key=lambda item: str(item)):
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(data_dir))
        except ValueError:
            rel = str(path)
        files.append({"path": rel, "bytes": path.stat().st_size})
    schema: dict[str, Any] = {}
    if review_db_path is not None and review_db_path.is_file():
        try:
            workspace = ReviewWorkspace.open(data_dir)
            schema = workspace.schema()
        except Exception as exc:
            schema = {"error": str(exc)}
    return {
        "workspace_kind": "oel_evidence_studio_agent_workspace",
        "source_output_dir": str(source_output),
        "workspace_dir": str(workspace_dir),
        "data_dir": str(data_dir),
        "generated_dir": str(generated_dir),
        "review_db": str(review_db_path) if review_db_path is not None else "",
        "copied_files": copied_files,
        "skipped_files": skipped_files,
        "files": files[:1000],
        "files_truncated": len(files) > 1000,
        "review_schema": schema,
        "guardrails": evidence_agent_workspace_guardrails(),
    }


def _agent_workspace_agents_md() -> str:
    return textwrap.dedent(
        """\
        # OEL Evidence Studio Agent Instructions

        You are operating inside a bounded Evidence Studio workspace for one completed OEL run.

        ## Hard boundaries

        - Use only files in this workspace.
        - Do not run OEL simulations, scenario generators, validation harnesses, or campaign workflows.
        - Do not invent, synthesize, or backfill missing data.
        - Do not modify files under `data/`; treat them as read-only evidence.
        - Write new plots, tables, notes, or reports under `generated/`.
        - When making a claim, cite the source file, review table, SQL query, or artifact used.
        - Prefer `data/review/run.sqlite` for quantitative work when it exists.

        ## Useful local files

        - `README.md`: quick orientation and example commands.
        - `evidence_manifest.json`: copied files, review schema, and guardrails.
        - `task_packet.json`: machine-readable review-store context when available.
        - `tools/query_review.py`: run read-only SQL against `data/review/run.sqlite`.
        - `tools/evidence_plot.py`: create simple OEL-styled plots from read-only SQL.

        ## Output expectations

        Put deliverables in `generated/`. Use concise, descriptive filenames. If you create a figure
        or report, include a sidecar note or embedded caption that names the data source and query.
        """
    )


def _agent_workspace_readme_md() -> str:
    return textwrap.dedent(
        """\
        # OEL Evidence Studio Workspace

        This folder is a bounded workspace for a CLI agent helping with a live technical brief.
        The original completed-run evidence has been copied into `data/`. New artifacts belong in
        `generated/`, where Evidence Studio can refresh and open them in the viewer.

        Start with:

        ```bash
        python tools/query_review.py --tables
        python tools/query_review.py --sql "SELECT * FROM run_metadata"
        ```

        Make a plot from review evidence:

        ```bash
        python tools/evidence_plot.py \\
          --sql "SELECT time_s, range_km FROM relative_state ORDER BY time_s" \\
          --x time_s --y range_km \\
          --title "Relative range over time" \\
          --output generated/relative_range.png
        ```

        Do not run new simulations from this workspace. Use only the copied evidence in `data/`.
        """
    )


def _write_agent_workspace_tools(tools_dir: Path) -> None:
    (tools_dir / "query_review.py").write_text(_query_review_tool_source(), encoding="utf-8")
    (tools_dir / "evidence_plot.py").write_text(_evidence_plot_tool_source(), encoding="utf-8")


def _query_review_tool_source() -> str:
    return textwrap.dedent(
        '''\
        from __future__ import annotations

        import argparse
        import csv
        import json
        import sqlite3
        import sys
        from pathlib import Path


        ROOT = Path(__file__).resolve().parents[1]
        DB_PATH = ROOT / "data" / "review" / "run.sqlite"


        def _validate_sql(sql: str) -> str:
            text = sql.strip()
            if not text:
                raise ValueError("SQL is required.")
            first = text.split(None, 1)[0].lower()
            if first not in {"select", "with"}:
                raise ValueError("Only SELECT or WITH queries are allowed.")
            return text


        def main() -> int:
            parser = argparse.ArgumentParser(description="Query copied OEL review evidence.")
            parser.add_argument("--sql", help="Read-only SELECT/WITH query.")
            parser.add_argument("--tables", action="store_true", help="List review tables.")
            parser.add_argument("--csv", action="store_true", help="Emit query rows as CSV instead of JSON.")
            args = parser.parse_args()
            if not DB_PATH.is_file():
                raise SystemExit(f"Missing review DB: {DB_PATH}")
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                if args.tables:
                    rows = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    ).fetchall()
                    print("\\n".join(row["name"] for row in rows))
                    return 0
                sql = _validate_sql(args.sql or "")
                rows = conn.execute(sql).fetchall()
                columns = list(rows[0].keys()) if rows else [item[0] for item in conn.execute(sql).description or []]
                if args.csv:
                    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(dict(row))
                else:
                    print(json.dumps({"columns": columns, "rows": [dict(row) for row in rows]}, indent=2))
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    )


def _evidence_plot_tool_source() -> str:
    return textwrap.dedent(
        '''\
        from __future__ import annotations

        import argparse
        import os
        import sqlite3
        from pathlib import Path


        ROOT = Path(__file__).resolve().parents[1]
        DB_PATH = ROOT / "data" / "review" / "run.sqlite"
        MPLCONFIGDIR = ROOT / ".matplotlib"
        XDG_CACHE_HOME = ROOT / ".cache"
        MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
        XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
        os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt


        DARK = {
            "figure": "#07111f",
            "axes": "#101a2b",
            "grid": "#28354a",
            "text": "#dce8f7",
            "line": "#38bdf8",
            "accent": "#f59e0b",
        }
        LIGHT = {
            "figure": "#f8fafc",
            "axes": "#ffffff",
            "grid": "#d7dee8",
            "text": "#142033",
            "line": "#0f76b7",
            "accent": "#b45309",
        }


        def _validate_sql(sql: str) -> str:
            text = sql.strip()
            if not text:
                raise ValueError("SQL is required.")
            first = text.split(None, 1)[0].lower()
            if first not in {"select", "with"}:
                raise ValueError("Only SELECT or WITH queries are allowed.")
            return text


        def _values(rows, column):
            out = []
            for row in rows:
                value = row[column]
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    out.append(value)
            return out


        def main() -> int:
            parser = argparse.ArgumentParser(description="Create an OEL-styled plot from copied review evidence.")
            parser.add_argument("--sql", required=True)
            parser.add_argument("--x", required=True)
            parser.add_argument("--y", action="append", required=True)
            parser.add_argument("--kind", choices=["line", "scatter", "bar"], default="line")
            parser.add_argument("--style", choices=["dark", "light"], default="dark")
            parser.add_argument("--title", default="")
            parser.add_argument("--xlabel", default="")
            parser.add_argument("--ylabel", default="")
            parser.add_argument("--output", required=True)
            args = parser.parse_args()
            if not DB_PATH.is_file():
                raise SystemExit(f"Missing review DB: {DB_PATH}")
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(_validate_sql(args.sql)).fetchall()
            if not rows:
                raise SystemExit("Query returned no rows.")
            columns = set(rows[0].keys())
            missing = [column for column in [args.x, *args.y] if column not in columns]
            if missing:
                raise SystemExit(f"Missing query columns: {', '.join(missing)}")

            palette = LIGHT if args.style == "light" else DARK
            plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
            fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=palette["figure"])
            ax.set_facecolor(palette["axes"])
            x = _values(rows, args.x)
            colors = [palette["line"], palette["accent"], "#22c55e", "#a855f7"]
            for idx, y_col in enumerate(args.y):
                y = _values(rows, y_col)
                color = colors[idx % len(colors)]
                if args.kind == "scatter":
                    ax.scatter(x, y, label=y_col, color=color, s=24)
                elif args.kind == "bar":
                    ax.bar(x, y, label=y_col, color=color)
                else:
                    ax.plot(x, y, label=y_col, color=color, linewidth=2.0, marker="o", markersize=3)
            ax.set_title(args.title or "OEL Evidence Plot", color=palette["text"])
            ax.set_xlabel(args.xlabel or args.x, color=palette["text"])
            ax.set_ylabel(args.ylabel or ", ".join(args.y), color=palette["text"])
            ax.tick_params(colors=palette["text"])
            for spine in ax.spines.values():
                spine.set_color(palette["grid"])
            ax.grid(True, color=palette["grid"], alpha=0.45)
            if len(args.y) > 1:
                legend = ax.legend()
                for text in legend.get_texts():
                    text.set_color(palette["text"])
            fig.tight_layout()
            output = Path(args.output)
            if not output.is_absolute():
                output = ROOT / output
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
            print(output)
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    )


def validate_evidence_plan(
    workspace: ReviewWorkspace,
    plan_payload: dict[str, Any],
    *,
    instruction: str = "",
    selected_context: dict[str, str] | None = None,
) -> ReviewPlotSpec:
    allowed = set(evidence_plan_schema()["properties"])
    unknown = sorted(set(plan_payload) - allowed)
    if unknown:
        raise ValueError(f"Evidence plan contains unsupported fields: {', '.join(unknown)}")
    plan = EvidencePlotPlan.from_dict(plan_payload)
    if plan.schema_version != EVIDENCE_PLAN_SCHEMA_VERSION:
        raise ValueError(f"Evidence plan schema_version must be {EVIDENCE_PLAN_SCHEMA_VERSION}.")
    if plan.action != "plot":
        raise ValueError("Evidence plan action must be 'plot'.")
    if plan.plot_type not in PLOT_TYPES:
        raise ValueError(f"Evidence plan plot_type must be one of: {', '.join(PLOT_TYPES)}.")
    if plan.style_name not in STYLE_NAMES:
        raise ValueError(f"Evidence plan style_name must be one of: {', '.join(STYLE_NAMES)}.")
    if plan.file_format not in {"png", "svg", "pdf"}:
        raise ValueError("Evidence plan file_format must be png, svg, or pdf.")
    if not plan.sql.strip():
        raise ValueError("Evidence plan sql is required.")
    if not plan.x_column.strip():
        raise ValueError("Evidence plan x_column is required.")
    if not plan.y_columns:
        raise ValueError("Evidence plan requires at least one y_column.")

    result = workspace.query(plan.sql, max_rows=5000)
    columns = set(result.columns)
    if plan.x_column not in columns:
        raise ValueError(f"Evidence plan x_column '{plan.x_column}' is not in query results.")
    for column in plan.y_columns:
        if column not in columns:
            raise ValueError(f"Evidence plan y_column '{column}' is not in query results.")
    if plan.group_column and plan.group_column not in columns:
        raise ValueError(f"Evidence plan group_column '{plan.group_column}' is not in query results.")
    numeric = set(numeric_columns(result))
    for column in plan.y_columns:
        if column not in numeric:
            raise ValueError(f"Evidence plan y_column '{column}' must contain numeric values.")
    if plan.group_column and len(plan.y_columns) != 1:
        raise ValueError("Evidence plan grouped plots support exactly one y_column.")
    return plan.to_spec(instruction=instruction, selected_context=selected_context or {})


def execute_evidence_plan(
    output_dir: str | Path,
    plan_payload: dict[str, Any],
    *,
    instruction: str = "",
    selected_context: dict[str, str] | None = None,
    dry_run: bool = False,
) -> EvidenceStudioResult:
    try:
        workspace = ReviewWorkspace.open(output_dir)
        spec = validate_evidence_plan(
            workspace,
            plan_payload,
            instruction=instruction,
            selected_context=selected_context,
        )
        if dry_run:
            return EvidenceStudioResult(
                status="planned",
                message="Validated an EvidencePlotPlan without writing an artifact.",
                sql=spec.sql,
                recipe_id=str(dict(spec.extra or {}).get("planner_recipe_id", "") or ""),
                plot_spec=spec,
                selected_context=dict(selected_context or {}),
            )
        artifact = save_review_plot(workspace, spec)
        return EvidenceStudioResult(
            status="ok",
            message=f"Executed EvidencePlotPlan and generated {artifact.relative_path}.",
            artifact=artifact,
            sql=spec.sql,
            recipe_id=str(dict(spec.extra or {}).get("planner_recipe_id", "") or ""),
            plot_spec=spec,
            selected_context=dict(selected_context or {}),
        )
    except Exception as exc:
        return EvidenceStudioResult(status="failed", message=f"Evidence plan rejected: {exc}")


def run_external_planner(command: list[str], packet: dict[str, Any], *, timeout_s: int = 120) -> dict[str, Any]:
    if not command:
        raise ValueError("Planner command is required.")
    proc = subprocess.run(
        command,
        input=json.dumps(packet, indent=2),
        text=True,
        capture_output=True,
        timeout=max(int(timeout_s), 1),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"planner command failed with exit {proc.returncode}: {proc.stderr.strip()}")
    return _load_json_object(proc.stdout)


def _request_plot_spec(
    *,
    workspace: ReviewWorkspace,
    instruction: str,
    selection: EvidenceSelection,
    style_name: str,
    file_format: str,
) -> tuple[ReviewPlotSpec, str]:
    tables = set(workspace.tables())

    if selection.recipe_id:
        return _recipe_spec(selection.recipe_id, style_name=style_name, file_format=file_format), selection.recipe_id

    recipe_id = _choose_recipe(instruction, tables)
    if recipe_id:
        return _recipe_spec(recipe_id, style_name=style_name, file_format=file_format), recipe_id

    if selection.sql:
        result = workspace.query(selection.sql, max_rows=5000)
        spec = default_plot_spec(selection.sql, result, artifact_id=_artifact_id_from_instruction(instruction))
        return _style_spec(spec, style_name=style_name, file_format=file_format, title=instruction), ""

    if selection.table:
        sql = f"SELECT * FROM {_quote_identifier(selection.table)} LIMIT 1000"
        result = workspace.query(sql, max_rows=1000)
        spec = default_plot_spec(sql, result, artifact_id=_artifact_id_from_instruction(instruction))
        return _style_spec(spec, style_name=style_name, file_format=file_format, title=instruction), ""

    if "relative_state" in tables:
        return _recipe_spec("relative_range", style_name=style_name, file_format=file_format), "relative_range"

    for candidate in ("campaign_closest_approach", "sensitivity_effects", "burn_activity"):
        recipe = PLOT_RECIPES[candidate]
        if all(table in tables for table in recipe.supported_tables):
            return _recipe_spec(candidate, style_name=style_name, file_format=file_format), candidate

    raise ValueError("No supported plot recipe matched the instruction or selected evidence source.")


def _choose_recipe(instruction: str, tables: set[str]) -> str:
    text = instruction.lower()
    candidates: list[tuple[str, tuple[str, ...]]] = [
        ("relative_velocity_components", ("relative velocity", "velocity component", "velocity over time")),
        ("relative_range_rate", ("range rate", "closing", "closing speed", "approach speed", "relative speed")),
        ("relative_range", ("range", "distance", "separation", "rendezvous")),
        ("burn_activity", ("burn", "thrust", "accel", "acceleration", "delta-v", "dv")),
        ("campaign_closest_approach", ("campaign", "monte carlo", "iteration", "closest approach")),
        ("sensitivity_effects", ("sensitivity", "effect", "ranking", "parameter")),
    ]
    for recipe_id, phrases in candidates:
        recipe = PLOT_RECIPES[recipe_id]
        if not all(table in tables for table in recipe.supported_tables):
            continue
        if any(phrase in text for phrase in phrases):
            return recipe_id
    return ""


def _recipe_spec(recipe_id: str, *, style_name: str, file_format: str) -> ReviewPlotSpec:
    recipe = PLOT_RECIPES.get(recipe_id)
    if recipe is None:
        raise ValueError(f"Unknown plot recipe '{recipe_id}'.")
    return ReviewPlotSpec(
        sql=recipe.sql,
        x_column=recipe.x_column,
        y_columns=list(recipe.y_columns),
        plot_type=recipe.plot_type,
        style_name=style_name,
        title=recipe.title,
        x_label=recipe.x_label,
        y_label=recipe.y_label,
        artifact_id=recipe.artifact_id,
        file_format=file_format,
    )


def _style_spec(
    spec: ReviewPlotSpec,
    *,
    style_name: str,
    file_format: str,
    title: str,
) -> ReviewPlotSpec:
    return ReviewPlotSpec(
        sql=spec.sql,
        x_column=spec.x_column,
        y_columns=list(spec.y_columns),
        plot_type=spec.plot_type,
        group_column=spec.group_column,
        style_name=style_name,
        title=title or spec.title,
        subtitle=spec.subtitle,
        x_label=spec.x_label,
        y_label=spec.y_label,
        artifact_id=spec.artifact_id,
        file_format=file_format,
        dpi=spec.dpi,
        max_rows=spec.max_rows,
        extra=dict(spec.extra or {}),
    )


def _with_agent_provenance(
    spec: ReviewPlotSpec,
    *,
    instruction: str,
    selected_context: dict[str, str],
    generated_by: str = "oel_evidence_studio_agent",
) -> ReviewPlotSpec:
    extra = dict(spec.extra or {})
    extra.update(
        {
            "generated_by": generated_by,
            "user_instruction": instruction,
            "selected_context": selected_context,
        }
    )
    return ReviewPlotSpec(
        sql=spec.sql,
        x_column=spec.x_column,
        y_columns=list(spec.y_columns),
        plot_type=spec.plot_type,
        group_column=spec.group_column,
        style_name=spec.style_name,
        title=spec.title,
        subtitle=spec.subtitle,
        x_label=spec.x_label,
        y_label=spec.y_label,
        artifact_id=spec.artifact_id,
        file_format=spec.file_format,
        dpi=spec.dpi,
        max_rows=spec.max_rows,
        extra=extra,
    )


def _artifact_id_from_instruction(instruction: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")
    if not slug:
        slug = "custom_plot"
    return f"evidence_{slug[:48].strip('_')}"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _load_json_object(text: str) -> dict[str, Any]:
    data = json.loads(str(text or "").strip())
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    return data


def _load_plan_arg(*, plan_json: str = "", plan_file: str = "") -> dict[str, Any]:
    if plan_json:
        return _load_json_object(plan_json)
    if plan_file:
        return _load_json_object(Path(plan_file).read_text(encoding="utf-8"))
    raise ValueError("Plan JSON or plan file is required.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the guarded OEL Evidence Studio agent against a completed output folder."
    )
    parser.add_argument("output_dir", nargs="?", help="Completed OEL output folder.")
    parser.add_argument("--ask", "-a", help="Natural-language plot request.")
    parser.add_argument("--style", default="oel_dark", choices=("oel_dark", "oel_light"), help="OEL plot style.")
    parser.add_argument("--format", default="png", choices=("png", "svg", "pdf"), help="Output figure format.")
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--table", help="Use a review table as selected context.")
    selection_group.add_argument("--saved-query", help="Use a built-in saved review query as selected context.")
    selection_group.add_argument("--sql", help="Use a read-only SELECT/WITH query as selected context.")
    selection_group.add_argument("--recipe", help="Force a built-in Evidence Studio plot recipe.")
    plan_group = parser.add_mutually_exclusive_group()
    plan_group.add_argument("--plan-json", help="Validate/execute an EvidencePlotPlan JSON object.")
    plan_group.add_argument("--plan-file", help="Validate/execute an EvidencePlotPlan JSON file.")
    plan_group.add_argument(
        "--planner-command",
        help=(
            "External planner command. OEL sends a task packet on stdin and expects exactly one "
            "EvidencePlotPlan JSON object on stdout."
        ),
    )
    parser.add_argument("--task-packet", action="store_true", help="Print the bounded planner task packet and exit.")
    parser.add_argument("--prepare-workspace", action="store_true", help="Prepare a bounded CLI-agent workspace and exit.")
    parser.add_argument("--plan-schema", action="store_true", help="Print the EvidencePlotPlan schema and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Plan/validate without writing an artifact.")
    parser.add_argument("--list-recipes", action="store_true", help="List guarded plot recipes and exit.")
    parser.add_argument("--guardrails", action="store_true", help="Print the Evidence Studio guardrails and exit.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.plan_schema:
        _print_cli_payload({"plan_schema": evidence_plan_schema()}, json_mode=True)
        return 0
    if args.guardrails:
        payload = {"guardrails": evidence_studio_guardrails()}
        _print_cli_payload(payload, json_mode=args.json)
        return 0
    if args.list_recipes:
        payload = {"recipes": list_evidence_plot_recipes()}
        _print_cli_payload(payload, json_mode=args.json)
        return 0
    if not args.output_dir:
        parser.error("output_dir is required unless --list-recipes, --guardrails, or --plan-schema is used")

    if args.prepare_workspace:
        workspace = prepare_evidence_agent_workspace(args.output_dir)
        _print_cli_payload({"agent_workspace": workspace.to_dict()}, json_mode=args.json)
        return 0

    selection = _selection_from_cli(args)
    if args.task_packet:
        if not args.ask:
            parser.error("--ask is required when emitting a planner task packet")
        packet = build_planner_task_packet(args.output_dir, instruction=args.ask, selection=selection)
        _print_cli_payload({"task_packet": packet}, json_mode=True)
        return 0
    if args.plan_json or args.plan_file:
        plan = _load_plan_arg(plan_json=args.plan_json or "", plan_file=args.plan_file or "")
        result = execute_evidence_plan(
            args.output_dir,
            plan,
            instruction=args.ask or "",
            selected_context=selection.to_dict(),
            dry_run=bool(args.dry_run),
        )
        _print_cli_payload(result.to_dict(), json_mode=args.json)
        return 0 if result.ok else 2
    if args.planner_command:
        if not args.ask:
            parser.error("--ask is required when running an external planner")
        packet = build_planner_task_packet(args.output_dir, instruction=args.ask, selection=selection)
        try:
            plan = run_external_planner(shlex.split(args.planner_command), packet)
        except Exception as exc:
            print(f"planner failed: {exc}", file=sys.stderr)
            return 2
        result = execute_evidence_plan(
            args.output_dir,
            plan,
            instruction=args.ask,
            selected_context=selection.to_dict(),
            dry_run=bool(args.dry_run),
        )
        _print_cli_payload(result.to_dict(), json_mode=args.json)
        return 0 if result.ok else 2
    if not args.ask:
        parser.error("--ask is required when generating or planning a plot")
    result = handle_evidence_studio_request(
        EvidenceStudioRequest(
            output_dir=Path(args.output_dir),
            instruction=args.ask,
            style_name=args.style,
            file_format=args.format,
            selection=selection,
            dry_run=bool(args.dry_run),
        )
    )
    payload = result.to_dict()
    _print_cli_payload(payload, json_mode=args.json)
    return 0 if result.ok else 2


def _selection_from_cli(args: argparse.Namespace) -> EvidenceSelection:
    if args.table:
        return EvidenceSelection(kind="table", label=f"Table: {args.table}", table=args.table)
    if args.sql:
        return EvidenceSelection(kind="query", label="Custom SQL", sql=args.sql)
    if args.recipe:
        return EvidenceSelection(kind="plot_recipe", label=f"Plot Recipe: {args.recipe}", recipe_id=args.recipe)
    if args.saved_query:
        saved = get_saved_review_query(args.saved_query)
        if saved is None:
            raise SystemExit(f"unknown saved review query: {args.saved_query}")
        return EvidenceSelection(
            kind="saved_query",
            label=f"Saved Query: {saved.name}",
            saved_query=saved.name,
            sql=saved.sql,
        )
    return EvidenceSelection()


def _print_cli_payload(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2))
        return
    if "guardrails" in payload and len(payload) == 1:
        for item in payload["guardrails"]:
            print(f"- {item}")
        return
    if "recipes" in payload:
        for item in payload["recipes"]:
            print(f"{item.get('recipe_id')}: {item.get('title')}")
        return
    if "plan_schema" in payload or "task_packet" in payload:
        print(json.dumps(payload, indent=2))
        return
    print(f"status: {payload.get('status')}")
    print(f"message: {payload.get('message')}")
    if payload.get("recipe_id"):
        print(f"recipe_id: {payload.get('recipe_id')}")
    if payload.get("sql"):
        print("sql:")
        print(payload.get("sql"))
    artifact = payload.get("artifact")
    if isinstance(artifact, dict) and artifact.get("path"):
        print(f"artifact: {artifact.get('path')}")


if __name__ == "__main__":
    raise SystemExit(main())
