from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sim.agent_task.failures import diagnose_failure
from sim.agent_task.models import AgentTaskRecipe, EvidencePacket
from sim.agent_task.plot_recipes import get_plot_recipe, review_plot_spec
from sim.agent_task.recipes import get_recipe
from sim.agent_task.semantics import semantic_metric_dicts
from sim.api import SimulationWorkspace
from sim.execution import run_simulation_config_file
from sim.review import (
    ReviewQueryError,
    ReviewStoreNotFoundError,
    ReviewWorkspace,
    get_saved_review_query,
    load_workflow_manifest,
)
from sim.review.plotting import ReviewPlotArtifact, save_review_plot

DEFAULT_INSPECTION_QUERIES = ("run_metadata", "objects", "artifacts")
DEFAULT_COMPARISON_METRICS = ("initial_range_km", "final_range_km", "closest_approach_km", "closest_approach_time_s")


def run_recipe(
    recipe_id: str,
    *,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    dry_run: bool = False,
    make_plots: bool = False,
    style_name: str = "oel_dark",
    max_rows: int = 50,
) -> dict[str, Any]:
    recipe = get_recipe(recipe_id)
    if recipe is None:
        raise ValueError(f"Unknown agent task recipe: {recipe_id}")
    prepared = prepare_recipe_config(recipe, output_dir=output_dir, output_root=output_root)
    task_id = recipe.recipe_id
    validation = _workspace().validate(prepared["config_path"])
    packet = EvidencePacket(
        task_id=task_id,
        status="validated" if bool(validation.get("ok")) else "failed",
        generated_utc=_utc_now(),
        recipe=recipe.to_dict(),
        configs=[prepared],
        validation=validation,
        semantic_metrics=semantic_metric_dicts(recipe.semantic_metric_names),
    )
    if not bool(validation.get("ok")):
        packet.failure_hints = [hint.to_dict() for hint in diagnose_failure(validation=validation, output_dir=prepared["output_dir"])]
        return _write_packet(packet, prepared["output_dir"])
    if dry_run:
        packet.caveats.append("Dry run requested: scenario was validated but not executed.")
        return _write_packet(packet, prepared["output_dir"])

    try:
        payload = run_simulation_config_file(prepared["config_path"])
        packet.run = _run_summary(payload)
        packet.status = "completed"
    except Exception as exc:
        packet.status = "failed"
        packet.run = {"error": str(exc)}
        packet.failure_hints = [
            hint.to_dict() for hint in diagnose_failure(str(exc), validation=validation, output_dir=prepared["output_dir"])
        ]
        return _write_packet(packet, prepared["output_dir"])

    inspection = inspect_output(
        prepared["output_dir"],
        query_names=recipe.query_names,
        max_rows=max_rows,
        semantic_metric_names=recipe.semantic_metric_names,
        write_packet=False,
    )
    packet.review = dict(inspection.get("review", {}) or {})
    packet.artifacts = list(inspection.get("artifacts", []) or [])
    packet.failure_hints.extend(list(inspection.get("failure_hints", []) or []))
    if make_plots:
        packet.plots = _make_plots_for_recipe(
            prepared["output_dir"],
            recipe.plot_recipe_ids,
            style_name=style_name,
        )
    return _write_packet(packet, prepared["output_dir"])


def inspect_output(
    output_dir: str | Path,
    *,
    query_names: tuple[str, ...] | list[str] | None = None,
    max_rows: int = 50,
    semantic_metric_names: tuple[str, ...] | list[str] = (),
    write_packet: bool = True,
) -> dict[str, Any]:
    outdir = Path(output_dir).expanduser().resolve()
    query_names = tuple(query_names or DEFAULT_INSPECTION_QUERIES)
    packet = EvidencePacket(
        task_id=f"inspect_{outdir.name}",
        status="completed",
        generated_utc=_utc_now(),
        task_type="inspect_output",
        configs=[{"output_dir": str(outdir)}],
        semantic_metrics=semantic_metric_dicts(tuple(semantic_metric_names)),
    )
    review: dict[str, Any] = {"output_dir": str(outdir), "queries": []}
    try:
        workspace = ReviewWorkspace.open(outdir)
        review.update(
            {
                "db_path": str(workspace.db_path),
                "tables": workspace.tables(),
                "schema": workspace.schema(),
                "saved_views": workspace.saved_views(),
            }
        )
        review["queries"] = _run_saved_queries(workspace, query_names, max_rows=max_rows)
        packet.artifacts = _artifact_rows(review["queries"])
    except (ReviewStoreNotFoundError, ReviewQueryError, ValueError) as exc:
        packet.status = "partial"
        packet.failure_hints = [hint.to_dict() for hint in diagnose_failure(str(exc), output_dir=outdir)]
        review["error"] = str(exc)

    workflow_manifest = _maybe_workflow_manifest(outdir)
    if workflow_manifest:
        review["workflow_manifest"] = workflow_manifest
        packet.artifacts.extend(_workflow_artifacts(workflow_manifest))
    packet.review = review
    if write_packet:
        return _write_packet(packet, outdir)
    return packet.to_dict()


def create_plot(
    output_dir: str | Path,
    plot_recipe_id: str,
    *,
    style_name: str = "oel_dark",
    file_format: str = "png",
    artifact_id: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    recipe = get_plot_recipe(plot_recipe_id)
    if recipe is None:
        raise ValueError(f"Unknown agent plot recipe: {plot_recipe_id}")
    workspace = ReviewWorkspace.open(output_dir)
    spec = review_plot_spec(
        recipe,
        style_name=style_name,
        file_format=file_format,
        artifact_id=artifact_id or recipe.artifact_id,
    )
    artifact = save_review_plot(workspace, spec, path=path)
    return _plot_artifact_dict(artifact, recipe_id=recipe.recipe_id)


def compare_configs(
    base_config: str | Path,
    candidate_config: str | Path,
    *,
    output_dir: str | Path,
    metric_names: tuple[str, ...] | list[str] | None = None,
    max_rows: int = 50,
) -> dict[str, Any]:
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = tuple(metric_names or DEFAULT_COMPARISON_METRICS)
    configs = [
        _prepare_config(base_config, outdir / "base", label="base"),
        _prepare_config(candidate_config, outdir / "candidate", label="candidate"),
    ]
    workspace = _workspace()
    validations = {item["label"]: workspace.validate(item["config_path"]) for item in configs}
    packet = EvidencePacket(
        task_id=f"compare_{Path(base_config).stem}_vs_{Path(candidate_config).stem}",
        task_type="compare_configs",
        status="completed",
        generated_utc=_utc_now(),
        configs=configs,
        validation=validations,
        semantic_metrics=semantic_metric_dicts(metrics),
    )
    if not all(bool(report.get("ok")) for report in validations.values()):
        packet.status = "failed"
        packet.failure_hints = [
            hint.to_dict()
            for report in validations.values()
            for hint in diagnose_failure(validation=report, output_dir=report.get("output_dir"))
        ]
        return _write_packet(packet, outdir)

    runs: dict[str, Any] = {}
    inspections: dict[str, Any] = {}
    for item in configs:
        label = str(item["label"])
        try:
            runs[label] = _run_summary(run_simulation_config_file(item["config_path"]))
            inspections[label] = inspect_output(
                item["output_dir"],
                query_names=("run_metadata", "rendezvous_metrics", "rendezvous_closest_approach", "artifacts"),
                max_rows=max_rows,
                semantic_metric_names=metrics,
                write_packet=False,
            )
        except Exception as exc:
            packet.status = "failed"
            runs[label] = {"error": str(exc)}
            packet.failure_hints.extend(
                hint.to_dict() for hint in diagnose_failure(str(exc), output_dir=item["output_dir"])
            )
            return _write_packet(packet, outdir)

    metric_table = {label: _extract_metric_values(data, metrics) for label, data in inspections.items()}
    packet.run = runs
    packet.review = {"base": inspections.get("base", {}).get("review", {}), "candidate": inspections.get("candidate", {}).get("review", {})}
    packet.artifacts = [
        {"label": label, **artifact}
        for label, inspection in inspections.items()
        for artifact in list(inspection.get("artifacts", []) or [])
    ]
    packet.comparison = {
        "metric_names": list(metrics),
        "metrics": metric_table,
        "deltas": _metric_deltas(metric_table.get("base", {}), metric_table.get("candidate", {})),
    }
    return _write_packet(packet, outdir)


def prepare_recipe_config(
    recipe: AgentTaskRecipe,
    *,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _repo_root()
    config_path = (root / recipe.config_path).resolve()
    if output_dir is None and output_root is not None:
        output_dir = Path(output_root).expanduser().resolve() / recipe.recipe_id
    if output_dir is None:
        data = _load_yaml(config_path)
        output_dir = data.get("outputs", {}).get("output_dir", "")
    return _prepare_config(config_path, output_dir, label=recipe.recipe_id)


def _prepare_config(config_path: str | Path, output_dir: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(config_path).expanduser().resolve()
    outdir = Path(output_dir).expanduser().resolve()
    data = _load_yaml(source)
    data.setdefault("outputs", {})["output_dir"] = str(outdir)
    data.setdefault("outputs", {}).setdefault("review", {})
    data["outputs"]["review"]["enabled"] = True
    data["outputs"]["review"].setdefault("detail", "standard")
    task_config = outdir / "agent_task_config.yaml"
    outdir.mkdir(parents=True, exist_ok=True)
    task_config.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return {
        "label": str(label),
        "source_config": str(source),
        "config_path": str(task_config),
        "output_dir": str(outdir),
        "review_enabled": True,
    }


def _run_saved_queries(workspace: ReviewWorkspace, query_names: tuple[str, ...], *, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in query_names:
        saved = get_saved_review_query(name)
        if saved is None:
            rows.append({"name": name, "status": "unknown_query"})
            continue
        try:
            result = workspace.query(saved.sql, max_rows=max_rows)
            rows.append(
                {
                    "name": saved.name,
                    "description": saved.description,
                    "sql": saved.sql,
                    "status": "ok",
                    "columns": result.columns,
                    "rows": result.rows,
                    "row_count": result.row_count,
                    "truncated": result.truncated,
                }
            )
        except ReviewQueryError as exc:
            rows.append(
                {
                    "name": saved.name,
                    "description": saved.description,
                    "sql": saved.sql,
                    "status": "failed",
                    "error": str(exc),
                    "failure_hints": [hint.to_dict() for hint in diagnose_failure(str(exc))],
                }
            )
    return rows


def _make_plots_for_recipe(output_dir: str | Path, plot_recipe_ids: tuple[str, ...], *, style_name: str) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    for recipe_id in plot_recipe_ids:
        try:
            plots.append(create_plot(output_dir, recipe_id, style_name=style_name))
        except Exception as exc:
            plots.append(
                {
                    "recipe_id": recipe_id,
                    "status": "failed",
                    "error": str(exc),
                    "failure_hints": [hint.to_dict() for hint in diagnose_failure(str(exc), output_dir=output_dir)],
                }
            )
    return plots


def _extract_metric_values(inspection: dict[str, Any], metric_names: tuple[str, ...]) -> dict[str, Any]:
    wanted = set(metric_names)
    values: dict[str, Any] = {}
    review = dict(inspection.get("review", {}) or {})
    for query in list(review.get("queries", []) or []):
        for row in list(query.get("rows", []) or []):
            metric_name = str(row.get("metric_name", "") or "")
            if metric_name in wanted and "value" in row:
                values[metric_name] = row.get("value")
            if query.get("name") == "rendezvous_closest_approach" and "closest_approach_km" in wanted:
                values.setdefault("closest_approach_km", row.get("range_km"))
                values.setdefault("closest_approach_time_s", row.get("time_s"))
    return values


def _metric_deltas(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for name in sorted(set(base) & set(candidate)):
        try:
            deltas[name] = float(candidate[name]) - float(base[name])
        except (TypeError, ValueError):
            continue
    return deltas


def _artifact_rows(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for query in query_rows:
        if query.get("name") not in {"artifacts", "workflow_artifacts"} or query.get("status") != "ok":
            continue
        for row in list(query.get("rows", []) or []):
            artifacts.append(dict(row))
    return artifacts


def _workflow_artifacts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(manifest.get("artifacts", []) or []) if isinstance(item, dict)]


def _maybe_workflow_manifest(output_dir: Path) -> dict[str, Any] | None:
    try:
        return load_workflow_manifest(output_dir)
    except Exception:
        return None


def _run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload.get("run", {})
    return {
        "summary": dict(summary or {}),
        "analysis": dict(payload.get("analysis", {}) or {}),
        "monte_carlo": dict(payload.get("monte_carlo", {}) or {}),
    }


def _plot_artifact_dict(artifact: ReviewPlotArtifact, *, recipe_id: str) -> dict[str, Any]:
    return {
        "recipe_id": recipe_id,
        "artifact_id": artifact.artifact_id,
        "path": str(artifact.path),
        "relative_path": artifact.relative_path,
        "row_count": artifact.row_count,
        "truncated": artifact.truncated,
        "spec": asdict(artifact.spec),
        "status": "ok",
    }


def _write_packet(packet: EvidencePacket, output_dir: str | Path) -> dict[str, Any]:
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(packet.to_dict())
    packet_path = outdir / "agent_evidence_packet.json"
    payload["packet_path"] = str(packet_path)
    packet_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Scenario config must be a YAML mapping: {path}")
    return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _workspace() -> SimulationWorkspace:
    root = _repo_root()
    return SimulationWorkspace(workspace_root=root, read_roots=(root,), write_roots=(root,))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
