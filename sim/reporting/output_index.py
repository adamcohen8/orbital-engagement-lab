from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any


def _scalar(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value).strip()
    return text if text else "not available"


def _path_link(path_text: Any, *, base_dir: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return "not available"
    try:
        path = Path(text)
        resolved = path if path.is_absolute() else Path.cwd() / path
        display = str(resolved.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        try:
            path = Path(text)
            resolved = path if path.is_absolute() else Path.cwd() / path
            display = str(Path(os.path.relpath(str(resolved.resolve()), str(base_dir.resolve()))))
        except Exception:
            display = text
    except Exception:
        display = text
    href = f"<{display}>" if any(ch.isspace() for ch in display) else display
    return f"[`{display}`]({href})"


def _literal(value: Any) -> str:
    text = _scalar(value).replace("`", "\\`")
    return f"`{text}`"


def _artifact_value(key: str, value: Any, *, base_dir: Path) -> str:
    key_lower = key.lower()
    if "error" in key_lower or "warning" in key_lower or key_lower.endswith("status"):
        return _literal(value)
    if isinstance(value, (str, os.PathLike)):
        return _path_link(value, base_dir=base_dir)
    return _literal(value)


def _flatten_artifacts(value: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, child in sorted(value.items()):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_artifacts(child, prefix=child_prefix))
        return rows
    if isinstance(value, list):
        rows = []
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            rows.extend(_flatten_artifacts(child, prefix=child_prefix))
        return rows
    return [(prefix or "artifact", value)]


def _artifact_lines(artifacts: dict[str, Any], *, base_dir: Path, limit: int = 60) -> list[str]:
    rows = [(key, value) for key, value in _flatten_artifacts(artifacts) if str(value or "").strip()]
    if not rows:
        return ["- No saved artifacts were listed for this run."]
    lines = [f"- `{key}`: {_artifact_value(key, value, base_dir=base_dir)}" for key, value in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"- ... {len(rows) - limit} additional artifacts omitted from this index")
    return lines


def _artifact_basename(artifacts: dict[str, Any], key: str, fallback: str) -> str:
    value = artifacts.get(key)
    if isinstance(value, (str, os.PathLike)) and str(value).strip():
        return Path(value).name
    return fallback


def _default_next_steps(*, workflow: str, artifacts: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    if workflow == "single_run":
        if "summary_json" in artifacts:
            name = _artifact_basename(artifacts, "summary_json", "master_run_summary.json")
            steps.append(f"Open `{name}` for stable run metadata and metrics.")
        if "run_log_json" in artifacts:
            name = _artifact_basename(artifacts, "run_log_json", "master_run_log.json")
            steps.append(f"Open `{name}` for saved time histories and custom plotting data.")
        steps.append("Inspect generated plot or animation artifacts listed below.")
    elif workflow == "monte_carlo":
        if "summary_json" in artifacts:
            name = _artifact_basename(artifacts, "summary_json", "master_monte_carlo_summary.json")
            steps.append(f"Open `{name}` for aggregate campaign results.")
        if "commander_brief_md" in artifacts:
            name = _artifact_basename(artifacts, "commander_brief_md", "master_monte_carlo_commander_brief.md")
            steps.append(f"Open `{name}` for the human-readable campaign brief.")
        steps.append("Inspect campaign plots and AI report artifacts when present.")
    elif workflow == "sensitivity":
        if "report_md" in artifacts:
            name = _artifact_basename(artifacts, "report_md", "master_analysis_sensitivity_report.md")
            steps.append(f"Open `{name}` for the human-readable study report.")
        if "rankings_csv" in artifacts:
            name = _artifact_basename(artifacts, "rankings_csv", "master_analysis_sensitivity_rankings.csv")
            steps.append(f"Open `{name}` to inspect ranked parameter effects.")
        steps.append("Inspect generated response, scatter, grid, or ranking figures listed below.")
    else:
        steps.append("Inspect the artifact inventory below.")
    return steps or ["Inspect the artifact inventory below."]


def _single_run_metrics(summary: dict[str, Any]) -> list[str]:
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    total_dv = 0.0
    for row in thrust_stats.values():
        try:
            total_dv += float(dict(row or {}).get("total_dv_m_s", 0.0))
        except (TypeError, ValueError):
            pass
    return [
        f"- Samples: `{_scalar(summary.get('samples'))}`",
        f"- Duration: `{_scalar(summary.get('duration_s'))} s`",
        f"- Objects: `{_scalar(', '.join(list(summary.get('objects', []) or [])))}`",
        f"- Terminated early: `{_scalar(summary.get('terminated_early'))}`",
        f"- Termination reason: `{_scalar(summary.get('termination_reason'))}`",
        f"- Total delta-v: `{total_dv:.4g} m/s`",
        f"- Plots: `{len(dict(summary.get('plot_outputs', {}) or {}))}`",
        f"- Animations: `{len(dict(summary.get('animation_outputs', {}) or {}))}`",
    ]


def _monte_carlo_metrics(payload: dict[str, Any]) -> list[str]:
    aggregate = dict(payload.get("aggregate_stats", {}) or {})
    commander = dict(payload.get("commander_brief", {}) or {})
    runs = list(payload.get("runs", []) or [])
    return [
        f"- Iterations: `{len(runs) if runs else _scalar(dict(payload.get('monte_carlo', {}) or {}).get('iterations'))}`",
        f"- Pass rate: `{_scalar(aggregate.get('pass_rate', commander.get('p_success')))}`",
        f"- Closest approach mean: `{_scalar(aggregate.get('closest_approach_km_mean'))} km`",
        f"- Keepout violation probability: `{_scalar(aggregate.get('p_keepout_violation', commander.get('p_keepout_violation')))}`",
        f"- Total delta-v mean: `{_scalar(aggregate.get('total_dv_m_s_mean'))} m/s`",
    ]


def _sensitivity_metrics(payload: dict[str, Any]) -> list[str]:
    analysis = dict(payload.get("analysis", {}) or {})
    rankings = list(payload.get("parameter_rankings", []) or [])
    top_driver = "not available"
    if rankings:
        top_driver = str(dict(rankings[0] or {}).get("parameter_path", "not available"))
    return [
        f"- Method: `{_scalar(analysis.get('method'))}`",
        f"- Total runs: `{_scalar(analysis.get('run_count'))}`",
        f"- Successful runs: `{_scalar(analysis.get('successful_run_count'))}`",
        f"- Failed runs: `{_scalar(analysis.get('failed_run_count'))}`",
        f"- Top ranked parameter: `{top_driver}`",
    ]


def write_output_index(
    *,
    outdir: Path,
    workflow: str,
    title: str,
    summary: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    next_steps: list[str] | None = None,
) -> Path:
    """Write a human-readable index for an output directory."""
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = outdir / "index.md"
    summary = dict(summary or {})
    payload = dict(payload or {})
    artifacts = dict(artifacts or {})
    scenario_name = summary.get("scenario_name", payload.get("scenario_name", title))
    scenario_description = str(summary.get("scenario_description", payload.get("scenario_description", "")) or "").strip()

    if workflow == "single_run":
        key_metrics = _single_run_metrics(summary)
    elif workflow == "monte_carlo":
        key_metrics = _monte_carlo_metrics(payload)
    elif workflow == "sensitivity":
        key_metrics = _sensitivity_metrics(payload)
    else:
        key_metrics = ["- No workflow-specific metrics are available yet."]

    steps = list(next_steps or _default_next_steps(workflow=workflow, artifacts=artifacts))
    lines = [
        "# Output Index",
        "",
        f"- Workflow: `{workflow}`",
        f"- Scenario: `{_scalar(scenario_name)}`",
    ]
    if scenario_description:
        lines.append(f"- Description: {scenario_description}")
    lines.extend(
        [
            f"- Output directory: `{outdir}`",
            f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
            "",
            "## Key Results",
            *key_metrics,
            "",
            "## Open First",
            *[f"{idx}. {step}" for idx, step in enumerate(steps, start=1)],
            "",
            "## Artifact Inventory",
            *_artifact_lines(artifacts, base_dir=outdir),
            "",
        ]
    )
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path
