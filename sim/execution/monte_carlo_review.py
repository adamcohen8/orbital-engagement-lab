# ruff: noqa: F401,F403,F405,I001
from .campaign_common import *

def _monte_carlo_review_run_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        summary = dict(run.get("summary", {}) or {})
        assessment = dict(run.get("assessment", {}) or {})
        artifacts = dict(summary.get("artifacts", {}) or {})
        rows.append(
            {
                "iteration": run.get("iteration"),
                "passed": assessment.get("passed", summary.get("passed")),
                "closest_approach_km": run.get("closest_approach_km", summary.get("closest_approach_km")),
                "duration_s": summary.get("duration_s"),
                "total_dv_m_s": summary.get("total_dv_m_s"),
                "output_dir": summary.get("output_dir", artifacts.get("output_dir")),
                "sampled_parameters_json": json.dumps(dict(run.get("sampled_parameters", {}) or {}), sort_keys=True),
            }
        )
    return rows


def _monte_carlo_review_metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        iteration = run.get("iteration")
        summary = dict(run.get("summary", {}) or {})
        metrics = dict(summary.get("metrics", {}) or {})
        for key in ("closest_approach_km", "duration_s", "total_dv_m_s"):
            if key in run:
                metrics.setdefault(key, run.get(key))
            elif key in summary:
                metrics.setdefault(key, summary.get(key))
        for name, value in sorted(metrics.items()):
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True)
            rows.append({"iteration": iteration, "metric_name": name, "metric_value": value})
    return rows

__all__ = [name for name in globals() if not name.startswith("__")]
