from __future__ import annotations

from pathlib import Path
from typing import Any

from sim.agent_task.models import FailureHint


def diagnose_failure(
    message: str = "",
    *,
    validation: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> list[FailureHint]:
    """Return actionable, public-safe hints for common agent workflow failures."""

    text = " ".join([str(message or ""), " ".join(str(item) for item in (validation or {}).get("errors", []) or [])])
    lower = text.lower()
    hints: list[FailureHint] = []

    if "plugin validation failed" in lower or "import" in lower and "plugin" in lower:
        hints.append(
            FailureHint(
                code="plugin_validation_failed",
                severity="error",
                message="The scenario references a plugin, module, or class that did not pass validation.",
                next_step="Inspect the config plugin pointers and run the normal validate-only command before execution.",
            )
        )
    if "review store not found" in lower or "review/run.sqlite" in lower:
        hints.append(
            FailureHint(
                code="review_store_missing",
                severity="warning",
                message="No review SQLite store was found for this output directory.",
                next_step="Enable outputs.review.enabled in the scenario and rerun, or fall back to index.md and summary JSON.",
            )
        )
    if "no such table" in lower or "no such column" in lower:
        hints.append(
            FailureHint(
                code="review_schema_mismatch",
                severity="warning",
                message="The selected review query does not match the tables or columns in this run.",
                next_step="Inspect review/schema.json or run a schema discovery query before using custom SQL.",
            )
        )
    if "yaml" in lower or "mapping" in lower or "parser" in lower:
        hints.append(
            FailureHint(
                code="config_yaml_invalid",
                severity="error",
                message="The scenario YAML could not be loaded or normalized.",
                next_step="Fix YAML structure first, then rerun validate-only before executing the scenario.",
            )
        )
    if "duration" in lower or "dt_s" in lower or "timestep" in lower:
        hints.append(
            FailureHint(
                code="timing_config_invalid",
                severity="error",
                message="The simulator duration or timestep settings look invalid.",
                next_step="Check simulator.duration_s and simulator.dt_s and keep the first repair loop minimal.",
            )
        )
    if output_dir is not None:
        review_db = Path(output_dir) / "review" / "run.sqlite"
        if not review_db.is_file():
            hints.append(
                FailureHint(
                    code="review_db_absent_after_run",
                    severity="warning",
                    message="The output directory does not contain review/run.sqlite.",
                    next_step="Confirm outputs.review.enabled is true and that the run completed artifact writing.",
                )
            )
    if not hints and text.strip():
        hints.append(
            FailureHint(
                code="agent_task_failed",
                severity="error",
                message="The agent task failed before producing complete evidence.",
                next_step="Use the validation report and generated artifacts to choose the smallest deterministic repair.",
            )
        )
    return hints
