from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

AGENT_EVIDENCE_PACKET_VERSION = 1
AGENT_TASK_MATURITY_LEVELS = frozenset({"supported", "prototype", "experimental"})


@dataclass(frozen=True)
class AgentTaskRecipe:
    """Machine-readable workflow recipe for an agent-safe OEL task."""

    recipe_id: str
    title: str
    description: str
    config_path: str
    maturity: str = "supported"
    workflow: str = "scenario_run"
    query_names: tuple[str, ...] = ()
    plot_recipe_ids: tuple[str, ...] = ()
    plots_generated_by_default: bool = False
    plot_cli_option: str = "--plot"
    semantic_metric_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.maturity not in AGENT_TASK_MATURITY_LEVELS:
            allowed = ", ".join(sorted(AGENT_TASK_MATURITY_LEVELS))
            raise ValueError(f"Unknown agent task recipe maturity {self.maturity!r}; expected one of: {allowed}")
        if self.plot_recipe_ids and not self.plot_cli_option:
            raise ValueError("Recipes with plot_recipe_ids must name the CLI option that renders them.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticMetric:
    """Review-store metric definition agents can cite in answers."""

    name: str
    description: str
    units: str = ""
    table: str = ""
    saved_query: str = ""
    sql: str = ""
    interpretation: str = ""
    caveats: tuple[str, ...] = ()
    maturity: str = "supported"
    source_tables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.maturity not in AGENT_TASK_MATURITY_LEVELS:
            allowed = ", ".join(sorted(AGENT_TASK_MATURITY_LEVELS))
            raise ValueError(f"Unknown semantic metric maturity {self.maturity!r}; expected one of: {allowed}")
        if self.sql and not self.sql.lstrip().upper().startswith(("SELECT", "WITH")):
            raise ValueError(f"Semantic metric {self.name!r} must use read-only SELECT/WITH SQL.")
        if not (self.table or self.saved_query or self.sql):
            raise ValueError(f"Semantic metric {self.name!r} must declare table, saved_query, or SQL evidence.")
        if not self.source_tables:
            source_tables = _dedupe_tables(
                tuple(item for item in (self.table,) if item) + _infer_source_tables(self.sql)
            )
            if not source_tables:
                raise ValueError(f"Semantic metric {self.name!r} must declare source_tables.")
            object.__setattr__(self, "source_tables", source_tables)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentPlotRecipe:
    """Named plot recipe over a review store."""

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
    maturity: str = "supported"
    semantic_metric_names: tuple[str, ...] = ()
    supported_tables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.maturity not in AGENT_TASK_MATURITY_LEVELS:
            allowed = ", ".join(sorted(AGENT_TASK_MATURITY_LEVELS))
            raise ValueError(f"Unknown agent plot recipe maturity {self.maturity!r}; expected one of: {allowed}")
        if not self.sql.lstrip().upper().startswith(("SELECT", "WITH")):
            raise ValueError(f"Agent plot recipe {self.recipe_id!r} must use read-only SELECT/WITH SQL.")
        if not self.supported_tables:
            raise ValueError(f"Agent plot recipe {self.recipe_id!r} must declare supported_tables.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_source_tables(sql: str) -> tuple[str, ...]:
    if not sql:
        return ()
    cte_names = {
        match.group(1)
        for match in re.finditer(r"(?:\bWITH|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", sql, flags=re.IGNORECASE)
    }
    names = [
        match.group(1)
        for match in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\b", sql, flags=re.IGNORECASE)
        if match.group(1).lower() not in {"select"} and match.group(1) not in cte_names
    ]
    return _dedupe_tables(tuple(names))


def _dedupe_tables(names: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for name in names:
        if name and name not in out:
            out.append(name)
    return tuple(out)


@dataclass(frozen=True)
class FailureHint:
    code: str
    severity: str
    message: str
    next_step: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class EvidencePacket:
    """Portable answer-support packet produced by sim.agent_task."""

    task_id: str
    status: str
    generated_utc: str
    task_type: str = "agent_task"
    schema_version: int = AGENT_EVIDENCE_PACKET_VERSION
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    recipe: dict[str, Any] | None = None
    configs: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    semantic_metric_requests: list[dict[str, Any]] = field(default_factory=list)
    semantic_metrics: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    artifact_summary: dict[str, Any] = field(default_factory=dict)
    plots: list[dict[str, Any]] = field(default_factory=list)
    plot_summary: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    failure_hints: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
