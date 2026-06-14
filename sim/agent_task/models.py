from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

AGENT_EVIDENCE_PACKET_VERSION = 1


@dataclass(frozen=True)
class AgentTaskRecipe:
    """Machine-readable workflow recipe for an agent-safe OEL task."""

    recipe_id: str
    title: str
    description: str
    config_path: str
    query_names: tuple[str, ...] = ()
    plot_recipe_ids: tuple[str, ...] = ()
    semantic_metric_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

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
    semantic_metric_names: tuple[str, ...] = ()
    supported_tables: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    recipe: dict[str, Any] | None = None
    configs: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    semantic_metrics: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    plots: list[dict[str, Any]] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)
    failure_hints: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
