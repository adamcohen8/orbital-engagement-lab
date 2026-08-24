# ruff: noqa: E402 -- automation cache policy must run before plotting imports.
"""Agent-oriented OEL workflow recipes and evidence packets."""

from sim.runtime_environment import configure_headless_runtime, configure_runtime_caches

configure_runtime_caches()
configure_headless_runtime()

from sim.agent_task.models import (
    AGENT_EVIDENCE_PACKET_VERSION,
    AgentPlotRecipe,
    AgentTaskRecipe,
    EvidencePacket,
    FailureHint,
    SemanticMetric,
)
from sim.agent_task.plot_recipes import get_plot_recipe, list_plot_recipes
from sim.agent_task.recipes import get_recipe, list_recipes
from sim.agent_task.runner import compare_configs, compare_outputs, create_plot, inspect_output, run_recipe
from sim.agent_task.semantics import get_semantic_metric, list_semantic_metrics, semantic_metric_request_rows

__all__ = [
    "AGENT_EVIDENCE_PACKET_VERSION",
    "AgentPlotRecipe",
    "AgentTaskRecipe",
    "EvidencePacket",
    "FailureHint",
    "SemanticMetric",
    "compare_configs",
    "compare_outputs",
    "create_plot",
    "get_plot_recipe",
    "get_recipe",
    "get_semantic_metric",
    "inspect_output",
    "list_plot_recipes",
    "list_recipes",
    "list_semantic_metrics",
    "run_recipe",
    "semantic_metric_request_rows",
]
