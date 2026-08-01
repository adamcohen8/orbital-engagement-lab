from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any, Iterable

from integrations.oel_mcp.contracts import DEPLOYMENT_PROFILES, ToolContract

RESOURCE_SCHEMA_VERSION = 1
MAX_RESOURCE_BYTES = 500_000
PUBLIC_RESOURCE_URIS = (
    "oel://capabilities/tools/v1",
    "oel://review/saved-queries/v1",
    "oel://agent/tasks/v1",
    "oel://docs/operator-guide/v1",
)
PUBLIC_SAVED_QUERY_NAMES = frozenset(
    {
        "artifacts",
        "attitude_rates_first_last",
        "attitude_state_first_last",
        "burn_activity",
        "burn_events",
        "ground_access_no_access_reasons",
        "ground_access_summary",
        "mission_recovery_burns",
        "mission_recovery_candidates",
        "mission_recovery_elements",
        "mission_recovery_summary",
        "objects",
        "passive_final_state",
        "relative_final_state",
        "rendezvous_closest_approach",
        "rendezvous_metrics",
        "run_metadata",
    }
)


@dataclass(frozen=True)
class ResourceContract:
    uri: str
    name: str
    title: str
    description: str
    mime_type: str
    source: str
    maturity: str = "supported"


@dataclass(frozen=True)
class PublishedResource:
    contract: ResourceContract
    text: str

    @property
    def size(self) -> int:
        return len(self.text.encode("utf-8"))


PUBLIC_RESOURCE_CONTRACTS = (
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[0],
        name="oel-public-tool-schemas-v1",
        title="OEL public MCP tool schemas",
        description="Versioned schemas and effect annotations for the public OEL MCP tool registry.",
        mime_type="application/json",
        source="integrations.oel_mcp.public_registry",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[1],
        name="oel-public-saved-review-queries-v1",
        title="OEL public saved review queries",
        description="Bounded metadata for supported public read-only review queries.",
        mime_type="application/json",
        source="sim.review.queries",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[2],
        name="oel-public-agent-tasks-v1",
        title="OEL public agent task definitions",
        description="Public agent task definitions suitable for deterministic OEL workflow routing.",
        mime_type="application/json",
        source="sim.agent_task.recipes",
    ),
    ResourceContract(
        uri=PUBLIC_RESOURCE_URIS[3],
        name="oel-mcp-operator-guide-v1",
        title="OEL MCP operator guide",
        description="Packaged guidance for the supported local stdio MCP surface.",
        mime_type="text/markdown",
        source="integrations.oel_mcp.resource_data/operator-guide.md",
    ),
)


def build_public_resource_catalog(
    *,
    profile: str,
    tool_contracts: Iterable[ToolContract],
) -> tuple[PublishedResource, ...]:
    if profile not in DEPLOYMENT_PROFILES:
        raise ValueError("Unknown OEL MCP deployment profile.")
    public_tools = tuple(contract for contract in tool_contracts if not contract.tool_id.startswith("oel.pro."))
    loaders = {
        PUBLIC_RESOURCE_URIS[0]: lambda: _json_text(_tool_schema_payload(profile, public_tools)),
        PUBLIC_RESOURCE_URIS[1]: lambda: _json_text(_saved_query_payload()),
        PUBLIC_RESOURCE_URIS[2]: lambda: _json_text(_agent_task_payload()),
        PUBLIC_RESOURCE_URIS[3]: _operator_guide_text,
    }
    published: list[PublishedResource] = []
    for contract in PUBLIC_RESOURCE_CONTRACTS:
        text = loaders[contract.uri]()
        size = len(text.encode("utf-8"))
        if size > MAX_RESOURCE_BYTES:
            raise ValueError(f"Packaged MCP resource exceeds the {MAX_RESOURCE_BYTES}-byte limit: {contract.uri}")
        published.append(PublishedResource(contract=contract, text=text))
    return tuple(published)


def public_resource_map(resources: Iterable[PublishedResource]) -> dict[str, PublishedResource]:
    return {resource.contract.uri: resource for resource in resources}


def _tool_schema_payload(profile: str, contracts: tuple[ToolContract, ...]) -> dict[str, Any]:
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[0],
        "deployment_profile": profile,
        "source": "integrations.oel_mcp.public_registry",
        "tools": [contract.mcp_definition() for contract in contracts],
        "non_claims": [
            "Resource discovery is not execution authority.",
            "Only the listed deployment registry is available through this resource.",
        ],
    }


def _saved_query_payload() -> dict[str, Any]:
    from sim.review.queries import list_saved_review_queries

    queries = [
        {
            "name": query.name,
            "description": query.description,
            "sql": query.sql,
            "source_tables": list(query.source_tables),
            "maturity": query.maturity,
            "allow_empty": query.allow_empty,
        }
        for query in list_saved_review_queries()
        if query.name in PUBLIC_SAVED_QUERY_NAMES
    ]
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[1],
        "source": "sim.review.queries",
        "queries": queries,
        "query_count": len(queries),
        "rules": {
            "read_only_prefixes": ["SELECT", "WITH"],
            "units_are_never_inferred": True,
            "review_database_mutation_allowed": False,
        },
    }


def _agent_task_payload() -> dict[str, Any]:
    from sim.agent_task.recipes import list_recipes

    recipes = [asdict(recipe) for recipe in list_recipes() if "public" in recipe.tags]
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "resource_uri": PUBLIC_RESOURCE_URIS[2],
        "source": "sim.agent_task.recipes",
        "tasks": recipes,
        "task_count": len(recipes),
        "non_claims": [
            "Task resources describe documented workflows; reading them does not run a task.",
            "Examples and recipes do not replace scenario validation or deterministic OEL evidence.",
        ],
    }


def _operator_guide_text() -> str:
    resource = files("integrations.oel_mcp").joinpath("resource_data/operator-guide.md")
    return resource.read_text(encoding="utf-8")


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


__all__ = [
    "MAX_RESOURCE_BYTES",
    "PUBLIC_RESOURCE_CONTRACTS",
    "PUBLIC_RESOURCE_URIS",
    "PublishedResource",
    "ResourceContract",
    "build_public_resource_catalog",
    "public_resource_map",
]
