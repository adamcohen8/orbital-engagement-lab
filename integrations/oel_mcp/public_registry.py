from __future__ import annotations

from typing import Any

from integrations.oel_mcp.contracts import (
    HANDLING_SCHEMA,
    MAX_ROWS,
    MAX_VM_STEPS,
    ToolContract,
    handling_properties,
    object_schema,
)

PUBLIC_PROFILES = ("public_local", "direct_frontier_restricted")

DESCRIBE_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"const": "available"},
        "integration": {"type": "string"},
        "transport": {"const": "stdio"},
        "deployment_profile": {"type": "string"},
        "capabilities": {"type": "array", "items": {"type": "object"}},
        "dependency_direction": {"const": "mcp_consumes_oel"},
        "compatibility": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "integration",
        "transport",
        "deployment_profile",
        "capabilities",
        "dependency_direction",
        "compatibility",
        "non_claims",
    ),
)

INSPECT_RUN_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "output_dir": {"type": "string"},
        "status": {"enum": ["completed", "partial"]},
        "evidence_summary": {"type": "object"},
        "review": {"type": "object"},
        "artifact_summary": {"type": "object"},
        "failure_hints": {"type": "array", "items": {"type": "object"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "freshness": {"type": "object"},
    },
    required=(
        "output_dir",
        "status",
        "evidence_summary",
        "review",
        "artifact_summary",
        "failure_hints",
        "caveats",
        "freshness",
    ),
)

QUERY_REVIEW_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "output_dir": {"type": "string"},
        "review_store": {"const": "review/run.sqlite"},
        "sql": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "object"}},
        "row_count": {"type": "integer", "minimum": 0, "maximum": MAX_ROWS},
        "empty_result": {"type": "boolean"},
        "empty_result_semantics": {"type": "string"},
        "truncated": {"type": "boolean"},
        "units_semantics": {"type": "string"},
    },
    required=(
        "output_dir",
        "review_store",
        "sql",
        "columns",
        "rows",
        "row_count",
        "empty_result",
        "empty_result_semantics",
        "truncated",
        "units_semantics",
    ),
)

PUBLIC_TOOL_CONTRACTS: tuple[ToolContract, ...] = (
    ToolContract(
        tool_id="oel.describe_capabilities.v1",
        title="Describe OEL MCP capabilities",
        description="List the OEL MCP tools available in the active deployment profile, including maturity and non-claims.",
        risk_class="R0_read",
        oel_api="integration-owned registry",
        maturity="prototype",
        install_profile="core",
        deployment_profiles=PUBLIC_PROFILES,
        data_classes=("capability_metadata",),
        writes=False,
        input_schema=object_schema({}, required=()),
        result_schema=DESCRIBE_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.inspect_run.v1",
        title="Inspect completed OEL run",
        description="Inspect an existing OEL output directory without writing a new evidence packet.",
        risk_class="R0_read",
        oel_api="sim.agent_task.runner.inspect_output",
        maturity="prototype",
        install_profile="core",
        deployment_profiles=PUBLIC_PROFILES,
        data_classes=("run_evidence",),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "output_dir": {"type": "string", "minLength": 1},
                    "query_names": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS, "default": 50},
                }
            ),
            required=("output_dir", "handling"),
        ),
        result_schema=INSPECT_RUN_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.query_review.v1",
        title="Query OEL review evidence",
        description="Run one bounded read-only SELECT or WITH query against an existing OEL review store.",
        risk_class="R0_read",
        oel_api="sim.review.ReviewWorkspace.query",
        maturity="prototype",
        install_profile="core",
        deployment_profiles=PUBLIC_PROFILES,
        data_classes=("run_evidence",),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "output_dir": {"type": "string", "minLength": 1},
                    "sql": {"type": "string", "minLength": 1, "maxLength": 100_000},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS, "default": 100},
                    "max_vm_steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_VM_STEPS,
                        "default": 250_000,
                    },
                }
            ),
            required=("output_dir", "sql", "handling"),
        ),
        result_schema=QUERY_REVIEW_RESULT_SCHEMA,
    ),
)


def public_contracts_for_profile(profile: str) -> tuple[ToolContract, ...]:
    return tuple(contract for contract in PUBLIC_TOOL_CONTRACTS if profile in contract.deployment_profiles)


def public_contract_map(profile: str) -> dict[str, ToolContract]:
    return {contract.tool_id: contract for contract in public_contracts_for_profile(profile)}


def public_tool_definitions(profile: str) -> list[dict[str, Any]]:
    return [contract.mcp_definition() for contract in public_contracts_for_profile(profile)]


__all__ = [
    "HANDLING_SCHEMA",
    "PUBLIC_TOOL_CONTRACTS",
    "public_contract_map",
    "public_contracts_for_profile",
    "public_tool_definitions",
]
