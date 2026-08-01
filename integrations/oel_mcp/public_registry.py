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
from integrations.oel_mcp.execution import M4_RESOURCE_PROFILES

PUBLIC_PROFILES = ("public_local", "direct_frontier_restricted")
M4_LOCAL_PROFILES = ("public_local",)
M3_PUBLIC_TOOL_IDS = (
    "oel.describe_capabilities.v1",
    "oel.inspect_run.v1",
    "oel.query_review.v1",
)

APPROVAL_SCHEMA: dict[str, Any] = object_schema(
    {
        "approval_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "scope": {"type": "string", "enum": ["trust", "write", "execute"]},
    },
    required=("approval_id", "scope"),
)

SCENARIO_INPUT_PROPERTIES: dict[str, Any] = {
    "config_path": {"type": "string", "minLength": 1},
    "output_dir": {"type": "string", "minLength": 1},
    "resource_profile": {"type": "string", "enum": list(M4_RESOURCE_PROFILES), "default": "laptop-safe"},
}

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
        "execution_provenance": {"type": "object"},
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
        "execution_provenance",
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

PLAN_RUN_RESULT_SCHEMA = object_schema(
    {
        "scenario": {"type": "object"},
        "identity": {"type": "object"},
        "safe_validation": {"type": "object"},
        "resource_estimate": {"type": "object"},
        "phases": {"type": "array", "items": {"type": "string"}},
        "approval": {"type": "object"},
        "execution_authorized": {"const": False},
    },
    required=(
        "scenario",
        "identity",
        "safe_validation",
        "resource_estimate",
        "phases",
        "approval",
        "execution_authorized",
    ),
)

VALIDATE_SCENARIO_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["validated", "safe_only", "failed"]},
        "scenario": {"type": "object"},
        "identity": {"type": "object"},
        "safe_validation": {"type": "object"},
        "trusted_validation": {"type": "object"},
        "resource_estimate": {"type": "object"},
        "execution_ready": {"type": "boolean"},
        "execution_authorized": {"const": False},
        "next_step": {"type": "string"},
    },
    required=(
        "status",
        "scenario",
        "identity",
        "safe_validation",
        "trusted_validation",
        "resource_estimate",
        "execution_ready",
        "execution_authorized",
        "next_step",
    ),
)

RUN_SCENARIO_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial", "cancelled"]},
        "scenario": {"type": "object"},
        "identity": {"type": "object"},
        "output_dir": {"type": "string"},
        "run": {"type": "object"},
        "artifacts": {"type": "array", "items": {"type": "string"}},
        "manifest_path": {"type": "string"},
    },
    required=("status", "scenario", "identity", "output_dir", "run", "artifacts", "manifest_path"),
)

COMPARE_RUNS_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial"]},
        "base_output_dir": {"type": "string"},
        "candidate_output_dir": {"type": "string"},
        "metric_names": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "object"},
        "deltas": {"type": "object"},
        "metric_status": {"type": "array", "items": {"type": "object"}},
        "summary": {"type": "object"},
    },
    required=(
        "status",
        "base_output_dir",
        "candidate_output_dir",
        "metric_names",
        "metrics",
        "deltas",
        "metric_status",
        "summary",
    ),
)

PLOT_EVIDENCE_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial"]},
        "output_dir": {"type": "string"},
        "recipe_id": {"type": "string"},
        "artifact": {"type": "object"},
        "manifest_path": {"type": "string"},
    },
    required=("status", "output_dir", "recipe_id", "artifact", "manifest_path"),
)

RUN_AGENT_TASK_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial", "failed", "cancelled"]},
        "recipe_id": {"type": "string"},
        "recipe_maturity": {"const": "supported"},
        "output_dir": {"type": "string"},
        "evidence_summary": {"type": "object"},
        "packet_path": {"type": "string"},
        "manifest_path": {"type": "string"},
        "artifacts": {"type": "array", "items": {"type": "object"}},
        "failure_hints": {"type": "array", "items": {"type": "object"}},
    },
    required=(
        "status",
        "recipe_id",
        "recipe_maturity",
        "output_dir",
        "evidence_summary",
        "packet_path",
        "manifest_path",
        "artifacts",
        "failure_hints",
    ),
)

PREPARE_REPORT_PACKET_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial"]},
        "packet_id": {"type": "string"},
        "source_output_dir": {"type": "string"},
        "packet_output_dir": {"type": "string"},
        "packet_path": {"type": "string"},
        "brief_path": {"type": "string"},
        "manifest_path": {"type": "string"},
        "packet_sha256": {"type": "string"},
        "artifact_count": {"type": "integer", "minimum": 0, "maximum": 100},
        "evidence_summary": {"type": "object"},
        "provider_call_made": {"const": False},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "packet_id",
        "source_output_dir",
        "packet_output_dir",
        "packet_path",
        "brief_path",
        "manifest_path",
        "packet_sha256",
        "artifact_count",
        "evidence_summary",
        "provider_call_made",
        "non_claims",
    ),
)

AUDIT_REPORT_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["passed", "needs_review"]},
        "packet_id": {"type": "string"},
        "report_path": {"type": "string"},
        "packet_path": {"type": "string"},
        "audit_output_dir": {"type": "string"},
        "audit_json_path": {"type": "string"},
        "audit_markdown_path": {"type": "string"},
        "manifest_path": {"type": "string"},
        "checks": {"type": "object"},
        "unknown_evidence_references": {"type": "array", "items": {"type": "string"}},
        "unavailable_evidence_references": {"type": "array", "items": {"type": "string"}},
        "missing_required_sections": {"type": "array", "items": {"type": "string"}},
        "provider_call_made": {"const": False},
        "semantic_claim_review_performed": {"const": False},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "packet_id",
        "report_path",
        "packet_path",
        "audit_output_dir",
        "audit_json_path",
        "audit_markdown_path",
        "manifest_path",
        "checks",
        "unknown_evidence_references",
        "unavailable_evidence_references",
        "missing_required_sections",
        "provider_call_made",
        "semantic_claim_review_performed",
        "non_claims",
    ),
)

PUBLIC_TOOL_CONTRACTS: tuple[ToolContract, ...] = (
    ToolContract(
        tool_id="oel.describe_capabilities.v1",
        title="Describe OEL MCP capabilities",
        description="List the OEL MCP tools available in the active deployment profile, including maturity and non-claims.",
        risk_class="R0_read",
        oel_api="integration-owned registry",
        maturity="supported",
        install_profile="mcp",
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
        maturity="supported",
        install_profile="mcp",
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
        maturity="supported",
        install_profile="mcp",
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
    ToolContract(
        tool_id="oel.plan_run.v1",
        title="Plan a bounded OEL scenario run",
        description="Inspect a trusted local scenario structurally and report its normalized identity, phases, and resource estimate without executing it.",
        risk_class="R0_read",
        oel_api="sim.api.SimulationWorkspace.validate_safe; sim.resource_limits.estimate_resource_requirements",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("scenario_config", "resource_estimate"),
        writes=False,
        input_schema=object_schema(
            handling_properties(dict(SCENARIO_INPUT_PROPERTIES)), required=("config_path", "output_dir", "handling")
        ),
        result_schema=PLAN_RUN_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.validate_scenario.v1",
        title="Validate an OEL scenario for a bounded run",
        description="Run safe validation first and optionally trusted plugin validation, returning a content-bound validation id without authorizing execution.",
        risk_class="R0_read",
        oel_api="sim.api.SimulationWorkspace.validate",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("scenario_config", "validation_evidence"),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    **SCENARIO_INPUT_PROPERTIES,
                    "trust_plugins": {"type": "boolean", "default": False},
                    "trust_approval": APPROVAL_SCHEMA,
                }
            ),
            required=("config_path", "output_dir", "trust_plugins", "handling"),
        ),
        result_schema=VALIDATE_SCENARIO_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.run_scenario.v1",
        title="Run one validated deterministic OEL scenario",
        description="Execute a content-bound, trusted, resource-preflighted public scenario into a new approved output directory.",
        risk_class="R2_execute",
        oel_api="sim.execution.run_simulation_config_file",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("scenario_config", "run_evidence"),
        writes=True,
        executes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    **SCENARIO_INPUT_PROPERTIES,
                    "validation_id": {"type": "string", "minLength": 1},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("config_path", "output_dir", "resource_profile", "validation_id", "approval", "handling"),
        ),
        result_schema=RUN_SCENARIO_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.compare_runs.v1",
        title="Compare two completed OEL runs",
        description="Compare allowlisted semantic metrics from two existing review stores without rerunning either scenario.",
        risk_class="R0_read",
        oel_api="sim.agent_task.runner.compare_outputs",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence",),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "base_output_dir": {"type": "string", "minLength": 1},
                    "candidate_output_dir": {"type": "string", "minLength": 1},
                    "metric_names": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                }
            ),
            required=("base_output_dir", "candidate_output_dir", "metric_names", "handling"),
        ),
        result_schema=COMPARE_RUNS_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.plot_evidence.v1",
        title="Plot completed OEL review evidence",
        description="Generate one allowlisted plot recipe inside an approved completed-run output directory.",
        risk_class="R1_write",
        oel_api="sim.agent_task.runner.create_plot",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "plot_artifact"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "output_dir": {"type": "string", "minLength": 1},
                    "recipe_id": {"type": "string", "minLength": 1},
                    "style": {"type": "string", "enum": ["oel_dark", "oel_light"], "default": "oel_dark"},
                    "format": {"type": "string", "enum": ["png", "svg", "pdf"], "default": "png"},
                    "artifact_id": {"type": "string", "minLength": 1, "maxLength": 80},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("output_dir", "recipe_id", "artifact_id", "approval", "handling"),
        ),
        result_schema=PLOT_EVIDENCE_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.run_agent_task.v1",
        title="Run one supported public OEL agent task",
        description="Execute one checked-in supported public task recipe with bounded review rows and optional allowlisted plots.",
        risk_class="R2_execute",
        oel_api="sim.agent_task.runner.run_recipe",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("scenario_config", "run_evidence", "task_evidence_packet"),
        writes=True,
        executes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "recipe_id": {"type": "string", "minLength": 1},
                    "output_dir": {"type": "string", "minLength": 1},
                    "resource_profile": {
                        "type": "string",
                        "enum": list(M4_RESOURCE_PROFILES),
                        "default": "laptop-safe",
                    },
                    "make_plots": {"type": "boolean", "default": False},
                    "style": {"type": "string", "enum": ["oel_dark", "oel_light"], "default": "oel_dark"},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("recipe_id", "output_dir", "resource_profile", "approval", "handling"),
        ),
        result_schema=RUN_AGENT_TASK_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.prepare_report_packet.v1",
        title="Prepare a provider-neutral OEL report evidence packet",
        description="Write a bounded, hashed evidence packet and authoring brief from one completed local OEL run without calling a model provider.",
        risk_class="R1_write",
        oel_api="sim.agent_task.runner.inspect_output; integration-owned packet writer",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "report_evidence_packet"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "source_output_dir": {"type": "string", "minLength": 1},
                    "packet_output_dir": {"type": "string", "minLength": 1},
                    "packet_id": {"type": "string", "minLength": 1, "maxLength": 80},
                    "query_names": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("source_output_dir", "packet_output_dir", "packet_id", "approval", "handling"),
        ),
        result_schema=PREPARE_REPORT_PACKET_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.audit_report.v1",
        title="Audit an agent-authored report against an OEL evidence packet",
        description="Verify packet and artifact hashes, report structure, and explicit evidence references without calling a model or judging narrative claims.",
        risk_class="R1_write",
        oel_api="integration-owned deterministic report audit",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("report_evidence_packet", "agent_authored_report", "report_audit"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "report_path": {"type": "string", "minLength": 1},
                    "packet_path": {"type": "string", "minLength": 1},
                    "audit_output_dir": {"type": "string", "minLength": 1},
                    "author": {"type": "string", "minLength": 1, "maxLength": 120},
                    "model": {"type": "string", "maxLength": 200, "default": ""},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("report_path", "packet_path", "audit_output_dir", "author", "approval", "handling"),
        ),
        result_schema=AUDIT_REPORT_RESULT_SCHEMA,
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
    "M3_PUBLIC_TOOL_IDS",
    "PUBLIC_TOOL_CONTRACTS",
    "public_contract_map",
    "public_contracts_for_profile",
    "public_tool_definitions",
]
