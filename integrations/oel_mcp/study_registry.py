"""Restricted local MCP contracts for BYO frontier study planning."""

from __future__ import annotations

from copy import deepcopy

from integrations.oel_mcp.contracts import ToolContract, handling_properties, object_schema
from integrations.oel_mcp.public_registry import public_contract_map
from sim.hosted_client.package_contract import HOSTED_EXECUTION_PACKAGE_VALIDATION_SCHEMA_ID
from sim.study_planning.schemas import (
    CAPABILITY_REQUEST_SCHEMA,
    PLANNING_RESULT_SCHEMA,
    STUDY_CAPABILITY_MANIFEST_SCHEMA,
    STUDY_PLAN_REVIEW_SCHEMA,
    STUDY_PLAN_SCHEMA,
    STUDY_REQUEST_SCHEMA,
)

PROFILE = "direct_frontier_restricted"


def _proposal_schema(schema: dict, *generated_fields: str) -> dict:
    value = deepcopy(schema)
    value["required"] = [item for item in value.get("required", []) if item not in generated_fields]
    return value


REQUEST_PROPOSAL_SCHEMA = _proposal_schema(STUDY_REQUEST_SCHEMA, "schema", "request_id", "request_sha256")
PLAN_PROPOSAL_SCHEMA = _proposal_schema(
    STUDY_PLAN_SCHEMA,
    "schema",
    "plan_id",
    "plan_sha256",
    "request_id",
    "request_sha256",
    "capability_manifest_sha256",
    "planner_version",
)
NULLABLE_PLAN_PROPOSAL_SCHEMA = deepcopy(PLAN_PROPOSAL_SCHEMA)
NULLABLE_PLAN_PROPOSAL_SCHEMA["type"] = ["object", "null"]

CONFIG_INPUT_SCHEMA = object_schema(
    {
        "config_ref": {"type": "string", "minLength": 1, "maxLength": 160},
        "path": {"type": "string", "minLength": 1, "maxLength": 2000},
    },
    required=("config_ref", "path"),
)

CAPABILITY_MANIFEST_RESULT_SCHEMA = deepcopy(STUDY_CAPABILITY_MANIFEST_SCHEMA)
CAPABILITY_MANIFEST_RESULT_SCHEMA["properties"]["catalog_scope"] = {
    "const": "public_agent_discovery"
}


def study_contracts() -> dict[str, ToolContract]:
    describe = public_contract_map(PROFILE)["oel.describe_capabilities.v1"]
    contracts = (
        describe,
        ToolContract(
            tool_id="oel.study.capabilities.v1",
            title="Describe OEL study capabilities",
            description=(
                "Return the explicit public and Pro operation vocabulary available to a BYO frontier planner. "
                "This reveals contracts and limitations, not private implementation source."
            ),
            risk_class="R0_read",
            oel_api="sim.study_planning.discovery_capability_catalog",
            maturity="prototype",
            install_profile="public",
            deployment_profiles=(PROFILE,),
            data_classes=("capability_metadata",),
            writes=False,
            input_schema=object_schema(handling_properties({}), required=("handling",)),
            result_schema=CAPABILITY_MANIFEST_RESULT_SCHEMA,
        ),
        ToolContract(
            tool_id="oel.study.preflight.v1",
            title="Preflight a proposed OEL StudyPlan",
            description=(
                "Deterministically compile a frontier-model proposal into PLAN_VALID, "
                "CLARIFICATION_REQUIRED, or UNSUPPORTED. It never executes, charges, or approves."
            ),
            risk_class="R0_read",
            oel_api="sim.study_planning.preflight_study_plan",
            maturity="prototype",
            install_profile="public",
            deployment_profiles=(PROFILE,),
            data_classes=("user_study_intent", "selected_project_inputs"),
            writes=False,
            input_schema=object_schema(
                handling_properties(
                    {
                        "request": deepcopy(REQUEST_PROPOSAL_SCHEMA),
                        "plan": deepcopy(NULLABLE_PLAN_PROPOSAL_SCHEMA),
                        "configs": {"type": "array", "items": CONFIG_INPUT_SCHEMA, "maxItems": 32},
                    }
                ),
                required=("request", "plan", "handling"),
            ),
            result_schema=deepcopy(PLANNING_RESULT_SCHEMA),
            limits={"max_configs": 32, "max_config_bytes": 2_000_000, "execution_allowed": False},
        ),
        ToolContract(
            tool_id="oel.study.plan_review.v1",
            title="Render a preflight-bound StudyPlan review",
            description=(
                "Render the exact content-bound plan, transfer set, evidence, access route, and limitations for user review. "
                "The review does not authorize execution and the agent cannot authorize it."
            ),
            risk_class="R0_read",
            oel_api="sim.study_planning.build_plan_review",
            maturity="prototype",
            install_profile="public",
            deployment_profiles=(PROFILE,),
            data_classes=("user_study_intent",),
            writes=False,
            input_schema=object_schema(
                handling_properties({"planning_result": deepcopy(PLANNING_RESULT_SCHEMA)}),
                required=("planning_result", "handling"),
            ),
            result_schema=deepcopy(STUDY_PLAN_REVIEW_SCHEMA),
        ),
        ToolContract(
            tool_id="oel.hosted.validate_package.v1",
            title="Validate a local Hosted execution package",
            description=(
                "Validate and content-bind a public Hosted execution package without uploading bytes, "
                "importing Pro code, executing OEL, authorizing payment, or approving a run."
            ),
            risk_class="R0_read",
            oel_api="sim.hosted_client.validate_hosted_execution_package",
            maturity="prototype",
            install_profile="public",
            deployment_profiles=(PROFILE,),
            data_classes=("user_study_intent", "selected_project_inputs"),
            writes=False,
            input_schema=object_schema(
                handling_properties(
                    {"package_path": {"type": "string", "minLength": 1, "maxLength": 2000}}
                ),
                required=("package_path", "handling"),
            ),
            result_schema={
                "type": "object",
                "properties": {
                    "schema": {"const": HOSTED_EXECUTION_PACKAGE_VALIDATION_SCHEMA_ID},
                    "status": {"enum": ["CLIENT_VALID", "CLIENT_INVALID"]},
                },
                "required": ("schema", "status"),
            },
            limits={
                "execution_allowed": False,
                "upload_allowed": False,
                "payment_authorization_allowed": False,
            },
        ),
        ToolContract(
            tool_id="oel.study.prepare_capability_request.v1",
            title="Prepare an opt-in capability-request preview",
            description=(
                "Prepare a sanitized editable preview after UNSUPPORTED. It never sends feedback and cannot "
                "authorize its own submission."
            ),
            risk_class="R0_read",
            oel_api="sim.study_planning.prepare_capability_request",
            maturity="prototype",
            install_profile="public",
            deployment_profiles=(PROFILE,),
            data_classes=("user_study_intent", "product_feedback"),
            writes=False,
            input_schema=object_schema(
                handling_properties(
                    {
                        "planning_result": deepcopy(PLANNING_RESULT_SCHEMA),
                        "user_summary": {"type": "string", "minLength": 1, "maxLength": 4000},
                        "would_pay": {"type": ["boolean", "null"]},
                        "contact_permission": {"type": "boolean"},
                        "contact": {"type": ["string", "null"], "maxLength": 500},
                        "attachments": {"type": "array", "items": {"type": "object"}, "maxItems": 16},
                    }
                ),
                required=(
                    "planning_result",
                    "user_summary",
                    "would_pay",
                    "contact_permission",
                    "contact",
                    "attachments",
                    "handling",
                ),
            ),
            result_schema=deepcopy(CAPABILITY_REQUEST_SCHEMA),
            limits={"submission_allowed": False, "automatic_project_attachment": False},
        ),
    )
    return {contract.tool_id: contract for contract in contracts}


__all__ = ["PROFILE", "study_contracts"]
