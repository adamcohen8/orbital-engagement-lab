from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

TOOL_CONTRACT_VERSION = 1
MAX_ROWS = 1000
MAX_VM_STEPS = 2_000_000
MAX_MANIFEST_BYTES = 5_000_000
MAX_REVIEW_STORE_BYTES = 1_000_000_000
MAX_RESPONSE_BYTES = 1_000_000

DEPLOYMENT_PROFILES = frozenset(
    {
        "public_local",
        "pro_local",
        "mendicant_sealed",
        "mendicant_tandem",
        "direct_frontier_restricted",
    }
)

HANDLING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "marking": {"type": "string", "minLength": 1, "maxLength": 120},
        "release_scope": {
            "type": "string",
            "enum": ["public", "local_only", "frontier_eligible"],
        },
        "owner": {"type": "string", "maxLength": 200, "default": ""},
    },
    "required": ["marking", "release_scope"],
    "additionalProperties": False,
}

ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_contract_version": {"const": TOOL_CONTRACT_VERSION},
        "tool_id": {"type": "string"},
        "risk_class": {"enum": ["R0_read", "R1_write"]},
        "status": {"enum": ["completed", "partial", "failed"]},
        "effects": {
            "type": "object",
            "properties": {
                "reads": {"type": "boolean"},
                "writes": {"type": "boolean"},
                "executes": {"const": False},
                "external_communication": {"const": False},
            },
            "required": ["reads", "writes", "executes", "external_communication"],
            "additionalProperties": False,
        },
        "evidence": {
            "type": "object",
            "properties": {
                "complete": {"type": "boolean"},
                "empty": {"type": "boolean"},
                "truncated": {"type": "boolean"},
            },
            "required": ["complete", "empty", "truncated"],
            "additionalProperties": False,
        },
        "error": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {"type": {"type": "string"}, "message": {"type": "string"}},
                    "required": ["type", "message"],
                    "additionalProperties": False,
                },
            ]
        },
        "audit": {
            "type": "object",
            "properties": {
                "schema_version": {"const": 1},
                "generated_utc": {"type": "string"},
                "deployment_profile": {"type": "string"},
                "tool_id": {"type": "string"},
                "status": {"type": "string"},
                "arguments_sha256": {"type": "string"},
                "payload_retained": {"const": False},
            },
            "required": [
                "schema_version",
                "generated_utc",
                "deployment_profile",
                "tool_id",
                "status",
                "arguments_sha256",
                "payload_retained",
            ],
            "additionalProperties": False,
        },
        "result": {},
    },
    "required": [
        "tool_contract_version",
        "tool_id",
        "risk_class",
        "status",
        "effects",
        "evidence",
        "error",
        "audit",
        "result",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ToolContract:
    tool_id: str
    title: str
    description: str
    risk_class: str
    oel_api: str
    maturity: str
    install_profile: str
    deployment_profiles: tuple[str, ...]
    data_classes: tuple[str, ...]
    writes: bool
    input_schema: dict[str, Any]
    result_schema: dict[str, Any]
    deprecated: bool = False
    replacement: str = ""

    def capability(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "title": self.title,
            "risk_class": self.risk_class,
            "oel_api": self.oel_api,
            "maturity": self.maturity,
            "required_install_profile": self.install_profile,
            "deployment_profiles": list(self.deployment_profiles),
            "data_classes": list(self.data_classes),
            "effects": effects(writes=self.writes),
            "deprecated": self.deprecated,
            "replacement": self.replacement,
            "input_schema_sha256": _schema_sha256(self.input_schema),
            "result_schema_sha256": _schema_sha256(self.result_schema),
            "limits": _tool_limits(self.tool_id),
        }

    def mcp_definition(self) -> dict[str, Any]:
        output_schema = deepcopy(ENVELOPE_SCHEMA)
        output_schema["properties"]["tool_id"] = {"const": self.tool_id}
        output_schema["properties"]["risk_class"] = {"const": self.risk_class}
        output_schema["properties"]["result"] = deepcopy(self.result_schema)
        return {
            "name": self.tool_id,
            "title": self.title,
            "description": self.description,
            "inputSchema": deepcopy(self.input_schema),
            "outputSchema": output_schema,
        }


def object_schema(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": deepcopy(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def handling_properties(properties: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(properties)
    out["handling"] = deepcopy(HANDLING_SCHEMA)
    return out


def effects(*, writes: bool) -> dict[str, bool]:
    return {
        "reads": True,
        "writes": writes,
        "executes": False,
        "external_communication": False,
    }


def _schema_sha256(schema: dict[str, Any]) -> str:
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tool_limits(tool_id: str) -> dict[str, Any]:
    limits: dict[str, Any] = {"max_response_bytes": MAX_RESPONSE_BYTES}
    if tool_id == "oel.inspect_run.v1":
        limits.update({"max_rows_per_saved_query": MAX_ROWS, "max_review_store_bytes": MAX_REVIEW_STORE_BYTES})
    elif tool_id == "oel.query_review.v1":
        limits.update(
            {
                "max_rows": MAX_ROWS,
                "max_vm_steps": MAX_VM_STEPS,
                "max_review_store_bytes": MAX_REVIEW_STORE_BYTES,
            }
        )
    elif tool_id.startswith("oel.pro."):
        limits["max_input_file_bytes"] = MAX_MANIFEST_BYTES
    return limits
