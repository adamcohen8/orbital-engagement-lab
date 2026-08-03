from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable

from integrations.oel_mcp.contracts import MAX_RESPONSE_BYTES, TOOL_CONTRACT_VERSION, ToolContract, effects
from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.policy import MCPPathPolicy, validate_handling


class BaseOELMCPHandlers:
    """Shared transport-independent contract, policy, audit, and envelope behavior."""

    def __init__(
        self,
        *,
        profile: str,
        contracts: dict[str, ToolContract],
        read_roots: tuple[str | Path, ...] | None = None,
        write_roots: tuple[str | Path, ...] | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        approval_policy: ExecutionApprovalPolicy | None = None,
    ) -> None:
        self.profile = profile
        self.contracts = dict(contracts)
        self.path_policy = MCPPathPolicy.configured(read_roots=read_roots, write_roots=write_roots)
        self.max_response_bytes = max(1, min(int(max_response_bytes), MAX_RESPONSE_BYTES))
        self.approval_policy = approval_policy or ExecutionApprovalPolicy.configured()

    def describe_capabilities(self) -> dict[str, Any]:
        contract = self.contracts["oel.describe_capabilities.v1"]
        result = {
            "status": "available",
            "integration": "oel_mcp_v1",
            "transport": "stdio",
            "deployment_profile": self.profile,
            "capabilities": [item.capability() for item in self.contracts.values()],
            "dependency_direction": "mcp_consumes_oel",
            "compatibility": {
                "additive_optional_fields_allowed": True,
                "new_major_required_for": [
                    "required_argument_change",
                    "unit_or_field_meaning_change",
                    "risk_or_effect_change",
                    "hidden_truth_boundary_change",
                    "quality_gate_weakening",
                ],
            },
            "non_claims": [
                "The MCP adapter does not implement physics, private scoring workflows, or dataset validation.",
                "MCP discovery and transport do not replace deployment authorization or release policy.",
                "Supported execution is local, explicitly operator-approved, and never communicates externally.",
            ],
        }
        return self._envelope(contract=contract, arguments={}, operation=lambda: result)

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        cancel_event: Event | None = None,
        progress: Callable[[float, float | None, str], None] | None = None,
    ) -> dict[str, Any]:
        tool_id = str(tool_name)
        contract = self.contracts.get(tool_id)
        if contract is None:
            raise PermissionError("Tool is not available in this deployment profile.")
        args = dict(arguments or {})
        if tool_id == "oel.describe_capabilities.v1":
            _validate_arguments(contract, args)
            return self.describe_capabilities()
        validate_handling(self.profile, args.get("handling"))
        _validate_arguments(contract, args)
        if contract.writes or contract.executes:
            self.approval_policy.require(args.get("approval"), executes=contract.executes)
        requires_plugin_trust = (
            tool_id == "oel.validate_scenario.v1" and bool(args.get("trust_plugins", False))
            or tool_id == "oel.run_scenario.v1"
            or tool_id in {"oel.materialize_onp_handoff.v1", "oel.materialize_scenario_patch.v1"}
            and bool(args.get("trust_plugins", False))
        )
        if requires_plugin_trust:
            self.approval_policy.require_trust(args.get("trust_approval"))
        return self._call_contract(contract, args, cancel_event=cancel_event, progress=progress)

    def _call_contract(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None = None,
        progress: Callable[[float, float | None, str], None] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _envelope(
        self,
        *,
        contract: ToolContract,
        arguments: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
        evidence: Callable[[dict[str, Any]], dict[str, bool]] | None = None,
        outcome_status: Callable[[dict[str, Any]], str] | None = None,
    ) -> dict[str, Any]:
        argument_digest = _sha256_json(_redacted_arguments(arguments))
        try:
            raw_result = operation()
            status = outcome_status(raw_result) if outcome_status else "completed"
            evidence_status = evidence(raw_result) if evidence else _evidence(complete=status == "completed")
            result = _json_safe_value(self._project_result(raw_result))
            _validate_value(result, contract.result_schema, path="tool result")
            _enforce_response_size(result, self.max_response_bytes)
            error = None
        except Exception as exc:
            result = None
            status = "failed"
            evidence_status = _evidence(complete=False)
            error = {
                "type": type(exc).__name__,
                "message": _safe_error_message(exc, authorized_roots=self._authorized_roots()),
            }
        return {
            "tool_contract_version": TOOL_CONTRACT_VERSION,
            "tool_id": contract.tool_id,
            "risk_class": contract.risk_class,
            "status": status,
            "effects": effects(writes=contract.writes, executes=contract.executes),
            "evidence": evidence_status,
            "error": error,
            "audit": {
                "schema_version": 1,
                "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "deployment_profile": self.profile,
                "tool_id": contract.tool_id,
                "status": status,
                "arguments_sha256": argument_digest,
                "arguments_sha256_semantics": "argument_names_and_handling_labels_only",
                "argument_values_retained": False,
                "payload_retained": False,
            },
            "result": result,
        }

    def _project_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.profile != "direct_frontier_restricted":
            return result
        return _frontier_safe_value(result, authorized_roots=self._authorized_roots())

    def _authorized_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(self.path_policy.read_roots + self.path_policy.write_roots))


def require_file_size(path: Path, *, maximum: int) -> None:
    if path.stat().st_size > maximum:
        raise ValueError("Authorized input exceeds the tool file-size budget.")


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    return value


def _validate_arguments(contract: ToolContract, arguments: dict[str, Any]) -> None:
    _validate_value(arguments, contract.input_schema, path="tool arguments")


def _validate_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if value is None and "null" in expected:
            return
        candidates = [item for item in expected if item != "null"]
        if len(candidates) != 1:
            raise ValueError(f"{path} has an unsupported union schema.")
        schema = {**schema, "type": candidates[0]}
        expected = candidates[0]
    if expected == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object.")
        properties = dict(schema.get("properties", {}) or {})
        required = set(schema.get("required", []) or [])
        missing = sorted(required - set(value))
        unexpected = sorted(set(value) - set(properties)) if schema.get("additionalProperties") is False else []
        if missing:
            raise ValueError(f"Missing required {path}: {', '.join(missing)}")
        if unexpected:
            raise ValueError(f"Unexpected {path}: {', '.join(unexpected)}")
        for key, item in value.items():
            if key in properties:
                _validate_value(item, dict(properties[key]), path=f"{path}.{key}")
    elif expected == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array.")
        if len(value) < int(schema.get("minItems", len(value))) or len(value) > int(schema.get("maxItems", len(value))):
            raise ValueError(f"{path} is outside its supported item count.")
        item_schema = dict(schema.get("items", {}) or {})
        for index, item in enumerate(value):
            _validate_value(item, item_schema, path=f"{path}[{index}]")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer.")
        if value < int(schema.get("minimum", value)) or value > int(schema.get("maximum", value)):
            raise ValueError(f"{path} is outside its supported range.")
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{path} must be a finite number.")
        if float(value) < float(schema.get("minimum", value)) or float(value) > float(schema.get("maximum", value)):
            raise ValueError(f"{path} is outside its supported range.")
    elif expected == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean.")
    elif expected == "string":
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{path} must be a string.")
        text = str(value)
        if len(text) < int(schema.get("minLength", len(text))) or len(text) > int(schema.get("maxLength", len(text))):
            raise ValueError(f"{path} is outside its supported length.")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not one of the supported values.")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} does not match the required value.")


def _redacted_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "argument_names": sorted(arguments),
        "handling": {
            "marking": str(dict(arguments.get("handling", {}) or {}).get("marking", "")),
            "release_scope": str(dict(arguments.get("handling", {}) or {}).get("release_scope", "")),
            "data_label": str(dict(arguments.get("handling", {}) or {}).get("data_label", "")),
            "hidden_truth_access": dict(arguments.get("handling", {}) or {}).get("hidden_truth_access"),
        },
    }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _enforce_response_size(result: dict[str, Any], maximum: int) -> None:
    size = len(json.dumps(result, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    if size > maximum:
        raise ValueError("Tool result exceeds the configured response-size budget.")


def _safe_error_message(exc: Exception, *, authorized_roots: tuple[Path, ...]) -> str:
    if not isinstance(exc, (PermissionError, FileNotFoundError, ValueError)):
        return "Tool operation failed without disclosing local details."
    message = str(exc)
    if any(str(root) in message for root in authorized_roots):
        return "Tool operation failed without disclosing local details."
    return message


def _frontier_safe_value(value: Any, *, authorized_roots: tuple[Path, ...]) -> Any:
    if isinstance(value, dict):
        return {key: _frontier_safe_value(item, authorized_roots=authorized_roots) for key, item in value.items()}
    if isinstance(value, list):
        return [_frontier_safe_value(item, authorized_roots=authorized_roots) for item in value]
    if isinstance(value, tuple):
        return [_frontier_safe_value(item, authorized_roots=authorized_roots) for item in value]
    if not isinstance(value, str):
        return value
    if not any(str(root) in value for root in authorized_roots):
        return value
    normalized = value
    for index, root in enumerate(authorized_roots):
        normalized = normalized.replace(str(root), f"<AUTHORIZED_ROOT_{index}>")
    normalized = normalized.replace("\\", "/")
    return f"oel-local-ref:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}"


def _evidence(*, complete: bool, empty: bool = False, truncated: bool = False) -> dict[str, bool]:
    return {"complete": complete, "empty": empty, "truncated": truncated}


__all__ = ["BaseOELMCPHandlers", "require_file_size"]
