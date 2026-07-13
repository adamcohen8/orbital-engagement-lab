from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from integrations.oel_mcp.contracts import MAX_RESPONSE_BYTES, TOOL_CONTRACT_VERSION, ToolContract, effects
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
    ) -> None:
        self.profile = profile
        self.contracts = dict(contracts)
        self.path_policy = MCPPathPolicy.configured(read_roots=read_roots, write_roots=write_roots)
        self.max_response_bytes = max(1, min(int(max_response_bytes), MAX_RESPONSE_BYTES))

    def describe_capabilities(self) -> dict[str, Any]:
        contract = self.contracts["oel.describe_capabilities.v1"]
        result = {
            "status": "available",
            "integration": "oel_mcp_pre_v2",
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
                "The MCP adapter does not implement physics, IHE scoring, or dataset validation.",
                "MCP discovery and transport do not replace deployment authorization or release policy.",
                "Pre-v2 tools do not execute simulation physics or communicate externally.",
            ],
        }
        return self._envelope(contract=contract, arguments={}, operation=lambda: result)

    def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
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
        return self._call_contract(contract, args)

    def _call_contract(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
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
            result = operation()
            _enforce_response_size(result, self.max_response_bytes)
            status = outcome_status(result) if outcome_status else "completed"
            evidence_status = evidence(result) if evidence else _evidence(complete=status == "completed")
            error = None
        except Exception as exc:
            result = None
            status = "failed"
            evidence_status = _evidence(complete=False)
            error = {"type": type(exc).__name__, "message": _safe_error_message(exc)}
        return {
            "tool_contract_version": TOOL_CONTRACT_VERSION,
            "tool_id": contract.tool_id,
            "risk_class": contract.risk_class,
            "status": status,
            "effects": effects(writes=contract.writes),
            "evidence": evidence_status,
            "error": error,
            "audit": {
                "schema_version": 1,
                "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "deployment_profile": self.profile,
                "tool_id": contract.tool_id,
                "status": status,
                "arguments_sha256": argument_digest,
                "payload_retained": False,
            },
            "result": result,
        }


def require_file_size(path: Path, *, maximum: int) -> None:
    if path.stat().st_size > maximum:
        raise ValueError("Authorized input exceeds the tool file-size budget.")


def _validate_arguments(contract: ToolContract, arguments: dict[str, Any]) -> None:
    schema = contract.input_schema
    properties = dict(schema.get("properties", {}) or {})
    required = set(schema.get("required", []) or [])
    missing = sorted(required - set(arguments))
    unexpected = sorted(set(arguments) - set(properties))
    if missing:
        raise ValueError(f"Missing required tool arguments: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected tool arguments: {', '.join(unexpected)}")
    for name, value in arguments.items():
        definition = dict(properties.get(name, {}) or {})
        if definition.get("type") == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Tool argument {name} must be an integer.")
            if value < int(definition.get("minimum", value)) or value > int(definition.get("maximum", value)):
                raise ValueError(f"Tool argument {name} is outside its supported range.")
        if definition.get("type") == "string" and not isinstance(value, (str, Path)):
            raise ValueError(f"Tool argument {name} must be a string.")
        if definition.get("type") == "array" and not isinstance(value, list):
            raise ValueError(f"Tool argument {name} must be an array.")


def _redacted_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "argument_names": sorted(arguments),
        "handling": {
            "marking": str(dict(arguments.get("handling", {}) or {}).get("marking", "")),
            "release_scope": str(dict(arguments.get("handling", {}) or {}).get("release_scope", "")),
        },
    }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _enforce_response_size(result: dict[str, Any], maximum: int) -> None:
    size = len(json.dumps(result, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    if size > maximum:
        raise ValueError("Tool result exceeds the configured response-size budget.")


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, PermissionError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return str(exc)
    return str(exc)


def _evidence(*, complete: bool, empty: bool = False, truncated: bool = False) -> dict[str, bool]:
    return {"complete": complete, "empty": empty, "truncated": truncated}


__all__ = ["BaseOELMCPHandlers", "require_file_size"]
