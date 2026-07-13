from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from integrations.oel_mcp.protocol import MCP_PROTOCOL_VERSION, OELMCPServer


class ConformanceClient(Protocol):
    def initialize(self) -> dict[str, Any]: ...

    def list_tools(self) -> list[dict[str, Any]]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def ping(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ConformanceResult:
    passed: bool
    checks: tuple[dict[str, Any], ...]


class DispatchConformanceClient:
    """Adapter for the pre-v2 prototype; SDK clients can implement the same protocol."""

    def __init__(self, server: OELMCPServer) -> None:
        self.server = server
        self._next_id = 1

    def initialize(self) -> dict[str, Any]:
        return self._result(
            "initialize",
            {"protocolVersion": MCP_PROTOCOL_VERSION, "clientInfo": {"name": "oel-conformance", "version": "1"}},
        )

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._result("tools/list").get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._result("tools/call", {"name": name, "arguments": arguments})

    def ping(self) -> dict[str, Any]:
        return self._result("ping")

    def _result(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        response = self.server.dispatch(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params or {})}
        )
        if response is None or "error" in response:
            raise AssertionError(f"Conformance request failed for {method}: {response}")
        return dict(response.get("result", {}) or {})


def run_conformance(client: ConformanceClient, *, expected_tool_ids: tuple[str, ...]) -> ConformanceResult:
    checks: list[dict[str, Any]] = []

    initialized = client.initialize()
    _check(checks, "protocol_version", initialized.get("protocolVersion") == MCP_PROTOCOL_VERSION)
    _check(checks, "tool_capability", "tools" in dict(initialized.get("capabilities", {}) or {}))
    _check(checks, "ping", client.ping() == {})

    tools = client.list_tools()
    names = tuple(str(item.get("name", "")) for item in tools)
    _check(checks, "tool_registry", names == expected_tool_ids, detail={"actual": names})
    _check(
        checks,
        "schemas_present",
        all(isinstance(item.get("inputSchema"), dict) and isinstance(item.get("outputSchema"), dict) for item in tools),
    )

    described = client.call_tool("oel.describe_capabilities.v1", {})
    structured = dict(described.get("structuredContent", {}) or {})
    _check(checks, "structured_result", structured.get("status") == "completed")
    _check(checks, "no_external_effect", not bool(dict(structured.get("effects", {}) or {}).get("external_communication")))
    _check(checks, "audit_without_payload", dict(structured.get("audit", {}) or {}).get("payload_retained") is False)

    return ConformanceResult(passed=all(bool(row["passed"]) for row in checks), checks=tuple(checks))


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    detail: dict[str, Any] | None = None,
) -> None:
    checks.append({"check_id": check_id, "passed": bool(passed), "detail": dict(detail or {})})
