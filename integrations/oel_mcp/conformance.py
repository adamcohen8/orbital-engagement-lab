from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

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


class SDKConformanceClient:
    """Synchronous conformance adapter over the official SDK's in-memory client."""

    def __init__(self, server: Any) -> None:
        self.server = server

    def initialize(self) -> dict[str, Any]:
        return self._request("initialize")

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._request("list_tools"))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._request("call_tool", name=name, arguments=arguments)

    def ping(self) -> dict[str, Any]:
        return self._request("ping")

    def _request(self, operation: str, **kwargs: Any) -> Any:
        try:
            import anyio
        except ImportError as exc:  # pragma: no cover - covered by no-MCP installation checks
            raise RuntimeError(
                'The optional MCP SDK is not installed. Install the OEL MCP profile with `pip install ".[mcp]"`.'
            ) from exc
        try:
            return anyio.run(self._request_async, operation, kwargs)
        except BaseException as exc:
            from mcp import MCPError

            nested = _find_nested_exception(exc, MCPError)
            if nested is not None:
                raise nested from None
            raise

    async def _request_async(self, operation: str, kwargs: dict[str, Any]) -> Any:
        import warnings

        try:
            from mcp import Client, MCPDeprecationWarning
        except ImportError as exc:  # pragma: no cover - covered by no-MCP installation checks
            raise RuntimeError(
                'The optional MCP SDK is not installed. Install the OEL MCP profile with `pip install ".[mcp]"`.'
            ) from exc

        async with Client(self.server, mode="legacy", cache=None) as client:
            operations: dict[str, Callable[[], Any]] = {
                "initialize": lambda: {
                    "protocolVersion": client.protocol_version,
                    "capabilities": client.server_capabilities.model_dump(by_alias=True, exclude_none=True),
                    "serverInfo": client.server_info.model_dump(by_alias=True, exclude_none=True),
                },
                "list_tools": lambda: client.list_tools(cache_mode="reload"),
                "call_tool": lambda: client.call_tool(str(kwargs["name"]), dict(kwargs["arguments"])),
                "ping": client.send_ping,
            }
            if operation not in operations:
                raise ValueError(f"Unsupported conformance operation: {operation}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", MCPDeprecationWarning)
                result = operations[operation]()
                if hasattr(result, "__await__"):
                    result = await result

        if operation == "list_tools":
            return [tool.model_dump(by_alias=True, exclude_none=True) for tool in result.tools]
        if operation in {"call_tool", "ping"}:
            return result.model_dump(by_alias=True, exclude_none=True)
        return result


class SDKStdioConformanceClient:
    """Official SDK client adapter that starts a fresh stdio server per operation."""

    def __init__(
        self,
        *,
        command: str,
        args: tuple[str, ...],
        cwd: str | Path,
        env: dict[str, str],
        mode: str = "auto",
    ) -> None:
        self.command = command
        self.args = args
        self.cwd = Path(cwd)
        self.env = dict(env)
        self.mode = mode

    def initialize(self) -> dict[str, Any]:
        return self._request("initialize")

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._request("list_tools"))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._request("call_tool", name=name, arguments=arguments)

    def ping(self) -> dict[str, Any]:
        return self._request("ping")

    def list_resources(self) -> list[dict[str, Any]]:
        return list(self._request("list_resources"))

    def read_resource(self, uri: str) -> dict[str, Any]:
        return self._request("read_resource", uri=uri)

    def _request(self, operation: str, **kwargs: Any) -> Any:
        try:
            import anyio
        except ImportError as exc:  # pragma: no cover - covered by no-MCP installation checks
            raise RuntimeError(
                'The optional MCP SDK is not installed. Install the OEL MCP profile with `pip install ".[mcp]"`.'
            ) from exc
        try:
            return anyio.run(self._request_async, operation, kwargs)
        except BaseException as exc:
            from mcp import MCPError

            nested = _find_nested_exception(exc, MCPError)
            if nested is not None:
                raise nested from None
            raise

    async def _request_async(self, operation: str, kwargs: dict[str, Any]) -> Any:
        import warnings

        try:
            from mcp import Client, MCPDeprecationWarning, StdioServerParameters, stdio_client
        except ImportError as exc:  # pragma: no cover - covered by no-MCP installation checks
            raise RuntimeError(
                'The optional MCP SDK is not installed. Install the OEL MCP profile with `pip install ".[mcp]"`.'
            ) from exc

        parameters = StdioServerParameters(
            command=self.command,
            args=list(self.args),
            cwd=self.cwd,
            env=self.env,
        )
        async with Client(stdio_client(parameters), mode=self.mode, cache=None) as client:
            operations: dict[str, Callable[[], Any]] = {
                "initialize": lambda: {
                    "protocolVersion": client.protocol_version,
                    "capabilities": client.server_capabilities.model_dump(by_alias=True, exclude_none=True),
                    "serverInfo": client.server_info.model_dump(by_alias=True, exclude_none=True),
                },
                "list_tools": lambda: client.list_tools(cache_mode="reload"),
                "call_tool": lambda: client.call_tool(str(kwargs["name"]), dict(kwargs["arguments"])),
                "ping": client.send_ping,
                "list_resources": lambda: client.list_resources(cache_mode="reload"),
                "read_resource": lambda: client.read_resource(str(kwargs["uri"]), cache_mode="reload"),
            }
            if operation not in operations:
                raise ValueError(f"Unsupported conformance operation: {operation}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", MCPDeprecationWarning)
                result = operations[operation]()
                if hasattr(result, "__await__"):
                    result = await result

        if operation == "list_tools":
            return [tool.model_dump(by_alias=True, exclude_none=True) for tool in result.tools]
        if operation == "list_resources":
            return [resource.model_dump(by_alias=True, exclude_none=True) for resource in result.resources]
        if operation in {"call_tool", "ping", "read_resource"}:
            return result.model_dump(by_alias=True, exclude_none=True)
        return result


def run_conformance(
    client: ConformanceClient,
    *,
    expected_tool_ids: tuple[str, ...],
    expected_protocol_versions: tuple[str, ...] = (MCP_PROTOCOL_VERSION,),
    check_ping: bool = True,
) -> ConformanceResult:
    checks: list[dict[str, Any]] = []

    initialized = client.initialize()
    _check(
        checks,
        "protocol_version",
        initialized.get("protocolVersion") in expected_protocol_versions,
        detail={"actual": initialized.get("protocolVersion"), "expected": expected_protocol_versions},
    )
    _check(checks, "tool_capability", "tools" in dict(initialized.get("capabilities", {}) or {}))
    if check_ping:
        _check(checks, "ping", client.ping() == {})
    else:
        _check(
            checks,
            "ping",
            True,
            detail={"skipped": True, "reason": "ping was removed from MCP protocol revision 2026-07-28"},
        )

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


def _find_nested_exception(exception: BaseException, exception_type: type[BaseException]) -> BaseException | None:
    if isinstance(exception, exception_type):
        return exception
    for nested in getattr(exception, "exceptions", ()):
        found = _find_nested_exception(nested, exception_type)
        if found is not None:
            return found
    return None
