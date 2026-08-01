from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from threading import Event
from typing import Any

from integrations.oel_mcp.base_handlers import BaseOELMCPHandlers
from integrations.oel_mcp.protocol import SERVER_INFO, tool_definitions
from integrations.oel_mcp.resources import (
    PublishedResource,
    build_public_resource_catalog,
    public_resource_map,
)

MCP_SDK_REQUIREMENT = "mcp>=2.0.0,<3"
MCP_SDK_REVIEWED_VERSION = "2.0.0"
MCP_SDK_PROTOCOL_VERSION = "2026-07-28"
MCP_SDK_LEGACY_PROTOCOL_VERSION = "2025-11-25"
SDK_SERVER_INFO = {"name": SERVER_INFO["name"], "version": "2.0.0"}


@dataclass(frozen=True)
class SDKLifecycleState:
    resources: tuple[PublishedResource, ...]


def installed_sdk_version() -> str | None:
    """Return the installed optional SDK version without importing the SDK."""

    try:
        return version("mcp")
    except PackageNotFoundError:
        return None


def build_sdk_server(handlers: BaseOELMCPHandlers) -> Any:
    """Build an official-SDK server over OEL's frozen transport-neutral handlers."""

    try:
        from mcp import MCPError
        from mcp.server import Server
        from mcp_types import (
            Annotations,
            CallToolResult,
            ListResourcesResult,
            ListToolsResult,
            ReadResourceResult,
            Resource,
            TextContent,
            TextResourceContents,
            Tool,
        )
    except ImportError as exc:  # pragma: no cover - exercised in no-MCP installation checks
        raise RuntimeError(
            'The optional MCP SDK is not installed. Install the OEL MCP profile with `pip install ".[mcp]"`.'
        ) from exc

    async def list_tools(_context: Any, _params: Any) -> Any:
        return ListToolsResult(tools=[Tool(**definition) for definition in tool_definitions(handlers)])

    async def call_tool(context: Any, params: Any) -> Any:
        import anyio

        cancel_event = Event()

        def progress(value: float, total: float | None, message: str) -> None:
            anyio.from_thread.run(context.session.report_progress, value, total, message)

        try:
            payload = await anyio.to_thread.run_sync(
                lambda: handlers.call(
                    str(params.name),
                    dict(params.arguments or {}),
                    **(
                        {"cancel_event": cancel_event, "progress": progress}
                        if isinstance(handlers, BaseOELMCPHandlers)
                        else {}
                    ),
                ),
                abandon_on_cancel=True,
            )
        except anyio.get_cancelled_exc_class():
            cancel_event.set()
            raise
        except (TypeError, ValueError, PermissionError) as exc:
            raise MCPError(-32602, str(exc)) from exc
        except Exception as exc:
            raise MCPError(-32603, "Internal server error without local diagnostic details.") from exc
        return CallToolResult(
            content=[TextContent(text=json.dumps(payload, sort_keys=True))],
            structuredContent=payload,
            isError=payload.get("status") == "failed",
        )

    @asynccontextmanager
    async def lifespan(_server: Any) -> Any:
        resources = build_public_resource_catalog(
            profile=str(getattr(handlers, "profile", "public_local")),
            tool_contracts=handlers.contracts.values(),
        )
        yield SDKLifecycleState(resources=resources)

    async def list_resources(context: Any, _params: Any) -> Any:
        return ListResourcesResult(
            cacheScope="public",
            ttlMs=300_000,
            resources=[
                Resource(
                    name=resource.contract.name,
                    title=resource.contract.title,
                    uri=resource.contract.uri,
                    description=resource.contract.description,
                    mimeType=resource.contract.mime_type,
                    size=resource.size,
                    annotations=Annotations(audience=["assistant"], priority=0.8),
                )
                for resource in context.lifespan_context.resources
            ],
        )

    async def read_resource(context: Any, params: Any) -> Any:
        resources = public_resource_map(context.lifespan_context.resources)
        resource = resources.get(str(params.uri))
        if resource is None:
            raise MCPError(-32602, "Resource is not available in this deployment profile.")
        return ReadResourceResult(
            cacheScope="public",
            ttlMs=300_000,
            contents=[
                TextResourceContents(
                    uri=resource.contract.uri,
                    mimeType=resource.contract.mime_type,
                    text=resource.text,
                )
            ],
        )

    return Server(
        SDK_SERVER_INFO["name"],
        version=SDK_SERVER_INFO["version"],
        lifespan=lifespan,
        instructions=(
            "Use OEL as the deterministic physics and evidence authority. "
            "The active local surface requires its declared validation, entitlement, data-handling, and operator-approval policies."
        ),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
    )


async def _serve_sdk(server: Any) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def serve_sdk(handlers: BaseOELMCPHandlers) -> None:
    """Serve one local stdio connection through the official MCP SDK."""

    try:
        import anyio
    except ImportError as exc:  # pragma: no cover - exercised in no-MCP installation checks
        raise RuntimeError(
            'The optional MCP SDK is not installed. Install the OEL MCP profile with `pip install ".[mcp]"`.'
        ) from exc
    anyio.run(_serve_sdk, build_sdk_server(handlers))


__all__ = [
    "MCP_SDK_LEGACY_PROTOCOL_VERSION",
    "MCP_SDK_PROTOCOL_VERSION",
    "MCP_SDK_REQUIREMENT",
    "MCP_SDK_REVIEWED_VERSION",
    "SDK_SERVER_INFO",
    "SDKLifecycleState",
    "build_sdk_server",
    "installed_sdk_version",
    "serve_sdk",
]
