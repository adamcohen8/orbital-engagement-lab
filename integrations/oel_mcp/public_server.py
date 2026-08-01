import os
from typing import Sequence

from integrations.oel_mcp.diagnostics import run_server_cli
from integrations.oel_mcp.protocol import OELMCPServer
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from integrations.oel_mcp.public_registry import M3_PUBLIC_TOOL_IDS, PUBLIC_PROFILES


def handlers_for_profile(profile: str) -> PublicOELMCPHandlers:
    selected = str(profile or "").strip()
    if selected not in PUBLIC_PROFILES:
        raise PermissionError("Deployment profile is not authorized by the public MCP server.")
    return PublicOELMCPHandlers(profile=selected)


def _serve(profile: str = "public_local") -> None:
    handlers = handlers_for_profile(profile)
    adapter = os.environ.get("OEL_MCP_ADAPTER", "sdk").strip().lower()
    if adapter == "legacy":
        handlers.contracts = {tool_id: handlers.contracts[tool_id] for tool_id in M3_PUBLIC_TOOL_IDS}
        OELMCPServer(handlers).serve()
        return
    if adapter == "sdk":
        from integrations.oel_mcp.sdk_protocol import serve_sdk

        serve_sdk(handlers)
        return
    raise ValueError("OEL_MCP_ADAPTER must be either 'legacy' or 'sdk'.")


def main(argv: Sequence[str] | None = None) -> int:
    return run_server_cli(
        argv=None if argv is None else list(argv),
        default_profile="public_local",
        serve=_serve,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["handlers_for_profile", "main"]
