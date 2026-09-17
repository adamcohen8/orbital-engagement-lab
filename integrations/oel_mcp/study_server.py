"""Local MCP server for the BYO frontier planning pilot (Phase 3A)."""

from __future__ import annotations

import os
import shutil
import sys
from typing import Sequence

from integrations.oel_mcp.diagnostics import run_server_cli
from integrations.oel_mcp.protocol import OELMCPServer
from integrations.oel_mcp.study_handlers import StudyPlanningMCPHandlers


def _serve(_profile: str) -> None:
    handlers = StudyPlanningMCPHandlers()
    adapter = os.environ.get("OEL_MCP_ADAPTER", "sdk").strip().lower()
    if adapter == "legacy":
        OELMCPServer(handlers).serve()
        return
    if adapter == "sdk":
        from integrations.oel_mcp.sdk_protocol import serve_sdk

        serve_sdk(handlers)
        return
    raise ValueError("OEL_MCP_ADAPTER must be either 'legacy' or 'sdk'.")


def _doctor(_profile: str, adapter: str) -> dict[str, object]:
    handlers = StudyPlanningMCPHandlers()
    return {
        "status": "ready",
        "adapter": adapter,
        "deployment_profile": handlers.profile,
        "tool_ids": sorted(handlers.contracts),
        "effects": {"writes": False, "executes": False, "external_communication": False},
        "model_credentials_received_by_oel": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--print-host-config" in arguments and "--command" not in arguments:
        entrypoint = shutil.which("oel-study-mcp")
        if entrypoint:
            arguments.extend(("--command", entrypoint))
        else:
            arguments.extend(
                (
                    "--command",
                    str(sys.executable),
                    "--arg=-m",
                    "--arg=integrations.oel_mcp.study_server",
                )
            )
    return run_server_cli(
        argv=arguments,
        default_profile="direct_frontier_restricted",
        serve=_serve,
        doctor=_doctor,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
