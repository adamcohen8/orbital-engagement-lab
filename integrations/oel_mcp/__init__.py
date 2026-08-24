"""Optional MCP adapter over documented Orbital Engagement Lab workflows."""

from sim.runtime_environment import configure_runtime_caches

configure_runtime_caches()

from integrations.oel_mcp.handlers import OELMCPHandlers  # noqa: E402
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers  # noqa: E402

__all__ = ["OELMCPHandlers", "PublicOELMCPHandlers"]
