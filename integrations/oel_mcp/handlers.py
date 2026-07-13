"""Compatibility import for the public pre-v2 handler name."""

from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers

OELMCPHandlers = PublicOELMCPHandlers

__all__ = ["OELMCPHandlers", "PublicOELMCPHandlers"]
