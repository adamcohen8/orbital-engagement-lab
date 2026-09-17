"""Resolve OEL resources in a checkout or an installed Python environment."""

from __future__ import annotations

import sysconfig
from importlib.resources import files
from pathlib import Path


class ResourceNotFoundError(FileNotFoundError):
    """Raised when a required installed OEL resource cannot be located."""


def resource_path(*parts: str) -> Path:
    """Return one maintained resource without depending on the current cwd."""

    relative = Path(*parts)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("OEL resource paths must be relative and may not escape their resource root.")
    checkout = Path(__file__).resolve().parents[2] / relative
    installed = Path(sysconfig.get_path("data")) / "share" / "oel" / relative
    for candidate in (checkout, installed):
        if candidate.is_file():
            return candidate
    raise ResourceNotFoundError(f"Required OEL resource was not installed: {relative.as_posix()}")


def quickstart_config_path() -> Path:
    return resource_path("configs", "quickstart_5min.yaml")


def agent_bootstrap_text() -> str:
    """One packaged operational guide for managed workspaces and MCP."""
    return files("sim.installation").joinpath("data/agent-bootstrap.md").read_text(encoding="utf-8")
