"""Managed installation, update, and user-workspace support for OEL."""

from .contracts import (
    CHANNEL_CONFIG_SCHEMA,
    INSTALLATION_RECORD_SCHEMA,
    RELEASE_MANIFEST_SCHEMA,
    WORKSPACE_SCHEMA,
)
from .paths import InstallationPaths

__all__ = [
    "CHANNEL_CONFIG_SCHEMA",
    "INSTALLATION_RECORD_SCHEMA",
    "RELEASE_MANIFEST_SCHEMA",
    "WORKSPACE_SCHEMA",
    "InstallationPaths",
]
