"""Sanitized engine/workspace identity recorded in deterministic run evidence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sim.project_version import source_project_version


def execution_provenance() -> dict[str, Any]:
    version = os.environ.get("OEL_ENGINE_VERSION", "").strip() or source_project_version() or "unknown"
    disposition = os.environ.get("OEL_INSTALLATION_DISPOSITION", "").strip() or "developer"
    payload: dict[str, Any] = {
        "schema_version": "oel.execution-provenance.v1",
        "engine_version": version,
        "installation_disposition": disposition,
        "edition": os.environ.get("OEL_ENGINE_EDITION", "").strip() or None,
        "release_manifest_sha256": os.environ.get("OEL_RELEASE_MANIFEST_SHA256", "").strip() or None,
        "installation_transaction_id": os.environ.get("OEL_INSTALLATION_TRANSACTION_ID", "").strip() or None,
        "workspace": None,
    }
    workspace_root = os.environ.get("OEL_WORKSPACE_ROOT", "").strip()
    if workspace_root:
        try:
            from .workspace import load_workspace

            workspace = load_workspace(Path(workspace_root))
            payload["workspace"] = {
                "workspace_id": workspace["workspace_id"],
                "manifest_sha256": workspace["manifest_sha256"],
                "locked_version": workspace["engine"]["locked_version"],
                "contracts": workspace["contracts"],
            }
        except (OSError, RuntimeError, ValueError):
            payload["workspace"] = {"status": "unavailable"}
    return payload
