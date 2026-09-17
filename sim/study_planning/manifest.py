"""Authoritative content-bound capability-manifest construction."""

from __future__ import annotations

from typing import Any

from .capabilities import CapabilityCatalog
from .contracts import (
    STUDY_CAPABILITY_MANIFEST_SCHEMA_ID,
    canonical_sha256,
    require_valid_document,
)

PLANNER_VERSION = "oel-study-planner-3a.v1"


def build_capability_manifest(catalog: CapabilityCatalog) -> dict[str, Any]:
    """Build the one manifest used by CLI, MCP, preflight, and feedback."""

    payload = {
        "schema": STUDY_CAPABILITY_MANIFEST_SCHEMA_ID,
        "planner_version": PLANNER_VERSION,
        "catalog_scope": catalog.scope,
        "capabilities": catalog.to_list(),
        "access": {
            "descriptor_visibility": "public",
            "public_capabilities_available_locally": True,
            "pro_capabilities_available_locally": False,
            "pro_execution_tools_exposed": False,
            "hosted_preflight_required_for_pro": True,
        },
        "effects": {"writes": False, "executes": False, "external_communication": False},
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return require_valid_document(payload, expected_schema=STUDY_CAPABILITY_MANIFEST_SCHEMA_ID)


__all__ = ["PLANNER_VERSION", "build_capability_manifest"]
