"""Privacy-scoped local drafts for refusal feedback and plan-bound complaints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .contracts import (
    CAPABILITY_REQUEST_SCHEMA_ID,
    PLANNING_RESULT_SCHEMA_ID,
    STUDY_COMPLAINT_SCHEMA_ID,
    canonical_sha256,
    require_valid_document,
)

_REFUSAL_CATEGORIES = {
    "missing_physics",
    "missing_data",
    "unsupported_analysis",
    "insufficient_validation",
    "resource_limit",
    "deployment_limit",
    "planner_error",
}


def prepare_capability_request(
    planning_result: Mapping[str, Any],
    *,
    user_summary: str,
    engine_version: str,
    would_pay: bool | None = None,
    contact_permission: bool = False,
    contact: str | None = None,
    attachments: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Prepare, but never submit, a sanitized feedback payload after refusal."""

    result = require_valid_document(planning_result, expected_schema=PLANNING_RESULT_SCHEMA_ID)
    if result["status"] != "UNSUPPORTED":
        raise ValueError("A capability request may only be prepared from an UNSUPPORTED planning result.")
    blockers = [item for item in result["findings"] if item["severity"] == "blocker"]
    first = blockers[0] if blockers else {}
    category = str(first.get("category", "planner_error"))
    if category not in _REFUSAL_CATEGORIES:
        category = "planner_error"
    missing_capability = str(first.get("message", "The planner could not identify an executable OEL study."))
    provisional = {
        "schema": CAPABILITY_REQUEST_SCHEMA_ID,
        "request_id": result["request"]["request_id"],
        "refusal_category": category,
        "missing_capability": missing_capability,
        "user_summary": str(user_summary),
        "would_pay": would_pay,
        "contact_permission": bool(contact_permission),
        "contact": str(contact) if contact_permission and contact else None,
        "versions": {
            "planner": str(result["planner_version"]),
            "capability_manifest": str(result["capability_manifest_sha256"]),
            "engine": str(engine_version),
        },
        "attachments": [deepcopy(dict(item)) for item in attachments],
        "submission_authorized": False,
    }
    provisional["capability_request_id"] = f"capability-request:{canonical_sha256(provisional)[:24]}"
    return require_valid_document(provisional, expected_schema=CAPABILITY_REQUEST_SCHEMA_ID)


def prepare_study_complaint(
    *,
    request_id: str,
    plan_id: str,
    plan_sha256: str,
    receipt_id: str,
    alleged_failure: str,
    violated_criterion_ids: Sequence[str],
    selected_evidence_refs: Sequence[str],
    product_feedback_permission: bool = False,
) -> dict[str, Any]:
    """Prepare a local complaint preview without sending or authorizing it."""

    provisional = {
        "schema": STUDY_COMPLAINT_SCHEMA_ID,
        "request_id": request_id,
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "receipt_id": receipt_id,
        "alleged_failure": alleged_failure,
        "violated_criterion_ids": list(violated_criterion_ids),
        "selected_evidence_refs": list(selected_evidence_refs),
        "product_feedback_permission": bool(product_feedback_permission),
        "submission_authorized": False,
    }
    provisional["study_complaint_id"] = f"complaint:{canonical_sha256(provisional)[:24]}"
    return require_valid_document(provisional, expected_schema=STUDY_COMPLAINT_SCHEMA_ID)


__all__ = ["prepare_capability_request", "prepare_study_complaint"]
