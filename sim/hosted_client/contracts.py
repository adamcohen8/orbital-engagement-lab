"""Public-safe contracts for routing a planned study toward local or hosted execution."""

from __future__ import annotations

from typing import Any, Mapping

from sim.study_planning.contracts import (
    PLANNING_RESULT_SCHEMA_ID,
    canonical_sha256,
    require_valid_document,
)

ROUTE_DECISION_SCHEMA = "oel.hosted_route_decision.v2"


def route_planning_result(
    planning_result: Mapping[str, Any], *, hosted_profile_configured: bool = False,
) -> dict[str, Any]:
    """Return the deterministic user-facing route without executing anything."""

    result = require_valid_document(
        planning_result,
        expected_schema=PLANNING_RESULT_SCHEMA_ID,
    )
    route = str(result["execution_route"]["route"])
    if route == "LOCAL_FREE_AVAILABLE":
        disposition = "execution_options_available"
        next_action = "review_local_recommendation_or_request_hosted_quote"
        message = (
            "This plan uses public OEL capabilities. Local execution is free and recommended, "
            "an invited alpha operator may request a simulated Hosted quote."
        )
        recommendation = {
            "option": "local",
            "reason": "The complete plan is available in public OEL without an OEL execution fee.",
            "requirement": False,
        }
        execution_options = {
            "local": {"available": True, "oel_execution_amount_usd": 0, "quote_required": False},
            "hosted": {"available": True, "oel_execution_amount_usd": None, "quote_required": True},
        }
    elif route == "HOSTED_PRO_REQUIRED":
        disposition = "hosted_pro_offer"
        next_action = "review_offer_then_explicitly_approve"
        message = "This plan requires hosted OEL Pro and explicit user approval before execution."
        recommendation = {
            "option": "hosted",
            "reason": "The plan includes one or more Pro capabilities that are not executable in public OEL.",
            "requirement": True,
        }
        execution_options = {
            "local": {"available": False, "oel_execution_amount_usd": None, "quote_required": False},
            "hosted": {"available": True, "oel_execution_amount_usd": None, "quote_required": True},
        }
    else:
        disposition = "not_eligible"
        next_action = "clarify_or_prepare_capability_feedback"
        message = "OEL cannot execute this proposal as currently planned."
        recommendation = {"option": None, "reason": message, "requirement": True}
        execution_options = {
            "local": {"available": False, "oel_execution_amount_usd": None, "quote_required": False},
            "hosted": {"available": False, "oel_execution_amount_usd": None, "quote_required": False},
        }
    # Configuration is a presentation prerequisite, never execution authority.
    # The service still verifies the session, exact offer, and explicit approval.
    if not hosted_profile_configured and route != "NOT_ELIGIBLE":
        execution_options["hosted"]["available"] = False
        message = (
            "Local execution is free and recommended. " if route == "LOCAL_FREE_AVAILABLE" else
            "This plan requires Pro capabilities unavailable in public local OEL. "
        ) + "Hosted OEL is a closed alpha; access is not publicly available."
        next_action = "execute_locally" if route == "LOCAL_FREE_AVAILABLE" else "use_public_fallback_or_revise_plan"
        if route == "HOSTED_PRO_REQUIRED":
            recommendation = {"option": None, "reason": message, "requirement": True}
    decision = {
        "schema": ROUTE_DECISION_SCHEMA,
        "hosted_access": {
            "status": "closed_alpha",
            "public_registration_available": False,
            "profile_configured": bool(hosted_profile_configured),
            "execution_authorized": False,
        },
        "planning_result_sha256": canonical_sha256(result),
        "planning_status": result["status"],
        "execution_route": route,
        "disposition": disposition,
        "next_action": next_action,
        "message": message,
        "execution_options": execution_options,
        "recommendation": recommendation,
        "oel_execution_amount_usd": 0 if route == "LOCAL_FREE_AVAILABLE" else None,
        "hosted_preflight_required": execution_options["hosted"]["available"],
        "user_approval_required": execution_options["hosted"]["available"],
        "feedback_preview_available": result["status"] == "UNSUPPORTED",
    }
    decision["decision_sha256"] = canonical_sha256(decision)
    return decision


__all__ = ["ROUTE_DECISION_SCHEMA", "route_planning_result"]
