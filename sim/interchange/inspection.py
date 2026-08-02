from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import HANDOFF_MANIFEST_SCHEMA_ID, PRODUCT_ENVELOPE_SCHEMA_ID
from .validation import load_interchange_document, validate_document


def inspect_document(
    document: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Return a compact read-only summary plus complete validation diagnostics."""

    report = validate_document(document, source_path=source_path, verify_sources=verify_sources)
    schema_id = str(document.get("schema_id", "") or "")
    if schema_id == PRODUCT_ENVELOPE_SCHEMA_ID:
        quality = dict(document.get("quality", {}) or {})
        freshness = dict(document.get("freshness", {}) or {})
        producer = dict(document.get("producer", {}) or {})
        markings = dict(document.get("data_markings", {}) or {})
        summary = {
            "document_type": "product",
            "schema_id": schema_id,
            "schema_version": document.get("schema_version"),
            "product_kind": document.get("product_kind"),
            "product_id": document.get("product_id"),
            "created_utc": document.get("created_utc"),
            "producer": {
                "capability_id": producer.get("capability_id"),
                "oel_version": producer.get("oel_version"),
                "run_id": producer.get("run_id"),
            },
            "quality": {
                "disposition": quality.get("disposition"),
                "producer_status": quality.get("producer_status"),
                "warning_count": len(quality.get("warnings", []) or []),
                "non_claim_count": len(quality.get("non_claims", []) or []),
            },
            "freshness": {
                "integrity_status": freshness.get("integrity_status"),
                "age_status": freshness.get("age_status"),
                "reference_epoch_jd_utc": freshness.get("reference_epoch_jd_utc"),
            },
            "data_markings": {
                "scope": markings.get("scope"),
                "handling": markings.get("handling"),
                "approved_for_public_export": markings.get("approved_for_public_export"),
                "contains_customer_data": markings.get("contains_customer_data"),
                "contains_hidden_truth": markings.get("contains_hidden_truth"),
            },
        }
        if document.get("product_kind") == "oel.state_estimate":
            payload = dict(document.get("payload", {}) or {})
            state = dict(payload.get("state", {}) or {})
            epoch = dict(state.get("epoch", {}) or {})
            covariance = dict(payload.get("covariance", {}) or {})
            summary["state"] = {
                "object_id": dict(payload.get("object", {}) or {}).get("object_id"),
                "representation": state.get("representation"),
                "frame": state.get("frame"),
                "epoch_jd_utc": epoch.get("value"),
                "time_system": epoch.get("time_system"),
                "covariance_present": covariance.get("present"),
            }
        elif document.get("product_kind") == "oel.relative_state_estimate":
            payload = dict(document.get("payload", {}) or {})
            state = dict(payload.get("relative_state", {}) or {})
            epoch = dict(state.get("epoch", {}) or {})
            summary["relative_state"] = {
                "chief_id": dict(payload.get("chief", {}) or {}).get("object_id"),
                "deputy_id": dict(payload.get("deputy", {}) or {}).get("object_id"),
                "frame": state.get("frame"),
                "convention": state.get("convention"),
                "epoch_jd_utc": epoch.get("value"),
                "covariance_present": dict(payload.get("covariance", {}) or {}).get("present"),
            }
        elif document.get("product_kind") == "oel.scenario_patch":
            payload = dict(document.get("payload", {}) or {})
            source = dict(payload.get("source_scenario", {}) or {})
            patch = dict(payload.get("patch", {}) or {})
            selection = dict(payload.get("selection", {}) or {})
            summary["scenario_patch"] = {
                "patch_type": patch.get("patch_type"),
                "selection_id": selection.get("selection_id"),
                "selection_kind": selection.get("selection_kind"),
                "rank": selection.get("rank"),
                "source_scenario_name": source.get("scenario_name"),
                "source_scenario_sha256": source.get("sha256"),
                "operation_count": len(patch.get("operations", []) or []),
            }
    elif schema_id == HANDOFF_MANIFEST_SCHEMA_ID:
        adapter = dict(document.get("adapter", {}) or {})
        output = dict(document.get("output", {}) or {})
        summary = {
            "document_type": "manifest",
            "schema_id": schema_id,
            "schema_version": document.get("schema_version"),
            "manifest_id": document.get("manifest_id"),
            "created_utc": document.get("created_utc"),
            "source_product_ids": list(document.get("source_product_ids", []) or []),
            "adapter": adapter,
            "output": {
                "kind": output.get("kind"),
                "path": output.get("path"),
                "status": output.get("status"),
            },
            "execution_occurred": document.get("execution_occurred"),
            "recommended_next_action": document.get("recommended_next_action"),
        }
    else:
        summary = {
            "document_type": "unknown",
            "schema_id": schema_id,
            "schema_version": document.get("schema_version"),
        }
    summary["validation"] = report.to_dict()
    return summary


def inspect_path(path: str | Path, *, verify_sources: bool = True) -> dict[str, Any]:
    source = Path(path)
    document = load_interchange_document(source)
    result = inspect_document(document, source_path=source, verify_sources=verify_sources)
    result["source_path"] = str(source)
    return result
