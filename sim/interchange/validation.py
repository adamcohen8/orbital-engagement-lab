from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    AGE_STATUSES,
    DATA_SCOPES,
    HANDOFF_MANIFEST_SCHEMA_ID,
    HANDOFF_MANIFEST_SCHEMA_VERSION,
    INTEGRITY_STATUSES,
    PRODUCT_ENVELOPE_SCHEMA_ID,
    PRODUCT_ENVELOPE_SCHEMA_VERSION,
    QUALITY_DISPOSITIONS,
)
from .provenance import compute_manifest_id, compute_product_id, sha256_file

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRODUCT_ID_RE = re.compile(r"^oel\.[a-z0-9_.-]+:[0-9a-f]{64}$")
_ENVELOPE_FIELDS = {
    "schema_id",
    "schema_version",
    "product_kind",
    "product_id",
    "created_utc",
    "producer",
    "payload",
    "quality",
    "freshness",
    "provenance",
    "data_markings",
}


@dataclass(frozen=True)
class InterchangeValidationIssue:
    code: str
    path: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class InterchangeValidationReport:
    document_type: str
    schema_id: str
    schema_version: int | None
    identifier: str
    valid: bool
    promotable: bool
    issues: tuple[InterchangeValidationIssue, ...]

    @property
    def errors(self) -> tuple[InterchangeValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def blockers(self) -> tuple[InterchangeValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "blocker")

    @property
    def warnings(self) -> tuple[InterchangeValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "identifier": self.identifier,
            "valid": self.valid,
            "promotable": self.promotable,
            "issue_counts": {
                "errors": len(self.errors),
                "blockers": len(self.blockers),
                "warnings": len(self.warnings),
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_interchange_document(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"interchange document does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"interchange document is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("interchange document root must be a JSON object")
    return value


def validate_document(
    document: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    verify_sources: bool = True,
) -> InterchangeValidationReport:
    schema_id = str(document.get("schema_id", "") or "")
    if schema_id == PRODUCT_ENVELOPE_SCHEMA_ID:
        return validate_product(document, source_path=source_path, verify_sources=verify_sources)
    if schema_id == HANDOFF_MANIFEST_SCHEMA_ID:
        return _validate_manifest(document)
    issue = InterchangeValidationIssue(
        code="schema.unsupported",
        path="$.schema_id",
        message=(
            f"Unsupported schema_id {schema_id!r}. Expected {PRODUCT_ENVELOPE_SCHEMA_ID!r} "
            f"or {HANDOFF_MANIFEST_SCHEMA_ID!r}."
        ),
    )
    return InterchangeValidationReport(
        document_type="unknown",
        schema_id=schema_id,
        schema_version=_integer_or_none(document.get("schema_version")),
        identifier="",
        valid=False,
        promotable=False,
        issues=(issue,),
    )


def validate_product(
    document: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    verify_sources: bool = True,
) -> InterchangeValidationReport:
    issues: list[InterchangeValidationIssue] = []
    _closed_mapping(document, _ENVELOPE_FIELDS, "$", issues)
    _require_exact(document, "schema_id", PRODUCT_ENVELOPE_SCHEMA_ID, "$", issues)
    _require_exact(document, "schema_version", PRODUCT_ENVELOPE_SCHEMA_VERSION, "$", issues)
    product_kind = _required_string(document, "product_kind", "$", issues)
    product_id = _required_string(document, "product_id", "$", issues)
    created_utc = _required_string(document, "created_utc", "$", issues)
    if created_utc:
        _validate_utc(created_utc, "$.created_utc", issues)
    if product_id and (not _PRODUCT_ID_RE.fullmatch(product_id) or not product_id.startswith(f"{product_kind}:")):
        _error(issues, "identity.invalid", "$.product_id", "product_id must be '<product_kind>:<64 lowercase hex>'.")

    producer = _required_mapping(document, "producer", "$", issues)
    _closed_mapping(producer, {"capability_id", "oel_version", "run_id"}, "$.producer", issues)
    _required_string(producer, "capability_id", "$.producer", issues)
    _required_string(producer, "oel_version", "$.producer", issues)
    if "run_id" in producer and producer.get("run_id") is not None:
        _nonempty_string(producer.get("run_id"), "$.producer.run_id", issues)

    payload = _required_mapping(document, "payload", "$", issues)
    quality = _required_mapping(document, "quality", "$", issues)
    _validate_quality(quality, issues)
    freshness = _required_mapping(document, "freshness", "$", issues)
    _validate_freshness(freshness, issues)
    provenance = _required_mapping(document, "provenance", "$", issues)
    _validate_provenance(
        provenance,
        freshness=freshness,
        source_path=source_path,
        verify_sources=verify_sources,
        issues=issues,
    )
    markings = _required_mapping(document, "data_markings", "$", issues)
    _validate_markings(markings, issues)

    if product_kind == "oel.state_estimate":
        _validate_state_estimate(payload, provenance=provenance, issues=issues)
    elif product_kind == "oel.relative_state_estimate":
        _validate_relative_state_estimate(payload, provenance=provenance, issues=issues)
    elif product_kind == "oel.scenario_patch":
        _validate_scenario_patch(payload, provenance=provenance, issues=issues)
    elif product_kind == "oel.ogp_mean_element_product":
        _validate_ogp_mean_element_product(payload, provenance=provenance, issues=issues)
    elif product_kind == "oel.completed_run_state":
        _validate_completed_run_state(payload, provenance=provenance, issues=issues)
    elif product_kind:
        _error(
            issues,
            "product_kind.unsupported",
            "$.product_kind",
            f"No semantic validator is registered for {product_kind!r}.",
        )

    if not any(issue.severity == "error" for issue in issues):
        try:
            expected_id = compute_product_id(document)
        except (TypeError, ValueError) as exc:
            _error(issues, "identity.canonicalization_failed", "$", str(exc))
        else:
            if product_id != expected_id:
                _error(
                    issues,
                    "identity.mismatch",
                    "$.product_id",
                    f"Declared product_id does not match canonical identity; expected {expected_id}.",
                )

    disposition = str(quality.get("disposition", "") or "")
    if disposition and disposition != "accepted":
        _block(
            issues,
            f"quality.{disposition}",
            "$.quality.disposition",
            f"Disposition {disposition!r} is valid evidence but is not eligible for automatic promotion.",
        )
    integrity_status = str(freshness.get("integrity_status", "") or "")
    if integrity_status in {"stale", "not_evaluated"}:
        _block(
            issues,
            f"freshness.integrity_{integrity_status}",
            "$.freshness.integrity_status",
            f"Integrity status {integrity_status!r} blocks promotion.",
        )
    age_status = str(freshness.get("age_status", "") or "")
    if age_status == "stale":
        _block(
            issues,
            "freshness.age_stale",
            "$.freshness.age_status",
            "The named age policy failed; this product is not promotable for that policy.",
        )

    valid = not any(issue.severity == "error" for issue in issues)
    promotable = valid and not any(issue.severity == "blocker" for issue in issues)
    return InterchangeValidationReport(
        document_type="product",
        schema_id=str(document.get("schema_id", "") or ""),
        schema_version=_integer_or_none(document.get("schema_version")),
        identifier=product_id,
        valid=valid,
        promotable=promotable,
        issues=tuple(_deduplicate_issues(issues)),
    )


def _validate_quality(value: Mapping[str, Any], issues: list[InterchangeValidationIssue]) -> None:
    path = "$.quality"
    _closed_mapping(value, {"disposition", "producer_status", "gates", "warnings", "non_claims"}, path, issues)
    disposition = _required_string(value, "disposition", path, issues)
    if disposition and disposition not in QUALITY_DISPOSITIONS:
        _error(issues, "quality.disposition_unknown", f"{path}.disposition", f"Expected one of {QUALITY_DISPOSITIONS}.")
    _required_string(value, "producer_status", path, issues)
    _required_mapping(value, "gates", path, issues)
    _string_list(value, "warnings", path, issues)
    _string_list(value, "non_claims", path, issues)


def _validate_freshness(value: Mapping[str, Any], issues: list[InterchangeValidationIssue]) -> None:
    path = "$.freshness"
    _closed_mapping(
        value,
        {"integrity_status", "age_status", "reference_epoch_jd_utc", "evaluated_utc", "policy"},
        path,
        issues,
    )
    integrity = _required_string(value, "integrity_status", path, issues)
    if integrity and integrity not in INTEGRITY_STATUSES:
        _error(issues, "freshness.integrity_unknown", f"{path}.integrity_status", f"Expected one of {INTEGRITY_STATUSES}.")
    age = _required_string(value, "age_status", path, issues)
    if age and age not in AGE_STATUSES:
        _error(issues, "freshness.age_unknown", f"{path}.age_status", f"Expected one of {AGE_STATUSES}.")
    if value.get("reference_epoch_jd_utc") is not None:
        _finite_number(value.get("reference_epoch_jd_utc"), f"{path}.reference_epoch_jd_utc", issues)
    if value.get("evaluated_utc") is not None:
        evaluated = _nonempty_string(value.get("evaluated_utc"), f"{path}.evaluated_utc", issues)
        if evaluated:
            _validate_utc(evaluated, f"{path}.evaluated_utc", issues)
    policy = _required_mapping(value, "policy", path, issues)
    if age in {"current", "stale"} and not policy:
        _error(
            issues,
            "freshness.policy_missing",
            f"{path}.policy",
            f"age_status {age!r} requires the named consumer policy and its result.",
        )


def _validate_provenance(
    value: Mapping[str, Any],
    *,
    freshness: Mapping[str, Any],
    source_path: str | Path | None,
    verify_sources: bool,
    issues: list[InterchangeValidationIssue],
) -> None:
    path = "$.provenance"
    _closed_mapping(value, {"source_artifacts", "source_product_ids", "transformations"}, path, issues)
    artifacts = _required_sequence(value, "source_artifacts", path, issues)
    source_product_ids = _required_sequence(value, "source_product_ids", path, issues)
    transformations = _required_sequence(value, "transformations", path, issues)
    for index, product_id in enumerate(source_product_ids):
        if not isinstance(product_id, str) or not _PRODUCT_ID_RE.fullmatch(product_id):
            _error(
                issues,
                "provenance.source_product_id_invalid",
                f"{path}.source_product_ids[{index}]",
                "Source product IDs must use '<product_kind>:<64 lowercase hex>'.",
            )
    for index, transformation in enumerate(transformations):
        item_path = f"{path}.transformations[{index}]"
        if not isinstance(transformation, Mapping):
            _error(issues, "type.mapping", item_path, "Transformation records must be objects.")
            continue
        _closed_mapping(transformation, {"transformation_id", "version", "details"}, item_path, issues)
        _required_string(transformation, "transformation_id", item_path, issues)
        _required_string(transformation, "version", item_path, issues)
        _required_mapping(transformation, "details", item_path, issues)

    base = Path(source_path).resolve().parent if source_path is not None else None
    integrity = str(freshness.get("integrity_status", "") or "")
    compared = 0
    for index, artifact in enumerate(artifacts):
        item_path = f"{path}.source_artifacts[{index}]"
        if not isinstance(artifact, Mapping):
            _error(issues, "type.mapping", item_path, "Source artifact records must be objects.")
            continue
        _closed_mapping(artifact, {"artifact_id", "sha256", "path", "media_type", "size_bytes"}, item_path, issues)
        _required_string(artifact, "artifact_id", item_path, issues)
        digest = _required_string(artifact, "sha256", item_path, issues)
        if digest and not _SHA256_RE.fullmatch(digest):
            _error(issues, "provenance.sha256_invalid", f"{item_path}.sha256", "sha256 must be 64 lowercase hex.")
        raw_path = artifact.get("path")
        if raw_path is not None:
            raw_path = _nonempty_string(raw_path, f"{item_path}.path", issues)
        if artifact.get("media_type") is not None:
            _nonempty_string(artifact.get("media_type"), f"{item_path}.media_type", issues)
        if artifact.get("size_bytes") is not None:
            size = artifact.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                _error(issues, "provenance.size_invalid", f"{item_path}.size_bytes", "size_bytes must be nonnegative.")
        if not verify_sources or not raw_path:
            continue
        if base is None:
            _warn(
                issues,
                "provenance.source_base_missing",
                f"{item_path}.path",
                "Source path was declared but no product path was supplied for relative resolution.",
            )
            continue
        candidate = Path(raw_path)
        resolved = candidate if candidate.is_absolute() else base / candidate
        if not resolved.is_file():
            severity = "error" if integrity == "current" else "blocker"
            _add(
                issues,
                "provenance.source_missing",
                f"{item_path}.path",
                f"Source artifact does not exist: {resolved}",
                severity,
            )
            continue
        compared += 1
        actual = sha256_file(resolved)
        if digest and actual != digest:
            severity = "error" if integrity == "current" else "blocker"
            _add(
                issues,
                "provenance.source_hash_mismatch",
                f"{item_path}.sha256",
                f"Source hash mismatch for {resolved.name}; observed {actual}.",
                severity,
            )
    if not verify_sources and integrity == "current" and artifacts:
        _block(
            issues,
            "provenance.verification_skipped",
            path,
            "Source fingerprint comparison was skipped; the product cannot be promoted from this validation result.",
        )
    if verify_sources and integrity == "current" and artifacts and compared == 0:
        _block(
            issues,
            "provenance.not_verifiable",
            path,
            "Integrity is declared current, but no source artifact could be compared in this inspection context.",
        )


def _validate_markings(value: Mapping[str, Any], issues: list[InterchangeValidationIssue]) -> None:
    path = "$.data_markings"
    _closed_mapping(
        value,
        {
            "scope",
            "handling",
            "approved_for_public_export",
            "contains_customer_data",
            "contains_hidden_truth",
            "source_markings",
        },
        path,
        issues,
    )
    scope = _required_string(value, "scope", path, issues)
    if scope and scope not in DATA_SCOPES:
        _error(issues, "markings.scope_unknown", f"{path}.scope", f"Expected one of {DATA_SCOPES}.")
    _required_string(value, "handling", path, issues)
    for field in ("approved_for_public_export", "contains_customer_data", "contains_hidden_truth"):
        if field not in value or not isinstance(value.get(field), bool):
            _error(issues, "type.boolean", f"{path}.{field}", f"{field} must be a boolean.")
    if "source_markings" in value:
        _required_sequence(value, "source_markings", path, issues)


def _validate_state_estimate(
    payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    issues: list[InterchangeValidationIssue],
) -> None:
    path = "$.payload"
    _closed_mapping(
        payload,
        {"object", "state", "covariance", "object_specs", "model_assumptions", "estimator_evidence"},
        path,
        issues,
    )
    obj = _required_mapping(payload, "object", path, issues)
    _closed_mapping(obj, {"object_id", "role", "kind"}, f"{path}.object", issues)
    _required_string(obj, "object_id", f"{path}.object", issues)
    _required_string(obj, "role", f"{path}.object", issues)
    if _required_string(obj, "kind", f"{path}.object", issues) != "satellite":
        _error(issues, "state.object_kind_incompatible", f"{path}.object.kind", "State Estimate v1 requires kind 'satellite'.")

    state = _required_mapping(payload, "state", path, issues)
    state_path = f"{path}.state"
    _closed_mapping(state, {"representation", "frame", "epoch", "component_order", "units", "values"}, state_path, issues)
    if _required_string(state, "representation", state_path, issues) != "cartesian_position_velocity":
        _error(issues, "state.representation_incompatible", f"{state_path}.representation", "Expected cartesian_position_velocity.")
    frame = _required_string(state, "frame", state_path, issues)
    if frame and frame != "ECI":
        _error(
            issues,
            "state.frame_incompatible",
            f"{state_path}.frame",
            f"State Estimate v1 requires canonical ECI; {frame!r} needs an explicit named transformation.",
        )
    epoch = _required_mapping(state, "epoch", state_path, issues)
    _closed_mapping(epoch, {"value", "format", "time_system"}, f"{state_path}.epoch", issues)
    epoch_value = _finite_number(epoch.get("value"), f"{state_path}.epoch.value", issues)
    if epoch_value is not None and epoch_value <= 0.0:
        _error(issues, "state.epoch_invalid", f"{state_path}.epoch.value", "Julian date must be positive.")
    if _required_string(epoch, "format", f"{state_path}.epoch", issues) != "jd":
        _error(issues, "state.epoch_format_incompatible", f"{state_path}.epoch.format", "Expected 'jd'.")
    if _required_string(epoch, "time_system", f"{state_path}.epoch", issues) != "UTC":
        _error(issues, "state.time_system_incompatible", f"{state_path}.epoch.time_system", "Expected 'UTC'.")
    expected_order = ["x", "y", "z", "vx", "vy", "vz"]
    expected_units = ["km", "km", "km", "km/s", "km/s", "km/s"]
    _exact_string_sequence(state.get("component_order"), expected_order, f"{state_path}.component_order", issues)
    _exact_string_sequence(state.get("units"), expected_units, f"{state_path}.units", issues)
    _finite_vector(state.get("values"), 6, f"{state_path}.values", issues)

    covariance = _required_mapping(payload, "covariance", path, issues)
    _validate_covariance(covariance, state_frame=frame, state_epoch=epoch_value, issues=issues)
    _required_mapping(payload, "object_specs", path, issues)
    assumptions = _required_mapping(payload, "model_assumptions", path, issues)
    _closed_mapping(assumptions, {"orbit_force_model", "attitude"}, f"{path}.model_assumptions", issues)
    _required_mapping(assumptions, "orbit_force_model", f"{path}.model_assumptions", issues)
    _required_mapping(assumptions, "attitude", f"{path}.model_assumptions", issues)

    evidence = _required_mapping(payload, "estimator_evidence", path, issues)
    evidence_path = f"{path}.estimator_evidence"
    _closed_mapping(evidence, {"method", "selected_parameters", "od_solution_id", "source_report_sha256"}, evidence_path, issues)
    _required_string(evidence, "method", evidence_path, issues)
    _string_list(evidence, "selected_parameters", evidence_path, issues)
    if evidence.get("od_solution_id") is not None:
        _nonempty_string(evidence.get("od_solution_id"), f"{evidence_path}.od_solution_id", issues)
    report_hash = _required_string(evidence, "source_report_sha256", evidence_path, issues)
    if report_hash and not _SHA256_RE.fullmatch(report_hash):
        _error(issues, "state.source_report_hash_invalid", f"{evidence_path}.source_report_sha256", "Expected 64 lowercase hex.")
    artifact_hashes = {
        str(item.get("sha256", "") or "")
        for item in provenance.get("source_artifacts", [])
        if isinstance(item, Mapping)
    }
    if report_hash and report_hash not in artifact_hashes:
        _error(
            issues,
            "state.source_report_unbound",
            f"{evidence_path}.source_report_sha256",
            "source_report_sha256 must match a provenance source artifact hash.",
        )


def _validate_ogp_mean_element_product(
    payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    issues: list[InterchangeValidationIssue],
) -> None:
    path = "$.payload"
    _closed_mapping(payload, {"object", "mean_elements", "propagation", "estimator_evidence"}, path, issues)
    obj = _required_mapping(payload, "object", path, issues)
    _closed_mapping(
        obj,
        {"object_id", "kind", "norad_number", "international_designator"},
        f"{path}.object",
        issues,
    )
    _required_string(obj, "object_id", f"{path}.object", issues)
    _require_exact(obj, "kind", "satellite", f"{path}.object", issues)

    state = _required_mapping(payload, "mean_elements", path, issues)
    state_path = f"{path}.mean_elements"
    _closed_mapping(state, {"representation", "frame", "epoch", "values"}, state_path, issues)
    _require_exact(state, "representation", "ogp_mean_elements", state_path, issues)
    _require_exact(state, "frame", "TEME", state_path, issues)
    epoch = _required_mapping(state, "epoch", state_path, issues)
    _closed_mapping(epoch, {"value", "format", "time_system"}, f"{state_path}.epoch", issues)
    epoch_value = _finite_number(epoch.get("value"), f"{state_path}.epoch.value", issues)
    if epoch_value is not None and epoch_value <= 0.0:
        _error(issues, "ogp.epoch_invalid", f"{state_path}.epoch.value", "Julian date must be positive.")
    _require_exact(epoch, "format", "jd", f"{state_path}.epoch", issues)
    _require_exact(epoch, "time_system", "UTC", f"{state_path}.epoch", issues)
    values = _required_mapping(state, "values", state_path, issues)
    required = (
        "epoch_jd_utc", "inclination_deg", "raan_deg", "eccentricity", "argp_deg",
        "mean_anomaly_deg", "mean_motion_rev_per_day",
    )
    for field in required:
        _finite_number(values.get(field), f"{state_path}.values.{field}", issues)
    eccentricity = values.get("eccentricity")
    if isinstance(eccentricity, (int, float)) and not 0.0 <= float(eccentricity) < 1.0:
        _error(issues, "ogp.eccentricity_invalid", f"{state_path}.values.eccentricity", "Expected [0, 1).")
    mean_motion = values.get("mean_motion_rev_per_day")
    if isinstance(mean_motion, (int, float)) and float(mean_motion) <= 0.0:
        _error(issues, "ogp.mean_motion_invalid", f"{state_path}.values.mean_motion_rev_per_day", "Must be positive.")
    if "line1" in values or "line2" in values:
        _error(issues, "ogp.tle_text_forbidden", f"{state_path}.values", "Native fitted elements must not contain TLE text.")

    propagation = _required_mapping(payload, "propagation", path, issues)
    _closed_mapping(propagation, {"family", "regime", "propagator_name", "not_tle_text"}, f"{path}.propagation", issues)
    _require_exact(propagation, "family", "OGP", f"{path}.propagation", issues)
    if _required_string(propagation, "regime", f"{path}.propagation", issues) not in {"sgp4", "sdp4"}:
        _error(issues, "ogp.regime_invalid", f"{path}.propagation.regime", "Expected sgp4 or sdp4.")
    _required_string(propagation, "propagator_name", f"{path}.propagation", issues)
    if propagation.get("not_tle_text") is not True:
        _error(issues, "ogp.tle_claim_invalid", f"{path}.propagation.not_tle_text", "Must be true.")

    evidence = _required_mapping(payload, "estimator_evidence", path, issues)
    _closed_mapping(evidence, {"method", "parameterization", "selected_parameters", "fit", "holdout", "source_report_sha256"}, f"{path}.estimator_evidence", issues)
    _required_string(evidence, "method", f"{path}.estimator_evidence", issues)
    _required_string(evidence, "parameterization", f"{path}.estimator_evidence", issues)
    _string_list(evidence, "selected_parameters", f"{path}.estimator_evidence", issues)
    _required_mapping(evidence, "fit", f"{path}.estimator_evidence", issues)
    _required_mapping(evidence, "holdout", f"{path}.estimator_evidence", issues)
    report_hash = _required_string(evidence, "source_report_sha256", f"{path}.estimator_evidence", issues)
    artifact_hashes = {
        str(item.get("sha256", "") or "")
        for item in provenance.get("source_artifacts", [])
        if isinstance(item, Mapping)
    }
    if report_hash and report_hash not in artifact_hashes:
        _error(issues, "ogp.source_report_unbound", f"{path}.estimator_evidence.source_report_sha256", "Source report hash is not bound in provenance.")


def _validate_completed_run_state(
    payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    issues: list[InterchangeValidationIssue],
) -> None:
    path = "$.payload"
    _closed_mapping(
        payload,
        {"object", "state", "covariance", "object_specs", "model_assumptions", "source_run", "selection"},
        path,
        issues,
    )
    obj = _required_mapping(payload, "object", path, issues)
    _closed_mapping(obj, {"object_id", "role", "kind"}, f"{path}.object", issues)
    _required_string(obj, "object_id", f"{path}.object", issues)
    _required_string(obj, "role", f"{path}.object", issues)
    if _required_string(obj, "kind", f"{path}.object", issues) != "satellite":
        _error(
            issues,
            "continuation.object_kind_incompatible",
            f"{path}.object.kind",
            "Completed Run State v1 requires kind 'satellite'.",
        )

    state = _required_mapping(payload, "state", path, issues)
    state_path = f"{path}.state"
    _closed_mapping(state, {"representation", "frame", "epoch", "component_order", "units", "values"}, state_path, issues)
    _require_exact(state, "representation", "cartesian_position_velocity", state_path, issues)
    frame = _required_string(state, "frame", state_path, issues)
    if frame and frame != "ECI":
        _error(
            issues,
            "continuation.frame_incompatible",
            f"{state_path}.frame",
            "Completed Run State v1 requires canonical ECI evidence.",
        )
    epoch = _required_mapping(state, "epoch", state_path, issues)
    _closed_mapping(epoch, {"value", "format", "time_system"}, f"{state_path}.epoch", issues)
    epoch_value = _finite_number(epoch.get("value"), f"{state_path}.epoch.value", issues)
    if epoch_value is not None and epoch_value <= 0.0:
        _error(issues, "continuation.epoch_invalid", f"{state_path}.epoch.value", "Julian date must be positive.")
    _require_exact(epoch, "format", "jd", f"{state_path}.epoch", issues)
    _require_exact(epoch, "time_system", "UTC", f"{state_path}.epoch", issues)
    expected_order = ["x", "y", "z", "vx", "vy", "vz"]
    expected_units = ["km", "km", "km", "km/s", "km/s", "km/s"]
    _exact_string_sequence(state.get("component_order"), expected_order, f"{state_path}.component_order", issues)
    _exact_string_sequence(state.get("units"), expected_units, f"{state_path}.units", issues)
    _finite_vector(state.get("values"), 6, f"{state_path}.values", issues)

    covariance = _required_mapping(payload, "covariance", path, issues)
    _validate_covariance(covariance, state_frame=frame, state_epoch=epoch_value, issues=issues)
    _required_mapping(payload, "object_specs", path, issues)
    assumptions = _required_mapping(payload, "model_assumptions", path, issues)
    _closed_mapping(assumptions, {"orbit_force_model", "attitude"}, f"{path}.model_assumptions", issues)
    _required_mapping(assumptions, "orbit_force_model", f"{path}.model_assumptions", issues)
    _required_mapping(assumptions, "attitude", f"{path}.model_assumptions", issues)

    source_run = _required_mapping(payload, "source_run", path, issues)
    source_path = f"{path}.source_run"
    _closed_mapping(
        source_run,
        {
            "run_id",
            "scenario_name",
            "review_schema_version",
            "generated_utc",
            "config_sha256",
            "review_db_sha256",
            "initial_jd_utc",
        },
        source_path,
        issues,
    )
    _required_string(source_run, "run_id", source_path, issues)
    _required_string(source_run, "scenario_name", source_path, issues)
    _required_string(source_run, "review_schema_version", source_path, issues)
    generated = _required_string(source_run, "generated_utc", source_path, issues)
    if generated:
        _validate_utc(generated, f"{source_path}.generated_utc", issues)
    config_hash = _required_string(source_run, "config_sha256", source_path, issues)
    review_hash = _required_string(source_run, "review_db_sha256", source_path, issues)
    for field, value in (("config_sha256", config_hash), ("review_db_sha256", review_hash)):
        if value and not _SHA256_RE.fullmatch(value):
            _error(issues, "continuation.hash_invalid", f"{source_path}.{field}", "Expected 64 lowercase hex.")
    artifact_hashes = {
        str(item.get("sha256", "") or "")
        for item in provenance.get("source_artifacts", [])
        if isinstance(item, Mapping)
    }
    if review_hash and review_hash not in artifact_hashes:
        _error(
            issues,
            "continuation.review_store_unbound",
            f"{source_path}.review_db_sha256",
            "review_db_sha256 must match a provenance source artifact hash.",
        )
    initial_jd = _finite_number(source_run.get("initial_jd_utc"), f"{source_path}.initial_jd_utc", issues)
    if initial_jd is not None and initial_jd <= 0.0:
        _error(
            issues,
            "continuation.initial_epoch_invalid",
            f"{source_path}.initial_jd_utc",
            "Initial Julian date must be positive.",
        )

    selection = _required_mapping(payload, "selection", path, issues)
    selection_path = f"{path}.selection"
    _closed_mapping(
        selection,
        {"selector_kind", "requested", "sample_index", "time_s", "state_row_sha256", "associated_event"},
        selection_path,
        issues,
    )
    selector_kind = _required_string(selection, "selector_kind", selection_path, issues)
    if selector_kind and selector_kind not in {"final", "sample_index", "time_s", "event"}:
        _error(
            issues,
            "continuation.selector_unsupported",
            f"{selection_path}.selector_kind",
            "Expected final, sample_index, time_s, or event.",
        )
    requested = _required_mapping(selection, "requested", selection_path, issues)
    expected_request_fields = {
        "final": {"sample"},
        "sample_index": {"sample_index"},
        "time_s": {"time_s"},
        "event": {"event_id"},
    }.get(selector_kind, set())
    _closed_mapping(requested, expected_request_fields, f"{selection_path}.requested", issues)
    if selector_kind == "final":
        _require_exact(requested, "sample", "final", f"{selection_path}.requested", issues)
    elif selector_kind == "sample_index":
        requested_index = requested.get("sample_index")
        if isinstance(requested_index, bool) or not isinstance(requested_index, int) or requested_index < 0:
            _error(
                issues,
                "continuation.requested_sample_invalid",
                f"{selection_path}.requested.sample_index",
                "Requested sample_index must be a non-negative integer.",
            )
    elif selector_kind == "time_s":
        requested_time = _finite_number(requested.get("time_s"), f"{selection_path}.requested.time_s", issues)
        if requested_time is not None and requested_time < 0.0:
            _error(
                issues,
                "continuation.requested_time_invalid",
                f"{selection_path}.requested.time_s",
                "Requested time_s must be non-negative.",
            )
    elif selector_kind == "event":
        _required_string(requested, "event_id", f"{selection_path}.requested", issues)
    selected_index = selection.get("sample_index")
    if isinstance(selected_index, bool) or not isinstance(selected_index, int) or selected_index < 0:
        _error(
            issues,
            "continuation.sample_invalid",
            f"{selection_path}.sample_index",
            "Selected sample_index must be a non-negative integer.",
        )
    selected_time = _finite_number(selection.get("time_s"), f"{selection_path}.time_s", issues)
    if selected_time is not None and selected_time < 0.0:
        _error(issues, "continuation.time_invalid", f"{selection_path}.time_s", "Selected time_s must be non-negative.")
    state_hash = _required_string(selection, "state_row_sha256", selection_path, issues)
    if state_hash and not _SHA256_RE.fullmatch(state_hash):
        _error(
            issues,
            "continuation.state_hash_invalid",
            f"{selection_path}.state_row_sha256",
            "Expected 64 lowercase hex.",
        )
    associated_event = selection.get("associated_event")
    if selector_kind == "event":
        if not isinstance(associated_event, Mapping):
            _error(
                issues,
                "continuation.event_missing",
                f"{selection_path}.associated_event",
                "Event selection requires associated event evidence.",
            )
        else:
            event_path = f"{selection_path}.associated_event"
            _closed_mapping(
                associated_event,
                {
                    "event_id",
                    "time_s",
                    "sample_index",
                    "object_id",
                    "event_type",
                    "severity",
                    "message",
                    "source",
                    "event_row_sha256",
                },
                event_path,
                issues,
            )
            _required_string(associated_event, "event_id", event_path, issues)
            event_hash = _required_string(associated_event, "event_row_sha256", event_path, issues)
            if event_hash and not _SHA256_RE.fullmatch(event_hash):
                _error(issues, "continuation.event_hash_invalid", f"{event_path}.event_row_sha256", "Expected 64 lowercase hex.")
            if associated_event.get("sample_index") != selected_index:
                _error(
                    issues,
                    "continuation.event_sample_mismatch",
                    f"{event_path}.sample_index",
                    "Associated event and selected state must share sample_index.",
                )
            event_time = _finite_number(associated_event.get("time_s"), f"{event_path}.time_s", issues)
            if event_time is not None and selected_time is not None and not math.isclose(
                event_time, selected_time, rel_tol=0.0, abs_tol=1.0e-9
            ):
                _error(
                    issues,
                    "continuation.event_time_mismatch",
                    f"{event_path}.time_s",
                    "Associated event and selected state must share time_s.",
                )
    elif associated_event is not None:
        _error(
            issues,
            "continuation.event_unexpected",
            f"{selection_path}.associated_event",
            "Only the event selector may carry associated event evidence.",
        )
    if initial_jd is not None and selected_time is not None and epoch_value is not None:
        expected_epoch = initial_jd + selected_time / 86400.0
        if not math.isclose(epoch_value, expected_epoch, rel_tol=0.0, abs_tol=1.0e-12):
            _error(
                issues,
                "continuation.epoch_derivation_mismatch",
                f"{state_path}.epoch.value",
                "State epoch must equal initial_jd_utc + selected time_s / 86400.",
            )


def _validate_covariance(
    value: Mapping[str, Any],
    *,
    state_frame: str,
    state_epoch: float | None,
    issues: list[InterchangeValidationIssue],
) -> None:
    path = "$.payload.covariance"
    present = value.get("present")
    if not isinstance(present, bool):
        _error(issues, "type.boolean", f"{path}.present", "present must be a boolean.")
        return
    if not present:
        _closed_mapping(value, {"present", "reason"}, path, issues)
        _required_string(value, "reason", path, issues)
        return
    allowed = {
        "present",
        "frame",
        "epoch_jd_utc",
        "component_order",
        "units",
        "matrix",
        "mathematically_valid",
        "calibrated",
        "calibration_scope",
    }
    _closed_mapping(value, allowed, path, issues)
    frame = _required_string(value, "frame", path, issues)
    if frame and frame != state_frame:
        _error(issues, "covariance.frame_mismatch", f"{path}.frame", "Covariance and state frames must match in v1.")
    epoch = _finite_number(value.get("epoch_jd_utc"), f"{path}.epoch_jd_utc", issues)
    if epoch is not None and state_epoch is not None and epoch != state_epoch:
        _error(issues, "covariance.epoch_mismatch", f"{path}.epoch_jd_utc", "Covariance and state epochs must match in v1.")
    expected_order = ["x", "y", "z", "vx", "vy", "vz"]
    expected_units = ["km", "km", "km", "km/s", "km/s", "km/s"]
    _exact_string_sequence(value.get("component_order"), expected_order, f"{path}.component_order", issues)
    _exact_string_sequence(value.get("units"), expected_units, f"{path}.units", issues)
    matrix = value.get("matrix")
    array: np.ndarray | None = None
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        _error(issues, "covariance.matrix_invalid", f"{path}.matrix", "Covariance matrix must contain finite numbers.")
    if array is not None:
        if array.shape != (6, 6):
            _error(issues, "covariance.shape_invalid", f"{path}.matrix", "Covariance matrix must be 6x6.")
        elif not np.all(np.isfinite(array)):
            _error(issues, "covariance.nonfinite", f"{path}.matrix", "Covariance matrix must contain finite numbers.")
        elif not np.allclose(array, array.T, rtol=0.0, atol=1.0e-12):
            _error(issues, "covariance.not_symmetric", f"{path}.matrix", "Covariance matrix must be symmetric.")
        elif float(np.min(np.linalg.eigvalsh(array))) < -1.0e-12:
            _error(issues, "covariance.not_psd", f"{path}.matrix", "Covariance matrix must be positive semidefinite.")
    for field in ("mathematically_valid", "calibrated"):
        if not isinstance(value.get(field), bool):
            _error(issues, "type.boolean", f"{path}.{field}", f"{field} must be a boolean.")
    if value.get("mathematically_valid") is not True:
        _block(issues, "covariance.producer_invalid", f"{path}.mathematically_valid", "Producer did not mark covariance mathematically valid.")
    calibration_scope = value.get("calibration_scope")
    if calibration_scope is not None and not isinstance(calibration_scope, str):
        _error(issues, "type.string", f"{path}.calibration_scope", "calibration_scope must be a string or null.")
    if value.get("calibrated") is True and not calibration_scope:
        _error(issues, "covariance.calibration_scope_missing", f"{path}.calibration_scope", "A calibrated covariance requires a bounded calibration scope.")


def _validate_relative_state_estimate(
    payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    issues: list[InterchangeValidationIssue],
) -> None:
    path = "$.payload"
    _closed_mapping(
        payload,
        {"chief", "deputy", "relative_state", "covariance", "object_specs", "model_assumptions", "estimator_evidence"},
        path,
        issues,
    )
    chief = _required_mapping(payload, "chief", path, issues)
    _closed_mapping(chief, {"object_id", "state_product_id"}, f"{path}.chief", issues)
    chief_id = _required_string(chief, "object_id", f"{path}.chief", issues)
    chief_product_id = _required_string(chief, "state_product_id", f"{path}.chief", issues)
    if chief_product_id and not _PRODUCT_ID_RE.fullmatch(chief_product_id):
        _error(issues, "relative.chief_product_id_invalid", f"{path}.chief.state_product_id", "Invalid chief state product ID.")
    if chief_product_id and chief_product_id not in provenance.get("source_product_ids", []):
        _error(
            issues,
            "relative.chief_product_unbound",
            f"{path}.chief.state_product_id",
            "Chief state product ID must appear in provenance.source_product_ids.",
        )

    deputy = _required_mapping(payload, "deputy", path, issues)
    _closed_mapping(deputy, {"object_id", "role", "kind"}, f"{path}.deputy", issues)
    deputy_id = _required_string(deputy, "object_id", f"{path}.deputy", issues)
    _required_string(deputy, "role", f"{path}.deputy", issues)
    if _required_string(deputy, "kind", f"{path}.deputy", issues) != "satellite":
        _error(issues, "relative.deputy_kind_incompatible", f"{path}.deputy.kind", "Relative State Estimate v1 requires kind 'satellite'.")
    if chief_id and deputy_id and chief_id == deputy_id:
        _error(issues, "relative.object_identity_conflict", f"{path}.deputy.object_id", "Chief and deputy IDs must differ.")

    state = _required_mapping(payload, "relative_state", path, issues)
    state_path = f"{path}.relative_state"
    _closed_mapping(
        state,
        {"representation", "frame", "convention", "axes", "sign_convention", "epoch", "component_order", "units", "values"},
        state_path,
        issues,
    )
    if _required_string(state, "representation", state_path, issues) != "cartesian_relative_position_velocity":
        _error(issues, "relative.representation_incompatible", f"{state_path}.representation", "Expected cartesian_relative_position_velocity.")
    if _required_string(state, "frame", state_path, issues) != "RIC":
        _error(issues, "relative.frame_incompatible", f"{state_path}.frame", "Relative State Estimate v1 requires RIC.")
    if _required_string(state, "convention", state_path, issues) != "rectangular":
        _error(issues, "relative.convention_incompatible", f"{state_path}.convention", "Relative State Estimate v1 requires rectangular RIC.")
    axes = _required_mapping(state, "axes", state_path, issues)
    _closed_mapping(axes, {"radial", "intrack", "crosstrack"}, f"{state_path}.axes", issues)
    _require_exact(axes, "radial", "chief_position_outward", f"{state_path}.axes", issues)
    _require_exact(axes, "intrack", "chief_motion_direction", f"{state_path}.axes", issues)
    _require_exact(axes, "crosstrack", "right_handed_orbit_normal", f"{state_path}.axes", issues)
    _require_exact(
        state,
        "sign_convention",
        "deputy_minus_chief_expressed_in_chief_ric",
        state_path,
        issues,
    )
    epoch = _required_mapping(state, "epoch", state_path, issues)
    _closed_mapping(epoch, {"value", "format", "time_system"}, f"{state_path}.epoch", issues)
    epoch_value = _finite_number(epoch.get("value"), f"{state_path}.epoch.value", issues)
    if epoch_value is not None and epoch_value <= 0.0:
        _error(issues, "relative.epoch_invalid", f"{state_path}.epoch.value", "Julian date must be positive.")
    _require_exact(epoch, "format", "jd", f"{state_path}.epoch", issues)
    _require_exact(epoch, "time_system", "UTC", f"{state_path}.epoch", issues)
    order = ["r_radial", "i_intrack", "c_crosstrack", "vr_radial", "vi_intrack", "vc_crosstrack"]
    units = ["km", "km", "km", "km/s", "km/s", "km/s"]
    _exact_string_sequence(state.get("component_order"), order, f"{state_path}.component_order", issues)
    _exact_string_sequence(state.get("units"), units, f"{state_path}.units", issues)
    _finite_vector(state.get("values"), 6, f"{state_path}.values", issues)

    covariance = _required_mapping(payload, "covariance", path, issues)
    _validate_relative_covariance(covariance, state_epoch=epoch_value, issues=issues)
    _required_mapping(payload, "object_specs", path, issues)
    assumptions = _required_mapping(payload, "model_assumptions", path, issues)
    _closed_mapping(assumptions, {"relative_dynamics_model", "dynamics_metadata"}, f"{path}.model_assumptions", issues)
    model = _required_string(assumptions, "relative_dynamics_model", f"{path}.model_assumptions", issues)
    if model and model not in {"hcw", "ss_j2", "th", "ya"}:
        _error(issues, "relative.dynamics_model_incompatible", f"{path}.model_assumptions.relative_dynamics_model", "Unsupported relative dynamics model.")
    _required_mapping(assumptions, "dynamics_metadata", f"{path}.model_assumptions", issues)
    evidence = _required_mapping(payload, "estimator_evidence", path, issues)
    _closed_mapping(evidence, {"method", "source_report_sha256"}, f"{path}.estimator_evidence", issues)
    _required_string(evidence, "method", f"{path}.estimator_evidence", issues)
    report_hash = _required_string(evidence, "source_report_sha256", f"{path}.estimator_evidence", issues)
    artifact_hashes = {
        str(item.get("sha256", "") or "")
        for item in provenance.get("source_artifacts", [])
        if isinstance(item, Mapping)
    }
    if report_hash and report_hash not in artifact_hashes:
        _error(issues, "relative.source_report_unbound", f"{path}.estimator_evidence.source_report_sha256", "Source report hash is not bound in provenance.")


def _validate_relative_covariance(
    value: Mapping[str, Any], *, state_epoch: float | None, issues: list[InterchangeValidationIssue]
) -> None:
    path = "$.payload.covariance"
    present = value.get("present")
    if not isinstance(present, bool):
        _error(issues, "type.boolean", f"{path}.present", "present must be a boolean.")
        return
    if not present:
        _closed_mapping(value, {"present", "reason"}, path, issues)
        _required_string(value, "reason", path, issues)
        return
    allowed = {
        "present", "frame", "convention", "epoch_jd_utc", "component_order", "units", "matrix",
        "mathematically_valid", "calibrated", "calibration_scope",
    }
    _closed_mapping(value, allowed, path, issues)
    _require_exact(value, "frame", "RIC", path, issues)
    _require_exact(value, "convention", "rectangular", path, issues)
    epoch = _finite_number(value.get("epoch_jd_utc"), f"{path}.epoch_jd_utc", issues)
    if epoch is not None and state_epoch is not None and epoch != state_epoch:
        _error(issues, "relative.covariance_epoch_mismatch", f"{path}.epoch_jd_utc", "Relative covariance and state epochs must match.")
    order = ["r_radial", "i_intrack", "c_crosstrack", "vr_radial", "vi_intrack", "vc_crosstrack"]
    units = ["km", "km", "km", "km/s", "km/s", "km/s"]
    _exact_string_sequence(value.get("component_order"), order, f"{path}.component_order", issues)
    _exact_string_sequence(value.get("units"), units, f"{path}.units", issues)
    try:
        matrix = np.asarray(value.get("matrix"), dtype=float)
    except (TypeError, ValueError):
        matrix = np.empty((0, 0))
        _error(issues, "relative.covariance_invalid", f"{path}.matrix", "Relative covariance must contain finite numbers.")
    if matrix.shape != (6, 6):
        _error(issues, "relative.covariance_shape_invalid", f"{path}.matrix", "Relative covariance must be 6x6.")
    elif not np.all(np.isfinite(matrix)):
        _error(issues, "relative.covariance_nonfinite", f"{path}.matrix", "Relative covariance must be finite.")
    elif not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        _error(issues, "relative.covariance_not_symmetric", f"{path}.matrix", "Relative covariance must be symmetric.")
    elif float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-12:
        _error(issues, "relative.covariance_not_psd", f"{path}.matrix", "Relative covariance must be positive semidefinite.")
    for field in ("mathematically_valid", "calibrated"):
        if not isinstance(value.get(field), bool):
            _error(issues, "type.boolean", f"{path}.{field}", f"{field} must be a boolean.")
    if value.get("mathematically_valid") is not True:
        _block(issues, "relative.covariance_producer_invalid", f"{path}.mathematically_valid", "Producer did not mark relative covariance mathematically valid.")
    scope = value.get("calibration_scope")
    if scope is not None and not isinstance(scope, str):
        _error(issues, "type.string", f"{path}.calibration_scope", "calibration_scope must be a string or null.")
    if value.get("calibrated") is True and not scope:
        _error(issues, "relative.covariance_scope_missing", f"{path}.calibration_scope", "Calibrated covariance requires a scope.")


def _validate_scenario_patch(
    payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    issues: list[InterchangeValidationIssue],
) -> None:
    path = "$.payload"
    _closed_mapping(payload, {"source_scenario", "patch", "selection", "evidence"}, path, issues)
    source = _required_mapping(payload, "source_scenario", path, issues)
    source_path = f"{path}.source_scenario"
    _closed_mapping(source, {"scenario_name", "sha256", "canonical_digest"}, source_path, issues)
    _required_string(source, "scenario_name", source_path, issues)
    source_hash = _required_string(source, "sha256", source_path, issues)
    source_digest = _required_string(source, "canonical_digest", source_path, issues)
    for field, value in (("sha256", source_hash), ("canonical_digest", source_digest)):
        if value and not _SHA256_RE.fullmatch(value):
            _error(issues, "patch.hash_invalid", f"{source_path}.{field}", f"{field} must be 64 lowercase hex.")
    artifact_hashes = {
        str(item.get("sha256", "") or "")
        for item in provenance.get("source_artifacts", [])
        if isinstance(item, Mapping)
    }
    if source_hash and source_hash not in artifact_hashes:
        _error(issues, "patch.source_unbound", f"{source_path}.sha256", "Source scenario hash must match a provenance source artifact.")

    patch = _required_mapping(payload, "patch", path, issues)
    patch_path = f"{path}.patch"
    _closed_mapping(patch, {"patch_type", "operations"}, patch_path, issues)
    patch_type = _required_string(patch, "patch_type", patch_path, issues)
    if patch_type and patch_type not in {"mission_recovery_candidate", "controller_optimized_variant"}:
        _error(issues, "patch.type_unsupported", f"{patch_path}.patch_type", "Unsupported scenario patch type.")
    operations = _required_sequence(patch, "operations", patch_path, issues)
    if not operations:
        _error(issues, "patch.operations_empty", f"{patch_path}.operations", "At least one typed operation is required.")
    for index, operation in enumerate(operations):
        operation_path = f"{patch_path}.operations[{index}]"
        if not isinstance(operation, Mapping):
            _error(issues, "type.mapping", operation_path, "Patch operations must be objects.")
            continue
        _closed_mapping(operation, {"op", "kind", "path", "value", "reason"}, operation_path, issues)
        op = _required_string(operation, "op", operation_path, issues)
        kind = _required_string(operation, "kind", operation_path, issues)
        dotted = _required_string(operation, "path", operation_path, issues)
        _required_string(operation, "reason", operation_path, issues)
        if "value" not in operation:
            _error(issues, "patch.value_missing", f"{operation_path}.value", "Patch operation value is required.")
        if op not in {"replace", "append"}:
            _error(issues, "patch.operation_unsupported", f"{operation_path}.op", "Only replace and append are supported.")
        if kind not in {"mission_burn", "duration_extension", "controller_pointer", "scenario_override"}:
            _error(issues, "patch.operation_kind_unsupported", f"{operation_path}.kind", "Unsupported typed operation kind.")
        if dotted:
            _validate_patch_path(op=op, kind=kind, path=dotted, issue_path=f"{operation_path}.path", issues=issues)
        _validate_patch_operation_value(
            patch_type=patch_type,
            kind=kind,
            value=operation.get("value"),
            issue_path=f"{operation_path}.value",
            issues=issues,
        )

    selection = _required_mapping(payload, "selection", path, issues)
    selection_path = f"{path}.selection"
    _closed_mapping(
        selection,
        {"selection_id", "selection_kind", "rank", "recommended_modes", "case_name", "variant_name"},
        selection_path,
        issues,
    )
    _required_string(selection, "selection_id", selection_path, issues)
    selection_kind = _required_string(selection, "selection_kind", selection_path, issues)
    if selection_kind and selection_kind not in {"mission_recovery_candidate", "controller_optimized_variant"}:
        _error(issues, "patch.selection_kind_unsupported", f"{selection_path}.selection_kind", "Unsupported selection kind.")
    if patch_type and selection_kind and selection_kind != patch_type:
        _error(
            issues,
            "patch.selection_kind_mismatch",
            f"{selection_path}.selection_kind",
            "selection_kind must match patch_type.",
        )
    rank = selection.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        _error(issues, "patch.rank_invalid", f"{selection_path}.rank", "Selection rank must be a positive integer.")
    _string_list(selection, "recommended_modes", selection_path, issues)
    for optional in ("case_name", "variant_name"):
        if selection.get(optional) is not None:
            _nonempty_string(selection.get(optional), f"{selection_path}.{optional}", issues)

    evidence = _required_mapping(payload, "evidence", path, issues)
    evidence_path = f"{path}.evidence"
    _closed_mapping(evidence, {"objective", "constraints", "producer_evidence"}, evidence_path, issues)
    _required_mapping(evidence, "objective", evidence_path, issues)
    _required_mapping(evidence, "constraints", evidence_path, issues)
    _required_mapping(evidence, "producer_evidence", evidence_path, issues)


def _validate_patch_path(
    *, op: str, kind: str, path: str, issue_path: str, issues: list[InterchangeValidationIssue]
) -> None:
    tokens = [token for token in path.split(".") if token]
    valid = False
    if kind == "mission_burn":
        valid = op == "append" and len(tokens) == 3 and tokens[0] == "objects" and tokens[2] == "mission_objectives"
    elif kind == "duration_extension":
        valid = op == "replace" and tokens == ["simulator", "duration_s"]
    elif kind == "controller_pointer":
        valid = (
            op == "replace"
            and len(tokens) == 3
            and tokens[0] == "objects"
            and tokens[2] in {"orbit_control", "attitude_control", "base_guidance"}
        )
    elif kind == "scenario_override":
        valid = op == "replace" and bool(tokens) and tokens[0] in {"objects", "simulator", "analysis", "ground_stations"}
    if not valid:
        _error(issues, "patch.path_not_allowed", issue_path, f"Path {path!r} is not allowed for operation kind {kind!r}.")


def _validate_patch_operation_value(
    *,
    patch_type: str,
    kind: str,
    value: Any,
    issue_path: str,
    issues: list[InterchangeValidationIssue],
) -> None:
    allowed_kinds = {
        "mission_recovery_candidate": {"mission_burn", "duration_extension"},
        "controller_optimized_variant": {"controller_pointer", "scenario_override"},
    }
    if patch_type in allowed_kinds and kind not in allowed_kinds[patch_type]:
        _error(
            issues,
            "patch.operation_kind_mismatch",
            issue_path,
            f"Operation kind {kind!r} is not valid for patch type {patch_type!r}.",
        )
    if kind == "duration_extension":
        duration = _finite_number(value, issue_path, issues)
        if duration is not None and duration <= 0.0:
            _error(issues, "patch.duration_invalid", issue_path, "Scenario duration must be positive.")
        return
    if kind == "controller_pointer":
        if not isinstance(value, Mapping) or not value:
            _error(issues, "patch.controller_pointer_invalid", issue_path, "Controller pointer must be a non-empty object.")
        return
    if kind != "mission_burn":
        return
    if not isinstance(value, Mapping):
        _error(issues, "patch.mission_burn_invalid", issue_path, "Mission burn must be an object.")
        return
    _closed_mapping(value, {"module", "class_name", "params"}, issue_path, issues)
    if value.get("module") != "sim.mission.modules":
        _error(
            issues,
            "patch.mission_module_incompatible",
            f"{issue_path}.module",
            "Mission patches must use the checked-in sim.mission.modules contract.",
        )
    if value.get("class_name") != "ScheduledVectorBurnMissionModule":
        _error(
            issues,
            "patch.mission_module_incompatible",
            f"{issue_path}.class_name",
            "Mission patches must use ScheduledVectorBurnMissionModule.",
        )
    params = _required_mapping(value, "params", issue_path, issues)
    params_path = f"{issue_path}.params"
    _closed_mapping(
        params,
        {
            "target_id",
            "frame",
            "delta_v_m_s",
            "burn_start_s",
            "burn_duration_s",
            "require_finite_reference",
        },
        params_path,
        issues,
    )
    _required_string(params, "target_id", params_path, issues)
    frame = _required_string(params, "frame", params_path, issues)
    if frame and frame not in {"eci", "ric"}:
        _error(issues, "patch.mission_frame_incompatible", f"{params_path}.frame", "Mission burn frame must be eci or ric.")
    _finite_vector(params.get("delta_v_m_s"), 3, f"{params_path}.delta_v_m_s", issues)
    start = _finite_number(params.get("burn_start_s"), f"{params_path}.burn_start_s", issues)
    if start is not None and start < 0.0:
        _error(issues, "patch.mission_start_invalid", f"{params_path}.burn_start_s", "Burn start must be non-negative.")
    duration = _finite_number(params.get("burn_duration_s"), f"{params_path}.burn_duration_s", issues)
    if duration is not None and duration <= 0.0:
        _error(issues, "patch.mission_duration_invalid", f"{params_path}.burn_duration_s", "Burn duration must be positive.")
    if not isinstance(params.get("require_finite_reference"), bool):
        _error(
            issues,
            "type.boolean",
            f"{params_path}.require_finite_reference",
            "require_finite_reference must be a boolean.",
        )


def _validate_manifest(document: Mapping[str, Any]) -> InterchangeValidationReport:
    issues: list[InterchangeValidationIssue] = []
    fields = {
        "schema_id",
        "schema_version",
        "manifest_id",
        "created_utc",
        "source_product_ids",
        "source_hashes",
        "adapter",
        "materialization_options",
        "defaults_applied",
        "overrides",
        "source_markings",
        "output_markings",
        "output",
        "validation",
        "warnings",
        "failures",
        "recommended_next_action",
        "execution_occurred",
    }
    _closed_mapping(document, fields, "$", issues)
    _require_exact(document, "schema_id", HANDOFF_MANIFEST_SCHEMA_ID, "$", issues)
    _require_exact(document, "schema_version", HANDOFF_MANIFEST_SCHEMA_VERSION, "$", issues)
    identifier = _required_string(document, "manifest_id", "$", issues)
    created = _required_string(document, "created_utc", "$", issues)
    if created:
        _validate_utc(created, "$.created_utc", issues)
    source_ids = _required_sequence(document, "source_product_ids", "$", issues)
    if not source_ids:
        _error(issues, "manifest.sources_missing", "$.source_product_ids", "At least one source product ID is required.")
    for index, product_id in enumerate(source_ids):
        if not isinstance(product_id, str) or not _PRODUCT_ID_RE.fullmatch(product_id):
            _error(issues, "manifest.source_id_invalid", f"$.source_product_ids[{index}]", "Invalid product ID.")
    _required_mapping(document, "source_hashes", "$", issues)
    adapter = _required_mapping(document, "adapter", "$", issues)
    _closed_mapping(adapter, {"adapter_id", "adapter_version"}, "$.adapter", issues)
    _required_string(adapter, "adapter_id", "$.adapter", issues)
    _required_string(adapter, "adapter_version", "$.adapter", issues)
    for field in (
        "materialization_options",
        "defaults_applied",
        "source_markings",
        "output_markings",
        "output",
        "validation",
    ):
        _required_mapping(document, field, "$", issues)
    _required_sequence(document, "overrides", "$", issues)
    _string_list(document, "warnings", "$", issues)
    _required_sequence(document, "failures", "$", issues)
    _required_string(document, "recommended_next_action", "$", issues)
    if document.get("execution_occurred") is not False:
        _error(
            issues,
            "manifest.execution_boundary",
            "$.execution_occurred",
            "Handoff manifests must record execution_occurred=false.",
        )
    if not any(issue.severity == "error" for issue in issues):
        expected = compute_manifest_id(document)
        if identifier != expected:
            _error(issues, "identity.mismatch", "$.manifest_id", f"Expected canonical manifest ID {expected}.")
    valid = not any(issue.severity == "error" for issue in issues)
    return InterchangeValidationReport(
        document_type="manifest",
        schema_id=str(document.get("schema_id", "") or ""),
        schema_version=_integer_or_none(document.get("schema_version")),
        identifier=identifier,
        valid=valid,
        promotable=False,
        issues=tuple(_deduplicate_issues(issues)),
    )


def _closed_mapping(
    value: Mapping[str, Any], allowed: set[str], path: str, issues: list[InterchangeValidationIssue]
) -> None:
    for field in sorted(set(value) - allowed):
        _error(issues, "schema.unknown_field", f"{path}.{field}", f"Unknown field {field!r} is not allowed in schema v1.")


def _require_exact(
    value: Mapping[str, Any], field: str, expected: Any, path: str, issues: list[InterchangeValidationIssue]
) -> None:
    if field not in value:
        _error(issues, "schema.required", f"{path}.{field}", f"Missing required field {field!r}.")
    elif value.get(field) != expected:
        _error(issues, "schema.unsupported_version", f"{path}.{field}", f"Expected {expected!r}; got {value.get(field)!r}.")


def _required_string(
    value: Mapping[str, Any], field: str, path: str, issues: list[InterchangeValidationIssue]
) -> str:
    if field not in value:
        _error(issues, "schema.required", f"{path}.{field}", f"Missing required field {field!r}.")
        return ""
    return _nonempty_string(value.get(field), f"{path}.{field}", issues)


def _nonempty_string(value: Any, path: str, issues: list[InterchangeValidationIssue]) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(issues, "type.string", path, "Expected a non-empty string.")
        return ""
    return value


def _required_mapping(
    value: Mapping[str, Any], field: str, path: str, issues: list[InterchangeValidationIssue]
) -> Mapping[str, Any]:
    if field not in value:
        _error(issues, "schema.required", f"{path}.{field}", f"Missing required field {field!r}.")
        return {}
    item = value.get(field)
    if not isinstance(item, Mapping):
        _error(issues, "type.mapping", f"{path}.{field}", "Expected an object.")
        return {}
    return item


def _required_sequence(
    value: Mapping[str, Any], field: str, path: str, issues: list[InterchangeValidationIssue]
) -> Sequence[Any]:
    if field not in value:
        _error(issues, "schema.required", f"{path}.{field}", f"Missing required field {field!r}.")
        return ()
    item = value.get(field)
    if not isinstance(item, list):
        _error(issues, "type.array", f"{path}.{field}", "Expected an array.")
        return ()
    return item


def _string_list(value: Mapping[str, Any], field: str, path: str, issues: list[InterchangeValidationIssue]) -> None:
    items = _required_sequence(value, field, path, issues)
    for index, item in enumerate(items):
        if not isinstance(item, str):
            _error(issues, "type.string", f"{path}.{field}[{index}]", "Expected a string.")


def _exact_string_sequence(
    value: Any, expected: list[str], path: str, issues: list[InterchangeValidationIssue]
) -> None:
    if not isinstance(value, list) or value != expected:
        _error(issues, "state.sequence_incompatible", path, f"Expected exactly {expected!r}.")


def _finite_vector(value: Any, size: int, path: str, issues: list[InterchangeValidationIssue]) -> None:
    if not isinstance(value, list) or len(value) != size:
        _error(issues, "state.vector_shape", path, f"Expected exactly {size} components.")
        return
    for index, item in enumerate(value):
        _finite_number(item, f"{path}[{index}]", issues)


def _finite_number(value: Any, path: str, issues: list[InterchangeValidationIssue]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(issues, "type.number", path, "Expected a finite JSON number.")
        return None
    result = float(value)
    if not math.isfinite(result):
        _error(issues, "type.nonfinite", path, "Expected a finite JSON number.")
        return None
    return result


def _validate_utc(value: str, path: str, issues: list[InterchangeValidationIssue]) -> None:
    if not value.endswith("Z"):
        _error(issues, "time.utc_required", path, "Timestamp must be ISO-8601 UTC and end in 'Z'.")
        return
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _error(issues, "time.invalid", path, "Timestamp must be valid ISO-8601 UTC.")


def _integer_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _add(
    issues: list[InterchangeValidationIssue], code: str, path: str, message: str, severity: str
) -> None:
    issues.append(InterchangeValidationIssue(code=code, path=path, message=message, severity=severity))


def _error(issues: list[InterchangeValidationIssue], code: str, path: str, message: str) -> None:
    _add(issues, code, path, message, "error")


def _block(issues: list[InterchangeValidationIssue], code: str, path: str, message: str) -> None:
    _add(issues, code, path, message, "blocker")


def _warn(issues: list[InterchangeValidationIssue], code: str, path: str, message: str) -> None:
    _add(issues, code, path, message, "warning")


def _deduplicate_issues(issues: list[InterchangeValidationIssue]) -> list[InterchangeValidationIssue]:
    seen: set[tuple[str, str, str, str]] = set()
    result = []
    for issue in issues:
        key = (issue.code, issue.path, issue.message, issue.severity)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result
