"""Versioned, read-only interchange contracts for OEL capability handoffs."""

from .comparison import (
    HANDOFF_COMPARISON_SCHEMA_ID,
    HANDOFF_COMPARISON_SCHEMA_VERSION,
    HandoffComparisonError,
    compare_handoff,
)
from .completed_runs import (
    CompletedRunStateExportError,
    build_completed_run_state_product,
    export_completed_run_state,
)
from .contracts import (
    AGE_STATUSES,
    DATA_SCOPES,
    INTEGRITY_STATUSES,
    PRODUCT_ENVELOPE_SCHEMA_ID,
    PRODUCT_ENVELOPE_SCHEMA_VERSION,
    QUALITY_DISPOSITIONS,
    AgeStatus,
    DataScope,
    HandoffManifest,
    IntegrityStatus,
    ProductEnvelope,
    QualityDisposition,
)
from .inspection import inspect_document, inspect_path
from .materialization import OGPMaterializationError, canonical_scenario_digest, materialize_ogp, materialize_onp
from .provenance import canonical_json_bytes, compute_manifest_id, compute_product_id, sha256_file
from .scenario_patches import materialize_scenario_patch, select_patch_product
from .validation import (
    InterchangeValidationIssue,
    InterchangeValidationReport,
    load_interchange_document,
    validate_document,
    validate_product,
)

__all__ = [
    "AGE_STATUSES",
    "DATA_SCOPES",
    "INTEGRITY_STATUSES",
    "PRODUCT_ENVELOPE_SCHEMA_ID",
    "PRODUCT_ENVELOPE_SCHEMA_VERSION",
    "QUALITY_DISPOSITIONS",
    "AgeStatus",
    "DataScope",
    "HandoffManifest",
    "IntegrityStatus",
    "InterchangeValidationIssue",
    "InterchangeValidationReport",
    "ProductEnvelope",
    "HANDOFF_COMPARISON_SCHEMA_ID",
    "HANDOFF_COMPARISON_SCHEMA_VERSION",
    "HandoffComparisonError",
    "CompletedRunStateExportError",
    "OGPMaterializationError",
    "QualityDisposition",
    "canonical_json_bytes",
    "canonical_scenario_digest",
    "compare_handoff",
    "compute_manifest_id",
    "compute_product_id",
    "build_completed_run_state_product",
    "export_completed_run_state",
    "inspect_document",
    "inspect_path",
    "load_interchange_document",
    "materialize_onp",
    "materialize_ogp",
    "materialize_scenario_patch",
    "select_patch_product",
    "sha256_file",
    "validate_document",
    "validate_product",
]
