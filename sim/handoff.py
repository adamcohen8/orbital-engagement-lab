"""Stable facade for OEL product inspection and bounded materialization."""

from sim.interchange.cli import main
from sim.interchange.comparison import (
    HANDOFF_COMPARISON_SCHEMA_ID,
    HANDOFF_COMPARISON_SCHEMA_VERSION,
    HandoffComparisonError,
    compare_handoff,
)
from sim.interchange.completed_runs import (
    CompletedRunStateExportError,
    build_completed_run_state_product,
    export_completed_run_state,
)
from sim.interchange.contracts import HandoffManifest, ProductEnvelope
from sim.interchange.inspection import inspect_document, inspect_path
from sim.interchange.maneuver_detection import (
    ManeuverDetectionExportError,
    build_maneuver_detection_product,
    export_event_centered_observations,
    export_maneuver_detection_product,
)
from sim.interchange.materialization import (
    OGPMaterializationError,
    canonical_scenario_digest,
    materialize_ogp,
    materialize_onp,
)
from sim.interchange.overlays import ScenarioOverlayError, emit_scenario_overlay, load_scenario_overlay
from sim.interchange.provenance import (
    canonical_json_bytes,
    compute_manifest_id,
    compute_product_id,
    sha256_file,
)
from sim.interchange.satellite_checkpoints import (
    SatelliteCheckpointError,
    export_satellite_checkpoint,
    materialize_satellite_checkpoint,
)
from sim.interchange.scenario_patches import materialize_scenario_patch, select_patch_product
from sim.interchange.snapshots import (
    CompletedRunSnapshotError,
    export_completed_run_snapshot,
    materialize_snapshot_onp,
)
from sim.interchange.validation import (
    load_interchange_document,
    validate_document,
    validate_product,
)

__all__ = [
    "HandoffManifest",
    "ProductEnvelope",
    "HANDOFF_COMPARISON_SCHEMA_ID",
    "HANDOFF_COMPARISON_SCHEMA_VERSION",
    "HandoffComparisonError",
    "CompletedRunStateExportError",
    "CompletedRunSnapshotError",
    "OGPMaterializationError",
    "ManeuverDetectionExportError",
    "ScenarioOverlayError",
    "SatelliteCheckpointError",
    "canonical_json_bytes",
    "canonical_scenario_digest",
    "compare_handoff",
    "compute_manifest_id",
    "compute_product_id",
    "build_completed_run_state_product",
    "build_maneuver_detection_product",
    "export_event_centered_observations",
    "export_completed_run_state",
    "export_completed_run_snapshot",
    "export_maneuver_detection_product",
    "export_satellite_checkpoint",
    "emit_scenario_overlay",
    "inspect_document",
    "inspect_path",
    "load_interchange_document",
    "load_scenario_overlay",
    "materialize_onp",
    "materialize_ogp",
    "materialize_scenario_patch",
    "materialize_snapshot_onp",
    "materialize_satellite_checkpoint",
    "select_patch_product",
    "sha256_file",
    "validate_document",
    "validate_product",
]


if __name__ == "__main__":
    raise SystemExit(main())
