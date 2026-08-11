from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterchangeCapabilityFamily:
    name: str
    module: str
    facade: str
    capabilities: tuple[str, ...]


INTERCHANGE_CAPABILITY_FAMILIES = (
    InterchangeCapabilityFamily(
        name="contracts",
        module="sim.interchange.contracts",
        facade="sim.handoff",
        capabilities=("ProductEnvelope", "HandoffManifest"),
    ),
    InterchangeCapabilityFamily(
        name="provenance",
        module="sim.interchange.provenance",
        facade="sim.handoff",
        capabilities=("canonical_json_bytes", "compute_product_id", "compute_manifest_id", "sha256_file"),
    ),
    InterchangeCapabilityFamily(
        name="validation",
        module="sim.interchange.validation",
        facade="sim.handoff",
        capabilities=("load_interchange_document", "validate_document", "validate_product"),
    ),
    InterchangeCapabilityFamily(
        name="inspection",
        module="sim.interchange.inspection",
        facade="sim.handoff",
        capabilities=("inspect_document", "inspect_path"),
    ),
    InterchangeCapabilityFamily(
        name="materialization",
        module="sim.interchange.materialization",
        facade="sim.handoff",
        capabilities=("canonical_scenario_digest", "materialize_onp", "materialize_ogp", "OGPMaterializationError"),
    ),
    InterchangeCapabilityFamily(
        name="scenario_patches",
        module="sim.interchange.scenario_patches",
        facade="sim.handoff",
        capabilities=("materialize_scenario_patch", "select_patch_product"),
    ),
    InterchangeCapabilityFamily(
        name="overlays",
        module="sim.interchange.overlays",
        facade="sim.handoff",
        capabilities=("ScenarioOverlayError", "emit_scenario_overlay", "load_scenario_overlay"),
    ),
    InterchangeCapabilityFamily(
        name="comparison",
        module="sim.interchange.comparison",
        facade="sim.handoff",
        capabilities=(
            "HANDOFF_COMPARISON_SCHEMA_ID",
            "HANDOFF_COMPARISON_SCHEMA_VERSION",
            "HandoffComparisonError",
            "compare_handoff",
        ),
    ),
    InterchangeCapabilityFamily(
        name="completed_runs",
        module="sim.interchange.completed_runs",
        facade="sim.handoff",
        capabilities=(
            "CompletedRunStateExportError",
            "build_completed_run_state_product",
            "export_completed_run_state",
        ),
    ),
    InterchangeCapabilityFamily(
        name="maneuver_detection",
        module="sim.interchange.maneuver_detection",
        facade="sim.handoff",
        capabilities=(
            "ManeuverDetectionExportError",
            "build_maneuver_detection_product",
            "export_event_centered_observations",
            "export_maneuver_detection_product",
        ),
    ),
    InterchangeCapabilityFamily(
        name="snapshots",
        module="sim.interchange.snapshots",
        facade="sim.handoff",
        capabilities=(
            "CompletedRunSnapshotError",
            "export_completed_run_snapshot",
            "materialize_snapshot_onp",
        ),
    ),
    InterchangeCapabilityFamily(
        name="satellite_checkpoints",
        module="sim.interchange.satellite_checkpoints",
        facade="sim.handoff",
        capabilities=(
            "SatelliteCheckpointError",
            "export_satellite_checkpoint",
            "materialize_satellite_checkpoint",
        ),
    ),
)
