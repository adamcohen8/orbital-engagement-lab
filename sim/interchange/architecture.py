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
)
