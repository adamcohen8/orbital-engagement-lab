from __future__ import annotations

import json
from pathlib import Path

from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from integrations.oel_mcp.public_registry import public_contracts_for_profile
from integrations.oel_mcp.resources import build_public_resource_catalog
from sim.tests.test_study_lifecycle import _build

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_HANDLING = {"marking": "PUBLIC_TEST", "release_scope": "public"}
OEM_REFERENCE = ROOT / "sim/interchange/examples/oel_earth_eme2000_utc_v3.oem"


def _handlers(tmp_path: Path) -> PublicOELMCPHandlers:
    return PublicOELMCPHandlers(
        read_roots=(ROOT, tmp_path),
        write_roots=(tmp_path,),
    )


def test_analysis_workflow_resource_routes_typed_problems_without_expanding_execution_surface() -> None:
    resources = build_public_resource_catalog(
        profile="public_local",
        tool_contracts=public_contracts_for_profile("public_local"),
    )
    workflow = next(resource for resource in resources if resource.contract.uri == "oel://analysis/workflows/v1")

    assert '"workflow_id": "study_lifecycle"' in workflow.text
    payload = json.loads(workflow.text)
    constellation = next(item for item in payload["workflows"] if item["workflow_id"] == "constellation_design")
    assert constellation["evidence"] == "oel.constellation_design_evidence.v1"
    assert constellation["authoritative_replay"] == "python -m sim.constellation_design replay"
    assert constellation["mcp_tools"] == []
    assert constellation["pro_escalation"] == {
        "availability": "coming_soon",
        "capability_ids": ["constellation_design.optimization"],
        "commercially_available": False,
        "estimated_launch": None,
        "execution_available": False,
        "mcp_tools": [],
        "product_family": "OEL Pro Constellation Design",
        "public_fallback": "Evaluate one explicit public constellation design with bounded objectives.",
        "recommendation_only": True,
        "use_when": "The request requires automated constellation optimization rather than one public design solve.",
    }
    tracking_od = next(
        item for item in payload["workflows"] if item["workflow_id"] == "tracking_data_orbit_determination"
    )
    assert tracking_od["pro_escalation"]["product_family"] == "OEL Pro Orbit Determination"
    assert tracking_od["pro_escalation"]["capability_ids"] == [
        "orbit_determination.reduced_tracking",
        "orbit_determination.ilrs_slr",
    ]
    collection = next(item for item in payload["workflows"] if item["workflow_id"] == "collection_analysis")
    assert "pro_escalation" not in collection
    assert payload["routing"]["pro_recommendations_are_not_execution_authority"] is True
    assert all(item["mcp_tools"] == [] for item in payload["cross_cutting_pro_escalations"])
    assert '"mcp_tools": []' in workflow.text
    assert "scenario YAML or typed orbital-analysis problem" in workflow.text
    assert "sim.pro_" not in workflow.text
    assert "agents/pro" not in workflow.text
    assert "/Users/" not in workflow.text


def test_study_inspect_replay_and_compare_are_read_only_and_content_bound(tmp_path: Path) -> None:
    artifacts = _build(tmp_path / "study")
    handlers = _handlers(tmp_path)

    inspected = handlers.inspect_study(bundle_dir=artifacts.output_dir, handling=PUBLIC_HANDLING)
    replayed = handlers.replay_study(bundle_dir=artifacts.output_dir, handling=PUBLIC_HANDLING)
    compared = handlers.compare_studies(
        left_bundle_dir=artifacts.output_dir,
        right_bundle_dir=artifacts.output_dir,
        handling=PUBLIC_HANDLING,
    )

    assert inspected["status"] == "completed"
    assert inspected["result"]["status"] == "verified"
    assert replayed["result"]["replay_status"] == "identity_verified"
    assert compared["result"]["same_bundle"] is True
    assert all(not payload["effects"]["writes"] for payload in (inspected, replayed, compared))


def test_ccsds_and_frame_time_adapters_return_bounded_nonexecuting_receipts(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)

    ccsds = handlers.inspect_ccsds(
        path=OEM_REFERENCE,
        product_kind="oem",
        handling=PUBLIC_HANDLING,
    )
    epoch = handlers.convert_frame_time(
        operation="convert_epoch",
        epoch="2024-01-01T00:00:00Z",
        from_scale="UTC",
        to_scale="TAI",
        handling=PUBLIC_HANDLING,
    )
    state = handlers.convert_frame_time(
        operation="transform_state",
        epoch="2024-01-01T00:00:00Z",
        time_scale="UTC",
        source_frame="EME2000",
        target_frame="EME2000",
        position_km=[7000.0, 0.0, 0.0],
        velocity_km_s=[0.0, 7.5, 0.0],
        handling=PUBLIC_HANDLING,
    )

    assert ccsds["result"]["inspection"]["valid_oem"] is True
    assert ccsds["result"]["execution_occurred"] is False
    assert epoch["result"]["result"]["scale"] == "TAI"
    assert state["result"]["result"]["position_km"] == [7000.0, 0.0, 0.0]
    assert epoch["effects"]["executes"] is False
    assert state["effects"]["writes"] is False
