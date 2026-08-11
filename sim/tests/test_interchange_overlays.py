from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from sim import handoff
from sim.interchange.overlays import ScenarioOverlayError, emit_scenario_overlay
from sim.interchange.provenance import compute_product_id
from sim.interchange.validation import validate_document

ROOT = Path(__file__).resolve().parents[2]


def _source(tmp_path: Path) -> Path:
    raw = yaml.safe_load(
        (ROOT / "agents" / "examples" / "public_agent_single_satellite.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["scenario_name"] = "overlay_source"
    raw["outputs"]["output_dir"] = str(tmp_path / "source_run")
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return source


def _overlay() -> dict:
    return {
        "ground_stations": [
            {
                "id": "overlay_station",
                "lat_deg": 38.803,
                "lon_deg": -104.526,
                "alt_km": 1.9,
                "min_elevation_deg": 10.0,
                "max_range_km": 2500.0,
            }
        ],
        "objects": {
            "target": {
                "flight_software": {
                    "stack": "fsw.passive",
                    "hardware_profile": "hardware.passive.v1",
                    "task_period_s": 2.0,
                },
                "knowledge": {
                    "refresh_rate_s": 10.0,
                    "targets": ["target"],
                    "conditions": {"require_line_of_sight": False},
                    "sensor_error": {
                        "pos_sigma_km": [0.0, 0.0, 0.0],
                        "vel_sigma_km_s": [0.0, 0.0, 0.0],
                    },
                    "estimation": {"type": "ekf"},
                },
            }
        },
        "simulator": {"termination": {"earth_impact_enabled": False}},
        "outputs": {"review": {"enabled": True, "detail": "standard"}},
    }


def test_typed_overlay_materializes_station_controller_safety_and_review(tmp_path: Path) -> None:
    source = _source(tmp_path)
    product_path = tmp_path / "overlay.json"
    emission = emit_scenario_overlay(
        source,
        _overlay(),
        overlay_id="access_and_control",
        output_path=product_path,
        rationale="Bind access and execution context to the accepted source state.",
    )
    assert emission["valid"] is True
    assert emission["promotable"] is True
    assert emission["operation_count"] == 5

    result = handoff.materialize_scenario_patch(
        product_path,
        source,
        scenario_name="overlay_materialized",
        scenario_path=tmp_path / "materialized.yaml",
        output_dir=tmp_path / "run",
        trust_plugins=True,
    )
    assert result["status"] == "materialized"
    scenario = yaml.safe_load((tmp_path / "materialized.yaml").read_text(encoding="utf-8"))
    assert scenario["ground_stations"][0]["id"] == "overlay_station"
    assert scenario["objects"]["target"]["flight_software"]["stack"] == "fsw.passive"
    assert scenario["objects"]["target"]["flight_software"]["task_period_s"] == 2.0
    assert scenario["objects"]["target"]["knowledge"]["targets"] == ["target"]
    assert scenario["simulator"]["termination"] == {"earth_impact_enabled": False}
    assert scenario["outputs"]["review"] == {"enabled": True, "detail": "standard"}
    assert handoff.compare_handoff(product_path, result["scenario_path"])["status"] == "equivalent"


def test_overlay_rejects_unbounded_fields_and_tampered_paths(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(ScenarioOverlayError, match="Unsupported overlay top-level"):
        emit_scenario_overlay(
            source,
            {"metadata": {"owner": "changed"}},
            overlay_id="bad",
            output_path=tmp_path / "bad.json",
            rationale="not allowed",
        )
    with pytest.raises(ScenarioOverlayError, match="Unsupported analysis overlay fields"):
        emit_scenario_overlay(
            source,
            {"analysis": {"workflow_policy": {"posture": "bounded"}}},
            overlay_id="bad_analysis",
            output_path=tmp_path / "bad_analysis.json",
            rationale="not allowed",
        )

    product_path = tmp_path / "overlay.json"
    emit_scenario_overlay(
        source,
        _overlay(),
        overlay_id="bounded",
        output_path=product_path,
        rationale="bounded",
    )
    product = json.loads(product_path.read_text(encoding="utf-8"))
    tampered = deepcopy(product)
    tampered["payload"]["patch"]["operations"][0]["path"] = "metadata.owner"
    tampered["product_id"] = compute_product_id(tampered)
    report = validate_document(tampered, source_path=product_path)
    assert report.valid is False
    assert any(issue.code == "patch.path_not_allowed" for issue in report.errors)


def test_scenario_patch_operation_target_participates_in_product_identity(tmp_path: Path) -> None:
    source = _source(tmp_path)
    product_path = tmp_path / "overlay.json"
    emit_scenario_overlay(
        source,
        _overlay(),
        overlay_id="identity_bound",
        output_path=product_path,
        rationale="identity regression",
    )
    product = json.loads(product_path.read_text(encoding="utf-8"))
    changed = deepcopy(product)
    changed["payload"]["patch"]["operations"][0]["path"] = "objects.target.knowledge"

    assert compute_product_id(changed) != compute_product_id(product)


@pytest.mark.parametrize("legacy_field", ("orbit_control", "attitude_control", "mission_execution"))
def test_overlay_rejects_removed_satellite_gnc_fields(tmp_path: Path, legacy_field: str) -> None:
    with pytest.raises(ScenarioOverlayError, match="Unsupported overlay fields for object"):
        emit_scenario_overlay(
            _source(tmp_path),
            {"objects": {"target": {legacy_field: {"module": "legacy.removed"}}}},
            overlay_id=f"legacy_{legacy_field}",
            output_path=tmp_path / f"{legacy_field}.json",
            rationale="Legacy GNC v1 fields must fail closed.",
        )


def test_overlay_cli_emits_product_without_materialization(tmp_path: Path, capsys) -> None:
    source = _source(tmp_path)
    overlay_path = tmp_path / "overlay.yaml"
    overlay_path.write_text(yaml.safe_dump(_overlay(), sort_keys=False), encoding="utf-8")
    output = tmp_path / "product.json"
    code = handoff.main(
        [
            "emit-overlay",
            "--source-scenario",
            str(source),
            "--overlay",
            str(overlay_path),
            "--overlay-id",
            "cli_overlay",
            "--rationale",
            "CLI overlay test",
            "--output",
            str(output),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "emitted"
    assert output.is_file()
