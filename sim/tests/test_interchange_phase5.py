from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml

from sim.execution import run_simulation_config_file
from sim.interchange.adapters.planning import emit_mission_recovery_scenario_patches
from sim.interchange.cli import main as handoff_main
from sim.interchange.inspection import inspect_path
from sim.interchange.provenance import compute_product_id
from sim.interchange.scenario_patches import materialize_scenario_patch, select_patch_product
from sim.interchange.validation import validate_document

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "sim" / "interchange" / "schemas"


def _source_scenario(path: Path, *, with_mission_recovery: bool = False) -> Path:
    source = yaml.safe_load(
        (ROOT / "agents" / "examples" / "public_agent_mission_recovery_plus_c_burn.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["scenario_name"] = "phase5_source"
    source["metadata"] = {"owner": "public", "public_surface": "phase5_test_fixture"}
    source["simulator"]["duration_s"] = 10.0
    if not with_mission_recovery:
        source["analysis"].pop("mission_recovery", None)
    source["outputs"]["output_dir"] = str(path.parent / "source_run")
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path


def _mission_recovery(*, duration_s: float | None = 2.0) -> dict:
    return {
        "object_id": "target",
        "goal": "orbit_shape",
        "assessment_time_s": 10.0,
        "planner": {
            "max_recovery_time_s": 100.0,
            "max_recovery_delta_v_m_s": 5.0,
            "recommended": {"min_delta_v": "candidate-a", "constrained": "candidate-a"},
            "candidates": [
                {
                    "candidate_id": "candidate-a",
                    "source": "analytic_reconstitution",
                    "source_family": "bounded_public_planner",
                    "target_basis": "initial_orbit_shape",
                    "feasible": True,
                    "verified": True,
                    "within_tolerances": True,
                    "planned_time_s": 4.0,
                    "planned_delta_v_m_s": 0.25,
                    "burn_sequence": [
                        {
                            "burn_index": 0,
                            "start_time_s": 1.0,
                            "duration_s": duration_s,
                            "frame": "eci",
                            "axis": "vector",
                            "delta_v_m_s": 0.25,
                            "delta_v_eci_m_s": [0.0, 0.25, 0.0],
                        }
                    ],
                    "verification": {"method": "deterministic_candidate_propagation"},
                }
            ],
        },
    }


def test_mission_recovery_candidate_emits_typed_patch_and_materializes_without_execution(
    tmp_path: Path,
) -> None:
    source = _source_scenario(tmp_path / "source.yaml")
    emission = emit_mission_recovery_scenario_patches(
        _mission_recovery(), source_scenario=source, output_dir=tmp_path / "patches"
    )
    assert emission["status"] == "emitted"
    assert emission["selection_required"] is True
    patch_path = select_patch_product(emission["index_path"], "candidate-a")
    inspection = inspect_path(patch_path)
    assert inspection["validation"]["promotable"] is True
    patch_summary = inspection["scenario_patch"]
    assert patch_summary["patch_type"] == "mission_recovery_candidate"
    assert patch_summary["selection_id"] == "candidate-a"
    assert patch_summary["rank"] == 1
    assert patch_summary["source_scenario_name"] == "phase5_source"
    assert patch_summary["operation_count"] == 2

    source_before = source.read_bytes()
    destination = tmp_path / "selected_candidate.yaml"
    result = materialize_scenario_patch(
        patch_path,
        source,
        scenario_name="selected_candidate",
        scenario_path=destination,
        output_dir=tmp_path / "run",
        trust_plugins=True,
    )
    assert result["status"] == "materialized"
    assert result["execution_occurred"] is False
    assert source.read_bytes() == source_before
    assert (tmp_path / "run").exists() is False
    scenario = yaml.safe_load(destination.read_text(encoding="utf-8"))
    burn = scenario["objects"]["target"]["mission_objectives"][-1]
    assert burn["class_name"] == "ScheduledVectorBurnMissionModule"
    assert burn["params"]["burn_start_s"] == 11.0
    assert burn["params"]["delta_v_m_s"] == [0.0, 0.25, 0.0]
    assert scenario["simulator"]["duration_s"] == 14.0
    assert scenario["metadata"]["handoff"]["selection"]["selection_id"] == "candidate-a"
    assert scenario["metadata"]["handoff"]["execution_occurred"] is False
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["source_hashes"][scenario["metadata"]["handoff"]["source_product_id"]]
    assert manifest["source_hashes"]["source_scenario:phase5_source"] == patch_summary[
        "source_scenario_sha256"
    ]


def test_normal_mission_recovery_run_emits_patch_index(tmp_path: Path) -> None:
    source = _source_scenario(tmp_path / "source.yaml", with_mission_recovery=True)
    run_simulation_config_file(source)
    summary = json.loads((tmp_path / "source_run" / "master_run_summary.json").read_text(encoding="utf-8"))
    emission = summary["mission_recovery"]["scenario_patch_emission"]
    assert emission["status"] == "emitted"
    assert emission["selection_required"] is True
    assert Path(emission["index_path"]).is_file()


def test_mission_recovery_duration_is_rounded_up_to_complete_source_timestep(
    tmp_path: Path,
) -> None:
    source = _source_scenario(tmp_path / "source.yaml")
    recovery = _mission_recovery(duration_s=2.45)
    recovery["planner"]["candidates"][0]["planned_time_s"] = 4.45
    emission = emit_mission_recovery_scenario_patches(
        recovery, source_scenario=source, output_dir=tmp_path / "patches"
    )
    patch_path = select_patch_product(emission["index_path"], "candidate-a")
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    duration_operation = patch["payload"]["patch"]["operations"][-1]

    assert duration_operation["value"] == 15.0
    result = materialize_scenario_patch(
        patch_path,
        source,
        scenario_name="aligned_duration",
        scenario_path=tmp_path / "aligned.yaml",
        output_dir=tmp_path / "run",
        trust_plugins=True,
    )
    assert result["status"] == "materialized"
    scenario = yaml.safe_load((tmp_path / "aligned.yaml").read_text(encoding="utf-8"))
    assert scenario["simulator"]["duration_s"] == 15.0


def test_patch_index_cli_requires_exact_selection_and_never_executes(tmp_path: Path, capsys) -> None:
    source = _source_scenario(tmp_path / "source.yaml")
    emission = emit_mission_recovery_scenario_patches(
        _mission_recovery(), source_scenario=source, output_dir=tmp_path / "patches"
    )
    destination = tmp_path / "cli_selected.yaml"
    code = handoff_main(
        [
            "materialize-scenario-patch",
            "--patch-index",
            emission["index_path"],
            "--selection-id",
            "candidate-a",
            "--source-scenario",
            str(source),
            "--scenario-name",
            "cli_selected",
            "--output",
            str(destination),
            "--run-output-dir",
            str(tmp_path / "run"),
            "--trust-plugins",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "materialized"
    assert payload["execution_occurred"] is False
    assert (tmp_path / "run").exists() is False

    missing_code = handoff_main(
        [
            "materialize-scenario-patch",
            "--patch-index",
            emission["index_path"],
            "--selection-id",
            "missing",
            "--source-scenario",
            str(source),
            "--scenario-name",
            "missing",
            "--output",
            str(tmp_path / "missing.yaml"),
            "--run-output-dir",
            str(tmp_path / "missing_run"),
        ]
    )
    assert missing_code == 2
    assert (tmp_path / "missing.yaml").exists() is False


def test_stale_source_scenario_fails_closed_with_manifest(tmp_path: Path) -> None:
    source = _source_scenario(tmp_path / "source.yaml")
    emission = emit_mission_recovery_scenario_patches(
        _mission_recovery(), source_scenario=source, output_dir=tmp_path / "patches"
    )
    patch_path = select_patch_product(emission["index_path"], "candidate-a")
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    destination = tmp_path / "stale.yaml"
    result = materialize_scenario_patch(
        patch_path,
        source,
        scenario_name="stale",
        scenario_path=destination,
        output_dir=tmp_path / "run",
    )
    assert result["status"] == "blocked"
    assert destination.exists() is False
    assert any(item["code"] == "patch.source_hash_mismatch" for item in result["failures"])
    assert Path(result["manifest_path"]).is_file()


def test_unbound_mission_burn_candidate_is_evidence_but_not_promotable(tmp_path: Path) -> None:
    source = _source_scenario(tmp_path / "source.yaml")
    emission = emit_mission_recovery_scenario_patches(
        _mission_recovery(duration_s=None), source_scenario=source, output_dir=tmp_path / "patches"
    )
    patch = select_patch_product(emission["index_path"], "candidate-a")
    report = validate_document(json.loads(patch.read_text(encoding="utf-8")), source_path=patch)
    assert report.valid is True
    assert report.promotable is False
    assert any(issue.code == "quality.review_required" for issue in report.blockers)


def test_scenario_patch_rejects_non_allowlisted_operation_path(tmp_path: Path) -> None:
    source = _source_scenario(tmp_path / "source.yaml")
    emission = emit_mission_recovery_scenario_patches(
        _mission_recovery(), source_scenario=source, output_dir=tmp_path / "patches"
    )
    patch = select_patch_product(emission["index_path"], "candidate-a")
    product = json.loads(patch.read_text(encoding="utf-8"))
    tampered = deepcopy(product)
    tampered["payload"]["patch"]["operations"][0]["path"] = "metadata.owner"
    tampered["product_id"] = compute_product_id(tampered)
    report = validate_document(tampered, source_path=patch)
    assert report.valid is False
    assert any(issue.code == "patch.path_not_allowed" for issue in report.errors)


def test_scenario_patch_rejects_type_mismatch_and_unchecked_mission_module(tmp_path: Path) -> None:
    source = _source_scenario(tmp_path / "source.yaml")
    emission = emit_mission_recovery_scenario_patches(
        _mission_recovery(), source_scenario=source, output_dir=tmp_path / "patches"
    )
    patch = select_patch_product(emission["index_path"], "candidate-a")
    product = json.loads(patch.read_text(encoding="utf-8"))

    mismatched = deepcopy(product)
    mismatched["payload"]["selection"]["selection_kind"] = "controller_optimized_variant"
    mismatched["product_id"] = compute_product_id(mismatched)
    mismatch_report = validate_document(mismatched, source_path=patch)
    assert any(issue.code == "patch.selection_kind_mismatch" for issue in mismatch_report.errors)

    unchecked = deepcopy(product)
    unchecked["payload"]["patch"]["operations"][0]["value"]["module"] = "custom.mission"
    unchecked["product_id"] = compute_product_id(unchecked)
    unchecked_report = validate_document(unchecked, source_path=patch)
    assert any(issue.code == "patch.mission_module_incompatible" for issue in unchecked_report.errors)


def test_scenario_patch_payload_schema_is_checked_in() -> None:
    schema = json.loads((SCHEMAS / "oel-scenario-patch-v1.schema.json").read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["patch"]["properties"]["operations"]["minItems"] == 1
    assert schema["properties"]["selection"]["required"] == [
        "selection_id",
        "selection_kind",
        "rank",
        "recommended_modes",
    ]
