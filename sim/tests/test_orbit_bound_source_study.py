from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sim.analysis.collection_opportunity import assess_collection_opportunities, write_collection_evidence
from sim.analysis.directed_link import DirectedLinkConfig, LinkTerminal, TerminalPattern
from sim.analysis.mission_scheduling import MissionSchedulingError
from sim.analysis.mission_scheduling_sources import build_mission_scheduling_problem_from_sources
from sim.analysis.orbit_bound_sources import verify_orbit_bound_link, write_orbit_bound_link
from sim.analysis.orbit_history_product import (
    OrbitHistoryProductError,
    export_orbit_history_product,
    orbit_history_semantic_sha256,
    verify_orbit_history_product,
)
from sim.analysis.schedule_power_study import (
    SchedulePowerStudyError,
    build_orbit_bound_source_study,
    inspect_orbit_bound_source_study,
)
from sim.analysis.spacecraft_power import power_history_from_mapping
from sim.collection import main as collection_main
from sim.study import main as study_main

ROOT = Path(__file__).resolve().parents[2]
EPOCH = 2451545.0


@pytest.fixture(scope="module")
def bound_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, dict]:
    root = tmp_path_factory.mktemp("orbit-bound-inputs")
    config = yaml.safe_load((ROOT / "agents/examples/public_agent_single_satellite.yaml").read_text())
    config["scenario_name"] = "orbit_bound_source_study_test"
    config["objects"] = {"SAT-A": config["objects"]["target"]}
    config["simulator"]["duration_s"] = 120.0
    config["simulator"]["initial_jd_utc"] = EPOCH
    config["outputs"]["output_dir"] = str(root / "completed_run")
    scenario = root / "scenario.yaml"
    scenario.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    for extra in (["--validate-only"], []):
        result = subprocess.run(
            [sys.executable, "run_simulation.py", "--config", str(scenario), *extra],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    orbit_dir = root / "orbit_history"
    export_orbit_history_product(root / "completed_run", object_id="SAT-A", output_dir=orbit_dir)
    orbit_manifest, history = verify_orbit_history_product(orbit_dir)
    collection = json.loads((ROOT / "examples/collection/public_equatorial_optical_collection.json").read_text())
    collection["name"] = "orbit_bound_collection"
    collection["spacecraft"]["asset_id"] = "SAT-A"
    collection["spacecraft"]["initial_state_eci_km_km_s"] = [
        *history.position_eci_km[0], *history.velocity_eci_km_s[0]
    ]
    collection["duration_s"] = 120.0
    collection["resources"] = {"enabled": False}
    evidence = assess_collection_opportunities(collection, orbit_history=history)
    assert evidence["summary"]["accepted_opportunity_count"] == 1
    write_collection_evidence(evidence, root / "collection.json")

    pattern = TerminalPattern(kind="constant", gain_dbi=30.0)
    config = DirectedLinkConfig(
        analysis_id="orbit_bound_link", link_id="orbit_bound_link",
        tx_terminal=LinkTerminal("sat-tx", "SAT-A", "body", (1, 0, 0, 0), pattern),
        rx_terminal=LinkTerminal("gs-rx", "GS-1", "enu", (1, 0, 0, 0), pattern),
        carrier_frequency_hz=2.2e9, tx_power_w=10.0, data_rate_bps=100e6,
        system_noise_temperature_k=500.0, required_eb_n0_db=-20.0,
        min_fixed_site_elevation_rad=0.0,
    )
    write_orbit_bound_link(
        config, history, station_latitude_deg=0.0, station_longitude_deg=79.53938163,
        station_height_km=0.0, sample_indices=list(range(90, 121)), output_dir=root / "link",
    )
    verify_orbit_bound_link(root / "link", history)

    plan = {
        "schema_version": "oel.mission_scheduling_source_plan.v1",
        "analysis_id": "orbit_bound_source_schedule",
        "epoch_jd_utc": EPOCH,
        "horizon_start_s": 0.0,
        "horizon_end_s": 120.0,
        "orbit_history_semantic_sha256": orbit_manifest["history_semantic_sha256"],
        "assets": [{
            "asset_id": "SAT-A", "storage_capacity_bytes": 150e6,
            "initial_storage_bytes": 0.0, "energy_budget_wh": 10.0,
            "maximum_payload_duty_cycle": 1.0, "maximum_slew_rate_rad_s": None,
            "settling_time_s": 2.0,
        }],
        "collection_sources": [{
            "source_id": "collection", "path": "collection.json", "asset_id": "SAT-A",
            "objective_scale": 1.0, "energy_cost_wh": 3.0,
        }],
        "link_sources": [{
            "source_id": "link", "path": "link", "asset_id": "SAT-A",
            "station_asset_id": "GS-1", "station_id": "GS-1", "energy_cost_wh": 1.0,
        }],
        "minimum_selected_observations": 1,
        "require_observation_delivery_by_horizon": True,
    }
    problem = json.loads((ROOT / "examples/spacecraft_power/public_schedule_power_problem.json").read_text())
    problem["epoch_jd_utc"] = EPOCH
    problem["horizon_end_s"] = 120.0
    problem["integration_step_s"] = 10.0
    (root / "source_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (root / "power_problem.json").write_text(json.dumps(problem), encoding="utf-8")
    return root, plan, problem


def _build(tmp_path: Path, bound_inputs: tuple[Path, dict, dict]) -> Path:
    root, plan, problem = bound_inputs
    output = tmp_path / "study"
    result = build_orbit_bound_source_study(
        orbit_history_dir=root / "orbit_history", source_plan=plan, base_dir=root,
        problem=problem, observation_load_w=180.0, downlink_load_w=120.0,
        output_dir=output,
    )
    assert result["status"] == "verified"
    return output


def test_completed_run_orbit_binds_collection_link_schedule_power_and_study(
    tmp_path: Path, bound_inputs: tuple[Path, dict, dict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _build(tmp_path, bound_inputs)
    result = inspect_orbit_bound_source_study(output)
    digest = result["history_semantic_sha256"]
    source_root = output / "source_study" / "source_schedule" / "source_products"
    collection = json.loads((source_root / "collection" / "collection_evidence.json").read_text())
    link = json.loads((source_root / "link" / "link_analysis_manifest.json").read_text())
    power = json.loads((output / "source_study" / "schedule_power_study" / "power" /
                        "spacecraft_power_summary.json").read_text())
    assert collection["orbit_history_semantic_sha256"] == digest
    assert link["orbit_binding"]["parent_history_sha256"] == digest
    assert link["orbit_binding"]["sample_indices"] == list(range(90, 121))
    assert power["history_semantic_sha256"] == digest
    assert result["feasibility"] == "feasible"
    assert study_main(["inspect-orbit-bound-source-study", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == result


def test_orbit_export_and_bound_link_cli(
    tmp_path: Path, bound_inputs: tuple[Path, dict, dict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _, _ = bound_inputs
    orbit_output = tmp_path / "orbit"
    assert study_main([
        "export-orbit-history", str(root / "completed_run"),
        "--object-id", "SAT-A", "--output-dir", str(orbit_output),
    ]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["history_semantic_sha256"] == verify_orbit_history_product(
        root / "orbit_history"
    )[0]["history_semantic_sha256"]
    link_manifest = json.loads((root / "link" / "link_analysis_manifest.json").read_text())
    config_path = tmp_path / "link_config.json"
    config_path.write_text(json.dumps(link_manifest["normalized_config"]))
    link_output = tmp_path / "link"
    assert study_main([
        "build-orbit-bound-link", "--orbit-history-dir", str(orbit_output),
        "--config", str(config_path), "--station-latitude-deg", "0",
        "--station-longitude-deg", "79.53938163", "--station-height-km", "0",
        "--first-sample-index", "90", "--last-sample-index", "120",
        "--output-dir", str(link_output),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["sample_indices"] == list(range(90, 121))
    _, history = verify_orbit_history_product(orbit_output)
    verify_orbit_bound_link(link_output, history)
    collection_problem = json.loads((root / "collection.json").read_text())["normalized_problem"]
    collection_problem_path = tmp_path / "collection_problem.json"
    collection_problem_path.write_text(json.dumps(collection_problem))
    collection_output = tmp_path / "collection_evidence.json"
    assert collection_main([
        str(collection_problem_path), "--orbit-history-dir", str(orbit_output),
        "--output", str(collection_output),
    ]) == 0
    capsys.readouterr()
    assert json.loads(collection_output.read_text())["orbit_history_semantic_sha256"] == exported[
        "history_semantic_sha256"
    ]


def test_bound_link_cli_reports_malformed_config_as_structured_error(
    tmp_path: Path, bound_inputs: tuple[Path, dict, dict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _, _ = bound_inputs
    config_path = tmp_path / "invalid_link_config.json"
    config_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "link"
    assert study_main([
        "build-orbit-bound-link", "--orbit-history-dir", str(root / "orbit_history"),
        "--config", str(config_path), "--station-latitude-deg", "0",
        "--station-longitude-deg", "0", "--station-height-km", "0",
        "--first-sample-index", "0", "--last-sample-index", "1",
        "--output-dir", str(output),
    ]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "error", "message": "Directed-link tx_terminal must be a JSON object."
    }
    assert not output.exists()


def _rewrite_orbit(root: Path, destination: Path, field: str) -> Path:
    shutil.copytree(root / "orbit_history", destination)
    history_path = destination / "normalized_history.json"
    payload = json.loads(history_path.read_text())
    if field == "state":
        payload["samples"][0]["position_eci_km"][0] += 1.0
    elif field == "epoch":
        payload["epoch_jd_utc"] += 1.0
    else:
        payload["asset_id"] = "SAT-B"
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    history_path.write_text(content)
    manifest_path = destination / "orbit_history_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    history = power_history_from_mapping(payload)
    manifest["asset_id"] = history.object_id
    manifest["epoch_jd_utc"] = history.initial_jd_utc
    manifest["history_semantic_sha256"] = orbit_history_semantic_sha256(history)
    manifest["history_file_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_orbit_history_product(destination)
    return destination


@pytest.mark.parametrize("field", ["state", "epoch", "asset"])
def test_changed_parent_orbit_fails_closed(
    tmp_path: Path, bound_inputs: tuple[Path, dict, dict], field: str,
) -> None:
    root, plan, problem = bound_inputs
    changed = _rewrite_orbit(root, tmp_path / "changed_orbit", field)
    with pytest.raises(SchedulePowerStudyError, match="parent orbit asset and digest"):
        build_orbit_bound_source_study(
            orbit_history_dir=changed, source_plan=plan, base_dir=root, problem=problem,
            observation_load_w=180.0, downlink_load_w=120.0,
            output_dir=tmp_path / "rejected",
        )
    assert not (tmp_path / "rejected").exists()


def test_source_scheduler_rejects_missing_collection_and_link_bindings(
    tmp_path: Path, bound_inputs: tuple[Path, dict, dict],
) -> None:
    root, plan, _ = bound_inputs
    copied = tmp_path / "sources"
    copied.mkdir()
    shutil.copyfile(root / "collection.json", copied / "collection.json")
    shutil.copytree(root / "link", copied / "link")
    collection_path = copied / "collection.json"
    collection = json.loads(collection_path.read_text())
    del collection["orbit_history_semantic_sha256"]
    collection_path.write_text(json.dumps(collection))
    with pytest.raises(MissionSchedulingError, match="orbit binding"):
        build_mission_scheduling_problem_from_sources(plan, base_dir=copied)

    shutil.copyfile(root / "collection.json", collection_path)
    link_path = copied / "link" / "link_analysis_manifest.json"
    link = json.loads(link_path.read_text())
    del link["orbit_binding"]
    link_path.write_text(json.dumps(link))
    with pytest.raises(MissionSchedulingError, match="orbit binding"):
        build_mission_scheduling_problem_from_sources(plan, base_dir=copied)


def test_study_inspection_rejects_changed_retained_history(
    tmp_path: Path, bound_inputs: tuple[Path, dict, dict],
) -> None:
    output = _build(tmp_path, bound_inputs)
    path = output / "orbit_history" / "normalized_history.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        inspect_orbit_bound_source_study(output)


def test_orbit_export_rejects_config_that_differs_from_review_store(
    tmp_path: Path, bound_inputs: tuple[Path, dict, dict],
) -> None:
    root, _, _ = bound_inputs
    copied = tmp_path / "completed_run"
    (copied / "review").mkdir(parents=True)
    shutil.copyfile(root / "completed_run" / "review" / "run.sqlite", copied / "review" / "run.sqlite")
    config_path = copied / "effective_config.json"
    config = json.loads((root / "completed_run" / "effective_config.json").read_text())
    config["scenario_name"] = "different_run"
    config_path.write_text(json.dumps(config))
    with pytest.raises(OrbitHistoryProductError, match="differs from review-store provenance"):
        export_orbit_history_product(copied, object_id="SAT-A", output_dir=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
