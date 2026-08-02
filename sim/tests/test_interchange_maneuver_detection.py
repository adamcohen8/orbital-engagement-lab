from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import sim.handoff as handoff
from sim import SimulationConfig, SimulationSession
from sim.interchange.adapters.maneuver_detection import ManeuverDetectionExportError
from sim.interchange.cli import main as handoff_main
from sim.interchange.provenance import compute_product_id
from sim.interchange.validation import validate_product


def _run_detection_fixture(tmp_path: Path) -> Path:
    source = Path("configs/validation_ekf_maneuver_detection_delayed_impulse.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["outputs"]["output_dir"] = str(tmp_path / "run")
    result = SimulationSession.from_config(SimulationConfig.from_dict(config)).run()
    return Path(result.summary["review_outputs"]["sqlite"]).parent.parent


def test_confirmed_detection_exports_a_versioned_traceable_product(tmp_path: Path) -> None:
    completed_run = _run_detection_fixture(tmp_path)
    product_path = tmp_path / "products" / "maneuver_detection.json"

    emission = handoff.export_maneuver_detection_product(
        completed_run,
        output_path=product_path,
        observer_id="chaser",
        target_id="target",
    )
    product = json.loads(product_path.read_text(encoding="utf-8"))
    report = validate_product(product, source_path=product_path)

    assert emission["status"] == "exported"
    assert emission["execution_occurred"] is False
    assert report.valid is True
    assert report.promotable is True
    assert product["product_kind"] == "oel.maneuver_detection"
    assert product["payload"]["observer"]["object_id"] == "chaser"
    assert product["payload"]["target"]["object_id"] == "target"
    assert product["payload"]["detection"]["status"] == "confirmed"
    assert product["payload"]["detection"]["epoch_jd_utc"] is None
    assert product["quality"]["gates"]["absolute_epoch_available"] is False
    assert product["quality"]["warnings"]

    observations_path = tmp_path / "event_window" / "observations.json"
    window = handoff.export_event_centered_observations(
        product_path,
        output_path=observations_path,
        pre_event_s=20.0,
        post_event_s=20.0,
    )
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    assert window["truth_included"] is False
    assert window["fit_duration_s"] == pytest.approx(20.0)
    assert window["holdout_duration_s"] == pytest.approx(20.0)
    assert {row["partition"] for row in observations["observations"]} == {"fit", "holdout"}
    assert observations["source"]["metadata"]["hidden_truth_included"] is False
    assert observations["source"]["metadata"]["detection_product_id"] == product["product_id"]


def test_detection_product_rejects_summary_event_time_drift(tmp_path: Path) -> None:
    completed_run = _run_detection_fixture(tmp_path)
    product_path = tmp_path / "maneuver_detection.json"
    handoff.export_maneuver_detection_product(completed_run, output_path=product_path)
    product = json.loads(product_path.read_text(encoding="utf-8"))
    tampered = deepcopy(product)
    tampered["payload"]["detector"]["summary"]["maneuver_first_confirmed_t_s"] += 1.0
    tampered["product_id"] = compute_product_id(tampered)

    report = validate_product(tampered, source_path=product_path, verify_sources=False)

    assert report.valid is False
    assert any(item.code == "maneuver_detection.summary_time_mismatch" for item in report.errors)


def test_detection_export_requires_exactly_one_confirmed_event(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/validation_ekf_maneuver_detection_delayed_impulse.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["simulator"]["duration_s"] = 10.0
    config["outputs"]["output_dir"] = str(tmp_path / "run")
    result = SimulationSession.from_config(SimulationConfig.from_dict(config)).run()

    with pytest.raises(ManeuverDetectionExportError, match="exactly one"):
        handoff.export_maneuver_detection_product(
            Path(result.summary["review_outputs"]["sqlite"]).parent.parent,
            output_path=tmp_path / "missing.json",
        )


def test_detection_cli_and_facade_are_publicly_reachable(tmp_path: Path, capsys) -> None:
    completed_run = _run_detection_fixture(tmp_path)
    product_path = tmp_path / "cli_detection.json"

    status = handoff_main(
        [
            "export-maneuver-detection",
            str(completed_run),
            "--output",
            str(product_path),
            "--observer-id",
            "chaser",
            "--target-id",
            "target",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["status"] == "exported"
    assert handoff.build_maneuver_detection_product
    assert handoff.export_maneuver_detection_product
