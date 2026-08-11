from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import sim.handoff as handoff
from sim import SimulationConfig, SimulationSession
from sim.api import SimulationWorkspace
from sim.interchange.provenance import compute_product_id
from sim.interchange.validation import validate_product
from sim.review import ReviewWorkspace


def _run_pair(output_dir: Path) -> Path:
    config = {
        "scenario_name": "atomic_snapshot_fixture",
        "metadata": {
            "owner": "public",
            "public_surface": "public-agent-workflow",
            "export_review": {"approved_for_public_export": True},
        },
        "objects": {
            "chief": {
                "enabled": True,
                "role": "target",
                "kind": "satellite",
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
            "deputy": {
                "enabled": True,
                "role": "chaser",
                "kind": "satellite",
                "specs": {"mass_kg": 80.0},
                "initial_state": {
                    "position_eci_km": [7000.1, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
        },
        "simulator": {
            "initial_jd_utc": 2461254.5,
            "duration_s": 2.0,
            "dt_s": 1.0,
            "dynamics": {
                "orbit": {"model": "two_body", "orbit_substep_s": 1.0},
                "attitude": {"enabled": False},
                "rocket": {"enabled": False},
            },
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": True, "save_full_log": False},
            "review": {"enabled": True, "detail": "standard"},
            "plots": {"enabled": False},
            "animations": {"enabled": False},
        },
    }
    result = SimulationSession.from_config(SimulationConfig.from_dict(config)).run()
    return Path(result.summary["review_outputs"]["sqlite"])


def test_atomic_pair_snapshot_materializes_and_preserves_both_states(tmp_path: Path) -> None:
    db_path = _run_pair(tmp_path / "source")
    product_path = tmp_path / "snapshot.json"

    exported = handoff.export_completed_run_snapshot(
        db_path.parent.parent,
        output_path=product_path,
        object_ids=["chief", "deputy"],
        selector="sample_index",
        sample_index=1,
    )
    product = json.loads(product_path.read_text(encoding="utf-8"))
    report = validate_product(product, source_path=product_path)

    assert exported["sample_index"] == 1
    assert exported["relative_pair_count"] == 1
    assert report.promotable is True
    assert len(product["payload"]["states"]) == 2
    assert len(product["payload"]["relative_pairs"]) == 1

    scenario_path = tmp_path / "continued.yaml"
    materialized = handoff.materialize_snapshot_onp(
        product_path,
        scenario_name="atomic_pair_continuation",
        scenario_path=scenario_path,
        output_dir=tmp_path / "continued",
        duration_s=1.0,
        dt_s=1.0,
        trust_plugins=True,
    )
    assert materialized["status"] == "materialized"
    assert Path(materialized["manifest_path"]).is_file()
    comparison = handoff.compare_handoff(product_path, scenario_path)
    assert comparison["status"] == "equivalent"
    assert comparison["summary"]["failed_count"] == 0
    SimulationWorkspace().run(scenario_path)
    rows = ReviewWorkspace.open(tmp_path / "continued").query(
        "SELECT object_id, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
        "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s "
        "FROM object_state WHERE sample_index = 0 ORDER BY object_id"
    ).rows
    expected = {
        dict(item["object"])["object_id"]: dict(item["state"])["values"]
        for item in product["payload"]["states"]
    }
    assert [row["object_id"] for row in rows] == ["chief", "deputy"]
    for row in rows:
        assert list(row.values())[1:] == pytest.approx(expected[row["object_id"]])


def test_event_snapshot_selects_all_objects_at_the_event_sample(tmp_path: Path) -> None:
    db_path = _run_pair(tmp_path / "source")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("failure:deputy", 1.0, 1, "deputy", "failure", "warning", "passivate deputy", "test"),
        )
        conn.commit()
    product_path = tmp_path / "event_snapshot.json"

    handoff.export_completed_run_snapshot(
        db_path.parent.parent,
        output_path=product_path,
        object_ids=["chief", "deputy"],
        selector="event",
        event_id="failure:deputy",
    )
    product = json.loads(product_path.read_text(encoding="utf-8"))

    assert product["payload"]["selection"]["selector_kind"] == "event"
    assert product["payload"]["selection"]["sample_index"] == 1
    assert product["payload"]["selection"]["associated_event"]["event_id"] == "failure:deputy"
    assert validate_product(product, source_path=product_path).promotable is True


def test_snapshot_validator_rejects_non_atomic_epochs(tmp_path: Path) -> None:
    db_path = _run_pair(tmp_path / "source")
    product_path = tmp_path / "snapshot.json"
    handoff.export_completed_run_snapshot(
        db_path.parent.parent,
        output_path=product_path,
        object_ids=["chief", "deputy"],
        selector="final",
    )
    product = json.loads(product_path.read_text(encoding="utf-8"))
    product["payload"]["states"][1]["state"]["epoch"]["value"] += 1.0 / 86400.0
    product["product_id"] = compute_product_id(product)

    report = validate_product(product, source_path=product_path, verify_sources=False)

    assert report.valid is False
    assert any(item.code == "snapshot.epoch_mismatch" for item in report.errors)
