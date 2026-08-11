from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import sim.handoff as handoff
from sim import SimulationConfig, SimulationSession
from sim.api import SimulationWorkspace
from sim.interchange.validation import validate_product
from sim.review import ReviewWorkspace


def _source_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "satellite_checkpoint_source",
        "metadata": {
            "owner": "public",
            "public_surface": "public-agent-workflow",
            "export_review": {"approved_for_public_export": True},
        },
        "objects": {
            "sat": {
                "enabled": True,
                "role": "chaser",
                "kind": "satellite",
                "specs": {
                    "dry_mass_kg": 10.0,
                    "fuel_mass_kg": 1.0,
                    "mass_kg": 11.0,
                    "isp_s": 200.0,
                    "mass_properties": {
                        "inertia_reference_point": "center_of_mass",
                        "inertia_kg_m2": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                    },
                },
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                    "attitude_quat_bn": [0.996194698, 0.0, 0.087155743, 0.0],
                    "angular_rate_body_rad_s": [0.001, -0.002, 0.0005],
                },
                "flight_software": {
                    "stack": "fsw.orbit_reference",
                    "hardware_profile": "hardware.ideal_wrench.v1",
                    "task_period_s": 1.0,
                    "params": {
                        "navigation_initialization": "ideal",
                        "assumed_mass_kg": 11.0,
                        "translation_mode": "scheduled_burn",
                        "max_acceleration_m_s2": 0.01,
                        "max_force_n": 0.1,
                        "scheduled_burns": [
                            {
                                "start_time_s": 0.0,
                                "duration_s": 2.0,
                                "delta_v_m_s": [0.02, 0.0, 0.0],
                                "frame": "eci",
                            }
                        ],
                    },
                },
            }
        },
        "simulator": {
            "initial_jd_utc": 2461254.5,
            "duration_s": 2.0,
            "dt_s": 1.0,
            "dynamics": {
                "orbit": {"model": "two_body", "orbit_substep_s": 1.0},
                "attitude": {"enabled": True, "attitude_substep_s": 0.25},
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


def _run_source(output_dir: Path) -> None:
    config = SimulationConfig.from_dict(_source_config(output_dir))
    SimulationSession.from_config(config).run()


def test_completed_run_products_preserve_current_mass_and_remaining_propellant(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _run_source(source)
    final_mass = ReviewWorkspace.open(source).query(
        "SELECT mass_kg FROM object_state WHERE object_id = 'sat' ORDER BY sample_index DESC LIMIT 1"
    ).rows[0]["mass_kg"]
    assert final_mass < 11.0

    product_path = tmp_path / "state.json"
    handoff.export_completed_run_state(source, output_path=product_path, object_id="sat", selector="final")
    product = json.loads(product_path.read_text(encoding="utf-8"))
    resource = product["payload"]["resource_state"]
    assert resource["mass_kg"] == pytest.approx(final_mass)
    assert resource["dry_mass_kg"] == 10.0
    assert resource["fuel_mass_kg"] == pytest.approx(final_mass - 10.0)

    scenario_path = tmp_path / "passive.yaml"
    result = handoff.materialize_onp(
        product_path,
        scenario_name="mass_preserving_passive_branch",
        scenario_path=scenario_path,
        output_dir=tmp_path / "passive_run",
        duration_s=1.0,
        dt_s=1.0,
        trust_plugins=True,
    )
    assert result["status"] == "materialized"
    specs = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))["objects"]["sat"]["specs"]
    assert specs["mass_kg"] == pytest.approx(final_mass)
    assert specs["fuel_mass_kg"] == pytest.approx(final_mass - 10.0)


def test_satellite_checkpoint_materializes_and_restores_complete_runtime(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _run_source(source)
    product_path = tmp_path / "satellite_checkpoint.json"
    exported = handoff.export_satellite_checkpoint(
        source,
        output_path=product_path,
        object_id="sat",
    )
    product = json.loads(product_path.read_text(encoding="utf-8"))
    report = validate_product(product, source_path=product_path)

    assert exported["status"] == "exported"
    assert report.promotable is True
    assert product["product_kind"] == "oel.satellite_checkpoint"
    assert product["payload"]["checkpoint"]["checkpoint_schema"] == "oel.satellite_runtime_checkpoint.v1"
    assert product["payload"]["checkpoint"]["profile_id"] is None

    generic = handoff.materialize_onp(
        product_path,
        scenario_name="unsafe_passive_downgrade",
        scenario_path=tmp_path / "unsafe_passive.yaml",
        output_dir=tmp_path / "unsafe_passive_run",
        duration_s=1.0,
        dt_s=1.0,
    )
    assert generic["status"] == "blocked"
    assert generic["failures"][0]["code"] == "compatibility.product_kind"
    assert not (tmp_path / "unsafe_passive.yaml").exists()

    scenario_path = tmp_path / "active.yaml"
    continued_output = tmp_path / "active_run"
    materialized = handoff.materialize_satellite_checkpoint(
        product_path,
        scenario_name="active_checkpoint_branch",
        scenario_path=scenario_path,
        output_dir=continued_output,
        duration_s=2.0,
        dt_s=1.0,
        trust_plugins=True,
    )
    assert materialized["status"] == "materialized"
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    assert scenario["objects"]["sat"]["flight_software"]["stack"] == "fsw.orbit_reference"
    assert scenario["objects"]["sat"]["flight_software"]["checkpoint"]["state_hash_sha256"] == (
        product["payload"]["checkpoint"]["state_hash_sha256"]
    )

    SimulationWorkspace().run(scenario_path)
    first = ReviewWorkspace.open(continued_output).query(
        "SELECT pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, "
        "vel_z_eci_km_s, quat_w, quat_x, quat_y, quat_z, omega_x_rad_s, omega_y_rad_s, "
        "omega_z_rad_s, mass_kg FROM object_state WHERE object_id = 'sat' AND sample_index = 0"
    ).rows[0]
    payload = product["payload"]
    assert list(first.values())[:6] == pytest.approx(payload["state"]["values"])
    assert list(first.values())[6:10] == pytest.approx(payload["attitude"]["quaternion_bn"])
    assert list(first.values())[10:13] == pytest.approx(payload["attitude"]["angular_rate_body_rad_s"])
    assert first["mass_kg"] == pytest.approx(payload["resource_state"]["mass_kg"])
    invocations = ReviewWorkspace.open(continued_output).query(
        "SELECT MIN(invocation_id) AS first_id, MAX(invocation_id) AS last_id FROM fsw_invocations"
    ).rows[0]
    assert invocations["first_id"] > product["payload"]["checkpoint"]["invocation_id"]


def test_hash_only_legacy_snapshot_cannot_be_promoted_to_satellite_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _run_source(source)
    db = source / "review" / "run.sqlite"
    import sqlite3

    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE fsw_snapshots SET detail_json = '{}' WHERE object_id = 'sat'")
        connection.commit()

    with pytest.raises(handoff.SatelliteCheckpointError, match="hash-only legacy"):
        handoff.export_satellite_checkpoint(
            source,
            output_path=tmp_path / "legacy.json",
            object_id="sat",
        )
