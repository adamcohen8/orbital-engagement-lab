from __future__ import annotations

import json
from pathlib import Path

import yaml

from sim import SimulationConfig, SimulationSession
from sim.reporting import build_maneuver_readiness_packet


def test_integrated_run_emits_explicit_maneuver_readiness_evidence(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/quickstart_5min.yaml").read_text(encoding="utf-8")
    )
    config["objects"]["chaser"]["specs"].pop("thruster", None)
    config["objects"]["chaser"]["specs"]["actuator_preset"] = "BASIC_RCS_6DOF"
    config["simulator"]["duration_s"] = 30.0
    config["outputs"]["output_dir"] = str(tmp_path / "run")
    config["outputs"]["stats"]["save_full_log"] = True
    config["outputs"]["stats"]["controller_debug"] = True
    config["outputs"]["stats"]["print_summary"] = False
    result = SimulationSession.from_config(SimulationConfig.from_dict(config)).run()
    completed_run = Path(result.summary["review_outputs"]["sqlite"]).parent.parent

    packet = build_maneuver_readiness_packet(
        completed_run,
        object_id="chaser",
        chief_id="target",
        thresholds={
            "max_final_range_km": 3.1,
            "max_allocation_force_residual_n": 5.0,
            "max_allocation_saturated_duration_s": 30.0,
            "max_pointing_error_deg": 180.0,
            "min_final_propellant_kg": 1.0,
            "min_burn_samples": 1,
            "require_no_attitude_guardrail_events": True,
        },
        output_path=tmp_path / "maneuver_readiness.json",
    )

    assert packet["verdict"] == "ready"
    assert all(item["status"] == "pass" for item in packet["gates"])
    assert packet["metrics"]["max_allocation_force_residual_n"] is not None
    assert packet["metrics"]["allocation_saturated_duration_s"] is not None
    assert packet["metrics"]["max_pointing_error_deg"] is not None
    assert packet["metrics"]["final_propellant_remaining_kg"] is not None
    saved = json.loads((tmp_path / "maneuver_readiness.json").read_text(encoding="utf-8"))
    assert saved["schema_id"] == "oel-maneuver-readiness-packet-v1"
    assert saved["evidence"]["review_store_sha256"]


def test_missing_required_metric_produces_unknown_not_optimistic_readiness(tmp_path: Path) -> None:
    config = {
        "scenario_name": "readiness_unknown_fixture",
        "objects": {
            "target": {
                "enabled": True,
                "role": "target",
                "kind": "satellite",
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
            "chaser": {
                "enabled": True,
                "role": "chaser",
                "kind": "satellite",
                "initial_state": {
                    "position_eci_km": [7001.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
        },
        "simulator": {
            "duration_s": 1.0,
            "dt_s": 1.0,
            "dynamics": {"orbit": {"model": "two_body"}, "attitude": {"enabled": False}},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(tmp_path / "run"),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": True},
            "review": {"enabled": True},
            "plots": {"enabled": False},
            "animations": {"enabled": False},
        },
    }
    result = SimulationSession.from_config(SimulationConfig.from_dict(config)).run()
    packet = build_maneuver_readiness_packet(
        Path(result.summary["review_outputs"]["sqlite"]).parent.parent,
        object_id="chaser",
        chief_id="target",
        thresholds={"max_pointing_error_deg": 10.0},
    )

    assert packet["verdict"] == "unknown"
    assert packet["gates"][0]["status"] == "unknown"
