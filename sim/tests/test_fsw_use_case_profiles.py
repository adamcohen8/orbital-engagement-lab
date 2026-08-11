from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sim import SimulationConfig, SimulationSession
from sim.config import scenario_config_from_dict
from sim.flight_software import (
    USE_CASE_PROFILES,
    materialize_use_case_profile,
    resolve_use_case_profile,
    use_case_profiles,
    validate_use_case_profile_catalog,
)


def _scenario(flight_software: dict[str, object], *, output_dir: Path | None = None) -> dict[str, object]:
    return {
        "scenario_name": "fsw_profile_selection",
        "objects": {
            "sat": {
                "kind": "satellite",
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
                "flight_software": flight_software,
            }
        },
        "simulator": {
            "duration_s": 1.0,
            "dt_s": 1.0,
            "dynamics": {"attitude": {"enabled": False}},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(output_dir) if output_dir is not None else "outputs/fsw_profile_selection",
            "mode": "save" if output_dir is not None else "interactive",
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
            "review": {"enabled": output_dir is not None, "detail": "standard"},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def test_profile_catalog_is_valid_json_serializable_and_broad() -> None:
    assert validate_use_case_profile_catalog() == ()
    assert len(USE_CASE_PROFILES) >= 15
    assert {item.domain for item in USE_CASE_PROFILES} >= {"attitude", "orbit", "rpo", "low_thrust"}
    maturity = {item.profile_id: item.maturity.value for item in USE_CASE_PROFILES}
    supported = {
        "fsw.profile.coast_monitor.v1",
        "fsw.profile.adcs_commissioning.v1",
        "fsw.profile.adcs_nadir_payload.v1",
        "fsw.profile.adcs_sun_pointing.v1",
        "fsw.profile.adcs_target_tracking.v1",
        "fsw.profile.orbit_maneuver_execution.v1",
        "fsw.profile.leo_stationkeeping.v1",
        "fsw.profile.orbital_element_maintenance.v1",
        "fsw.profile.atmospheric_pass_recovery.v1",
        "fsw.profile.rpo_far_field_rendezvous.v1",
        "fsw.profile.rpo_formation_hold.v1",
        "fsw.profile.rpo_corridor_approach.v1",
        "fsw.profile.rpo_waypoint_inspection.v1",
        "fsw.profile.rpo_terminal_proximity.v1",
        "fsw.profile.rpo_passive_retreat.v1",
        "fsw.profile.rpo_conjunction_response.v1",
        "fsw.profile.low_thrust_phasing.v1",
        "fsw.profile.low_thrust_element_maintenance.v1",
    }
    assert set(maturity) == supported
    assert maturity["fsw.profile.coast_monitor.v1"] == "supported"
    assert all(
        maturity[profile_id] == "supported"
        for profile_id in supported
    )
    qualification = {item.profile_id: item.qualification_status for item in USE_CASE_PROFILES}
    assert qualification["fsw.profile.coast_monitor.v1"] == "supported"
    assert all(
        qualification[profile_id] == "supported"
        for profile_id in supported
    )
    json.dumps([item.to_dict() for item in USE_CASE_PROFILES], allow_nan=False)


def test_profile_catalog_can_be_filtered_by_domain() -> None:
    rpo = use_case_profiles(domain="RPO")
    assert rpo
    assert all(item.domain == "rpo" for item in rpo)


def test_profile_materialization_requires_mission_specific_inputs() -> None:
    with pytest.raises(ValueError, match="requires params: reference_object_id"):
        materialize_use_case_profile("fsw.profile.rpo_far_field_rendezvous.v1")
    with pytest.raises(ValueError, match="requires params: scheduled_burns"):
        materialize_use_case_profile(
            "fsw.profile.orbit_maneuver_execution.v1",
            params={"scheduled_burns": []},
        )


def test_profile_materialization_merges_defaults_and_mission_inputs() -> None:
    selection = materialize_use_case_profile(
        "fsw.profile.rpo_far_field_rendezvous.v1",
        params={"reference_object_id": "chief"},
        hardware_profile="hardware.rcs.v1",
        task_period_s=0.2,
    )
    assert selection.stack_id == "fsw.rpo_reference"
    assert selection.hardware_profile == "hardware.rcs.v1"
    assert selection.task_period_s == 0.2
    assert selection.params["translation_mode"] == "ric_pd_transfer"
    assert selection.params["reference_object_id"] == "chief"
    assert selection.params["max_acceleration_m_s2"] == 0.01


def test_profile_materialization_enforces_mode_and_task_period_envelope() -> None:
    with pytest.raises(ValueError, match="qualified envelope"):
        materialize_use_case_profile("fsw.profile.coast_monitor.v1", task_period_s=999.0)
    with pytest.raises(ValueError, match="fixes params.translation_mode"):
        materialize_use_case_profile(
            "fsw.profile.rpo_corridor_approach.v1",
            params={"reference_object_id": "target", "translation_mode": "passive_retreat"},
        )
    with pytest.raises(ValueError, match="fixes params.max_acceleration_m_s2"):
        materialize_use_case_profile(
            "fsw.profile.rpo_corridor_approach.v1",
            params={"reference_object_id": "target", "max_acceleration_m_s2": 999.0},
        )
    with pytest.raises(ValueError, match="restricts params.max_acceleration_m_s2"):
        materialize_use_case_profile(
            "fsw.profile.low_thrust_phasing.v1",
            params={"reference_object_id": "target", "max_acceleration_m_s2": 999.0},
        )
    with pytest.raises(ValueError, match="fixes params.measurement_stale_after_s"):
        materialize_use_case_profile(
            "fsw.profile.coast_monitor.v1",
            params={"measurement_stale_after_s": 1.0e9},
        )


def test_profile_config_resolves_and_round_trips_with_provenance() -> None:
    raw = _scenario(
        {
            "profile": "fsw.profile.rpo_formation_hold.v1",
            "params": {
                "reference_object_id": "chief",
                "target_relative_state_ric_m": [0.0, 500.0, 0.0, 0.0, 0.0, 0.0],
            },
        }
    )
    first = scenario_config_from_dict(raw)
    section = first.objects["sat"].flight_software
    assert section is not None
    assert section.profile == "fsw.profile.rpo_formation_hold.v1"
    assert section.stack == "fsw.rpo_reference"
    assert section.hardware_profile == "hardware.ideal_wrench.v1"
    assert section.task_period_s == 0.5
    assert section.params["translation_mode"] == "ric_hold"

    serialized = first.to_dict()
    normalized = serialized["objects"]["sat"]["flight_software"]
    assert normalized["profile"] == section.profile
    assert normalized["stack"] == section.stack
    second = scenario_config_from_dict(serialized)
    assert second.objects["sat"].flight_software == section


def test_profile_rejects_mismatched_stack_and_undeclared_hardware() -> None:
    with pytest.raises(ValueError, match="does not match profile"):
        scenario_config_from_dict(
            _scenario(
                {
                    "profile": "fsw.profile.adcs_nadir_payload.v1",
                    "stack": "fsw.orbit_reference",
                }
            )
        )
    with pytest.raises(ValueError, match="does not declare hardware profile"):
        materialize_use_case_profile(
            "fsw.profile.adcs_nadir_payload.v1",
            hardware_profile="hardware.ideal_wrench.v1",
        )


def test_profile_runtime_and_review_store_retain_profile_identity(tmp_path: Path) -> None:
    result = SimulationSession.from_config(
        SimulationConfig.from_dict(
            _scenario(
                {"profile": "fsw.profile.coast_monitor.v1"},
                output_dir=tmp_path,
            )
        )
    ).run()
    evidence = result.payload["flight_software_evidence_by_object"]["sat"]
    assert {row["profile_id"] for row in evidence["invocations"]} == {
        "fsw.profile.coast_monitor.v1"
    }
    db_path = Path(result.summary["review_outputs"]["sqlite"])
    with sqlite3.connect(db_path) as conn:
        profile_ids = {
            row[0]
            for row in conn.execute("SELECT DISTINCT profile_id FROM fsw_invocations")
        }
        timing = conn.execute(
            "SELECT modeled_execution_duration_ns, execution_budget_ns, deadline_missed, detail_json "
            "FROM fsw_task_timing WHERE task_id = 'stack.step' ORDER BY invocation_id"
        ).fetchall()
    assert profile_ids == {"fsw.profile.coast_monitor.v1"}
    assert timing
    assert all(row[0] is not None and row[0] >= 0 for row in timing)
    assert all(row[1] == 1_000_000_000 for row in timing)
    assert all("release_reasons" in json.loads(row[3]) for row in timing)
    assert all(row[2] in {0, 1} for row in timing)
    assert all(
        row["profile_params"] == {"measurement_stale_after_s": 30.0}
        for row in evidence["invocations"]
    )


def test_profile_details_expose_maturation_contract() -> None:
    profile = resolve_use_case_profile("fsw.profile.low_thrust_element_maintenance.v1")
    assert "Monte Carlo robustness" in profile.qualification_gates
    assert profile.known_limits
    assert profile.required_parameters[0].name == "target_coes"
