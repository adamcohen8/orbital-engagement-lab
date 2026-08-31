from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from sim.analysis.conjunction_workflow import propagate_history
from sim.analysis.history_adapters import AnalysisHistory
from sim.analysis.mission_scheduling import (
    MissionSchedulingProblem,
    solve_mission_schedule,
    write_mission_scheduling_artifacts,
)
from sim.analysis.spacecraft_power import (
    SPACECRAFT_POWER_EVIDENCE_SCHEMA,
    SPACECRAFT_POWER_HISTORY_SCHEMA,
    SPACECRAFT_POWER_PROBLEM_SCHEMA,
    SpacecraftPowerError,
    SpacecraftPowerProblem,
    assess_spacecraft_power,
    power_history_from_mapping,
    power_history_to_dict,
    problem_with_mission_schedule,
    verify_spacecraft_power_artifacts,
    write_spacecraft_power_artifacts,
)
from sim.analysis.trajectory_targeting import PropagationSettings
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.epoch import resolve_sun_moon_positions
from sim.installation.cli import _dispatch_commands
from sim.spacecraft_power import main as power_main

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_EXAMPLE = ROOT / "examples/mission_scheduling/public_two_asset_collection_problem.json"
EPOCH_JD = 2461041.5


def _problem(*, horizon_s: float = 3600.0) -> dict:
    return {
        "schema_version": SPACECRAFT_POWER_PROBLEM_SCHEMA,
        "analysis_id": "public-power-hand-case",
        "asset_id": "SAT-A",
        "epoch_jd_utc": EPOCH_JD,
        "horizon_start_s": 0.0,
        "horizon_end_s": horizon_s,
        "integration_step_s": min(30.0, horizon_s / 2.0),
        "transition_time_tolerance_s": 0.01,
        "transition_max_iterations": 80,
        "shadow_model": "none",
        "ephemeris_model": "analytic_enhanced",
        "orientation_mode": "sun_tracking_ideal",
        "solar_array": {
            "area_m2": 1.0,
            "efficiency": 0.2,
            "solar_flux_w_m2": 1000.0,
            "maximum_generation_w": 200.0,
            "normal_body": [1.0, 0.0, 0.0],
        },
        "battery": {
            "capacity_wh": 100.0,
            "initial_soc_fraction": 0.5,
            "minimum_soc_fraction": 0.0,
            "maximum_soc_fraction": 1.0,
            "maximum_charge_power_w": 1000.0,
            "maximum_discharge_power_w": 1000.0,
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
        },
        "base_load_w": 100.0,
        "activities": [],
    }


def _constant_history(*, horizon_s: float = 3600.0, attitude: bool = False) -> AnalysisHistory:
    sun, _ = resolve_sun_moon_positions(
        {"jd_utc_start": EPOCH_JD, "ephemeris_mode": "analytic_enhanced"}, 0.0
    )
    position = 7000.0 * sun / np.linalg.norm(sun)
    times = np.array([0.0, horizon_s], dtype=float)
    attitudes = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1)) if attitude else None
    return AnalysisHistory(
        object_id="SAT-A",
        product_kind="synthetic_hand_case",
        state_provider_id="test:constant-history",
        frame="eci",
        initial_jd_utc=EPOCH_JD,
        times_s=times,
        position_eci_km=np.tile(position, (2, 1)),
        velocity_eci_km_s=np.zeros((2, 3)),
        attitude_quat_bn=attitudes,
        attitude_source_kind="analytic_ideal" if attitude else "not_required",
        attitude_provider_id="test:identity-attitude" if attitude else None,
    )


def _orbit_history(duration_s: float = 6000.0) -> AnalysisHistory:
    sun, _ = resolve_sun_moon_positions(
        {"jd_utc_start": EPOCH_JD, "ephemeris_mode": "analytic_enhanced"}, 0.0
    )
    radial = sun / np.linalg.norm(sun)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(radial, trial))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    cross_track = np.cross(radial, trial)
    cross_track /= np.linalg.norm(cross_track)
    in_track = np.cross(cross_track, radial)
    radius_km = 7000.0
    speed = np.sqrt(398600.4418 / radius_km)
    state = np.hstack((radius_km * radial, speed * in_track))
    propagated = propagate_history(state, duration_s, PropagationSettings(step_s=30.0))
    times, states = propagated.arrays()
    return AnalysisHistory(
        object_id="SAT-A",
        product_kind="onp_two_body_test",
        state_provider_id="test:onp-two-body",
        frame="eci",
        initial_jd_utc=EPOCH_JD,
        times_s=times,
        position_eci_km=states[:, :3],
        velocity_eci_km_s=states[:, 3:],
    )


def _grazing_eclipse_history(
    duration_s: float = 60.0,
    *,
    eclipse_center_s: float = 30.0,
) -> AnalysisHistory:
    sun, _ = resolve_sun_moon_positions(
        {"jd_utc_start": EPOCH_JD, "ephemeris_mode": "analytic_enhanced"}, 0.0
    )
    sun_direction = sun / np.linalg.norm(sun)
    cross_sun = np.array([0.0, 0.0, 1.0])
    cross_sun -= float(np.dot(cross_sun, sun_direction)) * sun_direction
    if np.linalg.norm(cross_sun) < 0.1:
        cross_sun = np.array([0.0, 1.0, 0.0])
        cross_sun -= float(np.dot(cross_sun, sun_direction)) * sun_direction
    cross_sun /= np.linalg.norm(cross_sun)
    radius_km = 7000.0
    mean_motion_rad_s = np.sqrt(EARTH_MU_KM3_S2 / radius_km**3)
    half_angle = mean_motion_rad_s * 10.0
    radius_ratio = EARTH_RADIUS_KM / radius_km
    beta_rad = np.arccos(np.sqrt((1.0 - radius_ratio**2) / np.cos(half_angle) ** 2))
    orbit_normal = np.sin(beta_rad) * sun_direction + np.cos(beta_rad) * cross_sun
    projected_sun = sun_direction - float(np.dot(sun_direction, orbit_normal)) * orbit_normal
    projected_sun /= np.linalg.norm(projected_sun)
    in_track = np.cross(orbit_normal, projected_sun)
    in_track /= np.linalg.norm(in_track)

    def state(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        angle = mean_motion_rad_s * (time_s - eclipse_center_s)
        position = radius_km * (-projected_sun * np.cos(angle) + in_track * np.sin(angle))
        velocity = radius_km * mean_motion_rad_s * (
            projected_sun * np.sin(angle) + in_track * np.cos(angle)
        )
        return position, velocity

    endpoints = [state(0.0), state(duration_s)]
    return AnalysisHistory(
        object_id="SAT-A",
        product_kind="analytic_circular_grazing_eclipse",
        state_provider_id="test:analytic-circular-grazing-eclipse",
        frame="eci",
        initial_jd_utc=EPOCH_JD,
        times_s=np.array([0.0, duration_s]),
        position_eci_km=np.asarray([item[0] for item in endpoints]),
        velocity_eci_km_s=np.asarray([item[1] for item in endpoints]),
    )


def test_constant_sunlight_energy_balance_and_saturation() -> None:
    result = assess_spacecraft_power(_problem(), _constant_history())

    assert result.summary["schema_version"] == SPACECRAFT_POWER_EVIDENCE_SCHEMA
    assert result.summary["status"] == "completed"
    assert result.summary["feasibility"] == "feasible"
    assert result.summary["totals"]["generated_energy_wh"] == pytest.approx(200.0)
    assert result.summary["totals"]["load_energy_wh"] == pytest.approx(100.0)
    assert result.summary["totals"]["charged_battery_energy_wh"] == pytest.approx(50.0)
    assert result.summary["totals"]["curtailed_energy_wh"] == pytest.approx(50.0)
    assert result.summary["battery"]["final_soc_fraction"] == pytest.approx(1.0)
    assert max(abs(value) for value in result.summary["conservation_residuals_wh"].values()) < 1.0e-10
    assert any(event.event_kind == "battery_maximum_soc" for event in result.events)


def test_discharge_limit_and_reserve_produce_explicit_unmet_load() -> None:
    payload = _problem()
    payload["solar_array"]["maximum_generation_w"] = 1.0
    payload["battery"]["initial_soc_fraction"] = 1.0
    payload["battery"]["maximum_discharge_power_w"] = 50.0
    payload["base_load_w"] = 101.0

    result = assess_spacecraft_power(payload, _constant_history())

    assert result.summary["feasibility"] == "infeasible"
    assert result.summary["totals"]["unmet_load_energy_wh"] == pytest.approx(50.0)
    assert result.summary["battery"]["final_soc_fraction"] == pytest.approx(0.5)
    unmet = next(event for event in result.events if event.event_kind == "unmet_load_start")
    assert unmet.time_s == pytest.approx(0.0)


def test_battery_reserve_event_is_timed_inside_an_interval() -> None:
    payload = _problem(horizon_s=3600.0)
    payload["integration_step_s"] = 60.0
    payload["solar_array"]["maximum_generation_w"] = 1.0
    payload["battery"]["initial_soc_fraction"] = 1.0
    payload["battery"]["minimum_soc_fraction"] = 0.25
    payload["battery"]["maximum_discharge_power_w"] = 100.0
    payload["base_load_w"] = 101.0

    result = assess_spacecraft_power(payload, _constant_history())
    event = next(item for item in result.events if item.event_kind == "battery_minimum_soc")

    assert event.time_s == pytest.approx(2700.0, abs=1.0e-9)
    assert result.summary["totals"]["unmet_load_energy_wh"] == pytest.approx(25.0)
    assert result.summary["battery"]["final_soc_fraction"] == pytest.approx(0.25)


def test_unmet_load_onset_uses_reserve_depletion_time() -> None:
    payload = _problem(horizon_s=60.0)
    payload["integration_step_s"] = 60.0
    payload["solar_array"]["maximum_generation_w"] = 1.0
    payload["battery"]["capacity_wh"] = 1.0
    payload["battery"]["initial_soc_fraction"] = 0.5
    payload["base_load_w"] = 101.0

    result = assess_spacecraft_power(payload, _constant_history(horizon_s=60.0))
    reserve = next(item for item in result.events if item.event_kind == "battery_minimum_soc")
    unmet = next(item for item in result.events if item.event_kind == "unmet_load_start")

    assert reserve.time_s == pytest.approx(18.0)
    assert unmet.time_s == pytest.approx(reserve.time_s)


def test_real_two_body_history_resolves_sunlight_penumbra_and_umbra() -> None:
    history = _orbit_history()
    payload = _problem(horizon_s=6000.0)
    payload["shadow_model"] = "conical"
    payload["integration_step_s"] = 30.0

    result = assess_spacecraft_power(payload, history)
    classes = {item.illumination_class for item in result.intervals}

    assert classes == {"sunlight", "penumbra", "umbra"}
    transitions = [item for item in result.events if item.event_kind == "illumination_transition"]
    assert len(transitions) >= 4
    assert all(item.disposition == "provider_refined" for item in transitions)
    assert all(item.bracket_end_s - item.bracket_start_s <= 0.01 for item in transitions)


def test_cylindrical_eclipse_duration_matches_closed_form_circular_case() -> None:
    radius_km = 7000.0
    payload = _problem(horizon_s=6000.0)
    payload["shadow_model"] = "cylindrical"
    result = assess_spacecraft_power(payload, _orbit_history())

    retained_umbra_s = sum(
        item.duration_s for item in result.intervals if item.illumination_class == "umbra"
    )
    mean_motion_rad_s = np.sqrt(EARTH_MU_KM3_S2 / radius_km**3)
    expected_umbra_s = 2.0 * np.arcsin(EARTH_RADIUS_KM / radius_km) / mean_motion_rad_s

    assert retained_umbra_s == pytest.approx(expected_umbra_s, abs=0.05)


@pytest.mark.parametrize("eclipse_center_s", (30.0, 22.5))
def test_whole_grazing_eclipse_inside_one_step_is_discovered_and_refined(
    eclipse_center_s: float,
) -> None:
    payload = _problem(horizon_s=60.0)
    payload["shadow_model"] = "cylindrical"
    payload["integration_step_s"] = 60.0
    payload["battery"]["initial_soc_fraction"] = 0.0

    result = assess_spacecraft_power(
        payload,
        _grazing_eclipse_history(eclipse_center_s=eclipse_center_s),
    )
    transitions = [item for item in result.events if item.event_kind == "illumination_transition"]

    assert {item.illumination_class for item in result.intervals} == {"sunlight", "umbra"}
    assert len(transitions) == 2
    assert all(item.disposition == "provider_refined" for item in transitions)
    assert all(item.bracket_end_s - item.bracket_start_s <= 0.01 for item in transitions)
    assert all(item.original_bracket_start_s == 0.0 for item in transitions)
    assert all(item.original_bracket_end_s == 60.0 for item in transitions)
    assert result.summary["totals"]["generated_energy_wh"] > 2.0
    if eclipse_center_s == 30.0:
        assert result.summary["totals"]["unmet_load_energy_wh"] == pytest.approx(0.0)
        assert result.summary["feasibility"] == "feasible"


def test_transition_refinement_fails_if_iteration_limit_cannot_meet_tolerance() -> None:
    payload = _problem(horizon_s=6000.0)
    payload["shadow_model"] = "conical"
    payload["integration_step_s"] = 60.0
    payload["transition_time_tolerance_s"] = 1.0e-6
    payload["transition_max_iterations"] = 1

    with pytest.raises(SpacecraftPowerError, match="exhausted transition_max_iterations"):
        assess_spacecraft_power(payload, _orbit_history())


def test_power_integration_is_stable_under_step_refinement() -> None:
    history = _orbit_history()
    coarse = _problem(horizon_s=6000.0)
    coarse["shadow_model"] = "conical"
    coarse["integration_step_s"] = 30.0
    fine = deepcopy(coarse)
    fine["integration_step_s"] = 15.0

    coarse_result = assess_spacecraft_power(coarse, history)
    fine_result = assess_spacecraft_power(fine, history)

    assert coarse_result.summary["totals"]["generated_energy_wh"] == pytest.approx(
        fine_result.summary["totals"]["generated_energy_wh"], abs=0.001
    )
    assert coarse_result.summary["battery"]["minimum_soc_fraction"] == pytest.approx(
        fine_result.summary["battery"]["minimum_soc_fraction"], abs=0.0005
    )


def test_body_fixed_orientation_requires_attitude_and_respects_array_normal() -> None:
    payload = _problem()
    payload["orientation_mode"] = "history_body_fixed"
    with pytest.raises(SpacecraftPowerError, match="requires retained attitude"):
        assess_spacecraft_power(payload, _constant_history())

    history = _constant_history(attitude=True)
    sun, _ = resolve_sun_moon_positions(
        {"jd_utc_start": EPOCH_JD, "ephemeris_mode": "analytic_enhanced"}, 0.0
    )
    payload["solar_array"]["normal_body"] = list(sun / np.linalg.norm(sun))
    result = assess_spacecraft_power(payload, history)
    assert result.samples[0].incidence_cosine > 0.999999


def test_history_round_trip_is_semantically_exact() -> None:
    history = _orbit_history(300.0)
    payload = power_history_to_dict(history)
    restored = power_history_from_mapping(payload)

    assert payload["schema_version"] == SPACECRAFT_POWER_HISTORY_SCHEMA
    assert power_history_to_dict(restored) == payload


def test_published_schema_accepts_generated_records(tmp_path: Path) -> None:
    schema_path = ROOT / "docs/contracts/schemas/oel-spacecraft-power-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    history = _orbit_history(300.0)
    problem = SpacecraftPowerProblem.from_mapping(_problem(horizon_s=300.0))
    result = assess_spacecraft_power(problem, history)
    artifacts = write_spacecraft_power_artifacts(result, history, tmp_path / "schema")

    for value in (
        problem.to_dict(),
        power_history_to_dict(history),
        result.summary,
        json.loads(artifacts.manifest_json.read_text(encoding="utf-8")),
    ):
        validator.validate(value)


def test_artifacts_replay_and_tamper_fail_closed(tmp_path: Path) -> None:
    history = _constant_history()
    result = assess_spacecraft_power(_problem(), history)
    artifacts = write_spacecraft_power_artifacts(result, history, tmp_path / "power")

    replay = verify_spacecraft_power_artifacts(artifacts.output_dir)
    assert replay["status"] == "verified"
    assert replay["result_semantic_sha256"] == result.summary["result_semantic_sha256"]
    with pytest.raises(SpacecraftPowerError, match="must not already exist"):
        write_spacecraft_power_artifacts(result, history, artifacts.output_dir)

    artifacts.timeseries_csv.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SpacecraftPowerError, match="receipt mismatch"):
        verify_spacecraft_power_artifacts(artifacts.output_dir)


def test_replay_rejects_self_consistent_forged_derived_csv(tmp_path: Path) -> None:
    history = _constant_history()
    artifacts = write_spacecraft_power_artifacts(
        assess_spacecraft_power(_problem(), history), history, tmp_path / "power"
    )
    forged = b"time_s,forged\n0,true\n"
    artifacts.timeseries_csv.write_bytes(forged)
    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    receipt = next(
        item for item in manifest["artifacts"] if item["path"] == artifacts.timeseries_csv.name
    )
    receipt["bytes"] = len(forged)
    receipt["sha256"] = hashlib.sha256(forged).hexdigest()
    artifacts.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(SpacecraftPowerError, match="authoritative replay"):
        verify_spacecraft_power_artifacts(artifacts.output_dir)


def test_schedule_adapter_verifies_and_binds_selected_activity_loads(tmp_path: Path) -> None:
    schedule_problem = MissionSchedulingProblem.from_mapping(
        json.loads(SCHEDULE_EXAMPLE.read_text(encoding="utf-8"))
    )
    schedule = write_mission_scheduling_artifacts(
        solve_mission_schedule(schedule_problem), tmp_path / "schedule"
    )
    payload = _problem(horizon_s=120.0)
    history = _constant_history(horizon_s=120.0)
    converted = problem_with_mission_schedule(
        payload,
        schedule.output_dir,
        activity_power_w={"observation": 120.0, "downlink": 80.0},
    )

    assert [item.category for item in converted.activities] == ["observation", "downlink"]
    assert len({item.source_product_sha256 for item in converted.activities}) == 1
    result = assess_spacecraft_power(converted, history)
    assert result.summary["source_product_sha256s"] == [
        converted.activities[0].source_product_sha256
    ]


def test_schedule_adapter_rejects_receipt_updated_forged_schedule_csv(tmp_path: Path) -> None:
    schedule_problem = MissionSchedulingProblem.from_mapping(
        json.loads(SCHEDULE_EXAMPLE.read_text(encoding="utf-8"))
    )
    schedule = write_mission_scheduling_artifacts(
        solve_mission_schedule(schedule_problem), tmp_path / "schedule"
    )
    with schedule.schedule_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["start_s"] = str(float(rows[0]["start_s"]) + 1.0)
    rows[0]["end_s"] = str(float(rows[0]["end_s"]) + 1.0)
    with schedule.schedule_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    content = schedule.schedule_csv.read_bytes()
    manifest = json.loads(schedule.manifest_json.read_text(encoding="utf-8"))
    receipt = next(
        item for item in manifest["artifacts"] if item["path"] == "mission_schedule.csv"
    )
    receipt["bytes"] = len(content)
    receipt["sha256"] = hashlib.sha256(content).hexdigest()
    schedule.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(SpacecraftPowerError, match="did not verify"):
        problem_with_mission_schedule(
            _problem(horizon_s=120.0),
            schedule.output_dir,
            activity_power_w={"observation": 120.0, "downlink": 80.0},
        )


def test_strict_contract_rejects_unknown_nonfinite_and_mismatched_inputs() -> None:
    unknown = _problem()
    unknown["surprise"] = True
    with pytest.raises(SpacecraftPowerError, match="unknown fields"):
        SpacecraftPowerProblem.from_mapping(unknown)

    nonfinite = _problem()
    nonfinite["base_load_w"] = float("nan")
    with pytest.raises(SpacecraftPowerError, match="finite"):
        SpacecraftPowerProblem.from_mapping(nonfinite)

    history = _constant_history()
    mismatch = _problem()
    mismatch["asset_id"] = "OTHER"
    with pytest.raises(SpacecraftPowerError, match="asset_id"):
        assess_spacecraft_power(mismatch, history)

    direct = deepcopy(_problem())
    direct["solar_array"]["efficiency"] = float("nan")
    with pytest.raises(SpacecraftPowerError, match="finite"):
        SpacecraftPowerProblem.from_mapping(direct)


def test_cli_analyze_schedule_replay_and_error_paths(tmp_path: Path, capsys) -> None:
    assert "power" in _dispatch_commands()
    problem_path = tmp_path / "problem.json"
    history_path = tmp_path / "history.json"
    problem_path.write_text(json.dumps(_problem(horizon_s=120.0)), encoding="utf-8")
    history_path.write_text(
        json.dumps(power_history_to_dict(_constant_history(horizon_s=120.0))), encoding="utf-8"
    )
    schedule_problem = MissionSchedulingProblem.from_mapping(
        json.loads(SCHEDULE_EXAMPLE.read_text(encoding="utf-8"))
    )
    schedule = write_mission_scheduling_artifacts(
        solve_mission_schedule(schedule_problem), tmp_path / "schedule"
    )
    output = tmp_path / "power"

    assert power_main(
        [
            "analyze",
            str(problem_path),
            str(history_path),
            "--output-dir",
            str(output),
            "--mission-schedule",
            str(schedule.output_dir),
            "--observation-load-w",
            "120",
            "--downlink-load-w",
            "80",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert power_main(["replay", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"

    invalid = deepcopy(_problem())
    invalid["asset_id"] = "OTHER"
    problem_path.write_text(json.dumps(invalid), encoding="utf-8")
    assert power_main(["validate", str(problem_path), str(history_path)]) == 2
    assert "asset_id" in json.loads(capsys.readouterr().out)["message"]
    assert power_main(
        ["analyze", str(problem_path), str(history_path), "--output-dir", str(tmp_path / "bad")]
    ) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"
