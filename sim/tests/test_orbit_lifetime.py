from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from sim.analysis.orbit_lifetime import (
    ORBIT_LIFETIME_COMPARISON_EVIDENCE_SCHEMA,
    ORBIT_LIFETIME_COMPARISON_PROBLEM_SCHEMA,
    ORBIT_LIFETIME_EVIDENCE_SCHEMA,
    ORBIT_LIFETIME_PROBLEM_SCHEMA,
    LifetimeAtmosphere,
    OrbitLifetimeComparisonProblem,
    OrbitLifetimeError,
    OrbitLifetimeProblem,
    assess_orbit_lifetime,
    compare_orbit_lifetime_models,
    verify_orbit_lifetime_artifacts,
    verify_orbit_lifetime_comparison_artifacts,
    write_orbit_lifetime_artifacts,
    write_orbit_lifetime_comparison_artifacts,
)
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.elements import coe_to_rv_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin
from sim.installation.cli import _dispatch_commands
from sim.orbit_lifetime import main as lifetime_main

ROOT = Path(__file__).resolve().parents[2]
EPOCH_JD = 2461041.5


def _problem(
    *,
    altitude_km: float = 200.0,
    duration_s: float = 7200.0,
    integration_step_s: float = 10.0,
) -> dict:
    radius_km = 6378.137 + altitude_km
    position, velocity = coe_to_rv_eci(
        a_km=radius_km,
        ecc=0.0,
        inc_deg=51.6,
        raan_deg=20.0,
        argp_deg=30.0,
        true_anomaly_deg=40.0,
    )
    return {
        "schema_version": ORBIT_LIFETIME_PROBLEM_SCHEMA,
        "analysis_id": "public-lifetime-test",
        "asset_id": "SAT-A",
        "epoch_jd_utc": EPOCH_JD,
        "initial_position_eci_km": position.tolist(),
        "initial_velocity_eci_km_s": velocity.tolist(),
        "duration_s": duration_s,
        "integration_step_s": integration_step_s,
        "output_step_s": max(integration_step_s, 300.0),
        "transition_time_tolerance_s": 0.01,
        "transition_max_iterations": 80,
        "mass_kg": 100.0,
        "drag_area_m2": 10.0,
        "drag_coefficient": 2.2,
        "drag_enabled": True,
        "include_j2": False,
        "stop_at_reentry": True,
        "atmosphere": {
            "model": "exponential",
            "parameters": {
                "reference_density_kg_m3": 1.0e-9,
                "reference_altitude_km": 200.0,
                "scale_height_km": 30.0,
                "ceiling_altitude_km": 1000.0,
            },
        },
        "thresholds": {
            "warning_altitude_km": 190.0,
            "disposal_altitude_km": 150.0,
            "reentry_altitude_km": 120.0,
        },
    }


def _comparison() -> dict:
    base = _problem(duration_s=1800.0)
    base["stop_at_reentry"] = False
    return {
        "schema_version": ORBIT_LIFETIME_COMPARISON_PROBLEM_SCHEMA,
        "comparison_id": "public-atmosphere-comparison-test",
        "base_problem": base,
        "cases": [
            {
                "case_id": "low-density",
                "atmosphere": {
                    "model": "constant",
                    "parameters": {"density_kg_m3": 1.0e-13},
                },
            },
            {
                "case_id": "high-density",
                "atmosphere": {
                    "model": "constant",
                    "parameters": {"density_kg_m3": 1.0e-9},
                },
            },
        ],
    }


def test_drag_off_conserves_two_body_energy_and_angular_momentum() -> None:
    orbital_period_s = 2.0 * np.pi * np.sqrt((6578.137**3) / EARTH_MU_KM3_S2)
    payload = _problem(duration_s=orbital_period_s, integration_step_s=5.0)
    payload["output_step_s"] = 300.0
    payload["drag_enabled"] = False
    result = assess_orbit_lifetime(payload)

    assert result.summary["outcome"] == "horizon_complete"
    assert result.summary["propagator"]["force_models"] == []
    assert abs(result.summary["changes"]["specific_energy_km2_s2"]) < 1.0e-9
    assert abs(result.summary["changes"]["angular_momentum_km2_s"]) < 4.0e-8
    assert abs(result.summary["energy_accounting"]["residual_km2_s2"]) < 1.0e-9


def test_constant_density_short_arc_matches_instantaneous_drag_energy_limit() -> None:
    payload = _problem(duration_s=0.1, integration_step_s=0.1)
    payload["output_step_s"] = 0.1
    payload["atmosphere"] = {
        "model": "constant",
        "parameters": {"density_kg_m3": 2.0e-10},
    }
    problem = OrbitLifetimeProblem.from_mapping(payload)
    state = problem.initial_state()
    env = problem.atmosphere.environment(problem.epoch_jd_utc)
    env.update({"drag_area_m2": problem.drag_area_m2, "drag_coefficient": problem.drag_coefficient})
    context = OrbitContext(
        mu_km3_s2=EARTH_MU_KM3_S2,
        mass_kg=problem.mass_kg,
        area_m2=problem.drag_area_m2,
        cd=problem.drag_coefficient,
    )
    expected_rate = float(np.dot(state[3:], drag_plugin(0.0, state, env, context)))

    result = assess_orbit_lifetime(problem)
    measured_rate = result.summary["changes"]["specific_energy_km2_s2"] / 0.1

    assert measured_rate == pytest.approx(expected_rate, rel=2.0e-5)
    assert abs(result.summary["energy_accounting"]["residual_km2_s2"]) < 1.0e-10


def test_decay_reaches_all_thresholds_and_stops_at_refined_reentry() -> None:
    result = assess_orbit_lifetime(_problem())

    assert result.summary["schema_version"] == ORBIT_LIFETIME_EVIDENCE_SCHEMA
    assert result.summary["status"] == "completed"
    assert result.summary["outcome"] == "reentry_threshold_reached"
    assert [item.threshold_kind for item in result.events] == ["warning", "disposal", "reentry"]
    assert all(item.disposition == "provider_refined" for item in result.events)
    assert all(item.bracket_end_s - item.bracket_start_s <= 0.01 for item in result.events)
    assert result.samples[-1].altitude_km == pytest.approx(120.0, abs=5.0e-4)
    assert result.summary["resource_use"]["propagated_duration_s"] < 7200.0


def test_horizon_complete_does_not_extrapolate_unreached_lifetime() -> None:
    payload = _problem(duration_s=600.0)
    payload["atmosphere"] = {"model": "constant", "parameters": {"density_kg_m3": 1.0e-15}}
    result = assess_orbit_lifetime(payload)

    assert result.summary["outcome"] == "horizon_complete"
    assert result.summary["thresholds"]["reentry"] == {
        "altitude_km": 120.0,
        "reached": False,
        "time_s": None,
        "elapsed_days": None,
        "disposition": None,
    }
    assert "not evidence of infinite lifetime" in result.summary["claim_limits"][1]


def test_decay_solution_is_stable_under_step_refinement() -> None:
    coarse = _problem(duration_s=1800.0, integration_step_s=10.0)
    coarse["output_step_s"] = 300.0
    fine = deepcopy(coarse)
    fine["integration_step_s"] = 5.0
    coarse_result = assess_orbit_lifetime(coarse)
    fine_result = assess_orbit_lifetime(fine)

    assert coarse_result.summary["final"]["semi_major_axis_km"] == pytest.approx(
        fine_result.summary["final"]["semi_major_axis_km"], abs=2.0e-4
    )
    assert coarse_result.summary["thresholds"]["warning"]["time_s"] == pytest.approx(
        fine_result.summary["thresholds"]["warning"]["time_s"], abs=0.02
    )


def test_interior_perigee_crossing_is_detected_when_step_endpoints_are_above() -> None:
    perigee_radius_km = EARTH_RADIUS_KM + 79.99
    eccentricity = 0.8
    position, velocity = coe_to_rv_eci(
        a_km=perigee_radius_km / (1.0 - eccentricity),
        ecc=eccentricity,
        inc_deg=0.0,
        raan_deg=0.0,
        argp_deg=0.0,
        true_anomaly_deg=0.0,
    )
    perigee_state = np.hstack((position, velocity))
    propagator = OrbitPropagator(model="two_body", integrator="rk4", plugins=[], acceleration_mode="off")
    context = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
    initial_state = propagator.propagate(
        perigee_state,
        -60.0,
        0.0,
        np.zeros(3),
        {},
        context,
    )
    payload = _problem(duration_s=120.0, integration_step_s=120.0)
    payload.update(
        {
            "initial_position_eci_km": initial_state[:3].tolist(),
            "initial_velocity_eci_km_s": initial_state[3:].tolist(),
            "output_step_s": 120.0,
            "drag_enabled": False,
            "atmosphere": {"model": "constant", "parameters": {"density_kg_m3": 1.0e-15}},
            "thresholds": {
                "warning_altitude_km": 80.2,
                "disposal_altitude_km": 80.1,
                "reentry_altitude_km": 80.0,
            },
        }
    )

    result = assess_orbit_lifetime(payload)

    assert result.samples[0].altitude_km > 80.2
    assert result.summary["outcome"] == "reentry_threshold_reached"
    assert [event.threshold_kind for event in result.events] == ["warning", "disposal", "reentry"]
    assert result.events[-1].time_s < 60.0


def test_stop_at_reentry_false_terminates_at_earth_surface_without_invalid_samples() -> None:
    payload = _problem(duration_s=7200.0)
    payload["stop_at_reentry"] = False

    result = assess_orbit_lifetime(payload)

    assert result.summary["outcome"] == "earth_surface_reached"
    assert result.summary["thresholds"]["reentry"]["reached"] is True
    assert result.samples[-1].altitude_km >= 0.0
    assert result.samples[-1].altitude_km < 0.01
    assert result.events[-1].threshold_kind == "earth_surface"


def test_harris_priester_effective_table_and_altitude_domain_are_enforced() -> None:
    unsupported = _problem(duration_s=10.0, integration_step_s=10.0)
    unsupported["atmosphere"] = {"model": "harris_priester", "parameters": {"f107": 149.0}}
    with pytest.raises(OrbitLifetimeError, match="supported Harris-Priester table"):
        OrbitLifetimeProblem.from_mapping(unsupported)

    below_domain = _problem(duration_s=10.0, integration_step_s=10.0)
    below_domain["atmosphere"] = {"model": "harris_priester", "parameters": {"f107": 150.0}}
    below_domain["thresholds"]["reentry_altitude_km"] = 110.0
    with pytest.raises(OrbitLifetimeError, match="lower domain limit"):
        OrbitLifetimeProblem.from_mapping(below_domain)

    valid = _problem(duration_s=10.0, integration_step_s=10.0)
    valid["output_step_s"] = 10.0
    valid["atmosphere"] = {"model": "harris_priester", "parameters": {"f107": 150.0}}
    result = assess_orbit_lifetime(valid)
    assert result.summary["atmosphere_effective"]["harris_priester_f107_table"] == 150.0
    assert result.summary["atmosphere_effective"]["altitude_domain"] == {
        "minimum_altitude_km": 110.0,
        "minimum_inclusive": False,
        "maximum_altitude_km": 2000.0,
        "maximum_inclusive": False,
    }


def test_comparison_semantic_identity_excludes_unused_base_atmosphere() -> None:
    original = _comparison()
    changed = deepcopy(original)
    changed["base_problem"]["atmosphere"] = {
        "model": "constant",
        "parameters": {"density_kg_m3": 9.99e-2},
    }

    original_result = compare_orbit_lifetime_models(original)
    changed_result = compare_orbit_lifetime_models(changed)

    assert original_result.rows == changed_result.rows
    assert original_result.summary["cases"] == changed_result.summary["cases"]
    assert (
        original_result.summary["comparison_semantic_sha256"]
        == changed_result.summary["comparison_semantic_sha256"]
    )
    assert original_result.summary["result_semantic_sha256"] == changed_result.summary["result_semantic_sha256"]


def test_epoch_and_atmosphere_runtime_failures_are_structured(monkeypatch) -> None:
    invalid_epoch = _problem(duration_s=10.0, integration_step_s=10.0)
    invalid_epoch["epoch_jd_utc"] = 1.0e308
    with pytest.raises(OrbitLifetimeError, match="UTC conversion"):
        OrbitLifetimeProblem.from_mapping(invalid_epoch)

    payload = _problem(duration_s=10.0, integration_step_s=10.0)
    payload["output_step_s"] = 10.0
    payload["atmosphere"] = {"model": "ussa1976", "parameters": {}}

    def fail_density(*args, **kwargs):
        raise OverflowError("synthetic atmosphere overflow")

    monkeypatch.setattr("sim.analysis.orbit_lifetime.density_from_model", fail_density)
    with pytest.raises(OrbitLifetimeError, match="Atmosphere model 'ussa1976' failed"):
        assess_orbit_lifetime(payload)


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        ("constant", {"density_kg_m3": 1.0e-12}),
        (
            "exponential",
            {
                "reference_density_kg_m3": 1.0e-9,
                "reference_altitude_km": 200.0,
                "scale_height_km": 30.0,
                "ceiling_altitude_km": 1000.0,
            },
        ),
        ("ussa1976", {}),
        ("nrlmsise00", {"f107": 150.0, "f107a": 150.0, "ap": 4.0, "ap_a": [4.0] * 7}),
        ("harris_priester", {"f107": 150.0}),
    ],
)
def test_supported_atmospheres_are_explicit_and_finite(model: str, parameters: dict) -> None:
    atmosphere = LifetimeAtmosphere.from_mapping({"model": model, "parameters": parameters})
    problem = OrbitLifetimeProblem.from_mapping(_problem(duration_s=10.0, integration_step_s=10.0))
    density = atmosphere.density(problem.initial_state(), 0.0, problem.epoch_jd_utc)

    assert np.isfinite(density)
    assert density >= 0.0
    assert atmosphere.to_dict()["parameters"] == parameters


def test_model_comparison_preserves_identical_non_atmosphere_inputs() -> None:
    problem = OrbitLifetimeComparisonProblem.from_mapping(_comparison())
    result = compare_orbit_lifetime_models(problem)

    assert result.summary["schema_version"] == ORBIT_LIFETIME_COMPARISON_EVIDENCE_SCHEMA
    assert result.summary["identical_non_atmosphere_inputs"] is True
    assert [item["case_id"] for item in result.rows] == ["high-density", "low-density"]
    assert result.rows[0]["final_semi_major_axis_km"] < result.rows[1]["final_semi_major_axis_km"]
    assert len({item["result_semantic_sha256"] for item in result.rows}) == 2


def test_single_and_comparison_artifacts_replay_and_fail_closed(tmp_path: Path) -> None:
    single = write_orbit_lifetime_artifacts(
        assess_orbit_lifetime(_problem(duration_s=600.0)), tmp_path / "single"
    )
    comparison = write_orbit_lifetime_comparison_artifacts(
        compare_orbit_lifetime_models(_comparison()), tmp_path / "comparison"
    )

    assert verify_orbit_lifetime_artifacts(single.output_dir)["status"] == "verified"
    assert verify_orbit_lifetime_comparison_artifacts(comparison.output_dir)["status"] == "verified"

    original_manifest = single.manifest_json.read_text(encoding="utf-8")
    identity_forgery = json.loads(original_manifest)
    identity_forgery["implementation_identity"]["source_tree_sha256"] = "0" * 64
    single.manifest_json.write_text(
        json.dumps(identity_forgery, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OrbitLifetimeError, match="implementation_identity"):
        verify_orbit_lifetime_artifacts(single.output_dir)
    single.manifest_json.write_text(original_manifest, encoding="utf-8")

    forged = b"sample_index,forged\n0,true\n"
    single.timeseries_csv.write_bytes(forged)
    manifest = json.loads(single.manifest_json.read_text(encoding="utf-8"))
    receipt = next(item for item in manifest["artifacts"] if item["path"] == single.timeseries_csv.name)
    receipt["bytes"] = len(forged)
    receipt["sha256"] = hashlib.sha256(forged).hexdigest()
    single.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(OrbitLifetimeError, match="authoritative replay"):
        verify_orbit_lifetime_artifacts(single.output_dir)


def test_contract_rejects_unknown_nonfinite_and_unbounded_inputs() -> None:
    unknown = _problem()
    unknown["surprise"] = True
    with pytest.raises(OrbitLifetimeError, match="unknown fields"):
        OrbitLifetimeProblem.from_mapping(unknown)

    nonfinite = _problem()
    nonfinite["mass_kg"] = float("nan")
    with pytest.raises(OrbitLifetimeError, match="finite"):
        OrbitLifetimeProblem.from_mapping(nonfinite)

    too_many_steps = _problem(duration_s=7200.0)
    too_many_steps["integration_step_s"] = 0.001
    too_many_steps["output_step_s"] = 1.0
    with pytest.raises(OrbitLifetimeError, match="integration steps"):
        OrbitLifetimeProblem.from_mapping(too_many_steps)

    duplicate = _comparison()
    duplicate["cases"][1]["case_id"] = duplicate["cases"][0]["case_id"]
    with pytest.raises(OrbitLifetimeError, match="unique"):
        OrbitLifetimeComparisonProblem.from_mapping(duplicate)


def test_published_schema_accepts_generated_records(tmp_path: Path) -> None:
    schema = json.loads(
        (ROOT / "docs/contracts/schemas/oel-orbit-lifetime-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    problem = OrbitLifetimeProblem.from_mapping(_problem(duration_s=600.0))
    result = assess_orbit_lifetime(problem)
    artifacts = write_orbit_lifetime_artifacts(result, tmp_path / "single")
    comparison_problem = OrbitLifetimeComparisonProblem.from_mapping(_comparison())
    comparison_result = compare_orbit_lifetime_models(comparison_problem)
    comparison_artifacts = write_orbit_lifetime_comparison_artifacts(
        comparison_result, tmp_path / "comparison"
    )

    for value in (
        problem.to_dict(),
        result.summary,
        json.loads(artifacts.manifest_json.read_text(encoding="utf-8")),
        comparison_problem.to_dict(),
        comparison_result.summary,
        json.loads(comparison_artifacts.manifest_json.read_text(encoding="utf-8")),
    ):
        validator.validate(value)


def test_cli_analyze_compare_replay_validation_and_error_paths(tmp_path: Path, capsys) -> None:
    assert "lifetime" in _dispatch_commands()
    problem_path = tmp_path / "problem.json"
    comparison_path = tmp_path / "comparison.json"
    problem_path.write_text(json.dumps(_problem(duration_s=600.0)), encoding="utf-8")
    comparison_path.write_text(json.dumps(_comparison()), encoding="utf-8")

    assert lifetime_main(["validate", str(problem_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    assert lifetime_main(["validate-comparison", str(comparison_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"

    single = tmp_path / "single"
    comparison = tmp_path / "comparison"
    assert lifetime_main(["analyze", str(problem_path), "--output-dir", str(single)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert lifetime_main(["replay", str(single)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert lifetime_main(["compare", str(comparison_path), "--output-dir", str(comparison)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert lifetime_main(["replay-comparison", str(comparison)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"

    invalid = _problem()
    invalid["mass_kg"] = -1.0
    problem_path.write_text(json.dumps(invalid), encoding="utf-8")
    assert lifetime_main(["validate", str(problem_path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"
