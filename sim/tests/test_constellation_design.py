from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sim.analysis.constellation_design import (
    CONSTELLATION_DESIGN_EVIDENCE_SCHEMA,
    ConstellationCandidate,
    ConstellationDesignError,
    ConstellationDesignProblem,
    generate_constellation_members,
    solve_constellation_design,
    verify_constellation_design_artifacts,
    write_constellation_design_artifacts,
)
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM


def _problem() -> dict:
    return {
        "schema_version": "oel.constellation_design_problem.v1",
        "analysis_id": "test_constellation_trade",
        "initial_jd_utc": 2461041.5,
        "duration_s": 600.0,
        "sample_step_s": 600.0,
        "propagation": {"model": "onp_two_body", "integration_step_s": 60.0},
        "coverage": {"order": 5, "half_angle_deg": 35.0, "required_multiplicity": 1},
        "ground_sites": [
            {
                "site_id": "equator",
                "geodetic_latitude_deg": 0.0,
                "longitude_deg": 0.0,
                "ellipsoidal_height_km": 0.0,
            }
        ],
        "link_budget": {
            "carrier_frequency_hz": 2.2e9,
            "tx_power_w": 20.0,
            "data_rate_bps": 1.0e6,
            "system_noise_temperature_k": 290.0,
            "required_eb_n0_db": 8.0,
            "tx_gain_dbi": 6.0,
            "rx_gain_dbi": 24.0,
            "tx_line_loss_db": 1.0,
            "rx_line_loss_db": 1.0,
            "misc_loss_db": 2.0,
            "minimum_elevation_deg": 5.0,
        },
        "objective": {
            "coverage_weight": 1.0,
            "network_weight": 0.5,
            "satellite_penalty": 0.005,
            "ground_site_penalty": 0.01,
            "minimum_coverage_fraction": 0.0,
            "minimum_network_availability_fraction": 0.0,
        },
        "designs": [
            {
                "design_id": "delta_2_1_0",
                "pattern": "walker_delta",
                "satellite_count": 2,
                "plane_count": 1,
                "phasing": 0,
                "altitude_km": 550.0,
                "inclination_deg": 53.0,
                "raan_start_deg": 10.0,
                "initial_phase_deg": 20.0,
                "raan_span_deg": None,
                "ground_site_ids": ["equator"],
            }
        ],
    }


def test_walker_generation_matches_declared_delta_phasing_and_radius() -> None:
    design = ConstellationCandidate.from_mapping(
        {
            **_problem()["designs"][0],
            "satellite_count": 6,
            "plane_count": 3,
            "phasing": 1,
        }
    )
    members = generate_constellation_members(design)

    assert len(members) == 6
    assert [member["raan_deg"] for member in members[::2]] == pytest.approx([10.0, 130.0, 250.0])
    assert [member["argument_of_latitude_deg"] for member in members[::2]] == pytest.approx([20.0, 80.0, 140.0])
    assert [member["argument_of_latitude_deg"] for member in members[1::2]] == pytest.approx([200.0, 260.0, 320.0])
    for member in members:
        assert np.linalg.norm(member["position_eci_km"]) == pytest.approx(EARTH_RADIUS_KM + 550.0)


def test_star_and_shell_raan_spans_are_explicit_and_distinct() -> None:
    base = _problem()["designs"][0]
    star = ConstellationCandidate.from_mapping(
        {**base, "pattern": "walker_star", "satellite_count": 4, "plane_count": 2, "phasing": 1}
    )
    shell = ConstellationCandidate.from_mapping(
        {
            **base,
            "pattern": "shell",
            "satellite_count": 4,
            "plane_count": 2,
            "phasing": 1,
            "raan_span_deg": 90.0,
        }
    )

    assert [member["raan_deg"] for member in generate_constellation_members(star)[::2]] == pytest.approx([10.0, 100.0])
    assert [member["raan_deg"] for member in generate_constellation_members(shell)[::2]] == pytest.approx([10.0, 55.0])


def test_problem_rejects_unknown_sites_and_unbounded_work() -> None:
    unknown_site = _problem()
    unknown_site["designs"][0]["ground_site_ids"] = ["missing"]
    with pytest.raises(ConstellationDesignError, match="unknown ground sites"):
        ConstellationDesignProblem.from_mapping(unknown_site)

    unbounded = _problem()
    unbounded["duration_s"] = 720000.0
    with pytest.raises(ConstellationDesignError, match="public bound"):
        ConstellationDesignProblem.from_mapping(unbounded)

    invalid_longitude = _problem()
    invalid_longitude["ground_sites"][0]["longitude_deg"] = 181.0
    with pytest.raises(ConstellationDesignError, match="longitude"):
        ConstellationDesignProblem.from_mapping(invalid_longitude)


def test_solve_is_deterministic_and_retains_transparent_ranking_evidence() -> None:
    problem = ConstellationDesignProblem.from_mapping(_problem())
    first = solve_constellation_design(problem)
    second = solve_constellation_design(problem)

    assert first.evidence == second.evidence
    assert first.evidence["schema_version"] == CONSTELLATION_DESIGN_EVIDENCE_SCHEMA
    assert first.evidence["status"] == "complete"
    assert first.evidence["ranking"] == ["delta_2_1_0"]
    candidate = first.evidence["candidate_results"][0]
    assert candidate["rank"] == 1
    assert candidate["feasible"] is True
    assert set(candidate["score_components"]) == {
        "coverage_service",
        "network_service",
        "satellite_penalty",
        "ground_site_penalty",
    }
    assert 0.0 <= candidate["coverage"]["time_weighted_mean_covered_fraction"] <= 1.0
    assert 0.0 <= candidate["network"]["union_sampled_available_fraction"] <= 1.0
    times = np.asarray(candidate["network"]["sample_times_s"], dtype=float)
    available = np.asarray(candidate["network"]["union_available_by_sample"], dtype=bool)
    independently_aggregated = float(np.dot(available[:-1], np.diff(times)) / (times[-1] - times[0]))
    assert candidate["network"]["union_sampled_available_fraction"] == independently_aggregated
    coverage_times = np.asarray(candidate["coverage"]["sample_times_s"], dtype=float)
    fractions = np.asarray(candidate["coverage"]["instantaneous_covered_fraction"], dtype=float)
    independently_time_weighted = float(
        np.dot(fractions[:-1], np.diff(coverage_times)) / (coverage_times[-1] - coverage_times[0])
    )
    assert candidate["coverage"]["time_weighted_mean_covered_fraction"] == pytest.approx(independently_time_weighted)
    assert len(candidate["generated_members"]) == 2


def test_all_infeasible_trade_retains_ranking_without_a_recommendation() -> None:
    raw = _problem()
    raw["objective"]["minimum_coverage_fraction"] = 1.0
    raw["objective"]["minimum_network_availability_fraction"] = 1.0

    evidence = solve_constellation_design(ConstellationDesignProblem.from_mapping(raw)).evidence

    assert evidence["ranking"] == ["delta_2_1_0"]
    assert evidence["candidate_results"][0]["feasible"] is False
    assert evidence["recommended_design_id"] is None


def test_artifacts_replay_authoritatively_and_reject_tampering(tmp_path: Path) -> None:
    result = solve_constellation_design(ConstellationDesignProblem.from_mapping(_problem()))
    artifacts = write_constellation_design_artifacts(result, tmp_path / "evidence")

    replay = verify_constellation_design_artifacts(artifacts.output_dir)
    assert replay["status"] == "verified"
    assert replay["result_semantic_sha256"] == result.evidence["result_semantic_sha256"]
    with pytest.raises(ConstellationDesignError, match="must be absent"):
        write_constellation_design_artifacts(result, artifacts.output_dir)

    payload = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    payload["candidate_results"][0]["score"] += 1.0
    artifacts.evidence_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConstellationDesignError, match="receipt mismatch"):
        verify_constellation_design_artifacts(artifacts.output_dir)


def test_normalization_is_order_independent_and_rejects_unknown_fields() -> None:
    raw = _problem()
    raw["ground_sites"].append(
        {
            "site_id": "north",
            "geodetic_latitude_deg": 45.0,
            "longitude_deg": 10.0,
            "ellipsoidal_height_km": 0.1,
        }
    )
    reversed_raw = copy.deepcopy(raw)
    reversed_raw["ground_sites"].reverse()
    assert (
        ConstellationDesignProblem.from_mapping(raw).to_dict()
        == ConstellationDesignProblem.from_mapping(reversed_raw).to_dict()
    )

    raw["unexpected"] = True
    with pytest.raises(ConstellationDesignError, match="unknown fields"):
        ConstellationDesignProblem.from_mapping(raw)
