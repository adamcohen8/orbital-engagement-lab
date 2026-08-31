from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.trajectory_design import (
    TrajectoryTargetingProblem,
    execute_trajectory,
    finite_difference_jacobian,
    solve_trajectory_target,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "trajectory_targeting_orekit_13_1_7.json"
JAVA_SOURCE = Path(__file__).with_name("external_references") / "OrekitTrajectoryTargetingReference.java"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _problem(
    *,
    segments: list[dict],
    constraints: list[dict],
    variables: list[dict] | None = None,
    step_s: float = 1.0,
) -> TrajectoryTargetingProblem:
    radius_km = 7000.0
    return TrajectoryTargetingProblem.from_mapping(
        {
            "schema_version": "oel.trajectory_targeting_problem.v1",
            "name": "orekit_acceptance",
            "initial_state_eci_km_km_s": [
                radius_km,
                0.0,
                0.0,
                0.0,
                math.sqrt(EARTH_MU_KM3_S2 / radius_km),
                0.0,
            ],
            "propagation": {"step_s": step_s},
            "segments": segments,
            "variables": variables or [],
            "constraints": constraints,
        }
    )


def test_orekit_fixture_is_bound_to_public_generator_source() -> None:
    fixture = _fixture()
    source_hash = hashlib.sha256(JAVA_SOURCE.read_bytes()).hexdigest()
    raw_csv = ROOT / fixture["raw_csv"]
    raw_hash = hashlib.sha256(raw_csv.read_bytes()).hexdigest()

    assert JAVA_SOURCE.relative_to(ROOT).as_posix() == fixture["generator_source"]
    assert source_hash == fixture["generator_source_sha256"]
    assert raw_hash == fixture["raw_csv_sha256"]
    assert fixture["provider"] == "Orekit"
    assert fixture["provider_version"] == "13.1.7"
    assert fixture["orekit_data_required"] is False

    with raw_csv.open(newline="", encoding="utf-8") as handle:
        rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == set(fixture["cases"])
    for case_id, expected in fixture["cases"].items():
        row = rows[case_id]
        assert float(row["time_s"]) == expected["time_s"]
        assert [
            float(row[column])
            for column in ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")
        ] == expected["state_eci_km_km_s"]


def test_hohmann_solution_repropagates_to_pinned_orekit_state() -> None:
    fixture = _fixture()
    reference = fixture["cases"]["hohmann_half_transfer"]
    r1_km = 7000.0
    r2_km = 9000.0
    circular_speed = math.sqrt(EARTH_MU_KM3_S2 / r1_km)
    transfer_speed = math.sqrt(EARTH_MU_KM3_S2 * (2.0 / r1_km - 2.0 / (r1_km + r2_km)))
    departure_m_s = 1000.0 * (transfer_speed - circular_speed)
    problem = _problem(
        segments=[
            {
                "type": "impulsive_burn",
                "name": "departure",
                "frame": "ric",
                "delta_v_m_s": [0.0, departure_m_s, 0.0],
            },
            {"type": "coast", "name": "transfer", "duration_s": reference["time_s"]},
        ],
        constraints=[{"name": "arrival_radius", "quantity": "radius_km", "target": r2_km, "tolerance": 1.0e-5}],
    )

    execution = execute_trajectory(problem)
    difference = np.asarray(execution["final_state_eci_km_km_s"]) - np.asarray(reference["state_eci_km_km_s"])

    # The reviewed envelope preserves the independent RK4-versus-analytical
    # implementation residual instead of requiring machine-zero parity.
    assert np.linalg.norm(difference[:3]) < 1.0e-5
    assert np.linalg.norm(difference[3:]) < 1.0e-8


def test_cartesian_rendezvous_target_and_jacobian_match_orekit() -> None:
    fixture = _fixture()
    cases = fixture["cases"]
    target = cases["rendezvous_seed"]["state_eci_km_km_s"]
    problem = _problem(
        segments=[
            {
                "type": "impulsive_burn",
                "name": "departure",
                "frame": "eci",
                "delta_v_m_s": [0.0, 0.0, 2.0],
            },
            {"type": "coast", "name": "rendezvous", "duration_s": 900.0},
        ],
        variables=[
            {
                "name": "dv_x",
                "segment": "departure",
                "field": "delta_v_x_m_s",
                "initial": 5.0,
                "perturbation": 0.1,
            },
            {
                "name": "dv_y",
                "segment": "departure",
                "field": "delta_v_y_m_s",
                "initial": -2.0,
                "perturbation": 0.1,
            },
        ],
        constraints=[
            {"name": "arrival_x", "quantity": "position_x_km", "target": target[0], "tolerance": 1.0e-5},
            {"name": "arrival_y", "quantity": "position_y_km", "target": target[1], "tolerance": 1.0e-5},
        ],
    )

    evidence = solve_trajectory_target(problem)
    oel_jacobian, _ = finite_difference_jacobian(problem, [12.0, -6.0])
    perturbation_m_s = 0.1
    orekit_jacobian_raw = np.column_stack(
        (
            (
                np.asarray(cases["rendezvous_plus_x"]["state_eci_km_km_s"][:2])
                - np.asarray(cases["rendezvous_minus_x"]["state_eci_km_km_s"][:2])
            )
            / (2.0 * perturbation_m_s),
            (
                np.asarray(cases["rendezvous_plus_y"]["state_eci_km_km_s"][:2])
                - np.asarray(cases["rendezvous_minus_y"]["state_eci_km_km_s"][:2])
            )
            / (2.0 * perturbation_m_s),
        )
    )
    orekit_jacobian_normalized = orekit_jacobian_raw / 1.0e-5

    assert evidence["converged"] is True
    assert evidence["decision_values"] == pytest.approx([12.0, -6.0], abs=2.0e-5)
    assert np.max(np.abs(oel_jacobian - orekit_jacobian_normalized)) < 0.02
    repropagated = evidence["authoritative_repropagation"]["execution"]["final_state_eci_km_km_s"]
    assert np.linalg.norm(np.asarray(repropagated)[:2] - np.asarray(target)[:2]) < 1.0e-5


def test_phasing_energy_target_matches_closed_form_burn() -> None:
    initial_radius_km = 7000.0
    target_semi_major_axis_km = 7200.0
    circular_speed = math.sqrt(EARTH_MU_KM3_S2 / initial_radius_km)
    target_speed = math.sqrt(EARTH_MU_KM3_S2 * (2.0 / initial_radius_km - 1.0 / target_semi_major_axis_km))
    expected_burn_m_s = 1000.0 * (target_speed - circular_speed)
    problem = _problem(
        segments=[
            {
                "type": "impulsive_burn",
                "name": "phasing_burn",
                "frame": "ric",
                "delta_v_m_s": [0.0, 0.0, 0.0],
            }
        ],
        variables=[
            {
                "name": "intrack_burn",
                "segment": "phasing_burn",
                "field": "delta_v_i_m_s",
                "initial": 50.0,
                "perturbation": 0.01,
            }
        ],
        constraints=[
            {
                "name": "phasing_sma",
                "quantity": "semi_major_axis_km",
                "target": target_semi_major_axis_km,
                "tolerance": 1.0e-6,
            }
        ],
    )

    evidence = solve_trajectory_target(problem)

    assert evidence["converged"] is True
    assert evidence["decision_values"] == pytest.approx([expected_burn_m_s], abs=1.0e-5)
