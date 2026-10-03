"""Opt-in whole-workflow parity for native targeting and exact scheduling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.analysis.mission_scheduling import solve_mission_schedule
from sim.analysis.trajectory_targeting import (
    TrajectoryTargetingError,
    finite_difference_jacobian,
    solve_trajectory_target,
)

native = pytest.importorskip("oel_rust_orbit")
pytestmark = pytest.mark.skipif(
    not all(hasattr(native, name) for name in ("mission_schedule_search", "targeting_rollouts_bytes")),
    reason="requires the native workflow batch kernels",
)


@pytest.mark.parametrize("delivery", [False, True])
@pytest.mark.parametrize("minimum", [0, 1, 2])
def test_exact_schedule_native_matches_python_evidence(delivery: bool, minimum: int) -> None:
    path = Path(__file__).resolve().parents[2] / "examples/mission_scheduling/public_two_asset_collection_problem.json"
    problem = json.loads(path.read_text(encoding="utf-8"))
    problem["require_observation_delivery_by_horizon"] = delivery
    problem["minimum_selected_observations"] = minimum
    python = solve_mission_schedule(problem)
    rust = solve_mission_schedule(problem, numeric_backend="rust")
    assert rust.selected_opportunity_ids == python.selected_opportunity_ids
    assert rust.feasible_subset_count == python.feasible_subset_count
    assert rust.evaluated_subset_count == python.evaluated_subset_count
    assert rust.schedule_semantic_sha256 == python.schedule_semantic_sha256


def test_native_schedule_matches_fractional_objective_and_tie_semantics() -> None:
    path = Path(__file__).resolve().parents[2] / "examples/mission_scheduling/public_two_asset_collection_problem.json"
    problem = json.loads(path.read_text(encoding="utf-8"))
    problem["opportunities"][0]["objective_value"] = 9.25
    expected = solve_mission_schedule(problem)
    actual = solve_mission_schedule(problem, numeric_backend="rust")
    assert actual.schedule_semantic_sha256 == expected.schedule_semantic_sha256


def _targeting_problem() -> dict:
    return {
        "schema_version": "oel.trajectory_targeting_problem.v1",
        "name": "native_jacobian_parity",
        "initial_state_eci_km_km_s": [7000.0, 0.0, 0.0, 0.0, 7.546053290107542, 0.0],
        "propagation": {"step_s": 10.0, "integrator": "rk4"},
        "segments": [
            {"type": "coast", "name": "before", "duration_s": 120.0},
            {"type": "impulsive_burn", "name": "burn", "frame": "ric", "delta_v_m_s": [0.0, 0.0, 0.0]},
            {"type": "coast", "name": "after", "duration_s": 180.0},
        ],
        "variables": [
            {"name": "in_track", "segment": "burn", "field": "delta_v_i_m_s",
             "initial": 0.0, "perturbation": 0.1},
        ],
        "constraints": [
            {"name": "vy", "quantity": "velocity_y_km_s", "target": 7.15, "tolerance": 0.0001},
        ],
    }


def test_native_targeting_jacobian_and_solver_retain_candidate_order_and_receipts() -> None:
    problem = _targeting_problem()
    python_jacobian, python_accounting = finite_difference_jacobian(problem, [0.0])
    rust_jacobian, rust_accounting = finite_difference_jacobian(problem, [0.0], numeric_backend="rust")
    np.testing.assert_allclose(rust_jacobian, python_jacobian, rtol=0.0, atol=1.0e-10)
    assert rust_accounting == python_accounting
    python = solve_trajectory_target(problem)
    rust = solve_trajectory_target(problem, numeric_backend="rust")
    assert rust["status"] == python["status"] == "converged"
    assert rust["decision_values"] == python["decision_values"]
    assert rust["convergence_history"] == python["convergence_history"]
    assert rust["resources"] == python["resources"]


def test_native_targeting_retains_elapsed_event_jacobian_and_accounting() -> None:
    problem = _targeting_problem()
    problem["segments"][0] = {
        "type": "coast", "name": "before",
        "stop": {"quantity": "elapsed_time_s", "target": 120.0,
                 "direction": "increasing", "max_duration_s": 200.0},
    }
    expected, expected_accounting = finite_difference_jacobian(problem, [0.0])
    actual, actual_accounting = finite_difference_jacobian(problem, [0.0], numeric_backend="rust")
    np.testing.assert_array_equal(actual, expected)
    assert actual_accounting == expected_accounting
    problem["propagation"]["integrator"] = "dopri5"
    with pytest.raises(TrajectoryTargetingError, match="RK4"):
        finite_difference_jacobian(problem, [0.0], numeric_backend="rust")


@pytest.mark.parametrize("forces", [[], ["j2"], ["j4", "j2", "j3"]])
def test_zonal_targeting_fixed_and_event_trials_preserve_reference(forces):
    from sim.analysis.trajectory_targeting import execute_trajectory

    problem = _targeting_problem()
    problem["propagation"]["force_model"] = forces
    reference, reference_counts = finite_difference_jacobian(problem, [0.0])
    actual, counts = finite_difference_jacobian(problem, [0.0], numeric_backend="rust")
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=2e-9)
    assert counts == reference_counts
    problem["initial_state_eci_km_km_s"][4] += 0.25
    problem["segments"][-1] = {
        "type": "coast", "name": "after", "stop": {
            "quantity": "radial_velocity_km_s", "target": 0.0, "direction": "decreasing",
            "minimum_elapsed_s": 60.0, "max_duration_s": 4500.0,
        },
    }
    reference = execute_trajectory(problem, [0.1])
    actual = execute_trajectory(problem, [0.1], numeric_backend="rust")
    np.testing.assert_allclose(actual["final_state_eci_km_km_s"], reference["final_state_eci_km_km_s"],
                               rtol=0.0, atol=1e-10)
    assert actual["resources"] == reference["resources"]
    assert actual["segments"][-1]["stop_event"] == reference["segments"][-1]["stop_event"]
    reference, reference_counts = finite_difference_jacobian(problem, [0.0])
    actual, counts = finite_difference_jacobian(problem, [0.0], numeric_backend="rust")
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=2e-9)
    assert counts == reference_counts


@pytest.mark.parametrize("quantity,target", [("elapsed_time_s", 1000.0), ("true_anomaly_deg", 90.0)])
def test_native_event_miss_preserves_failure_receipt(quantity, target):
    from sim.analysis.trajectory_targeting import MissedEventError, execute_trajectory

    problem = _targeting_problem()
    problem["segments"][0] = {"type": "coast", "name": "before", "stop": {
        "quantity": quantity, "target": target, "direction": "increasing", "max_duration_s": 1.0,
    }}
    receipts = []
    for backend in ("python", "rust"):
        with pytest.raises(MissedEventError) as caught:
            execute_trajectory(problem, numeric_backend=backend)
        receipts.append(caught.value.receipt)
    assert receipts[0] == receipts[1]


def test_native_angular_wrap_and_refinement_failure_receipts():
    from sim.analysis.trajectory_targeting import EventRefinementError, execute_trajectory

    problem = _targeting_problem()
    problem["segments"][0] = {"type": "coast", "name": "before", "stop": {
        "quantity": "true_anomaly_deg", "target": 15.0, "direction": "increasing",
        "minimum_elapsed_s": 5500.0, "max_duration_s": 6500.0,
    }}
    reference = execute_trajectory(problem)
    actual = execute_trajectory(problem, numeric_backend="rust")
    assert actual == reference
    problem["propagation"].update(event_max_iterations=1, event_time_tolerance_s=1e-12, event_value_tolerance=1e-12)
    receipts = []
    for backend in ("python", "rust"):
        with pytest.raises(EventRefinementError) as caught:
            execute_trajectory(problem, numeric_backend=backend)
        receipts.append(caught.value.receipt)
    assert receipts[0] == receipts[1]
