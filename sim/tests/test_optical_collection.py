from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import erfa
import numpy as np
import pytest

from sim.analysis.collection_opportunity import CollectionOpportunityProblem, assess_collection_opportunities
from sim.analysis.collection_opportunity_resources import CollectionResources, screen_collection_resources
from sim.analysis.optical_collection import (
    GroundTarget,
    OpticalPayload,
    footprint_boundary_evidence,
    local_nadir_frame_sensor_from_eci,
    optical_quality_metrics,
    sensor_frame_and_gimbal_vector,
)
from sim.collection import _read_problem
from sim.dynamics.orbit.epoch import AU_KM, sun_position_eci_km_enhanced

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "collection" / "public_equatorial_optical_collection.json"


def _problem() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _payload() -> OpticalPayload:
    return OpticalPayload.from_mapping(_problem()["sensor"])


def test_optical_quality_matches_closed_form_equations() -> None:
    payload = _payload()
    result = optical_quality_metrics(slant_range_km=500.0, incidence_angle_rad=0.0, payload=payload)
    assert result["diffraction_limited_resolution_m"] == pytest.approx(1.22 * 550.0e-9 * 500_000.0 / 0.2)
    assert result["ground_sample_distance_m"] == pytest.approx(5.0e-6 * 500_000.0 / 0.5)
    assert result["effective_resolution_m"] == pytest.approx(10.0)


def test_nadir_and_target_tracking_frames_are_proper_and_center_target() -> None:
    state = np.array([6878.137, 0.0, 0.0, 0.0, 7.612608, 0.0])
    target = np.array([6378.137, 0.0, 0.0])
    nadir = local_nadir_frame_sensor_from_eci(state)
    tracking, gimbal, angle = sensor_frame_and_gimbal_vector(state, target, pointing_mode="target_track_gimbal")
    for rotation in (nadir, tracking):
        assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1.0e-14)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-14)
    assert gimbal == pytest.approx([0.0, 0.0, 1.0], abs=1.0e-14)
    assert angle == pytest.approx(0.0)
    line_of_sight = (target - state[:3]) / np.linalg.norm(target - state[:3])
    assert tracking @ line_of_sight == pytest.approx([0.0, 0.0, 1.0], abs=1.0e-14)

    # Public mounting semantics are +X cross-track, +Y along-track, +Z boresight.
    assert nadir @ np.array([0.0, 0.0, 1.0]) == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-14)
    assert nadir @ np.array([0.0, 1.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1.0e-14)


def test_nadir_fixed_uses_fov_without_inventing_gimbal_motion() -> None:
    problem = _problem()
    problem["sensor"].update(
        pointing_mode="nadir_fixed",
        maximum_gimbal_off_nadir_deg=0.0,
        maximum_slew_rate_deg_s=0.0,
        settling_time_s=0.0,
        minimum_collection_duration_s=1.0,
    )
    problem["resources"] = {"enabled": False, "storage_capacity_bytes": 0.0, "initial_storage_bytes": 0.0}
    evidence = assess_collection_opportunities(problem)
    assert evidence["sample_ledger"][0]["gimbal_off_nadir_deg"] == 0.0
    assert evidence["sample_ledger"][0]["required_slew_rate_deg_s"] == 0.0
    interior = min(evidence["sample_ledger"], key=lambda row: abs(row["time_s"] - 5.0))
    assert interior["sensor_vertical_angle_deg"] != 0.0
    assert interior["available"] is True
    assert evidence["summary"]["accepted_opportunity_count"] == 1


def test_asymmetric_pushbroom_uses_cross_track_x_and_along_track_y() -> None:
    problem = _problem()
    problem["sensor"].update(
        pointing_mode="nadir_fixed",
        maximum_gimbal_off_nadir_deg=0.0,
        maximum_slew_rate_deg_s=0.0,
        settling_time_s=0.0,
        minimum_collection_duration_s=0.1,
    )
    problem["sensor"]["pattern"] = {
        "kind": "pushbroom_hard_fov",
        "x_half_angle_deg": 1.0,
        "y_half_angle_deg": 10.0,
    }
    problem["resources"] = {"enabled": False, "storage_capacity_bytes": 0.0, "initial_storage_bytes": 0.0}
    evidence = assess_collection_opportunities(problem)
    sample = min(evidence["sample_ledger"], key=lambda row: abs(row["time_s"] - 5.0))
    assert sample["sensor_horizontal_angle_deg"] == pytest.approx(0.0, abs=1.0e-10)
    assert 1.0 < abs(sample["sensor_vertical_angle_deg"]) < 10.0
    assert sample["available"] is True


def test_wgs84_footprint_boundary_is_complete_and_on_ellipsoid() -> None:
    payload = _payload()
    target = GroundTarget("target", 0.0, 0.0, 0.0)
    state = np.array([6878.137, 0.0, 0.0, 0.0, 7.612608, 0.0])
    sensor_from_ecef = local_nadir_frame_sensor_from_eci(state)
    result = footprint_boundary_evidence(
        observer_ecef_km=state[:3],
        dcm_sensor_from_ecef=sensor_from_ecef,
        payload=payload,
        target=target,
    )
    assert result["disposition"] == "complete"
    assert all(result["boundary_hit"])
    assert result["tangent_plane_area_km2"] > 0.0
    assert result["corner_chord_width_km"] == pytest.approx(176.62, rel=2.0e-3)
    assert result["corner_chord_height_km"] == pytest.approx(105.28, rel=2.0e-3)


def test_enhanced_sun_direction_agrees_with_erfa_epv00_at_j2000() -> None:
    position = sun_position_eci_km_enhanced(2451545.0)
    heliocentric_earth, _barycentric_earth = erfa.epv00(2451545.0, 0.0)
    reference = -np.asarray(heliocentric_earth[0], dtype=float) * AU_KM
    angular_error = math.degrees(
        math.acos(float(np.clip(position @ reference / np.linalg.norm(position) / np.linalg.norm(reference), -1.0, 1.0)))
    )
    assert angular_error < 0.02
    assert np.linalg.norm(position) == pytest.approx(np.linalg.norm(reference), rel=5.0e-4)


def test_resource_screen_fails_closed_for_storage_and_downlink() -> None:
    problem = _problem()
    resources = CollectionResources.from_mapping(problem["resources"])
    accepted = screen_collection_resources(resources, collection_end_s=100.0, generated_data_bytes=1.0e8)
    assert accepted["resource_feasible"] is True

    storage_raw = copy.deepcopy(problem["resources"])
    storage_raw["storage_capacity_bytes"] = 1.0e6
    storage = screen_collection_resources(
        CollectionResources.from_mapping(storage_raw), collection_end_s=100.0, generated_data_bytes=1.0e8
    )
    assert storage["reason"] == "storage_exceeded"

    downlink_raw = copy.deepcopy(problem["resources"])
    downlink_raw["downlink_windows"][0]["delivered_data_bytes"] = 1.0e6
    downlink = screen_collection_resources(
        CollectionResources.from_mapping(downlink_raw), collection_end_s=100.0, generated_data_bytes=1.0e8
    )
    assert downlink["reason"] == "downlink_insufficient"


def test_resource_screen_rejects_duplicate_and_overlapping_downlink_capacity() -> None:
    base = {
        "enabled": True,
        "storage_capacity_bytes": 1000.0,
        "initial_storage_bytes": 0.0,
        "require_downlink_by_horizon": True,
        "downlink_windows": [
            {
                "window_id": "w1",
                "source_product_sha256": "a" * 64,
                "start_s": 20.0,
                "end_s": 30.0,
                "delivered_data_bytes": 60.0,
            },
            {
                "window_id": "w2",
                "source_product_sha256": "a" * 64,
                "start_s": 20.0,
                "end_s": 30.0,
                "delivered_data_bytes": 60.0,
            },
        ],
    }
    with pytest.raises(ValueError, match="semantic duplicates"):
        CollectionResources.from_mapping(base)

    base["downlink_windows"][1]["source_product_sha256"] = "b" * 64
    base["downlink_windows"][1]["start_s"] = 25.0
    base["downlink_windows"][1]["end_s"] = 35.0
    with pytest.raises(ValueError, match="must not overlap"):
        CollectionResources.from_mapping(base)


def test_equatorial_collection_builds_refined_resource_screened_opportunity() -> None:
    evidence = assess_collection_opportunities(CollectionOpportunityProblem.from_mapping(_problem()))
    assert evidence["status"] == "completed"
    assert evidence["summary"]["accepted_opportunity_count"] == 1
    candidate = evidence["opportunity_candidates"][0]
    assert candidate["raw_geometry_start_s"] == 0.0
    assert candidate["raw_geometry_end_s"] == pytest.approx(79.15, abs=0.02)
    assert candidate["collection_start_s"] == pytest.approx(5.0)
    assert candidate["accepted"] is True
    assert candidate["resource_screen"]["resource_feasible"] is True
    assert candidate["midpoint_footprint"]["disposition"] == "complete"
    assert evidence["task_opportunities"][0]["kind"] == "observation"
    assert evidence["normalized_problem"]["resources"]["downlink_windows"][0]["source_product_sha256"] == "a" * 64
    assert candidate["resource_screen"]["eligible_downlink_sources"][0]["source_product_sha256"] == "a" * 64
    json.dumps(evidence, allow_nan=False)


def test_interior_discovery_finds_false_endpoint_opportunity() -> None:
    problem = _problem()
    problem["target"]["longitude_deg"] += 0.296
    problem["sensor"].update(
        maximum_slew_rate_deg_s=100.0,
        settling_time_s=0.0,
        minimum_collection_duration_s=0.1,
    )
    problem["constraints"]["maximum_effective_resolution_m"] = 10.02
    problem["resources"] = {"enabled": False, "storage_capacity_bytes": 0.0, "initial_storage_bytes": 0.0}
    problem["propagation"]["step_s"] = 10.0
    evidence = assess_collection_opportunities(problem)
    assert evidence["sample_ledger"][0]["available"] is False
    assert next(row for row in evidence["sample_ledger"] if row["time_s"] == 10.0)["available"] is False
    assert evidence["summary"]["accepted_opportunity_count"] == 1
    candidate = evidence["opportunity_candidates"][0]
    assert candidate["raw_geometry_start_s"] == pytest.approx(1.79, abs=0.02)
    assert candidate["raw_geometry_end_s"] == pytest.approx(8.20, abs=0.02)
    assert evidence["resources"]["maximum_interior_discovery_step_s"] < problem["propagation"]["step_s"]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (("sensor", "maximum_slew_rate_deg_s", 0.1), "slew_rate_exceeded"),
        (("constraints", "maximum_effective_resolution_m", 5.0), "resolution_exceeded"),
        (("constraints", "minimum_sun_elevation_deg", 20.0), "illumination_rejected"),
    ],
)
def test_collection_constraints_retain_rejection_reasons(
    mutation: tuple[str, str, float], expected_reason: str
) -> None:
    problem = _problem()
    section, key, value = mutation
    problem[section][key] = value
    evidence = assess_collection_opportunities(problem)
    assert evidence["summary"]["accepted_opportunity_count"] == 0
    assert expected_reason in evidence["summary"]["reason_counts"]


def test_settling_storage_and_downlink_failures_remain_in_candidate_evidence() -> None:
    settling = _problem()
    settling["sensor"]["settling_time_s"] = 100.0
    evidence = assess_collection_opportunities(settling)
    assert evidence["opportunity_candidates"][0]["disposition"] == "insufficient_collection_duration"

    storage = _problem()
    storage["resources"]["storage_capacity_bytes"] = 1.0e6
    evidence = assess_collection_opportunities(storage)
    assert evidence["opportunity_candidates"][0]["disposition"] == "storage_exceeded"

    downlink = _problem()
    downlink["resources"]["downlink_windows"][0]["delivered_data_bytes"] = 1.0e6
    evidence = assess_collection_opportunities(downlink)
    assert evidence["opportunity_candidates"][0]["disposition"] == "downlink_insufficient"


def test_problem_rejects_non_surface_target_and_unbound_downlink() -> None:
    target = _problem()
    target["target"]["altitude_km"] = 1.0
    with pytest.raises(ValueError, match="surface targets"):
        CollectionOpportunityProblem.from_mapping(target)

    downlink = _problem()
    downlink["resources"]["downlink_windows"][0]["source_product_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="source_product_sha256"):
        CollectionOpportunityProblem.from_mapping(downlink)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda value: value["constraints"].__setitem__("maximum_effective_resolution_metres", 5.0), "unknown fields"),
        (lambda value: value["resources"].__setitem__("enabled", "false"), "must be a boolean"),
        (lambda value: value["sensor"].__setitem__("boundary_samples_per_edge", 2.9), "must be an integer"),
        (lambda value: value.__setitem__("transition_max_iterations", 3.9), "must be an integer"),
        (lambda value: value["propagation"].__setitem__("event_max_iterations", 3.9), "must be an integer"),
    ],
)
def test_problem_rejects_unknown_fields_and_type_coercion(mutation, expected: str) -> None:
    problem = _problem()
    mutation(problem)
    with pytest.raises(ValueError, match=expected):
        CollectionOpportunityProblem.from_mapping(problem)


def test_problem_reader_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"oel.collection_opportunity_problem.v1","duration_s":1,"duration_s":2}')
    with pytest.raises(ValueError, match="Duplicate JSON field 'duration_s'"):
        _read_problem(path)
