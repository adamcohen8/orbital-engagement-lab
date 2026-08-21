from __future__ import annotations

import json

import numpy as np
import pytest

from sim.analysis.global_coverage import (
    GlobalCoverageConfig,
    evaluate_global_coverage,
    summarize_sampled_coverage_mask,
    write_global_coverage_artifacts,
)
from sim.analysis.healpix import (
    WGS84_AUTHALIC_RADIUS_KM,
    WGS84_SURFACE_AREA_KM2,
    authalic_latitude_rad,
    geodetic_latitude_from_authalic_rad,
    healpix_nested_centers_authalic,
    healpix_npix,
    healpix_wgs84_centers,
)
from sim.analysis.observer_target_geometry import evaluate_surface_targets_ecef
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM
from sim.utils.quaternion import dcm_to_quaternion_bn


def _attitude_for_boresight_eci(boresight_eci: np.ndarray) -> np.ndarray:
    body_z = np.asarray(boresight_eci, dtype=float)
    body_z /= np.linalg.norm(body_z)
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(reference, body_z))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    body_x = np.cross(reference, body_z)
    body_x /= np.linalg.norm(body_x)
    body_y = np.cross(body_z, body_x)
    return dcm_to_quaternion_bn(np.vstack((body_x, body_y, body_z)))


def _fixed_ecef_evidence(
    times_s: np.ndarray,
    *,
    boresight_ecef: np.ndarray,
) -> tuple[FrameContext, np.ndarray, np.ndarray]:
    context = FrameContext(jd_utc_start=2451545.0)
    position_ecef = np.array([WGS84_A_KM + 500.0, 0.0, 0.0], dtype=float)
    positions_eci = []
    attitudes = []
    for time_s in times_s:
        rotation = eci_to_ecef_rotation_context(float(time_s), context)
        positions_eci.append(rotation.T @ position_ecef)
        attitudes.append(_attitude_for_boresight_eci(rotation.T @ boresight_ecef))
    return context, np.asarray(positions_eci), np.asarray(attitudes)


def _config(*, chunk_size: int = 1024, max_comparisons: int = 300_000_000) -> GlobalCoverageConfig:
    return GlobalCoverageConfig(
        analysis_id="phase1_reference",
        source_asset_id="spacecraft",
        state_provider_id="spacecraft.truth",
        attitude_source_kind="achieved",
        attitude_provider_id="spacecraft.attitude_truth",
        sensor_id="spacecraft.imager",
        order=5,
        half_angle_rad=float(np.deg2rad(20.0)),
        quat_body_from_sensor=(1.0, 0.0, 0.0, 0.0),
        chunk_size=chunk_size,
        max_cell_time_comparisons=max_comparisons,
    )


def test_healpix_order_zero_nested_centers_match_official_geometry() -> None:
    beta, longitude = healpix_nested_centers_authalic(0)
    polar_latitude = float(np.arcsin(2.0 / 3.0))
    expected_beta = np.array([polar_latitude] * 4 + [0.0] * 4 + [-polar_latitude] * 4)
    expected_longitude_deg = np.array(
        [45.0, 135.0, -135.0, -45.0, 0.0, 90.0, -180.0, -90.0, 45.0, 135.0, -135.0, -45.0]
    )
    np.testing.assert_allclose(beta, expected_beta, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(np.rad2deg(longitude), expected_longitude_deg, rtol=0.0, atol=1.0e-13)


def test_authalic_round_trip_surface_area_and_wgs84_centers() -> None:
    latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 1001)
    recovered = geodetic_latitude_from_authalic_rad(authalic_latitude_rad(latitude))
    # Forward authalic latitude loses a small amount of precision close to the
    # poles; the bracketed inverse remains within two solver tolerances.
    np.testing.assert_allclose(recovered, latitude, rtol=0.0, atol=2.0e-13)
    assert WGS84_AUTHALIC_RADIUS_KM == pytest.approx(6371.007180918474, abs=2.0e-12)
    assert WGS84_SURFACE_AREA_KM2 == pytest.approx(510065621.72408843, abs=2.0e-7)
    assert WGS84_SURFACE_AREA_KM2 / healpix_npix(6) == pytest.approx(10377.3116398944, abs=1.0e-7)

    centers = healpix_wgs84_centers(0)
    ellipsoid_level = (
        (centers.ecef_km[:, 0] / WGS84_A_KM) ** 2
        + (centers.ecef_km[:, 1] / WGS84_A_KM) ** 2
        + (centers.ecef_km[:, 2] / WGS84_B_KM) ** 2
    )
    np.testing.assert_allclose(ellipsoid_level, 1.0, rtol=0.0, atol=3.0e-15)
    np.testing.assert_allclose(
        np.linalg.norm(centers.outward_normal_ecef, axis=1),
        1.0,
        rtol=0.0,
        atol=2.0e-16,
    )


def test_sampled_interval_dwell_revisit_and_censoring_semantics() -> None:
    times = np.arange(6, dtype=float)
    mask = np.array(
        [
            [False, True, False],
            [True, True, False],
            [True, False, False],
            [False, False, False],
            [True, True, False],
            [False, True, False],
        ],
        dtype=bool,
    )
    metrics = summarize_sampled_coverage_mask(mask, times, cell_indices=np.array([10, 20, 30]))
    np.testing.assert_array_equal(metrics.intervals.cell_index, [10, 20])
    np.testing.assert_array_equal(metrics.intervals.interval_offset, [0, 2, 4])
    np.testing.assert_array_equal(metrics.intervals.start_sample_index, [1, 4, 0, 4])
    np.testing.assert_array_equal(metrics.intervals.end_sample_index_exclusive, [3, 5, 2, 6])
    np.testing.assert_allclose(metrics.dwell_s, [3.0, 3.0, 0.0])
    np.testing.assert_array_equal(metrics.interval_count, [2, 2, 0])
    np.testing.assert_array_equal(metrics.observed_acquisition_count, [2, 1, 0])
    np.testing.assert_allclose(metrics.max_complete_revisit_gap_s[:2], [1.0, 2.0])
    assert np.isnan(metrics.max_complete_revisit_gap_s[2])
    np.testing.assert_allclose(metrics.prefix_boundary_gap_s[[0, 2]], [1.0, 5.0])
    assert np.isnan(metrics.prefix_boundary_gap_s[1])
    np.testing.assert_allclose(metrics.suffix_boundary_gap_s[[0, 2]], [0.0, 5.0])
    assert np.isnan(metrics.suffix_boundary_gap_s[1])
    np.testing.assert_array_equal(metrics.start_censored, [False, True, False])
    np.testing.assert_array_equal(metrics.end_censored, [False, True, False])


def test_surface_target_geometry_blocks_tangent_and_enforces_range() -> None:
    target = np.array([[WGS84_A_KM, 0.0, 0.0]])
    normal = np.array([[1.0, 0.0, 0.0]])
    visible = evaluate_surface_targets_ecef(
        observer_ecef_km=np.array([WGS84_A_KM + 500.0, 0.0, 0.0]),
        target_ecef_km=target,
        target_outward_normal_ecef=normal,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
        half_angle_rad=0.2,
    )
    np.testing.assert_array_equal(visible.available, [True])

    tangent = evaluate_surface_targets_ecef(
        observer_ecef_km=np.array([WGS84_A_KM, 500.0, 0.0]),
        target_ecef_km=target,
        target_outward_normal_ecef=normal,
        boresight_ecef=np.array([0.0, -1.0, 0.0]),
        half_angle_rad=0.2,
    )
    np.testing.assert_array_equal(tangent.visible, [False])
    np.testing.assert_array_equal(tangent.available, [False])

    range_limited = evaluate_surface_targets_ecef(
        observer_ecef_km=np.array([WGS84_A_KM + 500.0, 0.0, 0.0]),
        target_ecef_km=target,
        target_outward_normal_ecef=normal,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
        half_angle_rad=0.2,
        max_range_km=100.0,
    )
    np.testing.assert_array_equal(range_limited.inside_range, [False])
    np.testing.assert_array_equal(range_limited.available, [False])


def test_achieved_attitude_changes_global_conical_coverage() -> None:
    times = np.array([0.0, 60.0, 120.0])
    context, positions, nadir_attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    nadir = evaluate_global_coverage(
        _config(),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=nadir_attitudes,
        frame_context=context,
    )
    np.testing.assert_array_equal(nadir.covered_cell_count, [4, 4, 4])
    assert nadir.summary["domain_disposition"] == "global_earth"
    assert nadir.summary["ever_covered_cell_count"] == 4
    covered = nadir.cell_metrics.interval_count > 0
    np.testing.assert_allclose(nadir.cell_metrics.dwell_s[covered], 120.0)
    assert np.all(nadir.cell_metrics.start_censored[covered])
    assert np.all(nadir.cell_metrics.end_censored[covered])

    _, _, outward_attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([1.0, 0.0, 0.0]),
    )
    outward = evaluate_global_coverage(
        _config(),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=outward_attitudes,
        frame_context=context,
    )
    np.testing.assert_array_equal(outward.covered_cell_count, 0)
    assert outward.summary["ever_covered_cell_count"] == 0


def test_chunk_size_does_not_change_results_or_semantic_hash() -> None:
    times = np.array([0.0, 30.0, 90.0])
    context, positions, attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    first = evaluate_global_coverage(
        _config(chunk_size=257),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    second = evaluate_global_coverage(
        _config(chunk_size=4096),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    np.testing.assert_array_equal(first.covered_cell_count, second.covered_cell_count)
    np.testing.assert_array_equal(first.cell_metrics.interval_count, second.cell_metrics.interval_count)
    np.testing.assert_array_equal(
        first.cell_metrics.intervals.interval_offset,
        second.cell_metrics.intervals.interval_offset,
    )
    np.testing.assert_array_equal(
        first.cell_metrics.intervals.start_sample_index,
        second.cell_metrics.intervals.start_sample_index,
    )
    np.testing.assert_allclose(first.cell_metrics.dwell_s, second.cell_metrics.dwell_s, rtol=0.0, atol=0.0)
    assert first.interval_semantic_sha256 == second.interval_semantic_sha256
    sign_equivalent = evaluate_global_coverage(
        _config(chunk_size=257),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=-attitudes,
        frame_context=context,
    )
    np.testing.assert_array_equal(sign_equivalent.covered_cell_count, first.covered_cell_count)
    assert sign_equivalent.input_evidence_sha256 != first.input_evidence_sha256
    assert sign_equivalent.interval_semantic_sha256 != first.interval_semantic_sha256


def test_artifacts_preserve_sparse_intervals_and_semantic_digest(tmp_path) -> None:
    times = np.array([0.0, 60.0])
    context, positions, attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    result = evaluate_global_coverage(
        _config(),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    artifacts = write_global_coverage_artifacts(result, tmp_path / "coverage")
    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    with np.load(artifacts.intervals_npz) as interval_data:
        np.testing.assert_array_equal(
            interval_data["cell_index"],
            result.cell_metrics.intervals.cell_index,
        )
        np.testing.assert_array_equal(
            interval_data["interval_offset"],
            result.cell_metrics.intervals.interval_offset,
        )
    assert manifest["artifacts"]["coverage_intervals.npz"]["semantic_sha256"] == (
        result.interval_semantic_sha256
    )
    assert manifest["input_evidence_sha256"] == result.input_evidence_sha256
    assert summary["status"] == "complete"
    assert artifacts.samples_csv.is_file()
    assert artifacts.cells_csv is not None and artifacts.cells_csv.is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        write_global_coverage_artifacts(result, artifacts.output_dir)


def test_validation_fails_closed_for_attitude_epoch_and_resource_limits() -> None:
    with pytest.raises(ValueError, match="normalized"):
        GlobalCoverageConfig(
            analysis_id="bad",
            source_asset_id="spacecraft",
            state_provider_id="truth",
            attitude_source_kind="achieved",
            attitude_provider_id="attitude",
            sensor_id="sensor",
            order=5,
            half_angle_rad=0.2,
            quat_body_from_sensor=(2.0, 0.0, 0.0, 0.0),
        )

    times = np.array([0.0, 60.0])
    context, positions, attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="cell-time comparisons"):
        evaluate_global_coverage(
            _config(max_comparisons=1),
            times_s=times,
            positions_eci_km=positions,
            attitudes_quat_bn=attitudes,
            frame_context=context,
        )
    with pytest.raises(ValueError, match="absolute UTC epoch"):
        evaluate_global_coverage(
            _config(),
            times_s=times,
            positions_eci_km=positions,
            attitudes_quat_bn=attitudes,
            frame_context=FrameContext(),
        )
