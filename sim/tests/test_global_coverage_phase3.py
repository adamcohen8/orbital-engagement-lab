from __future__ import annotations

import json

import numpy as np
import pytest

from sim.analysis.coverage_plotting import _boundary_plot_vectors
from sim.analysis.coverage_queries import CoverageRegionMask, evaluate_coverage_queries
from sim.analysis.global_coverage import GlobalCoverageConfig, evaluate_global_coverage
from sim.analysis.healpix import healpix_npix
from sim.analysis.rich_coverage import (
    BOUNDARY_DISPOSITION_NAMES,
    RichCoverageConfig,
    evaluate_rich_coverage,
    write_rich_coverage_artifacts,
)
from sim.analysis.sensor_footprint_geometry import (
    PRIMARY_REASON_NAMES,
    HardFOVPattern,
    SurfaceServiceConstraints,
    evaluate_rich_surface_targets_ecef,
    fov_boundary_rays_sensor,
    intersect_rays_wgs84,
)
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM, geodetic_to_ecef_km
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


def _fixed_ecef_positions_in_eci(
    times_s: np.ndarray,
    context: FrameContext,
    position_ecef: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            eci_to_ecef_rotation_context(float(time_s), context).T @ position_ecef
            for time_s in times_s
        ]
    )


def _fixed_geodetic_evidence(
    times_s: np.ndarray,
    *,
    latitude_deg: float,
    longitude_deg: float,
) -> tuple[FrameContext, np.ndarray, np.ndarray]:
    context = FrameContext(jd_utc_start=2451545.0)
    surface_ecef = geodetic_to_ecef_km(latitude_deg, longitude_deg, 0.0)
    position_ecef = geodetic_to_ecef_km(latitude_deg, longitude_deg, 500.0)
    boresight_ecef = surface_ecef - position_ecef
    boresight_ecef /= np.linalg.norm(boresight_ecef)
    positions_eci = []
    attitudes = []
    for time_s in times_s:
        rotation = eci_to_ecef_rotation_context(float(time_s), context)
        positions_eci.append(rotation.T @ position_ecef)
        attitudes.append(_attitude_for_boresight_eci(rotation.T @ boresight_ecef))
    return context, np.asarray(positions_eci), np.asarray(attitudes)


def _rich_config(
    *,
    pattern: HardFOVPattern | None = None,
    constraints: SurfaceServiceConstraints | None = None,
    sun_provider_id: str | None = None,
    chunk_size: int = 1024,
    max_comparisons: int = 300_000_000,
) -> RichCoverageConfig:
    return RichCoverageConfig(
        analysis_id="phase3_reference",
        source_asset_id="spacecraft",
        state_provider_id="spacecraft.truth",
        attitude_source_kind="achieved",
        attitude_provider_id="spacecraft.attitude_truth",
        sensor_id="spacecraft.imager",
        order=5,
        quat_body_from_sensor=(1.0, 0.0, 0.0, 0.0),
        pattern=pattern or HardFOVPattern.axisymmetric_cone(np.deg2rad(20.0)),
        constraints=constraints or SurfaceServiceConstraints(),
        sun_provider_id=sun_provider_id,
        boundary_samples_per_edge=8,
        chunk_size=chunk_size,
        max_cell_time_comparisons=max_comparisons,
    )


def _phase1_config(*, chunk_size: int = 1024) -> GlobalCoverageConfig:
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
    )


def _surface_targets() -> tuple[np.ndarray, np.ndarray]:
    coordinates = ((0.0, 0.0), (0.0, 0.2), (0.2, 0.0), (0.0, 2.0))
    targets = np.asarray([geodetic_to_ecef_km(lat, lon, 0.0) for lat, lon in coordinates])
    normals = np.asarray(
        [
            np.array(
                [
                    point[0] / (WGS84_A_KM**2),
                    point[1] / (WGS84_A_KM**2),
                    point[2] / (WGS84_B_KM**2),
                ]
            )
            for point in targets
        ]
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return targets, normals


def _nadir_sensor_from_ecef() -> np.ndarray:
    # +Xs is east, +Ys is south, and +Zs is geodetic nadir at this fixture.
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])


def test_rich_cone_matches_frozen_phase1_membership_and_intervals() -> None:
    times = np.array([0.0, 60.0, 120.0])
    context, positions, attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    phase1 = evaluate_global_coverage(
        _phase1_config(),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    phase3 = evaluate_rich_coverage(
        _rich_config(),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    np.testing.assert_array_equal(phase3.covered_cell_count, phase1.covered_cell_count)
    np.testing.assert_array_equal(phase3.cell_metrics.interval_count, phase1.cell_metrics.interval_count)
    np.testing.assert_array_equal(
        phase3.cell_metrics.intervals.start_sample_index,
        phase1.cell_metrics.intervals.start_sample_index,
    )
    assert phase3.summary["domain_disposition"] == "global_earth"
    assert phase3.summary["boundary_disposition_sample_count"]["complete"] == times.size
    np.testing.assert_array_equal(
        phase3.primary_reason_count[:, PRIMARY_REASON_NAMES.index("available")],
        phase3.covered_cell_count,
    )


def test_rectangular_and_pushbroom_axes_have_distinct_membership() -> None:
    targets, normals = _surface_targets()
    observer = np.array([WGS84_A_KM + 500.0, 0.0, 0.0])
    rectangular = evaluate_rich_surface_targets_ecef(
        observer_ecef_km=observer,
        target_ecef_km=targets,
        target_outward_normal_ecef=normals,
        dcm_sensor_from_ecef=_nadir_sensor_from_ecef(),
        pattern=HardFOVPattern.rectangular(np.deg2rad(5.0), np.deg2rad(1.0)),
    )
    pushbroom = evaluate_rich_surface_targets_ecef(
        observer_ecef_km=observer,
        target_ecef_km=targets,
        target_outward_normal_ecef=normals,
        dcm_sensor_from_ecef=_nadir_sensor_from_ecef(),
        pattern=HardFOVPattern.pushbroom(np.deg2rad(1.0), np.deg2rad(5.0)),
    )
    np.testing.assert_array_equal(rectangular.inside_pattern[:3], [True, True, False])
    np.testing.assert_array_equal(pushbroom.inside_pattern[:3], [True, False, True])
    assert rectangular.sensor_horizontal_angle_rad[1] > 0.0
    assert rectangular.sensor_vertical_angle_rad[2] < 0.0


def test_service_constraints_and_primary_reason_precedence() -> None:
    targets, normals = _surface_targets()
    observer = np.array([WGS84_A_KM + 500.0, 0.0, 0.0])
    wide = HardFOVPattern.axisymmetric_cone(np.deg2rad(60.0))
    off_nadir = evaluate_rich_surface_targets_ecef(
        observer_ecef_km=observer,
        target_ecef_km=targets[:2],
        target_outward_normal_ecef=normals[:2],
        dcm_sensor_from_ecef=_nadir_sensor_from_ecef(),
        pattern=wide,
        constraints=SurfaceServiceConstraints(maximum_target_off_nadir_rad=np.deg2rad(1.0)),
    )
    np.testing.assert_array_equal(off_nadir.available, [True, False])
    assert off_nadir.primary_reason_code[1] == PRIMARY_REASON_NAMES.index("off_nadir_exceeded")

    incidence = evaluate_rich_surface_targets_ecef(
        observer_ecef_km=observer,
        target_ecef_km=targets[[0, 3]],
        target_outward_normal_ecef=normals[[0, 3]],
        dcm_sensor_from_ecef=_nadir_sensor_from_ecef(),
        pattern=wide,
        constraints=SurfaceServiceConstraints(maximum_incidence_rad=np.deg2rad(1.0)),
    )
    np.testing.assert_array_equal(incidence.available, [True, False])
    assert incidence.primary_reason_code[1] == PRIMARY_REASON_NAMES.index("incidence_exceeded")

    daylight = evaluate_rich_surface_targets_ecef(
        observer_ecef_km=observer,
        target_ecef_km=targets[:1],
        target_outward_normal_ecef=normals[:1],
        dcm_sensor_from_ecef=_nadir_sensor_from_ecef(),
        pattern=wide,
        constraints=SurfaceServiceConstraints(minimum_sun_elevation_rad=0.0),
        sun_ecef_km=np.array([1.5e8, 0.0, 0.0]),
    )
    night = evaluate_rich_surface_targets_ecef(
        observer_ecef_km=observer,
        target_ecef_km=targets[:1],
        target_outward_normal_ecef=normals[:1],
        dcm_sensor_from_ecef=_nadir_sensor_from_ecef(),
        pattern=wide,
        constraints=SurfaceServiceConstraints(minimum_sun_elevation_rad=0.0),
        sun_ecef_km=np.array([-1.5e8, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(daylight.available, [True])
    np.testing.assert_array_equal(night.available, [False])
    assert night.primary_reason_code[0] == PRIMARY_REASON_NAMES.index("illumination_rejected")

    outward = np.diag([1.0, -1.0, -1.0])
    pattern_precedes_night = evaluate_rich_surface_targets_ecef(
        observer_ecef_km=observer,
        target_ecef_km=targets[:1],
        target_outward_normal_ecef=normals[:1],
        dcm_sensor_from_ecef=outward,
        pattern=HardFOVPattern.axisymmetric_cone(np.deg2rad(20.0)),
        constraints=SurfaceServiceConstraints(minimum_sun_elevation_rad=0.0),
        sun_ecef_km=np.array([-1.5e8, 0.0, 0.0]),
    )
    assert pattern_precedes_night.primary_reason_code[0] == PRIMARY_REASON_NAMES.index(
        "outside_pattern"
    )


def test_boundary_rays_and_wgs84_intersections_cover_complete_partial_and_none() -> None:
    rectangle = fov_boundary_rays_sensor(
        HardFOVPattern.rectangular(np.deg2rad(10.0), np.deg2rad(5.0)),
        samples_per_edge=4,
    )
    assert rectangle.shape == (16, 3)
    np.testing.assert_allclose(np.linalg.norm(rectangle, axis=1), 1.0)

    observer = np.array([WGS84_A_KM + 500.0, 0.0, 0.0])
    mixed = intersect_rays_wgs84(
        observer,
        np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_equal(mixed.hit, [True, False])
    np.testing.assert_allclose(mixed.point_ecef_km[0], [WGS84_A_KM, 0.0, 0.0], atol=1.0e-9)
    assert np.all(np.isnan(mixed.point_ecef_km[1]))
    for invalid in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and non-negative"):
            intersect_rays_wgs84(
                observer,
                np.array([[1.0, 0.0, 0.0]]),
                distance_tolerance_km=invalid,
            )
        with pytest.raises(ValueError, match="finite and non-negative"):
            intersect_rays_wgs84(
                observer,
                np.array([[1.0, 0.0, 0.0]]),
                discriminant_tolerance=invalid,
            )

    polar = intersect_rays_wgs84(
        np.array([0.0, 0.0, WGS84_B_KM + 500.0]),
        np.array([[0.0, 0.0, -1.0]]),
    )
    np.testing.assert_array_equal(polar.hit, [True])
    np.testing.assert_allclose(polar.point_ecef_km[0], [0.0, 0.0, WGS84_B_KM], atol=1.0e-9)

    split_lon, split_lat = _boundary_plot_vectors(
        np.array([178.0, 179.0, -179.0, -178.0]),
        np.array([0.0, 1.0, 1.0, 0.0]),
        np.ones(4, dtype=bool),
    )
    assert np.count_nonzero(np.isnan(split_lon)) >= 1
    assert np.count_nonzero(np.isnan(split_lat)) >= 1

    times = np.array([0.0, 60.0])
    dispositions = {}
    for name, angle_deg, boresight in (
        ("complete", 10.0, np.array([-1.0, 0.0, 0.0])),
        ("partial", 10.0, np.array([-np.cos(np.deg2rad(65.0)), np.sin(np.deg2rad(65.0)), 0.0])),
        ("none", 10.0, np.array([1.0, 0.0, 0.0])),
    ):
        context, positions, attitudes = _fixed_ecef_evidence(times, boresight_ecef=boresight)
        result = evaluate_rich_coverage(
            _rich_config(pattern=HardFOVPattern.axisymmetric_cone(np.deg2rad(angle_deg))),
            times_s=times,
            positions_eci_km=positions,
            attitudes_quat_bn=attitudes,
            frame_context=context,
        )
        dispositions[name] = BOUNDARY_DISPOSITION_NAMES[
            int(result.footprint_boundary.boundary_disposition_code[0])
        ]
    assert dispositions == {"complete": "complete", "partial": "partial", "none": "no_intersection"}


def test_physical_rich_boundaries_cross_antimeridian_and_reach_high_latitude() -> None:
    times = np.array([0.0, 60.0])
    config = _rich_config(pattern=HardFOVPattern.axisymmetric_cone(np.deg2rad(20.0)))

    context, positions, attitudes = _fixed_geodetic_evidence(
        times,
        latitude_deg=0.0,
        longitude_deg=179.5,
    )
    dateline = evaluate_rich_coverage(
        config,
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    hit = dateline.footprint_boundary.boundary_hit[0]
    longitude = dateline.footprint_boundary.boundary_longitude_deg[0, hit]
    latitude = dateline.footprint_boundary.boundary_geodetic_latitude_deg[0, hit]
    assert np.all(np.isfinite(longitude)) and np.all(np.isfinite(latitude))
    assert np.min(longitude) < -179.0 and np.max(longitude) > 179.0
    plot_lon, plot_lat = _boundary_plot_vectors(longitude, latitude, np.ones(hit.sum(), dtype=bool))
    assert np.any(np.isnan(plot_lon)) and np.any(np.isnan(plot_lat))

    context, positions, attitudes = _fixed_geodetic_evidence(
        times,
        latitude_deg=89.5,
        longitude_deg=0.0,
    )
    polar = evaluate_rich_coverage(
        config,
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    polar_hit = polar.footprint_boundary.boundary_hit[0]
    polar_latitude = polar.footprint_boundary.boundary_geodetic_latitude_deg[0, polar_hit]
    polar_longitude = polar.footprint_boundary.boundary_longitude_deg[0, polar_hit]
    assert np.all(np.isfinite(polar_latitude)) and np.all(np.isfinite(polar_longitude))
    assert np.max(np.abs(polar_latitude)) > 88.0
    assert BOUNDARY_DISPOSITION_NAMES[
        int(polar.footprint_boundary.boundary_disposition_code[0])
    ] == "complete"


def test_explicit_sun_evidence_controls_illumination_coverage() -> None:
    times = np.array([0.0, 60.0])
    context, positions, attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    constraints = SurfaceServiceConstraints(minimum_sun_elevation_rad=0.0)
    config = _rich_config(constraints=constraints, sun_provider_id="fixture.sun")
    day_sun = _fixed_ecef_positions_in_eci(times, context, np.array([1.5e8, 0.0, 0.0]))
    night_sun = _fixed_ecef_positions_in_eci(times, context, np.array([-1.5e8, 0.0, 0.0]))
    day = evaluate_rich_coverage(
        config,
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
        sun_positions_eci_km=day_sun,
    )
    night = evaluate_rich_coverage(
        config,
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
        sun_positions_eci_km=night_sun,
    )
    assert np.all(day.covered_cell_count > 0)
    np.testing.assert_array_equal(night.covered_cell_count, 0)
    assert night.summary["primary_reason_total"]["illumination_rejected"] > 0
    assert day.input_evidence_sha256 != night.input_evidence_sha256
    assert day.interval_semantic_sha256 != night.interval_semantic_sha256


def test_chunk_parity_and_phase2_queries_accept_rich_product() -> None:
    times = np.array([0.0, 30.0, 90.0])
    context, positions, attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    first = evaluate_rich_coverage(
        _rich_config(
            pattern=HardFOVPattern.rectangular(np.deg2rad(20.0), np.deg2rad(10.0)),
            chunk_size=257,
        ),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    second = evaluate_rich_coverage(
        _rich_config(
            pattern=HardFOVPattern.rectangular(np.deg2rad(20.0), np.deg2rad(10.0)),
            chunk_size=4096,
        ),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    np.testing.assert_array_equal(first.covered_cell_count, second.covered_cell_count)
    np.testing.assert_array_equal(first.primary_reason_count, second.primary_reason_count)
    np.testing.assert_array_equal(
        first.footprint_boundary.boundary_hit,
        second.footprint_boundary.boundary_hit,
    )
    assert first.interval_semantic_sha256 == second.interval_semantic_sha256

    all_cells = tuple(range(healpix_npix(first.config.order)))
    query = evaluate_coverage_queries(
        first,
        region_masks=[
            CoverageRegionMask(
                region_id="whole_grid",
                mask_version="phase3.v1",
                provenance="phase3-query-compatibility-test",
                cell_indices=all_cells,
            )
        ],
    )
    np.testing.assert_array_equal(query.regions[0].covered_cell_count, first.covered_cell_count)


def test_artifacts_and_plot_are_deterministic_and_source_bound(tmp_path) -> None:
    times = np.array([0.0, 60.0, 120.0])
    context, positions, attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    result = evaluate_rich_coverage(
        _rich_config(pattern=HardFOVPattern.pushbroom(np.deg2rad(15.0), np.deg2rad(3.0))),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    first = write_rich_coverage_artifacts(
        result,
        tmp_path / "first",
        include_footprint_plot=True,
    )
    second = write_rich_coverage_artifacts(result, tmp_path / "second")
    assert first.summary_json.read_bytes() == second.summary_json.read_bytes()
    assert first.samples_csv.read_bytes() == second.samples_csv.read_bytes()
    assert first.footprint_plot_png is not None and first.footprint_plot_png.stat().st_size > 1000
    assert first.footprint_plot_quality_json is not None
    assert json.loads(first.footprint_plot_quality_json.read_text(encoding="utf-8"))["automated_status"] == "passed"
    manifest = json.loads(first.manifest_json.read_text(encoding="utf-8"))
    assert manifest["input_evidence_sha256"] == result.input_evidence_sha256
    assert manifest["semantic_sha256"] == result.interval_semantic_sha256
    assert manifest["normalized_scientific_config"]["pattern"]["kind"] == "pushbroom_hard_fov"
    with np.load(first.footprints_npz) as footprint_data:
        np.testing.assert_array_equal(
            footprint_data["boundary_hit"].astype(bool),
            result.footprint_boundary.boundary_hit,
        )
    with pytest.raises(FileExistsError, match="already exists"):
        write_rich_coverage_artifacts(result, first.output_dir)


def test_validation_fails_closed_for_patterns_sun_and_resources() -> None:
    with pytest.raises(ValueError, match="identical"):
        HardFOVPattern("axisymmetric_hard_cone", 0.1, 0.2)
    with pytest.raises(ValueError, match="must not exceed"):
        SurfaceServiceConstraints(
            minimum_sun_elevation_rad=0.5,
            maximum_sun_elevation_rad=0.1,
        )
    with pytest.raises(ValueError, match="outside the WGS84"):
        intersect_rays_wgs84(np.zeros(3), np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="sun_provider_id"):
        _rich_config(constraints=SurfaceServiceConstraints(minimum_sun_elevation_rad=0.0))

    times = np.array([0.0, 60.0])
    context, positions, attitudes = _fixed_ecef_evidence(
        times,
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    constrained = _rich_config(
        constraints=SurfaceServiceConstraints(minimum_sun_elevation_rad=0.0),
        sun_provider_id="fixture.sun",
    )
    with pytest.raises(ValueError, match="Explicit Sun ECI positions"):
        evaluate_rich_coverage(
            constrained,
            times_s=times,
            positions_eci_km=positions,
            attitudes_quat_bn=attitudes,
            frame_context=context,
        )
    with pytest.raises(ValueError, match="when sun_provider_id is declared"):
        evaluate_rich_coverage(
            _rich_config(sun_provider_id="unused.sun"),
            times_s=times,
            positions_eci_km=positions,
            attitudes_quat_bn=attitudes,
            frame_context=context,
        )
    with pytest.raises(ValueError, match="cell-time comparisons"):
        evaluate_rich_coverage(
            _rich_config(max_comparisons=1),
            times_s=times,
            positions_eci_km=positions,
            attitudes_quat_bn=attitudes,
            frame_context=context,
        )
