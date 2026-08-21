from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sim.analysis.coverage_queries import (
    COVERAGE_QUERY_SCHEMA_VERSION,
    CoveragePointQuery,
    CoverageRegionMask,
    evaluate_coverage_queries,
    write_coverage_query_artifacts,
)
from sim.analysis.global_coverage import (
    GlobalCoverageConfig,
    GlobalCoverageResult,
    summarize_sampled_coverage_mask,
)
from sim.analysis.healpix import (
    WGS84_SURFACE_AREA_KM2,
    healpix_npix,
    healpix_wgs84_centers,
    wgs84_points_to_healpix_nested,
)


def _synthetic_global_result() -> GlobalCoverageResult:
    order = 5
    cells = healpix_wgs84_centers(order)
    times = np.array([0.0, 1.0, 2.0, 4.0])
    mask = np.zeros((times.size, healpix_npix(order)), dtype=bool)
    mask[:, 10] = [False, True, True, False]
    mask[:, 20] = [True, True, False, True]
    metrics = summarize_sampled_coverage_mask(mask, times)
    covered_count = np.count_nonzero(mask, axis=1).astype(np.int64)
    config = GlobalCoverageConfig(
        analysis_id="phase2_synthetic",
        source_asset_id="spacecraft",
        state_provider_id="spacecraft.truth",
        attitude_source_kind="achieved",
        attitude_provider_id="spacecraft.attitude_truth",
        sensor_id="spacecraft.imager",
        order=order,
        half_angle_rad=0.2,
        quat_body_from_sensor=(1.0, 0.0, 0.0, 0.0),
    )
    return GlobalCoverageResult(
        config=config,
        frame_metadata={"frame": "synthetic"},
        times_s=times,
        covered_cell_count=covered_count,
        instantaneous_covered_fraction=covered_count.astype(float) / healpix_npix(order),
        cell_geodetic_latitude_deg=np.rad2deg(cells.geodetic_latitude_rad),
        cell_longitude_deg=np.rad2deg(cells.longitude_rad),
        cell_metrics=metrics,
        summary={
            "status": "complete",
            "domain_disposition": "global_earth",
            "analysis_id": "phase2_synthetic",
            "grid_identity": "healpix_nest_wgs84_authalic_v1",
            "order": order,
            "sample_count": int(times.size),
            "horizon_start_s": float(times[0]),
            "horizon_end_s": float(times[-1]),
        },
        resource_estimate={},
        input_evidence_sha256="0" * 64,
        interval_semantic_sha256="1" * 64,
    )


def _region(region_id: str, cells: tuple[int, ...]) -> CoverageRegionMask:
    return CoverageRegionMask(
        region_id=region_id,
        mask_version="2026-08-19.v1",
        provenance="unit-test-declared-mask",
        cell_indices=cells,
    )


def _point_at_cell(result: GlobalCoverageResult, cell_index: int, point_id: str) -> CoveragePointQuery:
    return CoveragePointQuery(
        point_id=point_id,
        longitude_deg=float(result.cell_longitude_deg[cell_index]),
        geodetic_latitude_deg=float(result.cell_geodetic_latitude_deg[cell_index]),
    )


def test_wgs84_cell_center_round_trip_and_periodic_longitude() -> None:
    for order in (0, 5):
        centers = healpix_wgs84_centers(order)
        mapped = wgs84_points_to_healpix_nested(
            order,
            np.rad2deg(centers.geodetic_latitude_rad),
            np.rad2deg(centers.longitude_rad),
        )
        np.testing.assert_array_equal(mapped, centers.cell_index)

    for order in (6, 7, 8):
        sampled_cells = np.unique(np.linspace(0, healpix_npix(order) - 1, 257, dtype=np.int64))
        centers = healpix_wgs84_centers(order, sampled_cells)
        mapped = wgs84_points_to_healpix_nested(
            order,
            np.rad2deg(centers.geodetic_latitude_rad),
            np.rad2deg(centers.longitude_rad),
        )
        np.testing.assert_array_equal(mapped, sampled_cells)

    antimeridian = wgs84_points_to_healpix_nested(5, [0.0, 0.0], [180.0, -180.0])
    np.testing.assert_array_equal(antimeridian, [6570, 6570])
    periodic = wgs84_points_to_healpix_nested(5, [20.0, -20.0], [725.0, -715.0])
    canonical = wgs84_points_to_healpix_nested(5, [20.0, -20.0], [5.0, 5.0])
    np.testing.assert_array_equal(periodic, canonical)
    poles = wgs84_points_to_healpix_nested(5, [90.0, -90.0], [0.0, 0.0])
    assert np.all((poles >= 0) & (poles < healpix_npix(5)))


def test_region_metrics_use_equal_area_and_match_source_subset() -> None:
    source = _synthetic_global_result()
    result = evaluate_coverage_queries(
        source,
        region_masks=[_region("three_cells", (10, 20, 30))],
    )
    region = result.regions[0]
    np.testing.assert_array_equal(region.covered_cell_count, [1, 2, 1, 1])
    np.testing.assert_allclose(region.instantaneous_covered_fraction, [1 / 3, 2 / 3, 1 / 3, 1 / 3])
    cell_area = WGS84_SURFACE_AREA_KM2 / healpix_npix(5)
    np.testing.assert_allclose(region.covered_area_km2, region.covered_cell_count * cell_area)
    np.testing.assert_array_equal(region.cell_metrics.cell_index, [10, 20, 30])
    np.testing.assert_allclose(region.cell_metrics.dwell_s, [3.0, 2.0, 0.0])
    np.testing.assert_array_equal(region.cell_metrics.interval_count, [1, 2, 0])
    assert region.summary["domain_disposition"] == "region_query"
    assert region.summary["region_area_km2"] == pytest.approx(3.0 * cell_area)
    assert region.summary["ever_covered_fraction"] == pytest.approx(2.0 / 3.0)
    assert region.summary["max_complete_revisit_gap_s"]["maximum"] == pytest.approx(2.0)
    assert len(region.mask_semantic_sha256) == 64


def test_all_cell_region_aggregation_equals_global_product() -> None:
    source = _synthetic_global_result()
    all_cells = tuple(range(healpix_npix(source.config.order)))
    result = evaluate_coverage_queries(source, region_masks=[_region("whole_grid", all_cells)])
    region = result.regions[0]
    np.testing.assert_array_equal(region.covered_cell_count, source.covered_cell_count)
    np.testing.assert_allclose(
        region.instantaneous_covered_fraction,
        source.instantaneous_covered_fraction,
        rtol=0.0,
        atol=0.0,
    )
    assert region.summary["ever_covered_cell_count"] == 2
    assert region.summary["cell_count"] == healpix_npix(source.config.order)


def test_point_query_inherits_containing_cell_intervals_and_censoring() -> None:
    source = _synthetic_global_result()
    point = _point_at_cell(source, 20, "cell_20_center")
    result = evaluate_coverage_queries(source, points=[point])
    queried = result.points[0]
    assert queried.cell_index == 20
    np.testing.assert_array_equal(queried.covered_by_sample, [True, True, False, True])
    np.testing.assert_allclose(queried.cell_metrics.dwell_s, [2.0])
    assert queried.summary["domain_disposition"] == "point_cell_query"
    assert queried.summary["resolution_dependent_cell_result"] is True
    assert queried.summary["start_censored"] is True
    assert queried.summary["end_censored"] is True
    assert queried.summary["max_complete_revisit_gap_s"] == pytest.approx(2.0)


def test_query_identity_is_sorted_and_semantically_bound() -> None:
    source = _synthetic_global_result()
    region_a = _region("a", (10, 20))
    region_b = _region("b", (20, 30))
    point_a = _point_at_cell(source, 10, "point_a")
    point_b = _point_at_cell(source, 20, "point_b")
    first = evaluate_coverage_queries(
        source,
        region_masks=[region_b, region_a],
        points=[point_b, point_a],
    )
    second = evaluate_coverage_queries(
        source,
        region_masks=[region_a, region_b],
        points=[point_a, point_b],
    )
    assert [item.mask.region_id for item in first.regions] == ["a", "b"]
    assert [item.query.point_id for item in first.points] == ["point_a", "point_b"]
    assert first.query_semantic_sha256 == second.query_semantic_sha256

    changed = CoverageRegionMask(
        region_id="a",
        mask_version=region_a.mask_version,
        provenance="different-provenance",
        cell_indices=region_a.cell_indices,
    )
    third = evaluate_coverage_queries(source, region_masks=[changed, region_b], points=[point_a, point_b])
    assert third.query_semantic_sha256 != first.query_semantic_sha256


def test_query_validation_fails_closed() -> None:
    source = _synthetic_global_result()
    with pytest.raises(ValueError, match="At least one"):
        evaluate_coverage_queries(source)
    with pytest.raises(ValueError, match="strictly increasing"):
        _region("bad", (20, 10))
    with pytest.raises(ValueError, match="one-dimensional integer"):
        CoverageRegionMask(
            region_id="bad",
            mask_version="v1",
            provenance="test",
            cell_indices=(10.0, 20.0),
        )
    with pytest.raises(ValueError, match="outside"):
        evaluate_coverage_queries(
            source,
            region_masks=[_region("bad", (healpix_npix(5),))],
        )
    duplicate_id = _point_at_cell(source, 10, "same")
    with pytest.raises(ValueError, match="identifiers must be unique"):
        evaluate_coverage_queries(
            source,
            region_masks=[_region("same", (10,))],
            points=[duplicate_id],
        )
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        CoveragePointQuery(point_id="bad", longitude_deg=0.0, geodetic_latitude_deg=91.0)
    normalized = CoveragePointQuery(point_id="normalized", longitude_deg=180.0, geodetic_latitude_deg=0.0)
    assert normalized.longitude_deg == -180.0

    with pytest.raises(ValueError, match="global-Earth source disposition"):
        evaluate_coverage_queries(
            replace(source, summary={"status": "complete", "domain_disposition": "region_query"}),
            points=[_point_at_cell(source, 10, "point")],
        )
    with pytest.raises(ValueError, match="sampled covered-cell counts"):
        evaluate_coverage_queries(
            replace(source, covered_cell_count=source.covered_cell_count + 1),
            points=[_point_at_cell(source, 10, "point")],
        )


def test_query_artifacts_are_deterministic_and_bind_source(tmp_path) -> None:
    source = _synthetic_global_result()
    query_result = evaluate_coverage_queries(
        source,
        region_masks=[_region("three_cells", (10, 20, 30))],
        points=[_point_at_cell(source, 20, "point")],
    )
    first = write_coverage_query_artifacts(query_result, tmp_path / "first")
    second = write_coverage_query_artifacts(query_result, tmp_path / "second")
    assert first.queries_json.read_bytes() == second.queries_json.read_bytes()
    assert first.region_samples_csv is not None and second.region_samples_csv is not None
    assert first.region_samples_csv.read_bytes() == second.region_samples_csv.read_bytes()
    assert first.point_samples_csv is not None and second.point_samples_csv is not None
    assert first.point_samples_csv.read_bytes() == second.point_samples_csv.read_bytes()

    manifest = json.loads(first.manifest_json.read_text(encoding="utf-8"))
    queries = json.loads(first.queries_json.read_text(encoding="utf-8"))
    assert manifest["query_schema_version"] == COVERAGE_QUERY_SCHEMA_VERSION
    assert manifest["source_interval_semantic_sha256"] == source.interval_semantic_sha256
    assert manifest["query_semantic_sha256"] == query_result.query_semantic_sha256
    assert queries["regions"][0]["cell_indices"] == [10, 20, 30]
    assert queries["points"][0]["cell_index"] == 20
    with pytest.raises(FileExistsError, match="already exists"):
        write_coverage_query_artifacts(query_result, first.output_dir)
