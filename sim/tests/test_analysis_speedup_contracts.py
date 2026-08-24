from __future__ import annotations

import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

import sim.analysis.healpix as healpix_module
from sim.analysis.communications_coverage import _pattern_pass
from sim.analysis.coverage_aggregation import ConstellationCoverageConfig
from sim.analysis.directed_link import TerminalPattern
from sim.analysis.healpix import (
    cached_healpix_wgs84_centers,
    clear_healpix_center_cache,
    healpix_wgs84_centers,
)
from sim.core.models import StateBelief, StateTruth
from sim.mission.modules import MissionExecutiveStrategy


def test_cached_healpix_centers_preserve_exact_requested_shape() -> None:
    clear_healpix_center_cache()
    indices = np.arange(4, 20, dtype=np.int64)
    expected = healpix_wgs84_centers(3, indices)
    first = cached_healpix_wgs84_centers(3, indices)
    second = cached_healpix_wgs84_centers(3, indices.copy())
    assert first is second
    for field_name in (
        "cell_index",
        "authalic_latitude_rad",
        "longitude_rad",
        "geodetic_latitude_rad",
        "ecef_km",
        "outward_normal_ecef",
    ):
        actual = getattr(first, field_name)
        np.testing.assert_array_equal(actual, getattr(expected, field_name))
        assert not actual.flags.writeable
        with pytest.raises(ValueError):
            actual.setflags(write=True)

    noncontiguous = cached_healpix_wgs84_centers(3, np.array([4, 20], dtype=np.int64))
    np.testing.assert_array_equal(noncontiguous.cell_index, [4, 20])


def test_cached_healpix_centers_do_not_freeze_or_alias_caller_indices() -> None:
    clear_healpix_center_cache()
    caller_indices = np.arange(4, 20, dtype=np.int64)
    centers = cached_healpix_wgs84_centers(3, caller_indices)
    assert caller_indices.flags.writeable
    caller_indices[0] = 3
    np.testing.assert_array_equal(centers.cell_index, np.arange(4, 20, dtype=np.int64))


def test_healpix_cache_enforces_entry_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(healpix_module, "_CENTER_CACHE_MAX_ENTRIES", 2)
    clear_healpix_center_cache()
    first = cached_healpix_wgs84_centers(3, np.array([0], dtype=np.int64))
    cached_healpix_wgs84_centers(3, np.array([1], dtype=np.int64))
    cached_healpix_wgs84_centers(3, np.array([2], dtype=np.int64))
    reloaded = cached_healpix_wgs84_centers(3, np.array([0], dtype=np.int64))
    assert reloaded is not first
    clear_healpix_center_cache()


def test_direct_cosine_pattern_gate_is_explicit_and_matches_nonboundary_cases() -> None:
    pattern = TerminalPattern("axisymmetric_hard_cone", 0.0, np.deg2rad(30.0))
    directions = np.array(
        [
            [0.0, 0.0, 1.0],
            [np.sin(np.deg2rad(20.0)), 0.0, np.cos(np.deg2rad(20.0))],
            [np.sin(np.deg2rad(40.0)), 0.0, np.cos(np.deg2rad(40.0))],
        ]
    )
    _, exact = _pattern_pass(pattern, directions, gate_mode="exact_arccos")
    _, optimized = _pattern_pass(pattern, directions, gate_mode="direct_cosine")
    np.testing.assert_array_equal(optimized, exact)


def test_spatial_screening_fuses_minimum_with_exact_sample_index_tie_order() -> None:
    screening = pytest.importorskip("sim.scale.screening")
    store_models = pytest.importorskip("sim.scale.store_models")
    PropagationProductSummary = store_models.PropagationProductSummary
    PropagationSampleRecord = store_models.PropagationSampleRecord
    products = [
        PropagationProductSummary("product:a", "a", "ogp", "scalar", "teme", 1.0, 2.0, 1.0, 2, "a"),
        PropagationProductSummary("product:b", "b", "ogp", "scalar", "teme", 1.0, 2.0, 1.0, 2, "b"),
    ]

    def sample(index: int, jd: float, x: float, velocity: float) -> PropagationSampleRecord:
        return PropagationSampleRecord(
            index,
            jd,
            float(index),
            x,
            0.0,
            0.0,
            velocity,
            0.0,
            0.0,
            "",
        )

    samples = {
        "product:b": [sample(0, 10.0, 1.0, 3.0), sample(1, 11.0, 1.0, 9.0)],
        "product:a": [sample(0, 10.0, 0.0, 1.0), sample(1, 11.0, 0.0, 1.0)],
    }
    exact = screening._spatial_candidate_pairs(
        products,
        samples_by_product=samples,
        distance_threshold_km=2.0,
    )
    squared = screening._spatial_candidate_pairs(
        products,
        samples_by_product=samples,
        distance_threshold_km=2.0,
        distance_comparison="squared_distance",
    )
    assert exact[0][2] == (1.0, 10.0, 2.0)
    assert squared[0][2] == exact[0][2]


def test_mission_transition_reuses_range_metric_for_rearm_and_fire() -> None:
    def mode(name: str) -> dict[str, object]:
        return {
            "name": name,
            "mission_strategy": {
                "module": "sim.mission.modules",
                "class_name": "SafeHoldMissionStrategy",
                "params": {"attitude_mode": "hold_current"},
            },
            "mission_execution": {
                "module": "sim.mission.modules",
                "class_name": "SafeHoldExecution",
                "params": {},
            },
        }

    executive = MissionExecutiveStrategy(
        initial_mode="hold",
        modes=[mode("hold"), mode("defend")],
        transitions=[
            {
                "from_mode": "hold",
                "to_mode": "defend",
                "trigger": "range_lt",
                "target_id": "target",
                "threshold_km": 10.0,
            }
        ],
    )
    truth = StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0]),
        velocity_eci_km_s=np.zeros(3),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
        mass_kg=100.0,
        t_s=0.0,
    )
    belief = StateBelief(
        state=np.array([7005.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(6),
        last_update_t_s=0.0,
    )
    with patch.object(executive, "_range_km", wraps=executive._range_km) as range_metric:
        output = executive.update(
            truth=truth,
            own_knowledge={"target": belief},
            t_s=0.0,
        )
    assert output["mission_mode"]["executive_mode"] == "defend"
    assert range_metric.call_count == 1


def test_constellation_member_count_fails_closed_before_uint16_wrap() -> None:
    member_ids = tuple(f"member-{index:05d}" for index in range(65_536))
    with pytest.raises(ValueError, match="uint16 multiplicity contract limit of 65535"):
        ConstellationCoverageConfig(
            analysis_id="too-many-members",
            member_analysis_ids=member_ids,
            order=5,
            service_definition_id="test-service",
        )


def test_bulk_sample_loader_chunks_above_conservative_bind_limit(tmp_path) -> None:
    store_propagation = pytest.importorskip("sim.scale.store_propagation")
    store = tmp_path / "scale.sqlite"
    with sqlite3.connect(store) as conn:
        conn.execute(
            """
            CREATE TABLE propagation_samples (
                propagation_product_id TEXT NOT NULL,
                sample_index INTEGER NOT NULL,
                jd_utc REAL NOT NULL,
                t_s REAL NOT NULL,
                pos_x_km REAL,
                pos_y_km REAL,
                pos_z_km REAL,
                vel_x_km_s REAL,
                vel_y_km_s REAL,
                vel_z_km_s REAL,
                error TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO propagation_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("product-0000", 0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ""),
                ("product-1000", 0, 2.0, 0.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, ""),
            ],
        )
    product_ids = tuple(f"product-{index:04d}" for index in range(1001))
    loaded = store_propagation.load_product_samples_by_id(
        store,
        propagation_product_ids=product_ids,
    )
    assert tuple(loaded) == product_ids
    assert loaded["product-0000"][0].pos_x_km == 1.0
    assert loaded["product-1000"][0].pos_x_km == 7.0
