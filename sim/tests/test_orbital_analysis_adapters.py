from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np
import pytest

from sim.analysis.global_coverage import GlobalCoverageConfig, evaluate_global_coverage
from sim.analysis.history_adapters import (
    AnalysisHistory,
    global_coverage_refinement_evaluator,
    history_from_ogp_product,
)
from sim.api import SimulationConfig, SimulationSession
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.utils.geodesy import WGS84_A_KM
from sim.utils.quaternion import dcm_to_quaternion_bn


def test_history_adapter_uses_hermite_and_shortest_arc_slerp() -> None:
    history = AnalysisHistory(
        object_id="sat", product_kind="onp_completed_run", state_provider_id="run:sat",
        frame="eci", initial_jd_utc=2451545.0, times_s=np.array([0.0, 2.0]),
        position_eci_km=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        velocity_eci_km_s=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        attitude_quat_bn=np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]),
        attitude_source_kind="achieved", attitude_provider_id="run:sat:attitude",
    )
    midpoint = history.state_at(1.0)
    np.testing.assert_allclose(midpoint.position_eci_km, [1.0, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(midpoint.velocity_eci_km_s, [1.0, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(midpoint.attitude_quat_bn, [1.0, 0.0, 0.0, 0.0], atol=1.0e-14)


@dataclass
class _OGPProduct:
    propagation_product_id: str = "ogp:one"
    object_id: str = "catalog_sat"
    output_frame: str = "eci"
    start_jd_utc: float = 2451545.0
    status: str = "completed"
    samples: list[dict] | None = None


def test_ogp_adapter_requires_eci_and_does_not_invent_attitude() -> None:
    samples = [
        {"t_s": 0.0, "pos_x_km": 7000.0, "pos_y_km": 0.0, "pos_z_km": 0.0,
         "vel_x_km_s": 0.0, "vel_y_km_s": 7.5, "vel_z_km_s": 0.0, "error": ""},
        {"t_s": 60.0, "pos_x_km": 6985.0, "pos_y_km": 449.0, "pos_z_km": 0.0,
         "vel_x_km_s": -0.48, "vel_y_km_s": 7.48, "vel_z_km_s": 0.0, "error": ""},
    ]
    history = history_from_ogp_product(_OGPProduct(samples=samples))
    assert history.product_kind == "ogp_product"
    assert history.attitude_quat_bn is None
    assert history.attitude_source_kind == "not_required"
    with pytest.raises(ValueError, match="Directional terminal"):
        history.link_endpoint(require_attitude=True)
    replay = history.with_attitude_replay(
        np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        attitude_source_kind="replay",
        attitude_provider_id="catalog_sat.attitude_replay",
    )
    assert replay.link_endpoint(require_attitude=True).attitude_source_kind == "replay"
    with pytest.raises(ValueError, match="ECI output product"):
        history_from_ogp_product(_OGPProduct(output_frame="teme", samples=samples))


def _attitude_for_boresight(boresight_eci: np.ndarray) -> np.ndarray:
    body_z = np.asarray(boresight_eci, dtype=float)
    body_z /= np.linalg.norm(body_z)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, body_z))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    body_x = np.cross(reference, body_z)
    body_x /= np.linalg.norm(body_x)
    body_y = np.cross(body_z, body_x)
    return dcm_to_quaternion_bn(np.vstack((body_x, body_y, body_z)))


def test_global_coverage_uses_provider_backed_transition_refinement() -> None:
    times = np.array([0.0, 10.0])
    frame_context = FrameContext(jd_utc_start=2451545.0)
    position_ecef = np.array([WGS84_A_KM + 500.0, 0.0, 0.0])
    positions = []
    attitudes = []
    for index, time_s in enumerate(times):
        rotation = eci_to_ecef_rotation_context(float(time_s), frame_context)
        positions.append(rotation.T @ position_ecef)
        boresight_ecef = np.array([1.0, 0.0, 0.0]) if index == 0 else np.array([-1.0, 0.0, 0.0])
        attitudes.append(_attitude_for_boresight(rotation.T @ boresight_ecef))
    history = AnalysisHistory(
        object_id="sat", product_kind="onp_completed_run", state_provider_id="run:sat",
        frame="eci", initial_jd_utc=2451545.0, times_s=times,
        position_eci_km=np.asarray(positions), velocity_eci_km_s=np.zeros((2, 3)),
        attitude_quat_bn=np.asarray(attitudes), attitude_source_kind="achieved",
        attitude_provider_id="run:sat:attitude",
    )
    config = GlobalCoverageConfig(
        analysis_id="refined_coverage", source_asset_id="sat", state_provider_id="run:sat",
        attitude_source_kind="achieved", attitude_provider_id="run:sat:attitude", sensor_id="sat.sensor",
        order=5, half_angle_rad=float(np.deg2rad(20.0)), quat_body_from_sensor=(1.0, 0.0, 0.0, 0.0),
        transition_time_tolerance_s=0.1, transition_max_iterations=20,
    )
    evaluator = global_coverage_refinement_evaluator(
        history, order=5, half_angle_rad=config.half_angle_rad,
        quat_body_from_sensor=config.quat_body_from_sensor, max_range_km=None,
        frame_context=frame_context,
    )
    result = evaluate_global_coverage(
        config, times_s=times, positions_eci_km=history.position_eci_km,
        attitudes_quat_bn=history.attitude_quat_bn, frame_context=frame_context,
        evaluator_at_time=evaluator, refinement_provider_id="run:sat:history_hermite_slerp",
    )
    assert result.refined_intervals
    assert result.refined_transitions
    assert all(
        transition.bracket_start_s <= transition.time_s <= transition.bracket_end_s
        for transition in result.refined_transitions
    )
    acquisitions = [row for row in result.refined_intervals if row.acquisition_disposition == "provider_refined"]
    assert acquisitions
    assert all(0.0 < row.start_s < 10.0 for row in acquisitions)
    assert result.summary["transition_refinement"]["evaluator_call_count"] > 0


def test_scenario_adapter_writes_coverage_and_link_review_tables(tmp_path) -> None:
    config = {
        "scenario_name": "orbital_analysis_adapter_smoke",
        "objects": {
            "tx": {
                "enabled": True,
                "initial_state": {"position_eci_km": [0.0, 0.0, 6878.137], "velocity_eci_km_s": [7.612, 0.0, 0.0]},
            },
            "rx": {
                "enabled": True,
                "initial_state": {"position_eci_km": [0.0, 0.0, 6888.137], "velocity_eci_km_s": [7.606, 0.0, 0.0]},
            },
        },
        "simulator": {
            "duration_s": 2.0, "dt_s": 1.0, "initial_jd_utc": 2451545.0,
            "dynamics": {"attitude": {"enabled": True}},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(tmp_path), "mode": "save",
            "stats": {"print_summary": False, "save_json": True, "save_full_log": True},
            "plots": {"enabled": False}, "animations": {"enabled": False},
            "review": {"enabled": True, "detail": "standard"},
            "orbital_analysis": {
                "enabled": True,
                "coverage": [{
                    "analysis_id": "earth_view", "source_object_id": "tx", "sensor_id": "tx.imager",
                    "order": 5, "half_angle_deg": 20.0,
                    "quat_body_from_sensor": [0.0, 1.0, 0.0, 0.0],
                    "transition_time_tolerance_s": 0.05, "transition_max_iterations": 20,
                }],
                "directed_links": [{
                    "analysis_id": "tx_to_rx", "link_id": "tx_to_rx", "tx_object_id": "tx", "rx_object_id": "rx",
                    "tx_terminal": {"terminal_id": "tx.rf", "pattern": {"kind": "constant", "gain_dbi": 20.0}},
                    "rx_terminal": {"terminal_id": "rx.rf", "pattern": {"kind": "constant", "gain_dbi": 20.0}},
                    "carrier_frequency_hz": 2.0e9, "tx_power_w": 10.0, "data_rate_bps": 1.0e6,
                    "system_noise_temperature_k": 300.0, "required_eb_n0_db": 5.0,
                    "transition_time_tolerance_s": 0.05, "transition_max_iterations": 20,
                    "include_margin_plot": False,
                }],
            },
        },
    }
    result = SimulationSession.from_config(SimulationConfig.from_dict(config)).run()
    assert result.summary["orbital_analysis"] == {
        "coverage_analysis_count": 1,
        "directed_link_analysis_count": 1,
        "schema_version": "oel.scenario-orbital-analysis.v1",
    }
    database = tmp_path / "review" / "run.sqlite"
    run_log = json.loads((tmp_path / "master_run_log.json").read_text(encoding="utf-8"))
    assert "intervals" not in run_log["orbital_analysis"]["coverage"][0]
    assert "samples" not in run_log["orbital_analysis"]["directed_links"][0]
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT analysis_id FROM coverage_summary").fetchone() == ("earth_view",)
        assert conn.execute("SELECT COUNT(*) FROM coverage_samples").fetchone()[0] == 3
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'coverage_transitions'"
        ).fetchone()[0] == 1
        assert conn.execute("SELECT analysis_id FROM link_summary").fetchone() == ("tx_to_rx",)
        assert conn.execute("SELECT COUNT(*) FROM link_samples").fetchone()[0] == 3
        disposition = conn.execute("SELECT acquisition_disposition FROM link_windows").fetchone()[0]
        assert disposition in {"study_start_censored", "provider_refined"}


def test_scenario_directional_analysis_fails_closed_without_attitude(tmp_path) -> None:
    config = {
        "scenario_name": "orbital_analysis_attitude_guard",
        "objects": {"sat": {"enabled": True, "initial_state": {
            "position_eci_km": [7000.0, 0.0, 0.0], "velocity_eci_km_s": [0.0, 7.5, 0.0]
        }}},
        "simulator": {"duration_s": 1.0, "dt_s": 1.0, "initial_jd_utc": 2451545.0,
                      "dynamics": {"attitude": {"enabled": False}}, "termination": {"earth_impact_enabled": False}},
        "outputs": {"output_dir": str(tmp_path), "stats": {"print_summary": False},
                    "plots": {"enabled": False}, "animations": {"enabled": False},
                    "orbital_analysis": {"enabled": True, "coverage": [{
                        "analysis_id": "invalid", "source_object_id": "sat", "sensor_id": "sensor",
                        "order": 5, "half_angle_deg": 20.0,
                    }]}},
    }
    with pytest.raises(ValueError, match="requires achieved attitude"):
        SimulationSession.from_config(SimulationConfig.from_dict(config)).run()
