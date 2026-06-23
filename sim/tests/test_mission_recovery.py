from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim import SimulationConfig
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.reporting.mission_recovery import build_mission_recovery_summary, write_mission_recovery_trade_space_plot


def scenario_config_from_dict(data: dict):
    return SimulationConfig.from_dict(data).to_scenario_config()


def test_mission_recovery_infers_retrograde_intrack_impulse_from_simulated_state() -> None:
    radius_km = EARTH_RADIUS_KM + 400.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    initial = np.zeros(14, dtype=float)
    initial[:6] = [radius_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    initial[13] = 100.0
    final = initial.copy()
    final[4] -= 0.005
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0, "max_thrust_n": 20.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_slot",
                    "slot_tolerance_deg": 1.0,
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0, 1.0]),
        truth_hist={"target": np.vstack([initial, final])},
    )

    estimate = summary["recovery_estimate"]
    assert estimate["method"] == "sim_state_inferred_intrack_impulse"
    assert estimate["disturbance_delta_v_m_s"] == pytest.approx(-5.0)
    assert estimate["recovery_delta_v_m_s"] == pytest.approx(5.0)
    assert estimate["propellant_kg"] == pytest.approx(0.231485, rel=1e-5)
    assert estimate["slot_recovery_found"] is True
    assert estimate["slot_recovery_time_s"] is not None
    assert summary["element_errors"]["a_km"] < 0.0


def test_mission_recovery_planner_returns_trade_space_recommendations() -> None:
    radius_km = EARTH_RADIUS_KM + 400.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    initial = np.zeros(14, dtype=float)
    initial[:6] = [radius_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    initial[13] = 100.0
    final = initial.copy()
    final[4] -= 0.005
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0, "max_thrust_n": 20.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_shape",
                    "element_tolerances": {"a_km": 1.0, "ecc": 0.001},
                    "planner": {
                        "enabled": True,
                        "modes": ["min_delta_v", "min_time", "constrained"],
                        "max_recovery_time_s": 7200.0,
                        "max_recovery_delta_v_m_s": 10.0,
                        "candidate_count": 5,
                    },
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0, 1.0]),
        truth_hist={"target": np.vstack([initial, final])},
    )

    planner = summary["planner"]
    assert planner["candidate_count"] >= 2
    assert set(planner["recommended"]) == {"min_delta_v", "min_time", "constrained"}
    assert planner["recommended"]["min_time"] is not None
    assert planner["recommended"]["min_delta_v"] is not None
    assert all(candidate["burn_sequence"] for candidate in planner["candidates"])
    assert planner["candidates"][0]["planned_delta_v_m_s"] <= 10.0
    assert all(candidate["burn_sequence"][0]["duration_s"] is not None for candidate in planner["candidates"])


def test_mission_recovery_planner_reports_signed_intrack_recovery_axis() -> None:
    radius_km = EARTH_RADIUS_KM + 400.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    initial = np.zeros(14, dtype=float)
    initial[:6] = [radius_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    initial[13] = 100.0
    final = initial.copy()
    final[4] -= 0.005
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0, "max_thrust_n": 20.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_slot",
                    "slot_tolerance_deg": 1.0,
                    "planner": {
                        "enabled": True,
                        "modes": ["min_delta_v"],
                        "max_recovery_time_s": 7200.0,
                        "max_recovery_delta_v_m_s": 10.0,
                        "candidate_count": 2,
                    },
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0, 1.0]),
        truth_hist={"target": np.vstack([initial, final])},
    )

    candidate = summary["planner"]["candidates"][0]
    burn = candidate["burn_sequence"][0]
    assert burn["axis"] == "+I"
    assert burn["duration_s"] == pytest.approx(25.0, rel=1e-3)


def test_mission_recovery_trade_space_plot_writes_png(tmp_path: Path) -> None:
    radius_km = EARTH_RADIUS_KM + 400.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    initial = np.zeros(14, dtype=float)
    initial[:6] = [radius_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    initial[13] = 100.0
    final = initial.copy()
    final[4] -= 0.005
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0, "max_thrust_n": 20.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_shape",
                    "element_tolerances": {"a_km": 1.0, "ecc": 0.001},
                    "planner": {
                        "enabled": True,
                        "modes": ["min_delta_v", "min_time", "constrained"],
                        "max_recovery_time_s": 7200.0,
                        "max_recovery_delta_v_m_s": 10.0,
                        "candidate_count": 5,
                    },
                }
            },
        }
    )
    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0, 1.0]),
        truth_hist={"target": np.vstack([initial, final])},
    )

    path = write_mission_recovery_trade_space_plot(
        mission_recovery=summary,
        outdir=tmp_path,
        mode="save",
        dpi=90,
    )

    assert path is not None
    assert Path(path).is_file()
    assert Path(path).suffix == ".png"


def test_mission_recovery_zero_disturbance_has_zero_shape_wait() -> None:
    radius_km = EARTH_RADIUS_KM + 400.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    initial = np.zeros(14, dtype=float)
    initial[:6] = [radius_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    initial[13] = 100.0
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_shape",
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0, 1.0]),
        truth_hist={"target": np.vstack([initial, initial.copy()])},
    )

    estimate = summary["recovery_estimate"]
    assert estimate["method"] == "sim_state_inferred_intrack_impulse"
    assert estimate["disturbance_apsis"] == "circular"
    assert estimate["recovery_delta_v_m_s"] == pytest.approx(0.0)
    assert estimate["recovery_time_s"] == pytest.approx(0.0)


def test_mission_recovery_config_requires_orbit_shape_or_orbit_slot() -> None:
    with pytest.raises(ValueError, match="analysis.mission_recovery.goal"):
        scenario_config_from_dict(
            {
                "analysis": {
                    "mission_recovery": {
                        "enabled": True,
                        "goal": "slot",
                    }
                }
            }
        )


def test_mission_recovery_planner_config_validates_modes() -> None:
    with pytest.raises(ValueError, match="analysis.mission_recovery.planner.modes"):
        scenario_config_from_dict(
            {
                "analysis": {
                    "mission_recovery": {
                        "enabled": True,
                        "goal": "orbit_shape",
                        "planner": {"enabled": True, "modes": ["fastest"]},
                    }
                }
            }
        )


def test_mission_recovery_cross_track_burn_uses_local_shape_fallback() -> None:
    radius_km = EARTH_RADIUS_KM + 400.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    initial = np.zeros(14, dtype=float)
    initial[:6] = [radius_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    initial[13] = 100.0
    final = initial.copy()
    final[5] += 0.005
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_shape",
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0, 1.0]),
        truth_hist={"target": np.vstack([initial, final])},
    )

    estimate = summary["recovery_estimate"]
    assert estimate["method"] == "local_orbit_shape_velocity_match"
    assert estimate["recovery_delta_v_m_s"] == pytest.approx(5.0, rel=1e-3)
    assert abs(summary["element_errors"]["inc_deg"]) > 0.01
