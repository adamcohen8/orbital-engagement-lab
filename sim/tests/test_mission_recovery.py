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
    assert planner["sources"] == ["analytic_reconstitution"]
    assert planner["recommendation_basis"] == "original_orbit_reconstitution"
    assert planner["candidate_count"] >= 2
    assert set(planner["recommended"]) == {"min_delta_v", "min_time", "constrained"}
    assert planner["recommended"]["min_time"] is not None
    assert planner["recommended"]["min_delta_v"] is not None
    assert all(candidate["burn_sequence"] for candidate in planner["candidates"])
    assert all(candidate["source_family"] == "analytic_reconstitution" for candidate in planner["candidates"])
    assert all(candidate["target_basis"] == "initial_orbit" for candidate in planner["candidates"])
    assert planner["candidates"][0]["planned_delta_v_m_s"] <= 10.0
    assert all(candidate["burn_sequence"][0]["duration_s"] is not None for candidate in planner["candidates"])


def test_same_apsis_shape_recovery_honors_element_tolerances() -> None:
    radius_km = EARTH_RADIUS_KM + 400.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    initial = np.zeros(14, dtype=float)
    initial[:6] = [radius_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    initial[13] = 100.0
    final = initial.copy()
    final[4] -= 0.005
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_shape",
                    "element_tolerances": {"a_km": 0.0, "ecc": 0.0},
                    "planner": {"enabled": True, "modes": ["min_delta_v"]},
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0, 1.0]),
        truth_hist={"target": np.vstack([initial, final])},
    )

    candidate = next(
        item for item in summary["planner"]["candidates"] if item["source"] == "same_apsis_shape_recovery"
    )
    assert candidate["within_tolerances"] is False
    assert candidate["verified"] is False


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


def test_existing_planner_source_is_a_compatibility_alias() -> None:
    cfg = scenario_config_from_dict(
        {
            "analysis": {
                "mission_recovery": {
                    "planner": {"enabled": True, "sources": ["existing"]},
                }
            }
        }
    )

    assert cfg.analysis.mission_recovery.planner["sources"] == ["analytic_reconstitution"]


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


def test_mission_recovery_orbit_transfer_planner_collapses_zero_impulse_slot_candidates() -> None:
    radius_km = EARTH_RADIUS_KM + 400.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    state = np.zeros(14, dtype=float)
    state[:6] = [radius_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    state[13] = 100.0
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
                        "sources": ["analytic_reconstitution", "orbit_transfer"],
                        "modes": ["min_delta_v", "min_time", "constrained"],
                        "max_recovery_time_s": 1200.0,
                        "max_recovery_delta_v_m_s": 1.0,
                        "candidate_count": 4,
                        "orbit_transfer": {
                            "enabled": True,
                            "departure_samples": 1,
                            "time_of_flight_samples": 2,
                            "min_time_of_flight_s": 600.0,
                            "max_time_of_flight_s": 1200.0,
                        },
                    },
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0]),
        truth_hist={"target": np.vstack([state])},
    )

    planner = summary["planner"]
    assert planner["sources"] == ["analytic_reconstitution", "orbit_transfer"]
    assert planner["recommended"]["min_delta_v"] is not None
    assert any(item["source_family"] == "analytic_reconstitution" for item in planner["candidates"])
    candidate = next(item for item in planner["candidates"] if item["source"] == "orbit_transfer_lambert")
    assert candidate["source"] == "orbit_transfer_lambert"
    assert candidate["transfer_type"] == "zero_impulse"
    assert candidate["departure_wait_s"] == pytest.approx(0.0)
    assert candidate["time_of_flight_s"] >= 600.0
    assert candidate["planned_delta_v_m_s"] <= 1.0
    assert candidate["burn_sequence"] == []
    assert any("Collapsed 2 Lambert impulse" in note for note in candidate["notes"])


def test_orbit_transfer_planner_rejects_unsupported_candidate_states_and_continues() -> None:
    initial = np.zeros(14, dtype=float)
    initial[:6] = [4792.866637, 0.0, -4792.866637, 0.0, 7.668558, 0.0]
    initial[13] = 100.0
    final = initial.copy()
    final[4] += 0.0001
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_slot",
                    "assessment_time_s": "final",
                    "slot_tolerance_deg": 1.0,
                    "max_phasing_orbits": 25,
                    "planner": {
                        "enabled": True,
                        "sources": ["analytic_reconstitution", "orbit_transfer"],
                        "modes": ["min_delta_v", "min_time", "constrained"],
                        "max_recovery_time_s": 86400.0,
                        "max_recovery_delta_v_m_s": 15.0,
                        "candidate_count": 12,
                        "simulate_candidates": True,
                    },
                    "propulsion": {
                        "spacecraft_mass_kg": 100.0,
                        "isp_s": 220.0,
                        "max_thrust_n": 20.0,
                    },
                    "element_tolerances": {"a_km": 1.0, "ecc": 0.001},
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0, 45.0]),
        truth_hist={"target": np.vstack([initial, final])},
    )

    planner = summary["planner"]
    rejections = planner["candidate_rejections"]
    assert rejections["by_reason"]["unsupported_post_transfer_orbit"] > 0
    assert rejections["total"] == sum(rejections["by_reason"].values())
    assert any(item["source_family"] == "analytic_reconstitution" for item in planner["candidates"])
    assert any(item["source_family"] == "orbit_transfer" for item in planner["candidates"])
    assert any("continued without aborting" in warning for warning in planner["warnings"])


def test_orbit_transfer_planner_reports_one_impulse_departure_when_arrival_burn_collapses() -> None:
    radius_km = 7000.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    period_s = float(2.0 * np.pi * np.sqrt(radius_km**3 / EARTH_MU_KM3_S2))
    time_of_flight_s = period_s / 6.0
    state = np.zeros(14, dtype=float)
    state[:6] = [radius_km, 0.0, 0.0, 0.0, 0.99 * circular_speed, 0.0]
    state[13] = 100.0
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0, "max_thrust_n": 20.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_slot",
                    "slot_tolerance_deg": 1.0,
                    "target_orbit": {
                        "coes": {
                            "a_km": radius_km,
                            "ecc": 0.0,
                            "inc_deg": 0.0,
                            "raan_deg": 0.0,
                            "argp_deg": 0.0,
                            "true_anomaly_deg": 0.0,
                        }
                    },
                    "planner": {
                        "enabled": True,
                        "sources": ["orbit_transfer"],
                        "modes": ["min_delta_v"],
                        "max_recovery_time_s": time_of_flight_s,
                        "max_recovery_delta_v_m_s": 100.0,
                        "candidate_count": 1,
                        "orbit_transfer": {
                            "enabled": True,
                            "departure_samples": 1,
                            "time_of_flight_samples": 1,
                            "min_time_of_flight_s": time_of_flight_s,
                            "max_time_of_flight_s": time_of_flight_s,
                            "impulse_epsilon_m_s": 1.0e-3,
                        },
                    },
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0]),
        truth_hist={"target": np.vstack([state])},
    )

    candidate = summary["planner"]["candidates"][0]
    assert candidate["source"] == "orbit_transfer_lambert"
    assert candidate["transfer_type"] == "one_impulse_departure"
    assert candidate["planned_delta_v_m_s"] == pytest.approx(0.01 * circular_speed * 1000.0, abs=1.0e-3)
    assert len(candidate["burn_sequence"]) == 1
    burn = candidate["burn_sequence"][0]
    assert burn["burn_index"] == 0
    assert burn["axis"] == "lambert_departure"
    assert burn["start_time_s"] == pytest.approx(0.0)
    assert burn["delta_v_m_s"] == pytest.approx(candidate["planned_delta_v_m_s"], abs=1.0e-3)


def test_orbit_transfer_planner_accepts_desired_target_orbit_shape() -> None:
    radius_km = EARTH_RADIUS_KM + 400.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    state = np.zeros(14, dtype=float)
    state[:6] = [radius_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    state[13] = 100.0
    target_a_km = radius_km + 100.0
    cfg = scenario_config_from_dict(
        {
            "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0, "max_thrust_n": 20.0}},
            "analysis": {
                "mission_recovery": {
                    "enabled": True,
                    "object_id": "target",
                    "goal": "orbit_shape",
                    "target_orbit": {
                        "coes": {
                            "a_km": target_a_km,
                            "ecc": 0.0,
                            "inc_deg": 0.0,
                            "raan_deg": 0.0,
                            "argp_deg": 0.0,
                            "true_anomaly_deg": 0.0,
                        }
                    },
                    "element_tolerances": {"a_km": 20.0, "ecc": 0.05},
                    "planner": {
                        "enabled": True,
                        "modes": ["min_delta_v"],
                        "max_recovery_time_s": 2400.0,
                        "max_recovery_delta_v_m_s": 5000.0,
                        "candidate_count": 3,
                        "orbit_transfer": {
                            "enabled": True,
                            "departure_samples": 1,
                            "time_of_flight_samples": 2,
                            "target_anomaly_samples": 8,
                            "min_time_of_flight_s": 600.0,
                            "max_time_of_flight_s": 2400.0,
                        },
                    },
                }
            },
        }
    )

    summary = build_mission_recovery_summary(
        cfg=cfg,
        t_s=np.array([0.0]),
        truth_hist={"target": np.vstack([state])},
    )

    assert summary["target_elements"]["a_km"] == pytest.approx(target_a_km)
    planner = summary["planner"]
    assert planner["sources"] == ["analytic_reconstitution", "orbit_transfer"]
    assert planner["recommendation_basis"] == "configured_target_orbit_lambert"
    assert planner["analytical_baseline_candidate_ids"]
    recommended_id = planner["recommended"]["min_delta_v"]
    candidate = next(item for item in planner["candidates"] if item["candidate_id"] == recommended_id)
    assert candidate["source"] == "orbit_transfer_lambert"
    assert candidate["source_family"] == "orbit_transfer"
    assert candidate["target_basis"] == "configured_target_orbit"
    assert candidate["planned_delta_v_m_s"] > 0.0
    assert len(candidate["burn_sequence"]) == 2
    assert abs(candidate["expected_final_elements"]["a_km"] - target_a_km) < 20.0
    baseline = next(
        item for item in planner["candidates"] if item["source_family"] == "analytic_reconstitution"
    )
    assert baseline["target_basis"] == "initial_orbit"
    assert baseline["candidate_id"] != recommended_id
    assert summary["recovery_estimate"]["scope"] == "original_orbit_reconstitution"
    assert summary["target_element_errors"]["a_km"] == pytest.approx(-100.0, abs=1.0e-6)


def test_orbit_transfer_planner_grid_refinement_approaches_hohmann_transfer() -> None:
    r1_km = 7000.0
    r2_km = 14000.0
    circular_speed = float(np.sqrt(EARTH_MU_KM3_S2 / r1_km))
    state = np.zeros(14, dtype=float)
    state[:6] = [r1_km, 0.0, 0.0, 0.0, circular_speed, 0.0]
    state[13] = 100.0

    def best_candidate(*, time_of_flight_samples: int, target_anomaly_samples: int) -> dict:
        cfg = scenario_config_from_dict(
            {
                "target": {"enabled": True, "specs": {"mass_kg": 100.0, "isp_s": 220.0, "max_thrust_n": 20.0}},
                "analysis": {
                    "mission_recovery": {
                        "enabled": True,
                        "object_id": "target",
                        "goal": "orbit_shape",
                        "target_orbit": {
                            "coes": {
                                "a_km": r2_km,
                                "ecc": 0.0,
                                "inc_deg": 0.0,
                                "raan_deg": 0.0,
                                "argp_deg": 0.0,
                                "true_anomaly_deg": 180.0,
                            }
                        },
                        "element_tolerances": {"a_km": 100.0, "ecc": 0.1},
                        "planner": {
                            "enabled": True,
                            "sources": ["orbit_transfer"],
                            "modes": ["min_delta_v"],
                            "max_recovery_time_s": 7000.0,
                            "max_recovery_delta_v_m_s": 10000.0,
                            "candidate_count": 3,
                            "orbit_transfer": {
                                "enabled": True,
                                "departure_samples": 1,
                                "time_of_flight_samples": time_of_flight_samples,
                                "target_anomaly_samples": target_anomaly_samples,
                                "min_time_of_flight_s": 1000.0,
                                "max_time_of_flight_s": 7000.0,
                            },
                        },
                    }
                },
            }
        )
        summary = build_mission_recovery_summary(
            cfg=cfg,
            t_s=np.array([0.0]),
            truth_hist={"target": np.vstack([state])},
        )
        return summary["planner"]["candidates"][0]

    coarse = best_candidate(time_of_flight_samples=3, target_anomaly_samples=4)
    refined = best_candidate(time_of_flight_samples=20, target_anomaly_samples=72)
    transfer_a_km = 0.5 * (r1_km + r2_km)
    expected_hohmann_delta_v_m_s = (
        abs(np.sqrt(EARTH_MU_KM3_S2 * (2.0 / r1_km - 1.0 / transfer_a_km)) - np.sqrt(EARTH_MU_KM3_S2 / r1_km))
        + abs(np.sqrt(EARTH_MU_KM3_S2 / r2_km) - np.sqrt(EARTH_MU_KM3_S2 * (2.0 / r2_km - 1.0 / transfer_a_km)))
    ) * 1000.0

    assert coarse["planned_delta_v_m_s"] > refined["planned_delta_v_m_s"] + 3000.0
    assert refined["planned_delta_v_m_s"] == pytest.approx(expected_hohmann_delta_v_m_s, abs=50.0)
    assert refined["target_phase_deg"] == pytest.approx(175.0)


def test_mission_recovery_planner_config_validates_orbit_transfer_sources() -> None:
    with pytest.raises(ValueError, match="analysis.mission_recovery.planner.sources"):
        scenario_config_from_dict(
            {
                "analysis": {
                    "mission_recovery": {
                        "enabled": True,
                        "goal": "orbit_shape",
                        "planner": {"enabled": True, "sources": ["magic"]},
                    }
                }
            }
        )


def test_mission_recovery_config_validates_target_orbit_coes() -> None:
    with pytest.raises(ValueError, match="analysis.mission_recovery.target_orbit.coes.a_km"):
        scenario_config_from_dict(
            {
                "analysis": {
                    "mission_recovery": {
                        "enabled": True,
                        "goal": "orbit_shape",
                        "target_orbit": {"coes": {"a_km": -1.0}},
                    }
                }
            }
        )


def test_orbit_transfer_planner_rejects_unsupported_multi_revolution_search() -> None:
    with pytest.raises(ValueError, match="multi_revolution_max"):
        scenario_config_from_dict(
            {
                "analysis": {
                    "mission_recovery": {
                        "enabled": True,
                        "goal": "orbit_shape",
                        "planner": {
                            "enabled": True,
                            "sources": ["orbit_transfer"],
                            "orbit_transfer": {"multi_revolution_max": 1},
                        },
                    }
                }
            }
        )


def test_orbit_transfer_planner_rejects_negative_impulse_epsilon() -> None:
    with pytest.raises(ValueError, match="impulse_epsilon_m_s"):
        scenario_config_from_dict(
            {
                "analysis": {
                    "mission_recovery": {
                        "enabled": True,
                        "goal": "orbit_shape",
                        "planner": {
                            "enabled": True,
                            "sources": ["orbit_transfer"],
                            "orbit_transfer": {"impulse_epsilon_m_s": -1.0e-3},
                        },
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
