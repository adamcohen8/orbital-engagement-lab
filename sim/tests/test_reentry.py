from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.dynamics.reentry import ReentryConfig, ReentryObjectProperties, reentry_metrics_for_state
from sim.master_outputs import AVAILABLE_FIGURE_IDS
from sim.single_run import _SingleRunEngine


def _reentry_config(tmp_path: Path) -> dict:
    return {
        "scenario_name": "reentry_smoke",
        "objects": {
            "capsule": {
                "kind": "satellite",
                "specs": {
                    "mass_kg": 100.0,
                    "drag_area_m2": 2.0,
                    "cd": 2.2,
                    "nose_radius_m": 0.4,
                },
                "initial_state": {
                    "position_eci_km": [6500.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.8, 0.0],
                },
            }
        },
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "dynamics": {
                "orbit": {"drag": False},
                "reentry": {
                    "enabled": True,
                    "begin_altitude_km": 130.0,
                    "object_ids": ["capsule"],
                    "atmosphere_model": "ussa1976",
                },
                "attitude": {"enabled": False},
            },
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(tmp_path),
            "mode": "save",
            "plots": {"enabled": False},
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
        },
    }


def test_reentry_metrics_activate_and_summarize(tmp_path: Path) -> None:
    cfg = scenario_config_from_dict(_reentry_config(tmp_path))
    engine = _SingleRunEngine(cfg)
    while not engine.done:
        engine.step()

    payload = engine.build_run_payload()
    metrics = payload["reentry_metrics_by_object"]["capsule"]
    summary = payload["summary"]["reentry_summary_by_object"]["capsule"]

    assert summary["entered_reentry"] is True
    assert summary["entry_time_s"] == 0.0
    assert np.nanmax(np.array(metrics["dynamic_pressure_pa"], dtype=float)) > 0.0
    assert np.nanmax(np.array(metrics["heat_rate_w_m2"], dtype=float)) > 0.0
    assert np.nanmax(np.array(metrics["heat_load_j_m2"], dtype=float)) > 0.0


def test_reentry_heat_rate_can_terminate_run(tmp_path: Path) -> None:
    raw = _reentry_config(tmp_path)
    raw["simulator"]["dynamics"]["reentry"]["termination"] = {
        "enabled": True,
        "max_heat_rate_w_m2": 1.0,
    }
    cfg = scenario_config_from_dict(raw)
    payload = _SingleRunEngine(cfg).run()
    summary = payload["summary"]

    assert summary["terminated_early"] is True
    assert summary["termination_reason"] == "reentry_heat_rate"
    assert summary["termination_object_id"] == "capsule"


def test_reentry_termination_by_object_can_enable_specific_vehicle(tmp_path: Path) -> None:
    raw = _reentry_config(tmp_path)
    raw["simulator"]["dynamics"]["reentry"]["termination"] = {
        "enabled": False,
        "by_object": {
            "capsule": {
                "enabled": True,
                "max_heat_rate_w_m2": 1.0,
            }
        },
    }
    cfg = scenario_config_from_dict(raw)
    payload = _SingleRunEngine(cfg).run()
    summary = payload["summary"]

    assert summary["terminated_early"] is True
    assert summary["termination_reason"] == "reentry_heat_rate"
    assert summary["termination_object_id"] == "capsule"


def test_reentry_termination_by_object_can_disable_specific_vehicle(tmp_path: Path) -> None:
    raw = _reentry_config(tmp_path)
    raw["simulator"]["dynamics"]["reentry"]["termination"] = {
        "enabled": True,
        "max_heat_rate_w_m2": 1.0,
        "by_object": {
            "capsule": {
                "enabled": False,
            }
        },
    }
    cfg = scenario_config_from_dict(raw)
    payload = _SingleRunEngine(cfg).run()
    summary = payload["summary"]

    assert summary["terminated_early"] is False
    assert summary["termination_reason"] is None
    assert summary["termination_object_id"] is None


def test_reentry_terminate_on_entry_can_stop_at_threshold(tmp_path: Path) -> None:
    raw = _reentry_config(tmp_path)
    raw["simulator"]["dynamics"]["reentry"]["termination"] = {
        "enabled": True,
        "terminate_on_entry": True,
    }
    cfg = scenario_config_from_dict(raw)
    payload = _SingleRunEngine(cfg).run()
    summary = payload["summary"]

    assert summary["terminated_early"] is True
    assert summary["termination_reason"] == "reentry_entry"
    assert summary["termination_time_s"] == 0.0
    assert summary["termination_object_id"] == "capsule"


def test_reentry_plot_ids_are_public() -> None:
    assert "reentry_summary" in AVAILABLE_FIGURE_IDS
    assert "reentry_aero" in AVAILABLE_FIGURE_IDS
    assert "reentry_thermal" in AVAILABLE_FIGURE_IDS
    assert "atmospheric_pass" in AVAILABLE_FIGURE_IDS


def test_reentry_plot_suite_writes_files(tmp_path: Path) -> None:
    raw = _reentry_config(tmp_path)
    raw["outputs"]["plots"] = {"enabled": True, "preset": "reentry"}
    cfg = scenario_config_from_dict(raw)
    payload = _SingleRunEngine(cfg).run()
    plots = payload["summary"]["plot_outputs"]

    assert Path(plots["reentry_summary"]).is_file()
    assert Path(plots["reentry_aero"]).is_file()
    assert Path(plots["reentry_thermal"]).is_file()


def test_atmospheric_pass_plot_writes_file(tmp_path: Path) -> None:
    raw = _reentry_config(tmp_path)
    raw["objects"]["capsule"]["specs"].update({"cl": 0.5, "lift_area_m2": 2.0, "lift_axis_body": [0.0, 0.0, 1.0]})
    raw["outputs"]["plots"] = {"enabled": True, "figure_ids": ["atmospheric_pass"]}
    cfg = scenario_config_from_dict(raw)
    payload = _SingleRunEngine(cfg).run()
    plots = payload["summary"]["plot_outputs"]

    assert Path(plots["atmospheric_pass"]).is_file()


def test_reentry_uses_per_object_nose_radius_for_multiple_satellites(tmp_path: Path) -> None:
    raw = _reentry_config(tmp_path)
    raw["objects"] = {
        "sharp": {
            **raw["objects"]["capsule"],
            "specs": {
                **raw["objects"]["capsule"]["specs"],
                "nose_radius_m": 0.25,
            },
        },
        "blunt": {
            **raw["objects"]["capsule"],
            "specs": {
                **raw["objects"]["capsule"]["specs"],
                "nose_radius_m": 1.0,
            },
        },
    }
    raw["simulator"]["dynamics"]["reentry"]["object_ids"] = []
    cfg = scenario_config_from_dict(raw)
    payload = _SingleRunEngine(cfg).run()
    metrics = payload["reentry_metrics_by_object"]

    sharp_heat = np.nanmax(np.array(metrics["sharp"]["heat_rate_w_m2"], dtype=float))
    blunt_heat = np.nanmax(np.array(metrics["blunt"]["heat_rate_w_m2"], dtype=float))

    assert sharp_heat > blunt_heat


def test_reentry_uses_nested_object_aero_specs(tmp_path: Path) -> None:
    raw = _reentry_config(tmp_path)
    raw["objects"]["capsule"]["specs"] = {
        "mass_kg": 100.0,
        "aero": {
            "drag_area_m2": 2.0,
            "cd": 2.2,
            "nose_radius_m": 0.25,
            "lift_area_m2": 5.0,
            "cl": 0.5,
        },
    }
    cfg = scenario_config_from_dict(raw)
    payload = _SingleRunEngine(cfg).run()
    metrics = payload["reentry_metrics_by_object"]["capsule"]

    assert np.nanmax(np.array(metrics["heat_rate_w_m2"], dtype=float)) > 0.0
    assert np.nanmax(np.array(metrics["lift_accel_m_s2"], dtype=float)) > 0.0


def test_reentry_metrics_remain_inactive_above_threshold() -> None:
    cfg = ReentryConfig(enabled=True, begin_altitude_km=100.0, atmosphere_model="ussa1976")
    out = reentry_metrics_for_state(
        r_eci_km=np.array([6500.0, 0.0, 0.0]),
        v_eci_km_s=np.array([0.0, 7.8, 0.0]),
        t_s=0.0,
        dt_s=1.0,
        cfg=cfg,
        props=ReentryObjectProperties(mass_kg=100.0, drag_area_m2=2.0, cd=2.2, nose_radius_m=0.5),
        env={},
        active=False,
    )
    assert out["active"] == 0.0
    assert np.isfinite(out["altitude_km"])
    assert np.isnan(out["heat_rate_w_m2"])


def test_reentry_current_active_can_exit_while_heat_load_is_preserved() -> None:
    cfg = ReentryConfig(enabled=True, begin_altitude_km=300.0, atmosphere_model="ussa1976")
    props = ReentryObjectProperties(mass_kg=100.0, drag_area_m2=2.0, cd=2.2, nose_radius_m=0.5)
    entered = reentry_metrics_for_state(
        r_eci_km=np.array([6378.137 + 250.0, 0.0, 0.0]),
        v_eci_km_s=np.array([0.0, 7.8, 0.0]),
        t_s=0.0,
        dt_s=1.0,
        cfg=cfg,
        props=props,
        env={},
        active=True,
    )
    exited = reentry_metrics_for_state(
        r_eci_km=np.array([6378.137 + 350.0, 0.0, 0.0]),
        v_eci_km_s=np.array([0.0, 7.8, 0.0]),
        t_s=1.0,
        dt_s=1.0,
        cfg=cfg,
        props=props,
        env={},
        active=False,
        previous_heat_load_j_m2=entered["heat_load_j_m2"],
    )

    assert entered["active"] == 1.0
    assert exited["active"] == 0.0
    assert exited["heat_load_j_m2"] == entered["heat_load_j_m2"]


def test_object_lift_axis_can_steer_atmospheric_pass(tmp_path: Path) -> None:
    raw = _reentry_config(tmp_path)
    raw["objects"]["capsule"]["specs"].update(
        {
            "cd": 0.0,
            "cl": 1.0,
            "lift_area_m2": 10.0,
            "lift_axis_body": [0.0, 0.0, 1.0],
        }
    )
    raw["simulator"]["duration_s"] = 1.0
    raw["simulator"]["dynamics"]["orbit"]["drag"] = True
    raw["simulator"]["environment"] = {"density_kg_m3": 1.0e-6}
    cfg = scenario_config_from_dict(raw)
    payload = _SingleRunEngine(cfg).run()
    truth = np.array(payload["truth_by_object"]["capsule"], dtype=float)

    assert truth[-1, 5] > 0.0


def test_aero_assisted_plane_change_demo_runs_and_exits_reentry(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "aero_assisted_plane_change_demo.yaml"
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["outputs"]["output_dir"] = str(tmp_path)
    raw["outputs"]["plots"]["enabled"] = False
    raw["outputs"]["stats"]["print_summary"] = False
    cfg = scenario_config_from_dict(raw)
    assert validate_scenario_plugins(cfg) == []

    payload = _SingleRunEngine(cfg).run()
    summary = payload["summary"]
    reentry = summary["reentry_summary_by_object"]["chaser"]
    truth = np.array(payload["truth_by_object"]["chaser"], dtype=float)
    alt = np.linalg.norm(truth[:, :3], axis=1) - 6378.137

    assert summary["terminated_early"] is False
    assert reentry["entered_reentry"] is True
    assert reentry["currently_in_reentry"] is False
    assert reentry["latest_exit_time_s"] is not None
    assert reentry["peak_lift_accel_m_s2"] > 0.0
    assert alt[-1] > 350.0
    assert truth[-1, 2] > 25.0
