from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from sim.config import scenario_config_from_dict
from sim.master_outputs import PLOT_PRESETS, plot_outputs
from sim.plotting import (
    plot_control_effort,
    plot_estimation_error,
    plot_estimation_error_components,
    plot_ground_track_from_payload,
    plot_rendezvous_summary,
    plot_run_dashboard,
    plot_sensor_access,
)


def _hist(pos: np.ndarray) -> np.ndarray:
    n = pos.shape[0]
    hist = np.zeros((n, 14), dtype=float)
    hist[:, :3] = pos
    hist[:, 3:6] = np.array([0.0, 7.5, 0.0], dtype=float)
    hist[:, 6] = 1.0
    hist[:, 10:13] = np.array([0.0, 0.0, 0.01], dtype=float)
    hist[:, 13] = 100.0
    return hist


def _payload() -> dict[str, object]:
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    target = _hist(
        np.array(
            [
                [7000.0, 0.0, 0.0],
                [7000.0, 7.5, 0.0],
                [6999.99, 15.0, 0.0],
                [6999.98, 22.5, 0.0],
            ],
            dtype=float,
        )
    )
    chaser = target.copy()
    chaser[:, 0] += np.array([1.0, 0.8, 0.5, 0.2], dtype=float)
    thrust = np.zeros((t.size, 3), dtype=float)
    thrust[:, 0] = [0.0, 1.0e-6, 1.0e-6, 0.0]
    belief = chaser.copy()
    belief[:, 0] += 0.01
    belief[:, 3] += 1.0e-4
    knowledge = chaser[:, :6].copy()
    knowledge[0, :] = np.nan
    knowledge[2, 0] += 0.02
    return {
        "summary": {"scenario_name": "plot_test"},
        "time_s": t.tolist(),
        "truth_by_object": {"target": target.tolist(), "chaser": chaser.tolist()},
        "belief_by_object": {"chaser": belief.tolist()},
        "applied_thrust_by_object": {"target": np.zeros_like(thrust).tolist(), "chaser": thrust.tolist()},
        "knowledge_by_observer": {"chaser": {"target": knowledge.tolist()}},
        "target_reference_orbit_truth": [],
    }


def test_payload_plotting_api_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _payload()
    outputs = {
        "dashboard": tmp_path / "dashboard.png",
        "rendezvous": tmp_path / "rendezvous.png",
        "control": tmp_path / "control.png",
        "estimation": tmp_path / "estimation.png",
        "estimation_components": tmp_path / "estimation_components.png",
        "sensor_access": tmp_path / "sensor_access.png",
        "ground": tmp_path / "ground.png",
    }

    plot_run_dashboard(payload, out_path=outputs["dashboard"], close=True)
    plot_rendezvous_summary(payload, out_path=outputs["rendezvous"], close=True)
    plot_control_effort(payload, out_path=outputs["control"], close=True)
    plot_estimation_error(payload, out_path=outputs["estimation"], close=True)
    plot_estimation_error_components(payload, out_path=outputs["estimation_components"], close=True)
    plot_sensor_access(payload, out_path=outputs["sensor_access"], close=True)
    plot_ground_track_from_payload(payload, out_path=outputs["ground"], close=True)

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_outputs_expands_public_plot_presets(tmp_path: Path) -> None:
    assert "run_dashboard" in PLOT_PRESETS["minimal"]
    cfg = scenario_config_from_dict(
        {
            "scenario_name": "plot_outputs_test",
            "target": {"enabled": True},
            "chaser": {"enabled": True},
            "simulator": {"duration_s": 3.0, "dt_s": 1.0},
            "outputs": {
                "output_dir": str(tmp_path),
                "mode": "save",
                "plots": {
                    "enabled": True,
                    "preset": "minimal",
                    "figure_ids": [
                        "rendezvous_summary",
                        "control_effort",
                        "estimation_error",
                        "estimation_error_components",
                        "sensor_access",
                        "ground_track_multi",
                    ],
                    "reference_object_id": "target",
                },
            },
            "monte_carlo": {"enabled": False},
        }
    )
    payload = _payload()
    t = np.array(payload["time_s"], dtype=float)
    truth = {k: np.array(v, dtype=float) for k, v in dict(payload["truth_by_object"]).items()}
    thrust = {k: np.array(v, dtype=float) for k, v in dict(payload["applied_thrust_by_object"]).items()}
    belief = {k: np.array(v, dtype=float) for k, v in dict(payload["belief_by_object"]).items()}
    knowledge = {
        obs: {tgt: np.array(arr, dtype=float) for tgt, arr in by_tgt.items()}
        for obs, by_tgt in dict(payload["knowledge_by_observer"]).items()
    }

    out = plot_outputs(
        cfg=cfg,
        t_s=t,
        truth_hist=truth,
        target_reference_orbit_truth=None,
        thrust_hist=thrust,
        desired_attitude_hist=None,
        knowledge_hist=knowledge,
        rocket_metrics=None,
        outdir=tmp_path,
        resolve_rocket_stack=lambda specs: None,
        resolve_satellite_isp_s=lambda specs: 0.0,
        belief_hist=belief,
    )

    assert set(out) >= {
        "run_dashboard",
        "rendezvous_summary",
        "control_effort",
        "estimation_error",
        "estimation_error_components",
        "sensor_access",
        "ground_track_multi",
    }
    for artifact in out.values():
        path = Path(artifact)
        assert path.exists()
        assert path.stat().st_size > 0
