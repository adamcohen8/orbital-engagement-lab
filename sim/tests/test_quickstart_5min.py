from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

import run_simulation
from sim.execution import run_simulation_config_file


def _single_run_summary_text(summary: dict) -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        run_simulation._print_single_run_summary(
            {
                "config_path": "",
                "scenario_name": summary.get("scenario_name", "summary_display"),
                "run": summary,
            }
        )
    return stream.getvalue()


def test_quickstart_5min_runs_headlessly_and_writes_start_here_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source_cfg = root / "configs" / "quickstart_5min.yaml"
    config = yaml.safe_load(source_cfg.read_text(encoding="utf-8"))
    outdir = tmp_path / "quickstart_5min"
    config["outputs"]["output_dir"] = str(outdir)

    cfg_path = tmp_path / "quickstart_5min.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_simulation_config_file(cfg_path)

    assert result["scenario_name"] == "quickstart_5min"
    assert (outdir / "index.md").is_file()
    assert (outdir / "master_run_summary.json").is_file()
    assert not any(outdir.glob("*.png"))

    index_text = (outdir / "index.md").read_text(encoding="utf-8")
    assert "# Start Here" in index_text
    assert "## What Happened" in index_text
    assert "Open [`master_run_summary.json`](master_run_summary.json)" in index_text
    assert "Closest approach:" in index_text
    assert ".venv/bin/python -m sim.review" in index_text
    assert (outdir / "review" / "run.sqlite").is_file()
    assert "Inspect generated plot or animation artifacts listed below." not in index_text

    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    assert summary["scenario_name"] == "quickstart_5min"
    assert summary["objects"] == ["chaser", "target"]
    assert summary["relative_range_summary"]["object_pair"] == ["chaser", "target"]
    assert "rocket" not in summary["objects"]
    profile = summary["runtime_profile"]
    assert profile["completed_steps"] > 0
    assert profile["object_count"] == 2
    assert profile["total_step_wall_s"] >= 0.0
    assert profile["executor"]["object_step_backend"] == "serial"
    assert profile["executor"]["object_step_workers"] == 1
    assert profile["stage_totals"]["object_step"]["count"] > 0
    assert set(profile["object_totals"]) == {"chaser", "target"}
    assert profile["object_totals"]["chaser"]["total_s"] >= 0.0
    assert profile["object_totals"]["chaser"]["nested_stage_total_s"] >= 0.0
    assert "satellite_step" in profile["object_totals"]["chaser"]["stages"]
    assert profile["slowest_objects"]


def test_quickstart_process_pool_object_executor_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source_cfg = root / "configs" / "quickstart_5min.yaml"
    config = yaml.safe_load(source_cfg.read_text(encoding="utf-8"))
    config["objects"]["observer"] = {
        "enabled": True,
        "kind": "satellite",
        "specs": {
            "mass_kg": 100.0,
            "mass_properties": {
                "inertia_reference_point": "center_of_mass",
                "inertia_kg_m2": [
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 10.0],
                ]
            },
        },
        "initial_state": {
            "position_eci_km": [7010.0, 0.0, 0.0],
            "velocity_eci_km_s": [0.0, 7.49, 0.0],
            "angular_rate_body_rad_s": [2.0e6, 0.0, 0.0],
        },
    }
    config["simulator"]["duration_s"] = 10.0
    config["simulator"]["resource_profile"] = "off"
    config["simulator"]["dynamics"]["attitude"]["guardrail_policy"] = "sanitize"
    config["outputs"]["plots"]["enabled"] = False
    config["outputs"]["plots"]["figure_ids"] = []
    config["outputs"]["animations"]["enabled"] = False
    config["outputs"]["animations"]["types"] = []

    serial_config = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
    serial_outdir = tmp_path / "quickstart_serial"
    serial_config["simulator"]["execution"] = {
        "object_parallelism": {
            "enabled": False,
            "backend": "serial",
        }
    }
    serial_config["outputs"]["output_dir"] = str(serial_outdir)
    serial_cfg_path = tmp_path / "quickstart_serial.yaml"
    serial_cfg_path.write_text(yaml.safe_dump(serial_config, sort_keys=False), encoding="utf-8")
    run_simulation_config_file(serial_cfg_path)

    outdir = tmp_path / "quickstart_process_pool"
    config["simulator"]["execution"] = {
        "object_parallelism": {
            "enabled": True,
            "backend": "process_pool",
            "workers": 2,
            "min_objects": 3,
        }
    }
    config["outputs"]["output_dir"] = str(outdir)

    cfg_path = tmp_path / "quickstart_process_pool.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    try:
        run_simulation_config_file(cfg_path)
    except RuntimeError as exc:
        if "ProcessPoolObjectStepExecutor is unavailable" in str(exc):
            pytest.skip(str(exc))
        raise

    serial_summary = json.loads((serial_outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    profile = summary["runtime_profile"]
    assert profile["executor"]["object_step_backend"] == "process_pool"
    assert profile["executor"]["object_step_workers"] == 2
    assert profile["stage_totals"]["object_step"]["count"] > 0
    assert set(profile["object_totals"]) == {"chaser", "target", "observer"}
    assert summary["attitude_guardrail_stats"] == serial_summary["attitude_guardrail_stats"]
    assert summary["attitude_guardrail_stats"]["rate_clamp_events"] > 0
    state_columns = (
        "pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
        "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s"
    )
    serial_db = sqlite3.connect(serial_outdir / "review" / "run.sqlite")
    process_db = sqlite3.connect(outdir / "review" / "run.sqlite")
    try:
        for object_id in summary["objects"]:
            serial_state = np.array(
                serial_db.execute(
                    f"SELECT {state_columns} FROM object_state WHERE object_id = ? ORDER BY time_s DESC LIMIT 1",
                    (object_id,),
                ).fetchone(),
                dtype=float,
            )
            process_state = np.array(
                process_db.execute(
                    f"SELECT {state_columns} FROM object_state WHERE object_id = ? ORDER BY time_s DESC LIMIT 1",
                    (object_id,),
                ).fetchone(),
                dtype=float,
            )
            assert np.allclose(process_state, serial_state, rtol=0.0, atol=1e-9)
    finally:
        serial_db.close()
        process_db.close()


def test_quickstart_cli_shortcut_validates() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "run_simulation.py", "--quickstart", "--validate-only"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "quickstart_5min" in proc.stdout
    assert "OK" in proc.stdout


def test_flagship_ric_pd_config_validates() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "run_simulation.py", "--config", "configs/ric_pd_10km_experiment.yaml", "--validate-only"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "ric_pd_10km_experiment" in proc.stdout
    assert "OK" in proc.stdout


def test_flagship_analysis_script_writes_custom_metrics(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source_cfg = root / "configs" / "ric_pd_10km_experiment.yaml"
    config = yaml.safe_load(source_cfg.read_text(encoding="utf-8"))
    outdir = tmp_path / "flagship_short"
    config["outputs"]["output_dir"] = str(outdir)
    config["outputs"]["plots"]["enabled"] = False
    config["outputs"]["plots"]["figure_ids"] = []
    config["simulator"]["duration_s"] = 2.0

    cfg_path = tmp_path / "flagship_short.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "examples/python/flagship_analysis.py",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(outdir),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    metrics_path = outdir / "custom_analysis" / "flagship_metrics.json"
    csv_path = outdir / "custom_analysis" / "flagship_metrics.csv"
    assert metrics_path.is_file()
    assert csv_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["scenario_name"] == "ric_pd_10km_experiment"
    assert metrics["deputy"] == "chaser"
    assert metrics["chief"] == "target"
    assert metrics["samples"] > 0


def test_single_run_summary_only_reports_insertion_for_enabled_rocket() -> None:
    base_summary = {
        "scenario_name": "summary_display",
        "objects": ["target"],
        "samples": 1,
        "dt_s": 1.0,
        "duration_s": 0.0,
        "terminated_early": False,
        "rocket_insertion_achieved": False,
        "rocket_insertion_time_s": None,
    }

    assert "Insertion  :" not in _single_run_summary_text(base_summary)

    rocket_summary = dict(
        base_summary, objects=["rocket"], rocket_insertion_achieved=True, rocket_insertion_time_s=12.0
    )

    assert "Insertion  : achieved at t=12.0" in _single_run_summary_text(rocket_summary)


def test_doctor_reports_quickstart_readiness() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "run_simulation.py", "--doctor"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert "ORBITAL ENGAGEMENT LAB DOCTOR" in proc.stdout
    assert "Quickstart validation" in proc.stdout
    assert proc.returncode == 0


def test_open_output_folder_uses_platform_opener(tmp_path: Path) -> None:
    with patch("run_simulation.subprocess.Popen") as popen:
        assert run_simulation._open_output_folder(tmp_path)

    popen.assert_called_once()
    assert str(tmp_path) in popen.call_args.args[0]
