from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

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
    assert "No default next command is defined for this workflow." in index_text
    assert "Inspect generated plot or animation artifacts listed below." not in index_text

    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    assert summary["scenario_name"] == "quickstart_5min"
    assert summary["objects"] == ["chaser", "target"]
    assert summary["relative_range_summary"]["object_pair"] == ["chaser", "target"]
    assert "rocket" not in summary["objects"]


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
