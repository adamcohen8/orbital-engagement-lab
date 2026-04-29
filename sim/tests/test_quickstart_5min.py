from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import yaml

import run_simulation
from sim.master_simulator import run_master_simulation


def test_quickstart_5min_runs_headlessly_and_writes_start_here_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source_cfg = root / "configs" / "quickstart_5min.yaml"
    config = yaml.safe_load(source_cfg.read_text(encoding="utf-8"))
    outdir = tmp_path / "quickstart_5min"
    config["outputs"]["output_dir"] = str(outdir)

    cfg_path = tmp_path / "quickstart_5min.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_master_simulation(cfg_path)

    assert result["scenario_name"] == "quickstart_5min"
    assert (outdir / "index.md").is_file()
    assert (outdir / "master_run_summary.json").is_file()
    assert not any(outdir.glob("*.png"))

    index_text = (outdir / "index.md").read_text(encoding="utf-8")
    assert "Open `master_run_summary.json`" in index_text
    assert "Inspect generated plot or animation artifacts listed below." not in index_text

    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    assert summary["scenario_name"] == "quickstart_5min"
    assert summary["objects"] == ["chaser", "target"]


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


def test_doctor_reports_quickstart_readiness() -> None:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "run_simulation.py", "--doctor"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "ORBITAL ENGAGEMENT LAB DOCTOR" in proc.stdout
    assert "Quickstart validation" in proc.stdout


def test_open_output_folder_uses_platform_opener(tmp_path: Path) -> None:
    with patch("run_simulation.subprocess.Popen") as popen:
        assert run_simulation._open_output_folder(tmp_path)

    popen.assert_called_once()
    assert str(tmp_path) in popen.call_args.args[0]
