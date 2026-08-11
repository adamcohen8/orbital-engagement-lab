from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import run_simulation
from sim.execution import run_simulation_config_file


def test_cli_native_math_thread_defaults_preserve_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in run_simulation._NATIVE_MATH_THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")

    run_simulation._configure_cli_native_math_threads()

    assert run_simulation.os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert all(
        run_simulation.os.environ[name] == "1"
        for name in run_simulation._NATIVE_MATH_THREAD_ENV_VARS
        if name != "OPENBLAS_NUM_THREADS"
    )


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
    assert config["simulator"]["duration_s"] == pytest.approx(300.0)
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
    assert "python -m sim.review" in index_text
    assert ".venv/bin/python" not in index_text
    assert "docs/installation.md" in index_text
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


def test_doctor_write_probe_does_not_follow_legacy_probe_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("preserve me\n", encoding="utf-8")
    legacy_probe = output_root / ".doctor_write_test"
    legacy_probe.symlink_to(victim)
    monkeypatch.chdir(tmp_path)

    run_simulation._print_doctor_report()

    assert victim.read_text(encoding="utf-8") == "preserve me\n"
    assert legacy_probe.is_symlink()
    assert not list(output_root.glob(".doctor_write_test-*"))


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
    config = yaml.safe_load((root / "configs" / "ric_pd_10km_experiment.yaml").read_text(encoding="utf-8"))
    chaser = config["objects"]["chaser"]
    params = chaser["flight_software"]["params"]

    assert chaser["flight_software"]["stack"] == "fsw.rpo_reference"
    assert params["translation_mode"] == "ric_pd_transfer"
    assert params["goal_type"] == "rpo.ric_pd_transfer"
    assert params["goal_mode"] == "maintenance"
    assert "orbit_control" not in chaser
    assert "attitude_control" not in chaser

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
    assert "Functional Python" in proc.stdout
    assert "Security baseline" in proc.stdout
    assert "Operating system" in proc.stdout
    assert "Python executable" in proc.stdout
    assert "OEL version" in proc.stdout
    assert "Install profile" in proc.stdout
    assert "Trainer" in proc.stdout
    assert "Quickstart validation" in proc.stdout
    assert "Recovery commands" in proc.stdout
    assert proc.returncode == 0


def test_doctor_fails_when_installed_oel_metadata_is_stale(monkeypatch, capsys) -> None:
    from sim.project_version import ProjectVersionStatus

    monkeypatch.setattr(
        "sim.doctor.inspect_project_version",
        lambda **_kwargs: ProjectVersionStatus(
            source_version="0.22.2",
            installed_version="0.21.1",
            ok=False,
            required=True,
            detail="source 0.22.2; installed 0.21.1",
        ),
    )

    assert not run_simulation._print_doctor_report(source_root=Path(__file__).resolve().parents[2])
    output = capsys.readouterr().out
    assert "OEL version" in output
    assert "FAIL - source 0.22.2; installed 0.21.1" in output


def test_open_output_folder_uses_platform_opener(tmp_path: Path) -> None:
    with patch("run_simulation.open_folder") as opener:
        assert run_simulation._open_output_folder(tmp_path)

    opener.assert_called_once_with(tmp_path)
