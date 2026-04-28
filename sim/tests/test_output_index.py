from __future__ import annotations

from pathlib import Path

from sim.reporting.output_index import write_output_index


def test_write_output_index_creates_single_run_start_here_file(tmp_path: Path) -> None:
    summary_path = tmp_path / "master_run_summary.json"
    log_path = tmp_path / "master_run_log.json"
    summary_path.write_text("{}", encoding="utf-8")
    log_path.write_text("{}", encoding="utf-8")

    index_path = write_output_index(
        outdir=tmp_path,
        workflow="single_run",
        title="demo",
        summary={
            "scenario_name": "demo",
            "scenario_description": "Output index test.",
            "objects": ["target", "chaser"],
            "samples": 4,
            "duration_s": 30.0,
            "terminated_early": False,
            "thrust_stats": {"chaser": {"total_dv_m_s": 1.25}},
            "plot_outputs": {"run_dashboard": str(tmp_path / "run_dashboard.png")},
            "animation_outputs": {},
        },
        artifacts={
            "summary_json": str(summary_path),
            "run_log_json": str(log_path),
            "plots": {"run_dashboard": str(tmp_path / "run_dashboard.png")},
        },
    )

    text = index_path.read_text(encoding="utf-8")

    assert index_path == tmp_path / "index.md"
    assert "# Output Index" in text
    assert "Workflow: `single_run`" in text
    assert "Scenario: `demo`" in text
    assert "Total delta-v: `1.25 m/s`" in text
    assert "master_run_summary.json" in text
    assert "master_run_log.json" in text
    assert "plots.run_dashboard" in text


def test_write_output_index_renders_error_artifacts_as_literals(tmp_path: Path) -> None:
    index_path = write_output_index(
        outdir=tmp_path,
        workflow="sensitivity",
        title="demo",
        payload={
            "scenario_name": "demo",
            "analysis": {"method": "one_at_a_time", "run_count": 1},
        },
        artifacts={
            "summary_json": str(tmp_path / "master_analysis_sensitivity_summary.json"),
            "sensitivity_plot_error": "ImportError: numpy.core.multiarray failed to import",
        },
    )

    text = index_path.read_text(encoding="utf-8")

    assert "`sensitivity_plot_error`: `ImportError: numpy.core.multiarray failed to import`" in text
    assert "](<../../ImportError:" not in text


def test_write_output_index_open_first_uses_saved_artifacts_only(tmp_path: Path) -> None:
    index_path = write_output_index(
        outdir=tmp_path,
        workflow="monte_carlo",
        title="demo",
        payload={"scenario_name": "demo", "monte_carlo": {"iterations": 2}},
        artifacts={"ops_dashboard_png": str(tmp_path / "master_monte_carlo_ops_dashboard.png")},
    )

    text = index_path.read_text(encoding="utf-8")

    assert "Open `master_monte_carlo_summary.json`" not in text
    assert "Open `master_monte_carlo_commander_brief.md`" not in text
    assert "Inspect campaign plots and AI report artifacts when present." in text
