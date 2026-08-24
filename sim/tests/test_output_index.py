from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from sim.reporting.output_index import _single_run_next_command, write_output_index


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
            "relative_range_summary": {
                "object_pair": ["chaser", "target"],
                "initial_range_km": 10.0,
                "closest_approach_km": 0.5,
                "closest_approach_time_s": 25.0,
                "final_range_km": 0.75,
            },
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
    assert "# Start Here" in text
    assert "## Run Status" in text
    assert "## What Happened" in text
    assert "Workflow: `single_run`" in text
    assert "Scenario: `demo`" in text
    assert "Closest approach: `0.5 km`" in text
    assert "Total delta-v: `1.25 m/s`" in text
    assert "Open [`run_dashboard.png`](run_dashboard.png) for the fastest visual overview." in text
    assert "Open [`master_run_summary.json`](master_run_summary.json)" in text
    assert "## Next Command" in text
    assert "## Evidence Provenance" in text
    assert "## Claim Limits And Success Criteria" in text
    assert "not, by itself, a mission-success verdict" in text
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

    assert "master_monte_carlo_summary.json" not in text
    assert "master_monte_carlo_commander_brief.md" not in text
    assert "Inspect campaign plots and AI report artifacts when present." in text


def test_write_output_index_does_not_recommend_missing_single_run_plots(tmp_path: Path) -> None:
    index_path = write_output_index(
        outdir=tmp_path,
        workflow="single_run",
        title="quickstart",
        summary={
            "scenario_name": "quickstart",
            "plot_outputs": {},
            "animation_outputs": {},
        },
        artifacts={"summary_json": str(tmp_path / "master_run_summary.json")},
    )

    text = index_path.read_text(encoding="utf-8")

    assert "Open [`master_run_summary.json`](master_run_summary.json)" in text
    assert "Inspect generated plot or animation artifacts listed below." not in text
    assert "No plot or animation artifacts were generated for this run" in text


def test_write_output_index_uses_review_query_as_next_command_when_review_exists(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / "run.sqlite").write_text("", encoding="utf-8")

    index_path = write_output_index(
        outdir=tmp_path,
        workflow="single_run",
        title="demo",
        summary={
            "scenario_name": "demo",
            "review_sqlite_path": str(review_dir / "run.sqlite"),
            "plot_outputs": {},
            "animation_outputs": {},
        },
        artifacts={"summary_json": str(tmp_path / "master_run_summary.json")},
    )

    text = index_path.read_text(encoding="utf-8")

    assert "python -m sim.review" in text
    assert ".venv/bin/python" not in text
    assert "docs/installation.md" in text
    assert "--saved-query object_final_state" in text


def test_single_run_next_command_quotes_windows_paths() -> None:
    with patch("sim.reporting.output_index.os.name", "nt"):
        command = _single_run_next_command({"config_source_path": r"C:\Users\Instructor\OEL Course\scenario.yaml"})

    assert command == (
        'python run_simulation.py --config "C:\\Users\\Instructor\\OEL Course\\scenario.yaml" --validate-only'
    )


def test_output_index_resolves_repo_relative_artifact_without_duplicating_output_dir(
    tmp_path: Path, monkeypatch
) -> None:
    outdir = tmp_path / "outputs" / "run"
    outdir.mkdir(parents=True)
    artifact = outdir / "ground_track.png"
    artifact.write_bytes(b"png")
    monkeypatch.chdir(tmp_path)

    index_path = write_output_index(
        outdir=outdir,
        workflow="single_run",
        title="paths",
        summary={"scenario_name": "paths"},
        artifacts={"plots": {"ground_track": "outputs/run/ground_track.png"}},
    )

    text = index_path.read_text(encoding="utf-8")
    assert "[`ground_track.png`](ground_track.png)" in text
    assert "outputs/run/outputs/run" not in text


def test_output_index_does_not_duplicate_output_prefix_for_pending_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    outdir = tmp_path / "outputs" / "run"
    outdir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    index_path = write_output_index(
        outdir=outdir,
        workflow="single_run",
        title="paths",
        summary={"scenario_name": "paths"},
        artifacts={"plots": {"ground_track": "outputs/run/pending-ground-track.png"}},
    )

    text = index_path.read_text(encoding="utf-8")
    assert "[`pending-ground-track.png`](pending-ground-track.png)" in text
    assert "outputs/run/outputs/run" not in text


def test_output_index_headline_closes_negative_access_and_link_results(tmp_path: Path) -> None:
    index_path = write_output_index(
        outdir=tmp_path,
        workflow="single_run",
        title="negative evidence",
        summary={
            "scenario_name": "negative-evidence",
            "ground_station_access_summary": {
                "station-a": {
                    "sat-a": {
                        "samples": 3,
                        "access_samples": 0,
                        "access_duration_s": 0.0,
                        "minimum_elevation_deg": 25.0,
                        "maximum_range_km": 1000.0,
                        "reason_sample_count": {"below_minimum_elevation": 3},
                    }
                }
            },
            "orbital_analysis": {
                "directed_links": [
                    {
                        "analysis_id": "link-a",
                        "sample_count": 3,
                        "available_sample_count": 0,
                        "interval_count": 0,
                        "sampled_available_fraction": 0.0,
                        "available_duration_s": 0.0,
                        "required_eb_n0_db": 6.0,
                        "minimum_fixed_site_elevation_deg": 25.0,
                        "maximum_range_km_threshold": 1000.0,
                        "primary_reason_sample_count": {"below_minimum_elevation": 3},
                    }
                ]
            },
        },
        artifacts={},
    )

    text = index_path.read_text(encoding="utf-8")
    assert "Ground-station accessible samples: `0/3`" in text
    assert "below_minimum_elevation=3" in text
    assert "available samples/windows: `0/3; windows=0`" in text
    assert "required Eb/N0=6 dB" in text
