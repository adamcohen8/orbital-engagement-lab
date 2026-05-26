from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sim import ReviewWorkspace as TopLevelReviewWorkspace
from sim import SimulationConfig, SimulationSession
from sim.config import scenario_config_from_dict
from sim.execution import run_simulation_config_file
from sim.review import ReviewQueryError, ReviewWorkspace


def _review_store_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "review_store_smoke",
        "scenario_description": "Review store smoke test",
        "rocket": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        },
        "chaser": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "relative_to": "target",
                "relative_ric_rect": [1.0, 0.0, 0.0, -0.001, 0.0, 0.0],
            },
            "orbit_control": {
                "module": "sim.control.orbit.zero_controller",
                "class_name": "ZeroController",
            },
        },
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {
                "print_summary": False,
                "save_json": True,
                "save_full_log": True,
            },
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
            "review": {"enabled": True, "detail": "standard"},
        },
        "monte_carlo": {"enabled": False},
    }


def test_single_run_review_store_writes_queryable_sqlite(tmp_path: Path) -> None:
    result = SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    review_outputs = dict(result.summary.get("review_outputs", {}) or {})
    db_path = Path(review_outputs["sqlite"])
    schema_path = Path(review_outputs["schema_json"])

    assert db_path.is_file()
    assert schema_path.is_file()
    assert db_path.parent == tmp_path / "review"
    assert json.loads(schema_path.read_text(encoding="utf-8"))["schema_version"] == "0.1"

    with sqlite3.connect(db_path) as conn:
        scenario_name = conn.execute("SELECT scenario_name FROM run_metadata").fetchone()[0]
        object_count = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
        sample_count = conn.execute("SELECT COUNT(*) FROM time_samples").fetchone()[0]
        state_count = conn.execute("SELECT COUNT(*) FROM object_state").fetchone()[0]
        relative_count = conn.execute("SELECT COUNT(*) FROM relative_state").fetchone()[0]
        min_range = conn.execute("SELECT MIN(range_km) FROM relative_state").fetchone()[0]
        artifact_paths = [row[0] for row in conn.execute("SELECT path FROM artifacts ORDER BY artifact_id")]

    assert scenario_name == "review_store_smoke"
    assert object_count == 2
    assert sample_count == 3
    assert state_count == 6
    assert relative_count == 3
    assert min_range == pytest.approx(result.min_range("chaser", "target"))
    assert "master_run_summary.json" in artifact_paths
    assert "master_run_log.json" in artifact_paths


def test_review_store_config_defaults_disabled_and_validates_detail(tmp_path: Path) -> None:
    cfg = scenario_config_from_dict(_review_store_config(tmp_path))

    assert cfg.outputs.review.enabled is True
    assert cfg.outputs.review.detail == "standard"
    assert cfg.outputs.review.strict is False

    raw = _review_store_config(tmp_path)
    raw["outputs"]["review"]["detail"] = "dense"
    with pytest.raises(ValueError, match="outputs.review.detail"):
        scenario_config_from_dict(raw)


def test_review_workspace_query_api_allows_safe_selects(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    workspace = ReviewWorkspace.open(tmp_path)
    top_level_workspace = TopLevelReviewWorkspace.open(tmp_path / "review" / "run.sqlite")

    assert workspace.db_path == tmp_path / "review" / "run.sqlite"
    assert top_level_workspace.output_dir == tmp_path
    assert "relative_state" in workspace.tables()
    assert "object_state" in workspace.schema()["columns"]
    assert workspace.saved_views() == []

    result = workspace.query(
        """
        -- safe review query
        SELECT time_s, range_km
        FROM relative_state
        WHERE deputy_id = ?
        ORDER BY time_s;
        """,
        ("chaser",),
        max_rows=2,
    )

    assert result.columns == ["time_s", "range_km"]
    assert result.row_count == 2
    assert result.truncated is True
    assert result.rows[0]["time_s"] == 0.0

    aggregate = workspace.query(
        "WITH ranges AS (SELECT range_km FROM relative_state) SELECT MIN(range_km) AS min_range FROM ranges"
    )
    assert 0.9 < float(aggregate.rows[0]["min_range"]) < 1.0


def test_review_workspace_rejects_unsafe_queries(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)

    unsafe_queries = [
        "",
        "DELETE FROM metrics",
        "DROP TABLE metrics",
        "PRAGMA table_info(metrics)",
        "SELECT 1; SELECT 2",
        "WITH selected AS (SELECT 1) DELETE FROM metrics",
    ]

    for sql in unsafe_queries:
        with pytest.raises(ReviewQueryError):
            workspace.query(sql)


def test_review_cli_queries_output_folder(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review",
            str(tmp_path),
            "--query",
            "SELECT scenario_name FROM run_metadata",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["rows"][0]["scenario_name"] == "review_store_smoke"

    unsafe = subprocess.run(
        [sys.executable, "-m", "sim.review", str(tmp_path), "--query", "DELETE FROM metrics"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert unsafe.returncode == 2
    assert "must start with SELECT or WITH" in unsafe.stderr


def test_quickstart_config_can_emit_review_store_tables(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs" / "quickstart_5min.yaml").read_text(encoding="utf-8"))
    outdir = tmp_path / "quickstart_review"
    config["outputs"]["output_dir"] = str(outdir)
    config["outputs"]["stats"]["print_summary"] = False
    config["outputs"].setdefault("review", {})
    config["outputs"]["review"] = {"enabled": True, "detail": "standard"}

    cfg_path = tmp_path / "quickstart_review.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_simulation_config_file(cfg_path)

    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    db_path = outdir / "review" / "run.sqlite"
    assert db_path.is_file()
    assert Path(summary["review_outputs"]["sqlite"]) == db_path

    counts = _review_table_counts(db_path)
    assert counts["time_samples"] == summary["samples"]
    assert counts["objects"] == 2
    assert counts["object_state"] == summary["samples"] * 2
    assert counts["relative_state"] == summary["samples"]
    assert counts["thrust"] == summary["samples"] * 2
    assert counts["metrics"] >= 4
    assert counts["artifacts"] >= 1
    with sqlite3.connect(db_path) as conn:
        artifact_paths = [row[0] for row in conn.execute("SELECT path FROM artifacts ORDER BY artifact_id")]
    assert "master_run_summary.json" in artifact_paths


def test_plotting_config_review_store_indexes_plot_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs" / "plotting_rendezvous_demo.yaml").read_text(encoding="utf-8"))
    outdir = tmp_path / "plotting_review"
    config["outputs"]["output_dir"] = str(outdir)
    config["outputs"]["stats"]["print_summary"] = False
    config["simulator"]["duration_s"] = 5.0
    config["outputs"]["plots"]["enabled"] = True
    config["outputs"]["plots"]["preset"] = []
    config["outputs"]["plots"]["figure_ids"] = ["relative_range"]
    config["outputs"]["animations"]["enabled"] = False
    config["outputs"].setdefault("review", {})
    config["outputs"]["review"] = {"enabled": True, "detail": "standard"}

    cfg_path = tmp_path / "plotting_review.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_simulation_config_file(cfg_path)

    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    db_path = outdir / "review" / "run.sqlite"
    assert db_path.is_file()
    assert (outdir / "relative_ranges.png").is_file()
    assert "relative_ranges" in dict(summary.get("plot_outputs", {}) or {})

    with sqlite3.connect(db_path) as conn:
        scenario_name = conn.execute("SELECT scenario_name FROM run_metadata").fetchone()[0]
        artifact_rows = conn.execute(
            "SELECT artifact_id, artifact_type, path FROM artifacts ORDER BY artifact_id"
        ).fetchall()
        relative_rows = conn.execute("SELECT COUNT(*) FROM relative_state").fetchone()[0]
        ground_access_rows = conn.execute("SELECT COUNT(*) FROM ground_access").fetchone()[0]

    assert scenario_name == "plotting_rendezvous_demo"
    assert relative_rows == int(summary["samples"])
    assert ground_access_rows > 0
    assert ("plots:relative_ranges", "plots", "relative_ranges.png") in artifact_rows


def _review_table_counts(db_path: Path) -> dict[str, int]:
    tables = [
        "run_metadata",
        "objects",
        "time_samples",
        "object_state",
        "relative_state",
        "thrust",
        "ground_access",
        "events",
        "metrics",
        "artifacts",
    ]
    with sqlite3.connect(db_path) as conn:
        return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}
