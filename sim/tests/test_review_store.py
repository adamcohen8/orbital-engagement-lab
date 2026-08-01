from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import sim.reporting.review_store as review_store_module
from sim import ReviewWorkspace as TopLevelReviewWorkspace
from sim import SimulationConfig, SimulationSession
from sim.config import scenario_config_from_dict
from sim.execution import run_simulation_config_file
from sim.reporting.review_store import (
    REVIEW_SCHEMA_COMPATIBILITY_POLICY,
    REVIEW_SCHEMA_STABLE_TABLES,
    REVIEW_SCHEMA_VERSION,
)
from sim.review import (
    EVIDENCE_PLOT_RECIPES,
    SAVED_QUERY_MATURITY_LEVELS,
    SAVED_REVIEW_QUERIES,
    EvidencePlotter,
    ReviewPlotSpec,
    ReviewQueryError,
    ReviewStoreNotFoundError,
    ReviewWorkspace,
    SavedReviewQuery,
    load_workflow_manifest,
    numeric_columns,
    save_review_plot,
    write_workflow_review,
)

ISS_LINE1 = "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003"
ISS_LINE2 = "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004"


def _review_store_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "review_store_smoke",
        "scenario_description": "Review store smoke test",
        "objects": {
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
    }


def test_single_run_review_store_writes_queryable_sqlite(tmp_path: Path) -> None:
    result = SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    review_outputs = dict(result.summary.get("review_outputs", {}) or {})
    db_path = Path(review_outputs["sqlite"])
    schema_path = Path(review_outputs["schema_json"])

    assert db_path.is_file()
    assert schema_path.is_file()
    assert db_path.parent == tmp_path / "review"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["schema_version"] == REVIEW_SCHEMA_VERSION
    assert schema["compatibility"]["policy"] == REVIEW_SCHEMA_COMPATIBILITY_POLICY
    assert schema["compatibility"]["breaking_change_requires_schema_version_bump"] is True
    assert tuple(schema["compatibility"]["stable_tables"]) == REVIEW_SCHEMA_STABLE_TABLES
    assert set(REVIEW_SCHEMA_STABLE_TABLES).issubset(schema["tables"])

    with sqlite3.connect(db_path) as conn:
        scenario_name, config_sha256, config_json = conn.execute(
            "SELECT scenario_name, config_sha256, config_json FROM run_metadata"
        ).fetchone()
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        object_count = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
        sample_count = conn.execute("SELECT COUNT(*) FROM time_samples").fetchone()[0]
        state_count = conn.execute("SELECT COUNT(*) FROM object_state").fetchone()[0]
        relative_count = conn.execute("SELECT COUNT(*) FROM relative_state").fetchone()[0]
        min_range = conn.execute("SELECT MIN(range_km) FROM relative_state").fetchone()[0]
        artifact_paths = [row[0] for row in conn.execute("SELECT path FROM artifacts ORDER BY artifact_id")]

    assert scenario_name == "review_store_smoke"
    assert set(REVIEW_SCHEMA_STABLE_TABLES).issubset(table_names)
    assert config_sha256 == hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    assert json.loads(config_json)["objects"]["target"]["enabled"] is True
    assert object_count == 2
    assert sample_count == 3
    assert state_count == 6
    assert relative_count == 3
    assert min_range == pytest.approx(result.min_range("chaser", "target"))
    assert {
        "index.md",
        "master_run_log.json",
        "master_run_summary.json",
        "review/run.sqlite",
        "review/schema.json",
    }.issubset(set(artifact_paths))


def test_review_store_closes_sqlite_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_connect = sqlite3.connect
    real_replace = Path.replace
    state = {"closed": False, "replace_observed": False}

    class TrackedConnection:
        def __init__(self, path: Path) -> None:
            self.connection = real_connect(path)

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

        def close(self) -> None:
            self.connection.close()
            state["closed"] = True

    def tracked_connect(path: Path) -> TrackedConnection:
        return TrackedConnection(path)

    def checked_replace(path: Path, target: Path) -> Path:
        if path.name == "run.sqlite.tmp":
            assert state["closed"], "Windows cannot replace an open SQLite file"
            state["replace_observed"] = True
        return real_replace(path, target)

    monkeypatch.setattr(review_store_module.sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(Path, "replace", checked_replace)

    result = SimulationSession.from_config(
        SimulationConfig.from_dict(_review_store_config(tmp_path))
    ).run()

    assert state["replace_observed"]
    assert Path(result.summary["review_outputs"]["sqlite"]).is_file()


def test_review_store_supports_workspace_paths_with_spaces(tmp_path: Path) -> None:
    output_dir = tmp_path / "Orbital Engagement Lab" / "Windows Review Output"
    SimulationSession.from_config(
        SimulationConfig.from_dict(_review_store_config(output_dir))
    ).run()

    workspace = ReviewWorkspace.open(output_dir)
    result = workspace.query("SELECT scenario_name FROM run_metadata")

    assert result.rows == [{"scenario_name": "review_store_smoke"}]
    with pytest.raises(ReviewQueryError, match="read-only"):
        workspace.query("UPDATE run_metadata SET scenario_name = 'changed'")


def test_single_run_review_store_writes_all_configured_relative_pairs(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["scenario_name"] = "review_store_multi_pair"
    raw["simulator"]["duration_s"] = 1.0
    raw["objects"]["chaser_a"] = raw["objects"].pop("chaser")
    raw["objects"]["chaser_b"] = {
        "enabled": True,
        "specs": {"mass_kg": 100.0},
        "initial_state": {
            "relative_to": "target",
            "relative_ric_rect": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }

    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    db_path = Path(result.summary["review_outputs"]["sqlite"])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT deputy_id, chief_id, COUNT(*)
            FROM relative_state
            GROUP BY deputy_id, chief_id
            ORDER BY deputy_id, chief_id
            """
        ).fetchall()
        final_rows = conn.execute(SAVED_REVIEW_QUERIES["relative_final_state"].sql).fetchall()

    assert rows == [
        ("chaser_a", "target", 2),
        ("chaser_b", "target", 2),
    ]
    assert [(row[1], row[2]) for row in final_rows] == [
        ("chaser_a", "target"),
        ("chaser_b", "target"),
    ]


def test_saved_review_queries_have_machine_readable_contract() -> None:
    assert SAVED_REVIEW_QUERIES
    for key, query in SAVED_REVIEW_QUERIES.items():
        assert key == query.name
        assert query.description
        assert query.sql.lstrip().upper().startswith(("SELECT", "WITH"))
        assert query.source_tables, key
        assert query.maturity in SAVED_QUERY_MATURITY_LEVELS
        assert query.max_vm_steps > 0

    assert SAVED_REVIEW_QUERIES["burn_events"].allow_empty is True
    try:
        SavedReviewQuery(name="bad", description="bad", sql="DELETE FROM metrics")
    except ValueError as exc:
        assert "read-only SELECT/WITH" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("SavedReviewQuery accepted mutating SQL")


def test_flagship_scale_burn_activity_uses_vetted_saved_query_budget(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    db_path = review_dir / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE thrust (
                sample_index INTEGER,
                time_s REAL,
                object_id TEXT,
                accel_x_eci_km_s2 REAL,
                accel_y_eci_km_s2 REAL,
                accel_z_eci_km_s2 REAL,
                accel_norm_km_s2 REAL,
                burn_active INTEGER
            );
            CREATE INDEX idx_thrust_object_time ON thrust(object_id, time_s);
            """
        )
        rows = (
            (index, float(index), object_id, 0.0, 0.0, 0.0, 1.0e-6 if object_id == "chaser" else 0.0, int(object_id == "chaser"))
            for object_id in ("chaser", "target")
            for index in range(12_001)
        )
        conn.executemany("INSERT INTO thrust VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    workspace = ReviewWorkspace.open(tmp_path)
    saved = SAVED_REVIEW_QUERIES["burn_activity"]
    with pytest.raises(ReviewQueryError, match="step budget"):
        workspace.query(saved.sql)

    result = workspace.query(saved.sql, max_vm_steps=saved.max_vm_steps)

    assert result.row_count == 2
    assert result.rows[0]["object_id"] == "chaser"
    assert result.rows[0]["samples"] == 12_001


def test_relative_evidence_plot_recipes_group_by_pair_id() -> None:
    for recipe_id in ("relative_range", "relative_range_rate"):
        recipe = EVIDENCE_PLOT_RECIPES[recipe_id]
        assert recipe.group_column == "pair_id"
        assert "pair_id" in recipe.sql


def test_saved_review_query_source_tables_match_review_schema(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)
    available_tables = set(workspace.tables())
    single_run_queries = [
        "run_metadata",
        "objects",
        "artifacts",
        "passive_final_state",
        "rendezvous_metrics",
        "rendezvous_closest_approach",
        "relative_final_state",
        "burn_activity",
        "burn_events",
        "attitude_rates_first_last",
        "attitude_state_first_last",
    ]

    for query_name in single_run_queries:
        query = SAVED_REVIEW_QUERIES[query_name]
        assert set(query.source_tables).issubset(available_tables), query_name


def test_review_store_writes_object_propagation_metadata_for_sgp4(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["objects"]["chaser"]["enabled"] = False
    raw["objects"]["target"]["propagation_method"] = "general"
    raw["objects"]["target"]["general"] = {"model": "sgp4", "output_frame": "eci", "frame_transform": "teme_as_eci"}
    raw["objects"]["target"]["initial_state"] = {"tle": {"line1": ISS_LINE1, "line2": ISS_LINE2}}
    raw["simulator"]["initial_jd_utc"] = 2460310.5
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    db_path = Path(dict(result.summary.get("review_outputs", {}) or {})["sqlite"])
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT object_id, propagation_method, general_model, native_frame, output_frame, frame_transform
            FROM object_propagation
            """
        ).fetchone()

    assert row == ("target", "general", "sgp4", "teme", "eci", "teme_as_eci")


def test_review_store_writes_frame_provenance(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["simulator"]["frames"] = {"model": "simple_gmst"}
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    db_path = Path(dict(result.summary.get("review_outputs", {}) or {})["sqlite"])
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT scope, model, legacy_frame_model, time_scale_model,
                   polar_motion_applied, nutation_corrections_applied
            FROM frame_provenance
            """
        ).fetchone()

    assert row == ("scenario", "simple_gmst", "simple", "utc_only", 0, 0)
    assert result.payload["frame_provenance"]["model"] == "simple_gmst"


def test_review_store_writes_tle_initialization_metadata_for_special_propagation(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["objects"]["chaser"]["enabled"] = False
    raw["objects"]["target"]["initial_state"] = {"tle": {"line1": ISS_LINE1, "line2": ISS_LINE2}}
    raw["simulator"]["initial_jd_utc"] = 2460310.75
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    db_path = Path(dict(result.summary.get("review_outputs", {}) or {})["sqlite"])
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT object_id, source, initialization_model, initialization_propagator_name,
                   handoff_propagation_method, native_frame, output_frame, frame_transform,
                   tle_age_initialization_days
            FROM object_initialization
            """
        ).fetchone()

    assert row[:8] == (
        "target",
        "tle",
        "ogp",
        "OGP-SGP4",
        "special",
        "teme",
        "eci",
        "teme_to_eci_iau80",
    )
    assert row[8] == pytest.approx(0.25)


def test_review_store_records_canonical_eci_state_frame_for_sgp4(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["objects"]["chaser"]["enabled"] = False
    raw["objects"]["target"]["propagation_method"] = "general"
    raw["objects"]["target"]["general"] = {"model": "sgp4", "output_frame": "teme"}
    raw["objects"]["target"]["initial_state"] = {"tle": {"line1": ISS_LINE1, "line2": ISS_LINE2}}
    raw["simulator"]["initial_jd_utc"] = 2460310.5
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    assert result.payload["object_state_frames"]["target"] == "eci"
    db_path = Path(dict(result.summary.get("review_outputs", {}) or {})["sqlite"])
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT p.output_frame, p.frame_transform, f.state_frame
            FROM object_propagation p
            JOIN object_state_frame f USING (object_id)
            """
        ).fetchone()

    assert row == ("teme", "native", "eci")


def test_review_store_config_defaults_disabled_and_validates_detail(tmp_path: Path) -> None:
    cfg = scenario_config_from_dict(_review_store_config(tmp_path))

    assert cfg.outputs.review.enabled is True
    assert cfg.outputs.review.detail == "standard"
    assert cfg.outputs.review.strict is False

    raw = _review_store_config(tmp_path)
    raw["outputs"]["review"]["detail"] = "dense"
    with pytest.raises(ValueError, match="outputs.review.detail"):
        scenario_config_from_dict(raw)


def test_review_store_writes_mission_recovery_tables(tmp_path: Path) -> None:
    raw = _review_store_config(tmp_path)
    raw["objects"]["target"]["specs"]["isp_s"] = 220.0
    raw["analysis"] = {
        "mission_recovery": {
            "enabled": True,
            "object_id": "target",
            "goal": "orbit_shape",
            "element_tolerances": {"a_km": 1.0, "ecc": 0.01},
            "planner": {
                "enabled": True,
                "modes": ["min_delta_v", "min_time", "constrained"],
                "candidate_count": 4,
                "max_recovery_time_s": 7200.0,
                "max_recovery_delta_v_m_s": 20.0,
            },
        }
    }
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    db_path = Path(result.summary["review_outputs"]["sqlite"])

    with sqlite3.connect(db_path) as conn:
        summary_rows = conn.execute(
            "SELECT object_id, goal, method, recovery_delta_v_m_s FROM mission_recovery_summary"
        ).fetchall()
        element_rows = conn.execute(
            "SELECT state_label, a_km, ecc FROM mission_recovery_elements ORDER BY state_label"
        ).fetchall()
        metric_rows = conn.execute(
            "SELECT metric_name, units FROM metrics WHERE metric_name = 'recovery_delta_v_m_s'"
        ).fetchall()
        candidate_rows = conn.execute(
            "SELECT candidate_id, source, source_family, target_basis, feasible, verified "
            "FROM mission_recovery_candidates"
        ).fetchall()
        burn_rows = conn.execute(
            "SELECT candidate_id, burn_index, delta_v_m_s FROM mission_recovery_burns"
        ).fetchall()
        candidate_element_rows = conn.execute(
            "SELECT candidate_id, a_km, ecc FROM mission_recovery_candidate_elements"
        ).fetchall()

    assert len(summary_rows) == 1
    assert summary_rows[0][0] == "target"
    assert summary_rows[0][1] == "orbit_shape"
    assert summary_rows[0][2] in {"sim_state_inferred_intrack_impulse", "local_orbit_shape_velocity_match"}
    assert summary_rows[0][3] is not None
    assert [row[0] for row in element_rows] == ["final", "initial", "target"]
    assert metric_rows == [("recovery_delta_v_m_s", "m/s")]
    assert candidate_rows
    assert burn_rows
    assert candidate_element_rows
    assert all(row[2] == "analytic_reconstitution" for row in candidate_rows)
    assert all(row[3] == "initial_orbit" for row in candidate_rows)
    assert all(row[4] in (0, 1) for row in candidate_rows)
    assert all(row[5] in (0, 1) for row in candidate_rows)


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


def test_review_workspace_rejects_expensive_queries(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)

    with pytest.raises(ReviewQueryError, match="step budget"):
        workspace.query(
            "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt LIMIT 1000000) SELECT max(x) FROM cnt",
            max_vm_steps=1,
        )


def test_review_plot_creator_saves_styled_figure_with_provenance(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)
    result = workspace.query("SELECT time_s, range_km FROM relative_state ORDER BY time_s")

    assert numeric_columns(result) == ["time_s", "range_km"]

    artifact = save_review_plot(
        workspace,
        ReviewPlotSpec(
            sql="SELECT time_s, range_km FROM relative_state ORDER BY time_s",
            x_column="time_s",
            y_columns=["range_km"],
            plot_type="line",
            style_name="oel_light",
            title="Range Over Time",
            x_label="Time (s)",
            y_label="Range (km)",
            artifact_id="range_over_time",
        ),
    )

    assert artifact.path.is_file()
    assert artifact.relative_path == "review/figures/range_over_time.png"

    manifest = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    row = manifest["artifacts"][-1]
    assert row["artifact_id"] == "range_over_time"
    assert row["source"] == "oel_review_plot_api"
    assert row["plot_type"] == "line"
    assert row["style_name"] == "oel_light"
    assert row["x_column"] == "time_s"
    assert row["y_columns"] == ["range_km"]
    assert "FROM relative_state" in row["source_query"]


def test_evidence_plotter_api_creates_multi_series_recipe_and_custom_plots(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)
    plotter = EvidencePlotter(workspace)

    range_artifact = plotter.line(
        sql="SELECT time_s, range_km, range_rate_km_s FROM relative_state ORDER BY time_s",
        x="time_s",
        y=["range_km", "range_rate_km_s"],
        title="Range and rate",
        y_label="Value",
        artifact_id="api_range_and_rate",
        style="light",
    )

    assert range_artifact.relative_path == "review/figures/api_range_and_rate.png"
    assert range_artifact.path.is_file()

    recipe_artifact = plotter.relative_range_rate(artifact_id="api_range_rate_recipe")
    assert recipe_artifact.relative_path == "review/figures/api_range_rate_recipe.png"

    manifest = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][-2]["source"] == "oel_review_plot_api"
    assert manifest["artifacts"][-2]["style_name"] == "oel_light"
    assert manifest["artifacts"][-2]["y_columns"] == ["range_km", "range_rate_km_s"]
    assert manifest["artifacts"][-1]["extra"]["recipe_id"] == "relative_range_rate"


def test_evidence_plotter_histogram_heatmap_and_helpful_errors(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    with sqlite3.connect(review_dir / "run.sqlite") as conn:
        conn.execute(
            "CREATE TABLE sample_grid (region TEXT, bucket TEXT, value REAL, label TEXT)"
        )
        conn.executemany(
            "INSERT INTO sample_grid VALUES (?, ?, ?, ?)",
            [
                ("A", "early", 1.0, "one"),
                ("A", "late", 2.0, "two"),
                ("B", "early", 3.0, "three"),
                ("B", "late", 4.0, "four"),
            ],
        )

    plotter = EvidencePlotter(tmp_path)
    hist = plotter.histogram(
        sql="SELECT value FROM sample_grid",
        y="value",
        artifact_id="api_value_hist",
    )
    heatmap = plotter.heatmap(
        sql="SELECT bucket, region, value FROM sample_grid ORDER BY region, bucket",
        x="bucket",
        y="region",
        value="value",
        artifact_id="api_value_heatmap",
    )

    assert hist.path.is_file()
    assert heatmap.path.is_file()
    with pytest.raises(ValueError, match="Available columns"):
        plotter.line(sql="SELECT value FROM sample_grid", x="time_s", y="value")
    with pytest.raises(ValueError, match="Numeric columns"):
        plotter.line(sql="SELECT bucket, label FROM sample_grid", x="bucket", y="label")


def test_review_plot_preview_can_skip_provenance_manifest(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()
    workspace = ReviewWorkspace.open(tmp_path)
    preview_path = tmp_path / "preview.png"

    artifact = save_review_plot(
        workspace,
        ReviewPlotSpec(
            sql="SELECT time_s, range_km FROM relative_state ORDER BY time_s",
            x_column="time_s",
            y_columns=["range_km"],
            plot_type="scatter",
            style_name="oel_dark",
        ),
        path=preview_path,
        record=False,
    )

    assert artifact.path == preview_path
    assert preview_path.is_file()
    assert not (tmp_path / "review" / "generated_artifacts.json").exists()


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


def test_review_plot_cli_dry_run_and_creates_artifact(tmp_path: Path) -> None:
    SimulationSession.from_config(SimulationConfig.from_dict(_review_store_config(tmp_path))).run()

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review.plot",
            str(tmp_path),
            "--sql",
            "SELECT time_s, range_km FROM relative_state ORDER BY time_s",
            "--x",
            "time_s",
            "--y",
            "range_km",
            "--title",
            "CLI range",
            "--artifact-id",
            "cli_range",
            "--dry-run",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert dry_run.returncode == 0, dry_run.stderr
    dry_payload = json.loads(dry_run.stdout)
    assert dry_payload["spec"]["artifact_id"] == "cli_range"
    assert dry_payload["row_count"] == 3
    assert not (tmp_path / "review" / "figures" / "cli_range.png").exists()

    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review",
            "plot",
            str(tmp_path),
            "--recipe",
            "relative_range",
            "--artifact-id",
            "cli_recipe_range",
            "--style",
            "light",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    assert payload["relative_path"] == "review/figures/cli_recipe_range.png"
    assert Path(payload["path"]).is_file()
    manifest = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    row = manifest["artifacts"][-1]
    assert row["source"] == "oel_review_plot_cli"
    assert row["extra"]["recipe_id"] == "relative_range"


def test_workflow_review_manifest_writes_queryable_tables_and_cli_summary(tmp_path: Path) -> None:
    outputs = write_workflow_review(
        output_dir=tmp_path,
        workflow_type="controller_bench",
        title="Demo Bench",
        scenario_name="demo_bench",
        status="complete",
        summary={"run_count": 2, "passed_runs": 1},
        artifacts={"summary_json": str(tmp_path / "controller_bench_summary.json")},
        recommended_queries=[
            {
                "name": "controller_bench_runs",
                "description": "Run rows.",
                "sql": "SELECT variant_name, case_name, passed FROM bench_runs",
            }
        ],
        recommended_review_order=["Query bench_runs."],
        source_config="bench.yaml",
        tables={
            "bench_runs": [
                {"variant_name": "a", "case_name": "nominal", "passed": True, "failure_count": 0},
                {"variant_name": "b", "case_name": "nominal", "passed": False, "failure_count": 1},
            ]
        },
    )

    assert Path(outputs["workflow_manifest_json"]).is_file()
    assert Path(outputs["sqlite"]).is_file()
    assert load_workflow_manifest(tmp_path)["workflow_type"] == "controller_bench"

    workspace = ReviewWorkspace.open(tmp_path)
    assert "bench_runs" in workspace.tables()
    query = workspace.query("SELECT variant_name, passed FROM bench_runs ORDER BY variant_name")
    assert query.rows == [{"variant_name": "a", "passed": 1}, {"variant_name": "b", "passed": 0}]

    proc = subprocess.run(
        [sys.executable, "-m", "sim.review", str(tmp_path), "--manifest"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "workflow_type: controller_bench" in proc.stdout


def test_workflow_review_manifest_removes_stale_store_when_tables_are_absent(tmp_path: Path) -> None:
    write_workflow_review(
        output_dir=tmp_path,
        workflow_type="validation",
        tables={
            "validation_benchmarks": [
                {
                    "benchmark_name": "old",
                    "kind": "smoke",
                    "passed": True,
                    "duration_s": 1.0,
                    "output_dir": "old",
                }
            ]
        },
        recommended_queries=[{"name": "old", "sql": "SELECT benchmark_name FROM validation_benchmarks"}],
    )

    outputs = write_workflow_review(
        output_dir=tmp_path,
        workflow_type="validation",
        artifacts={"summary_json": str(tmp_path / "summary.json")},
        tables={},
        recommended_queries=[],
    )
    manifest = load_workflow_manifest(tmp_path)

    assert "sqlite" not in outputs
    assert manifest["sqlite"] == ""
    assert manifest["schema_json"] == ""
    assert manifest["saved_views_json"] == ""
    assert not (tmp_path / "review" / "run.sqlite").exists()
    assert not (tmp_path / "review" / "schema.json").exists()
    assert not (tmp_path / "review" / "saved_views.json").exists()
    with pytest.raises(ReviewStoreNotFoundError):
        ReviewWorkspace.open(tmp_path)


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
        config_path = conn.execute("SELECT config_path FROM run_metadata").fetchone()[0]
    assert {
        "index.md",
        "master_run_summary.json",
        "review/run.sqlite",
        "review/schema.json",
    }.issubset(set(artifact_paths))
    assert config_path == str(cfg_path.resolve())


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
