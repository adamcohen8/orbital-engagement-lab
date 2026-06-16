from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from sim.review.evidence_studio import (
    EVIDENCE_AGENT_WORKSPACE_DIRNAME,
    EVIDENCE_PLAN_SCHEMA_VERSION,
    EvidenceSelection,
    EvidenceStudioRequest,
    build_planner_task_packet,
    execute_evidence_plan,
    handle_evidence_studio_request,
    prepare_evidence_agent_workspace,
)


def _write_review_store(output_dir: Path) -> None:
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True)
    with sqlite3.connect(review_dir / "run.sqlite") as conn:
        conn.execute(
            "CREATE TABLE run_metadata (scenario_name TEXT, duration_s REAL, dt_s REAL, samples INTEGER, "
            "oel_version TEXT, review_schema_version TEXT)"
        )
        conn.execute("INSERT INTO run_metadata VALUES ('evidence_studio_smoke', 2.0, 1.0, 3, 'test', '0.3')")
        conn.execute(
            "CREATE TABLE relative_state (time_s REAL, deputy_id TEXT, chief_id TEXT, range_km REAL, "
            "v_radial_km_s REAL, v_intrack_km_s REAL, v_crosstrack_km_s REAL, range_rate_km_s REAL)"
        )
        conn.executemany(
            "INSERT INTO relative_state VALUES (?, 'chaser', 'target', ?, ?, ?, ?, ?)",
            [
                (0.0, 1.0, -0.10, 0.02, 0.00, -0.10),
                (1.0, 0.6, -0.05, 0.01, 0.00, -0.05),
                (2.0, 0.25, -0.01, 0.00, 0.00, -0.01),
            ],
        )


def test_evidence_studio_generates_styled_plot_with_instruction_provenance(tmp_path: Path) -> None:
    _write_review_store(tmp_path)

    result = handle_evidence_studio_request(
        EvidenceStudioRequest(
            output_dir=tmp_path,
            instruction="Plot relative range over time for the brief.",
            style_name="oel_light",
            file_format="svg",
        )
    )

    assert result.ok
    assert result.artifact is not None
    assert result.artifact.relative_path == "review/figures/evidence_relative_range.svg"
    assert result.artifact.path.is_file()

    manifest = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    row = manifest["artifacts"][-1]
    assert row["style_name"] == "oel_light"
    assert row["extra"]["generated_by"] == "oel_evidence_studio_agent"
    assert row["extra"]["user_instruction"] == "Plot relative range over time for the brief."


def test_evidence_studio_can_use_selected_table_as_context(tmp_path: Path) -> None:
    _write_review_store(tmp_path)

    result = handle_evidence_studio_request(
        EvidenceStudioRequest(
            output_dir=tmp_path,
            instruction="Plot the selected table.",
            selection=EvidenceSelection(kind="table", label="Table: relative_state", table="relative_state"),
        )
    )

    assert result.ok
    assert result.artifact is not None
    assert result.selected_context["table"] == "relative_state"
    assert result.artifact.path.is_file()


def test_evidence_studio_routes_relative_velocity_prompt_to_velocity_components(tmp_path: Path) -> None:
    _write_review_store(tmp_path)

    result = handle_evidence_studio_request(
        EvidenceStudioRequest(
            output_dir=tmp_path,
            instruction="Can you give me a plot of relative velocity over time?",
        )
    )

    assert result.ok
    assert result.recipe_id == "relative_velocity_components"
    assert result.artifact is not None
    assert result.artifact.relative_path == "review/figures/evidence_relative_velocity.png"
    assert result.artifact.spec.y_columns == ["v_radial_km_s", "v_intrack_km_s", "v_crosstrack_km_s"]


def test_evidence_studio_reports_missing_review_store(tmp_path: Path) -> None:
    result = handle_evidence_studio_request(
        EvidenceStudioRequest(output_dir=tmp_path, instruction="Plot relative range over time.")
    )

    assert not result.ok
    assert "review/run.sqlite" in result.message


def test_evidence_studio_cli_dry_run_emits_guarded_plan(tmp_path: Path) -> None:
    _write_review_store(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review.evidence_studio",
            str(tmp_path),
            "--ask",
            "Can you give me a plot of relative velocity over time?",
            "--dry-run",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "planned"
    assert payload["recipe_id"] == "relative_velocity_components"
    assert payload["artifact"] is None
    assert any("Does not execute arbitrary Python" in item for item in payload["guardrails"])
    assert not (tmp_path / "review" / "figures").exists()


def test_evidence_studio_cli_generates_plot_artifact(tmp_path: Path) -> None:
    _write_review_store(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review.evidence_studio",
            str(tmp_path),
            "--ask",
            "Plot relative range over time.",
            "--style",
            "oel_light",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["artifact"]["relative_path"] == "review/figures/evidence_relative_range.png"
    assert Path(payload["artifact"]["path"]).is_file()
    assert payload["plot_spec"]["style_name"] == "oel_light"


def test_planner_task_packet_exposes_schema_and_review_context(tmp_path: Path) -> None:
    _write_review_store(tmp_path)

    packet = build_planner_task_packet(tmp_path, instruction="Plot relative velocity.")

    assert packet["instruction"] == "Plot relative velocity."
    assert "relative_state" in packet["available_tables"]
    assert packet["plan_schema"]["schema_version"] == EVIDENCE_PLAN_SCHEMA_VERSION
    assert any(item["recipe_id"] == "relative_velocity_components" for item in packet["plot_recipes"])
    assert any("EvidencePlotPlan JSON" in item for item in packet["guardrails"])


def test_prepare_evidence_agent_workspace_copies_run_evidence_and_instructions(tmp_path: Path) -> None:
    _write_review_store(tmp_path)
    (tmp_path / "index.md").write_text("# Run\n", encoding="utf-8")

    workspace = prepare_evidence_agent_workspace(tmp_path)

    assert workspace.workspace_dir == tmp_path / EVIDENCE_AGENT_WORKSPACE_DIRNAME
    assert (workspace.data_dir / "review" / "run.sqlite").is_file()
    assert (workspace.generated_dir).is_dir()
    assert "Do not run OEL simulations" in workspace.agents_path.read_text(encoding="utf-8")
    manifest = json.loads(workspace.manifest_path.read_text(encoding="utf-8"))
    assert manifest["workspace_kind"] == "oel_evidence_studio_agent_workspace"
    assert "relative_state" in manifest["review_schema"]["columns"]
    assert workspace.task_packet_path is not None
    packet = json.loads(workspace.task_packet_path.read_text(encoding="utf-8"))
    assert packet["review_db"] == str(workspace.review_db_path)


def test_evidence_studio_cli_prepares_agent_workspace(tmp_path: Path) -> None:
    _write_review_store(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review.evidence_studio",
            str(tmp_path),
            "--prepare-workspace",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    workspace_dir = Path(payload["agent_workspace"]["workspace_dir"])
    assert workspace_dir.name == EVIDENCE_AGENT_WORKSPACE_DIRNAME
    assert (workspace_dir / "AGENTS.md").is_file()
    assert (workspace_dir / "generated").is_dir()


def test_execute_evidence_plan_accepts_valid_plot_plan(tmp_path: Path) -> None:
    _write_review_store(tmp_path)
    plan = {
        "schema_version": EVIDENCE_PLAN_SCHEMA_VERSION,
        "action": "plot",
        "sql": "SELECT time_s, range_km FROM relative_state ORDER BY time_s",
        "x_column": "time_s",
        "y_columns": ["range_km"],
        "plot_type": "line",
        "style_name": "oel_light",
        "file_format": "png",
        "title": "Planner Range",
        "x_label": "Time (s)",
        "y_label": "Range (km)",
        "artifact_id": "planner_range",
        "rationale": "Range answers the brief question.",
    }

    result = execute_evidence_plan(tmp_path, plan, instruction="Plot range from planner.")

    assert result.ok
    assert result.artifact is not None
    assert result.artifact.relative_path == "review/figures/planner_range.png"
    manifest = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    row = manifest["artifacts"][-1]
    assert row["extra"]["generated_by"] == "oel_evidence_studio_codex_plan"
    assert row["extra"]["planner_rationale"] == "Range answers the brief question."


def test_execute_evidence_plan_rejects_mutating_sql(tmp_path: Path) -> None:
    _write_review_store(tmp_path)
    plan = {
        "schema_version": EVIDENCE_PLAN_SCHEMA_VERSION,
        "action": "plot",
        "sql": "DELETE FROM relative_state",
        "x_column": "time_s",
        "y_columns": ["range_km"],
    }

    result = execute_evidence_plan(tmp_path, plan)

    assert not result.ok
    assert "SELECT or WITH" in result.message
    assert not (tmp_path / "review" / "figures").exists()


def test_evidence_studio_cli_emits_task_packet_and_executes_plan_file(tmp_path: Path) -> None:
    _write_review_store(tmp_path)

    packet_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review.evidence_studio",
            str(tmp_path),
            "--ask",
            "Plot relative velocity.",
            "--task-packet",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert packet_proc.returncode == 0, packet_proc.stderr
    packet = json.loads(packet_proc.stdout)["task_packet"]
    assert packet["plan_schema"]["schema_version"] == EVIDENCE_PLAN_SCHEMA_VERSION

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": EVIDENCE_PLAN_SCHEMA_VERSION,
                "action": "plot",
                "sql": "SELECT time_s, v_radial_km_s FROM relative_state ORDER BY time_s",
                "x_column": "time_s",
                "y_columns": ["v_radial_km_s"],
                "artifact_id": "planner_velocity_radial",
            }
        ),
        encoding="utf-8",
    )

    exec_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.review.evidence_studio",
            str(tmp_path),
            "--ask",
            "Plot radial relative velocity.",
            "--plan-file",
            str(plan_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert exec_proc.returncode == 0, exec_proc.stderr
    payload = json.loads(exec_proc.stdout)
    assert payload["status"] == "ok"
    assert payload["artifact"]["relative_path"] == "review/figures/planner_velocity_radial.png"
