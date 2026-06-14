from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from sim.agent_task import compare_configs, create_plot, inspect_output, run_recipe
from sim.agent_task.failures import diagnose_failure
from sim.execution import run_simulation_config_file

ROOT = Path(__file__).resolve().parents[2]


def _review_store_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "agent_task_smoke",
        "scenario_description": "Agent task smoke test",
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


def _write_config(path: Path, output_dir: Path) -> Path:
    path.write_text(yaml.safe_dump(_review_store_config(output_dir), sort_keys=False), encoding="utf-8")
    return path


def test_agent_task_recipe_dry_run_writes_evidence_packet(tmp_path: Path) -> None:
    payload = run_recipe("quickstart_review", output_root=tmp_path, dry_run=True)

    assert payload["status"] == "validated"
    assert payload["recipe"]["recipe_id"] == "quickstart_review"
    assert payload["configs"][0]["review_enabled"] is True
    assert payload["caveats"] == ["Dry run requested: scenario was validated but not executed."]
    assert Path(payload["packet_path"]).is_file()

    config = yaml.safe_load(Path(payload["configs"][0]["config_path"]).read_text(encoding="utf-8"))
    assert config["outputs"]["review"]["enabled"] is True
    assert Path(config["outputs"]["output_dir"]) == tmp_path / "quickstart_review"


def test_agent_task_inspects_completed_run_and_creates_plot(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cfg_path = _write_config(tmp_path / "scenario.yaml", outdir)
    run_simulation_config_file(cfg_path)

    payload = inspect_output(
        outdir,
        query_names=("run_metadata", "rendezvous_closest_approach", "artifacts"),
        semantic_metric_names=("closest_approach_km",),
    )

    assert payload["status"] == "completed"
    assert payload["semantic_metrics"][0]["name"] == "closest_approach_km"
    assert (outdir / "agent_evidence_packet.json").is_file()
    query_statuses = {query["name"]: query["status"] for query in payload["review"]["queries"]}
    assert query_statuses["run_metadata"] == "ok"
    assert query_statuses["rendezvous_closest_approach"] == "ok"

    plot = create_plot(outdir, "relative_range", style_name="oel_light")
    assert plot["status"] == "ok"
    assert Path(plot["path"]).is_file()
    generated = json.loads((outdir / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    assert generated["artifacts"][-1]["source"] == "output_review_workbench"


def test_agent_task_compare_configs_writes_metric_deltas(tmp_path: Path) -> None:
    base_cfg = _write_config(tmp_path / "base.yaml", tmp_path / "base_source")
    candidate_cfg = _write_config(tmp_path / "candidate.yaml", tmp_path / "candidate_source")

    payload = compare_configs(base_cfg, candidate_cfg, output_dir=tmp_path / "compare")

    assert payload["status"] == "completed"
    assert payload["comparison"]["metrics"]["base"]["closest_approach_km"] is not None
    assert payload["comparison"]["metrics"]["candidate"]["closest_approach_km"] is not None
    assert payload["comparison"]["deltas"]["closest_approach_km"] == 0.0
    assert (tmp_path / "compare" / "agent_evidence_packet.json").is_file()


def test_agent_task_cli_lists_recipes_as_json() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "sim.agent_task", "list", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert any(item["recipe_id"] == "quickstart_review" for item in payload["items"])


def test_agent_task_failure_hints_include_review_store_missing(tmp_path: Path) -> None:
    hints = diagnose_failure("Review store not found", output_dir=tmp_path / "missing")

    codes = {hint.code for hint in hints}
    assert "review_store_missing" in codes
    assert "review_db_absent_after_run" in codes
