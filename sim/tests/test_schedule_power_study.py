from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.analysis.history_adapters import AnalysisHistory
from sim.analysis.mission_scheduling import (
    MissionSchedulingProblem,
    solve_mission_schedule,
    write_mission_scheduling_artifacts,
)
from sim.analysis.schedule_power_study import (
    SchedulePowerStudyError,
    build_schedule_power_study,
    build_source_schedule_power_study,
    inspect_schedule_power_study,
    inspect_source_schedule_power_study,
)
from sim.analysis.spacecraft_power import power_history_to_dict
from sim.study import main as study_main

ROOT = Path(__file__).resolve().parents[2]
EPOCH_JD_UTC = 2461041.5
SOURCE_EPOCH_JD_UTC = 2451545.0


@pytest.fixture(scope="module")
def source_chain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("source-schedule-power")
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/python/mission_scheduling_source_chain.py"),
         "--output-root", str(root)],
        cwd=ROOT,
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["source_status"] == "verified"
    return root


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, dict, dict]:
    schedule_problem = MissionSchedulingProblem.from_mapping(
        json.loads((ROOT / "examples/mission_scheduling/public_two_asset_collection_problem.json").read_text())
    )
    schedule = write_mission_scheduling_artifacts(
        solve_mission_schedule(schedule_problem), tmp_path / "source_schedule"
    )
    power_problem = json.loads((ROOT / "examples/spacecraft_power/public_schedule_power_problem.json").read_text())
    power_problem["horizon_end_s"] = 120.0
    power_problem["integration_step_s"] = 10.0
    history = AnalysisHistory(
        object_id="SAT-A",
        product_kind="synthetic_integration_fixture",
        state_provider_id="test:schedule-power",
        frame="eci",
        initial_jd_utc=EPOCH_JD_UTC,
        times_s=np.array([0.0, 120.0]),
        position_eci_km=np.array([[7000.0, 0.0, 0.0], [7000.0, 0.0, 0.0]]),
        velocity_eci_km_s=np.zeros((2, 3)),
    )
    return schedule.output_dir, power_problem, power_history_to_dict(history)


def _build(tmp_path: Path, inputs: tuple[Path, dict, dict]) -> Path:
    schedule, problem, history = inputs
    output = tmp_path / "combined"
    result = build_schedule_power_study(
        schedule_dir=schedule,
        problem=problem,
        history=history,
        observation_load_w=180.0,
        downlink_load_w=120.0,
        schedule_epoch_jd_utc=EPOCH_JD_UTC,
        output_dir=output,
    )
    assert result["status"] == "verified"
    return output


def test_build_inspect_and_replay_complete_two_domain_study(
    tmp_path: Path, inputs: tuple[Path, dict, dict], capsys: pytest.CaptureFixture[str]
) -> None:
    output = _build(tmp_path, inputs)
    result = inspect_schedule_power_study(output)
    schedule = json.loads((output / "schedule" / "mission_schedule_summary.json").read_text())
    power = json.loads((output / "power" / "spacecraft_power_summary.json").read_text())
    plan = json.loads((output / "study" / "study_plan.json").read_text())

    assert result["schedule_semantic_sha256"] == schedule["schedule_semantic_sha256"]
    assert result["schedule_semantic_sha256"] in power["source_product_sha256s"]
    assert {step["step_id"] for step in plan["steps"]} == {"schedule", "power"}
    assert next(step for step in plan["steps"] if step["step_id"] == "power")["depends_on"] == ["schedule"]
    assert result["feasibility"] in {"feasible", "infeasible"}
    assert study_main(["inspect-schedule-power", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["study_id"] == result["study_id"]


def test_cli_builds_from_retained_schedule_and_json_inputs(
    tmp_path: Path, inputs: tuple[Path, dict, dict], capsys: pytest.CaptureFixture[str]
) -> None:
    schedule, problem, history = inputs
    problem_path = tmp_path / "power_problem.json"
    history_path = tmp_path / "history.json"
    output = tmp_path / "cli_study"
    problem_path.write_text(json.dumps(problem), encoding="utf-8")
    history_path.write_text(json.dumps(history), encoding="utf-8")

    assert study_main([
        "build-schedule-power",
        "--schedule-dir", str(schedule),
        "--problem", str(problem_path),
        "--history", str(history_path),
        "--schedule-epoch-jd-utc", str(EPOCH_JD_UTC),
        "--observation-load-w", "180",
        "--downlink-load-w", "120",
        "--output-dir", str(output),
        "--study-id", "cli-schedule-power",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["study_id"] == "cli-schedule-power"
    assert inspect_schedule_power_study(output)["status"] == "verified"


def test_rejects_mismatched_asset_epoch_and_schedule_horizon(
    tmp_path: Path, inputs: tuple[Path, dict, dict]
) -> None:
    schedule, problem, history = inputs
    kwargs = {
        "schedule_dir": schedule,
        "problem": problem,
        "history": history,
        "observation_load_w": 180.0,
        "downlink_load_w": 120.0,
        "schedule_epoch_jd_utc": EPOCH_JD_UTC,
        "output_dir": tmp_path / "rejected",
    }
    with pytest.raises(SchedulePowerStudyError, match="schedule epoch"):
        build_schedule_power_study(**{**kwargs, "schedule_epoch_jd_utc": EPOCH_JD_UTC + 1.0})
    with pytest.raises(SchedulePowerStudyError, match="no activities"):
        build_schedule_power_study(**{
            **kwargs,
            "problem": {**problem, "asset_id": "SAT-C"},
            "history": {**history, "asset_id": "SAT-C"},
        })
    with pytest.raises(SchedulePowerStudyError, match="Schedule horizon"):
        build_schedule_power_study(**{**kwargs, "problem": {**problem, "horizon_end_s": 100.0}})
    assert not kwargs["output_dir"].exists()


@pytest.mark.parametrize(
    "relative",
    [
        "schedule/mission_schedule.csv",
        "power/spacecraft_power_summary.json",
        "study/evidence/power.json",
    ],
)
def test_inspection_rejects_changed_retained_evidence(
    tmp_path: Path, inputs: tuple[Path, dict, dict], relative: str
) -> None:
    output = _build(tmp_path, inputs)
    target = output / relative
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        inspect_schedule_power_study(output)


def _source_inputs(inputs: tuple[Path, dict, dict]) -> tuple[dict, dict]:
    _, problem, history = inputs
    return (
        {**problem, "epoch_jd_utc": SOURCE_EPOCH_JD_UTC},
        {**history, "epoch_jd_utc": SOURCE_EPOCH_JD_UTC},
    )


def test_source_to_schedule_to_power_study_cli(
    tmp_path: Path, source_chain: Path, inputs: tuple[Path, dict, dict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    problem, history = _source_inputs(inputs)
    problem_path = tmp_path / "problem.json"
    history_path = tmp_path / "history.json"
    output = tmp_path / "source_study"
    problem_path.write_text(json.dumps(problem), encoding="utf-8")
    history_path.write_text(json.dumps(history), encoding="utf-8")

    assert study_main([
        "build-source-schedule-power",
        "--source-plan", str(source_chain / "source_plan.json"),
        "--problem", str(problem_path),
        "--history", str(history_path),
        "--observation-load-w", "180",
        "--downlink-load-w", "120",
        "--output-dir", str(output),
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified"
    assert result["source_count"] == 5
    assert result["schedule_semantic_sha256"] == json.loads(
        (output / "source_schedule" / "mission_schedule_source_manifest.json").read_text()
    )["schedule_semantic_sha256"]
    assert inspect_source_schedule_power_study(output) == result
    assert study_main(["inspect-source-schedule-power", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("relative", [
    "source_schedule/source_products/sat-a-collection/collection_evidence.json",
    "source_schedule/schedule/mission_schedule.csv",
    "schedule_power_study/power/spacecraft_power_summary.json",
])
def test_source_study_rejects_changed_evidence(
    tmp_path: Path, source_chain: Path, inputs: tuple[Path, dict, dict], relative: str,
) -> None:
    problem, history = _source_inputs(inputs)
    output = tmp_path / "source_study"
    build_source_schedule_power_study(
        source_plan=json.loads((source_chain / "source_plan.json").read_text()),
        base_dir=source_chain,
        problem=problem,
        history=history,
        observation_load_w=180.0,
        downlink_load_w=120.0,
        output_dir=output,
    )
    target = output / relative
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        inspect_source_schedule_power_study(output)


def test_source_study_rejects_mismatched_epoch_and_extra_artifact(
    tmp_path: Path, source_chain: Path, inputs: tuple[Path, dict, dict],
) -> None:
    _, problem, history = inputs
    kwargs = {
        "source_plan": json.loads((source_chain / "source_plan.json").read_text()),
        "base_dir": source_chain,
        "problem": problem,
        "history": history,
        "observation_load_w": 180.0,
        "downlink_load_w": 120.0,
        "output_dir": tmp_path / "source_study",
    }
    with pytest.raises(SchedulePowerStudyError, match="schedule epoch"):
        build_source_schedule_power_study(**kwargs)
    assert not kwargs["output_dir"].exists()

    source_problem, source_history = _source_inputs(inputs)
    build_source_schedule_power_study(**{
        **kwargs, "problem": source_problem, "history": source_history,
    })
    (kwargs["output_dir"] / "source_schedule" / "unexpected.txt").write_text("extra")
    with pytest.raises(SchedulePowerStudyError, match="artifact inventory"):
        inspect_source_schedule_power_study(kwargs["output_dir"])


def test_source_study_consumes_exported_completed_run_history(
    tmp_path: Path, source_chain: Path, inputs: tuple[Path, dict, dict],
) -> None:
    config = yaml.safe_load((ROOT / "agents/examples/public_agent_single_satellite.yaml").read_text())
    config["scenario_name"] = "schedule_power_completed_run_test"
    config["objects"] = {"SAT-A": config["objects"]["target"]}
    config["simulator"]["duration_s"] = 120.0
    config["simulator"]["initial_jd_utc"] = SOURCE_EPOCH_JD_UTC
    run_dir = tmp_path / "completed_run"
    config["outputs"]["output_dir"] = str(run_dir)
    config_path = tmp_path / "scenario.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    for extra in (["--validate-only"], []):
        result = subprocess.run(
            [sys.executable, "run_simulation.py", "--config", str(config_path), *extra],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert (run_dir / "review" / "run.sqlite").is_file()

    history_path = tmp_path / "exported_history.json"
    export = subprocess.run(
        [sys.executable, "-m", "sim.spacecraft_power", "export-review-history",
         str(run_dir), "--object-id", "SAT-A", "--output", str(history_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert export.returncode == 0, export.stdout + export.stderr
    assert json.loads(export.stdout)["sample_count"] == 121

    problem, _ = _source_inputs(inputs)
    output = tmp_path / "integrated_study"
    result = build_source_schedule_power_study(
        source_plan=json.loads((source_chain / "source_plan.json").read_text()),
        base_dir=source_chain,
        problem=problem,
        history=json.loads(history_path.read_text()),
        observation_load_w=180.0,
        downlink_load_w=120.0,
        output_dir=output,
    )
    assert result["status"] == "verified"
    assert result["source_count"] == 5
    assert inspect_source_schedule_power_study(output) == result
