from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sim.execution import run_simulation_config_file

ROOT = Path(__file__).resolve().parents[2]
AGENT_EXAMPLES = [
    ROOT / "agents" / "examples" / "public_agent_single_satellite.yaml",
    ROOT / "agents" / "examples" / "public_agent_rendezvous_lqr.yaml",
    ROOT / "agents" / "examples" / "public_agent_ground_access.yaml",
    ROOT / "agents" / "examples" / "public_agent_attitude_hold.yaml",
]
AGENT_EXAMPLE_IDS = [path.stem for path in AGENT_EXAMPLES]


def test_public_agent_docs_define_boundaries_and_commands() -> None:
    root_agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    agents_readme = (ROOT / "agents" / "README.md").read_text(encoding="utf-8")
    public_agents = (ROOT / "agents" / "public" / "AGENTS.md").read_text(encoding="utf-8")
    rubric = (ROOT / "agents" / "public" / "evaluation-rubric.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "oel-agents.md").read_text(encoding="utf-8")

    assert "orchestrate documented workflows" in root_agents
    assert "python run_simulation.py --config <path> --validate-only" in root_agents
    assert "Scenario Generation Rules" in public_agents
    assert "Natural User Requests" in public_agents
    assert "Agent Scenario Evaluation Rubric" in rubric
    assert "python run_simulation.py --config <scenario.yaml> --validate-only" in docs
    assert "Natural Requests" in docs
    assert "Agent Example Cookbook" in docs
    assert "not included in the public core" in root_agents

    public_agent_docs = {
        "AGENTS.md": root_agents,
        "agents/README.md": agents_readme,
        "agents/public/AGENTS.md": public_agents,
        "agents/public/evaluation-rubric.md": rubric,
        "docs/oel-agents.md": docs,
    }
    for path, text in public_agent_docs.items():
        assert re.search(r"\bPro\b", text) is None, path
        assert "agents/pro" not in text, path


def test_public_agent_docs_support_natural_user_requests() -> None:
    public_agents = (ROOT / "agents" / "public" / "AGENTS.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "oel-agents.md").read_text(encoding="utf-8")

    assert not (ROOT / "agents" / "examples" / "prompts").exists()
    for text in (public_agents, docs):
        assert "ordinary language" in text or "plain language" in text
        assert "Do not require" in text or "should not need special wording" in text
        assert "validate before running" in text or "Validate the config" in text
        assert "Make me a simple satellite propagation scenario" in text


@pytest.mark.parametrize("example_path", AGENT_EXAMPLES, ids=AGENT_EXAMPLE_IDS)
def test_public_agent_generated_examples_validate_with_cli(example_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "run_simulation.py", "--config", str(example_path), "--validate-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert example_path.stem in proc.stdout
    assert "OK" in proc.stdout


@pytest.mark.parametrize("example_path", AGENT_EXAMPLES, ids=AGENT_EXAMPLE_IDS)
def test_public_agent_generated_examples_run_headlessly(example_path: Path, tmp_path: Path) -> None:
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    outdir = tmp_path / example_path.stem
    config["outputs"]["output_dir"] = str(outdir)

    cfg_path = tmp_path / example_path.name
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_simulation_config_file(cfg_path)

    assert result["scenario_name"] == config["scenario_name"]
    assert (outdir / "index.md").is_file()
    assert (outdir / "master_run_summary.json").is_file()
    assert not any(outdir.glob("*.png"))

    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    assert summary["scenario_name"] == config["scenario_name"]
    assert summary["duration_s"] == config["simulator"]["duration_s"]
    assert sorted(summary["objects"]) == sorted(
        object_id for object_id, object_cfg in config["objects"].items() if object_cfg.get("enabled", True)
    )
