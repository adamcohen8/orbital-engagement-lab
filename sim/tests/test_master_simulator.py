from sim.config import load_simulation_yaml
from sim.execution import run_simulation_config_file


def test_public_smoke_config_loads():
    cfg = load_simulation_yaml("configs/automation_smoke.yaml")

    assert cfg.scenario_name == "automation_smoke"
    assert cfg.target.enabled is True
    assert cfg.simulator.duration_s == 120.0


def test_public_master_runner_executes_smoke_config():
    payload = run_simulation_config_file("configs/automation_smoke.yaml")

    assert payload["scenario_name"] == "automation_smoke"
    assert payload["run"]["duration_s"] == 120.0
