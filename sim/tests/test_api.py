from pathlib import Path

from sim import SimulationConfig, SimulationSession


def test_public_api_loads_and_runs_smoke_config():
    cfg = SimulationConfig.from_yaml("configs/automation_smoke.yaml")
    session = SimulationSession.from_config(cfg)

    result = session.run()

    assert result.summary["scenario_name"] == "automation_smoke"
    assert result.summary["duration_s"] == 120.0


def test_public_api_round_trips_config_dict():
    cfg = SimulationConfig.from_yaml("configs/automation_smoke.yaml")
    data = cfg.to_dict()

    reloaded = SimulationConfig.from_dict(data)

    assert reloaded.scenario_name == "automation_smoke"
    assert reloaded.to_scenario_config().target.enabled is True
    assert Path(reloaded.to_scenario_config().outputs.output_dir).as_posix() == "outputs/automation_smoke"
