from .service import (
    SimulationExecutionService,
    create_single_run_engine,
    run_simulation_config_file,
    run_simulation_scenario,
)

_PRIVATE_EXPORTS = {
    "can_run_monte_carlo_campaign": ("sim.execution.campaigns", "can_run_monte_carlo_campaign"),
    "prepare_monte_carlo_runs": ("sim.execution.campaigns", "prepare_monte_carlo_runs"),
    "run_monte_carlo_campaign": ("sim.execution.campaigns", "run_monte_carlo_campaign"),
    "run_monte_carlo_runs": ("sim.execution.campaigns", "run_monte_carlo_runs"),
    "run_serial_monte_carlo_runs": ("sim.execution.campaigns", "run_serial_monte_carlo_runs"),
    "prepare_sensitivity_runs": ("sim.execution.sensitivity", "prepare_sensitivity_runs"),
    "run_sensitivity_runs": ("sim.execution.sensitivity", "run_sensitivity_runs"),
}

__all__ = [
    "SimulationExecutionService",
    "can_run_monte_carlo_campaign",
    "create_single_run_engine",
    "prepare_monte_carlo_runs",
    "prepare_sensitivity_runs",
    "run_monte_carlo_campaign",
    "run_monte_carlo_runs",
    "run_serial_monte_carlo_runs",
    "run_sensitivity_runs",
    "run_simulation_config_file",
    "run_simulation_scenario",
]


def __getattr__(name: str):
    target = _PRIVATE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'sim.execution' has no attribute '{name}'")
    import importlib

    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
