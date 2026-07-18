_PRIVATE_EXPORTS = {
    "SimulationExecutionService": ("sim.execution.service", "SimulationExecutionService"),
    "create_single_run_engine": ("sim.execution.service", "create_single_run_engine"),
    "run_simulation_config_file": ("sim.execution.service", "run_simulation_config_file"),
    "run_simulation_scenario": ("sim.execution.service", "run_simulation_scenario"),
    "can_run_monte_carlo_campaign": ("sim.execution.campaigns", "can_run_monte_carlo_campaign"),
    "load_config_campaign": ("sim.execution.campaigns", "load_config_campaign"),
    "prepare_monte_carlo_runs": ("sim.execution.campaigns", "prepare_monte_carlo_runs"),
    "run_config_campaign": ("sim.execution.campaigns", "run_config_campaign"),
    "run_monte_carlo_campaign": ("sim.execution.campaigns", "run_monte_carlo_campaign"),
    "run_monte_carlo_runs": ("sim.execution.campaigns", "run_monte_carlo_runs"),
    "run_serial_monte_carlo_runs": ("sim.execution.campaigns", "run_serial_monte_carlo_runs"),
    "validate_config_campaign": ("sim.execution.campaigns", "validate_config_campaign"),
    "prepare_sensitivity_runs": ("sim.execution.sensitivity", "prepare_sensitivity_runs"),
    "run_sensitivity_runs": ("sim.execution.sensitivity", "run_sensitivity_runs"),
    "prepare_batch_run_configs": ("sim.execution.validation", "prepare_batch_run_configs"),
    "validate_generated_batch_configs": ("sim.execution.validation", "validate_generated_batch_configs"),
    "run_covariance_analysis": ("sim.execution.covariance", "run_covariance_analysis"),
}

__all__ = [
    "SimulationExecutionService",
    "can_run_monte_carlo_campaign",
    "create_single_run_engine",
    "load_config_campaign",
    "prepare_monte_carlo_runs",
    "prepare_batch_run_configs",
    "prepare_sensitivity_runs",
    "run_config_campaign",
    "run_covariance_analysis",
    "run_monte_carlo_campaign",
    "run_monte_carlo_runs",
    "run_serial_monte_carlo_runs",
    "run_sensitivity_runs",
    "run_simulation_config_file",
    "run_simulation_scenario",
    "validate_config_campaign",
    "validate_generated_batch_configs",
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
