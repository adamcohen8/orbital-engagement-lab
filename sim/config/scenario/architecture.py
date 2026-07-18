"""Static ownership map for scenario configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfigFamily:
    name: str
    module: str
    capabilities: tuple[str, ...]
    facade: str = "sim.config.scenario_yaml"


SCENARIO_CONFIG_FAMILIES: tuple[ScenarioConfigFamily, ...] = (
    ScenarioConfigFamily("models", "sim.config.scenario.models", ("SimulationScenarioConfig", "AgentSection")),
    ScenarioConfigFamily("primitives", "sim.config.scenario.primitives", ("_as_dict", "_reject_unknown_fields")),
    ScenarioConfigFamily("presets", "sim.config.scenario.presets", ("_resolve_preset_path", "_resolve_agent_presets")),
    ScenarioConfigFamily("objects", "sim.config.scenario.objects", ("_parse_objects_section", "_parse_initial_state_section")),
    ScenarioConfigFamily("simulator", "sim.config.scenario.simulator", ("_parse_simulator_section",)),
    ScenarioConfigFamily("analysis", "sim.config.scenario.analysis", ("_parse_analysis_section",)),
    ScenarioConfigFamily("outputs", "sim.config.scenario.outputs", ("_parse_outputs_section",)),
    ScenarioConfigFamily("paths", "sim.config.scenario.paths", ("_validate_config_read_paths",)),
    ScenarioConfigFamily(
        "validation",
        "sim.config.scenario.validation",
        ("_validate_physics_runtime_settings", "_validate_object_references"),
    ),
    ScenarioConfigFamily(
        "loader",
        "sim.config.scenario.loader",
        ("scenario_config_from_dict", "load_simulation_yaml"),
    ),
)
