"""Compatibility façade for scenario YAML schema and loading."""

from __future__ import annotations

from sim.config.scenario.analysis import (
    _monte_carlo_from_analysis as _monte_carlo_from_analysis,
)
from sim.config.scenario.analysis import (
    _parse_analysis_baseline_section as _parse_analysis_baseline_section,
)
from sim.config.scenario.analysis import (
    _parse_analysis_execution_section as _parse_analysis_execution_section,
)
from sim.config.scenario.analysis import (
    _parse_analysis_monte_carlo_section as _parse_analysis_monte_carlo_section,
)
from sim.config.scenario.analysis import (
    _parse_analysis_section as _parse_analysis_section,
)
from sim.config.scenario.analysis import (
    _parse_covariance_collision_screening_section as _parse_covariance_collision_screening_section,
)
from sim.config.scenario.analysis import (
    _parse_covariance_diagonal as _parse_covariance_diagonal,
)
from sim.config.scenario.analysis import (
    _parse_covariance_matrix as _parse_covariance_matrix,
)
from sim.config.scenario.analysis import (
    _parse_covariance_object_section as _parse_covariance_object_section,
)
from sim.config.scenario.analysis import (
    _parse_covariance_pair_section as _parse_covariance_pair_section,
)
from sim.config.scenario.analysis import (
    _parse_covariance_section as _parse_covariance_section,
)
from sim.config.scenario.analysis import (
    _parse_mc_variation as _parse_mc_variation,
)
from sim.config.scenario.analysis import (
    _parse_mission_recovery_planner_section as _parse_mission_recovery_planner_section,
)
from sim.config.scenario.analysis import (
    _parse_mission_recovery_section as _parse_mission_recovery_section,
)
from sim.config.scenario.analysis import (
    _parse_mission_recovery_target_orbit_section as _parse_mission_recovery_target_orbit_section,
)
from sim.config.scenario.analysis import (
    _parse_orbit_transfer_planner_section as _parse_orbit_transfer_planner_section,
)
from sim.config.scenario.analysis import (
    _parse_orbital_delivery_section as _parse_orbital_delivery_section,
)
from sim.config.scenario.analysis import (
    _parse_sensitivity_parameter as _parse_sensitivity_parameter,
)
from sim.config.scenario.analysis import (
    _parse_sensitivity_section as _parse_sensitivity_section,
)
from sim.config.scenario.loader import (
    load_simulation_yaml as load_simulation_yaml,
)
from sim.config.scenario.loader import (
    scenario_config_from_dict as scenario_config_from_dict,
)
from sim.config.scenario.models import (
    AgentSection as AgentSection,
)
from sim.config.scenario.models import (
    AlgorithmPointer as AlgorithmPointer,
)
from sim.config.scenario.models import (
    AnalysisBaselineSection as AnalysisBaselineSection,
)
from sim.config.scenario.models import (
    AnalysisExecutionSection as AnalysisExecutionSection,
)
from sim.config.scenario.models import (
    AnalysisMonteCarloSection as AnalysisMonteCarloSection,
)
from sim.config.scenario.models import (
    AnalysisSection as AnalysisSection,
)
from sim.config.scenario.models import (
    BridgePointer as BridgePointer,
)
from sim.config.scenario.models import (
    CovarianceCollisionScreeningSection as CovarianceCollisionScreeningSection,
)
from sim.config.scenario.models import (
    CovarianceFiniteDifferenceSection as CovarianceFiniteDifferenceSection,
)
from sim.config.scenario.models import (
    CovarianceObjectSection as CovarianceObjectSection,
)
from sim.config.scenario.models import (
    CovariancePairSection as CovariancePairSection,
)
from sim.config.scenario.models import (
    CovarianceProcessNoiseSection as CovarianceProcessNoiseSection,
)
from sim.config.scenario.models import (
    CovarianceSection as CovarianceSection,
)
from sim.config.scenario.models import (
    GroundStationSection as GroundStationSection,
)
from sim.config.scenario.models import (
    MissionRecoverySection as MissionRecoverySection,
)
from sim.config.scenario.models import (
    MonteCarloSection as MonteCarloSection,
)
from sim.config.scenario.models import (
    MonteCarloVariation as MonteCarloVariation,
)
from sim.config.scenario.models import (
    OutputAIConfigSection as OutputAIConfigSection,
)
from sim.config.scenario.models import (
    OutputAIReportSection as OutputAIReportSection,
)
from sim.config.scenario.models import (
    OutputAnimationsSection as OutputAnimationsSection,
)
from sim.config.scenario.models import (
    OutputMonteCarloSection as OutputMonteCarloSection,
)
from sim.config.scenario.models import (
    OutputPlotsSection as OutputPlotsSection,
)
from sim.config.scenario.models import (
    OutputResourceLimitsSection as OutputResourceLimitsSection,
)
from sim.config.scenario.models import (
    OutputReviewSection as OutputReviewSection,
)
from sim.config.scenario.models import (
    OutputsSection as OutputsSection,
)
from sim.config.scenario.models import (
    OutputStatsSection as OutputStatsSection,
)
from sim.config.scenario.models import (
    SensitivityParameter as SensitivityParameter,
)
from sim.config.scenario.models import (
    SensitivitySection as SensitivitySection,
)
from sim.config.scenario.models import (
    SimulationScenarioConfig as SimulationScenarioConfig,
)
from sim.config.scenario.models import (
    SimulatorAccelerationSection as SimulatorAccelerationSection,
)
from sim.config.scenario.models import (
    SimulatorDynamicsSection as SimulatorDynamicsSection,
)
from sim.config.scenario.models import (
    SimulatorEnvironmentSection as SimulatorEnvironmentSection,
)
from sim.config.scenario.models import (
    SimulatorExecutionSection as SimulatorExecutionSection,
)
from sim.config.scenario.models import (
    SimulatorFramesSection as SimulatorFramesSection,
)
from sim.config.scenario.models import (
    SimulatorPluginValidationSection as SimulatorPluginValidationSection,
)
from sim.config.scenario.models import (
    SimulatorSection as SimulatorSection,
)
from sim.config.scenario.models import (
    SimulatorTerminationSection as SimulatorTerminationSection,
)
from sim.config.scenario.models import (
    _TypedConfigDict as _TypedConfigDict,
)
from sim.config.scenario.objects import (
    _INITIAL_STATE_ALLOWED_KEYS as _INITIAL_STATE_ALLOWED_KEYS,
)
from sim.config.scenario.objects import (
    _INITIAL_STATE_AUX_KEYS as _INITIAL_STATE_AUX_KEYS,
)
from sim.config.scenario.objects import (
    _INITIAL_STATE_FORM_KEYS as _INITIAL_STATE_FORM_KEYS,
)
from sim.config.scenario.objects import (
    _parse_agent_section as _parse_agent_section,
)
from sim.config.scenario.objects import (
    _parse_algorithm_pointer as _parse_algorithm_pointer,
)
from sim.config.scenario.objects import (
    _parse_bridge_pointer as _parse_bridge_pointer,
)
from sim.config.scenario.objects import (
    _parse_ground_station_measurements as _parse_ground_station_measurements,
)
from sim.config.scenario.objects import (
    _parse_ground_station_section as _parse_ground_station_section,
)
from sim.config.scenario.objects import (
    _parse_ground_stations_section as _parse_ground_stations_section,
)
from sim.config.scenario.objects import (
    _parse_initial_state_section as _parse_initial_state_section,
)
from sim.config.scenario.objects import (
    _parse_objects_section as _parse_objects_section,
)
from sim.config.scenario.objects import (
    _reject_unsupported_agent_body_overrides as _reject_unsupported_agent_body_overrides,
)
from sim.config.scenario.outputs import (
    _parse_outputs_section as _parse_outputs_section,
)
from sim.config.scenario.paths import (
    _resolve_geometry_profile_path_in_specs as _resolve_geometry_profile_path_in_specs,
)
from sim.config.scenario.paths import (
    _resolve_geometry_profile_paths as _resolve_geometry_profile_paths,
)
from sim.config.scenario.paths import (
    _validate_config_read_paths as _validate_config_read_paths,
)
from sim.config.scenario.presets import (
    _AGENT_FRAGMENT_KEYS as _AGENT_FRAGMENT_KEYS,
)
from sim.config.scenario.presets import (
    _AGENT_PRESET_KEYS as _AGENT_PRESET_KEYS,
)
from sim.config.scenario.presets import (
    _PRESET_METADATA_KEYS as _PRESET_METADATA_KEYS,
)
from sim.config.scenario.presets import (
    _agent_fragment_from_preset as _agent_fragment_from_preset,
)
from sim.config.scenario.presets import (
    _deep_merge_dicts as _deep_merge_dicts,
)
from sim.config.scenario.presets import (
    _load_yaml_mapping as _load_yaml_mapping,
)
from sim.config.scenario.presets import (
    _resolve_agent_preset as _resolve_agent_preset,
)
from sim.config.scenario.presets import (
    _resolve_agent_presets as _resolve_agent_presets,
)
from sim.config.scenario.presets import (
    _resolve_preset_path as _resolve_preset_path,
)
from sim.config.scenario.primitives import (
    _OUTPUT_ANIMATIONS_UNSUPPORTED_ALIASES as _OUTPUT_ANIMATIONS_UNSUPPORTED_ALIASES,
)
from sim.config.scenario.primitives import (
    _OUTPUT_PLOTS_UNSUPPORTED_ALIASES as _OUTPUT_PLOTS_UNSUPPORTED_ALIASES,
)
from sim.config.scenario.primitives import (
    _OUTPUTS_UNSUPPORTED_ALIASES as _OUTPUTS_UNSUPPORTED_ALIASES,
)
from sim.config.scenario.primitives import (
    _REENTRY_TERMINATION_LIMIT_FIELDS as _REENTRY_TERMINATION_LIMIT_FIELDS,
)
from sim.config.scenario.primitives import (
    _ROOT_UNSUPPORTED_ALIASES as _ROOT_UNSUPPORTED_ALIASES,
)
from sim.config.scenario.primitives import (
    _SIMULATOR_UNSUPPORTED_ALIASES as _SIMULATOR_UNSUPPORTED_ALIASES,
)
from sim.config.scenario.primitives import (
    _as_dict as _as_dict,
)
from sim.config.scenario.primitives import (
    _enforce_strict_booleans as _enforce_strict_booleans,
)
from sim.config.scenario.primitives import (
    _is_bool_like_key as _is_bool_like_key,
)
from sim.config.scenario.primitives import (
    _parse_bool as _parse_bool,
)
from sim.config.scenario.primitives import (
    _parse_float as _parse_float,
)
from sim.config.scenario.primitives import (
    _parse_optional_float as _parse_optional_float,
)
from sim.config.scenario.primitives import (
    _plain_config_data as _plain_config_data,
)
from sim.config.scenario.primitives import (
    _reject_unknown_fields as _reject_unknown_fields,
)
from sim.config.scenario.primitives import (
    _reject_unsupported_aliases as _reject_unsupported_aliases,
)
from sim.config.scenario.primitives import (
    _UnsupportedAliasMap as _UnsupportedAliasMap,
)
from sim.config.scenario.primitives import (
    _validate_integer_multiple as _validate_integer_multiple,
)
from sim.config.scenario.primitives import (
    _validate_sim_timing as _validate_sim_timing,
)
from sim.config.scenario.simulator import (
    _normalize_dynamics_section as _normalize_dynamics_section,
)
from sim.config.scenario.simulator import (
    _normalize_reentry_section as _normalize_reentry_section,
)
from sim.config.scenario.simulator import (
    _normalize_reentry_termination_block as _normalize_reentry_termination_block,
)
from sim.config.scenario.simulator import (
    _normalize_simulator_termination_block as _normalize_simulator_termination_block,
)
from sim.config.scenario.simulator import (
    _parse_acceleration_section as _parse_acceleration_section,
)
from sim.config.scenario.simulator import (
    _parse_resource_profile as _parse_resource_profile,
)
from sim.config.scenario.simulator import (
    _parse_simulator_execution_section as _parse_simulator_execution_section,
)
from sim.config.scenario.simulator import (
    _parse_simulator_frames_section as _parse_simulator_frames_section,
)
from sim.config.scenario.simulator import (
    _parse_simulator_section as _parse_simulator_section,
)
from sim.config.scenario.validation import (
    _validate_object_references as _validate_object_references,
)
from sim.config.scenario.validation import (
    _validate_physics_runtime_settings as _validate_physics_runtime_settings,
)

AlgorithmPointer.__module__ = __name__
BridgePointer.__module__ = __name__
AgentSection.__module__ = __name__
GroundStationSection.__module__ = __name__
_TypedConfigDict.__module__ = __name__
SimulatorAccelerationSection.__module__ = __name__
SimulatorExecutionSection.__module__ = __name__
SimulatorFramesSection.__module__ = __name__
SimulatorSection.__module__ = __name__
SimulatorDynamicsSection.__module__ = __name__
SimulatorEnvironmentSection.__module__ = __name__
SimulatorPluginValidationSection.__module__ = __name__
SimulatorTerminationSection.__module__ = __name__
OutputStatsSection.__module__ = __name__
OutputPlotsSection.__module__ = __name__
OutputAnimationsSection.__module__ = __name__
OutputMonteCarloSection.__module__ = __name__
OutputAIReportSection.__module__ = __name__
OutputAIConfigSection.__module__ = __name__
OutputResourceLimitsSection.__module__ = __name__
OutputReviewSection.__module__ = __name__
OutputsSection.__module__ = __name__
MonteCarloVariation.__module__ = __name__
MonteCarloSection.__module__ = __name__
AnalysisExecutionSection.__module__ = __name__
AnalysisBaselineSection.__module__ = __name__
AnalysisMonteCarloSection.__module__ = __name__
SensitivityParameter.__module__ = __name__
SensitivitySection.__module__ = __name__
CovarianceObjectSection.__module__ = __name__
CovarianceCollisionScreeningSection.__module__ = __name__
CovariancePairSection.__module__ = __name__
CovarianceFiniteDifferenceSection.__module__ = __name__
CovarianceProcessNoiseSection.__module__ = __name__
CovarianceSection.__module__ = __name__
MissionRecoverySection.__module__ = __name__
AnalysisSection.__module__ = __name__
SimulationScenarioConfig.__module__ = __name__
