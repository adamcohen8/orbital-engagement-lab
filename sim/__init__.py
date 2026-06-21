from __future__ import annotations

import importlib
from typing import Any

_API_EXPORTS = [
    "MetricStudyResult",
    "ScenarioArtifact",
    "ScenarioBuilder",
    "SimulationConfig",
    "SimulationResult",
    "SimulationSession",
    "SimulationSnapshot",
    "SimulationWorkspace",
    "ValidationIssue",
    "ValidationReport",
]

_ACTUATOR_EXPORTS = [
    "ActuatorLimits",
    "SimpleActuator",
    "OrbitalActuator",
    "OrbitalActuatorLimits",
    "AttitudeActuator",
    "ReactionWheelLimits",
    "MagnetorquerLimits",
    "ThrusterPulseLimits",
    "ControlMomentGyroLimits",
    "WheelDesaturationLimits",
    "RcsThruster",
    "RcsClusterLimits",
    "ElectricPropulsionLimits",
    "GimbaledThrusterLimits",
    "ActuatorFaultConfig",
    "FaultedActuator",
    "apply_actuator_faults",
]

_ACTUATOR_PRESET_EXPORTS = [
    "BASIC_RCS_6DOF",
    "BASIC_ELECTRIC_PROPULSION",
    "BASIC_MAGNETORQUER_TRIAD",
    "BASIC_CMG_TRIAD",
    "BASIC_GIMBALED_THRUSTER",
    "ACTUATOR_PRESETS",
    "available_actuator_preset_names",
    "actuator_preset_to_specs",
    "resolve_actuator_specs_from_satellite_specs",
]

_CONTROL_EXPORTS = [
    "ZeroController",
    "DeltaVManeuver",
    "ThrustLimitedDeltaVManeuver",
    "ThrustLimitedDeltaVManeuverResult",
    "ImpulsiveManeuver",
    "ImpulsiveManeuverResult",
    "AttitudeAgnosticImpulsiveManeuverer",
    "IntegratedManeuverCommand",
    "IntegratedManeuverDecision",
    "HCWLQRController",
    "HCWNoRadialLQRController",
    "HCWNoRadialManualController",
    "HCWCurvInputRectOutputController",
    "HCWInTrackCrossTrackMPCController",
    "RelativeOrbitMPCController",
    "RCSAllocationAwareController",
    "ElectricPropulsionController",
    "GimbaledThrusterController",
    "PredictiveBurnConfig",
    "PredictiveBurnScheduler",
    "OrbitalAttitudeManeuverCoordinator",
    "StationkeepingController",
    "SafetyBarrierController",
    "RiskThresholdController",
    "RobustMPCController",
    "StochasticPolicyController",
    "ZeroTorqueController",
    "PoseCommandGenerator",
    "DetumbleThenSlewController",
    "ECIDetumblePDController",
    "RICDetumblePDController",
    "SnapAttitudeController",
    "SnapAndHoldRICAttitudeController",
    "SurrogateSnapECIController",
    "SurrogateSnapRICController",
    "QuaternionPDController",
    "ReactionWheelPDController",
    "ReactionWheelPIDController",
    "MagnetorquerBdotController",
    "WheelDesaturationController",
    "CMGSteeringController",
    "SmallAngleLQRController",
    "RICFrameLQRController",
    "RICFramePDController",
    "RICFramePIDController",
]

_CONFIG_EXPORTS = [
    "SimulationProfileName",
    "SimulationProfile",
    "PROFILE_FAST",
    "PROFILE_OPS",
    "PROFILE_HIGH_FIDELITY",
    "profile_choices",
    "get_simulation_profile",
    "resolve_dt_s",
    "resolve_steps_for_duration",
    "default_env_for_profile",
    "default_disturbance_config_for_profile",
    "build_orbit_propagator_for_profile",
    "build_default_ops_orbit_propagator",
    "AlgorithmPointer",
    "BridgePointer",
    "AgentSection",
    "GroundStationSection",
    "SimulatorSection",
    "SimulatorDynamicsSection",
    "SimulatorEnvironmentSection",
    "SimulatorPluginValidationSection",
    "SimulatorTerminationSection",
    "MonteCarloVariation",
    "MonteCarloSection",
    "OutputStatsSection",
    "OutputPlotsSection",
    "OutputAnimationsSection",
    "OutputMonteCarloSection",
    "OutputAIReportSection",
    "OutputAIConfigSection",
    "OutputsSection",
    "SimulationScenarioConfig",
    "scenario_config_from_dict",
    "load_simulation_yaml",
    "validate_scenario_plugins",
]

_CORE_EXPORTS = [
    "SimObject",
    "Command",
    "ObjectConfig",
    "SimConfig",
    "SimLog",
    "StateBelief",
    "StateTruth",
]

_DYNAMICS_EXPORTS = [
    "OrbitalAttitudeDynamics",
]

_ATTITUDE_DYNAMICS_EXPORTS = [
    "DisturbanceTorqueConfig",
    "DisturbanceTorqueModel",
]

_ORBIT_DYNAMICS_EXPORTS = [
    "EARTH_MU_KM3_S2",
    "EARTH_RADIUS_KM",
    "EARTH_J2",
    "EARTH_J3",
    "EARTH_J4",
    "AtmosphereModelName",
    "SphericalHarmonicTerm",
    "datetime_to_julian_date",
    "julian_date_to_datetime",
    "gmst_angle_rad_from_jd",
    "sun_position_eci_km_enhanced",
    "sun_position_eci_km_simple",
    "moon_position_eci_km_enhanced",
    "moon_position_eci_km_simple",
    "resolved_jd_utc",
    "resolve_body_position_eci_km",
    "resolve_sun_moon_positions",
    "resolve_time_dependent_env",
    "srp_shadow_factor",
    "spice_sun_moon_positions_eci_km",
    "spice_supported_body_names",
    "OrbitContext",
    "OrbitPropagator",
    "density_exponential",
    "density_ussa1976",
    "density_msis86",
    "density_nrlmsise00",
    "density_jacchia70",
    "density_jb2006",
    "density_jb2008",
    "density_harris_priester",
    "density_from_model",
    "parse_spherical_harmonic_terms",
    "accel_spherical_harmonics_terms",
    "load_hpop_ggm03_terms",
    "load_icgem_gfc_terms",
    "load_real_earth_gravity_terms",
    "j2_plugin",
    "j3_plugin",
    "j4_plugin",
    "spherical_harmonics_plugin",
    "drag_plugin",
    "srp_plugin",
    "third_body_planets_plugin",
    "third_body_moon_plugin",
    "third_body_sun_plugin",
]

_ESTIMATION_EXPORTS = [
    "OrbitEKFEstimator",
    "OrbitUKFEstimator",
    "AttitudeEKFEstimator",
    "JointStateEKFEstimator",
    "JointStateEstimator",
    "AoITrackingEstimator",
    "build_dynamics_od_quality_gates",
    "build_orbit_od_parameter_set",
    "selected_orbit_od_parameters",
    "solve_dynamics_orbit_determination",
]

_METRIC_EXPORTS = [
    "ScoreSummary",
    "compute_scores",
    "EngagementMetrics",
    "compute_engagement_metrics",
]

_MISSION_EXPORTS = [
    "AttitudeDetumbleGateMissionModule",
    "BudgetedEndStateExecution",
    "DesiredStateMissionStrategy",
    "DefensiveMissionStrategy",
    "MissionExecutiveStrategy",
    "RocketPursuitMissionStrategy",
    "RocketPredefinedOrbitMissionStrategy",
    "RocketGoNowExecution",
    "RocketGoWhenPossibleExecution",
    "RocketWaitOptimalExecution",
    "SatelliteMissionModule",
    "DefensiveRICAxisBurnMissionModule",
    "RocketMissionModule",
    "EndStateManeuverMissionModule",
    "IntegratedCommandMissionModule",
    "IntegratedCommandExecution",
    "PredictiveIntegratedCommandMissionModule",
]

_OPTIMIZATION_EXPORTS = [
    "ParameterBound",
    "OptimizationResult",
    "PSOConfig",
    "ParticleSwarmOptimizer",
    "ControllerAlgorithm",
    "AttitudeTuneCase",
    "TuneCaseResult",
    "GainTuningResult",
    "default_case_cost",
    "default_parameter_bounds",
    "preset_tuning_cases",
    "tune_controller_gains",
]

_KNOWLEDGE_EXPORTS = [
    "KnowledgeConditionConfig",
    "KnowledgeNoiseConfig",
    "KnowledgeEKFConfig",
    "TrackedObjectConfig",
    "ObjectKnowledgeBase",
]

_SENSOR_EXPORTS = [
    "NoisyOwnStateSensor",
    "JointStateSensor",
    "SensorNoiseConfig",
    "OwnStateSensor",
    "RelativeSensor",
    "CompositeSensorModel",
    "AccessConfig",
    "AccessModel",
    "GroundSite",
]

_UTIL_EXPORTS = [
    "ground_track_from_eci_history",
    "split_ground_track_dateline",
    "plot_ground_track",
    "plot_quaternion_components",
    "plot_body_rates",
    "plot_trajectory_frame",
    "plot_multi_trajectory_frame",
    "plot_ric_2d_projections",
    "plot_multi_ric_2d_projections",
    "plot_control_commands",
    "plot_multi_control_commands",
    "animate_rectangular_prism_attitude",
    "animate_multi_rectangular_prism_ric_curv",
    "animate_trajectory_frame",
    "animate_ground_track",
    "animate_multi_ground_track",
]

_ROCKET_EXPORTS = [
    "GuidanceCommand",
    "RocketAeroConfig",
    "RocketGuidanceLaw",
    "RocketSimConfig",
    "RocketVehicleConfig",
    "RocketState",
    "RocketSimResult",
    "RocketAscentSimulator",
    "OpenLoopPitchProgramGuidance",
    "MaxQThrottleLimiterGuidance",
    "HoldAttitudeGuidance",
]

_REVIEW_EXPORTS = [
    "EvidencePlotter",
    "ReviewQueryError",
    "ReviewQueryResult",
    "ReviewStoreNotFoundError",
    "ReviewWorkspace",
]

_INGESTION_EXPORTS = [
    "MissionInputPacket",
    "build_basic_propagation_scenario",
    "build_basic_rpo_scenario",
    "ingest_coes",
    "ingest_ephemeris_object_set",
    "ingest_ephemeris_samples",
    "ingest_relative_ric_state",
    "ingest_satellite_card",
    "ingest_state_vector",
    "ingest_tle",
    "inspect_packet",
    "load_packet",
    "merge_packets",
    "packet_from_dict",
    "render_ingestion_summary",
]

_OBSERVATION_EXPORTS = [
    "ObservationPacket",
    "fit_state_from_position_observations",
    "ingest_observations",
    "inspect_observation_packet",
    "kalman_filter_position_observations",
    "load_observation_packet",
    "observation_packet_from_dict",
    "sample_sgp4_observations_from_tle",
]

__all__ = [
    *_ACTUATOR_EXPORTS,
    *_ACTUATOR_PRESET_EXPORTS,
    *_CONTROL_EXPORTS,
    *_API_EXPORTS,
    *_CONFIG_EXPORTS,
    *_CORE_EXPORTS,
    *_ATTITUDE_DYNAMICS_EXPORTS,
    *_DYNAMICS_EXPORTS,
    *_ORBIT_DYNAMICS_EXPORTS,
    *_ESTIMATION_EXPORTS,
    *_METRIC_EXPORTS,
    *_MISSION_EXPORTS,
    *_OPTIMIZATION_EXPORTS,
    *_KNOWLEDGE_EXPORTS,
    *_SENSOR_EXPORTS,
    *_UTIL_EXPORTS,
    "run_master_simulation",
    *_ROCKET_EXPORTS,
    *_REVIEW_EXPORTS,
    *_INGESTION_EXPORTS,
    *_OBSERVATION_EXPORTS,
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {}


def _register(names: list[str], module_name: str) -> None:
    for name in names:
        _LAZY_IMPORTS[name] = (module_name, name)


_register(_API_EXPORTS, "sim.api")
_register(_ACTUATOR_EXPORTS, "sim.actuators")
_register(_ACTUATOR_PRESET_EXPORTS, "sim.actuators.presets")
_register(_CONTROL_EXPORTS, "sim.control")
_register(_CONFIG_EXPORTS, "sim.config")
_register(_CORE_EXPORTS, "sim.core.models")
_register(_DYNAMICS_EXPORTS, "sim.dynamics.model")
_register(_ATTITUDE_DYNAMICS_EXPORTS, "sim.dynamics.attitude")
_register(_ORBIT_DYNAMICS_EXPORTS, "sim.dynamics.orbit")
_register(_ESTIMATION_EXPORTS, "sim.estimation")
_register(_METRIC_EXPORTS, "sim.metrics")
_register(_MISSION_EXPORTS, "sim.mission")
_register(_OPTIMIZATION_EXPORTS, "sim.optimization")
_register(_KNOWLEDGE_EXPORTS, "sim.knowledge")
_register(_SENSOR_EXPORTS, "sim.sensors")
_register(_UTIL_EXPORTS, "sim.utils")
_register(_ROCKET_EXPORTS, "sim.rocket")
_register(_REVIEW_EXPORTS, "sim.review")
_register(_INGESTION_EXPORTS, "sim.ingestion")
_register(_OBSERVATION_EXPORTS, "sim.observations")
_LAZY_IMPORTS["run_master_simulation"] = ("sim.master_simulator", "run_master_simulation")


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'sim' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
