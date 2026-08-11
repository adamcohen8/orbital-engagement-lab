"""Versioned use-case profiles for complete OEL flight-software stacks.

Profiles are deliberately thinner than stack implementations.  They select a
complete stack, physical hardware family, cadence, and a small set of coherent
defaults for a recognizable spacecraft use case.  A profile does not inherit
the maturity of its component algorithms or its underlying stack; each profile
must eventually earn its own exact-version evidence.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from .catalog import resolve_stack, validate_stack_hardware, validate_stack_params
from .reference_stacks import STACK_VERSION, StackMaturity

PROFILE_CATALOG_VERSION = "1.0.0"
PROFILE_QUALIFICATION_GATES = (
    "closed-loop propagated-truth outcome",
    "sensor and navigation envelope",
    "actuator command receipt and physical realization",
    "mode transition and recovery behavior",
    "deterministic snapshot and replay",
    "off-nominal and fault response",
    "Monte Carlo robustness",
    "documented assumptions, limits, and tuning envelope",
)


@dataclass(frozen=True, slots=True)
class ProfileParameterRequirement:
    name: str
    summary: str
    units: str
    example: object

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "summary": self.summary,
            "units": self.units,
            "example": deepcopy(self.example),
        }


@dataclass(frozen=True, slots=True)
class FlightSoftwareUseCaseProfile:
    profile_id: str
    display_name: str
    domain: str
    summary: str
    stack_id: str
    default_hardware_profile: str
    compatible_hardware_profiles: tuple[str, ...]
    default_task_period_s: float
    capabilities: tuple[str, ...]
    default_params: tuple[tuple[str, object], ...] = ()
    qualified_optional_params: tuple[tuple[str, object], ...] = ()
    qualified_parameter_choices: tuple[tuple[str, tuple[object, ...]], ...] = ()
    mission_parameters: tuple[str, ...] = ()
    required_parameters: tuple[ProfileParameterRequirement, ...] = ()
    assumptions: tuple[str, ...] = ()
    known_limits: tuple[str, ...] = ()
    example_configs: tuple[str, ...] = ()
    maturity: StackMaturity = StackMaturity.EXPERIMENTAL
    qualification_status: str = "unqualified"
    catalog_version: str = PROFILE_CATALOG_VERSION
    stack_version: str = STACK_VERSION
    qualification_gates: tuple[str, ...] = PROFILE_QUALIFICATION_GATES

    @property
    def task_period_envelope_s(self) -> tuple[float, float]:
        return {
            "baseline": (0.1, 10.0),
            "attitude": (0.05, 0.2),
            "orbit": (0.25, 1.0),
            "rpo": (0.1, 0.5),
            "low_thrust": (1.0, 20.0),
        }[self.domain]

    def params_dict(self) -> dict[str, object]:
        return {name: deepcopy(value) for name, value in self.default_params}

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "domain": self.domain,
            "summary": self.summary,
            "stack_id": self.stack_id,
            "stack_version": self.stack_version,
            "default_hardware_profile": self.default_hardware_profile,
            "compatible_hardware_profiles": list(self.compatible_hardware_profiles),
            "default_task_period_s": self.default_task_period_s,
            "task_period_envelope_s": list(self.task_period_envelope_s),
            "capabilities": list(self.capabilities),
            "default_params": self.params_dict(),
            "qualified_optional_params": {
                name: deepcopy(value) for name, value in self.qualified_optional_params
            },
            "qualified_parameter_choices": {
                name: [deepcopy(value) for value in choices]
                for name, choices in self.qualified_parameter_choices
            },
            "mission_parameters": list(self.mission_parameters),
            "required_parameters": [item.to_dict() for item in self.required_parameters],
            "assumptions": list(self.assumptions),
            "known_limits": list(self.known_limits),
            "example_configs": list(self.example_configs),
            "maturity": self.maturity.value,
            "qualification_status": self.qualification_status,
            "catalog_version": self.catalog_version,
            "qualification_gates": list(self.qualification_gates),
        }


@dataclass(frozen=True, slots=True)
class MaterializedFlightSoftwareProfile:
    profile_id: str
    stack_id: str
    hardware_profile: str
    task_period_s: float
    params: dict[str, object]

    def to_config(self) -> dict[str, object]:
        return {
            "profile": self.profile_id,
            "stack": self.stack_id,
            "hardware_profile": self.hardware_profile,
            "task_period_s": self.task_period_s,
            "params": deepcopy(self.params),
        }


def _required(name: str, summary: str, units: str, example: object) -> ProfileParameterRequirement:
    return ProfileParameterRequirement(name, summary, units, example)


def _profile(
    profile_id: str,
    display_name: str,
    domain: str,
    summary: str,
    stack_id: str,
    hardware_profile: str,
    task_period_s: float,
    capabilities: tuple[str, ...],
    *,
    compatible_hardware_profiles: tuple[str, ...] | None = None,
    default_params: Mapping[str, object] | None = None,
    qualified_optional_params: Mapping[str, object] | None = None,
    qualified_parameter_choices: Mapping[str, tuple[object, ...]] | None = None,
    mission_parameters: tuple[str, ...] = (),
    required_parameters: tuple[ProfileParameterRequirement, ...] = (),
    assumptions: tuple[str, ...] = (),
    known_limits: tuple[str, ...] = (),
    example_configs: tuple[str, ...] = (),
    maturity: StackMaturity = StackMaturity.EXPERIMENTAL,
    qualification_status: str = "unqualified",
) -> FlightSoftwareUseCaseProfile:
    return FlightSoftwareUseCaseProfile(
        profile_id=profile_id,
        display_name=display_name,
        domain=domain,
        summary=summary,
        stack_id=stack_id,
        default_hardware_profile=hardware_profile,
        compatible_hardware_profiles=compatible_hardware_profiles or (hardware_profile,),
        default_task_period_s=task_period_s,
        capabilities=capabilities,
        default_params=tuple((str(name), deepcopy(value)) for name, value in (default_params or {}).items()),
        qualified_optional_params=tuple(
            (str(name), deepcopy(value)) for name, value in (qualified_optional_params or {}).items()
        ),
        qualified_parameter_choices=tuple(
            (str(name), tuple(deepcopy(value) for value in choices))
            for name, choices in (qualified_parameter_choices or {}).items()
        ),
        mission_parameters=tuple(str(name) for name in mission_parameters),
        required_parameters=required_parameters,
        assumptions=assumptions,
        known_limits=known_limits,
        example_configs=example_configs,
        maturity=maturity,
        qualification_status=qualification_status,
    )


_ATTITUDE_ASSUMPTIONS = (
    "Rigid-body attitude dynamics and center-of-mass-referenced inertia are configured.",
    "Sensor and actuator limits must be matched to the simulated spacecraft before qualification.",
)
_RPO_ASSUMPTIONS = (
    "The reference object is observable through the declared onboard sensing path.",
    "Reference-stack relative guidance is currently bounded to near-Earth, near-circular RIC use.",
)


USE_CASE_PROFILES: tuple[FlightSoftwareUseCaseProfile, ...] = (
    _profile(
        "fsw.profile.coast_monitor.v1",
        "Coast and Input Monitor",
        "baseline",
        "Passive propagation with typed input, timing, snapshot, and review evidence.",
        "fsw.passive",
        "hardware.passive.v1",
        1.0,
        ("coast", "typed-input-evidence"),
        default_params={"measurement_stale_after_s": 30.0},
        known_limits=("Produces no actuator commands or autonomous recovery behavior.",),
        example_configs=("configs/qualification_fsw_coast_monitor.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.adcs_commissioning.v1",
        "ADCS Commissioning",
        "attitude",
        "Qualified rate damping, coarse-Sun recovery, nominal Sun pointing, and wheel momentum management.",
        "fsw.attitude_reference",
        "hardware.reaction_wheels_magnetorquer.v1",
        0.1,
        ("detumble", "coarse-sun", "sun-pointing", "momentum-unloading", "fdir"),
        default_params={
            "reference_mode": "sun",
            "navigation_initialization": "cold",
            "kp": 0.20,
            "kd": 0.65,
            "max_torque_n_m": 0.08,
            "wheel_max_torque_n_m": (0.08,),
            "wheel_max_momentum_n_m_s": (6.0,),
            "detumble_entry_rate_rad_s": 0.25,
            "detumble_exit_rate_rad_s": 0.02,
        },
        assumptions=_ATTITUDE_ASSUMPTIONS,
        known_limits=("Electrical energy and magnetic-geometry generality require mission-specific models.",),
        example_configs=("configs/qualification_fsw_adcs_commissioning.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.adcs_nadir_payload.v1",
        "Nadir Payload Pointing",
        "attitude",
        "Nadir reference tracking with reaction-wheel control and magnetorquer momentum unloading.",
        "fsw.attitude_reference",
        "hardware.reaction_wheels_magnetorquer.v1",
        0.1,
        ("nadir-pointing", "detumble", "momentum-unloading", "fdir"),
        default_params={
            "reference_mode": "nadir",
            "kp": 0.20,
            "kd": 0.65,
            "max_torque_n_m": 0.08,
            "wheel_max_torque_n_m": (0.08,),
            "wheel_max_momentum_n_m_s": (3.0,),
        },
        assumptions=_ATTITUDE_ASSUMPTIONS,
        known_limits=("Payload jitter, flexible modes, and line-of-sight keep-outs are not yet profiled.",),
        example_configs=("configs/qualification_fsw_adcs_nadir_payload.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.adcs_sun_pointing.v1",
        "Sun Pointing",
        "attitude",
        "Sun-vector pointing with physical ADCS allocation options.",
        "fsw.attitude_reference",
        "hardware.reaction_wheels_magnetorquer.v1",
        0.1,
        ("sun-pointing", "detumble", "momentum-unloading", "fdir"),
        default_params={
            "reference_mode": "sun",
            "kp": 0.20,
            "kd": 0.65,
            "max_torque_n_m": 0.08,
            "wheel_max_torque_n_m": (0.08,),
            "wheel_max_momentum_n_m_s": (3.0,),
        },
        assumptions=_ATTITUDE_ASSUMPTIONS,
        known_limits=("Power-positive behavior requires a spacecraft power model and is not part of this profile claim.",),
        example_configs=("configs/qualification_fsw_adcs_sun_pointing.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.adcs_target_tracking.v1",
        "Inertial Target Tracking",
        "attitude",
        "Track a supplied inertial target position with a declared body boresight.",
        "fsw.attitude_reference",
        "hardware.reaction_wheels_magnetorquer.v1",
        0.1,
        ("target-pointing", "detumble", "momentum-unloading", "fdir"),
        default_params={
            "reference_mode": "target",
            "boresight_body": (1.0, 0.0, 0.0),
            "kp": 0.20,
            "kd": 0.65,
            "max_torque_n_m": 0.08,
            "wheel_max_torque_n_m": (0.08,),
            "wheel_max_momentum_n_m_s": (3.0,),
        },
        required_parameters=(
            _required("target_position_eci_m", "Target position in the inertial frame.", "m", (7.0e6, 1.0e6, 0.0)),
        ),
        assumptions=_ATTITUDE_ASSUMPTIONS,
        known_limits=("Autonomous optical acquisition and target-motion prediction remain outside the declared envelope.",),
        example_configs=("configs/qualification_fsw_adcs_target_tracking.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.orbit_maneuver_execution.v1",
        "Finite Maneuver Execution",
        "orbit",
        "Execute a supplied schedule of finite burns through the typed actuator boundary.",
        "fsw.orbit_reference",
        "hardware.ideal_wrench.v1",
        0.5,
        ("scheduled-burn", "command-receipts", "resource-gating", "fdir"),
        compatible_hardware_profiles=(
            "hardware.ideal_wrench.v1",
            "hardware.rcs.v1",
            "hardware.continuous_engine.v1",
        ),
        default_params={"translation_mode": "scheduled_burn", "max_acceleration_m_s2": 0.02},
        qualified_optional_params={
            "attitude_reference_mode": "quaternion",
            "attitude_boresight_body": (1.0, 0.0, 0.0),
            "max_attitude_torque_n_m": 0.08,
            "attitude_kp": (0.20, 0.20, 0.20),
            "attitude_kd": (0.70, 0.70, 0.70),
            "pointing_tolerance_rad": 0.08726646259971647,
            "require_pointing_for_translation": True,
        },
        qualified_parameter_choices={"max_acceleration_m_s2": (0.005, 0.02)},
        required_parameters=(
            _required(
                "scheduled_burns",
                "Finite burn schedule with start time, duration, delta-v, and frame.",
                "structured SI",
                ({"start_time_s": 10.0, "duration_s": 5.0, "delta_v_m_s": (0.1, 0.0, 0.0), "frame": "eci"},),
            ),
        ),
        known_limits=(
            "This profile executes a supplied immutable plan; dynamic cancellation and general maneuver optimization are outside the v1 Supported envelope.",
        ),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.leo_stationkeeping.v1",
        "LEO State Stationkeeping",
        "orbit",
        "Track a supplied absolute ECI position and velocity reference.",
        "fsw.orbit_reference",
        "hardware.ideal_wrench.v1",
        1.0,
        ("stationkeeping", "operational-navigation", "resource-gating", "fdir"),
        default_params={
            "translation_mode": "stationkeeping",
            "kp_position_s2": 4.0e-5,
            "kd_velocity_s_inv": 1.3e-2,
            "max_acceleration_m_s2": 0.01,
        },
        required_parameters=(
            _required("target_state_eci_m_m_s", "Target ECI position and velocity.", "m and m/s", (7.0e6, 0.0, 0.0, 0.0, 7546.0, 0.0)),
        ),
        known_limits=("Ground-track, local-time, and drag-makeup policies are not yet specialized.",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.orbital_element_maintenance.v1",
        "Orbital Element Maintenance",
        "orbit",
        "Regulate selected classical orbital elements using the reference element-feedback law.",
        "fsw.orbit_reference",
        "hardware.continuous_engine.v1",
        1.0,
        ("orbital-elements", "continuous-thrust", "resource-gating", "fdir"),
        compatible_hardware_profiles=("hardware.continuous_engine.v1", "hardware.ideal_wrench.v1"),
        default_params={
            "translation_mode": "orbital_elements",
            "controlled_elements": ("a", "ecc"),
            "max_acceleration_m_s2": 0.002,
        },
        qualified_parameter_choices={"max_acceleration_m_s2": (0.001, 0.002)},
        mission_parameters=("target_semi_major_axis_m", "target_eccentricity"),
        required_parameters=(
            _required(
                "target_coes",
                "Target classical orbital-element values keyed by element name.",
                "km/rad",
                {"a_km": 7000.0, "ecc": 0.001},
            ),
        ),
        known_limits=("Coupled long-horizon element maintenance and perturbation-specific tuning remain unqualified.",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.atmospheric_pass_recovery.v1",
        "Atmospheric Pass Recovery",
        "orbit",
        "State-driven atmospheric-pass detection followed by a bounded prograde recovery burn.",
        "fsw.orbit_reference",
        "hardware.continuous_engine.v1",
        0.5,
        ("atmospheric-pass", "continuous-thrust", "resource-gating", "fdir"),
        default_params={"translation_mode": "atmospheric_pass", "max_acceleration_m_s2": 0.002},
        required_parameters=(
            _required("pass_entry_altitude_m", "Altitude at or below which pass entry is recognized.", "m", 180000.0),
            _required("pass_exit_altitude_m", "Ascending altitude at or above which pass exit is recognized.", "m", 190000.0),
            _required("recovery_delta_v_m_s", "Prograde delta-v delivered after pass exit.", "m/s", 0.1),
            _required("prograde_acceleration_m_s2", "Requested recovery acceleration.", "m/s^2", 0.001),
        ),
        known_limits=("Pass detection uses onboard altitude thresholds; atmospheric trajectory optimization is not included.",),
        example_configs=("configs/aero_assisted_plane_change_demo.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.rpo_far_field_rendezvous.v1",
        "Far-Field RIC Rendezvous",
        "rpo",
        "Planned transfer, coast, correction, braking, and terminal-cleanup guidance toward an RIC target.",
        "fsw.rpo_reference",
        "hardware.ideal_wrench.v1",
        0.1,
        ("planned-rendezvous-transfer", "operational-navigation", "resource-gating", "fdir"),
        compatible_hardware_profiles=(
            "hardware.ideal_wrench.v1",
            "hardware.rcs.v1",
            "hardware.continuous_engine.v1",
        ),
        default_params={
            "translation_mode": "ric_pd_transfer",
            "target_relative_state_ric_m": (0.0,) * 6,
            "transfer_time_s": 300.0,
            "final_brake_start_s": 60.0,
            "terminal_start_s": 240.0,
            "terminal_range_m": 150.0,
            "max_acceleration_m_s2": 0.01,
        },
        required_parameters=(
            _required("reference_object_id", "Chief/target object identifier.", "identifier", "target"),
        ),
        assumptions=_RPO_ASSUMPTIONS,
        known_limits=("The flagship evidence does not automatically qualify other initial separations or hardware.",),
        example_configs=("configs/ric_pd_10km_experiment.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.rpo_formation_hold.v1",
        "Relative Formation Hold",
        "rpo",
        "Maintain a supplied rectangular-RIC relative position and velocity.",
        "fsw.rpo_reference",
        "hardware.ideal_wrench.v1",
        0.5,
        ("ric-hold", "formation-flying", "operational-navigation", "fdir"),
        default_params={
            "translation_mode": "ric_hold",
            "kp_position_s2": 4.0e-5,
            "kd_velocity_s_inv": 0.013,
            "max_acceleration_m_s2": 0.01,
        },
        qualified_optional_params={"mean_motion_rad_s": 0.0010780076},
        required_parameters=(
            _required("reference_object_id", "Chief object identifier.", "identifier", "target"),
            _required("target_relative_state_ric_m", "Desired rectangular-RIC position and velocity.", "m and m/s", (0.0, 500.0, 0.0, 0.0, 0.0, 0.0)),
        ),
        assumptions=_RPO_ASSUMPTIONS,
        known_limits=("Distributed coordination and crosslink consensus are not included.",),
        example_configs=("configs/validation_fsw_rpo_nonzero_hold.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.rpo_corridor_approach.v1",
        "V-Bar Corridor Approach",
        "rpo",
        "Rate-limited V-bar approach with terminal slowdown.",
        "fsw.rpo_reference",
        "hardware.rcs.v1",
        0.1,
        ("approach", "terminal-slowdown", "operational-navigation", "fdir"),
        compatible_hardware_profiles=("hardware.rcs.v1", "hardware.ideal_wrench.v1"),
        default_params={
            "translation_mode": "v_bar_approach",
            "approach_speed_m_s": 0.5,
            "slowdown_distance_m": 250.0,
            "max_acceleration_m_s2": 0.01,
            "kp_position_s2": 4.0e-5,
            "kd_velocity_s_inv": 0.013,
        },
        required_parameters=(
            _required("reference_object_id", "Target object identifier.", "identifier", "target"),
        ),
        assumptions=_RPO_ASSUMPTIONS,
        known_limits=("Corridor geometry is currently represented by axis guidance rather than a full approach polytope.",),
        example_configs=("configs/reference_gnc_vbar_approach.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.rpo_waypoint_inspection.v1",
        "Waypoint Inspection",
        "rpo",
        "Follow a supplied sequence of tolerance-gated rectangular-RIC waypoints.",
        "fsw.rpo_reference",
        "hardware.rcs.v1",
        0.1,
        ("waypoint", "inspection", "operational-navigation", "fdir"),
        compatible_hardware_profiles=("hardware.rcs.v1", "hardware.ideal_wrench.v1"),
        default_params={
            "translation_mode": "waypoint",
            "position_tolerance_m": 25.0,
            "velocity_tolerance_m_s": 0.1,
            "max_acceleration_m_s2": 0.01,
            "kp_position_s2": 4.0e-5,
            "kd_velocity_s_inv": 0.013,
        },
        qualified_optional_params={
            "recovery_clear_dwell_s": 2.0,
            "recovery_constraint_kinds": ("mission_safety_envelope",),
            "recovery_mode": "passive_retreat",
            "retreat_coast_range_m": 150.0,
            "retreat_speed_m_s": 1.0,
        },
        qualified_parameter_choices={
            "position_tolerance_m": (25.0, 30.0),
            "velocity_tolerance_m_s": (0.1, 0.5),
        },
        mission_parameters=("constraints",),
        required_parameters=(
            _required("reference_object_id", "Inspection target object identifier.", "identifier", "target"),
            _required("waypoints_ric", "Ordered rectangular-RIC waypoint states.", "m and m/s", ((0.0, 500.0, 0.0, 0.0, 0.0, 0.0),)),
        ),
        assumptions=_RPO_ASSUMPTIONS,
        known_limits=("Surface coverage, lighting, payload collection, and flyaround optimization are not yet composed.",),
        example_configs=("configs/validation_fsw_rpo_nonzero_waypoint.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.rpo_terminal_proximity.v1",
        "Terminal Proximity Braking",
        "rpo",
        "Limit closing rate and settle the relative state inside the terminal region.",
        "fsw.rpo_reference",
        "hardware.rcs.v1",
        0.1,
        ("terminal-braking", "operational-navigation", "resource-gating", "fdir"),
        compatible_hardware_profiles=("hardware.rcs.v1", "hardware.ideal_wrench.v1"),
        default_params={
            "translation_mode": "terminal_braking",
            "terminal_box_m": 25.0,
            "terminal_max_closing_speed_m_s": 0.05,
            "max_acceleration_m_s2": 0.01,
            "kp_position_s2": 4.0e-5,
            "kd_velocity_s_inv": 0.013,
        },
        required_parameters=(
            _required("reference_object_id", "Target object identifier.", "identifier", "target"),
        ),
        assumptions=_RPO_ASSUMPTIONS,
        known_limits=("Contact dynamics, docking sensors, capture logic, and docking certification are not included.",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.rpo_passive_retreat.v1",
        "Passive-Safe Retreat",
        "rpo",
        "Acquire an outward drift rate and then coast away from the reference object.",
        "fsw.rpo_reference",
        "hardware.rcs.v1",
        0.1,
        ("passive-retreat", "recovery", "operational-navigation", "fdir"),
        compatible_hardware_profiles=("hardware.rcs.v1", "hardware.ideal_wrench.v1"),
        default_params={
            "translation_mode": "passive_retreat",
            "retreat_speed_m_s": 1.0,
            "retreat_coast_range_m": 500.0,
            "max_acceleration_m_s2": 0.01,
        },
        required_parameters=(
            _required("reference_object_id", "Object from which to retreat.", "identifier", "target"),
        ),
        assumptions=_RPO_ASSUMPTIONS,
        known_limits=("Retreat direction and keep-out geometry require mission-specific review.",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.rpo_conjunction_response.v1",
        "Relative Conjunction Response",
        "rpo",
        "Monitor a relative keep-out prediction and request an avoidance maneuver before resuming hold behavior.",
        "fsw.rpo_reference",
        "hardware.rcs.v1",
        0.5,
        ("conjunction-avoidance", "ric-hold", "recovery", "operational-navigation", "fdir"),
        compatible_hardware_profiles=("hardware.rcs.v1", "hardware.ideal_wrench.v1"),
        default_params={
            "translation_mode": "ric_hold",
            "conjunction_avoidance_enabled": True,
            "conjunction_keep_out_radius_m": 100.0,
            "conjunction_prediction_horizon_s": 300.0,
            "conjunction_avoidance_delta_v_m_s": 1.0,
            "conjunction_maneuver_lead_time_s": 0.1,
            "max_acceleration_m_s2": 0.02,
        },
        qualified_optional_params={
            "kp_position_s2": 4.0e-5,
            "kd_velocity_s_inv": 0.013,
            "recovery_clear_dwell_s": 2.0,
            "recovery_constraint_kinds": ("mission_safety_envelope",),
            "recovery_mode": "passive_retreat",
        },
        mission_parameters=("target_relative_state_ric_m", "constraints"),
        required_parameters=(
            _required("reference_object_id", "Screened relative object identifier.", "identifier", "target"),
        ),
        assumptions=_RPO_ASSUMPTIONS,
        known_limits=("This is a local relative-response prototype, not operational conjunction assessment.",),
        example_configs=("configs/validation_fsw_conjunction_avoidance.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.low_thrust_phasing.v1",
        "Low-Thrust Relative Phasing",
        "low_thrust",
        "Apply low-authority continuous thrust to reduce relative in-track phase error.",
        "fsw.low_thrust_reference",
        "hardware.continuous_engine.v1",
        1.0,
        ("low-thrust-phasing", "continuous-thrust", "operational-navigation", "thrust-windowing", "resource-gating"),
        default_params={
            "translation_mode": "low_thrust_phasing",
            "kp_position_s2": 1.0e-8,
            "kd_velocity_s_inv": 2.0e-4,
            "max_acceleration_m_s2": 2.0e-5,
            "thrust_command_deadband_m_s2": 1.0e-9,
        },
        qualified_optional_params={
            "target_relative_state_ric_m": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "mean_motion_rad_s": 0.0010780076,
            "prograde_acceleration_m_s2": 1.0e-4,
            "gimbal_limit_rad": 3.141592653589793,
            "position_tolerance_m": 250.0,
            "velocity_tolerance_m_s": 0.05,
        },
        qualified_parameter_choices={
            "max_acceleration_m_s2": (2.0e-5, 1.0e-4, 1.0e-3),
            "max_force_n": (0.002, 0.01),
            "thrust_window_period_s": (120.0, 600.0),
            "thrust_window_duration_s": (60.0, 300.0),
            "thrust_window_phase_s": (0.0, 100.0, 200.0),
        },
        required_parameters=(
            _required("reference_object_id", "Chief object identifier.", "identifier", "target"),
        ),
        assumptions=_RPO_ASSUMPTIONS,
        known_limits=(
            "The Supported envelope is near-circular relative phasing, not general low-thrust trajectory optimization.",
            "Periodic thrust windows and typed resource inhibition are supported; automatic eclipse-to-power generation and thermal scheduling are not included.",
        ),
        example_configs=("configs/qualification_fsw_low_thrust_family.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
    _profile(
        "fsw.profile.low_thrust_element_maintenance.v1",
        "Low-Thrust Element Maintenance",
        "low_thrust",
        "Use continuous thrust to regulate selected classical orbital elements.",
        "fsw.low_thrust_reference",
        "hardware.continuous_engine.v1",
        1.0,
        ("orbital-elements", "continuous-thrust", "resource-gating"),
        default_params={
            "translation_mode": "orbital_elements",
            "controlled_elements": ("a", "ecc"),
            "orbital_element_control_law": "energy_eccentricity",
            "max_acceleration_m_s2": 2.0e-5,
            "thrust_command_deadband_m_s2": 1.0e-9,
            "element_averaging_window_s": 600.0,
            "kd_velocity_s_inv": 5.0e-5,
        },
        qualified_optional_params={
            "gimbal_limit_rad": 3.141592653589793,
            "position_tolerance_m": 250.0,
            "eccentricity_tolerance": 5.0e-4,
        },
        qualified_parameter_choices={
            "max_acceleration_m_s2": (2.0e-5, 1.0e-4),
            "max_force_n": (0.002, 0.01),
            "thrust_window_period_s": (120.0, 600.0),
            "thrust_window_duration_s": (60.0, 300.0),
            "element_averaging_window_s": (120.0, 300.0, 600.0, 900.0),
        },
        required_parameters=(
            _required("target_coes", "Target classical orbital-element values keyed by element name.", "SI/rad", {"a": 7.0e6, "ecc": 0.001}),
        ),
        assumptions=("Qualification is restricted to the checked-in near-Earth ONP J2-plus-drag envelope.",),
        known_limits=(
            "The Supported claim covers bounded a/ecc maintenance, not arbitrary orbit raising or optimized element transfers.",
            "Periodic thrust windows and restart are modeled; mission-specific eclipse, power, thermal, and gimbal policies remain stack configuration responsibilities.",
        ),
        example_configs=("configs/qualification_fsw_low_thrust_family.yaml",),
        maturity=StackMaturity.SUPPORTED,
        qualification_status="supported",
    ),
)

_BY_ID = {item.profile_id: item for item in USE_CASE_PROFILES}


def use_case_profiles(*, domain: str | None = None) -> tuple[FlightSoftwareUseCaseProfile, ...]:
    if domain is None:
        return USE_CASE_PROFILES
    token = str(domain).strip().lower()
    return tuple(item for item in USE_CASE_PROFILES if item.domain == token)


def resolve_use_case_profile(profile_id: str) -> FlightSoftwareUseCaseProfile:
    try:
        return _BY_ID[str(profile_id)]
    except KeyError as exc:
        choices = ", ".join(sorted(_BY_ID))
        raise ValueError(f"Unknown flight-software profile {profile_id!r}; choose one of: {choices}.") from exc


def materialize_use_case_profile(
    profile_id: str,
    *,
    params: Mapping[str, object] | None = None,
    hardware_profile: str | None = None,
    task_period_s: float | None = None,
) -> MaterializedFlightSoftwareProfile:
    profile = resolve_use_case_profile(profile_id)
    supplied = {str(name): deepcopy(value) for name, value in dict(params or {}).items()}
    resolved_params = profile.params_dict()
    required_names = {item.name for item in profile.required_parameters}
    optional_params = dict(profile.qualified_optional_params)
    parameter_choices = dict(profile.qualified_parameter_choices)
    allowed_names = (
        set(resolved_params)
        | set(optional_params)
        | set(parameter_choices)
        | set(profile.mission_parameters)
        | required_names
    )
    undeclared = sorted(set(supplied) - allowed_names)
    if undeclared:
        raise ValueError(
            f"flight_software profile {profile.profile_id!r} does not qualify params: "
            + ", ".join(undeclared)
            + "; select an unqualified explicit stack configuration for custom tuning."
        )
    for parameter_name, qualified_value in resolved_params.items():
        if (
            parameter_name in supplied
            and parameter_name not in required_names
            and parameter_name not in parameter_choices
            and not _profile_values_equal(supplied[parameter_name], qualified_value)
        ):
            raise ValueError(
                f"flight_software profile {profile.profile_id!r} fixes "
                f"params.{parameter_name}={qualified_value!r}; select a different profile or "
                "an unqualified explicit stack configuration to change that parameter."
            )
    for parameter_name, qualified_value in optional_params.items():
        if parameter_name in supplied and not _profile_values_equal(
            supplied[parameter_name], qualified_value
        ):
            raise ValueError(
                f"flight_software profile {profile.profile_id!r} fixes optional "
                f"params.{parameter_name}={qualified_value!r}; select an unqualified explicit "
                "stack configuration to change that parameter."
            )
    for parameter_name, choices in parameter_choices.items():
        if parameter_name in supplied and not any(
            _profile_values_equal(supplied[parameter_name], choice) for choice in choices
        ):
            raise ValueError(
                f"flight_software profile {profile.profile_id!r} restricts "
                f"params.{parameter_name} to {choices!r}; select an unqualified explicit stack "
                "configuration for other values."
            )
    resolved_params.update(supplied)
    missing = [
        item.name
        for item in profile.required_parameters
        if not _has_required_parameter_value(resolved_params, item.name)
    ]
    if missing:
        raise ValueError(
            f"flight_software profile {profile.profile_id!r} requires params: {', '.join(missing)}."
        )
    resolved_hardware = str(hardware_profile or profile.default_hardware_profile)
    if resolved_hardware not in profile.compatible_hardware_profiles:
        choices = ", ".join(profile.compatible_hardware_profiles)
        raise ValueError(
            f"Flight-software profile {profile.profile_id!r} does not declare hardware profile "
            f"{resolved_hardware!r}; declared profiles: {choices}."
        )
    resolved_period = profile.default_task_period_s if task_period_s is None else float(task_period_s)
    if not isfinite(resolved_period) or resolved_period <= 0.0:
        raise ValueError("flight_software.task_period_s must be finite and positive.")
    period_min, period_max = profile.task_period_envelope_s
    if not period_min <= resolved_period <= period_max:
        raise ValueError(
            f"flight_software profile {profile.profile_id!r} task_period_s must remain within "
            f"its qualified envelope [{period_min:g}, {period_max:g}] s."
        )
    validate_stack_hardware(profile.stack_id, resolved_hardware)
    validate_stack_params(profile.stack_id, resolved_params)
    return MaterializedFlightSoftwareProfile(
        profile.profile_id,
        profile.stack_id,
        resolved_hardware,
        resolved_period,
        resolved_params,
    )


def _profile_values_equal(left: object, right: object) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _profile_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _profile_values_equal(left[key], right[key]) for key in left
        )
    return left == right


def _has_required_parameter_value(params: Mapping[str, object], name: str) -> bool:
    if name not in params:
        return False
    value = params[name]
    if value is None or value == "":
        return False
    if isinstance(value, (Mapping, list, tuple, set, frozenset)) and not value:
        return False
    return True


def validate_use_case_profile_catalog() -> tuple[str, ...]:
    errors: list[str] = []
    pattern = re.compile(r"^fsw\.profile\.[a-z0-9_]+\.v[1-9][0-9]*$")
    seen: set[str] = set()
    for profile in USE_CASE_PROFILES:
        if profile.profile_id in seen:
            errors.append(f"duplicate profile id: {profile.profile_id}")
        seen.add(profile.profile_id)
        if pattern.fullmatch(profile.profile_id) is None:
            errors.append(f"invalid profile id: {profile.profile_id}")
        try:
            stack = resolve_stack(profile.stack_id)
        except ValueError as exc:
            errors.append(f"{profile.profile_id}: {exc}")
            continue
        if profile.stack_version != stack.version:
            errors.append(
                f"{profile.profile_id}: stack version {profile.stack_version} does not match {stack.version}"
            )
        expected_qualification_status = {
            StackMaturity.EXPERIMENTAL: "unqualified",
            StackMaturity.SUPPORTED: "supported",
            StackMaturity.REFERENCE: "reference",
        }.get(profile.maturity)
        if expected_qualification_status is None:
            errors.append(f"{profile.profile_id}: built-in profiles cannot use OEL Unrated maturity")
        elif profile.qualification_status != expected_qualification_status:
            errors.append(
                f"{profile.profile_id}: {profile.maturity.value} maturity requires "
                f"qualification_status={expected_qualification_status!r}"
            )
        if profile.default_hardware_profile not in profile.compatible_hardware_profiles:
            errors.append(f"{profile.profile_id}: default hardware is not in the profile compatibility set")
        for hardware in profile.compatible_hardware_profiles:
            try:
                validate_stack_hardware(profile.stack_id, hardware)
            except ValueError as exc:
                errors.append(f"{profile.profile_id}: {exc}")
        if not isfinite(profile.default_task_period_s) or profile.default_task_period_s <= 0.0:
            errors.append(f"{profile.profile_id}: default task period must be finite and positive")
        required_names = [item.name for item in profile.required_parameters]
        if len(required_names) != len(set(required_names)):
            errors.append(f"{profile.profile_id}: required parameter names must be unique")
        optional_names = [name for name, _value in profile.qualified_optional_params]
        choice_names = [name for name, _choices in profile.qualified_parameter_choices]
        declared_names = (
            list(profile.params_dict())
            + required_names
            + optional_names
            + choice_names
            + list(profile.mission_parameters)
        )
        allowed_choice_overlap = set(profile.params_dict()) & set(choice_names)
        duplicate_names = {
            name
            for name in declared_names
            if declared_names.count(name) > 1 and name not in allowed_choice_overlap
        }
        if duplicate_names:
            errors.append(
                f"{profile.profile_id}: parameter envelope names overlap: "
                + ", ".join(sorted(duplicate_names))
            )
        for name, choices in profile.qualified_parameter_choices:
            if not choices:
                errors.append(f"{profile.profile_id}: params.{name} has an empty qualified choice set")
        example_params = profile.params_dict()
        example_params.update(
            {name: deepcopy(value) for name, value in profile.qualified_optional_params}
        )
        for name, choices in profile.qualified_parameter_choices:
            if name not in example_params and choices:
                example_params[name] = deepcopy(choices[0])
        example_params.update({item.name: deepcopy(item.example) for item in profile.required_parameters})
        try:
            validate_stack_params(profile.stack_id, example_params)
        except ValueError as exc:
            errors.append(f"{profile.profile_id}: example selection is invalid: {exc}")
        for path in profile.example_configs:
            if not path.strip():
                errors.append(f"{profile.profile_id}: example config path must be non-empty")
    return tuple(errors)
