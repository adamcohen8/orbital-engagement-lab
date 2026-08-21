"""Independent discovery catalog for complete flight-software stacks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, sqrt

from .reference_stacks import BUILTIN_STACKS, STACK_VERSION, StackMaturity


@dataclass(frozen=True, slots=True)
class StackCatalogEntry:
    stack_id: str
    version: str
    maturity: StackMaturity
    summary: str
    compatible_hardware_profiles: tuple[str, ...]
    capabilities: tuple[str, ...]


_HARDWARE = {
    "fsw.passive": ("hardware.passive.v1", "game.ideal_wrench.v1"),
    "fsw.attitude_reference": (
        "hardware.ideal_wrench.v1",
        "hardware.reaction_wheels.v1",
        "hardware.reaction_wheels_magnetorquer.v1",
        "hardware.magnetorquer.v1",
        "hardware.cmg.v1",
    ),
    "fsw.orbit_reference": (
        "hardware.ideal_wrench.v1",
        "hardware.rcs.v1",
        "hardware.continuous_engine.v1",
    ),
    "fsw.rpo_reference": (
        "hardware.ideal_wrench.v1",
        "hardware.rcs.v1",
        "hardware.continuous_engine.v1",
    ),
    "fsw.low_thrust_reference": ("hardware.continuous_engine.v1",),
    "fsw.game_pilot_reference": (
        "game.ideal_wrench.v1",
        "game.reaction_wheel_engine.v1",
        "game.variable_geometry_aero.v1",
    ),
}

_CAPABILITIES = {
    "fsw.passive": ("coast", "typed-input-evidence"),
    "fsw.attitude_reference": (
        "detumble",
        "inertial-pointing",
        "moving-reference-pointing",
        "momentum-unloading",
        "fdir",
    ),
    "fsw.orbit_reference": (
        "stationkeeping",
        "orbital-elements",
        "atmospheric-pass",
        "operational-navigation",
        "fdir",
        "resources",
    ),
    "fsw.rpo_reference": (
        "ric-hold",
        "approach",
        "waypoint",
        "planned-rendezvous-transfer",
        "terminal-braking",
        "passive-retreat",
        "operational-navigation",
        "autonomous-maneuver-planning",
        "conjunction-avoidance",
        "fdir",
        "resources",
    ),
    "fsw.low_thrust_reference": ("low-thrust-phasing", "orbital-elements"),
    "fsw.game_pilot_reference": (
        "pilot-input",
        "operator-command",
        "translation",
        "attitude-thrust",
        "aerodynamic-effector",
    ),
}

STACK_CATALOG = tuple(
    StackCatalogEntry(
        item.stack_id,
        STACK_VERSION,
        StackMaturity.EXPERIMENTAL,
        item.summary,
        _HARDWARE[item.stack_id],
        _CAPABILITIES[item.stack_id],
    )
    for item in BUILTIN_STACKS
)


def stack_catalog() -> tuple[StackCatalogEntry, ...]:
    return STACK_CATALOG


def resolve_stack(stack_id: str) -> StackCatalogEntry:
    for entry in STACK_CATALOG:
        if entry.stack_id == stack_id:
            return entry
    choices = ", ".join(entry.stack_id for entry in STACK_CATALOG)
    raise ValueError(f"Unknown flight-software stack {stack_id!r}; choose one of: {choices}.")


def validate_stack_hardware(stack_id: str, hardware_profile: str | None) -> None:
    entry = resolve_stack(stack_id)
    if hardware_profile is None or hardware_profile in entry.compatible_hardware_profiles:
        return
    compatible = ", ".join(entry.compatible_hardware_profiles)
    raise ValueError(
        f"Flight-software stack {stack_id!r} is not compatible with hardware profile "
        f"{hardware_profile!r}; compatible profiles: {compatible}."
    )


_COMMON_PARAMS = {
    "emit_diagnostics",
    "navigation_initialization",
    "reference_object_id",
    "loaded_position_eci_m",
    "loaded_velocity_eci_m_s",
    "loaded_epoch_s",
    "fdir_rejection_limit",
    "fdir_saturation_limit",
    "fdir_isolate_on_saturation",
    "fdir_clear_dwell_s",
    "actuator_fallbacks",
}
_PASSIVE_PARAMS = _COMMON_PARAMS | {"measurement_stale_after_s"}
_ATTITUDE_PARAMS = _COMMON_PARAMS | {
    "measurement_stale_after_s",
    "reference_mode",
    "quaternion_bn",
    "ric_axis",
    "boresight_body",
    "target_position_eci_m",
    "thrust_direction_eci",
    "replay_times_s",
    "replay_quaternions_bn",
    "kp",
    "kd",
    "max_torque_n_m",
    "detumble_entry_rate_rad_s",
    "detumble_exit_rate_rad_s",
    "wheel_axes_body",
    "wheel_max_torque_n_m",
    "wheel_max_momentum_n_m_s",
    "wheel_initial_momentum_n_m_s",
    "max_dipole_a_m2",
    "magnetic_field_body_t",
    "cmg_momentum_n_m_s",
    "max_gimbal_rate_rad_s",
    "momentum_dump_start_fraction",
    "momentum_dump_stop_fraction",
    "momentum_dump_gain",
    "momentum_dump_max_dipole_a_m2",
}
_TRANSLATION_PARAMS = _COMMON_PARAMS | {
    "measurement_stale_after_s",
    "translation_mode",
    "max_acceleration_m_s2",
    "max_force_n",
    "assumed_mass_kg",
    "target_semi_major_axis_m",
    "target_state_eci_m_m_s",
    "target_eccentricity",
    "eccentricity_tolerance",
    "target_relative_state_ric_m",
    "waypoints_ric",
    "control_axis_mask",
    "kp_position_s2",
    "kd_velocity_s_inv",
    "mean_motion_rad_s",
    "approach_speed_m_s",
    "slowdown_distance_m",
    "terminal_box_m",
    "terminal_max_closing_speed_m_s",
    "retreat_speed_m_s",
    "retreat_coast_range_m",
    "position_tolerance_m",
    "velocity_tolerance_m_s",
    "scheduled_burns",
    "raise_start_s",
    "raise_end_s",
    "prograde_acceleration_m_s2",
    "min_raise_altitude_m",
    "pass_entry_altitude_m",
    "pass_exit_altitude_m",
    "recovery_delta_v_m_s",
    "orbital_element_control_law",
    "target_coes",
    "controlled_elements",
    "energy_gain_per_s",
    "eccentricity_gain_per_s",
    "plane_gain_per_s",
    "control_law",
    "control_design_dt_s",
    "lqr_q_weights",
    "lqr_r_weights",
    "rmoe_target_radial_center_m",
    "rmoe_target_in_track_center_m",
    "rmoe_target_in_track_drift_rate_m_s",
    "rmoe_target_cross_track_amplitude_m",
    "rmoe_max_drift_rate_m_s",
    "rmoe_close_zone_m",
    "rmoe_cross_track_burn_gate_m",
    "transfer_time_s",
    "burn_time_constant_s",
    "correction_interval_s",
    "velocity_deadband_m_s",
    "final_brake_start_s",
    "terminal_start_s",
    "terminal_range_m",
    "thrust_window_period_s",
    "thrust_window_duration_s",
    "thrust_window_phase_s",
    "thrust_command_deadband_m_s2",
    "element_averaging_window_s",
    "rcs_thrusters",
    "rcs_pulse_window_s",
    "gimbal_limit_rad",
    "attitude_reference_mode",
    "attitude_quaternion_bn",
    "attitude_ric_axis",
    "attitude_boresight_body",
    "attitude_target_position_eci_m",
    "attitude_thrust_direction_eci",
    "attitude_replay_times_s",
    "attitude_replay_quaternions_bn",
    "max_attitude_torque_n_m",
    "attitude_kp",
    "attitude_kd",
    "pointing_tolerance_rad",
    "goal_id",
    "goal_type",
    "goal_mode",
    "goal_dwell_s",
    "recovery_mode",
    "recover_on_fault",
    "require_pointing_for_translation",
    "actions",
    "constraints",
    "recovery_clear_dwell_s",
    "recover_on_action_timeout",
    "recovery_constraint_kinds",
    "navigation_filter",
    "navigation_alpha",
    "navigation_beta",
    "navigation_ekf_step_s",
    "navigation_process_noise_diag_si",
    "navigation_measurement_noise_diag_si",
    "navigation_initial_covariance_diag_si",
    "navigation_relative_mean_motion_rad_s",
    "navigation_nis_limit",
    "minimum_battery_soc",
    "minimum_available_power_w",
    "maximum_temperature_k",
    "maximum_storage_fraction",
    "minimum_propellant_kg",
    "conjunction_avoidance_enabled",
    "conjunction_keep_out_radius_m",
    "conjunction_prediction_horizon_s",
    "conjunction_avoidance_delta_v_m_s",
    "conjunction_maneuver_lead_time_s",
    "autonomous_maneuver_enabled",
    "maneuver_transfer_time_s",
    "maneuver_target_position_ric_m",
    "maneuver_maximum_delta_v_m_s",
}


def validate_stack_params(stack_id: str, params: dict[str, object]) -> None:
    """Reject misspelled or structurally invalid built-in stack settings."""

    allowed = (
        _PASSIVE_PARAMS
        if stack_id == "fsw.passive"
        else _ATTITUDE_PARAMS
        if stack_id == "fsw.attitude_reference"
        else _TRANSLATION_PARAMS
        if stack_id in {"fsw.orbit_reference", "fsw.rpo_reference", "fsw.low_thrust_reference"}
        else None
    )
    if allowed is None:
        return
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"flight_software.params has unsupported field(s) for {stack_id}: {', '.join(unknown)}.")
    if "wheel_axes_body" in params:
        try:
            axes = [[float(component) for component in axis] for axis in params["wheel_axes_body"]]  # type: ignore[union-attr]
        except (TypeError, ValueError) as exc:
            raise ValueError("flight_software.params.wheel_axes_body must contain finite 3-vectors.") from exc
        if not axes or any(len(axis) != 3 or any(not isfinite(value) for value in axis) for axis in axes):
            raise ValueError("flight_software.params.wheel_axes_body must contain finite 3-vectors.")
        if any(
            not isclose(sqrt(sum(value * value for value in axis)), 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9)
            for axis in axes
        ):
            raise ValueError("flight_software.params.wheel_axes_body must contain unit vectors.")
    navigation = str(params.get("navigation_initialization", "ideal"))
    if navigation not in {"cold", "loaded", "ideal"}:
        raise ValueError("flight_software.params.navigation_initialization must be cold, loaded, or ideal.")
    mode = str(params.get("translation_mode", "") or "")
    supported_modes = {
        "fsw.orbit_reference": {"scheduled_burn", "stationkeeping", "orbital_elements", "atmospheric_pass"},
        "fsw.rpo_reference": {
            "ric_hold",
            "r_bar_approach",
            "v_bar_approach",
            "c_bar_approach",
            "waypoint",
            "ric_pd_transfer",
            "terminal_braking",
            "passive_retreat",
        },
        "fsw.low_thrust_reference": {"low_thrust_phasing", "orbital_elements"},
    }
    if mode and stack_id in supported_modes and mode not in supported_modes[stack_id]:
        choices = ", ".join(sorted(supported_modes[stack_id]))
        raise ValueError(
            f"flight_software.params.translation_mode {mode!r} is not supported by {stack_id}; "
            f"choose one of: {choices}."
        )
    control_law = str(params.get("control_law", "reference_pd"))
    if control_law not in {"reference_pd", "hcw_lqr", "curvilinear_ric_pd", "rmoe_if_then"}:
        raise ValueError(
            "flight_software.params.control_law must be reference_pd, hcw_lqr, curvilinear_ric_pd, or rmoe_if_then."
        )
    if control_law != "reference_pd" and stack_id != "fsw.rpo_reference":
        raise ValueError("advanced relative control laws are supported only by fsw.rpo_reference.")
    for name in (
        "max_acceleration_m_s2",
        "max_force_n",
        "assumed_mass_kg",
        "max_torque_n_m",
        "kp_position_s2",
        "kd_velocity_s_inv",
        "goal_dwell_s",
        "slowdown_distance_m",
        "terminal_box_m",
        "terminal_max_closing_speed_m_s",
        "transfer_time_s",
        "burn_time_constant_s",
        "correction_interval_s",
        "velocity_deadband_m_s",
        "final_brake_start_s",
        "terminal_start_s",
        "terminal_range_m",
        "retreat_speed_m_s",
        "retreat_coast_range_m",
        "thrust_window_period_s",
        "thrust_window_duration_s",
        "thrust_window_phase_s",
        "thrust_command_deadband_m_s2",
        "element_averaging_window_s",
    ):
        if name in params:
            try:
                value = float(params[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"flight_software.params.{name} must be finite.") from exc
            if not isfinite(value):
                raise ValueError(f"flight_software.params.{name} must be finite.")
    low_thrust_only = {
        "thrust_window_period_s",
        "thrust_window_duration_s",
        "thrust_window_phase_s",
        "thrust_command_deadband_m_s2",
        "element_averaging_window_s",
    }
    if stack_id != "fsw.low_thrust_reference" and any(name in params for name in low_thrust_only):
        raise ValueError("low-thrust windowing and element averaging parameters require fsw.low_thrust_reference.")
    for name, size in (
        ("quaternion_bn", 4),
        ("boresight_body", 3),
        ("target_position_eci_m", 3),
        ("thrust_direction_eci", 3),
        ("target_relative_state_ric_m", 6),
        ("target_state_eci_m_m_s", 6),
        ("loaded_position_eci_m", 3),
        ("loaded_velocity_eci_m_s", 3),
        ("control_axis_mask", 3),
        ("attitude_quaternion_bn", 4),
        ("attitude_boresight_body", 3),
        ("attitude_target_position_eci_m", 3),
        ("attitude_thrust_direction_eci", 3),
        ("attitude_kp", 3),
        ("attitude_kd", 3),
    ):
        if name not in params:
            continue
        try:
            values = [float(value) for value in params[name]]  # type: ignore[union-attr]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"flight_software.params.{name} must contain exactly {size} finite numeric values."
            ) from exc
        if len(values) != size or any(not isfinite(value) for value in values):
            raise ValueError(f"flight_software.params.{name} must contain exactly {size} finite numeric values.")
    attitude_mode = str(params.get("reference_mode", params.get("attitude_reference_mode", "")) or "")
    prefix = "" if "reference_mode" in params else "attitude_"
    if attitude_mode == "target" and params.get(f"{prefix}target_position_eci_m") is None:
        raise ValueError(f"flight_software.params.{prefix}target_position_eci_m is required for target pointing.")
    if attitude_mode == "thrust" and params.get(f"{prefix}thrust_direction_eci") is None:
        raise ValueError(f"flight_software.params.{prefix}thrust_direction_eci is required for thrust pointing.")
    if str(params.get("translation_mode", "")) == "waypoint" and not list(params.get("waypoints_ric", []) or []):
        raise ValueError("flight_software.params.waypoints_ric is required for waypoint mode.")
    if navigation == "loaded" and (
        params.get("loaded_position_eci_m") is None or params.get("loaded_velocity_eci_m_s") is None
    ):
        raise ValueError("loaded navigation initialization requires loaded_position_eci_m and loaded_velocity_eci_m_s.")
