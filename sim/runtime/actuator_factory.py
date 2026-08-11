"""Satellite actuator, propulsion, and mass-property construction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.actuators import (
    ActuatorFaultConfig,
    AttitudeActuator,
    CombinedActuator,
    ControlMomentGyroLimits,
    ElectricPropulsionLimits,
    FaultedActuator,
    GimbaledThrusterLimits,
    MagnetorquerLimits,
    OrbitalActuator,
    OrbitalActuatorLimits,
    RcsClusterLimits,
    RcsThruster,
    ReactionWheelLimits,
    ThrusterPulseLimits,
    WheelDesaturationLimits,
)
from sim.actuators.presets import resolve_actuator_specs_from_satellite_specs
from sim.digital_twin.mass_properties import resolve_inertia_kg_m2
from sim.presets.thrusters import (
    BASIC_CHEMICAL_BOTTOM_Z,
    resolve_thruster_max_thrust_n_from_specs,
    resolve_thruster_mount_from_specs,
)


def _resolve_satellite_isp_s(specs: dict[str, Any]) -> float:
    if "isp_s" in specs:
        return float(specs.get("isp_s", 0.0))
    if "thruster_isp_s" in specs:
        return float(specs.get("thruster_isp_s", 0.0))
    thr = str(specs.get("thruster", "")).strip().upper()
    if thr in ("BASIC_CHEMICAL_BOTTOM_Z", "BASIC_CHEMICAL_Z_BOTTOM"):
        return float(BASIC_CHEMICAL_BOTTOM_Z.isp_s)
    orbital = dict(dict(specs.get("actuators", {}) or {}).get("orbital", {}) or {})
    electric = dict(orbital.get("electric_propulsion", {}) or {})
    if electric.get("isp_s") is not None:
        return float(electric["isp_s"])
    rcs = dict(orbital.get("rcs_cluster", {}) or {})
    if rcs.get("isp_s") is not None:
        return float(rcs["isp_s"])
    thrusters = list(rcs.get("thrusters", []) or [])
    if thrusters and dict(thrusters[0] or {}).get("isp_s") is not None:
        return float(dict(thrusters[0] or {})["isp_s"])
    # The ideal-wrench reference profile historically accepted a fuel load
    # without naming a propulsion preset.  Give that abstract force source the
    # same chemical-reference Isp used by the legacy actuator stack so fuel is
    # depleted instead of silently ignored.  Physical profiles continue to
    # override this through their explicit Isp or preset.
    if float(specs.get("fuel_mass_kg", 0.0) or 0.0) > 0.0:
        return 220.0
    return 0.0


def _resolve_satellite_inertia_kg_m2(specs: dict[str, Any]) -> np.ndarray:
    return resolve_inertia_kg_m2(specs, default=np.diag([120.0, 100.0, 80.0]))


def _satellite_spec_float(
    specs: dict[str, Any],
    names: tuple[str, ...],
    *,
    default: float,
    min_value: float | None = 0.0,
) -> tuple[float, bool]:
    for name in names:
        if name not in specs or specs.get(name) is None:
            continue
        value = float(specs[name])
        if not np.isfinite(value):
            raise ValueError(f"specs.{name} must be finite.")
        if min_value is not None and value < min_value:
            raise ValueError(f"specs.{name} must be >= {min_value}.")
        return value, True
    return float(default), False


def _satellite_spec_vector3(value: Any, *, field_name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.array(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"specs.{field_name} must contain finite values.")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"specs.{field_name} must be non-zero.")
    return arr / norm


def _array_or_none(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.array(value, dtype=float)
    if shape is not None:
        arr = arr.reshape(shape)
    if not np.all(np.isfinite(arr)):
        raise ValueError("actuator vector values must be finite.")
    return arr


def _angle_value_rad(raw: dict[str, Any], *, rad_name: str, deg_name: str, default_rad: float = 0.0) -> float:
    if rad_name in raw and raw.get(rad_name) is not None:
        return float(raw[rad_name])
    if deg_name in raw and raw.get(deg_name) is not None:
        return float(np.deg2rad(float(raw[deg_name])))
    return float(default_rad)


def _build_rcs_cluster(raw: Any) -> RcsClusterLimits | None:
    if not isinstance(raw, dict) or not _strict_enabled(raw, "rcs_cluster"):
        return None
    thrusters_raw = list(raw.get("thrusters", []) or [])
    thrusters: list[RcsThruster] = []
    for idx, item in enumerate(thrusters_raw):
        row = dict(item or {})
        thrusters.append(
            RcsThruster(
                name=str(row.get("name", f"rcs_{idx}")),
                position_body_m=np.array(row.get("position_body_m", [0.0, 0.0, 0.0]), dtype=float).reshape(3),
                force_direction_body=np.array(row.get("force_direction_body", [1.0, 0.0, 0.0]), dtype=float).reshape(
                    3
                ),
                max_thrust_n=float(row.get("max_thrust_n", 0.0)),
                min_impulse_bit_n_s=float(row.get("min_impulse_bit_n_s", 0.0)),
                isp_s=float(row.get("isp_s", raw.get("isp_s", 220.0))),
            )
        )
    if not thrusters:
        return None
    return RcsClusterLimits(
        thrusters=tuple(thrusters),
        allocation_mode=str(raw.get("allocation_mode", "force_torque")),
        pulse_quantum_s=float(raw.get("pulse_quantum_s", 0.0)),
        duty_cycle=float(raw.get("duty_cycle", 1.0)),
        force_weight=float(raw.get("force_weight", 1.0)),
        torque_weight=float(raw.get("torque_weight", 1.0)),
    )


def _build_electric_propulsion(raw: Any) -> ElectricPropulsionLimits | None:
    if not isinstance(raw, dict) or not _strict_enabled(raw, "electric_propulsion"):
        return None
    return ElectricPropulsionLimits(
        max_thrust_n=float(raw.get("max_thrust_n", 0.0)),
        isp_s=float(raw.get("isp_s", 1500.0)),
        duty_cycle=float(raw.get("duty_cycle", 1.0)),
        max_power_w=(None if raw.get("max_power_w") is None else float(raw.get("max_power_w"))),
        power_per_newton_w=(None if raw.get("power_per_newton_w") is None else float(raw.get("power_per_newton_w"))),
        throttle_time_constant_s=float(raw.get("throttle_time_constant_s", 0.0)),
    )


def _build_gimbaled_thruster(raw: Any) -> GimbaledThrusterLimits | None:
    if not isinstance(raw, dict) or not _strict_enabled(raw, "gimbaled_thruster"):
        return None
    return GimbaledThrusterLimits(
        neutral_direction_body=np.array(raw.get("neutral_direction_body", [-1.0, 0.0, 0.0]), dtype=float).reshape(3),
        position_body_m=_array_or_none(raw.get("position_body_m"), shape=(3,)),
        max_gimbal_angle_rad=_angle_value_rad(raw, rad_name="max_gimbal_angle_rad", deg_name="max_gimbal_angle_deg"),
        max_gimbal_rate_rad_s=_angle_value_rad(
            raw,
            rad_name="max_gimbal_rate_rad_s",
            deg_name="max_gimbal_rate_deg_s",
            default_rad=float("inf"),
        ),
        response_time_constant_s=float(raw.get("response_time_constant_s", 0.0)),
    )


def _strict_enabled(raw: dict[str, Any], path: str) -> bool:
    value = raw.get("enabled", True)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.enabled must be a boolean true/false value.")
    return value


def _build_reaction_wheels(raw: Any) -> ReactionWheelLimits | None:
    if not isinstance(raw, dict) or not _strict_enabled(raw, "reaction_wheels"):
        return None
    return ReactionWheelLimits(
        max_torque_nm=np.array(raw.get("max_torque_nm", [0.05, 0.05, 0.05]), dtype=float).reshape(-1),
        max_momentum_nms=np.array(raw.get("max_momentum_nms", [0.2, 0.2, 0.2]), dtype=float).reshape(-1),
        wheel_axes_body=_array_or_none(raw.get("wheel_axes_body")),
        wheel_inertia_kg_m2=_array_or_none(raw.get("wheel_inertia_kg_m2")),
        max_speed_rad_s=_array_or_none(raw.get("max_speed_rad_s")),
        torque_time_constant_s=float(raw.get("torque_time_constant_s", 0.0)),
        viscous_friction_nms=raw.get("viscous_friction_nms", 0.0),
        coulomb_friction_nm=raw.get("coulomb_friction_nm", 0.0),
    )


def _enabled_device_mapping(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    resolved = dict(raw)
    if not _strict_enabled(resolved, "actuator device"):
        return None
    return resolved


def _build_satellite_actuator_stack_from_specs(specs: dict[str, Any]) -> tuple[Any | None, dict[str, Any], bool]:
    raw = resolve_actuator_specs_from_satellite_specs(specs)
    if not isinstance(raw, dict) or not _strict_enabled(raw, "actuators"):
        return None, {}, False
    orbital_raw = dict(raw.get("orbital", {}) or {})
    attitude_raw = dict(raw.get("attitude", {}) or {})
    mount = resolve_thruster_mount_from_specs(specs)
    max_thrust_n = orbital_raw.get("max_thrust_n", resolve_thruster_max_thrust_n_from_specs(specs))
    isp_s = float(orbital_raw.get("isp_s", _resolve_satellite_isp_s(specs) or 220.0))
    default_direction = None if mount is None else np.array(mount.thrust_direction_body, dtype=float)
    default_position = None if mount is None else np.array(mount.position_body_m, dtype=float)
    magnetorquers_raw = _enabled_device_mapping(attitude_raw.get("magnetorquers"))
    thruster_pulse_raw = _enabled_device_mapping(attitude_raw.get("thruster_pulse"))
    cmg_raw = _enabled_device_mapping(attitude_raw.get("control_moment_gyros"))
    wheel_desaturation_raw = _enabled_device_mapping(attitude_raw.get("wheel_desaturation"))

    orbital_limits = OrbitalActuatorLimits(
        max_accel_km_s2=float(orbital_raw.get("max_accel_km_s2", specs.get("max_accel_km_s2", 1.0e9))),
        max_thrust_n=(None if max_thrust_n is None else float(max_thrust_n)),
        min_impulse_bit_km_s=float(orbital_raw.get("min_impulse_bit_km_s", 0.0)),
        max_throttle_rate_km_s2_s=float(orbital_raw.get("max_throttle_rate_km_s2_s", 1.0e9)),
        isp_s=isp_s,
        thruster_direction_body=_array_or_none(orbital_raw.get("thruster_direction_body"), shape=(3,))
        if "thruster_direction_body" in orbital_raw
        else default_direction,
        thruster_position_body_m=_array_or_none(orbital_raw.get("thruster_position_body_m"), shape=(3,))
        if "thruster_position_body_m" in orbital_raw
        else default_position,
        couple_to_attitude=bool(orbital_raw.get("couple_to_attitude", True)),
        rcs_cluster=_build_rcs_cluster(orbital_raw.get("rcs_cluster")),
        electric_propulsion=_build_electric_propulsion(orbital_raw.get("electric_propulsion")),
        gimbaled_thruster=_build_gimbaled_thruster(orbital_raw.get("gimbaled_thruster")),
    )
    attitude_act = AttitudeActuator(
        reaction_wheels=_build_reaction_wheels(attitude_raw.get("reaction_wheels")),
        magnetorquers=(
            None
            if magnetorquers_raw is None
            else MagnetorquerLimits(
                max_dipole_a_m2=np.array(
                    magnetorquers_raw.get("max_dipole_a_m2", [0.0, 0.0, 0.0]),
                    dtype=float,
                ).reshape(-1)
            )
        ),
        thruster_pulse=(
            None
            if thruster_pulse_raw is None
            else ThrusterPulseLimits(
                max_torque_nm=np.array(
                    thruster_pulse_raw.get("max_torque_nm", [0.0, 0.0, 0.0]),
                    dtype=float,
                ).reshape(3),
                pulse_quantum_s=float(thruster_pulse_raw.get("pulse_quantum_s", 0.02)),
            )
        ),
        control_moment_gyros=(
            None
            if cmg_raw is None
            else ControlMomentGyroLimits(
                max_torque_nm=cmg_raw.get("max_torque_nm", 0.0),
                momentum_nms=cmg_raw.get("momentum_nms", 0.0),
                gimbal_rate_limit_rad_s=cmg_raw.get("gimbal_rate_limit_rad_s", np.inf),
                torque_time_constant_s=float(cmg_raw.get("torque_time_constant_s", 0.0)),
            )
        ),
        wheel_desaturation=(
            None
            if wheel_desaturation_raw is None
            else WheelDesaturationLimits(
                momentum_fraction_threshold=float(wheel_desaturation_raw.get("momentum_fraction_threshold", 0.8)),
                unload_gain_s_inv=float(wheel_desaturation_raw.get("unload_gain_s_inv", 0.02)),
                max_unload_torque_nm=float(wheel_desaturation_raw.get("max_unload_torque_nm", 0.01)),
            )
        ),
    )
    actuator: Any = CombinedActuator(
        orbital=OrbitalActuator(lag_tau_s=float(orbital_raw.get("lag_tau_s", 0.0))),
        attitude=attitude_act,
    )
    fault_raw = dict(raw.get("faults", {}) or {})
    if fault_raw:
        actuator = FaultedActuator(
            base=actuator,
            faults=ActuatorFaultConfig(
                stuck_off=bool(fault_raw.get("stuck_off", False)),
                thrust_scale=float(fault_raw.get("thrust_scale", 1.0)),
                torque_scale=float(fault_raw.get("torque_scale", 1.0)),
                thrust_bias_eci_km_s2=np.array(fault_raw.get("thrust_bias_eci_km_s2", [0.0, 0.0, 0.0]), dtype=float),
                torque_bias_body_nm=np.array(fault_raw.get("torque_bias_body_nm", [0.0, 0.0, 0.0]), dtype=float),
            ),
        )
    return actuator, {"orbital": orbital_limits}, True


def _initial_state_nonnegative_float(initial_state: dict[str, Any], name: str, *, default: float = 0.0) -> float:
    value = float(initial_state.get(name, default) if initial_state.get(name) is not None else default)
    if not np.isfinite(value):
        raise ValueError(f"initial_state.{name} must be finite.")
    if value < 0.0:
        raise ValueError(f"initial_state.{name} must be >= 0.0.")
    return value


def _apply_thruster_mount_defaults(module_obj: Any | None, pointer: Any | None, specs: dict[str, Any]) -> Any | None:
    if module_obj is None:
        return None
    mount = resolve_thruster_mount_from_specs(specs)
    if mount is None:
        return module_obj
    params = dict(getattr(pointer, "params", {}) or {}) if pointer is not None else {}
    if hasattr(module_obj, "thruster_direction_body") and "thruster_direction_body" not in params:
        try:
            module_obj.thruster_direction_body = np.array(mount.thrust_direction_body, dtype=float)
        except (TypeError, ValueError, AttributeError):
            pass
    if hasattr(module_obj, "thruster_position_body_m") and "thruster_position_body_m" not in params:
        try:
            module_obj.thruster_position_body_m = np.array(mount.position_body_m, dtype=float)
        except (TypeError, ValueError, AttributeError):
            pass
    return module_obj
