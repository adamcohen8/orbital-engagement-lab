from __future__ import annotations

from copy import deepcopy
from typing import Any


def _rcs_thruster(
    name: str,
    position_body_m: tuple[float, float, float],
    force_direction_body: tuple[float, float, float],
    *,
    max_thrust_n: float = 0.5,
    isp_s: float = 230.0,
    min_impulse_bit_n_s: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "position_body_m": list(position_body_m),
        "force_direction_body": list(force_direction_body),
        "max_thrust_n": float(max_thrust_n),
        "isp_s": float(isp_s),
        "min_impulse_bit_n_s": float(min_impulse_bit_n_s),
    }


BASIC_RCS_6DOF: dict[str, Any] = {
    "enabled": True,
    "orbital": {
        "max_accel_km_s2": 1.0e-3,
        "couple_to_attitude": True,
        "rcs_cluster": {
            "allocation_mode": "force_torque",
            "pulse_quantum_s": 0.0,
            "duty_cycle": 1.0,
            "thrusters": [
                _rcs_thruster("rcs-plus-x-y-plus", (0.0, 0.35, 0.0), (1.0, 0.0, 0.0)),
                _rcs_thruster("rcs-plus-x-y-minus", (0.0, -0.35, 0.0), (1.0, 0.0, 0.0)),
                _rcs_thruster("rcs-minus-x-y-plus", (0.0, 0.35, 0.0), (-1.0, 0.0, 0.0)),
                _rcs_thruster("rcs-minus-x-y-minus", (0.0, -0.35, 0.0), (-1.0, 0.0, 0.0)),
                _rcs_thruster("rcs-plus-y-x-plus", (0.35, 0.0, 0.0), (0.0, 1.0, 0.0)),
                _rcs_thruster("rcs-plus-y-x-minus", (-0.35, 0.0, 0.0), (0.0, 1.0, 0.0)),
                _rcs_thruster("rcs-minus-y-x-plus", (0.35, 0.0, 0.0), (0.0, -1.0, 0.0)),
                _rcs_thruster("rcs-minus-y-x-minus", (-0.35, 0.0, 0.0), (0.0, -1.0, 0.0)),
                _rcs_thruster("rcs-plus-z-x-plus", (0.35, 0.0, 0.0), (0.0, 0.0, 1.0)),
                _rcs_thruster("rcs-plus-z-x-minus", (-0.35, 0.0, 0.0), (0.0, 0.0, 1.0)),
                _rcs_thruster("rcs-minus-z-x-plus", (0.35, 0.0, 0.0), (0.0, 0.0, -1.0)),
                _rcs_thruster("rcs-minus-z-x-minus", (-0.35, 0.0, 0.0), (0.0, 0.0, -1.0)),
                _rcs_thruster("rcs-plus-z-y-plus", (0.0, 0.35, 0.0), (0.0, 0.0, 1.0)),
                _rcs_thruster("rcs-plus-z-y-minus", (0.0, -0.35, 0.0), (0.0, 0.0, 1.0)),
                _rcs_thruster("rcs-minus-z-y-plus", (0.0, 0.35, 0.0), (0.0, 0.0, -1.0)),
                _rcs_thruster("rcs-minus-z-y-minus", (0.0, -0.35, 0.0), (0.0, 0.0, -1.0)),
            ],
        },
    },
}

BASIC_ELECTRIC_PROPULSION: dict[str, Any] = {
    "enabled": True,
    "orbital": {
        "max_accel_km_s2": 1.0e-3,
        "couple_to_attitude": False,
        "electric_propulsion": {
            "max_thrust_n": 0.5,
            "isp_s": 1600.0,
            "duty_cycle": 1.0,
            "max_power_w": 100.0,
            "power_per_newton_w": 200.0,
            "throttle_time_constant_s": 0.0,
        },
    },
}

BASIC_MAGNETORQUER_TRIAD: dict[str, Any] = {
    "enabled": True,
    "attitude": {
        "magnetorquers": {
            "max_dipole_a_m2": [10.0, 10.0, 10.0],
        },
    },
}

BASIC_CMG_TRIAD: dict[str, Any] = {
    "enabled": True,
    "attitude": {
        "control_moment_gyros": {
            "max_torque_nm": [0.2, 0.2, 0.2],
            "momentum_nms": [1.0, 1.0, 1.0],
            "gimbal_rate_limit_rad_s": [0.1, 0.1, 0.1],
            "torque_time_constant_s": 0.0,
        },
    },
}

BASIC_GIMBALED_THRUSTER: dict[str, Any] = {
    "enabled": True,
    "orbital": {
        "max_thrust_n": 5.0,
        "isp_s": 235.0,
        "max_accel_km_s2": 1.0e-3,
        "couple_to_attitude": True,
        "gimbaled_thruster": {
            "neutral_direction_body": [-1.0, 0.0, 0.0],
            "position_body_m": [0.0, 0.0, -0.5],
            "max_gimbal_angle_deg": 5.0,
            "max_gimbal_rate_deg_s": 2.0,
            "response_time_constant_s": 0.0,
        },
    },
}

ACTUATOR_PRESETS: dict[str, dict[str, Any]] = {
    "BASIC_RCS_6DOF": BASIC_RCS_6DOF,
    "BASIC_ELECTRIC_PROPULSION": BASIC_ELECTRIC_PROPULSION,
    "BASIC_MAGNETORQUER_TRIAD": BASIC_MAGNETORQUER_TRIAD,
    "BASIC_CMG_TRIAD": BASIC_CMG_TRIAD,
    "BASIC_GIMBALED_THRUSTER": BASIC_GIMBALED_THRUSTER,
}


def available_actuator_preset_names() -> tuple[str, ...]:
    return tuple(sorted(ACTUATOR_PRESETS))


def actuator_preset_to_specs(name: str) -> dict[str, Any]:
    key = str(name or "").strip().upper()
    if key not in ACTUATOR_PRESETS:
        choices = ", ".join(available_actuator_preset_names())
        raise KeyError(f"Unknown actuator preset '{name}'. Available presets: {choices}.")
    return deepcopy(ACTUATOR_PRESETS[key])


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(dict(merged[key]), dict(value))
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_actuator_specs_from_satellite_specs(specs: dict[str, Any] | None) -> dict[str, Any] | None:
    raw_specs = dict(specs or {})
    raw_actuators = raw_specs.get("actuators", raw_specs.get("actuator_model"))
    actuator_block = dict(raw_actuators or {}) if isinstance(raw_actuators, dict) else None
    preset_name = raw_specs.get("actuator_preset")
    if actuator_block is not None and actuator_block.get("preset") not in (None, ""):
        preset_name = actuator_block.get("preset")

    if preset_name in (None, ""):
        return actuator_block

    resolved = actuator_preset_to_specs(str(preset_name))
    if actuator_block is None:
        return resolved
    local = {key: value for key, value in actuator_block.items() if key != "preset"}
    return _deep_merge_dicts(resolved, local)
