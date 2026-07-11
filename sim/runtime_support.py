from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

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
from sim.aero import aero_spec_get, resolve_vehicle_aero_properties
from sim.config import SimulationScenarioConfig
from sim.config.plugin_specs import instantiate_plugin_spec
from sim.control.attitude.zero_torque import ZeroTorqueController
from sim.control.orbit.zero_controller import ZeroController
from sim.core.models import Command, StateBelief, StateTruth
from sim.digital_twin.mass_properties import resolve_center_of_mass_body_m, resolve_inertia_kg_m2
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.cr3bp import (
    cr3bp_halo_seed_state_km_s,
    cr3bp_moon_state_km_s,
    cr3bp_system,
    propagate_cr3bp_state,
)
from sim.dynamics.orbit.elements import coe_to_rv_eci as _coe_to_rv_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import frame_context_from_environment, transform_position
from sim.dynamics.orbit.propagator import (
    OrbitPropagator,
    drag_plugin,
    j2_plugin,
    j3_plugin,
    j4_plugin,
    lift_plugin,
    spherical_harmonics_plugin,
    srp_plugin,
    third_body_moon_plugin,
    third_body_sun_plugin,
)
from sim.dynamics.orbit.tle import tle_block_to_rv_eci
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile
from sim.estimation.joint_state import JointStateEstimator
from sim.estimation.maneuver_detection import EKFManeuverDetectionConfig
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.presets.rockets import BASIC_1ST_STAGE, BASIC_SSTO_ROCKET, BASIC_TWO_STAGE_STACK, RocketStackPreset
from sim.presets.thrusters import (
    BASIC_CHEMICAL_BOTTOM_Z,
    resolve_thruster_max_thrust_n_from_specs,
    resolve_thruster_mount_from_specs,
)
from sim.rocket import (
    MaxQThrottleLimiterGuidance,
    OpenLoopPitchProgramGuidance,
    OrbitInsertionCutoffGuidance,
    RocketAscentSimulator,
    RocketGuidanceLaw,
    RocketSimConfig,
    RocketState,
    RocketVehicleConfig,
    TVCSteeringGuidance,
)
from sim.rocket.aero import RocketAeroConfig
from sim.sensors.joint_state import JointStateSensor
from sim.sensors.noisy_own_state import NoisyOwnStateSensor
from sim.utils.frames import eci_relative_to_ric_rect, ric_curv_to_rect, ric_rect_state_to_eci, ric_rect_to_curv
from sim.utils.geodesy import ecef_to_geodetic_deg_km
from sim.utils.quaternion import quaternion_to_dcm_bn

logger = logging.getLogger(__name__)


def _geometry_profile_path_from_specs(specs: dict[str, Any]) -> str | None:
    raw = dict(specs or {})
    for key in ("geometry_profile_path", "area_profile_path", "attitude_area_profile_path"):
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    geometry = raw.get("geometry")
    if isinstance(geometry, dict):
        for key in ("profile_path", "area_profile_path", "attitude_area_profile_path"):
            value = geometry.get(key)
            if value not in (None, ""):
                return str(value)
    aero = raw.get("aero")
    if isinstance(aero, dict):
        for key in ("geometry_profile_path", "area_profile_path"):
            value = aero.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _load_geometry_area_profile_from_specs(specs: dict[str, Any]) -> GeometryAreaProfile | None:
    path_text = _geometry_profile_path_from_specs(specs)
    if path_text is None:
        return None
    return GeometryAreaProfile.load(Path(path_text))


def _earth_impact_policy_for_object(termination: Any, object_id: str) -> dict[str, Any]:
    root = dict(termination or {})
    policy = {
        "earth_impact_enabled": bool(root.get("earth_impact_enabled", True)),
        "earth_radius_km": float(root.get("earth_radius_km", EARTH_RADIUS_KM)),
    }
    by_object = root.get("by_object", {}) or {}
    if not isinstance(by_object, dict):
        return policy
    for key in ("*", "all", str(object_id)):
        override = by_object.get(key)
        if not isinstance(override, dict):
            continue
        if "earth_impact_enabled" in override:
            policy["earth_impact_enabled"] = bool(override.get("earth_impact_enabled"))
        elif "enabled" in override:
            policy["earth_impact_enabled"] = bool(override.get("enabled"))
        if override.get("earth_radius_km") is not None:
            policy["earth_radius_km"] = float(override.get("earth_radius_km"))
    return policy


def _module_obj(pointer, *, extra_kwargs: dict[str, Any] | None = None) -> Any | None:
    if pointer is None or pointer.module is None:
        return None
    if extra_kwargs:
        from dataclasses import replace

        pointer = replace(pointer, params={**dict(pointer.params or {}), **dict(extra_kwargs)})
    return instantiate_plugin_spec(pointer)


def _compatible_keyword_args(method: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return None

    accepts_var_kwargs = False
    filtered: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            return None
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) and name in kwargs:
            filtered[name] = kwargs[name]

    for name, param in signature.parameters.items():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        if param.default is inspect.Signature.empty and name not in filtered:
            return None

    return dict(kwargs) if accepts_var_kwargs else filtered


def _call_with_compat_kwargs(
    method: Callable[..., Any],
    *,
    primary_kwargs: dict[str, Any],
    fallback_kwargs: dict[str, Any] | None = None,
) -> Any:
    compatible = _compatible_keyword_args(method, primary_kwargs)
    if compatible is not None:
        return method(**compatible)
    if fallback_kwargs is not None:
        compatible = _compatible_keyword_args(method, fallback_kwargs)
        if compatible is not None:
            return method(**compatible)
    return method(**primary_kwargs)


def _to_jsonable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable_value(v) for v in value]
    return value


def _command_to_dict(cmd: Command) -> dict[str, Any]:
    return {
        "thrust_eci_km_s2": np.array(cmd.thrust_eci_km_s2, dtype=float).tolist(),
        "torque_body_nm": np.array(cmd.torque_body_nm, dtype=float).tolist(),
        "mode_flags": _to_jsonable_value(dict(cmd.mode_flags or {})),
    }


def _deep_set(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = root
    for i, tok in enumerate(parts):
        last = i == len(parts) - 1
        if "[" in tok and tok.endswith("]"):
            key, idx_txt = tok[:-1].split("[", 1)
            idx = int(idx_txt)
            if key:
                cur = cur[key]
            if not isinstance(cur, list):
                raise TypeError(f"'{tok}' is not a list segment in path '{path}'.")
            if last:
                cur[idx] = value
                return
            cur = cur[idx]
            continue
        if last:
            cur[tok] = value
            return
        cur = cur[tok]


def _sample_variation(v, rng: np.random.Generator) -> Any:
    mode = v.mode.lower()
    if mode == "choice":
        if not v.options:
            raise ValueError(f"Variation '{v.parameter_path}' with mode=choice requires options.")
        return v.options[int(rng.integers(0, len(v.options)))]
    if mode == "uniform":
        if v.low is None or v.high is None:
            raise ValueError(f"Variation '{v.parameter_path}' with mode=uniform requires low/high.")
        return float(rng.uniform(v.low, v.high))
    if mode == "normal":
        if v.mean is None or v.std is None:
            raise ValueError(f"Variation '{v.parameter_path}' with mode=normal requires mean/std.")
        return float(rng.normal(v.mean, v.std))
    raise ValueError(f"Unsupported variation mode '{v.mode}'.")


def _combine_commands(orb: Command, att: Command) -> Command:
    return Command(
        thrust_eci_km_s2=np.array(orb.thrust_eci_km_s2, dtype=float),
        torque_body_nm=np.array(att.torque_body_nm, dtype=float),
        mode_flags={**dict(orb.mode_flags or {}), **dict(att.mode_flags or {})},
    )


def _rocket_state_to_truth(s: RocketState) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array(s.position_eci_km, dtype=float),
        velocity_eci_km_s=np.array(s.velocity_eci_km_s, dtype=float),
        attitude_quat_bn=np.array(s.attitude_quat_bn, dtype=float),
        angular_rate_body_rad_s=np.array(s.angular_rate_body_rad_s, dtype=float),
        mass_kg=float(s.mass_kg),
        t_s=float(s.t_s),
    )


def _truth_state6(truth: StateTruth, out: np.ndarray | None = None) -> np.ndarray:
    state = np.empty(6, dtype=float) if out is None else out
    state[0:3] = truth.position_eci_km
    state[3:6] = truth.velocity_eci_km_s
    return state


def _decision_truth_from_belief(agent: AgentRuntime) -> StateTruth | None:
    belief = agent.belief
    if belief is None or belief.state.size < 6:
        return None
    state = np.array(belief.state, dtype=float).reshape(-1)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w = np.zeros(3, dtype=float)
    resource_truth = getattr(agent, "truth", None)
    rocket_state = getattr(agent, "rocket_state", None)
    if resource_truth is not None:
        mass_kg = float(resource_truth.mass_kg)
    elif rocket_state is not None:
        mass_kg = float(rocket_state.mass_kg)
    else:
        mass_kg = 0.0
    if state.size >= 13:
        q = np.array(state[6:10], dtype=float)
        w = np.array(state[10:13], dtype=float)
    return StateTruth(
        position_eci_km=np.array(state[:3], dtype=float),
        velocity_eci_km_s=np.array(state[3:6], dtype=float),
        attitude_quat_bn=q,
        angular_rate_body_rad_s=w,
        mass_kg=mass_kg,
        t_s=float(belief.last_update_t_s),
    )


def _truth_from_state6(state6: np.ndarray, *, t_s: float, fallback_truth: StateTruth | None = None) -> StateTruth:
    state = np.array(state6, dtype=float).reshape(-1)
    if state.size < 6:
        raise ValueError("state6 must contain at least 6 elements.")
    return StateTruth(
        position_eci_km=np.array(state[:3], dtype=float),
        velocity_eci_km_s=np.array(state[3:6], dtype=float),
        attitude_quat_bn=(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if fallback_truth is None
            else np.array(fallback_truth.attitude_quat_bn, dtype=float)
        ),
        angular_rate_body_rad_s=(
            np.zeros(3, dtype=float)
            if fallback_truth is None
            else np.array(fallback_truth.angular_rate_body_rad_s, dtype=float)
        ),
        mass_kg=0.0 if fallback_truth is None else float(fallback_truth.mass_kg),
        t_s=float(t_s),
    )


def _attitude_state13_from_belief(
    belief: StateBelief,
    truth: StateTruth,
    out: np.ndarray | None = None,
) -> np.ndarray:
    state = np.empty(13, dtype=float) if out is None else out
    state[0:6] = belief.state[:6]
    if belief.state.size >= 13:
        state[6:10] = belief.state[6:10]
        state[10:13] = belief.state[10:13]
    else:
        state[6:10] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        state[10:13] = np.zeros(3, dtype=float)
    return state


def _relative_orbit_state12(
    chief_truth: StateTruth,
    deputy_truth: StateTruth,
    out: np.ndarray | None = None,
    deputy_state6: np.ndarray | None = None,
    chief_state6: np.ndarray | None = None,
) -> np.ndarray:
    state = np.empty(12, dtype=float) if out is None else out
    r_c = chief_truth.position_eci_km
    v_c = chief_truth.velocity_eci_km_s
    x_dep_eci = np.empty(6, dtype=float) if deputy_state6 is None else deputy_state6
    x_chief_eci = np.empty(6, dtype=float) if chief_state6 is None else chief_state6
    x_dep_eci[0:3] = deputy_truth.position_eci_km
    x_dep_eci[3:6] = deputy_truth.velocity_eci_km_s
    x_chief_eci[0:3] = r_c
    x_chief_eci[3:6] = v_c
    x_rect = eci_relative_to_ric_rect(x_dep_eci=x_dep_eci, x_chief_eci=x_chief_eci)
    state[0:6] = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_c)))
    state[6:9] = r_c
    state[9:12] = v_c
    return state


def _rv_from_initial_state(s0: dict[str, Any], *, target_jd_utc: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not s0 or bool(s0.get("default_circular_earth", False)):
        pos = np.array([7000.0, 0.0, 0.0], dtype=float)
        spd = float(np.sqrt(EARTH_MU_KM3_S2 / np.linalg.norm(pos)))
        return pos, np.array([0.0, spd, 0.0], dtype=float)

    if any(key in s0 for key in ("relative_to_target_ric", "relative_ric_rect", "source", "launch_lat_deg")):
        # These recognized state forms are resolved by their dedicated runtime
        # initializers after the object graph or launch/deployment state exists.
        pos = np.array([7000.0, 0.0, 0.0], dtype=float)
        spd = float(np.sqrt(EARTH_MU_KM3_S2 / np.linalg.norm(pos)))
        return pos, np.array([0.0, spd, 0.0], dtype=float)

    cr3bp_state = s0.get("cr3bp_rotating")
    if isinstance(cr3bp_state, dict):
        raw_state = cr3bp_state.get("state_km_s", cr3bp_state.get("state"))
        if raw_state is None:
            raise ValueError("initial_state.cr3bp_rotating.state_km_s must be a length-6 list.")
        state = np.array(raw_state, dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("initial_state.cr3bp_rotating.state_km_s must be length-6.")
        return state[:3], state[3:]

    halo = s0.get("cr3bp_halo")
    if isinstance(halo, dict):
        system = cr3bp_system(str(halo.get("system", "earth_moon") or "earth_moon"))
        family = str(halo.get("family", "l1_northern") or "l1_northern")
        state = cr3bp_halo_seed_state_km_s(system=system, family=family)
        phase_time_s = float(halo.get("phase_time_s", 0.0) or 0.0)
        if not np.isfinite(phase_time_s) or phase_time_s < 0.0:
            raise ValueError("initial_state.cr3bp_halo.phase_time_s must be a nonnegative finite number.")
        if phase_time_s > 0.0:
            remaining_s = phase_time_s
            current_t_s = 0.0
            substep_s = float(halo.get("phase_substep_s", 120.0) or 120.0)
            if not np.isfinite(substep_s) or substep_s <= 0.0:
                raise ValueError("initial_state.cr3bp_halo.phase_substep_s must be a positive finite number.")
            while remaining_s > 1.0e-9:
                dt_s = min(substep_s, remaining_s)
                state = propagate_cr3bp_state(state, dt_s, current_t_s, system=system)
                current_t_s += dt_s
                remaining_s -= dt_s
        return state[:3], state[3:]

    if "position_eci_km" in s0:
        pos = np.array(s0["position_eci_km"], dtype=float).reshape(3)
        if "velocity_eci_km_s" not in s0:
            raise ValueError("initial_state.position_eci_km requires initial_state.velocity_eci_km_s.")
        vel = np.array(s0["velocity_eci_km_s"], dtype=float).reshape(3)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            raise ValueError("Cartesian initial-state position and velocity entries must be finite.")
        return pos, vel

    tle = s0.get("tle")
    if isinstance(tle, dict):
        return tle_block_to_rv_eci(tle, target_jd_utc=target_jd_utc)

    coes = s0.get("coes")
    if isinstance(coes, dict):
        d = dict(coes)
        return _coe_to_rv_eci(
            a_km=float(d.get("a_km", d.get("semi_major_axis_km", 7000.0))),
            ecc=float(d.get("ecc", d.get("e", 0.0))),
            inc_deg=float(d.get("inc_deg", d.get("inclination_deg", 0.0))),
            raan_deg=float(d.get("raan_deg", 0.0)),
            argp_deg=float(d.get("argp_deg", d.get("arg_periapsis_deg", 0.0))),
            true_anomaly_deg=float(d.get("ta_deg", d.get("true_anomaly_deg", 0.0))),
        )

    raise ValueError(
        "initial_state does not contain a supported orbital-state form. "
        "Use Cartesian position/velocity, coes, tle, CR3BP, a relative state, "
        "or explicit default_circular_earth: true."
    )


def _default_truth_from_agent(agent_cfg: Any, t_s: float = 0.0, target_jd_utc: float | None = None) -> StateTruth:
    s0 = dict(agent_cfg.initial_state or {})
    specs = dict(agent_cfg.specs or {})
    if ("dry_mass_kg" in specs) or ("fuel_mass_kg" in specs):
        dry_mass_kg = float(specs.get("dry_mass_kg", 0.0))
        fuel_mass_kg = float(specs.get("fuel_mass_kg", 0.0))
        if dry_mass_kg < 0.0 or fuel_mass_kg < 0.0:
            raise ValueError("dry_mass_kg and fuel_mass_kg must be non-negative.")
        mass_kg = dry_mass_kg + fuel_mass_kg
    else:
        mass_kg = float(specs.get("mass_kg", 300.0))
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("Object mass must be a positive finite value.")
    pos, vel = _rv_from_initial_state(s0, target_jd_utc=target_jd_utc)
    return StateTruth(
        position_eci_km=pos,
        velocity_eci_km_s=vel,
        attitude_quat_bn=np.array(s0.get("attitude_quat_bn", [1.0, 0.0, 0.0, 0.0]), dtype=float),
        angular_rate_body_rad_s=np.array(s0.get("angular_rate_body_rad_s", [0.0, 0.0, 0.0]), dtype=float),
        mass_kg=mass_kg,
        t_s=t_s,
    )


def _resolve_satellite_isp_s(specs: dict[str, Any]) -> float:
    if "isp_s" in specs:
        return float(specs.get("isp_s", 0.0))
    if "thruster_isp_s" in specs:
        return float(specs.get("thruster_isp_s", 0.0))
    thr = str(specs.get("thruster", "")).strip().upper()
    if thr in ("BASIC_CHEMICAL_BOTTOM_Z", "BASIC_CHEMICAL_Z_BOTTOM"):
        return float(BASIC_CHEMICAL_BOTTOM_Z.isp_s)
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
    if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
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
    if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
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
    if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
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


def _build_reaction_wheels(raw: Any) -> ReactionWheelLimits | None:
    if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
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


def _build_satellite_actuator_stack_from_specs(specs: dict[str, Any]) -> tuple[Any | None, dict[str, Any], bool]:
    raw = resolve_actuator_specs_from_satellite_specs(specs)
    if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
        return None, {}, False
    orbital_raw = dict(raw.get("orbital", {}) or {})
    attitude_raw = dict(raw.get("attitude", {}) or {})
    mount = resolve_thruster_mount_from_specs(specs)
    max_thrust_n = orbital_raw.get("max_thrust_n", resolve_thruster_max_thrust_n_from_specs(specs))
    isp_s = float(orbital_raw.get("isp_s", _resolve_satellite_isp_s(specs) or 220.0))
    default_direction = None if mount is None else np.array(mount.thrust_direction_body, dtype=float)
    default_position = None if mount is None else np.array(mount.position_body_m, dtype=float)

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
            if not isinstance(attitude_raw.get("magnetorquers"), dict)
            else MagnetorquerLimits(
                max_dipole_a_m2=np.array(
                    dict(attitude_raw.get("magnetorquers") or {}).get("max_dipole_a_m2", [0.0, 0.0, 0.0]),
                    dtype=float,
                ).reshape(-1)
            )
        ),
        thruster_pulse=(
            None
            if not isinstance(attitude_raw.get("thruster_pulse"), dict)
            else ThrusterPulseLimits(
                max_torque_nm=np.array(
                    dict(attitude_raw.get("thruster_pulse") or {}).get("max_torque_nm", [0.0, 0.0, 0.0]),
                    dtype=float,
                ).reshape(3),
                pulse_quantum_s=float(dict(attitude_raw.get("thruster_pulse") or {}).get("pulse_quantum_s", 0.02)),
            )
        ),
        control_moment_gyros=(
            None
            if not isinstance(attitude_raw.get("control_moment_gyros"), dict)
            else ControlMomentGyroLimits(
                max_torque_nm=dict(attitude_raw.get("control_moment_gyros") or {}).get("max_torque_nm", 0.0),
                momentum_nms=dict(attitude_raw.get("control_moment_gyros") or {}).get("momentum_nms", 0.0),
                gimbal_rate_limit_rad_s=dict(attitude_raw.get("control_moment_gyros") or {}).get(
                    "gimbal_rate_limit_rad_s", np.inf
                ),
                torque_time_constant_s=float(
                    dict(attitude_raw.get("control_moment_gyros") or {}).get("torque_time_constant_s", 0.0)
                ),
            )
        ),
        wheel_desaturation=(
            None
            if not isinstance(attitude_raw.get("wheel_desaturation"), dict)
            else WheelDesaturationLimits(
                momentum_fraction_threshold=float(
                    dict(attitude_raw.get("wheel_desaturation") or {}).get("momentum_fraction_threshold", 0.8)
                ),
                unload_gain_s_inv=float(dict(attitude_raw.get("wheel_desaturation") or {}).get("unload_gain_s_inv", 0.02)),
                max_unload_torque_nm=float(
                    dict(attitude_raw.get("wheel_desaturation") or {}).get("max_unload_torque_nm", 0.01)
                ),
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


def _resolve_chaser_relative_ric_init(initial_state: dict[str, Any]) -> tuple[np.ndarray, str, str] | None:
    s0 = dict(initial_state or {})
    rel_block = s0.get("relative_to_target_ric")
    if isinstance(rel_block, dict):
        frame = str(rel_block.get("frame", "rect")).strip().lower()
        reference_frame = str(rel_block.get("reference_frame", rel_block.get("origin", "target"))).strip().lower()
        state = np.array(rel_block.get("state", []), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_to_target_ric.state must be length-6.")
        if frame not in ("rect", "curv"):
            raise ValueError("chaser.initial_state.relative_to_target_ric.frame must be 'rect' or 'curv'.")
        return state, frame, reference_frame
    if "relative_ric_rect" in s0:
        state = np.array(s0.get("relative_ric_rect"), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_ric_rect must be length-6.")
        return state, "rect", "target"
    if "relative_ric_curv" in s0:
        state = np.array(s0.get("relative_ric_curv"), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_ric_curv must be length-6.")
        return state, "curv", "target"
    return None


def _resolve_relative_cislunar_init(initial_state: dict[str, Any]) -> np.ndarray | None:
    s0 = dict(initial_state or {})
    rel_block = s0.get("relative_to_target_cislunar")
    if isinstance(rel_block, dict):
        state = np.array(rel_block.get("state", []), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_to_target_cislunar.state must be length-6.")
        return state
    if "relative_cislunar" in s0:
        state = np.array(s0.get("relative_cislunar"), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_cislunar must be length-6.")
        return state
    return None


@dataclass
class AgentRuntime:
    object_id: str
    kind: str
    enabled: bool
    active: bool
    truth: StateTruth | None
    belief: StateBelief | None
    sensor: Any | None
    estimator: Any | None
    orbit_controller: Any | None
    attitude_controller: Any | None
    dynamics: OrbitalAttitudeDynamics | None
    knowledge_base: ObjectKnowledgeBase | None
    bridge: Any | None
    mission_strategy: Any | None
    mission_execution: Any | None
    rocket_sim: RocketAscentSimulator | None
    rocket_state: RocketState | None
    rocket_guidance: Any | None
    deploy_source: str | None
    deploy_time_s: float | None
    deploy_dv_body_m_s: np.ndarray | None
    initialization_delay_s: float
    control_available_time_s: float | None
    mission_modules: list[Any]
    waiting_for_launch: bool
    orbital_isp_s: float | None = None
    dry_mass_kg: float | None = None
    fuel_capacity_kg: float | None = None
    orbital_max_thrust_n: float | None = None
    thruster_direction_body: np.ndarray | None = None
    thruster_position_body_m: np.ndarray | None = None
    actuator: Any | None = None
    actuator_limits: dict[str, Any] = field(default_factory=dict)
    use_actuator_stack: bool = False
    mass_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RateLimitedController:
    base: Any
    period_s: float
    _last_eval_t_s: float | None = None
    _last_cmd: Command = field(default_factory=Command.zero, init=False)

    def __post_init__(self) -> None:
        self.period_s = float(max(self.period_s, 1e-9))

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if self._last_eval_t_s is None or float(t_s) - float(self._last_eval_t_s) >= self.period_s - 1e-12:
            if hasattr(self.base, "set_actuation_interval"):
                self.base.set_actuation_interval(float(t_s), float(t_s + self.period_s))
            self._last_cmd = self.base.act(belief, t_s, budget_ms)
            self._last_eval_t_s = float(t_s)
        return self._last_cmd

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(dict(state))

    def __getattr__(self, item: str) -> Any:
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(item)
        return getattr(base, item)


def _apply_relative_init_from_reference(
    *,
    agent: AgentRuntime,
    reference: AgentRuntime,
    initial_state: dict[str, Any],
) -> None:
    rel = _resolve_chaser_relative_ric_init(initial_state)
    if rel is None or agent.truth is None or reference.truth is None:
        return
    x_rel, frame, reference_frame = rel
    moon_state = cr3bp_moon_state_km_s()
    use_moon_ric = reference_frame.replace("-", "_") in {"moon", "moon_ric", "lunar", "lunar_ric"}
    origin_r = moon_state[:3] if use_moon_ric else np.zeros(3, dtype=float)
    origin_v = moon_state[3:] if use_moon_ric else np.zeros(3, dtype=float)
    r_t_abs = np.array(reference.truth.position_eci_km, dtype=float)
    v_t_abs = np.array(reference.truth.velocity_eci_km_s, dtype=float)
    r_t = r_t_abs - origin_r
    v_t = v_t_abs - origin_v
    r0 = float(np.linalg.norm(r_t))
    if r0 <= 0.0:
        return
    x_rel_rect = ric_curv_to_rect(x_rel, r0_km=r0) if frame == "curv" else np.array(x_rel, dtype=float).reshape(6)
    x_agent_eci = ric_rect_state_to_eci(x_rel_rect, r_t, v_t)
    agent.truth.position_eci_km = x_agent_eci[:3] + origin_r
    agent.truth.velocity_eci_km_s = x_agent_eci[3:] + origin_v
    if agent.belief is not None and agent.belief.state.size >= 6:
        agent.belief.state[:3] = agent.truth.position_eci_km
        agent.belief.state[3:6] = agent.truth.velocity_eci_km_s


def _apply_relative_cislunar_init_from_reference(
    *,
    agent: AgentRuntime,
    reference: AgentRuntime,
    initial_state: dict[str, Any],
) -> None:
    rel = _resolve_relative_cislunar_init(initial_state)
    if rel is None or agent.truth is None or reference.truth is None:
        return
    ref_state = np.hstack((reference.truth.position_eci_km, reference.truth.velocity_eci_km_s))
    state = ref_state + np.array(rel, dtype=float).reshape(6)
    agent.truth.position_eci_km = state[:3]
    agent.truth.velocity_eci_km_s = state[3:]
    if agent.belief is not None and agent.belief.state.size >= 6:
        agent.belief.state[:3] = agent.truth.position_eci_km
        agent.belief.state[3:6] = agent.truth.velocity_eci_km_s


def _apply_chaser_relative_init_from_target(
    *,
    chaser: AgentRuntime,
    target: AgentRuntime,
    initial_state: dict[str, Any],
) -> None:
    _apply_relative_init_from_reference(agent=chaser, reference=target, initial_state=initial_state)
    _apply_relative_cislunar_init_from_reference(agent=chaser, reference=target, initial_state=initial_state)


def _scenario_uses_aerodynamic_lift(cfg: SimulationScenarioConfig) -> bool:
    for agent_cfg in cfg.objects.values():
        if not bool(getattr(agent_cfg, "enabled", True)):
            continue
        if str(getattr(agent_cfg, "kind", "") or "").strip().lower() != "satellite":
            continue
        specs = dict(getattr(agent_cfg, "specs", {}) or {})
        props = resolve_vehicle_aero_properties(specs)
        if props.lift_axis_body is not None and float(props.cl) != 0.0:
            return True
    return False


def _build_orbit_propagator(cfg: SimulationScenarioConfig) -> OrbitPropagator:
    orbit = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    acceleration = dict(getattr(cfg.simulator, "acceleration", {}) or {})
    sh = dict(orbit.get("spherical_harmonics", {}) or {})
    sh_enabled = bool(sh.get("enabled", False))
    plugins = []
    if bool(orbit.get("j2", False)) and not sh_enabled:
        plugins.append(j2_plugin)
    if bool(orbit.get("j3", False)) and not sh_enabled:
        plugins.append(j3_plugin)
    if bool(orbit.get("j4", False)) and not sh_enabled:
        plugins.append(j4_plugin)
    if sh_enabled:
        plugins.append(spherical_harmonics_plugin)
    if bool(orbit.get("drag", False)):
        plugins.append(drag_plugin)
        if _scenario_uses_aerodynamic_lift(cfg):
            plugins.append(lift_plugin)
    if bool(orbit.get("srp", False)):
        plugins.append(srp_plugin)
    if bool(orbit.get("third_body_sun", False)):
        plugins.append(third_body_sun_plugin)
    if bool(orbit.get("third_body_moon", False)):
        plugins.append(third_body_moon_plugin)
    return OrbitPropagator(
        model=str(orbit.get("model", "two_body") or "two_body"),
        cr3bp_system_name=str(orbit.get("cr3bp_system", "earth_moon") or "earth_moon"),
        integrator=str(orbit.get("integrator", "rk4")),
        plugins=plugins,
        adaptive_atol=float(orbit.get("adaptive_atol", 1e-9)),
        adaptive_rtol=float(orbit.get("adaptive_rtol", 1e-7)),
        acceleration_mode=str(acceleration.get("mode", "off") or "off"),
    )


def _create_satellite_runtime(
    object_id: str,
    agent_cfg: Any,
    cfg: SimulationScenarioConfig,
    rng: np.random.Generator,
) -> AgentRuntime:
    initial_state = dict(agent_cfg.initial_state or {})
    truth = _default_truth_from_agent(agent_cfg, t_s=0.0, target_jd_utc=cfg.simulator.initial_jd_utc)
    specs = dict(agent_cfg.specs or {})
    inertia_kg_m2 = _resolve_satellite_inertia_kg_m2(specs)
    center_of_mass_body_m = resolve_center_of_mass_body_m(specs)
    aero_props = resolve_vehicle_aero_properties(
        specs,
        default_reference_area_m2=1.0,
        default_cd=2.2,
        default_cl=0.0,
        default_nose_radius_m=0.5,
        default_reference_length_m=1.0,
    )
    area_m2 = aero_props.reference_area_m2
    area_specified = aero_spec_get(specs, ("area_ref_m2", "area_m2", "reference_area_m2")) is not None
    drag_area_m2 = aero_props.drag_area_m2
    drag_area_specified = aero_spec_get(specs, ("drag_area_m2",)) is not None
    lift_area_m2 = aero_props.drag_area_m2 if aero_props.lift_area_m2 is None else aero_props.lift_area_m2
    lift_area_specified = aero_spec_get(specs, ("lift_area_m2",)) is not None
    lift_coefficient = aero_props.cl
    lift_axis_body = aero_props.lift_axis_body
    cp_offset_specified = aero_spec_get(specs, ("cp_offset_body_m", "center_of_pressure_offset_body_m")) is not None
    srp_area_m2, srp_area_specified = _satellite_spec_float(
        specs, ("srp_area_m2", "solar_area_m2"), default=area_m2
    )
    cd = aero_props.cd
    cd_specified = aero_spec_get(specs, ("cd", "drag_cd")) is not None
    cr, cr_specified = _satellite_spec_float(specs, ("cr", "srp_cr"), default=1.2)
    geometry_area_profile = _load_geometry_area_profile_from_specs(specs)
    noise = dict((agent_cfg.knowledge or {}).get("sensor_error", {}) or {})
    pos_sigma = float(np.array(noise.get("pos_sigma_km", [0.001])).reshape(-1)[0])
    vel_sigma = float(np.array(noise.get("vel_sigma_km_s", [1e-5])).reshape(-1)[0])
    acceleration = dict(getattr(cfg.simulator, "acceleration", {}) or {})
    orbit_estimator = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=float(cfg.simulator.dt_s),
        process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]),
        acceleration_mode=str(acceleration.get("mode", "off") or "off"),
    )
    orbit_ctrl_base = _module_obj(agent_cfg.orbit_control) or ZeroController()
    att_ctrl_base = _module_obj(agent_cfg.attitude_control) or ZeroTorqueController()
    orbit_cfg = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    att_cfg = dict(cfg.simulator.dynamics.get("attitude", {}) or {})
    attitude_enabled = bool(att_cfg.get("enabled", True))
    if attitude_enabled:
        belief = StateBelief(
            state=np.hstack(
                (truth.position_eci_km, truth.velocity_eci_km_s, truth.attitude_quat_bn, truth.angular_rate_body_rad_s)
            ),
            covariance=np.eye(13) * 1e-4,
            last_update_t_s=0.0,
        )
        quat_sigma = float(np.array(noise.get("quat_sigma", [1e-3])).reshape(-1)[0])
        omega_sigma = float(np.array(noise.get("omega_sigma_rad_s", [1e-4])).reshape(-1)[0])
        sensor = JointStateSensor(
            pos_sigma_km=pos_sigma,
            vel_sigma_km_s=vel_sigma,
            quat_sigma=quat_sigma,
            omega_sigma_rad_s=omega_sigma,
            update_cadence_s=float(cfg.simulator.dt_s),
            rng=rng,
        )
        estimator = JointStateEstimator(
            orbit_estimator=orbit_estimator,
            dt_s=float(cfg.simulator.dt_s),
            inertia_kg_m2=inertia_kg_m2,
            acceleration_mode=str(acceleration.get("mode", "off") or "off"),
        )
    else:
        belief = StateBelief(
            state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
            covariance=np.eye(6) * 1e-4,
            last_update_t_s=0.0,
        )
        sensor = NoisyOwnStateSensor(
            pos_sigma_km=pos_sigma,
            vel_sigma_km_s=vel_sigma,
            rng=rng,
            update_cadence_s=float(cfg.simulator.dt_s),
        )
        estimator = orbit_estimator
    dist_cfg = dict(att_cfg.get("disturbance_torques", {}) or {})
    orbit_ctrl = _RateLimitedController(
        base=orbit_ctrl_base,
        period_s=float(max(float(orbit_cfg.get("orbit_substep_s", cfg.simulator.dt_s) or cfg.simulator.dt_s), 1e-9)),
    )
    att_ctrl = (
        _RateLimitedController(
            base=att_ctrl_base,
            period_s=float(
                max(float(att_cfg.get("attitude_substep_s", cfg.simulator.dt_s) or cfg.simulator.dt_s), 1e-9)
            ),
        )
        if attitude_enabled
        else None
    )
    disturbance_config_kwargs: dict[str, Any] = {
        "use_gravity_gradient": bool(dist_cfg.get("gravity_gradient", False)),
        "use_magnetic": bool(dist_cfg.get("magnetic", False)),
        "use_drag": bool(dist_cfg.get("drag", False)),
        "use_srp": bool(dist_cfg.get("srp", False)),
        "center_of_mass_body_m": center_of_mass_body_m,
    }
    if drag_area_specified or area_specified:
        disturbance_config_kwargs["drag_area_m2"] = drag_area_m2
    if cd_specified:
        disturbance_config_kwargs["drag_cd"] = cd
    if cp_offset_specified:
        disturbance_config_kwargs["drag_cp_offset_body_m"] = aero_props.cp_offset_body_m
    if srp_area_specified or area_specified:
        disturbance_config_kwargs["srp_area_m2"] = srp_area_m2
    if cr_specified:
        disturbance_config_kwargs["srp_cr"] = cr
    if geometry_area_profile is not None:
        disturbance_config_kwargs["geometry_area_profile"] = geometry_area_profile
    disturbance_model = DisturbanceTorqueModel(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=inertia_kg_m2,
        config=DisturbanceTorqueConfig(**disturbance_config_kwargs),
    )
    dynamics = OrbitalAttitudeDynamics(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=inertia_kg_m2,
        disturbance_model=disturbance_model if attitude_enabled else None,
        area_m2=area_m2,
        cd=cd,
        cr=cr,
        drag_area_m2=drag_area_m2 if drag_area_specified else None,
        lift_area_m2=lift_area_m2 if lift_area_specified else None,
        lift_coefficient=lift_coefficient,
        lift_axis_body=lift_axis_body,
        srp_area_m2=srp_area_m2 if srp_area_specified else None,
        geometry_area_profile=geometry_area_profile,
        orbit_substep_s=float(orbit_cfg["orbit_substep_s"]) if orbit_cfg.get("orbit_substep_s") is not None else None,
        attitude_substep_s=float(att_cfg["attitude_substep_s"])
        if att_cfg.get("attitude_substep_s") is not None
        else None,
        propagate_attitude=attitude_enabled,
        orbit_propagator=_build_orbit_propagator(cfg),
        acceleration_mode=str(acceleration.get("mode", "off") or "off"),
    )
    bridge = _module_obj(agent_cfg.bridge) if (agent_cfg.bridge is not None and agent_cfg.bridge.enabled) else None
    mission_strategy_pointer = getattr(agent_cfg, "mission_strategy", None)
    mission_execution_pointer = getattr(agent_cfg, "mission_execution", None)
    mission_strategy = _module_obj(mission_strategy_pointer)
    mission_execution = _apply_thruster_mount_defaults(
        _module_obj(mission_execution_pointer), mission_execution_pointer, specs
    )
    mission_modules = [_module_obj(pointer) for pointer in list(agent_cfg.mission_objectives or [])]
    mission_modules = [module for module in mission_modules if module is not None]
    sat_isp_s = _resolve_satellite_isp_s(specs)
    sat_max_thrust_n = resolve_thruster_max_thrust_n_from_specs(specs)
    dry_mass_kg = specs.get("dry_mass_kg")
    fuel_capacity_kg = specs.get("fuel_mass_kg")
    thruster_mount = resolve_thruster_mount_from_specs(specs)
    actuator, actuator_limits, use_actuator_stack = _build_satellite_actuator_stack_from_specs(specs)
    initialization_delay_s = _initial_state_nonnegative_float(initial_state, "initialization_delay_s")
    return AgentRuntime(
        object_id=object_id,
        kind="satellite",
        enabled=bool(agent_cfg.enabled),
        active=bool(agent_cfg.enabled),
        truth=truth,
        belief=belief,
        sensor=sensor,
        estimator=estimator,
        orbit_controller=orbit_ctrl,
        attitude_controller=att_ctrl,
        dynamics=dynamics,
        knowledge_base=None,
        bridge=bridge,
        mission_strategy=mission_strategy,
        mission_execution=mission_execution,
        rocket_sim=None,
        rocket_state=None,
        rocket_guidance=None,
        deploy_source=str(initial_state.get("source", "")) or None,
        deploy_time_s=(
            float(initial_state.get("deploy_time_s"))
            if initial_state.get("deploy_time_s") is not None
            else None
        ),
        deploy_dv_body_m_s=np.array(
            initial_state.get("deploy_dv_body_m_s", [0.0, 0.0, 0.0]), dtype=float
        ),
        initialization_delay_s=initialization_delay_s,
        control_available_time_s=initialization_delay_s if bool(agent_cfg.enabled) else None,
        mission_modules=mission_modules,
        waiting_for_launch=False,
        orbital_isp_s=(None if sat_isp_s <= 0.0 else float(sat_isp_s)),
        dry_mass_kg=(None if dry_mass_kg is None else float(dry_mass_kg)),
        fuel_capacity_kg=(None if fuel_capacity_kg is None else float(fuel_capacity_kg)),
        orbital_max_thrust_n=sat_max_thrust_n,
        thruster_direction_body=(
            None if thruster_mount is None else np.array(thruster_mount.thrust_direction_body, dtype=float)
        ),
        thruster_position_body_m=(
            None if thruster_mount is None else np.array(thruster_mount.position_body_m, dtype=float)
        ),
        actuator=actuator,
        actuator_limits=actuator_limits,
        use_actuator_stack=use_actuator_stack,
        mass_properties=dict(specs.get("mass_properties", {}) or {}),
    )


def _resolve_rocket_stack(specs: dict[str, Any]) -> RocketStackPreset:
    preset = str(specs.get("preset_stack", "BASIC_TWO_STAGE_STACK")).strip().upper()
    ssto_stack = RocketStackPreset(name="Basic SSTO Stack", stages=(BASIC_SSTO_ROCKET,))
    by_name: dict[str, RocketStackPreset] = {
        "BASIC_TWO_STAGE_STACK": BASIC_TWO_STAGE_STACK,
        "BASIC_SSTO_STACK": ssto_stack,
        "BASIC_SSTO_ROCKET": ssto_stack,
        "BASIC_1ST_STAGE_STACK": RocketStackPreset(name="Basic 1st Stage Stack", stages=(BASIC_1ST_STAGE,)),
    }
    if preset not in by_name:
        valid = ", ".join(sorted(by_name.keys()))
        raise ValueError(f"Unknown rocket.specs.preset_stack '{preset}'. Valid options: {valid}")
    stack = by_name[preset]
    scales = {
        "dry_mass_scale": float(specs.get("dry_mass_scale", 1.0)),
        "propellant_mass_scale": float(specs.get("propellant_mass_scale", 1.0)),
        "thrust_scale": float(specs.get("thrust_scale", 1.0)),
        "isp_scale": float(specs.get("isp_scale", 1.0)),
    }
    if any((not np.isfinite(value)) or value <= 0.0 for value in scales.values()):
        raise ValueError("Rocket stage performance scales must be finite and positive.")
    if all(abs(value - 1.0) <= 1e-15 for value in scales.values()):
        return stack
    stages = tuple(
        replace(
            stage,
            dry_mass_kg=float(stage.dry_mass_kg) * scales["dry_mass_scale"],
            propellant_mass_kg=float(stage.propellant_mass_kg) * scales["propellant_mass_scale"],
            max_thrust_n=float(stage.max_thrust_n) * scales["thrust_scale"],
            isp_s=float(stage.isp_s) * scales["isp_scale"],
            sea_level_thrust_n=(
                None
                if stage.sea_level_thrust_n is None
                else float(stage.sea_level_thrust_n) * scales["thrust_scale"]
            ),
            vacuum_thrust_n=(
                None
                if stage.vacuum_thrust_n is None
                else float(stage.vacuum_thrust_n) * scales["thrust_scale"]
            ),
            sea_level_isp_s=(
                None if stage.sea_level_isp_s is None else float(stage.sea_level_isp_s) * scales["isp_scale"]
            ),
            vacuum_isp_s=(
                None if stage.vacuum_isp_s is None else float(stage.vacuum_isp_s) * scales["isp_scale"]
            ),
        )
        for stage in stack.stages
    )
    return RocketStackPreset(name=f"{stack.name} (scaled)", stages=stages)


def _build_rocket_guidance(agent_cfg: Any) -> RocketGuidanceLaw:
    base_pointer = getattr(agent_cfg, "base_guidance", None) or getattr(agent_cfg, "guidance", None)
    guidance = _module_obj(base_pointer) or OpenLoopPitchProgramGuidance()
    for modifier_pointer in list(getattr(agent_cfg, "guidance_modifiers", []) or []):
        modifier_obj = _module_obj(modifier_pointer, extra_kwargs={"base_guidance": guidance})
        if modifier_obj is not None:
            guidance = modifier_obj
    return guidance


def _create_rocket_runtime(
    cfg: SimulationScenarioConfig,
    object_id: str = "rocket",
    agent_cfg: Any | None = None,
) -> AgentRuntime:
    rc = cfg.rocket if agent_cfg is None else agent_cfg
    r_init = dict(rc.initial_state or {})
    r_specs = dict(rc.specs or {})
    orbit_dyn = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    att_dyn = dict(cfg.simulator.dynamics.get("attitude", {}) or {})
    rocket_dyn = dict(cfg.simulator.dynamics.get("rocket", {}) or {})
    object_aero = resolve_vehicle_aero_properties(
        r_specs,
        default_reference_area_m2=10.0,
        default_cd=0.20,
        default_cl=0.0,
        default_nose_radius_m=0.5,
        default_reference_length_m=30.0,
    )
    aero_dyn = dict(rocket_dyn.get("aero", {}) or {})
    rocket_area_ref_m2 = (
        float(rocket_dyn.get("area_ref_m2"))
        if rocket_dyn.get("area_ref_m2") is not None
        else (
            object_aero.reference_area_m2
            if aero_spec_get(r_specs, ("reference_area_m2", "area_ref_m2", "area_m2")) is not None
            else None
        )
    )
    atmosphere_env = dict(cfg.simulator.environment.get("atmosphere_env", {}) or {})
    earth_impact_policy = _earth_impact_policy_for_object(cfg.simulator.termination, object_id)
    aero_cfg = RocketAeroConfig(
        enabled=bool(rocket_dyn.get("aero_model_enabled", True)),
        reference_area_m2=float(aero_dyn.get("reference_area_m2", object_aero.reference_area_m2)),
        reference_length_m=float(aero_dyn.get("reference_length_m", object_aero.reference_length_m)),
        cp_offset_body_m=np.array(aero_dyn.get("cp_offset_body_m", object_aero.cp_offset_body_m), dtype=float),
        cd_base=float(aero_dyn.get("cd_base", aero_dyn.get("cd", object_aero.cd))),
        cd_alpha2=float(aero_dyn.get("cd_alpha2", 0.10)),
        cd_supersonic=float(aero_dyn.get("cd_supersonic", 0.28)),
        transonic_peak_cd=float(aero_dyn.get("transonic_peak_cd", 0.22)),
        transonic_mach=float(aero_dyn.get("transonic_mach", 1.0)),
        transonic_width=float(aero_dyn.get("transonic_width", 0.22)),
        cl_alpha_per_rad=float(aero_dyn.get("cl_alpha_per_rad", 0.15)),
        cy_beta_per_rad=float(aero_dyn.get("cy_beta_per_rad", 0.15)),
        cm_alpha_per_rad=float(aero_dyn.get("cm_alpha_per_rad", -0.02)),
        cn_beta_per_rad=float(aero_dyn.get("cn_beta_per_rad", -0.02)),
        cl_roll_per_rad=float(aero_dyn.get("cl_roll_per_rad", -0.01)),
        alpha_limit_deg=float(aero_dyn.get("alpha_limit_deg", 20.0)),
        beta_limit_deg=float(aero_dyn.get("beta_limit_deg", 20.0)),
    )
    rocket_inertia_kg_m2 = resolve_inertia_kg_m2(
        r_specs,
        default=np.array([[8.0e5, 0.0, 0.0], [0.0, 8.0e5, 0.0], [0.0, 0.0, 2.0e4]], dtype=float),
    )
    sim_cfg = RocketSimConfig(
        dt_s=float(cfg.simulator.dt_s),
        max_time_s=float(cfg.simulator.duration_s),
        target_altitude_km=float(rocket_dyn.get("target_altitude_km", 400.0)),
        target_altitude_tolerance_km=float(rocket_dyn.get("target_altitude_tolerance_km", 25.0)),
        target_eccentricity_max=float(rocket_dyn.get("target_eccentricity_max", 0.02)),
        insertion_hold_time_s=float(rocket_dyn.get("insertion_hold_time_s", 30.0)),
        launch_lat_deg=float(r_init.get("launch_lat_deg", 0.0)),
        launch_lon_deg=float(r_init.get("launch_lon_deg", 0.0)),
        launch_alt_km=float(r_init.get("launch_alt_km", 0.0)),
        launch_azimuth_deg=float(r_init.get("launch_azimuth_deg", 90.0)),
        atmosphere_model=str(rocket_dyn.get("atmosphere_model", "ussa1976")),
        enable_drag=bool(orbit_dyn.get("drag", True)),
        enable_srp=bool(orbit_dyn.get("srp", False)),
        enable_j2=bool(orbit_dyn.get("j2", True)),
        enable_j3=bool(orbit_dyn.get("j3", False)),
        enable_j4=bool(orbit_dyn.get("j4", False)),
        terminate_on_earth_impact=bool(earth_impact_policy.get("earth_impact_enabled", True)),
        earth_impact_radius_km=float(earth_impact_policy.get("earth_radius_km", 6378.137)),
        area_ref_m2=rocket_area_ref_m2,
        use_stagewise_aero_geometry=bool(rocket_dyn.get("use_stagewise_aero_geometry", True)),
        cd=float(rocket_dyn.get("cd", 0.35)),
        cr=float(rocket_dyn.get("cr", 1.2)),
        aero=aero_cfg,
        atmosphere_env=atmosphere_env,
        use_wgs84_geodesy=bool(rocket_dyn.get("use_wgs84_geodesy", True)),
        wind_enu_m_s=np.array(rocket_dyn.get("wind_enu_m_s", [0.0, 0.0, 0.0]), dtype=float),
        inertia_kg_m2=rocket_inertia_kg_m2,
        attitude_substep_s=float(rocket_dyn.get("attitude_substep_s", att_dyn.get("attitude_substep_s", 0.02)) or 0.02),
        attitude_mode=str(rocket_dyn.get("attitude_mode", "dynamic")),
        tvc_time_constant_s=float(rocket_dyn.get("tvc_time_constant_s", 0.1)),
        tvc_max_gimbal_deg=float(rocket_dyn.get("tvc_max_gimbal_deg", 6.0)),
        tvc_rate_limit_deg_s=float(rocket_dyn.get("tvc_rate_limit_deg_s", 20.0)),
        tvc_pivot_offset_body_m=np.array(rocket_dyn.get("tvc_pivot_offset_body_m", [0.0, 0.0, 0.0]), dtype=float),
    )
    vehicle_cfg = RocketVehicleConfig(
        stack=_resolve_rocket_stack(dict(rc.specs or {})),
        payload_mass_kg=float(r_specs.get("payload_mass_kg", 150.0)),
        thrust_axis_body=np.array(r_specs.get("thrust_axis_body", [1.0, 0.0, 0.0]), dtype=float),
    )
    guidance = _build_rocket_guidance(rc)
    if bool(rocket_dyn.get("tvc_steering_enabled", False)):
        guidance = TVCSteeringGuidance(
            base_guidance=guidance, pass_through_attitude=bool(rocket_dyn.get("tvc_pass_through_attitude", True))
        )
    if bool(rocket_dyn.get("orbit_insertion_cutoff_enabled", False)):
        guidance = OrbitInsertionCutoffGuidance(
            base_guidance=guidance,
            min_cutoff_alt_km=float(rocket_dyn.get("cutoff_min_alt_km", 80.0)),
            min_periapsis_alt_km=float(rocket_dyn.get("cutoff_min_periapsis_alt_km", 120.0)),
            apoapsis_margin_km=float(rocket_dyn.get("cutoff_apoapsis_margin_km", 5.0)),
            energy_margin_km2_s2=float(rocket_dyn.get("cutoff_energy_margin_km2_s2", 0.0)),
            ecc_relax_factor=float(rocket_dyn.get("cutoff_ecc_relax_factor", 2.0)),
            hard_escape_cutoff=bool(rocket_dyn.get("cutoff_hard_escape_enabled", True)),
            near_escape_speed_margin_frac=float(rocket_dyn.get("cutoff_near_escape_speed_margin_frac", 0.03)),
        )
    if bool(rocket_dyn.get("max_q_limiter_enabled", False)):
        guidance = MaxQThrottleLimiterGuidance(
            base_guidance=guidance,
            max_q_pa=float(rocket_dyn.get("max_q_pa", 45000.0)),
            min_throttle=float(rocket_dyn.get("min_throttle", 0.0)),
        )
    rocket_sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=guidance)
    rocket_state = rocket_sim.initial_state()
    truth = _rocket_state_to_truth(rocket_state)
    belief = StateBelief(
        state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
        covariance=np.eye(6) * 1e-4,
        last_update_t_s=0.0,
    )
    bridge = _module_obj(rc.bridge) if (rc.bridge is not None and rc.bridge.enabled) else None
    mission_strategy = _module_obj(getattr(rc, "mission_strategy", None))
    mission_execution = _module_obj(getattr(rc, "mission_execution", None))
    mission_modules = [_module_obj(pointer) for pointer in list(rc.mission_objectives or [])]
    mission_modules = [module for module in mission_modules if module is not None]
    return AgentRuntime(
        object_id=object_id,
        kind="rocket",
        enabled=bool(rc.enabled),
        active=bool(rc.enabled),
        truth=truth,
        belief=belief,
        sensor=None,
        estimator=None,
        orbit_controller=None,
        attitude_controller=None,
        dynamics=None,
        knowledge_base=None,
        bridge=bridge,
        mission_strategy=mission_strategy,
        mission_execution=mission_execution,
        rocket_sim=rocket_sim,
        rocket_state=rocket_state,
        rocket_guidance=guidance,
        deploy_source=None,
        deploy_time_s=None,
        deploy_dv_body_m_s=None,
        initialization_delay_s=0.0,
        control_available_time_s=0.0,
        mission_modules=mission_modules,
        waiting_for_launch=False,
        orbital_isp_s=None,
        dry_mass_kg=None,
        fuel_capacity_kg=None,
        orbital_max_thrust_n=None,
        thruster_direction_body=None,
        thruster_position_body_m=None,
        mass_properties=dict(r_specs.get("mass_properties", {}) or {}),
    )


def _knowledge_ekf_diag(value: Any, default: list[float]) -> np.ndarray:
    arr = np.array(value if value is not None else default, dtype=float).reshape(-1)
    if arr.size != 6:
        return np.array(default, dtype=float)
    return arr


def _knowledge_maneuver_detection_config(value: Any) -> EKFManeuverDetectionConfig:
    raw = dict(value or {})
    return EKFManeuverDetectionConfig(
        enabled=bool(raw.get("enabled", False)),
        warning_probability=float(raw.get("warning_probability", 0.99)),
        detection_probability=float(raw.get("detection_probability", 0.999)),
        window_size=int(raw.get("window_size", 5)),
        warning_count=int(raw.get("warning_count", 3)),
        detection_count=int(raw.get("detection_count", 3)),
        min_updates=int(raw.get("min_updates", 3)),
        cooldown_updates=int(raw.get("cooldown_updates", 0)),
    )


def _build_knowledge_base(
    observer_id: str, agent_cfg: Any, dt_s: float, rng: np.random.Generator
) -> ObjectKnowledgeBase | None:
    knowledge = dict(agent_cfg.knowledge or {})
    targets = list(knowledge.get("targets", []) or [])
    if not targets:
        return None
    conditions = dict(knowledge.get("conditions", {}) or {})
    noise = dict(knowledge.get("sensor_error", {}) or {})
    estimation = dict(knowledge.get("estimation", {}) or {})
    ekf_cfg = dict(estimation.get("ekf", knowledge.get("ekf", {})) or {})
    maneuver_detection_cfg = dict(
        estimation.get("maneuver_detection", ekf_cfg.get("maneuver_detection", knowledge.get("maneuver_detection", {}))) or {}
    )
    initial_track_state = ekf_cfg.get("initial_state_eci_km_s", estimation.get("initial_state_eci_km_s"))
    tracked: list[TrackedObjectConfig] = []
    for target_id in targets:
        tracked.append(
            TrackedObjectConfig(
                target_id=str(target_id),
                conditions=KnowledgeConditionConfig(
                    refresh_rate_s=float(knowledge.get("refresh_rate_s", dt_s)),
                    max_range_km=conditions.get("max_range_km"),
                    fov_half_angle_rad=conditions.get("fov_half_angle_rad"),
                    solid_angle_sr=conditions.get("solid_angle_sr"),
                    require_line_of_sight=bool(conditions.get("require_line_of_sight", False)),
                    dropout_prob=float(conditions.get("dropout_prob", 0.0)),
                    sensor_position_body_m=np.array(
                        conditions.get("sensor_position_body_m", [0.0, 0.0, 0.0]), dtype=float
                    ),
                    sensor_boresight_body=(
                        np.array(conditions.get("sensor_boresight_body"), dtype=float)
                        if conditions.get("sensor_boresight_body") is not None
                        else None
                    ),
                ),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.array(noise.get("pos_sigma_km", [0.01, 0.01, 0.01]), dtype=float),
                    vel_sigma_km_s=np.array(noise.get("vel_sigma_km_s", [1e-4, 1e-4, 1e-4]), dtype=float),
                    pos_bias_km=np.array(noise.get("pos_bias_km", [0.0, 0.0, 0.0]), dtype=float),
                    vel_bias_km_s=np.array(noise.get("vel_bias_km_s", [0.0, 0.0, 0.0]), dtype=float),
                    range_sigma_km=float(noise.get("range_sigma_km", 0.01)),
                    range_rate_sigma_km_s=float(noise.get("range_rate_sigma_km_s", 1e-4)),
                    angle_sigma_rad=float(noise.get("angle_sigma_rad", 1e-4)),
                    range_bias_km=float(noise.get("range_bias_km", 0.0)),
                    range_rate_bias_km_s=float(noise.get("range_rate_bias_km_s", 0.0)),
                    az_bias_rad=float(noise.get("az_bias_rad", 0.0)),
                    el_bias_rad=float(noise.get("el_bias_rad", 0.0)),
                ),
                estimator=str(estimation.get("type", "ekf")),
                measurement_model=str(estimation.get("measurement_model", "state")),
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=_knowledge_ekf_diag(
                        ekf_cfg.get("process_noise_diag"),
                        [1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10],
                    ),
                    meas_noise_diag=_knowledge_ekf_diag(
                        ekf_cfg.get("meas_noise_diag"),
                        [1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10],
                    ),
                    init_cov_diag=_knowledge_ekf_diag(
                        ekf_cfg.get("init_cov_diag"),
                        [1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2],
                    ),
                    initial_state_eci_km_s=(
                        None
                        if initial_track_state is None
                        else np.array(initial_track_state, dtype=float).reshape(6)
                    ),
                    initial_state_ric=(
                        None
                        if ekf_cfg.get("initial_state_ric", estimation.get("initial_state_ric")) is None
                        else np.array(ekf_cfg.get("initial_state_ric", estimation.get("initial_state_ric")), dtype=float).reshape(6)
                    ),
                    mean_motion_rad_s=(
                        None
                        if ekf_cfg.get("mean_motion_rad_s", estimation.get("mean_motion_rad_s")) is None
                        else float(ekf_cfg.get("mean_motion_rad_s", estimation.get("mean_motion_rad_s")))
                    ),
                    measurement_origin=str(ekf_cfg.get("measurement_origin", estimation.get("measurement_origin", "deputy"))),
                    integration_substep_s=float(ekf_cfg.get("integration_substep_s", 10.0)),
                ),
                maneuver_detection=_knowledge_maneuver_detection_config(maneuver_detection_cfg),
            )
        )
    return ObjectKnowledgeBase(
        observer_id=observer_id, tracked_objects=tracked, dt_s=dt_s, rng=rng, mu_km3_s2=EARTH_MU_KM3_S2
    )


def _deploy_from_rocket(agent: AgentRuntime, rocket: AgentRuntime, t_next: float) -> None:
    if (
        agent.kind != "satellite"
        or agent.active
        or agent.deploy_source not in {"rocket_deployment", "rocket_insertion"}
        or rocket.rocket_state is None
    ):
        return
    c_bn = quaternion_to_dcm_bn(rocket.rocket_state.attitude_quat_bn)
    dv_body = np.array(agent.deploy_dv_body_m_s if agent.deploy_dv_body_m_s is not None else np.zeros(3), dtype=float)
    dv_eci_km_s = (c_bn.T @ dv_body) / 1e3
    rs = rocket.rocket_state
    mass_kg = float(agent.truth.mass_kg) if agent.truth is not None else 200.0
    agent.truth = StateTruth(
        position_eci_km=np.array(rs.position_eci_km, dtype=float),
        velocity_eci_km_s=np.array(rs.velocity_eci_km_s, dtype=float) + dv_eci_km_s,
        attitude_quat_bn=np.array(rs.attitude_quat_bn, dtype=float),
        angular_rate_body_rad_s=np.array(rs.angular_rate_body_rad_s, dtype=float),
        mass_kg=mass_kg,
        t_s=t_next,
    )
    if agent.belief is not None and agent.belief.state.size >= 13:
        agent.belief = StateBelief(
            state=np.hstack(
                (
                    agent.truth.position_eci_km,
                    agent.truth.velocity_eci_km_s,
                    agent.truth.attitude_quat_bn,
                    agent.truth.angular_rate_body_rad_s,
                )
            ),
            covariance=np.eye(13) * 1e-4,
            last_update_t_s=t_next,
        )
    else:
        agent.belief = StateBelief(
            state=np.hstack((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s)),
            covariance=np.eye(6) * 1e-4,
            last_update_t_s=t_next,
        )
    agent.control_available_time_s = float(t_next) + float(max(agent.initialization_delay_s, 0.0))
    agent.active = True


def _run_mission_modules(
    *,
    agent: AgentRuntime,
    t_s: float,
    dt_s: float,
    env: dict[str, Any],
    orbit_controller: Any | None = None,
    attitude_controller: Any | None = None,
    orb_belief: StateBelief | None = None,
    att_belief: StateBelief | None = None,
) -> dict[str, Any]:
    if not agent.mission_modules:
        return {}
    truth = _decision_truth_from_belief(agent)
    if truth is None:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    out: dict[str, Any] = {}
    for module in agent.mission_modules:
        if not hasattr(module, "update"):
            continue
        ret = _call_with_compat_kwargs(
            module.update,
            primary_kwargs={
                "object_id": agent.object_id,
                "truth": truth,
                "belief": agent.belief,
                "own_knowledge": own_knowledge,
                "env": env,
                "t_s": t_s,
                "dt_s": dt_s,
                "orbit_controller": orbit_controller,
                "attitude_controller": attitude_controller,
                "orb_belief": orb_belief,
                "att_belief": att_belief,
                "rocket_state": agent.rocket_state,
                "rocket_vehicle_cfg": (agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
            },
            fallback_kwargs={"truth": truth, "t_s": t_s},
        )
        if isinstance(ret, dict):
            out.update(ret)
    return out


def _run_mission_strategy(
    *,
    agent: AgentRuntime,
    t_s: float,
    dt_s: float,
    env: dict[str, Any],
    orbit_controller: Any | None = None,
    attitude_controller: Any | None = None,
    orb_belief: StateBelief | None = None,
    att_belief: StateBelief | None = None,
) -> dict[str, Any]:
    strategy = agent.mission_strategy
    if strategy is None:
        return {}
    truth = _decision_truth_from_belief(agent)
    if truth is None:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    for method_name in ("update", "plan", "decide"):
        if not hasattr(strategy, method_name):
            continue
        method = getattr(strategy, method_name)
        ret = _call_with_compat_kwargs(
            method,
            primary_kwargs={
                "object_id": agent.object_id,
                "truth": truth,
                "belief": agent.belief,
                "own_knowledge": own_knowledge,
                "env": env,
                "t_s": t_s,
                "dt_s": dt_s,
                "orbit_controller": orbit_controller,
                "attitude_controller": attitude_controller,
                "orb_belief": orb_belief,
                "att_belief": att_belief,
                "rocket_state": agent.rocket_state,
                "rocket_vehicle_cfg": (agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
                "dry_mass_kg": agent.dry_mass_kg,
                "fuel_capacity_kg": agent.fuel_capacity_kg,
            },
            fallback_kwargs={"truth": truth, "t_s": t_s},
        )
        return ret if isinstance(ret, dict) else {}
    return {}


def _run_mission_execution(
    *,
    agent: AgentRuntime,
    intent: dict[str, Any],
    t_s: float,
    dt_s: float,
    env: dict[str, Any],
    orbit_controller: Any | None = None,
    attitude_controller: Any | None = None,
    orb_belief: StateBelief | None = None,
    att_belief: StateBelief | None = None,
) -> dict[str, Any]:
    execution = intent.get("_mission_execution_override", agent.mission_execution)
    if execution is None:
        return {}
    truth = _decision_truth_from_belief(agent)
    if truth is None:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    for method_name in ("update", "execute", "act"):
        if not hasattr(execution, method_name):
            continue
        method = getattr(execution, method_name)
        ret = _call_with_compat_kwargs(
            method,
            primary_kwargs={
                "intent": dict(intent or {}),
                "object_id": agent.object_id,
                "truth": truth,
                "belief": agent.belief,
                "own_knowledge": own_knowledge,
                "env": env,
                "t_s": t_s,
                "dt_s": dt_s,
                "orbit_controller": orbit_controller,
                "attitude_controller": attitude_controller,
                "orb_belief": orb_belief,
                "att_belief": att_belief,
                "rocket_state": agent.rocket_state,
                "rocket_vehicle_cfg": (agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
                "dry_mass_kg": agent.dry_mass_kg,
                "fuel_capacity_kg": agent.fuel_capacity_kg,
                "orbital_isp_s": agent.orbital_isp_s,
                "orbit_command_period_s": float(env.get("orbit_command_period_s", dt_s)),
            },
            fallback_kwargs={"intent": dict(intent or {}), "truth": truth, "t_s": t_s},
        )
        return ret if isinstance(ret, dict) else {}
    return {}


def _rocket_altitude_km(r_eci_km: np.ndarray, t_s: float, sim_cfg: RocketSimConfig) -> float:
    if not bool(getattr(sim_cfg, "use_wgs84_geodesy", False)):
        return float(np.linalg.norm(r_eci_km) - EARTH_RADIUS_KM)
    frame_context = frame_context_from_environment(dict(getattr(sim_cfg, "atmosphere_env", {}) or {}))
    r_ecef = transform_position(np.array(r_eci_km, dtype=float), "eci", "ecef", t_s=float(t_s), context=frame_context)
    _, _, alt_km = ecef_to_geodetic_deg_km(r_ecef)
    return float(alt_km)


def _orbital_elements_basic(
    r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = EARTH_MU_KM3_S2
) -> tuple[float, float]:
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    a = np.inf if abs(eps) < 1e-14 else float(-mu_km3_s2 / (2.0 * eps))
    h = np.cross(r_km, v_km_s)
    e_vec = np.cross(v_km_s, h) / mu_km3_s2 - r_km / r
    return a, float(np.linalg.norm(e_vec))
