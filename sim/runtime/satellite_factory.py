"""Satellite propagator and runtime construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim.aero import aero_spec_get, resolve_vehicle_aero_properties
from sim.config import SimulationScenarioConfig
from sim.digital_twin.mass_properties import resolve_center_of_mass_body_m
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
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
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile
from sim.presets.thrusters import resolve_thruster_max_thrust_n_from_specs, resolve_thruster_mount_from_specs
from sim.runtime.actuator_factory import (
    _initial_state_nonnegative_float,
    _resolve_satellite_inertia_kg_m2,
    _resolve_satellite_isp_s,
    _satellite_spec_float,
)
from sim.runtime.models import AgentRuntime
from sim.runtime.state_initialization import _default_truth_from_agent


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


def _scenario_uses_aerodynamic_lift(cfg: SimulationScenarioConfig) -> bool:
    game = dict(getattr(cfg, "metadata", {}).get("game", {}) or {})
    if str(game.get("control_mode", "") or "").strip().lower() in {
        "aerodynamic",
        "aero",
        "aero_control",
        "aerodynamic_control",
    }:
        aero = dict(game.get("aerodynamic_control", {}) or {})
        if float(aero.get("lift_coefficient", 0.45) or 0.0) > 0.0 and float(
            aero.get("lift_area_m2", 20.0) or 0.0
        ) > 0.0:
            return True
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


def _build_orbit_propagator(
    cfg: SimulationScenarioConfig,
    *,
    scenario_uses_aerodynamic_lift: bool | None = None,
) -> OrbitPropagator:
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
        uses_aerodynamic_lift = (
            _scenario_uses_aerodynamic_lift(cfg)
            if scenario_uses_aerodynamic_lift is None
            else bool(scenario_uses_aerodynamic_lift)
        )
        if uses_aerodynamic_lift:
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
    *,
    scenario_uses_aerodynamic_lift: bool | None = None,
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
    acceleration = dict(getattr(cfg.simulator, "acceleration", {}) or {})
    orbit_cfg = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    att_cfg = dict(cfg.simulator.dynamics.get("attitude", {}) or {})
    attitude_enabled = bool(att_cfg.get("enabled", True))
    dist_cfg = dict(att_cfg.get("disturbance_torques", {}) or {})
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
        orbit_propagator=_build_orbit_propagator(
            cfg,
            scenario_uses_aerodynamic_lift=scenario_uses_aerodynamic_lift,
        ),
        acceleration_mode=str(acceleration.get("mode", "off") or "off"),
    )
    sat_isp_s = _resolve_satellite_isp_s(specs)
    sat_max_thrust_n = resolve_thruster_max_thrust_n_from_specs(specs)
    dry_mass_kg = specs.get("dry_mass_kg")
    fuel_capacity_kg = specs.get("fuel_mass_kg")
    thruster_mount = resolve_thruster_mount_from_specs(specs)
    initialization_delay_s = _initial_state_nonnegative_float(initial_state, "initialization_delay_s")
    runtime = AgentRuntime(
        object_id=object_id,
        kind="satellite",
        enabled=bool(agent_cfg.enabled),
        active=bool(agent_cfg.enabled),
        truth=truth,
        belief=None,
        sensor=None,
        estimator=None,
        orbit_controller=None,
        attitude_controller=None,
        dynamics=dynamics,
        knowledge_base=None,
        bridge=None,
        mission_strategy=None,
        mission_execution=None,
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
        mission_modules=[],
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
        actuator=None,
        actuator_limits={},
        use_actuator_stack=False,
        mass_properties=dict(specs.get("mass_properties", {}) or {}),
        runtime_profile=str(getattr(agent_cfg, "runtime_profile", "flight_software") or "flight_software"),
    )
    if runtime.runtime_profile != "trajectory_only":
        from sim.runtime.satellites.factory import build_satellite_flight_software_runtime

        runtime.flight_software_runtime = build_satellite_flight_software_runtime(
            object_id=object_id,
            agent_cfg=agent_cfg,
            scenario_cfg=cfg,
            mass_kg=float(truth.mass_kg),
            specific_impulse_s=(None if sat_isp_s <= 0.0 else float(sat_isp_s)),
            dry_mass_kg=(0.0 if dry_mass_kg is None else float(dry_mass_kg)),
        )
    return runtime
