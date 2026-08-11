from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import sim.aero.core as aero_core
from sim.acceleration.settings import acceleration_enabled_from_mode
from sim.core.interfaces import DynamicsModel
from sim.core.models import Command, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueModel
from sim.dynamics.attitude.rigid_body import guard_attitude_state_output, propagate_attitude_exponential_map
from sim.dynamics.coupled_satellite import (
    CoupledDerivative,
    CoupledIntegratorConfig,
    CoupledSatelliteIntegrator,
    CoupledSatelliteState,
)
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile, RectangularPrismGeometry
from sim.utils.quaternion import quaternion_to_dcm_bn


def _owned_default_orbit_propagator() -> OrbitPropagator:
    propagator = OrbitPropagator(integrator="rk4")
    propagator._pending_orbital_attitude_default_configuration = True
    return propagator


@dataclass(frozen=True)
class OrbitalAttitudeDynamics(DynamicsModel):
    mu_km3_s2: float
    inertia_kg_m2: np.ndarray
    disturbance_model: DisturbanceTorqueModel | None = None
    area_m2: float = 1.0
    cd: float = 2.2
    cr: float = 1.2
    drag_area_m2: float | None = None
    lift_area_m2: float | None = None
    lift_coefficient: float = 0.0
    lift_axis_body: np.ndarray | None = None
    srp_area_m2: float | None = None
    use_rectangular_prism_for_aero_srp: bool = False
    rectangular_prism_dims_m: tuple[float, float, float] | None = None
    geometry_area_profile: GeometryAreaProfile | None = None
    orbit_substep_s: float | None = None
    attitude_substep_s: float | None = None
    propagate_attitude: bool = True
    orbit_propagator: OrbitPropagator = field(default_factory=_owned_default_orbit_propagator)
    acceleration_mode: str = "off"
    _acceleration_enabled: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.geometry_area_profile is not None and self.use_rectangular_prism_for_aero_srp:
            raise ValueError("Use either geometry_area_profile or rectangular prism aero/SRP mode, not both.")
        if self.use_rectangular_prism_for_aero_srp:
            if self.rectangular_prism_dims_m is None:
                raise ValueError(
                    "rectangular_prism_dims_m must be provided when use_rectangular_prism_for_aero_srp=True."
                )
            if self.disturbance_model is None:
                raise ValueError(
                    "Rectangular prism aero/SRP mode requires coupled orbit+attitude disturbance simulation "
                    "(disturbance_model must be set)."
                )
        acceleration_enabled = acceleration_enabled_from_mode(self.acceleration_mode)
        object.__setattr__(self, "_acceleration_enabled", bool(acceleration_enabled))
        if bool(getattr(self.orbit_propagator, "_pending_orbital_attitude_default_configuration", False)):
            self.orbit_propagator.acceleration_mode = self.acceleration_mode
            delattr(self.orbit_propagator, "_pending_orbital_attitude_default_configuration")

    def step(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        force_eci_n = np.asarray(command.mode_flags.get("physical_force_eci_n", (0.0, 0.0, 0.0)), dtype=float)
        force_body_n = np.asarray(command.mode_flags.get("physical_force_body_n", (0.0, 0.0, 0.0)), dtype=float)
        mass_flow_kg_s = float(command.mode_flags.get("mass_flow_kg_s", 0.0) or 0.0)
        if (
            self.propagate_attitude
            and self.orbit_propagator.state_frame == "eci"
            and (np.any(force_eci_n) or np.any(force_body_n) or mass_flow_kg_s > 0.0)
        ):
            return self._step_coupled_v2(state, command, env, dt_s)
        env_local = dict(env)
        q_force_evaluation = np.asarray(state.attitude_quat_bn, dtype=float)
        if self.propagate_attitude and dt_s > 0.0:
            q_force_evaluation, _ = propagate_attitude_exponential_map(
                quat_bn=q_force_evaluation,
                omega_body_rad_s=np.asarray(state.angular_rate_body_rad_s, dtype=float),
                inertia_kg_m2=self.inertia_kg_m2,
                torque_body_nm=np.asarray(command.torque_body_nm, dtype=float),
                dt_s=0.5 * float(dt_s),
                acceleration_mode=self.acceleration_mode,
            )
        if self.drag_area_m2 is not None and "drag_area_m2" not in env_local:
            env_local["drag_area_m2"] = float(self.drag_area_m2)
        if self.lift_area_m2 is not None and "lift_area_m2" not in env_local:
            env_local["lift_area_m2"] = float(self.lift_area_m2)
        if (
            self.lift_axis_body is not None
            and float(self.lift_coefficient) != 0.0
            and not bool(env_local.get("physical_aerodynamics", False))
        ):
            c_bn = quaternion_to_dcm_bn(q_force_evaluation)
            lift_axis_body = np.array(self.lift_axis_body, dtype=float).reshape(3)
            axis_norm = float(np.linalg.norm(lift_axis_body))
            if axis_norm > 0.0:
                env_local["lift_coefficient"] = float(self.lift_coefficient)
                env_local["lift_direction_eci"] = c_bn.T @ (lift_axis_body / axis_norm)
        if self.srp_area_m2 is not None:
            env_local["srp_area_m2"] = float(self.srp_area_m2)
        area_profile = self.geometry_area_profile
        geom = self._rectangular_prism_geometry()
        if area_profile is not None:
            c_bn = quaternion_to_dcm_bn(q_force_evaluation)
            omega_raw = env_local.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
            v_rel_eci_km_s = aero_core.atmosphere_relative_velocity_eci_km_s(
                state.position_eci_km,
                state.velocity_eci_km_s,
                t_s=float(state.t_s),
                earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                frame_model=str(env_local.get("drag_frame_model", "inertial_z")),
                jd_utc_start=env_local.get("jd_utc_start"),
                eop_path=env_local.get("drag_eop_path"),
                dut1_s=env_local.get("dut1_s"),
                xp_arcsec=env_local.get("xp_arcsec"),
                yp_arcsec=env_local.get("yp_arcsec"),
                dat_s=env_local.get("dat_s"),
                tt_minus_utc_s=env_local.get("tt_minus_utc_s"),
                ddpsi_rad=float(env_local.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(env_local.get("ddeps_rad", 0.0) or 0.0),
                eop_extrapolation=str(env_local.get("eop_extrapolation", "error") or "error"),
            )
            v_rel_body = c_bn @ v_rel_eci_km_s
            env_local["drag_area_m2"] = area_profile.projected_area_for_direction_m2(-v_rel_body)

            srp_geometry = resolve_srp_geometry(state.position_eci_km, state.t_s, env_local)
            sun_dir_eci = np.array(srp_geometry["sun_dir_sc_eci"], dtype=float)
            if float(np.linalg.norm(sun_dir_eci)) > 0.0:
                sun_dir_body = c_bn @ sun_dir_eci
                env_local["srp_area_m2"] = area_profile.projected_area_for_direction_m2(-sun_dir_body)
        elif self.use_rectangular_prism_for_aero_srp and geom is not None and self.disturbance_model is not None:
            c_bn = quaternion_to_dcm_bn(q_force_evaluation)
            omega_raw = env_local.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
            v_rel_eci_km_s = aero_core.atmosphere_relative_velocity_eci_km_s(
                state.position_eci_km,
                state.velocity_eci_km_s,
                t_s=float(state.t_s),
                earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                frame_model=str(env_local.get("drag_frame_model", "inertial_z")),
                jd_utc_start=env_local.get("jd_utc_start"),
                eop_path=env_local.get("drag_eop_path"),
                dut1_s=env_local.get("dut1_s"),
                xp_arcsec=env_local.get("xp_arcsec"),
                yp_arcsec=env_local.get("yp_arcsec"),
                dat_s=env_local.get("dat_s"),
                tt_minus_utc_s=env_local.get("tt_minus_utc_s"),
                ddpsi_rad=float(env_local.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(env_local.get("ddeps_rad", 0.0) or 0.0),
                eop_extrapolation=str(env_local.get("eop_extrapolation", "error") or "error"),
            )
            v_rel_body = c_bn @ v_rel_eci_km_s
            env_local["drag_area_m2"] = geom.projected_area_m2(-v_rel_body)

            srp_geometry = resolve_srp_geometry(state.position_eci_km, state.t_s, env_local)
            sun_dir_eci = np.array(srp_geometry["sun_dir_sc_eci"], dtype=float)
            if float(np.linalg.norm(sun_dir_eci)) > 0.0:
                sun_dir_body = c_bn @ sun_dir_eci
                env_local["srp_area_m2"] = geom.projected_area_m2(-sun_dir_body)

        x_orbit = np.empty(6, dtype=float)
        x_orbit[:3] = state.position_eci_km
        x_orbit[3:] = state.velocity_eci_km_s
        orbit_ctx = OrbitContext(
            mu_km3_s2=self.mu_km3_s2,
            mass_kg=state.mass_kg,
            area_m2=self.area_m2,
            cd=self.cd,
            cr=self.cr,
        )
        orbit_dt = self._effective_substep(self.orbit_substep_s, dt_s)
        command_acceleration_eci_km_s2 = np.asarray(command.thrust_eci_km_s2, dtype=float)
        physical_force_eci = command.mode_flags.get("physical_force_eci_n")
        physical_force_body = command.mode_flags.get("physical_force_body_n")
        if physical_force_eci is not None and physical_force_body is not None:
            c_bn_force = quaternion_to_dcm_bn(q_force_evaluation)
            total_force_eci_n = np.asarray(physical_force_eci, dtype=float).reshape(3) + c_bn_force.T @ np.asarray(
                physical_force_body,
                dtype=float,
            ).reshape(3)
            command_acceleration_eci_km_s2 = total_force_eci_n / max(float(state.mass_kg), 1.0e-12) / 1.0e3
        x_orbit_next = x_orbit.copy()
        if orbit_dt >= dt_s:
            x_orbit_next = self.orbit_propagator.propagate(
                x_eci=x_orbit_next,
                dt_s=dt_s,
                t_s=state.t_s,
                command_accel_eci_km_s2=command_acceleration_eci_km_s2,
                env=env_local,
                ctx=orbit_ctx,
            )
        else:
            t_local = state.t_s
            for h in self._substep_sequence(dt_s, orbit_dt):
                x_orbit_next = self.orbit_propagator.propagate(
                    x_eci=x_orbit_next,
                    dt_s=h,
                    t_s=t_local,
                    command_accel_eci_km_s2=command_acceleration_eci_km_s2,
                    env=env_local,
                    ctx=orbit_ctx,
                )
                t_local += h

        q_next = state.attitude_quat_bn.copy()
        w_next = state.angular_rate_body_rad_s.copy()
        if self.propagate_attitude:
            midpoint_truth = self._midpoint_translational_truth(
                state=state,
                x_orbit_next=x_orbit_next,
                dt_s=dt_s,
            )
            disturbance_cfg = getattr(self.disturbance_model, "config", None)
            if self.disturbance_model is not None and bool(getattr(disturbance_cfg, "use_drag", False)):
                if "density_kg_m3" not in env_local:
                    env_local["density_kg_m3"] = density_from_model(
                        str(env_local.get("atmosphere_model", "exponential")).lower(),
                        midpoint_truth.position_eci_km,
                        midpoint_truth.t_s,
                        env=env_local,
                    )
                omega_raw = env_local.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
                v_rel_eci_km_s = aero_core.atmosphere_relative_velocity_eci_km_s(
                    midpoint_truth.position_eci_km,
                    midpoint_truth.velocity_eci_km_s,
                    t_s=float(midpoint_truth.t_s),
                    earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                    frame_model=str(env_local.get("drag_frame_model", "inertial_z")),
                    jd_utc_start=env_local.get("jd_utc_start"),
                    eop_path=env_local.get("drag_eop_path"),
                    dut1_s=env_local.get("dut1_s"),
                    xp_arcsec=env_local.get("xp_arcsec"),
                    yp_arcsec=env_local.get("yp_arcsec"),
                    dat_s=env_local.get("dat_s"),
                    tt_minus_utc_s=env_local.get("tt_minus_utc_s"),
                    ddpsi_rad=float(env_local.get("ddpsi_rad", 0.0) or 0.0),
                    ddeps_rad=float(env_local.get("ddeps_rad", 0.0) or 0.0),
                    eop_extrapolation=str(env_local.get("eop_extrapolation", "error") or "error"),
                )
                v_rel_eci_m_s = v_rel_eci_km_s * 1e3
                env_local["drag_v_rel_eci_m_s"] = v_rel_eci_m_s
                env_local["drag_v_rel_norm_m_s"] = float(np.linalg.norm(v_rel_eci_m_s))

            if self.disturbance_model is not None and bool(getattr(disturbance_cfg, "use_srp", False)):
                srp_geometry = resolve_srp_geometry(midpoint_truth.position_eci_km, midpoint_truth.t_s, env_local)
                env_local["sun_dir_eci_unit"] = np.asarray(srp_geometry["sun_dir_sc_eci"], dtype=float)
                env_local["srp_distance_scale"] = float(srp_geometry["distance_scale"])
                if "srp_shadow_factor" not in env_local:
                    env_local["srp_shadow_factor"] = srp_shadow_factor(
                        r_sc_eci_km=midpoint_truth.position_eci_km,
                        t_s=midpoint_truth.t_s,
                        env=env_local,
                        srp_geometry=srp_geometry,
                    )
            att_dt = self._effective_substep(self.attitude_substep_s, dt_s)
            attitude_substeps = np.asarray(self._substep_sequence(dt_s, att_dt), dtype=float)
            compiled_propagator = getattr(self.disturbance_model, "try_propagate_compiled", None)
            if type(self.disturbance_model) is DisturbanceTorqueModel:
                compiled_result = self.disturbance_model.try_propagate_compiled(
                    quat_bn=q_next,
                    omega_body_rad_s=w_next,
                    command_torque_body_nm=command.torque_body_nm,
                    position_eci_km=midpoint_truth.position_eci_km,
                    t_s=midpoint_truth.t_s,
                    env=env_local,
                    substeps_s=attitude_substeps,
                    acceleration_mode=self.acceleration_mode,
                    acceleration_enabled=self._acceleration_enabled,
                )
            elif callable(compiled_propagator):
                compiled_result = compiled_propagator(
                    quat_bn=q_next,
                    omega_body_rad_s=w_next,
                    command_torque_body_nm=command.torque_body_nm,
                    position_eci_km=midpoint_truth.position_eci_km,
                    t_s=midpoint_truth.t_s,
                    env=env_local,
                    substeps_s=attitude_substeps,
                    acceleration_mode=self.acceleration_mode,
                )
            else:
                compiled_result = None
            if compiled_result is not None:
                q_next, w_next = compiled_result
            else:
                att_state = StateTruth(
                    position_eci_km=np.array(midpoint_truth.position_eci_km, dtype=float),
                    velocity_eci_km_s=np.array(midpoint_truth.velocity_eci_km_s, dtype=float),
                    attitude_quat_bn=np.array(q_next, dtype=float),
                    angular_rate_body_rad_s=np.array(w_next, dtype=float),
                    mass_kg=float(state.mass_kg),
                    t_s=float(midpoint_truth.t_s),
                )
                for h in attitude_substeps:
                    att_state.attitude_quat_bn = q_next
                    att_state.angular_rate_body_rad_s = w_next
                    disturbance_torque = (
                        np.zeros(3)
                        if self.disturbance_model is None
                        else self.disturbance_model.total_torque_body_nm(att_state, env_local)
                    )
                    total_torque = command.torque_body_nm + disturbance_torque
                    q_next, w_next = propagate_attitude_exponential_map(
                        quat_bn=q_next,
                        omega_body_rad_s=w_next,
                        inertia_kg_m2=self.inertia_kg_m2,
                        torque_body_nm=total_torque,
                        dt_s=float(h),
                        acceleration_mode=self.acceleration_mode,
                    )

        # Optional direct attitude state override for surrogate controller testing.
        if self.propagate_attitude:
            att_override = dict(command.mode_flags.get("attitude_state_override", {}) or {})
            if att_override:
                q_cmd = np.array(att_override.get("q_next_bn", q_next), dtype=float).reshape(-1)
                w_cmd = np.array(att_override.get("w_next_body_rad_s", w_next), dtype=float).reshape(-1)
                q_next, w_next = guard_attitude_state_output(q_cmd, w_cmd)
        delta_mass_kg = float(command.mode_flags.get("delta_mass_kg", 0.0))
        min_mass_kg = float(command.mode_flags.get("min_mass_kg", 0.0))
        if not np.isfinite(min_mass_kg):
            min_mass_kg = 0.0
        min_mass_kg = max(min_mass_kg, 0.0)
        mass_next = max(min_mass_kg, state.mass_kg - delta_mass_kg)

        return StateTruth(
            position_eci_km=x_orbit_next[:3],
            velocity_eci_km_s=x_orbit_next[3:],
            attitude_quat_bn=q_next,
            angular_rate_body_rad_s=w_next,
            mass_kg=mass_next,
            t_s=state.t_s + dt_s,
        )

    def _step_coupled_v2(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        """Run the production v2 stage-consistent coupled physics path."""

        force_eci_n = np.asarray(command.mode_flags.get("physical_force_eci_n", (0.0, 0.0, 0.0)), dtype=float)
        force_body_n = np.asarray(command.mode_flags.get("physical_force_body_n", (0.0, 0.0, 0.0)), dtype=float)
        commanded_torque = np.asarray(command.torque_body_nm, dtype=float)
        mass_flow_kg_s = max(float(command.mode_flags.get("mass_flow_kg_s", 0.0)), 0.0)
        minimum_mass_kg = max(float(command.mode_flags.get("min_mass_kg", 0.0)), 0.0)
        environment = dict(env)

        def derivative(t_s: float, stage: CoupledSatelliteState, _control: object) -> CoupledDerivative:
            propellant_available = stage.mass_kg > minimum_mass_kg + 1.0e-12
            active_force_eci = force_eci_n if propellant_available or mass_flow_kg_s == 0.0 else np.zeros(3)
            active_force_body = force_body_n if propellant_available or mass_flow_kg_s == 0.0 else np.zeros(3)
            active_mass_flow = mass_flow_kg_s if propellant_available else 0.0
            c_bn = quaternion_to_dcm_bn(stage.attitude_quat_bn)
            total_force_eci_n = active_force_eci + c_bn.T @ active_force_body
            stage_truth = StateTruth(
                position_eci_km=stage.position_eci_km,
                velocity_eci_km_s=stage.velocity_eci_km_s,
                attitude_quat_bn=stage.attitude_quat_bn,
                angular_rate_body_rad_s=stage.angular_rate_body_rad_s,
                mass_kg=stage.mass_kg,
                t_s=t_s,
            )
            stage_env = self._coupled_stage_environment(stage_truth, environment)
            orbit_ctx = OrbitContext(
                mu_km3_s2=self.mu_km3_s2,
                mass_kg=stage.mass_kg,
                area_m2=self.area_m2,
                cd=self.cd,
                cr=self.cr,
            )
            state6 = np.hstack((stage.position_eci_km, stage.velocity_eci_km_s))
            acceleration = self.orbit_propagator.acceleration_at(
                t_s=t_s,
                x_eci=state6,
                command_accel_eci_km_s2=total_force_eci_n / stage.mass_kg / 1.0e3,
                env=stage_env,
                ctx=orbit_ctx,
            )
            disturbance = (
                np.zeros(3)
                if self.disturbance_model is None
                else self.disturbance_model.total_torque_body_nm(stage_truth, stage_env)
            )
            total_torque = commanded_torque + disturbance
            omega = stage.angular_rate_body_rad_s
            angular_acceleration = np.linalg.solve(
                self.inertia_kg_m2,
                total_torque - np.cross(omega, self.inertia_kg_m2 @ omega),
            )
            return CoupledDerivative(
                position_rate_km_s=stage.velocity_eci_km_s,
                velocity_rate_km_s2=acceleration,
                angular_acceleration_body_rad_s2=angular_acceleration,
                mass_rate_kg_s=-active_mass_flow,
                actuator_state_rate=np.empty(0, dtype=float),
            )

        orbit_step = self._effective_substep(self.orbit_substep_s, dt_s)
        attitude_step = self._effective_substep(self.attitude_substep_s, dt_s)
        if (
            not np.any(commanded_torque)
            and not np.any(state.angular_rate_body_rad_s)
            and (
                self.disturbance_model is None
                or not bool(getattr(self.disturbance_model, "_compiled_has_disturbances", True))
            )
        ):
            # A static attitude makes the body-to-inertial force transform
            # constant. Preserve the exact coupled equations while avoiding
            # needless attitude microsteps during long coasts/steady burns.
            attitude_step = orbit_step
        integrator = CoupledSatelliteIntegrator(
            CoupledIntegratorConfig(orbit_step, attitude_step),
            derivative,
        )
        initial = CoupledSatelliteState(
            position_eci_km=state.position_eci_km,
            velocity_eci_km_s=state.velocity_eci_km_s,
            attitude_quat_bn=state.attitude_quat_bn,
            angular_rate_body_rad_s=state.angular_rate_body_rad_s,
            mass_kg=state.mass_kg,
            actuator_state=np.empty(0, dtype=float),
            t_s=state.t_s,
        )
        final = integrator.propagate(initial, end_time_s=state.t_s + dt_s).final_state
        q_next = final.attitude_quat_bn
        w_next = final.angular_rate_body_rad_s
        attitude_override = dict(command.mode_flags.get("attitude_state_override", {}) or {})
        if attitude_override:
            q_next, w_next = guard_attitude_state_output(
                np.asarray(attitude_override.get("q_next_bn", q_next), dtype=float),
                np.asarray(attitude_override.get("w_next_body_rad_s", w_next), dtype=float),
            )
        return StateTruth(
            position_eci_km=final.position_eci_km,
            velocity_eci_km_s=final.velocity_eci_km_s,
            attitude_quat_bn=q_next,
            angular_rate_body_rad_s=w_next,
            mass_kg=max(float(final.mass_kg), minimum_mass_kg),
            t_s=final.t_s,
        )

    def _coupled_stage_environment(self, state: StateTruth, env: dict) -> dict:
        stage_env = dict(env)
        if self.drag_area_m2 is not None:
            stage_env["drag_area_m2"] = float(self.drag_area_m2)
        if self.lift_area_m2 is not None:
            stage_env["lift_area_m2"] = float(self.lift_area_m2)
        if self.srp_area_m2 is not None:
            stage_env["srp_area_m2"] = float(self.srp_area_m2)
        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        if self.lift_axis_body is not None and float(self.lift_coefficient) != 0.0:
            axis = np.asarray(self.lift_axis_body, dtype=float).reshape(3)
            norm = float(np.linalg.norm(axis))
            if norm > 0.0:
                stage_env["lift_coefficient"] = float(self.lift_coefficient)
                stage_env["lift_direction_eci"] = c_bn.T @ (axis / norm)
        area_profile = self.geometry_area_profile
        geometry = self._rectangular_prism_geometry()
        if area_profile is not None or geometry is not None:
            omega_raw = stage_env.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
            relative_velocity = aero_core.atmosphere_relative_velocity_eci_km_s(
                state.position_eci_km,
                state.velocity_eci_km_s,
                t_s=float(state.t_s),
                earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                frame_model=str(stage_env.get("drag_frame_model", "inertial_z")),
                jd_utc_start=stage_env.get("jd_utc_start"),
                eop_path=stage_env.get("drag_eop_path"),
                dut1_s=stage_env.get("dut1_s"),
                xp_arcsec=stage_env.get("xp_arcsec"),
                yp_arcsec=stage_env.get("yp_arcsec"),
                dat_s=stage_env.get("dat_s"),
                tt_minus_utc_s=stage_env.get("tt_minus_utc_s"),
                ddpsi_rad=float(stage_env.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(stage_env.get("ddeps_rad", 0.0) or 0.0),
                eop_extrapolation=str(stage_env.get("eop_extrapolation", "error") or "error"),
            )
            body_velocity = c_bn @ relative_velocity
            area_owner = area_profile if area_profile is not None else geometry
            stage_env["drag_area_m2"] = area_owner.projected_area_for_direction_m2(-body_velocity) if area_profile is not None else area_owner.projected_area_m2(-body_velocity)
            srp_geometry = resolve_srp_geometry(state.position_eci_km, state.t_s, stage_env)
            sun_direction = np.asarray(srp_geometry["sun_dir_sc_eci"], dtype=float)
            if float(np.linalg.norm(sun_direction)) > 0.0:
                body_sun = c_bn @ sun_direction
                stage_env["srp_area_m2"] = area_owner.projected_area_for_direction_m2(-body_sun) if area_profile is not None else area_owner.projected_area_m2(-body_sun)
        return stage_env

    def _rectangular_prism_geometry(self) -> RectangularPrismGeometry | None:
        dims = self.rectangular_prism_dims_m
        if dims is None:
            return None
        return RectangularPrismGeometry(lx_m=float(dims[0]), ly_m=float(dims[1]), lz_m=float(dims[2]))

    @staticmethod
    def _effective_substep(substep_s: float | None, dt_s: float) -> float:
        if substep_s is None:
            return dt_s
        return max(min(float(substep_s), dt_s), 1e-9)

    @staticmethod
    def _substep_sequence(total_dt_s: float, h_s: float) -> list[float]:
        if h_s >= total_dt_s:
            return [float(total_dt_s)]
        n = int(np.floor(total_dt_s / h_s))
        steps = [float(h_s)] * n
        rem = float(total_dt_s - n * h_s)
        if rem > 1e-12:
            steps.append(rem)
        return steps

    @staticmethod
    def _midpoint_translational_truth(state: StateTruth, x_orbit_next: np.ndarray, dt_s: float) -> StateTruth:
        x_orbit_now = np.empty(6, dtype=float)
        x_orbit_now[:3] = state.position_eci_km
        x_orbit_now[3:] = state.velocity_eci_km_s
        x_mid = 0.5 * (x_orbit_now + np.array(x_orbit_next, dtype=float).reshape(6))
        return StateTruth(
            position_eci_km=np.array(x_mid[:3], dtype=float),
            velocity_eci_km_s=np.array(x_mid[3:], dtype=float),
            attitude_quat_bn=np.array(state.attitude_quat_bn, dtype=float),
            angular_rate_body_rad_s=np.array(state.angular_rate_body_rad_s, dtype=float),
            mass_kg=float(state.mass_kg),
            t_s=float(state.t_s + 0.5 * dt_s),
        )
