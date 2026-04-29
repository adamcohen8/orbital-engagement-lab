from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any

import numpy as np

from sim.actuators.orbital import attitude_coupled_thrust_eci, effective_max_accel_km_s2, thruster_disturbance_torque_body_nm
from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.runtime_support import (
    AgentRuntime,
    _attitude_state13_from_belief,
    _combine_commands,
    _command_to_dict,
    _orbital_elements_basic,
    _relative_orbit_state12,
    _rocket_altitude_km,
    _rocket_state_to_truth,
    _to_jsonable_value,
    _truth_from_state6,
    _truth_state6,
)
from sim.utils.quaternion import quaternion_to_dcm_bn

logger = logging.getLogger(__name__)


@dataclass
class _DecisionContext:
    agent: AgentRuntime
    internal_world_truth: dict[str, StateTruth]
    t_s: float
    dt_s: float
    env: dict[str, Any]
    orbit_controller: Any | None = None
    attitude_controller: Any | None = None
    orb_belief: StateBelief | None = None
    att_belief: StateBelief | None = None


class _DecisionContextBuilder:
    def __init__(
        self,
        *,
        base_environment: dict[str, Any],
        attitude_enabled: bool,
        orbit_command_period_s: float,
    ) -> None:
        self.base_environment = base_environment
        self.attitude_enabled = bool(attitude_enabled)
        self.orbit_command_period_s = float(orbit_command_period_s)

    def outer_context(
        self,
        *,
        agent: AgentRuntime,
        internal_world_truth: dict[str, StateTruth],
        t_s: float,
        dt_s: float,
    ) -> _DecisionContext:
        return _DecisionContext(
            agent=agent,
            internal_world_truth=internal_world_truth,
            t_s=float(t_s),
            dt_s=float(dt_s),
            env={**self.base_environment, "attitude_disabled": (not self.attitude_enabled)},
        )

    def satellite_context(
        self,
        *,
        agent: AgentRuntime,
        internal_world_truth: dict[str, StateTruth],
        t_s: float,
        dt_s: float,
        orb_belief: StateBelief | None,
        att_belief: StateBelief | None,
    ) -> _DecisionContext:
        return _DecisionContext(
            agent=agent,
            internal_world_truth=internal_world_truth,
            t_s=float(t_s),
            dt_s=float(dt_s),
            env={**self.base_environment, "orbit_command_period_s": self.orbit_command_period_s},
            orbit_controller=agent.orbit_controller,
            attitude_controller=(agent.attitude_controller if self.attitude_enabled else None),
            orb_belief=orb_belief,
            att_belief=(att_belief if self.attitude_enabled else None),
        )


@dataclass
class _SatelliteStepResult:
    truth: StateTruth
    average_thrust_eci_km_s2: np.ndarray
    average_torque_body_nm: np.ndarray
    delta_v_m_s: float
    max_accel_km_s2: float
    burned: bool


@dataclass
class _RocketStepResult:
    truth: StateTruth
    throttle: float
    thrust_eci_km_s2: np.ndarray
    torque_body_nm: np.ndarray
    delta_v_m_s: float
    max_accel_km_s2: float
    burned: bool
    stage_index: float | None = None
    q_dyn_pa: float | None = None
    mach: float | None = None


@dataclass
class _SatelliteCommandResult:
    command_applied: Command
    command_orbit: Command
    command_attitude: Command
    command_raw: Command
    use_integrated_command: bool
    orbit_runtime_ms: float
    attitude_runtime_ms: float


@dataclass
class _SatelliteBeliefContext:
    orbit_belief: StateBelief | None
    attitude_belief: StateBelief | None


class _RocketStepper:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def step(
        self,
        *,
        agent: AgentRuntime,
        world_truth_decision: dict[str, StateTruth],
        t_s: float,
        t_next: float,
    ) -> _RocketStepResult:
        e = self.engine
        mission_out = e._run_agent_decision(
            e.decision_contexts.outer_context(
                agent=agent,
                internal_world_truth=world_truth_decision,
                t_s=t_s,
                dt_s=e.dt,
            ),
            include_external_intent=False,
        )
        launch_auth = bool(mission_out.get("launch_authorized", True))
        agent.waiting_for_launch = not launch_auth
        if not launch_auth:
            agent.rocket_state.t_s = float(t_next)
            agent.truth = _rocket_state_to_truth(agent.rocket_state)
            if agent.belief is not None:
                agent.belief.state[:6] = _truth_state6(agent.truth, agent.belief.state[:6])
                agent.belief.last_update_t_s = t_next
            return _RocketStepResult(
                truth=agent.truth,
                throttle=0.0,
                thrust_eci_km_s2=e.zero3.copy(),
                torque_body_nm=e.zero3.copy(),
                delta_v_m_s=0.0,
                max_accel_km_s2=0.0,
                burned=False,
                stage_index=float(agent.rocket_state.active_stage_index),
                q_dyn_pa=0.0,
                mach=0.0,
            )

        cmd = agent.rocket_guidance.command(agent.rocket_state, agent.rocket_sim.sim_cfg, agent.rocket_sim.vehicle_cfg)
        if "guidance_throttle" in mission_out:
            cmd = type(cmd)(
                throttle=float(mission_out.get("guidance_throttle", cmd.throttle)),
                attitude_quat_bn_cmd=cmd.attitude_quat_bn_cmd,
                torque_body_nm_cmd=cmd.torque_body_nm_cmd,
            )
        throttle = float(np.clip(cmd.throttle, 0.0, 1.0))
        agent.rocket_state = agent.rocket_sim.step(agent.rocket_state, cmd, dt_s=e.dt)
        agent.truth = _rocket_state_to_truth(agent.rocket_state)
        if agent.belief is not None:
            agent.belief.state[:6] = _truth_state6(agent.truth, agent.belief.state[:6])
            agent.belief.last_update_t_s = t_next
        thrust_n = float(getattr(agent.rocket_state, "_last_step_thrust_n", 0.0))
        axis_eci = quaternion_to_dcm_bn(agent.rocket_state.attitude_quat_bn).T @ np.array(agent.rocket_sim.vehicle_cfg.thrust_axis_body, dtype=float)
        accel = (thrust_n / max(agent.rocket_state.mass_kg, 1e-9)) * axis_eci / 1e3
        accel_mag = float(np.linalg.norm(accel))
        return _RocketStepResult(
            truth=agent.truth,
            throttle=throttle,
            thrust_eci_km_s2=accel,
            torque_body_nm=e.zero3.copy(),
            delta_v_m_s=accel_mag * e.dt * 1e3,
            max_accel_km_s2=accel_mag,
            burned=bool(accel_mag > 1e-15),
            stage_index=float(agent.rocket_state.active_stage_index),
            q_dyn_pa=float(getattr(agent.rocket_state, "_last_step_q_dyn_pa", 0.0)),
            mach=float(getattr(agent.rocket_state, "_last_step_mach", 0.0)),
        )


class _SatelliteBeliefAdapter:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.orbit_state12_scratch = np.empty(12, dtype=float)
        self.attitude_state13_scratch = np.empty(13, dtype=float)
        self.deputy_state6_scratch = np.empty(6, dtype=float)
        self.chief_state6_scratch = np.empty(6, dtype=float)
        self.orbit_belief_scratch = StateBelief(
            state=self.orbit_state12_scratch,
            covariance=engine.eye12,
            last_update_t_s=0.0,
        )
        self.attitude_belief_scratch = StateBelief(
            state=self.attitude_state13_scratch,
            covariance=engine.eye6,
            last_update_t_s=0.0,
        )

    def prepare(
        self,
        *,
        aid: str,
        agent: AgentRuntime,
        truth: StateTruth,
        t_s: float,
    ) -> _SatelliteBeliefContext:
        e = self.engine
        if agent.belief is None:
            agent.belief = StateBelief(state=_truth_state6(truth), covariance=e.eye6.copy(), last_update_t_s=t_s)

        orb_belief = agent.belief
        if agent.orbit_controller is not None and orb_belief is not None:
            chief_truth = None
            if agent.knowledge_base is not None:
                target_belief = agent.knowledge_base.snapshot().get("target")
                if target_belief is not None and target_belief.state.size >= 6:
                    chief_truth = _truth_from_state6(
                        target_belief.state[:6],
                        t_s=target_belief.last_update_t_s,
                    )
            if chief_truth is not None and aid != "target" and hasattr(agent.orbit_controller, "ric_curv_state_slice"):
                self.orbit_belief_scratch.last_update_t_s = orb_belief.last_update_t_s
                self.orbit_belief_scratch.state = _relative_orbit_state12(
                    chief_truth=chief_truth,
                    deputy_truth=truth,
                    out=self.orbit_state12_scratch,
                    deputy_state6=self.deputy_state6_scratch,
                    chief_state6=self.chief_state6_scratch,
                )
                orb_belief = self.orbit_belief_scratch

        att_belief = agent.belief
        if e.attitude_enabled and att_belief is not None and att_belief.state.size < 13:
            self.attitude_belief_scratch.covariance = att_belief.covariance
            self.attitude_belief_scratch.last_update_t_s = att_belief.last_update_t_s
            self.attitude_belief_scratch.state = _attitude_state13_from_belief(
                belief=att_belief,
                truth=truth,
                out=self.attitude_state13_scratch,
            )
            att_belief = self.attitude_belief_scratch
        if not e.attitude_enabled:
            att_belief = None

        return _SatelliteBeliefContext(orbit_belief=orb_belief, attitude_belief=att_belief)


class _SatelliteCommandBuilder:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def build(
        self,
        *,
        aid: str,
        agent: AgentRuntime,
        truth: StateTruth,
        mission_out: dict[str, Any],
        orb_belief: StateBelief | None,
        att_belief: StateBelief | None,
        t_s: float,
        dt_s: float,
        sample_index: int,
    ) -> _SatelliteCommandResult:
        e = self.engine
        if e.attitude_enabled and "desired_attitude_quat_bn" in mission_out and agent.attitude_controller is not None:
            q_des = np.array(mission_out["desired_attitude_quat_bn"], dtype=float).reshape(-1)
            if q_des.size == 4 and hasattr(agent.attitude_controller, "set_target"):
                try:
                    agent.attitude_controller.set_target(q_des)
                except (TypeError, ValueError, AttributeError) as exc:
                    logger.warning("Failed to set desired_attitude_quat_bn on %s controller: %s", aid, exc)
        if e.attitude_enabled and "desired_attitude_quat_bn" in mission_out:
            q_des_log = np.array(mission_out["desired_attitude_quat_bn"], dtype=float).reshape(-1)
            if q_des_log.size == 4 and np.all(np.isfinite(q_des_log)):
                e.desired_attitude_hist[aid][sample_index + 1, :] = q_des_log
        if (
            e.attitude_enabled
            and "desired_ric_euler_rad" in mission_out
            and agent.attitude_controller is not None
            and hasattr(agent.attitude_controller, "set_desired_ric_state")
        ):
            euler = np.array(mission_out["desired_ric_euler_rad"], dtype=float).reshape(-1)
            if euler.size == 3:
                try:
                    agent.attitude_controller.set_desired_ric_state(float(euler[0]), float(euler[1]), float(euler[2]))
                except (TypeError, ValueError, AttributeError) as exc:
                    logger.warning("Failed to set desired_ric_euler_rad on %s controller: %s", aid, exc)

        use_integrated_cmd = bool(mission_out.get("mission_use_integrated_command", False))
        orbit_runtime_ms = 0.0
        attitude_runtime_ms = 0.0
        c_orb = Command.zero()
        if (not use_integrated_cmd) and agent.orbit_controller is not None and orb_belief is not None:
            orbit_t0 = perf_counter()
            c_orb = agent.orbit_controller.act(orb_belief, t_s, 2.0)
            orbit_runtime_ms = (perf_counter() - orbit_t0) * 1000.0
        c_att = Command.zero()
        if e.attitude_enabled and (not use_integrated_cmd) and agent.attitude_controller is not None and att_belief is not None:
            attitude_t0 = perf_counter()
            c_att = agent.attitude_controller.act(att_belief, t_s, 2.0)
            attitude_runtime_ms = (perf_counter() - attitude_t0) * 1000.0

        if use_integrated_cmd:
            cmd = Command.zero()
            if "thrust_eci_km_s2" in mission_out:
                cmd.thrust_eci_km_s2 = np.array(mission_out["thrust_eci_km_s2"], dtype=float).reshape(3)
            if "torque_body_nm" in mission_out:
                cmd.torque_body_nm = np.array(mission_out["torque_body_nm"], dtype=float).reshape(3)
            if "command_mode_flags" in mission_out and isinstance(mission_out["command_mode_flags"], dict):
                cmd.mode_flags.update(dict(mission_out["command_mode_flags"]))
            cmd.mode_flags["mode"] = "mission_integrated"
            if "mission_mode" in mission_out:
                cmd.mode_flags["mission_mode"] = mission_out["mission_mode"]
        else:
            cmd = _combine_commands(c_orb, c_att)
            if "thrust_eci_km_s2" in mission_out:
                cmd.thrust_eci_km_s2 = np.array(mission_out["thrust_eci_km_s2"], dtype=float).reshape(3)
            if "torque_body_nm" in mission_out:
                cmd.torque_body_nm = np.array(mission_out["torque_body_nm"], dtype=float).reshape(3)

        orbital_command_due = (
            e._last_orbital_command_eval_t_s[aid] is None
            or float(t_s) - float(e._last_orbital_command_eval_t_s[aid]) >= e.orbit_command_period_s - 1e-12
        )
        if orbital_command_due:
            e._last_orbital_command_eval_t_s[aid] = float(t_s)
            e._latched_orbital_thrust_cmd_by_object[aid] = np.array(cmd.thrust_eci_km_s2, dtype=float).reshape(3)
        latched_thrust_cmd = np.array(e._latched_orbital_thrust_cmd_by_object[aid], dtype=float).reshape(3)

        if not e.attitude_enabled:
            cmd.torque_body_nm = e.zero3
        cmd_step = Command(
            thrust_eci_km_s2=latched_thrust_cmd,
            torque_body_nm=(e.zero3.copy() if not e.attitude_enabled else np.array(cmd.torque_body_nm, dtype=float)),
            mode_flags=dict(cmd.mode_flags or {}),
        )
        cmd_step.mode_flags["orbital_command_updated"] = bool(orbital_command_due)
        if e._last_orbital_command_eval_t_s[aid] is not None:
            cmd_step.mode_flags["orbital_command_sample_t_s"] = float(e._last_orbital_command_eval_t_s[aid])
        cmd_step.mode_flags["current_attitude_quat_bn"] = np.array(truth.attitude_quat_bn, dtype=float)
        if agent.thruster_direction_body is not None:
            cmd_step.mode_flags["thruster_direction_body"] = np.array(agent.thruster_direction_body, dtype=float)
        if agent.thruster_position_body_m is not None:
            cmd_step.mode_flags["thruster_position_body_m"] = np.array(agent.thruster_position_body_m, dtype=float)
        if agent.thruster_direction_body is not None:
            cmd_step.mode_flags["commanded_thrust_eci_km_s2"] = np.array(cmd_step.thrust_eci_km_s2, dtype=float)
            cmd_step.thrust_eci_km_s2 = attitude_coupled_thrust_eci(
                cmd_step.thrust_eci_km_s2,
                attitude_quat_bn=np.array(truth.attitude_quat_bn, dtype=float),
                thruster_direction_body=np.array(agent.thruster_direction_body, dtype=float),
            )

        min_mass_kg = 0.0
        if agent.dry_mass_kg is not None and np.isfinite(float(agent.dry_mass_kg)):
            min_mass_kg = float(max(float(agent.dry_mass_kg), 0.0))
        if bool(truth.mass_kg <= (min_mass_kg + 1e-12)):
            cmd_step.thrust_eci_km_s2 = np.zeros(3, dtype=float)
            cmd_step.mode_flags["fuel_depleted"] = True
        if agent.orbital_max_thrust_n is not None:
            eff_max_accel_km_s2 = effective_max_accel_km_s2(
                current_mass_kg=float(max(truth.mass_kg, 0.0)),
                max_accel_km_s2=0.0,
                max_thrust_n=agent.orbital_max_thrust_n,
            )
            accel_vec = np.array(cmd_step.thrust_eci_km_s2, dtype=float)
            accel_norm = float(np.linalg.norm(accel_vec))
            if accel_norm > eff_max_accel_km_s2 > 0.0:
                cmd_step.thrust_eci_km_s2 = accel_vec * (eff_max_accel_km_s2 / accel_norm)
                cmd_step.mode_flags["thrust_limited_scale"] = float(eff_max_accel_km_s2 / accel_norm)
            elif eff_max_accel_km_s2 == 0.0:
                cmd_step.thrust_eci_km_s2 = np.zeros(3, dtype=float)
            cmd_step.mode_flags["effective_max_accel_km_s2"] = float(eff_max_accel_km_s2)
            cmd_step.mode_flags["max_thrust_n"] = float(agent.orbital_max_thrust_n)
        cmd_step.mode_flags["min_mass_kg"] = float(min_mass_kg)

        isp_s = agent.orbital_isp_s
        if isp_s is not None and float(isp_s) > 0.0 and "delta_mass_kg" not in cmd_step.mode_flags:
            g0_m_s2 = 9.80665
            a_mag_m_s2 = float(np.linalg.norm(cmd_step.thrust_eci_km_s2) * 1e3)
            thrust_n = float(max(truth.mass_kg, 0.0) * a_mag_m_s2)
            mdot_kg_s = 0.0 if thrust_n <= 0.0 else float(thrust_n / (float(isp_s) * g0_m_s2))
            delta_mass_kg = float(max(mdot_kg_s, 0.0) * dt_s)
            available_propellant_kg = float(max(truth.mass_kg - min_mass_kg, 0.0))
            applied_delta_mass_kg = float(min(delta_mass_kg, available_propellant_kg))
            if delta_mass_kg > 1e-15 and applied_delta_mass_kg < (delta_mass_kg - 1e-15):
                propellant_scale = float(np.clip(applied_delta_mass_kg / delta_mass_kg, 0.0, 1.0))
                cmd_step.thrust_eci_km_s2 = np.array(cmd_step.thrust_eci_km_s2, dtype=float) * propellant_scale
                cmd_step.mode_flags["propellant_limited_scale"] = propellant_scale
            cmd_step.mode_flags["delta_mass_kg"] = applied_delta_mass_kg

        thruster_torque_body_nm = np.zeros(3, dtype=float)
        if e.attitude_enabled and agent.thruster_direction_body is not None and agent.thruster_position_body_m is not None:
            thruster_torque_body_nm = thruster_disturbance_torque_body_nm(
                cmd_step.thrust_eci_km_s2,
                current_mass_kg=float(max(truth.mass_kg, 0.0)),
                thruster_direction_body=np.array(agent.thruster_direction_body, dtype=float),
                thruster_position_body_m=np.array(agent.thruster_position_body_m, dtype=float),
            )
            cmd_step.torque_body_nm = np.array(cmd_step.torque_body_nm, dtype=float) + thruster_torque_body_nm
        cmd_step.mode_flags["thruster_torque_body_nm"] = thruster_torque_body_nm.tolist()

        return _SatelliteCommandResult(
            command_applied=cmd_step,
            command_orbit=c_orb,
            command_attitude=c_att,
            command_raw=cmd,
            use_integrated_command=use_integrated_cmd,
            orbit_runtime_ms=orbit_runtime_ms,
            attitude_runtime_ms=attitude_runtime_ms,
        )


class _SatelliteEstimatorUpdater:
    def update(
        self,
        *,
        agent: AgentRuntime,
        truth: StateTruth,
        world_truth: dict[str, StateTruth],
        t_s: float,
    ) -> None:
        meas = agent.sensor.measure(truth=truth, env={"world_truth": world_truth}, t_s=t_s) if agent.sensor is not None else None
        if agent.estimator is not None and agent.belief is not None:
            agent.belief = agent.estimator.update(agent.belief, meas, t_s)


class _ControllerDebugRecorder:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def record(
        self,
        *,
        aid: str,
        agent: AgentRuntime,
        orbit_belief: StateBelief | None,
        attitude_belief: StateBelief | None,
        command_result: _SatelliteCommandResult,
        t_s: float,
        interval_end_t_s: float,
        dt_s: float,
    ) -> None:
        cmd_step = command_result.command_applied
        self.engine.controller_debug_hist[aid].append(
            {
                "t_s": float(t_s),
                "interval_end_t_s": float(interval_end_t_s),
                "dt_s": float(dt_s),
                "belief": (np.array(agent.belief.state, dtype=float).tolist() if agent.belief is not None else None),
                "orbit_belief": (np.array(orbit_belief.state, dtype=float).tolist() if orbit_belief is not None else None),
                "attitude_belief": (np.array(attitude_belief.state, dtype=float).tolist() if attitude_belief is not None else None),
                "orbit_controller_runtime_ms": float(command_result.orbit_runtime_ms),
                "attitude_controller_runtime_ms": float(command_result.attitude_runtime_ms),
                "controller_runtime_ms": float(command_result.orbit_runtime_ms + command_result.attitude_runtime_ms),
                "command_orbit": _command_to_dict(command_result.command_orbit),
                "command_attitude": _command_to_dict(command_result.command_attitude),
                "command_raw": _command_to_dict(command_result.command_raw),
                "command_applied": _command_to_dict(cmd_step),
                "use_integrated_command": bool(command_result.use_integrated_command),
                "mode_flags": _to_jsonable_value(dict(cmd_step.mode_flags or {})),
            }
        )


class _SatelliteStepper:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.belief_adapter = _SatelliteBeliefAdapter(engine)
        self.command_builder = _SatelliteCommandBuilder(engine)
        self.estimator_updater = _SatelliteEstimatorUpdater()
        self.debug_recorder = _ControllerDebugRecorder(engine)

    def step(
        self,
        *,
        aid: str,
        agent: AgentRuntime,
        initial_truth: StateTruth,
        world_truth_decision: dict[str, StateTruth],
        t_s: float,
        t_next: float,
        sample_index: int,
    ) -> _SatelliteStepResult:
        e = self.engine
        t_inner = float(t_s)
        tr_inner = initial_truth
        accel_time_integral = e.zero3.copy()
        torque_time_integral = e.zero3.copy()
        step_delta_v_m_s = 0.0
        step_max_accel_km_s2 = 0.0
        burned_this_step = False
        world_truth_inner = dict(world_truth_decision)
        env_inner = {
            **e.base_environment,
            "world_truth": world_truth_inner,
            "attitude_disabled": (not e.attitude_enabled),
            "orbit_command_period_s": float(e.orbit_command_period_s),
        }

        while t_inner < t_next - 1e-12:
            h = float(min(e.sim_substep_s, t_next - t_inner))
            t_decision = float(t_inner)
            t_eval = t_inner + h
            world_truth_inner[aid] = tr_inner
            belief_ctx = self.belief_adapter.prepare(
                aid=aid,
                agent=agent,
                truth=tr_inner,
                t_s=t_decision,
            )
            orb_belief = belief_ctx.orbit_belief
            att_belief = belief_ctx.attitude_belief
            mission_out = e._run_agent_decision(
                e.decision_contexts.satellite_context(
                    agent=agent,
                    internal_world_truth=world_truth_inner,
                    t_s=t_decision,
                    dt_s=h,
                    orb_belief=orb_belief,
                    att_belief=att_belief,
                ),
            )
            command_result = self.command_builder.build(
                aid=aid,
                agent=agent,
                truth=tr_inner,
                mission_out=mission_out,
                orb_belief=orb_belief,
                att_belief=att_belief,
                t_s=t_decision,
                dt_s=h,
                sample_index=sample_index,
            )
            cmd_step = command_result.command_applied
            self.debug_recorder.record(
                aid=aid,
                agent=agent,
                orbit_belief=orb_belief,
                attitude_belief=att_belief,
                command_result=command_result,
                t_s=t_decision,
                interval_end_t_s=t_eval,
                dt_s=h,
            )
            tr_inner = agent.dynamics.step(state=tr_inner, command=cmd_step, env=env_inner, dt_s=h)
            world_truth_inner[aid] = tr_inner
            self.estimator_updater.update(
                agent=agent,
                truth=tr_inner,
                world_truth=world_truth_inner,
                t_s=t_eval,
            )
            applied_thrust = np.array(cmd_step.thrust_eci_km_s2, dtype=float)
            applied_torque = np.array(cmd_step.torque_body_nm, dtype=float)
            accel_time_integral += applied_thrust * h
            torque_time_integral += applied_torque * h
            accel_mag = float(np.linalg.norm(applied_thrust))
            step_delta_v_m_s += accel_mag * h * 1e3
            step_max_accel_km_s2 = max(step_max_accel_km_s2, accel_mag)
            burned_this_step = burned_this_step or (accel_mag > 1e-15)
            t_inner = t_eval

        return _SatelliteStepResult(
            truth=tr_inner,
            average_thrust_eci_km_s2=accel_time_integral / e.dt,
            average_torque_body_nm=e.zero3 if not e.attitude_enabled else (torque_time_integral / e.dt),
            delta_v_m_s=step_delta_v_m_s,
            max_accel_km_s2=step_max_accel_km_s2,
            burned=burned_this_step,
        )


class _KnowledgeSynchronizer:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def initialize(self) -> None:
        e = self.engine
        initial_world_truth = {
            aid: (agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state))
            for aid, agent in e.agents.items()
            if agent.active
        }
        for aid, agent in e.agents.items():
            if not agent.active or agent.knowledge_base is None:
                continue
            observer_truth = initial_world_truth.get(aid)
            if observer_truth is None:
                continue
            agent.knowledge_base.update(observer_truth=observer_truth, world_truth=initial_world_truth, t_s=0.0)
            self._record_snapshot(aid=aid, sample_index=0)

    def update_after_step(
        self,
        *,
        world_truth: dict[str, StateTruth],
        sample_index: int,
        t_s: float,
    ) -> None:
        e = self.engine
        for aid, agent in e.agents.items():
            if not agent.active or agent.knowledge_base is None:
                continue
            observer_truth = world_truth.get(aid)
            if observer_truth is None:
                continue
            agent.knowledge_base.update(observer_truth=observer_truth, world_truth=world_truth, t_s=t_s)
            self._record_snapshot(aid=aid, sample_index=sample_index)

    def _record_snapshot(self, *, aid: str, sample_index: int) -> None:
        e = self.engine
        agent = e.agents[aid]
        if agent.knowledge_base is None:
            return
        snap = agent.knowledge_base.snapshot()
        for tid, hist in e.knowledge_hist.get(aid, {}).items():
            belief = snap.get(tid)
            if belief is not None:
                hist[sample_index, :] = belief.state[:6]
            elif sample_index > 0:
                hist[sample_index, :] = hist[sample_index - 1, :]


class _TerminationMonitor:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def check_earth_impact(self, *, t_s: float) -> bool:
        e = self.engine
        if not bool(e.cfg.simulator.termination.get("earth_impact_enabled", True)):
            return False
        re = float(e.cfg.simulator.termination.get("earth_radius_km", EARTH_RADIUS_KM))
        for aid, agent in e.agents.items():
            if not agent.active:
                continue
            if agent.kind == "rocket" and agent.waiting_for_launch:
                continue
            truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            impact = float(np.linalg.norm(truth.position_eci_km)) <= re
            if agent.kind == "rocket" and agent.rocket_sim is not None:
                impact = bool(_rocket_altitude_km(truth.position_eci_km, truth.t_s, agent.rocket_sim.sim_cfg) <= 0.0)
            if impact:
                e.terminated_early = True
                e.termination_reason = "earth_impact"
                e.termination_time_s = float(t_s)
                e.termination_object_id = aid
                return True
        return False

    def update_rocket_insertion(self, *, t_s: float) -> None:
        e = self.engine
        rocket = e.rocket
        if (
            rocket is None
            or not rocket.active
            or rocket.waiting_for_launch
            or rocket.rocket_state is None
            or rocket.rocket_sim is None
        ):
            return
        rs = rocket.rocket_state
        sim_cfg = rocket.rocket_sim.sim_cfg
        alt_km = _rocket_altitude_km(rs.position_eci_km, rs.t_s, sim_cfg)
        near_alt = abs(float(alt_km) - float(sim_cfg.target_altitude_km)) <= float(sim_cfg.target_altitude_tolerance_km)
        _, ecc_now = _orbital_elements_basic(np.array(rs.position_eci_km, dtype=float), np.array(rs.velocity_eci_km_s, dtype=float))
        low_e = float(ecc_now) <= float(sim_cfg.target_eccentricity_max)
        stages_done = int(rs.active_stage_index) >= len(rocket.rocket_sim.vehicle_cfg.stack.stages)
        if near_alt and low_e and stages_done:
            e.rocket_insertion_hold_s += float(e.dt)
            if (not e.rocket_inserted) and e.rocket_insertion_hold_s >= float(sim_cfg.insertion_hold_time_s):
                e.rocket_inserted = True
                e.rocket_insertion_time_s = float(t_s)
        else:
            e.rocket_insertion_hold_s = 0.0
        if e.rocket_inserted and str(e.cfg.simulator.scenario_type).strip().lower() == "rocket_ascent":
            e.terminated_early = True
            e.termination_reason = "rocket_orbit_insertion"
            e.termination_time_s = float(e.rocket_insertion_time_s if e.rocket_insertion_time_s is not None else t_s)
            e.termination_object_id = "rocket"
