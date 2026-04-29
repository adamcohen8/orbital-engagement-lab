from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import logging
import os
from typing import Any, Callable

import numpy as np

from sim.config import SimulationScenarioConfig, scenario_config_from_dict
from sim.core.models import Command, StateTruth
from sim.dynamics.attitude.rigid_body import get_attitude_guardrail_stats, reset_attitude_guardrail_stats
from sim.dynamics.orbit.spherical_harmonics import configure_spherical_harmonics_env
from sim.ground_stations import evaluate_ground_station_access
from sim.master_outputs import animate_outputs as _animate_outputs_impl
from sim.master_outputs import plot_outputs as _plot_outputs_impl
from sim.reporting.output_index import write_output_index
from sim.runtime_support import (
    AgentRuntime,
    _apply_chaser_relative_init_from_target,
    _build_knowledge_base,
    _create_rocket_runtime,
    _create_satellite_runtime,
    _decision_truth_from_belief,
    _deploy_from_rocket,
    _resolve_rocket_stack,
    _resolve_satellite_isp_s,
    _rocket_state_to_truth,
    _run_mission_execution,
    _run_mission_modules,
    _run_mission_strategy,
)
from sim.single_run_support import (
    _DecisionContext,
    _DecisionContextBuilder,
    _KnowledgeSynchronizer,
    _RocketStepper,
    _SatelliteStepper,
    _TerminationMonitor,
)
from sim.utils.io import write_json

logger = logging.getLogger(__name__)


def _state_truth_to_array(truth: StateTruth) -> np.ndarray:
    return np.hstack(
        (
            truth.position_eci_km,
            truth.velocity_eci_km_s,
            truth.attitude_quat_bn,
            truth.angular_rate_body_rad_s,
            np.array([truth.mass_kg]),
        )
    )


def _plot_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    belief_hist: dict[str, np.ndarray] | None,
    thrust_hist: dict[str, np.ndarray],
    desired_attitude_hist: dict[str, np.ndarray] | None,
    knowledge_hist: dict[str, dict[str, np.ndarray]],
    rocket_metrics: dict[str, np.ndarray] | None,
    bridge_hist: dict[str, list[dict[str, Any]]] | None,
    outdir: Path,
) -> dict[str, str]:
    return _plot_outputs_impl(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        target_reference_orbit_truth=target_reference_orbit_truth,
        thrust_hist=thrust_hist,
        belief_hist=belief_hist,
        desired_attitude_hist=desired_attitude_hist,
        knowledge_hist=knowledge_hist,
        rocket_metrics=rocket_metrics,
        bridge_hist=bridge_hist,
        outdir=outdir,
        resolve_rocket_stack=_resolve_rocket_stack,
        resolve_satellite_isp_s=_resolve_satellite_isp_s,
    )


def _animate_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    thrust_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    outdir: Path,
) -> dict[str, str]:
    return _animate_outputs_impl(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        thrust_hist=thrust_hist,
        target_reference_orbit_truth=target_reference_orbit_truth,
        outdir=outdir,
        resolve_satellite_isp_s=_resolve_satellite_isp_s,
    )


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def _format_single_run_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    scenario_description = str(summary.get("scenario_description", "") or "").strip()
    lines.append("")
    lines.append("=" * 72)
    lines.append("MASTER SIMULATION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Scenario   : {summary.get('scenario_name', 'unknown')}")
    if scenario_description:
        lines.append(f"Desc       : {scenario_description}")
    lines.append(f"Objects    : {', '.join(summary.get('objects', []))}")
    lines.append(f"Samples    : {summary.get('samples', 0)}")
    lines.append(
        f"Timing     : dt={_fmt_float(float(summary.get('dt_s', 0.0)), 3)} s, "
        f"duration={_fmt_float(float(summary.get('duration_s', 0.0)), 1)} s"
    )
    lines.append("-" * 72)
    if bool(summary.get("terminated_early", False)):
        lines.append(
            "Termination: EARLY "
            f"(reason={summary.get('termination_reason')}, "
            f"t={summary.get('termination_time_s')}, "
            f"object={summary.get('termination_object_id')})"
        )
    else:
        lines.append("Termination: nominal (full duration reached)")
    if "rocket_insertion_achieved" in summary:
        ins_ok = bool(summary.get("rocket_insertion_achieved", False))
        ins_t = summary.get("rocket_insertion_time_s")
        lines.append(f"Insertion  : achieved at t={ins_t}" if ins_ok else "Insertion  : not achieved")
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    if thrust_stats:
        lines.append("-" * 72)
        lines.append("Thrust Stats")
        lines.append(f"{'Object':<14}{'Burn Samples':>14}{'Max Accel (km/s^2)':>24}{'Total dV (m/s)':>18}")
        for oid in sorted(thrust_stats.keys()):
            stats = dict(thrust_stats.get(oid, {}) or {})
            lines.append(
                f"{oid:<14}"
                f"{int(stats.get('burn_samples', 0)):>14d}"
                f"{float(stats.get('max_accel_km_s2', 0.0)):>24.3e}"
                f"{float(stats.get('total_dv_m_s', 0.0)):>18.3f}"
            )
    plot_outputs = dict(summary.get("plot_outputs", {}) or {})
    anim_outputs = dict(summary.get("animation_outputs", {}) or {})
    guardrails = dict(summary.get("attitude_guardrail_stats", {}) or {})
    lines.append("-" * 72)
    lines.append(f"Artifacts  : plots={len(plot_outputs)}  animations={len(anim_outputs)}")
    lines.append(f"Guardrails : attitude_events={int(sum(int(v) for v in guardrails.values())) if guardrails else 0}")
    lines.append("=" * 72)
    return "\n".join(lines)


class _SingleRunEngine:
    def __init__(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.active_step_callback = step_callback
        reset_attitude_guardrail_stats()

        self.dt = float(cfg.simulator.dt_s)
        self.n = int(np.floor(float(cfg.simulator.duration_s) / self.dt)) + 1
        self.t_s = np.arange(self.n, dtype=float) * self.dt
        self.outdir = Path(cfg.outputs.output_dir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        seed = int(cfg.metadata.get("seed", 123))
        rng = np.random.default_rng(seed)
        dynamics_cfg = dict(cfg.simulator.dynamics or {})
        orbit_cfg = dict(dynamics_cfg.get("orbit", {}) or {})
        att_cfg = dict(dynamics_cfg.get("attitude", {}) or {})
        self.base_environment = configure_spherical_harmonics_env(dict(cfg.simulator.environment or {}), orbit_cfg)
        if cfg.simulator.initial_jd_utc is not None and "jd_utc_start" not in self.base_environment:
            self.base_environment["jd_utc_start"] = float(cfg.simulator.initial_jd_utc)
        self.attitude_enabled = bool(att_cfg.get("enabled", True))
        orbit_substep_s = float(max(float(orbit_cfg.get("orbit_substep_s", self.dt) or self.dt), 1e-9))
        attitude_substep_s = float(max(float(att_cfg.get("attitude_substep_s", self.dt) or self.dt), 1e-9))
        self.orbit_command_period_s = orbit_substep_s
        self.sim_substep_s = float(min(orbit_substep_s, attitude_substep_s)) if self.attitude_enabled else orbit_substep_s
        self.eye6 = np.eye(6) * 1e-4
        self.eye12 = np.eye(12) * 1e-4
        self.zero3 = np.zeros(3, dtype=float)
        self.decision_contexts = _DecisionContextBuilder(
            base_environment=self.base_environment,
            attitude_enabled=self.attitude_enabled,
            orbit_command_period_s=self.orbit_command_period_s,
        )
        self.rocket_stepper = _RocketStepper(self)
        self.satellite_stepper = _SatelliteStepper(self)
        self.termination_monitor = _TerminationMonitor(self)

        self.rocket = _create_rocket_runtime(cfg) if cfg.rocket.enabled else None
        self.chaser = (
            _create_satellite_runtime("chaser", cfg.chaser, cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
            if cfg.chaser.enabled
            else None
        )
        self.target = (
            _create_satellite_runtime("target", cfg.target, cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
            if cfg.target.enabled
            else None
        )
        if self.chaser is not None and self.chaser.deploy_source == "rocket_deployment":
            self.chaser.active = False

        self.agents: dict[str, AgentRuntime] = {}
        if self.rocket is not None:
            self.agents["rocket"] = self.rocket
        if self.target is not None:
            self.agents["target"] = self.target
        if self.chaser is not None:
            self.agents["chaser"] = self.chaser

        if self.chaser is not None and self.target is not None and self.chaser.deploy_source != "rocket_deployment":
            _apply_chaser_relative_init_from_target(
                chaser=self.chaser,
                target=self.target,
                initial_state=dict(cfg.chaser.initial_state or {}),
            )

        target_reference_cfg = dict(cfg.target.reference_orbit or {})
        self.target_reference_truth = None
        self.target_reference_dynamics = None
        self.target_reference_orbit_hist = None
        if bool(target_reference_cfg.get("enabled", False)) and self.target is not None and self.target.truth is not None:
            self.target_reference_truth = self.target.truth.copy()
            self.target_reference_dynamics = replace(
                self.target.dynamics,
                disturbance_model=None,
                propagate_attitude=False,
                use_rectangular_prism_for_aero_srp=False,
                rectangular_prism_dims_m=None,
            )
            self.target_reference_orbit_hist = np.full((self.n, 6), np.nan)
            self.target_reference_orbit_hist[0, 0:3] = self.target_reference_truth.position_eci_km
            self.target_reference_orbit_hist[0, 3:6] = self.target_reference_truth.velocity_eci_km_s

        for aid, agent in self.agents.items():
            cfg_src = cfg.rocket if aid == "rocket" else (cfg.chaser if aid == "chaser" else cfg.target)
            agent.knowledge_base = _build_knowledge_base(
                observer_id=aid,
                agent_cfg=cfg_src,
                dt_s=self.dt,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )

        self.truth_hist = {aid: np.full((self.n, 14), np.nan) for aid in self.agents.keys()}
        self.belief_hist = {
            aid: np.full((self.n, int(agent.belief.state.size) if agent.belief is not None else 0), np.nan)
            for aid, agent in self.agents.items()
        }
        self.thrust_hist = {aid: np.full((self.n, 3), np.nan) for aid in self.agents.keys()}
        self.torque_hist = {aid: np.full((self.n, 3), np.nan) for aid in self.agents.keys()}
        self.desired_attitude_hist = {aid: np.full((self.n, 4), np.nan) for aid in self.agents.keys()}
        self.controller_debug_hist: dict[str, list[dict[str, Any]]] = {aid: [] for aid in self.agents.keys()}
        self._last_orbital_command_eval_t_s: dict[str, float | None] = {aid: None for aid in self.agents.keys()}
        self._latched_orbital_thrust_cmd_by_object: dict[str, np.ndarray] = {
            aid: self.zero3.copy() for aid in self.agents.keys()
        }
        self.throttle_hist = {"rocket": np.full(self.n, np.nan)} if self.rocket is not None else {}
        self.rocket_stage_hist = np.full(self.n, np.nan) if self.rocket is not None else None
        self.rocket_q_dyn_hist = np.full(self.n, np.nan) if self.rocket is not None else None
        self.rocket_mach_hist = np.full(self.n, np.nan) if self.rocket is not None else None
        self.knowledge_hist: dict[str, dict[str, np.ndarray]] = {}
        self.bridge_hist: dict[str, list[dict[str, Any]]] = {aid: [] for aid in self.agents.keys()}
        for aid, agent in self.agents.items():
            if agent.knowledge_base is not None:
                self.knowledge_hist[aid] = {}
                for tid in agent.knowledge_base.target_ids():
                    self.knowledge_hist[aid][tid] = np.full((self.n, 6), np.nan)
        self.knowledge_sync = _KnowledgeSynchronizer(self)
        self.knowledge_sync.initialize()

        self.terminated_early = False
        self.termination_reason: str | None = None
        self.termination_time_s: float | None = None
        self.termination_object_id: str | None = None
        self.rocket_inserted = False
        self.rocket_insertion_time_s: float | None = None
        self.rocket_insertion_hold_s = 0.0
        self.total_dv_m_s_by_object = {aid: 0.0 for aid in self.agents.keys()}
        self.burn_samples_by_object = {aid: 0 for aid in self.agents.keys()}
        self.max_accel_km_s2_by_object = {aid: 0.0 for aid in self.agents.keys()}
        self.current_index = 0
        self.external_intent_providers: dict[str, Callable[..., dict[str, Any] | None]] = {}

        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            self.truth_hist[aid][0, :] = _state_truth_to_array(truth)
            if agent.belief is not None:
                self._ensure_belief_hist_width(aid, agent.belief.state.size)
                self.belief_hist[aid][0, : agent.belief.state.size] = agent.belief.state
            if aid == "rocket" and agent.rocket_state is not None and self.rocket_stage_hist is not None:
                self.rocket_stage_hist[0] = float(agent.rocket_state.active_stage_index)
                if self.rocket_q_dyn_hist is not None:
                    self.rocket_q_dyn_hist[0] = float(getattr(agent.rocket_state, "_last_step_q_dyn_pa", 0.0))
                if self.rocket_mach_hist is not None:
                    self.rocket_mach_hist[0] = float(getattr(agent.rocket_state, "_last_step_mach", 0.0))

        self._emit_step_callback(0)

    @property
    def total_steps(self) -> int:
        return max(self.n - 1, 0)

    @property
    def done(self) -> bool:
        return bool(self.terminated_early or self.current_index >= max(self.n - 1, 0))

    def _emit_step_callback(self, step: int) -> None:
        if self.active_step_callback is None:
            return
        try:
            self.active_step_callback(int(step), self.total_steps)
        except (TypeError, ValueError) as exc:
            logger.warning("Disabling step callback after runtime error: %s", exc)
            self.active_step_callback = None

    def _ensure_belief_hist_width(self, aid: str, width: int) -> None:
        hist = self.belief_hist[aid]
        if hist.shape[1] >= int(width):
            return
        expanded = np.full((self.n, int(width)), np.nan)
        if hist.shape[1] > 0:
            expanded[:, : hist.shape[1]] = hist
        self.belief_hist[aid] = expanded

    def snapshot(self, step_index: int | None = None) -> dict[str, Any]:
        idx = self.current_index if step_index is None else int(step_index)
        if idx < 0 or idx >= self.n:
            raise IndexError(f"step_index {idx} is out of range for {self.n} samples.")
        truth = {oid: np.array(hist[idx], dtype=float) for oid, hist in self.truth_hist.items()}
        if self.target_reference_orbit_hist is not None:
            ref_state = np.array(self.target_reference_orbit_hist[idx], dtype=float).reshape(-1)
            if ref_state.size >= 6 and np.all(np.isfinite(ref_state[:6])):
                target_mass_kg = 0.0
                target_truth = truth.get("target")
                if target_truth is not None and np.array(target_truth).reshape(-1).size >= 14:
                    target_mass_kg = float(np.array(target_truth, dtype=float).reshape(-1)[13])
                truth["target_reference"] = np.hstack(
                    (
                        ref_state[:6],
                        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, target_mass_kg]),
                    )
                )
        return {
            "step_index": idx,
            "time_s": float(self.t_s[idx]),
            "truth": truth,
            "belief": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.belief_hist.items()},
            "applied_thrust": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.thrust_hist.items()},
            "applied_torque": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.torque_hist.items()},
        }

    def set_external_intent_provider(
        self,
        object_id: str,
        provider: Callable[..., dict[str, Any] | None] | None,
    ) -> None:
        oid = str(object_id)
        if provider is None:
            self.external_intent_providers.pop(oid, None)
            return
        self.external_intent_providers[oid] = provider

    def _external_intent(
        self,
        *,
        ctx: _DecisionContext,
    ) -> dict[str, Any]:
        agent = ctx.agent
        decision_truth = _decision_truth_from_belief(agent)
        out: dict[str, Any] = {}
        provider = self.external_intent_providers.get(str(agent.object_id))
        if provider is not None:
            out.update(self._call_external_intent_provider(provider, ctx=ctx, decision_truth=decision_truth))
        bridge = getattr(agent, "bridge", None)
        bridge_provider = getattr(bridge, "external_intent", None) if bridge is not None else None
        if callable(bridge_provider):
            out.update(self._call_external_intent_provider(bridge_provider, ctx=ctx, decision_truth=decision_truth))
        return out

    def _call_external_intent_provider(
        self,
        provider: Callable[..., dict[str, Any] | None],
        *,
        ctx: _DecisionContext,
        decision_truth: StateTruth | None,
    ) -> dict[str, Any]:
        agent = ctx.agent
        try:
            ret = provider(
                object_id=agent.object_id,
                truth=decision_truth,
                belief=agent.belief,
                own_knowledge=(agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}),
                world_truth={},
                env=ctx.env,
                t_s=ctx.t_s,
                dt_s=ctx.dt_s,
                orbit_controller=ctx.orbit_controller,
                attitude_controller=ctx.attitude_controller,
                orb_belief=ctx.orb_belief,
                att_belief=ctx.att_belief,
                dry_mass_kg=agent.dry_mass_kg,
                fuel_capacity_kg=agent.fuel_capacity_kg,
                thruster_direction_body=agent.thruster_direction_body,
            )
        except TypeError:
            ret = provider(truth=decision_truth, t_s=ctx.t_s, dt_s=ctx.dt_s)
        return ret if isinstance(ret, dict) else {}

    def _run_agent_decision(self, ctx: _DecisionContext, *, include_external_intent: bool = True) -> dict[str, Any]:
        agent = ctx.agent
        mission_out = _run_mission_modules(
            agent=agent,
            world_truth=ctx.internal_world_truth,
            t_s=ctx.t_s,
            dt_s=ctx.dt_s,
            env=ctx.env,
            orbit_controller=ctx.orbit_controller,
            attitude_controller=ctx.attitude_controller,
            orb_belief=ctx.orb_belief,
            att_belief=ctx.att_belief,
        )
        mission_out.update(
            _run_mission_strategy(
                agent=agent,
                world_truth=ctx.internal_world_truth,
                t_s=ctx.t_s,
                dt_s=ctx.dt_s,
                env=ctx.env,
                orbit_controller=ctx.orbit_controller,
                attitude_controller=ctx.attitude_controller,
                orb_belief=ctx.orb_belief,
                att_belief=ctx.att_belief,
            )
        )
        if include_external_intent:
            mission_out.update(self._external_intent(ctx=ctx))
        mission_out.update(
            _run_mission_execution(
                agent=agent,
                intent=mission_out,
                world_truth=ctx.internal_world_truth,
                t_s=ctx.t_s,
                dt_s=ctx.dt_s,
                env=ctx.env,
                orbit_controller=ctx.orbit_controller,
                attitude_controller=ctx.attitude_controller,
                orb_belief=ctx.orb_belief,
                att_belief=ctx.att_belief,
            )
        )
        return mission_out

    def step(self) -> dict[str, Any]:
        if self.done:
            return self.snapshot()

        k = int(self.current_index)
        t = float(self.t_s[k])
        t_next = float(self.t_s[k + 1])

        if self.chaser is not None and self.rocket is not None and (not self.chaser.active):
            if t_next >= float(self.chaser.deploy_time_s or 0.0):
                _deploy_from_rocket(self.chaser, self.rocket, t_next)

        world_truth_start = {
            aid: (agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state))
            for aid, agent in self.agents.items()
            if agent.active
        }
        world_truth_live = dict(world_truth_start)

        if self.target_reference_truth is not None and self.target_reference_dynamics is not None:
            env_ref = {**self.base_environment, "world_truth": world_truth_live, "attitude_disabled": True}
            self.target_reference_truth = self.target_reference_dynamics.step(
                state=self.target_reference_truth,
                command=Command.zero(),
                env=env_ref,
                dt_s=self.dt,
            )
            assert self.target_reference_orbit_hist is not None
            self.target_reference_orbit_hist[k + 1, 0:3] = self.target_reference_truth.position_eci_km
            self.target_reference_orbit_hist[k + 1, 3:6] = self.target_reference_truth.velocity_eci_km_s

        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            tr_now = world_truth_start[aid]
            world_truth_decision = dict(world_truth_start)

            if agent.kind == "rocket":
                rocket_result = self.rocket_stepper.step(
                    agent=agent,
                    world_truth_decision=world_truth_decision,
                    t_s=t,
                    t_next=t_next,
                )
                agent.truth = rocket_result.truth
                self.throttle_hist["rocket"][k] = rocket_result.throttle
                self.thrust_hist[aid][k + 1, :] = rocket_result.thrust_eci_km_s2
                self.torque_hist[aid][k + 1, :] = rocket_result.torque_body_nm
                self.total_dv_m_s_by_object[aid] += rocket_result.delta_v_m_s
                self.max_accel_km_s2_by_object[aid] = max(self.max_accel_km_s2_by_object[aid], rocket_result.max_accel_km_s2)
                if rocket_result.burned:
                    self.burn_samples_by_object[aid] += 1
                if self.rocket_stage_hist is not None and rocket_result.stage_index is not None:
                    self.rocket_stage_hist[k + 1] = rocket_result.stage_index
                if self.rocket_q_dyn_hist is not None and rocket_result.q_dyn_pa is not None:
                    self.rocket_q_dyn_hist[k + 1] = rocket_result.q_dyn_pa
                if self.rocket_mach_hist is not None and rocket_result.mach is not None:
                    self.rocket_mach_hist[k + 1] = rocket_result.mach
            else:
                sat_result = self.satellite_stepper.step(
                    aid=aid,
                    agent=agent,
                    initial_truth=tr_now,
                    world_truth_decision=world_truth_decision,
                    t_s=t,
                    t_next=t_next,
                    sample_index=k,
                )
                agent.truth = sat_result.truth
                self.thrust_hist[aid][k + 1, :] = sat_result.average_thrust_eci_km_s2
                self.torque_hist[aid][k + 1, :] = sat_result.average_torque_body_nm
                self.total_dv_m_s_by_object[aid] += sat_result.delta_v_m_s
                self.max_accel_km_s2_by_object[aid] = max(self.max_accel_km_s2_by_object[aid], sat_result.max_accel_km_s2)
                if sat_result.burned:
                    self.burn_samples_by_object[aid] += 1

            world_truth_live[aid] = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            if agent.bridge is not None:
                evt = {"t_s": t_next, "object_id": aid}
                if hasattr(agent.bridge, "step"):
                    try:
                        ret = agent.bridge.step(evt)
                        if ret is not None:
                            evt["bridge"] = ret
                    except Exception as ex:
                        evt["bridge_error"] = str(ex)
                self.bridge_hist[aid].append(evt)

        self.knowledge_sync.update_after_step(
            world_truth=world_truth_live,
            sample_index=k + 1,
            t_s=t_next,
        )

        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            self.truth_hist[aid][k + 1, :] = _state_truth_to_array(truth)
            if agent.belief is not None:
                self._ensure_belief_hist_width(aid, agent.belief.state.size)
                self.belief_hist[aid][k + 1, : agent.belief.state.size] = agent.belief.state

        self.current_index = k + 1
        self._emit_step_callback(self.current_index)

        if self.termination_monitor.check_earth_impact(t_s=t_next):
            return self.snapshot()
        self.termination_monitor.update_rocket_insertion(t_s=t_next)

        return self.snapshot()

    def run(self) -> dict[str, Any]:
        while not self.done:
            self.step()
        return self.build_payload()

    def build_payload(self) -> dict[str, Any]:
        n_used = self.current_index + 1
        t_out = self.t_s[:n_used].copy()
        truth_out = {k: v[:n_used, :].copy() for k, v in self.truth_hist.items()}
        target_reference_orbit_out = (
            None if self.target_reference_orbit_hist is None else self.target_reference_orbit_hist[:n_used, :].copy()
        )
        belief_out = {k: v[:n_used, :].copy() for k, v in self.belief_hist.items()}
        thrust_out = {k: v[:n_used, :].copy() for k, v in self.thrust_hist.items()}
        torque_out = {k: v[:n_used, :].copy() for k, v in self.torque_hist.items()}
        desired_attitude_out = {k: v[:n_used, :].copy() for k, v in self.desired_attitude_hist.items()}
        knowledge_out = {obs: {tgt: arr[:n_used, :].copy() for tgt, arr in by_tgt.items()} for obs, by_tgt in self.knowledge_hist.items()}
        ground_station_access, ground_station_access_summary = evaluate_ground_station_access(
            ground_stations=list(self.cfg.ground_stations),
            t_s=t_out,
            truth_hist=truth_out,
            jd_utc_start=self.cfg.simulator.initial_jd_utc,
        )
        rocket_metrics_out: dict[str, np.ndarray] = {}
        if self.rocket is not None:
            if self.rocket_stage_hist is not None:
                rocket_metrics_out["stage_index"] = self.rocket_stage_hist[:n_used].copy()
            if self.rocket_q_dyn_hist is not None:
                rocket_metrics_out["q_dyn_pa"] = self.rocket_q_dyn_hist[:n_used].copy()
            if self.rocket_mach_hist is not None:
                rocket_metrics_out["mach"] = self.rocket_mach_hist[:n_used].copy()
            if "rocket" in self.throttle_hist:
                rocket_metrics_out["throttle_cmd"] = self.throttle_hist["rocket"][:n_used].copy()

        plot_outputs = _plot_outputs(
            cfg=self.cfg,
            t_s=t_out,
            truth_hist=truth_out,
            target_reference_orbit_truth=target_reference_orbit_out,
            belief_hist=belief_out,
            thrust_hist=thrust_out,
            desired_attitude_hist=desired_attitude_out,
            knowledge_hist=knowledge_out,
            rocket_metrics=rocket_metrics_out if rocket_metrics_out else None,
            bridge_hist=self.bridge_hist,
            outdir=self.outdir,
        )
        animation_outputs = _animate_outputs(
            cfg=self.cfg,
            t_s=t_out,
            truth_hist=truth_out,
            thrust_hist=thrust_out,
            target_reference_orbit_truth=target_reference_orbit_out,
            outdir=self.outdir,
        )

        thrust_stats = {
            oid: {
                "burn_samples": int(self.burn_samples_by_object.get(oid, 0)),
                "max_accel_km_s2": float(self.max_accel_km_s2_by_object.get(oid, 0.0)),
                "total_dv_m_s": float(self.total_dv_m_s_by_object.get(oid, 0.0)),
            }
            for oid in thrust_out.keys()
        }
        summary = {
            "scenario_name": self.cfg.scenario_name,
            "scenario_description": self.cfg.scenario_description,
            "objects": sorted(list(self.agents.keys())),
            "samples": int(n_used),
            "dt_s": self.dt,
            "duration_s": float(t_out[-1]) if t_out.size else 0.0,
            "terminated_early": self.terminated_early,
            "termination_reason": self.termination_reason,
            "termination_time_s": self.termination_time_s,
            "termination_object_id": self.termination_object_id,
            "rocket_insertion_achieved": bool(self.rocket_inserted),
            "rocket_insertion_time_s": self.rocket_insertion_time_s,
            "target_reference_orbit_enabled": bool(target_reference_orbit_out is not None),
            "thrust_stats": thrust_stats,
            "attitude_guardrail_stats": get_attitude_guardrail_stats(),
            "knowledge_detection_by_observer": {
                aid: agent.knowledge_base.detection_summary()
                for aid, agent in self.agents.items()
                if agent.knowledge_base is not None
            },
            "knowledge_consistency_by_observer": {
                aid: agent.knowledge_base.consistency_summary()
                for aid, agent in self.agents.items()
                if agent.knowledge_base is not None
            },
            "ground_station_access_summary": ground_station_access_summary,
            "plot_outputs": plot_outputs,
            "animation_outputs": animation_outputs,
        }
        payload = {
            "summary": summary,
            "time_s": t_out.tolist(),
            "truth_by_object": {k: v.tolist() for k, v in truth_out.items()},
            "target_reference_orbit_truth": ([] if target_reference_orbit_out is None else target_reference_orbit_out.tolist()),
            "belief_by_object": {k: v.tolist() for k, v in belief_out.items()},
            "applied_thrust_by_object": {k: v.tolist() for k, v in thrust_out.items()},
            "applied_torque_by_object": {k: v.tolist() for k, v in torque_out.items()},
            "desired_attitude_by_object": {k: v.tolist() for k, v in desired_attitude_out.items()},
            "knowledge_by_observer": {o: {t: a.tolist() for t, a in bt.items()} for o, bt in knowledge_out.items()},
            "knowledge_detection_by_observer": dict(summary.get("knowledge_detection_by_observer", {}) or {}),
            "knowledge_consistency_by_observer": dict(summary.get("knowledge_consistency_by_observer", {}) or {}),
            "ground_station_access": ground_station_access,
            "ground_station_access_summary": ground_station_access_summary,
            "bridge_events_by_object": self.bridge_hist,
            "controller_debug_by_object": self.controller_debug_hist,
            "rocket_throttle_cmd": self.throttle_hist.get("rocket", np.array([])).tolist() if self.throttle_hist else [],
            "rocket_metrics": {k: v.tolist() for k, v in rocket_metrics_out.items()},
        }
        artifacts: dict[str, Any] = {}
        if bool(self.cfg.outputs.stats.get("save_json", True)):
            artifacts["summary_json"] = str(self.outdir / "master_run_summary.json")
        if bool(self.cfg.outputs.stats.get("save_full_log", True)):
            artifacts["run_log_json"] = str(self.outdir / "master_run_log.json")
        if plot_outputs:
            artifacts["plots"] = plot_outputs
        if animation_outputs:
            artifacts["animations"] = animation_outputs
        index_path = write_output_index(
            outdir=self.outdir,
            workflow="single_run",
            title=str(self.cfg.scenario_name or "single_run"),
            summary=summary,
            artifacts=artifacts,
        )
        summary["output_index_md"] = str(index_path)
        payload["output_index_md"] = str(index_path)
        if bool(self.cfg.outputs.stats.get("save_json", True)):
            write_json(str(self.outdir / "master_run_summary.json"), summary)
        if bool(self.cfg.outputs.stats.get("save_full_log", True)):
            write_json(str(self.outdir / "master_run_log.json"), payload)
        if bool(self.cfg.outputs.stats.get("print_summary", True)):
            print(_format_single_run_summary(summary))
        return payload


def _run_single_config(
    cfg: SimulationScenarioConfig,
    step_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    return _SingleRunEngine(cfg, step_callback=step_callback).run()


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_noninteractive_for_automation(cfg: SimulationScenarioConfig) -> SimulationScenarioConfig:
    if not (_is_truthy_env("SIM_AUTOMATION") or _is_truthy_env("CI")):
        return cfg
    root = cfg.to_dict()
    outputs = root.setdefault("outputs", {})
    mode = str(outputs.get("mode", "interactive")).strip().lower()
    if mode == "interactive":
        outputs["mode"] = "save"
    return scenario_config_from_dict(root)
