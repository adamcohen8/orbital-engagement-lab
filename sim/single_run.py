from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim.config import (
    SimulationScenarioConfig,
    configured_objects,
    default_reference_object_id,
    relative_reference_for_object,
    scenario_config_from_dict,
)
from sim.core.models import Command, StateTruth
from sim.dynamics.attitude.rigid_body import get_attitude_guardrail_stats, reset_attitude_guardrail_stats
from sim.dynamics.orbit.spherical_harmonics import configure_spherical_harmonics_env
from sim.reporting.single_run_artifacts import (
    SingleRunArtifactContext,
    write_single_run_artifacts,
)
from sim.reporting.single_run_payload import SingleRunPayloadContext, build_single_run_payload
from sim.resource_limits import (
    HistoryMemoryEstimate,
    bytes_from_mb,
    configured_history_memory_limit_mb,
    enforce_history_memory_budget,
)
from sim.runtime_support import (
    AgentRuntime,
    _apply_relative_init_from_reference,
    _build_knowledge_base,
    _create_rocket_runtime,
    _create_satellite_runtime,
    _decision_truth_from_belief,
    _deploy_from_rocket,
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


@dataclass(frozen=True)
class _SingleRunPayloadParts:
    n_used: int
    t_s: np.ndarray
    truth_hist: dict[str, np.ndarray]
    target_reference_orbit_truth: np.ndarray | None
    belief_hist: dict[str, np.ndarray]
    thrust_hist: dict[str, np.ndarray]
    torque_hist: dict[str, np.ndarray]
    desired_attitude_hist: dict[str, np.ndarray]
    knowledge_hist: dict[str, dict[str, np.ndarray]]
    rocket_metrics: dict[str, np.ndarray]
    thrust_stats: dict[str, dict[str, Any]]


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
        self.outdir = Path(cfg.outputs.output_dir)

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
        self.sim_substep_s = (
            float(min(orbit_substep_s, attitude_substep_s)) if self.attitude_enabled else orbit_substep_s
        )
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

        self.agents: dict[str, AgentRuntime] = {}
        self.object_configs = configured_objects(cfg)
        for aid, agent_cfg in self.object_configs.items():
            if not bool(agent_cfg.enabled):
                continue
            if str(agent_cfg.kind).strip().lower() == "rocket":
                self.agents[aid] = _create_rocket_runtime(cfg, object_id=aid, agent_cfg=agent_cfg)
            else:
                self.agents[aid] = _create_satellite_runtime(
                    aid,
                    agent_cfg,
                    cfg,
                    np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                )
        for agent in self.agents.values():
            if agent.kind == "satellite" and agent.deploy_source == "rocket_deployment":
                agent.active = False

        self.rocket = self.agents.get("rocket") or next((a for a in self.agents.values() if a.kind == "rocket"), None)
        self.chaser = self.agents.get("chaser")
        self.target = self.agents.get("target")

        for aid, agent in self.agents.items():
            if agent.kind != "satellite" or agent.deploy_source == "rocket_deployment":
                continue
            agent_cfg = self.object_configs.get(aid)
            initial_state = dict(getattr(agent_cfg, "initial_state", {}) or {})
            reference_id = str(relative_reference_for_object(cfg, aid) or "").strip()
            reference = self.agents.get(reference_id) if reference_id else None
            if reference is not None:
                _apply_relative_init_from_reference(agent=agent, reference=reference, initial_state=initial_state)

        target_reference_id = default_reference_object_id(cfg, available_ids=self.agents.keys()) or "target"
        target_reference_section = self.object_configs.get(target_reference_id)
        target_reference_cfg = dict(
            (target_reference_section.reference_orbit if target_reference_section is not None else {}) or {}
        )
        target_reference_agent = self.agents.get(target_reference_id)
        self.target_reference_truth = None
        self.target_reference_dynamics = None
        self.target_reference_orbit_hist = None
        if (
            bool(target_reference_cfg.get("enabled", False))
            and target_reference_agent is not None
            and target_reference_agent.truth is not None
        ):
            self.target_reference_truth = target_reference_agent.truth.copy()
            self.target_reference_dynamics = replace(
                target_reference_agent.dynamics,
                disturbance_model=None,
                propagate_attitude=False,
                use_rectangular_prism_for_aero_srp=False,
                rectangular_prism_dims_m=None,
            )

        for aid, agent in self.agents.items():
            cfg_src = self.object_configs[aid]
            agent.knowledge_base = _build_knowledge_base(
                observer_id=aid,
                agent_cfg=cfg_src,
                dt_s=self.dt,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )

        self.history_memory_estimate = self._estimate_history_memory()
        enforce_history_memory_budget(self.history_memory_estimate)

        self.t_s = np.arange(self.n, dtype=float) * self.dt
        self.outdir.mkdir(parents=True, exist_ok=True)

        if self.target_reference_truth is not None and self.target_reference_orbit_hist is None:
            self.target_reference_orbit_hist = np.full((self.n, 6), np.nan)
            self.target_reference_orbit_hist[0, 0:3] = self.target_reference_truth.position_eci_km
            self.target_reference_orbit_hist[0, 3:6] = self.target_reference_truth.velocity_eci_km_s

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
        self.throttle_hist = {
            aid: np.full(self.n, np.nan) for aid, agent in self.agents.items() if agent.kind == "rocket"
        }
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
            if agent.kind == "rocket" and agent.rocket_state is not None and self.rocket_stage_hist is not None:
                self.rocket_stage_hist[0] = float(agent.rocket_state.active_stage_index)
                if self.rocket_q_dyn_hist is not None:
                    self.rocket_q_dyn_hist[0] = float(getattr(agent.rocket_state, "_last_step_q_dyn_pa", 0.0))
                if self.rocket_mach_hist is not None:
                    self.rocket_mach_hist[0] = float(getattr(agent.rocket_state, "_last_step_mach", 0.0))

        self._emit_step_callback(0)

    def _estimate_history_memory(self) -> HistoryMemoryEstimate:
        itemsize = np.dtype(float).itemsize
        samples = int(max(self.n, 0))
        active_objects = int(len(self.agents))
        float_columns = 1  # t_s
        if self.target_reference_truth is not None:
            float_columns += 6
        for agent in self.agents.values():
            float_columns += 14  # truth
            float_columns += int(agent.belief.state.size) if agent.belief is not None else 0
            float_columns += 3  # thrust
            float_columns += 3  # torque
            float_columns += 4  # desired attitude
            if agent.kind == "rocket":
                float_columns += 1  # throttle history
        if self.rocket is not None:
            float_columns += 3  # stage index, dynamic pressure, mach

        knowledge_pairs = 0
        for agent in self.agents.values():
            if agent.knowledge_base is None:
                continue
            targets = list(agent.knowledge_base.target_ids())
            knowledge_pairs += len(targets)
            float_columns += 6 * len(targets)

        retained_python_bytes_per_sample = 0
        for agent in self.agents.values():
            if agent.kind != "rocket":
                retained_python_bytes_per_sample += 4096  # controller_debug_hist row estimate
            if agent.bridge is not None:
                retained_python_bytes_per_sample += 512  # bridge event row estimate

        array_bytes = int(samples * float_columns * itemsize) + int(samples * retained_python_bytes_per_sample)
        # Payload construction copies history arrays before serialization/plotting, so budget against likely peak.
        estimated_peak_bytes = int(array_bytes * 2)
        limit_bytes = bytes_from_mb(configured_history_memory_limit_mb(self.cfg))
        return HistoryMemoryEstimate(
            samples=samples,
            active_objects=active_objects,
            knowledge_pairs=knowledge_pairs,
            array_bytes=array_bytes,
            estimated_peak_bytes=estimated_peak_bytes,
            limit_bytes=limit_bytes,
        )

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
        extra_columns = int(width) - int(hist.shape[1])
        extra_array_bytes = int(self.n * extra_columns * np.dtype(float).itemsize)
        next_estimate = HistoryMemoryEstimate(
            samples=self.history_memory_estimate.samples,
            active_objects=self.history_memory_estimate.active_objects,
            knowledge_pairs=self.history_memory_estimate.knowledge_pairs,
            array_bytes=self.history_memory_estimate.array_bytes + extra_array_bytes,
            estimated_peak_bytes=self.history_memory_estimate.estimated_peak_bytes + (2 * extra_array_bytes),
            limit_bytes=self.history_memory_estimate.limit_bytes,
        )
        enforce_history_memory_budget(next_estimate)
        self.history_memory_estimate = next_estimate
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

        if self.rocket is not None:
            for agent in self.agents.values():
                if agent.kind == "satellite" and not agent.active and agent.deploy_source == "rocket_deployment":
                    if t_next >= float(agent.deploy_time_s or 0.0):
                        _deploy_from_rocket(agent, self.rocket, t_next)

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
                if aid in self.throttle_hist:
                    self.throttle_hist[aid][k] = rocket_result.throttle
                self.thrust_hist[aid][k + 1, :] = rocket_result.thrust_eci_km_s2
                self.torque_hist[aid][k + 1, :] = rocket_result.torque_body_nm
                self.total_dv_m_s_by_object[aid] += rocket_result.delta_v_m_s
                self.max_accel_km_s2_by_object[aid] = max(
                    self.max_accel_km_s2_by_object[aid], rocket_result.max_accel_km_s2
                )
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
                self.max_accel_km_s2_by_object[aid] = max(
                    self.max_accel_km_s2_by_object[aid], sat_result.max_accel_km_s2
                )
                if sat_result.burned:
                    self.burn_samples_by_object[aid] += 1

            world_truth_live[aid] = (
                agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            )
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

    def _build_payload_parts(self) -> _SingleRunPayloadParts:
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
        knowledge_out = {
            obs: {tgt: arr[:n_used, :].copy() for tgt, arr in by_tgt.items()}
            for obs, by_tgt in self.knowledge_hist.items()
        }
        rocket_metrics_out: dict[str, np.ndarray] = {}
        if self.rocket is not None:
            rocket_object_id = str(getattr(self.rocket, "object_id", "rocket") or "rocket")
            if self.rocket_stage_hist is not None:
                rocket_metrics_out["stage_index"] = self.rocket_stage_hist[:n_used].copy()
            if self.rocket_q_dyn_hist is not None:
                rocket_metrics_out["q_dyn_pa"] = self.rocket_q_dyn_hist[:n_used].copy()
            if self.rocket_mach_hist is not None:
                rocket_metrics_out["mach"] = self.rocket_mach_hist[:n_used].copy()
            if rocket_object_id in self.throttle_hist:
                rocket_metrics_out["throttle_cmd"] = self.throttle_hist[rocket_object_id][:n_used].copy()

        thrust_stats = {
            oid: {
                "burn_samples": int(self.burn_samples_by_object.get(oid, 0)),
                "max_accel_km_s2": float(self.max_accel_km_s2_by_object.get(oid, 0.0)),
                "total_dv_m_s": float(self.total_dv_m_s_by_object.get(oid, 0.0)),
            }
            for oid in thrust_out.keys()
        }
        return _SingleRunPayloadParts(
            n_used=n_used,
            t_s=t_out,
            truth_hist=truth_out,
            target_reference_orbit_truth=target_reference_orbit_out,
            belief_hist=belief_out,
            thrust_hist=thrust_out,
            torque_hist=torque_out,
            desired_attitude_hist=desired_attitude_out,
            knowledge_hist=knowledge_out,
            rocket_metrics=rocket_metrics_out,
            thrust_stats=thrust_stats,
        )

    def _payload_from_parts(self, parts: _SingleRunPayloadParts) -> dict[str, Any]:
        return build_single_run_payload(
            SingleRunPayloadContext(
                cfg=self.cfg,
                object_ids=list(self.agents.keys()),
                dt_s=self.dt,
                t_s=parts.t_s,
                truth_hist=parts.truth_hist,
                target_reference_orbit_truth=parts.target_reference_orbit_truth,
                belief_hist=parts.belief_hist,
                thrust_hist=parts.thrust_hist,
                torque_hist=parts.torque_hist,
                desired_attitude_hist=parts.desired_attitude_hist,
                knowledge_hist=parts.knowledge_hist,
                bridge_hist=self.bridge_hist,
                controller_debug_hist=self.controller_debug_hist,
                rocket_throttle_cmd=self._primary_rocket_throttle_history(),
                rocket_metrics=parts.rocket_metrics,
                thrust_stats=parts.thrust_stats,
                attitude_guardrail_stats=get_attitude_guardrail_stats(),
                knowledge_detection_by_observer={
                    aid: agent.knowledge_base.detection_summary()
                    for aid, agent in self.agents.items()
                    if agent.knowledge_base is not None
                },
                knowledge_consistency_by_observer={
                    aid: agent.knowledge_base.consistency_summary()
                    for aid, agent in self.agents.items()
                    if agent.knowledge_base is not None
                },
                terminated_early=self.terminated_early,
                termination_reason=self.termination_reason,
                termination_time_s=self.termination_time_s,
                termination_object_id=self.termination_object_id,
                rocket_inserted=self.rocket_inserted,
                rocket_insertion_time_s=self.rocket_insertion_time_s,
            )
        )

    def _primary_rocket_throttle_history(self) -> np.ndarray:
        if not self.throttle_hist or self.rocket is None:
            return np.array([])
        rocket_object_id = str(getattr(self.rocket, "object_id", "rocket") or "rocket")
        return self.throttle_hist.get(rocket_object_id, np.array([]))

    def build_run_payload(self) -> dict[str, Any]:
        """Build the in-memory run payload without rendering or writing artifacts."""

        return self._payload_from_parts(self._build_payload_parts())

    def _write_artifacts(self, payload: dict[str, Any], parts: _SingleRunPayloadParts) -> dict[str, Any]:
        return write_single_run_artifacts(
            payload,
            SingleRunArtifactContext(
                cfg=self.cfg,
                outdir=self.outdir,
                t_s=parts.t_s,
                truth_hist=parts.truth_hist,
                target_reference_orbit_truth=parts.target_reference_orbit_truth,
                belief_hist=parts.belief_hist,
                thrust_hist=parts.thrust_hist,
                desired_attitude_hist=parts.desired_attitude_hist,
                knowledge_hist=parts.knowledge_hist,
                rocket_metrics=parts.rocket_metrics,
                bridge_hist=self.bridge_hist,
            ),
        )

    def build_payload(self) -> dict[str, Any]:
        parts = self._build_payload_parts()
        payload = self._payload_from_parts(parts)
        return self._write_artifacts(payload, parts)


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
