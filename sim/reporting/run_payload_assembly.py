"""Single-run payload slicing, assembly, and artifact dispatch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from sim.acceleration.settings import acceleration_context_from_config
from sim.dynamics.attitude.rigid_body import get_attitude_guardrail_stats
from sim.reporting.single_run_artifacts import SingleRunArtifactContext, write_single_run_artifacts
from sim.reporting.single_run_payload import SingleRunPayloadContext, build_single_run_payload

if TYPE_CHECKING:
    from sim.single_run import _SingleRunEngine


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
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]]
    rocket_metrics: dict[str, np.ndarray]
    reentry_metrics: dict[str, dict[str, np.ndarray]]
    thrust_stats: dict[str, dict[str, Any]]


class SingleRunPayloadAssembler:
    """Build reporting views without owning engine lifecycle or physics."""

    def __init__(
        self,
        engine: _SingleRunEngine,
        *,
        initialization_metadata_builder: Callable[..., dict[str, dict[str, Any]]],
    ) -> None:
        self.engine = engine
        self.initialization_metadata_builder = initialization_metadata_builder

    def build_parts(self) -> _SingleRunPayloadParts:
        engine = self.engine
        n_used = engine.current_index + 1
        t_out = engine.t_s[:n_used]
        truth_out = {key: value[:n_used, :] for key, value in engine.truth_hist.items()}
        target_reference_orbit_out = (
            None
            if engine.target_reference_orbit_hist is None
            else engine.target_reference_orbit_hist[:n_used, :]
        )
        belief_out = {key: value[:n_used, :] for key, value in engine.belief_hist.items()}
        thrust_out = {key: value[:n_used, :] for key, value in engine.thrust_hist.items()}
        torque_out = {key: value[:n_used, :] for key, value in engine.torque_hist.items()}
        desired_attitude_out = {
            key: value[:n_used, :] for key, value in engine.desired_attitude_hist.items()
        }
        knowledge_out = {
            observer: {target: arr[:n_used, :] for target, arr in by_target.items()}
            for observer, by_target in engine.knowledge_hist.items()
        }
        knowledge_measurements_out = {
            observer: {target: arr[:n_used, :] for target, arr in by_target.items()}
            for observer, by_target in getattr(engine, "knowledge_measurement_hist", {}).items()
        }
        rocket_metrics_out: dict[str, np.ndarray] = {}
        if engine.rocket is not None:
            rocket_object_id = str(getattr(engine.rocket, "object_id", "rocket") or "rocket")
            if engine.rocket_stage_hist is not None:
                rocket_metrics_out["stage_index"] = engine.rocket_stage_hist[:n_used]
            if engine.rocket_q_dyn_hist is not None:
                rocket_metrics_out["q_dyn_pa"] = engine.rocket_q_dyn_hist[:n_used]
            if engine.rocket_mach_hist is not None:
                rocket_metrics_out["mach"] = engine.rocket_mach_hist[:n_used]
            if rocket_object_id in engine.throttle_hist:
                rocket_metrics_out["throttle_cmd"] = engine.throttle_hist[rocket_object_id][:n_used]
            for metric_key, metric_hist in getattr(engine, "rocket_metric_hists", {}).items():
                rocket_metrics_out[metric_key] = metric_hist[:n_used]
        reentry_metrics_out = {
            object_id: {key: hist[:n_used] for key, hist in metrics.items()}
            for object_id, metrics in getattr(engine, "reentry_metric_hists", {}).items()
        }
        thrust_stats = {
            object_id: {
                "burn_samples": int(engine.burn_samples_by_object.get(object_id, 0)),
                "max_accel_km_s2": float(engine.max_accel_km_s2_by_object.get(object_id, 0.0)),
                "total_dv_m_s": float(engine.total_dv_m_s_by_object.get(object_id, 0.0)),
            }
            for object_id in thrust_out
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
            knowledge_measurement_hist=knowledge_measurements_out,
            rocket_metrics=rocket_metrics_out,
            reentry_metrics=reentry_metrics_out,
            thrust_stats=thrust_stats,
        )

    def payload_from_parts(self, parts: _SingleRunPayloadParts) -> dict[str, Any]:
        engine = self.engine
        runtime_profile = engine.runtime_profiler.payload(
            completed_steps=int(max(parts.n_used - 1, 0)),
            object_count=len(engine.agents),
        )
        runtime_profile["executor"] = {
            "object_step_backend": str(getattr(engine.object_step_executor, "backend_name", "unknown")),
            "object_step_workers": int(getattr(engine.object_step_executor, "max_workers", 1) or 1),
            "planner": dict(getattr(engine, "object_execution_plan", {}) or {}),
        }
        return build_single_run_payload(
            SingleRunPayloadContext(
                cfg=engine.cfg,
                object_ids=list(engine.agents.keys()),
                dt_s=engine.dt,
                t_s=parts.t_s,
                truth_hist=parts.truth_hist,
                target_reference_orbit_truth=parts.target_reference_orbit_truth,
                belief_hist=parts.belief_hist,
                thrust_hist=parts.thrust_hist,
                torque_hist=parts.torque_hist,
                desired_attitude_hist=parts.desired_attitude_hist,
                knowledge_hist=parts.knowledge_hist,
                knowledge_measurement_hist=parts.knowledge_measurement_hist,
                bridge_hist=engine.bridge_hist,
                controller_debug_hist=engine.controller_debug_hist,
                rocket_throttle_cmd=self.primary_rocket_throttle_history(),
                rocket_metrics=parts.rocket_metrics,
                reentry_metrics=parts.reentry_metrics,
                thrust_stats=parts.thrust_stats,
                runtime_profile=runtime_profile,
                object_initialization=self.initialization_metadata_builder(engine.cfg, engine.object_configs),
                object_propagation={
                    object_id: asdict(provider.metadata())
                    for object_id, provider in engine.general_propagation.items()
                },
                attitude_guardrail_stats=get_attitude_guardrail_stats(engine.attitude_guardrail_stats),
                knowledge_detection_by_observer={
                    object_id: agent.knowledge_base.detection_summary()
                    for object_id, agent in engine.agents.items()
                    if agent.knowledge_base is not None
                }
                if engine.worker_knowledge_detection_by_observer is None
                else dict(engine.worker_knowledge_detection_by_observer),
                knowledge_consistency_by_observer={
                    object_id: agent.knowledge_base.consistency_summary()
                    for object_id, agent in engine.agents.items()
                    if agent.knowledge_base is not None
                }
                if engine.worker_knowledge_consistency_by_observer is None
                else dict(engine.worker_knowledge_consistency_by_observer),
                terminated_early=engine.terminated_early,
                termination_reason=engine.termination_reason,
                termination_time_s=engine.termination_time_s,
                termination_object_id=engine.termination_object_id,
                rocket_inserted=engine.rocket_inserted,
                rocket_insertion_time_s=engine.rocket_insertion_time_s,
            )
        )

    def primary_rocket_throttle_history(self) -> np.ndarray:
        engine = self.engine
        if not engine.throttle_hist or engine.rocket is None:
            return np.array([])
        rocket_object_id = str(getattr(engine.rocket, "object_id", "rocket") or "rocket")
        return engine.throttle_hist.get(rocket_object_id, np.array([]))

    def ensure_full_history_allowed(self) -> None:
        if self.engine.history_mode == "dynamic":
            raise RuntimeError("Dynamic history mode is only supported for step-driven game sessions.")

    def build_run_payload(self) -> dict[str, Any]:
        self.ensure_full_history_allowed()
        return self.payload_from_parts(self.build_parts())

    def write_artifacts(self, payload: dict[str, Any], parts: _SingleRunPayloadParts) -> dict[str, Any]:
        engine = self.engine
        return write_single_run_artifacts(
            payload,
            SingleRunArtifactContext(
                cfg=engine.cfg,
                outdir=engine.outdir,
                t_s=parts.t_s,
                truth_hist=parts.truth_hist,
                target_reference_orbit_truth=parts.target_reference_orbit_truth,
                belief_hist=parts.belief_hist,
                thrust_hist=parts.thrust_hist,
                torque_hist=parts.torque_hist,
                desired_attitude_hist=parts.desired_attitude_hist,
                knowledge_hist=parts.knowledge_hist,
                knowledge_measurement_hist=parts.knowledge_measurement_hist,
                rocket_metrics=parts.rocket_metrics,
                reentry_metrics=parts.reentry_metrics,
                bridge_hist=engine.bridge_hist,
            ),
        )

    def build_payload(self) -> dict[str, Any]:
        engine = self.engine
        self.ensure_full_history_allowed()
        if not bool(getattr(engine, "_acceleration_context_active", False)):
            with acceleration_context_from_config(engine.cfg):
                engine._acceleration_context_active = True
                try:
                    return self.build_payload()
                finally:
                    engine._acceleration_context_active = False
        parts = self.build_parts()
        payload = self.payload_from_parts(parts)
        return self.write_artifacts(payload, parts)
