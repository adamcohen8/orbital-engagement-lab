from __future__ import annotations

import logging
import os
import pickle
from copy import copy
from dataclasses import asdict, dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from traceback import format_exc
from typing import Any, Callable, Protocol

import numpy as np

from sim.acceleration.settings import acceleration_context_from_config
from sim.aero import resolve_vehicle_aero_properties
from sim.config import (
    SimulationScenarioConfig,
    configured_objects,
    default_reference_object_id,
    relative_reference_for_object,
    scenario_config_from_dict,
)
from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.attitude.rigid_body import (
    activate_attitude_guardrail_stats,
    get_attitude_guardrail_stats,
    new_attitude_guardrail_stats,
)
from sim.dynamics.orbit.atmosphere import altitude_km_from_eci
from sim.dynamics.orbit.epoch import TIME_DEPENDENT_ENV_CACHE_KEY
from sim.dynamics.orbit.frames import FrameContext, frame_context_from_mapping
from sim.dynamics.orbit.sgp4 import SGP4EphemerisProvider
from sim.dynamics.orbit.spherical_harmonics import configure_spherical_harmonics_env
from sim.dynamics.orbit.tle import tle_block_initialization_metadata
from sim.dynamics.reentry import (
    REENTRY_METRIC_KEYS,
    ReentryObjectProperties,
    reentry_config_from_dynamics,
    reentry_metrics_for_state,
)
from sim.pro_features import FEATURE_OBJECT_PARALLELISM, require_pro_feature
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
    resource_profile,
)
from sim.rocket.navigation import build_rocket_nav_state
from sim.runtime_support import (
    AgentRuntime,
    _apply_relative_cislunar_init_from_reference,
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
    _BudgetedControllerProxy,
    _DecisionContext,
    _DecisionContextBuilder,
    _KnowledgeSynchronizer,
    _RocketStepper,
    _SatelliteStepper,
    _TerminationMonitor,
)
from sim.utils.parallel import restore_env_vars, set_parallel_worker_thread_limits

_AUTO_OBJECT_PARALLEL_RUN_WORK_THRESHOLD = 250.0

logger = logging.getLogger(__name__)

OBJECT_WORKER_BUDGET_ENV = "OEL_OBJECT_WORKER_BUDGET"
CAMPAIGN_WORKER_COUNT_ENV = "OEL_CAMPAIGN_WORKER_COUNT"
TOTAL_PROCESS_BUDGET_ENV = "OEL_TOTAL_PROCESS_BUDGET"


def _effective_propagation_method(cfg: SimulationScenarioConfig, agent_cfg: Any) -> str:
    orbit = dict((cfg.simulator.dynamics or {}).get("orbit", {}) or {})
    default_method = str(orbit.get("propagation_method", "special") or "special").strip().lower()
    return str(getattr(agent_cfg, "propagation_method", "") or default_method or "special").strip().lower()


def _object_initialization_metadata(
    cfg: SimulationScenarioConfig, object_configs: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for object_id, agent_cfg in sorted(dict(object_configs or {}).items()):
        if _effective_propagation_method(cfg, agent_cfg) == "general":
            continue
        initial_state = dict(getattr(agent_cfg, "initial_state", {}) or {})
        tle = initial_state.get("tle")
        if not isinstance(tle, dict):
            continue
        metadata[str(object_id)] = tle_block_initialization_metadata(
            tle,
            target_jd_utc=getattr(cfg.simulator, "initial_jd_utc", None),
            duration_s=float(getattr(cfg.simulator, "duration_s", 0.0) or 0.0),
        )
    return metadata


def _apply_frame_context_to_environment(
    env: dict[str, Any],
    *,
    frame_context: FrameContext,
    orbit_cfg: dict[str, Any],
) -> dict[str, Any]:
    out = dict(env or {})
    out["frame_context"] = frame_context
    out["frame_model"] = frame_context.legacy_frame_model
    out["frame_model_canonical"] = frame_context.model
    out["frame_provenance"] = frame_context.metadata()
    out["time_scale_model"] = frame_context.time_scale_model
    out["tt_minus_utc_s"] = float(frame_context.tt_minus_utc_s)
    if frame_context.eop_path is not None:
        out["eop_path"] = frame_context.eop_path
    if frame_context.dut1_s is not None:
        out["dut1_s"] = float(frame_context.dut1_s)
    if frame_context.xp_arcsec is not None:
        out["xp_arcsec"] = float(frame_context.xp_arcsec)
    if frame_context.yp_arcsec is not None:
        out["yp_arcsec"] = float(frame_context.yp_arcsec)
    if frame_context.dat_s is not None:
        out["dat_s"] = float(frame_context.dat_s)
    out["ddpsi_rad"] = float(frame_context.ddpsi_rad)
    out["ddeps_rad"] = float(frame_context.ddeps_rad)

    legacy_model = frame_context.legacy_frame_model
    if "drag_frame_model" not in out and orbit_cfg.get("drag_frame_model") is None:
        out["drag_frame_model"] = legacy_model
    if "density_frame_model" not in out:
        out["density_frame_model"] = str(out.get("drag_frame_model", legacy_model))
    if frame_context.eop_path is not None:
        if "drag_eop_path" not in out and orbit_cfg.get("drag_eop_path") is None:
            out["drag_eop_path"] = frame_context.eop_path
        if "density_eop_path" not in out:
            out["density_eop_path"] = str(out.get("drag_eop_path", frame_context.eop_path))
        if "spherical_harmonics_eop_path" not in out:
            out["spherical_harmonics_eop_path"] = frame_context.eop_path
    if "spherical_harmonics_frame_model" not in out:
        out["spherical_harmonics_frame_model"] = legacy_model
    return out


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
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]]
    rocket_metrics: dict[str, np.ndarray]
    reentry_metrics: dict[str, dict[str, np.ndarray]]
    thrust_stats: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ObjectStepInput:
    object_id: str
    agent: AgentRuntime
    initial_truth: StateTruth
    world_truth_decision: dict[str, StateTruth]
    t_s: float
    t_next: float
    sample_index: int


@dataclass(frozen=True)
class ObjectStepMessage:
    object_id: str
    knowledge_base: Any | None
    initial_truth: StateTruth
    world_truth_decision: dict[str, StateTruth]
    t_s: float
    t_next: float
    sample_index: int


@dataclass(frozen=True)
class ObjectKnowledgeSyncResult:
    object_id: str
    knowledge_snapshot: dict[str, StateBelief] = field(default_factory=dict)
    measurement_snapshot: dict[str, np.ndarray] = field(default_factory=dict)
    detection_summary: dict[str, Any] = field(default_factory=dict)
    consistency_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectStepResult:
    object_id: str
    stage: str
    elapsed_s: float
    truth: StateTruth
    thrust_eci_km_s2: np.ndarray
    torque_body_nm: np.ndarray
    delta_v_m_s: float = 0.0
    max_accel_km_s2: float = 0.0
    burned: bool = False
    throttle: float | None = None
    stage_index: float | None = None
    q_dyn_pa: float | None = None
    mach: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    bridge_events: list[dict[str, Any]] = field(default_factory=list)
    bridge_elapsed_s: float = 0.0
    updated_agent: AgentRuntime | None = None
    controller_debug_events: list[dict[str, Any]] = field(default_factory=list)
    profile_stage_seconds: dict[str, float] = field(default_factory=dict)
    profile_stage_counts: dict[str, int] = field(default_factory=dict)
    last_orbital_command_eval_t_s: float | None = None
    latched_orbital_thrust_cmd: np.ndarray | None = None
    desired_attitude_quat_bn: np.ndarray | None = None
    belief_state: np.ndarray | None = None
    belief_covariance: np.ndarray | None = None
    belief_last_update_t_s: float | None = None
    attitude_guardrail_count_deltas: dict[str, int] = field(default_factory=dict)


class ObjectStepExecutor(Protocol):
    backend_name: str

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        """Advance one timestep for each object input and return ordered results."""

    def shutdown(self) -> None:
        """Release executor resources."""

    def sync_after_step(
        self,
        *,
        world_truth: dict[str, StateTruth],
        sample_index: int,
        t_s: float,
    ) -> list[ObjectKnowledgeSyncResult] | None:
        """Synchronize worker knowledge after the orchestrator has the step truth."""


class ObjectStepBackendUnavailable(RuntimeError):
    """Raised when process transport fails independently of object physics."""


class SerialObjectStepExecutor:
    backend_name = "serial"
    max_workers = 1

    def __init__(self, engine: _SingleRunEngine) -> None:
        self.engine = engine

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        return [self.engine._step_object_serial(item) for item in inputs]

    def shutdown(self) -> None:
        return None

    def sync_after_step(
        self,
        *,
        world_truth: dict[str, StateTruth],
        sample_index: int,
        t_s: float,
    ) -> list[ObjectKnowledgeSyncResult] | None:
        return None


class ProcessPoolObjectStepExecutor:
    backend_name = "process_pool"

    def __init__(self, engine: _SingleRunEngine, *, max_workers: int) -> None:
        self.engine = engine
        self.max_workers = int(max(1, max_workers))
        self._worker_processes: list[Any] = []
        self._worker_connections: list[Any] = []
        self._executor_index_by_object: dict[str, int] = {}

    def _initialize_workers(self, inputs: list[ObjectStepInput]) -> None:
        worker_count = min(self.max_workers, len(inputs))
        groups: list[list[ObjectStepInput]] = [[] for _ in range(worker_count)]
        for index, item in enumerate(inputs):
            groups[index % worker_count].append(item)
        thread_env_previous = set_parallel_worker_thread_limits(default_threads="1")
        try:
            for worker_index, group in enumerate(groups):
                snapshots = {
                    item.object_id: self.engine._process_worker_engine_snapshot(item)
                    for item in group
                }
                parent_connection, worker_connection = get_context().Pipe(duplex=True)
                process = get_context().Process(
                    target=_persistent_object_worker_loop,
                    args=(worker_connection, snapshots),
                    name=f"oel-object-worker-{worker_index}",
                )
                process.start()
                worker_connection.close()
                self._worker_processes.append(process)
                self._worker_connections.append(parent_connection)
                for item in group:
                    self._executor_index_by_object[item.object_id] = worker_index
        except Exception as exc:
            self.shutdown()
            raise ObjectStepBackendUnavailable(
                "ProcessPoolObjectStepExecutor is unavailable in this environment. "
                "Use simulator.execution.object_parallelism.backend=serial or disable object_parallelism."
            ) from exc
        finally:
            restore_env_vars(thread_env_previous)

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        if len(inputs) <= 1:
            return [self.engine._step_object_serial(item) for item in inputs]
        if not self._worker_processes:
            self._initialize_workers(inputs)
        input_ids = {item.object_id for item in inputs}
        if input_ids != set(self._executor_index_by_object):
            raise RuntimeError("Persistent object worker membership changed after initialization.")
        chunks: list[list[tuple[int, ObjectStepMessage]]] = [[] for _ in self._worker_processes]
        for index, item in enumerate(inputs):
            chunks[self._executor_index_by_object[item.object_id]].append(
                (
                    index,
                    ObjectStepMessage(
                        object_id=item.object_id,
                        knowledge_base=item.agent.knowledge_base,
                        initial_truth=item.initial_truth,
                        world_truth_decision=item.world_truth_decision,
                        t_s=item.t_s,
                        t_next=item.t_next,
                        sample_index=item.sample_index,
                    ),
                )
            )
        active_indices = [index for index, chunk in enumerate(chunks) if chunk]
        try:
            for index in active_indices:
                self._worker_connections[index].send(chunks[index])
            indexed: list[tuple[int, ObjectStepResult]] = []
            for index in active_indices:
                status, payload = self._worker_connections[index].recv()
                if status != "ok":
                    message, worker_traceback = payload
                    raise RuntimeError(
                        f"Persistent object worker {index} failed: {message}\n{worker_traceback}"
                    )
                indexed.extend(payload)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise ObjectStepBackendUnavailable(
                f"Persistent object-worker transport failed: {type(exc).__name__}: {exc}"
            ) from exc
        indexed.sort(key=lambda row: row[0])
        return [result for _index, result in indexed]

    def sync_after_step(
        self,
        *,
        world_truth: dict[str, StateTruth],
        sample_index: int,
        t_s: float,
    ) -> list[ObjectKnowledgeSyncResult] | None:
        # Knowledge synchronization is intentionally centralized. It avoids a
        # full-world broadcast to every object worker and preserves one
        # deterministic observer/target update order.
        return None

    def shutdown(self) -> None:
        for connection in self._worker_connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for connection in self._worker_connections:
            connection.close()
        for process in self._worker_processes:
            process.join()
        self._worker_connections.clear()
        self._worker_processes.clear()
        self._executor_index_by_object.clear()


_PERSISTENT_OBJECT_WORKER_ENGINES: dict[str, _SingleRunEngine] = {}


def _persistent_object_workers_init(engines: dict[str, _SingleRunEngine]) -> None:
    global _PERSISTENT_OBJECT_WORKER_ENGINES
    _PERSISTENT_OBJECT_WORKER_ENGINES = dict(engines)


def _persistent_object_worker_loop(connection: Any, engines: dict[str, _SingleRunEngine]) -> None:
    _persistent_object_workers_init(engines)
    try:
        while True:
            chunk = connection.recv()
            if chunk is None:
                return
            try:
                connection.send(("ok", _persistent_object_step_batch_worker(chunk)))
            except Exception as exc:
                connection.send(("error", (str(exc), format_exc())))
                return
    finally:
        connection.close()


def _persistent_object_step_worker(message: ObjectStepMessage) -> ObjectStepResult:
    aid = str(message.object_id)
    engine = _PERSISTENT_OBJECT_WORKER_ENGINES.get(aid)
    if engine is None:
        raise RuntimeError(f"Persistent object worker for {aid!r} has not been initialized.")
    return _run_object_step_worker(engine, message)


def _persistent_object_step_batch_worker(
    chunk: list[tuple[int, ObjectStepMessage]],
) -> list[tuple[int, ObjectStepResult]]:
    return [(index, _persistent_object_step_worker(message)) for index, message in chunk]


def _run_object_step_worker(engine: _SingleRunEngine, message: ObjectStepMessage) -> ObjectStepResult:
    aid = str(message.object_id)
    agent = engine.agents[aid]
    # Knowledge is updated centrally after each object step. Refresh the
    # persistent worker's copy before its next decision so controllers and
    # estimators observe the same state as the serial execution path.
    agent.knowledge_base = message.knowledge_base
    activate_attitude_guardrail_stats(engine.attitude_guardrail_stats)
    guardrail_counts_before = get_attitude_guardrail_stats(engine.attitude_guardrail_stats)
    profiler_enabled = bool(getattr(engine.runtime_profiler, "enabled", True))
    engine.runtime_profiler = _RuntimeProfiler(object_ids=[aid], enabled=profiler_enabled)
    engine.controller_debug_hist = {aid: []}
    desired_attitude_hist = engine.desired_attitude_hist.get(aid)
    required_rows = int(message.sample_index) + 2
    if desired_attitude_hist is None:
        desired_attitude_hist = np.full((required_rows, 4), np.nan)
    elif required_rows > int(desired_attitude_hist.shape[0]):
        grow_to = max(required_rows, int(desired_attitude_hist.shape[0]) * 2)
        grown = np.full((grow_to, 4), np.nan)
        grown[: desired_attitude_hist.shape[0], :] = desired_attitude_hist
        desired_attitude_hist = grown
    else:
        desired_attitude_hist[int(message.sample_index) + 1, :] = np.nan
    engine.desired_attitude_hist = {aid: desired_attitude_hist}
    item = ObjectStepInput(
        object_id=aid,
        agent=agent,
        initial_truth=message.initial_truth,
        world_truth_decision=message.world_truth_decision,
        t_s=message.t_s,
        t_next=message.t_next,
        sample_index=message.sample_index,
    )
    with acceleration_context_from_config(engine.cfg):
        result = engine._step_object_serial(item)
    guardrail_counts_after = get_attitude_guardrail_stats(engine.attitude_guardrail_stats)
    guardrail_count_deltas = {
        name: int(value) - int(guardrail_counts_before.get(name, 0))
        for name, value in guardrail_counts_after.items()
    }
    belief = agent.belief
    # The parent owns and updates knowledge after each timestep. Do not send
    # the unchanged worker copy back inside AgentRuntime as well as in the next
    # message; preserving it in the parent removes a large redundant pickle.
    updated_agent = replace(agent, knowledge_base=None)
    return replace(
        result,
        updated_agent=updated_agent,
        controller_debug_events=list(engine.controller_debug_hist.get(aid, [])),
        profile_stage_seconds=dict(engine.runtime_profiler.object_stage_seconds.get(aid, {})),
        profile_stage_counts=dict(engine.runtime_profiler.object_stage_counts.get(aid, {})),
        last_orbital_command_eval_t_s=engine._last_orbital_command_eval_t_s.get(aid),
        latched_orbital_thrust_cmd=np.array(
            engine._latched_orbital_thrust_cmd_by_object.get(aid, engine.zero3),
            dtype=float,
        ),
        desired_attitude_quat_bn=np.array(
            engine.desired_attitude_hist[aid][item.sample_index + 1, :],
            dtype=float,
        )
        if aid in engine.desired_attitude_hist
        else None,
        belief_state=(None if belief is None else np.array(belief.state, dtype=float)),
        belief_covariance=(None if belief is None else np.array(belief.covariance, dtype=float)),
        belief_last_update_t_s=(None if belief is None else float(belief.last_update_t_s)),
        attitude_guardrail_count_deltas=guardrail_count_deltas,
    )


class _RuntimeProfiler:
    _OBJECT_WALL_STAGES = frozenset(
        {
            "rocket_step",
            "general_propagation_step",
            "satellite_step",
            "bridge_step",
        }
    )

    def __init__(self, *, object_ids: list[str], enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.stage_seconds: dict[str, float] = {}
        self.stage_counts: dict[str, int] = {}
        self.object_stage_seconds: dict[str, dict[str, float]] = {str(oid): {} for oid in object_ids}
        self.object_stage_counts: dict[str, dict[str, int]] = {str(oid): {} for oid in object_ids}

    def record_stage(self, stage: str, elapsed_s: float) -> None:
        if not self.enabled:
            return
        elapsed = float(elapsed_s)
        if elapsed < 0.0:
            return
        key = str(stage)
        self.stage_seconds[key] = float(self.stage_seconds.get(key, 0.0) + elapsed)
        self.stage_counts[key] = int(self.stage_counts.get(key, 0) + 1)

    def record_object(self, object_id: str, stage: str, elapsed_s: float) -> None:
        if not self.enabled:
            return
        elapsed = float(elapsed_s)
        if elapsed < 0.0:
            return
        oid = str(object_id)
        key = str(stage)
        by_stage = self.object_stage_seconds.setdefault(oid, {})
        by_stage[key] = float(by_stage.get(key, 0.0) + elapsed)
        by_count = self.object_stage_counts.setdefault(oid, {})
        by_count[key] = int(by_count.get(key, 0) + 1)

    def payload(self, *, completed_steps: int, object_count: int) -> dict[str, Any]:
        steps = int(max(completed_steps, 0))
        total_step_wall_s = float(self.stage_seconds.get("step_wall", 0.0))
        stage_totals = {
            key: {
                "total_s": float(value),
                "count": int(self.stage_counts.get(key, 0)),
                "mean_ms": _mean_ms(value, self.stage_counts.get(key, 0)),
                "share_of_step_wall": _safe_share(value, total_step_wall_s),
            }
            for key, value in sorted(self.stage_seconds.items())
        }
        object_totals: dict[str, Any] = {}
        for oid, by_stage in sorted(self.object_stage_seconds.items()):
            wall_total = float(
                sum(value for key, value in by_stage.items() if key in self._OBJECT_WALL_STAGES)
            )
            nested_total = float(
                sum(value for key, value in by_stage.items() if key not in self._OBJECT_WALL_STAGES)
            )
            object_totals[oid] = {
                "total_s": wall_total,
                "mean_ms_per_completed_step": _mean_ms(wall_total, steps),
                "nested_stage_total_s": nested_total,
                "stages": {
                    key: {
                        "total_s": float(value),
                        "count": int(self.object_stage_counts.get(oid, {}).get(key, 0)),
                        "mean_ms": _mean_ms(value, self.object_stage_counts.get(oid, {}).get(key, 0)),
                    }
                    for key, value in sorted(by_stage.items())
                },
            }
        slowest = sorted(
            (
                {"object_id": oid, "total_s": float(data["total_s"])}
                for oid, data in object_totals.items()
                if float(data["total_s"]) > 0.0
            ),
            key=lambda item: (-float(item["total_s"]), str(item["object_id"])),
        )[:10]
        return {
            "schema_version": 1,
            "enabled": bool(self.enabled),
            "completed_steps": steps,
            "object_count": int(object_count),
            "total_step_wall_s": total_step_wall_s,
            "mean_step_wall_ms": _mean_ms(total_step_wall_s, steps),
            "stage_totals": stage_totals,
            "object_totals": object_totals,
            "slowest_objects": slowest,
            "notes": [
                "Profiler timings use wall-clock perf_counter measurements inside the single-run engine.",
                "Object total_s is the non-overlapping object wall time; nested_stage_total_s is diagnostic detail.",
            ],
        }


def _mean_ms(total_s: float, count: int | None) -> float:
    n = int(count or 0)
    return 0.0 if n <= 0 else float(total_s) * 1000.0 / float(n)


def _safe_share(value: float, total: float) -> float:
    denom = float(total)
    return 0.0 if denom <= 0.0 else float(value) / denom


class _SingleRunEngine:
    def __init__(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: Callable[[int, int], None] | None = None,
        history_mode: str = "full",
        initial_history_capacity: int = 4096,
        max_history_samples: int = 4096,
    ) -> None:
        self.cfg = cfg
        self.active_step_callback = step_callback
        self.history_mode = str(history_mode or "full").strip().lower()
        if self.history_mode not in {"full", "dynamic"}:
            raise ValueError("history_mode must be 'full' or 'dynamic'.")
        self.dt = float(cfg.simulator.dt_s)
        controller_execution = dict(dict(getattr(cfg.simulator, "execution", {}) or {}).get("controller", {}) or {})
        self.orbit_controller_budget_ms = float(controller_execution.get("orbit_budget_ms", 2.0) or 2.0)
        self.attitude_controller_budget_ms = float(controller_execution.get("attitude_budget_ms", 2.0) or 2.0)
        self.controller_deadline_policy = str(
            controller_execution.get("deadline_policy", "record") or "record"
        ).strip().lower()
        self.planned_samples = int(np.floor(float(cfg.simulator.duration_s) / self.dt)) + 1
        self.sample_offset = 0
        self.max_history_samples = int(max(2, max_history_samples))
        if self.history_mode == "dynamic":
            self.n = int(max(2, min(self.planned_samples, self.max_history_samples, int(initial_history_capacity))))
        else:
            self.n = self.planned_samples
        self.outdir = Path(cfg.outputs.output_dir)

        seed = int(cfg.metadata.get("seed", 123))
        rng = np.random.default_rng(seed)
        dynamics_cfg = dict(cfg.simulator.dynamics or {})
        orbit_cfg = dict(dynamics_cfg.get("orbit", {}) or {})
        att_cfg = dict(dynamics_cfg.get("attitude", {}) or {})
        self.attitude_guardrail_stats = new_attitude_guardrail_stats(
            policy=str(att_cfg.get("guardrail_policy", "error") or "error")
        )
        activate_attitude_guardrail_stats(self.attitude_guardrail_stats)
        self.frame_context = frame_context_from_mapping(
            dict(getattr(cfg.simulator, "frames", {}) or {}),
            jd_utc_start=cfg.simulator.initial_jd_utc,
        )
        self.base_environment = configure_spherical_harmonics_env(dict(cfg.simulator.environment or {}), orbit_cfg)
        atmosphere_env = self.base_environment.pop("atmosphere_env", None)
        if isinstance(atmosphere_env, dict):
            self.base_environment = {**dict(atmosphere_env), **self.base_environment}
        if orbit_cfg.get("atmosphere_model") not in (None, "") and "atmosphere_model" not in self.base_environment:
            self.base_environment["atmosphere_model"] = str(orbit_cfg.get("atmosphere_model")).strip().lower()
        if cfg.simulator.initial_jd_utc is not None and "jd_utc_start" not in self.base_environment:
            self.base_environment["jd_utc_start"] = float(cfg.simulator.initial_jd_utc)
        self.base_environment = _apply_frame_context_to_environment(
            self.base_environment,
            frame_context=self.frame_context,
            orbit_cfg=orbit_cfg,
        )
        self._time_dependent_env_cache: dict[tuple, dict[str, np.ndarray]] = {}
        self.reentry_cfg = reentry_config_from_dynamics(dynamics_cfg)
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
            if agent.kind == "satellite" and agent.deploy_source in {"rocket_deployment", "rocket_insertion"}:
                agent.active = False

        self.general_propagation: dict[str, SGP4EphemerisProvider] = {}
        for aid, agent in self.agents.items():
            agent_cfg = self.object_configs.get(aid)
            if agent.kind != "satellite" or agent_cfg is None:
                continue
            if _effective_propagation_method(cfg, agent_cfg) != "general":
                continue
            initial_state = dict(getattr(agent_cfg, "initial_state", {}) or {})
            general = dict(getattr(agent_cfg, "general", {}) or {})
            if str(general.get("model", "") or "").strip().lower() != "sgp4":
                continue
            if agent.truth is None:
                continue
            provider = SGP4EphemerisProvider.from_tle_block(
                dict(initial_state.get("tle", {}) or {}),
                mass_kg=float(agent.truth.mass_kg),
                start_jd_utc=cfg.simulator.initial_jd_utc,
                duration_s=float(cfg.simulator.duration_s),
                output_frame=str(general.get("output_frame", "teme") or "teme"),
                frame_transform=general.get("frame_transform"),
                attitude_quat_bn=agent.truth.attitude_quat_bn,
                angular_rate_body_rad_s=agent.truth.angular_rate_body_rad_s,
                max_tle_age_days_warning=(
                    None
                    if general.get("max_tle_age_days_warning") is None
                    else float(general.get("max_tle_age_days_warning"))
                ),
            )
            self.general_propagation[aid] = provider
            agent.truth = provider.canonical_state_at(0.0)
            if agent.belief is not None and agent.belief.state.size >= 6:
                agent.belief.state[:6] = np.hstack((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s))
                agent.belief.last_update_t_s = 0.0

        self.rocket = self.agents.get("rocket") or next((a for a in self.agents.values() if a.kind == "rocket"), None)
        self.chaser = self.agents.get("chaser")
        self.target = self.agents.get("target")
        execution_cfg = dict(getattr(cfg.simulator, "execution", {}) or {})
        runtime_profiler_cfg = dict(execution_cfg.get("runtime_profiler", {}) or {})
        self.runtime_profiler = _RuntimeProfiler(
            object_ids=list(self.agents.keys()),
            enabled=bool(runtime_profiler_cfg.get("enabled", True)),
        )
        self.controller_debug_enabled = bool(getattr(cfg.outputs.stats, "controller_debug", True))
        self.object_step_executor: ObjectStepExecutor = SerialObjectStepExecutor(self)
        self.reentry_object_ids = self._resolve_reentry_object_ids()
        self.reentry_active_by_object = {aid: False for aid in self.reentry_object_ids}

        for aid, agent in self.agents.items():
            if agent.kind != "satellite" or agent.deploy_source in {"rocket_deployment", "rocket_insertion"}:
                continue
            if aid in self.general_propagation:
                continue
            agent_cfg = self.object_configs.get(aid)
            initial_state = dict(getattr(agent_cfg, "initial_state", {}) or {})
            reference_id = str(relative_reference_for_object(cfg, aid) or "").strip()
            reference = self.agents.get(reference_id) if reference_id else None
            if reference is not None:
                _apply_relative_init_from_reference(agent=agent, reference=reference, initial_state=initial_state)
                _apply_relative_cislunar_init_from_reference(
                    agent=agent,
                    reference=reference,
                    initial_state=initial_state,
                )

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
        self.rocket_metric_hist_keys = [
            "altitude_km",
            "speed_km_s",
            "vertical_speed_km_s",
            "horizontal_speed_km_s",
            "flight_path_angle_deg",
            "apoapsis_alt_km",
            "periapsis_alt_km",
            "eccentricity",
            "alpha_deg",
            "beta_deg",
            "tvc_gimbal_deg",
            "aero_force_n",
            "aero_moment_nm",
            "thrust_to_weight",
            "propellant_remaining_kg",
            "propellant_remaining_fraction",
            "guidance_phase_code",
        ]
        self.rocket_metric_hists = (
            {key: np.full(self.n, np.nan) for key in self.rocket_metric_hist_keys}
            if self.rocket is not None
            else {}
        )
        self.reentry_metric_hists = {
            aid: {key: np.full(self.n, np.nan) for key in REENTRY_METRIC_KEYS}
            for aid in self.reentry_object_ids
        }
        self.knowledge_hist: dict[str, dict[str, np.ndarray]] = {}
        self.knowledge_measurement_hist: dict[str, dict[str, np.ndarray]] = {}
        self.bridge_hist: dict[str, list[dict[str, Any]]] = {aid: [] for aid in self.agents.keys()}
        for aid, agent in self.agents.items():
            if agent.knowledge_base is not None:
                self.knowledge_hist[aid] = {}
                self.knowledge_measurement_hist[aid] = {}
                for tid in agent.knowledge_base.target_ids():
                    self.knowledge_hist[aid][tid] = np.full((self.n, 6), np.nan)
                    self.knowledge_measurement_hist[aid][tid] = np.full((self.n, 6), np.nan)
        self.knowledge_sync = _KnowledgeSynchronizer(self)
        self.knowledge_sync.initialize()
        self.worker_knowledge_detection_by_observer: dict[str, Any] | None = None
        self.worker_knowledge_consistency_by_observer: dict[str, Any] | None = None

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
                if agent.rocket_sim is not None:
                    nav = build_rocket_nav_state(
                        agent.rocket_state,
                        agent.rocket_sim.sim_cfg,
                        agent.rocket_sim.vehicle_cfg,
                    )
                    initial_metrics = {
                        "altitude_km": nav.altitude_km,
                        "speed_km_s": nav.speed_km_s,
                        "vertical_speed_km_s": nav.vertical_speed_km_s,
                        "horizontal_speed_km_s": nav.horizontal_speed_km_s,
                        "flight_path_angle_deg": nav.flight_path_angle_deg,
                        "apoapsis_alt_km": nav.apoapsis_alt_km,
                        "periapsis_alt_km": nav.periapsis_alt_km,
                        "eccentricity": nav.eccentricity,
                        "alpha_deg": nav.alpha_deg,
                        "beta_deg": nav.beta_deg,
                        "tvc_gimbal_deg": 0.0,
                        "aero_force_n": 0.0,
                        "aero_moment_nm": 0.0,
                        "thrust_to_weight": nav.thrust_to_weight,
                        "propellant_remaining_kg": nav.propellant_remaining_kg,
                        "propellant_remaining_fraction": nav.propellant_remaining_fraction,
                    }
                    for metric_key, metric_value in initial_metrics.items():
                        if metric_key in self.rocket_metric_hists:
                            self.rocket_metric_hists[metric_key][0] = float(metric_value)
            self._record_reentry_metrics(aid=aid, truth=truth, sample_index=0, dt_s=0.0)

        self.termination_monitor.check_reentry(t_s=float(self.t_s[0]))
        self._emit_step_callback(0)
        self.object_step_executor = self._build_object_step_executor()

    def _build_object_step_executor(self) -> ObjectStepExecutor:
        execution_cfg = dict(getattr(self.cfg.simulator, "execution", {}) or {})
        object_parallelism = dict(execution_cfg.get("object_parallelism", {}) or {})
        policy = str(execution_cfg.get("policy", "configured") or "configured").strip().lower()
        enabled = bool(object_parallelism.get("enabled", False))
        backend = str(object_parallelism.get("backend", "serial") or "serial").strip().lower()
        profile = resource_profile(getattr(self.cfg.simulator, "resource_profile", None))
        active_ids = [aid for aid, agent in self.agents.items() if agent.active]
        active_objects = len(active_ids)
        min_objects = int(object_parallelism.get("min_objects", 3) or 3)
        workers = int(object_parallelism.get("workers", 0) or 0)
        if workers <= 0:
            reserve_workers = int(object_parallelism.get("reserve_workers", 1) or 0)
            workers = max(1, int(os.cpu_count() or 1) - max(0, reserve_workers))
        max_workers = int(object_parallelism.get("max_workers", 0) or 0)
        if max_workers > 0:
            workers = min(workers, max_workers)
        workers = max(1, min(workers, active_objects))
        if profile.max_parallel_workers is not None:
            workers = max(1, min(workers, int(profile.max_parallel_workers)))
        hierarchical_object_budget: int | None = None
        raw_hierarchical_budget = os.environ.get(OBJECT_WORKER_BUDGET_ENV)
        if raw_hierarchical_budget not in (None, ""):
            hierarchical_object_budget = max(int(raw_hierarchical_budget), 0)
            workers = min(workers, hierarchical_object_budget) if hierarchical_object_budget > 0 else 1

        incompatibilities: list[str] = []
        if any(
            agent.deploy_source in {"rocket_deployment", "rocket_insertion"}
            for agent in self.agents.values()
        ):
            incompatibilities.append("dynamic object deployment is configured")
        if self.general_propagation:
            incompatibilities.append("mixed OGP/general-propagation objects are configured")
        if any(agent.bridge is not None for agent in self.agents.values()):
            incompatibilities.append("an external object bridge is configured")
        if policy in {"auto", "parallel"} and not incompatibilities:
            try:
                for aid in active_ids:
                    pickle.dumps(self.agents[aid], protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                incompatibilities.append(
                    f"object {aid!r} runtime state is not process-serializable: {exc}"
                )

        work_score = self._estimated_object_parallel_work_score(active_ids)
        planned_steps = max(
            1,
            int(
                np.ceil(
                    float(getattr(self.cfg.simulator, "duration_s", 0.0))
                    / max(float(self.dt), 1.0e-12)
                )
            ),
        )
        run_work_score = float(work_score) * float(planned_steps)
        selected_backend = "serial"
        reason = "object parallelism is disabled"

        if policy == "serial":
            reason = "simulator.execution.policy=serial"
        elif policy == "parallel":
            failures = list(incompatibilities)
            if bool(profile.force_serial):
                failures.append(f"resource profile {profile.name!r} requires serial execution")
            if active_objects < max(1, min_objects):
                failures.append(
                    f"active object count {active_objects} is below min_objects={min_objects}"
                )
            if workers <= 1:
                failures.append("fewer than two object workers are available")
            if failures:
                raise RuntimeError(
                    "Forced object-parallel execution is unavailable: " + "; ".join(failures)
                )
            selected_backend = "process_pool"
            reason = "parallel policy explicitly requires supported object workers"
        elif policy == "auto":
            if bool(profile.force_serial):
                reason = f"resource profile {profile.name!r} requires serial execution"
            elif incompatibilities:
                reason = "auto planner selected serial: " + "; ".join(incompatibilities)
            elif active_objects < max(1, min_objects):
                reason = (
                    f"auto planner selected serial: active object count {active_objects} "
                    f"is below min_objects={min_objects}"
                )
            elif workers <= 1:
                reason = "auto planner selected serial: fewer than two object workers are available"
            elif work_score < 2.5:
                reason = (
                    "auto planner selected serial: estimated object-step work score "
                    f"{work_score:.2f} is below the 2.50 crossover threshold"
                )
            elif run_work_score < _AUTO_OBJECT_PARALLEL_RUN_WORK_THRESHOLD:
                reason = (
                    "auto planner selected serial: estimated run work score "
                    f"{run_work_score:.2f} is below the "
                    f"{_AUTO_OBJECT_PARALLEL_RUN_WORK_THRESHOLD:.2f} startup-amortization threshold"
                )
            else:
                selected_backend = "process_pool"
                reason = (
                    f"auto planner selected {workers} workers for {active_objects} active objects; "
                    f"estimated work score={work_score:.2f}"
                )
        elif enabled and backend == "process_pool" and not bool(profile.force_serial):
            if active_objects >= max(1, min_objects) and workers > 1:
                selected_backend = "process_pool"
                reason = "legacy configured object parallelism is enabled"
            else:
                reason = "legacy configured object parallelism did not meet worker/object thresholds"
        elif enabled and backend not in {"serial", "process_pool"}:
            raise ValueError(f"Unsupported object step executor backend: {backend!r}.")

        allocation = self._object_worker_allocation(active_ids, workers if selected_backend == "process_pool" else 1)
        self.object_execution_plan = {
            "policy": policy,
            "requested_backend": backend,
            "selected_backend": selected_backend,
            "selected_workers": workers if selected_backend == "process_pool" else 1,
            "active_objects": active_objects,
            "min_objects": min_objects,
            "estimated_work_score": float(work_score),
            "planned_steps": int(planned_steps),
            "estimated_run_work_score": float(run_work_score),
            "eligible": not incompatibilities,
            "reason": reason,
            "allocation": allocation,
            "hierarchical_budget": {
                "object_workers_per_run": hierarchical_object_budget,
                "campaign_workers": (
                    None
                    if os.environ.get(CAMPAIGN_WORKER_COUNT_ENV) in (None, "")
                    else int(os.environ[CAMPAIGN_WORKER_COUNT_ENV])
                ),
                "total_process_budget": (
                    None
                    if os.environ.get(TOTAL_PROCESS_BUDGET_ENV) in (None, "")
                    else int(os.environ[TOTAL_PROCESS_BUDGET_ENV])
                ),
            },
        }
        if selected_backend == "process_pool":
            require_pro_feature(FEATURE_OBJECT_PARALLELISM)
            return ProcessPoolObjectStepExecutor(self, max_workers=workers)
        return SerialObjectStepExecutor(self)

    def _estimated_object_parallel_work_score(self, active_ids: list[str]) -> float:
        dynamics = dict(getattr(self.cfg.simulator, "dynamics", {}) or {})
        orbit = dict(dynamics.get("orbit", {}) or {})
        attitude = dict(dynamics.get("attitude", {}) or {})
        score = 1.0
        score += 0.5 * sum(bool(orbit.get(name, False)) for name in ("j2", "j3", "j4", "drag", "srp"))
        score += 0.75 * sum(bool(orbit.get(name, False)) for name in ("third_body_sun", "third_body_moon"))
        spherical = dict(orbit.get("spherical_harmonics", {}) or {})
        if bool(spherical.get("enabled", False)):
            degree = int(spherical.get("degree", 0) or 0)
            order = int(spherical.get("order", 0) or 0)
            score += 2.0 + 0.1 * float(max(degree, order))
        if bool(attitude.get("enabled", True)):
            attitude_substep_s = float(attitude.get("attitude_substep_s", self.dt) or self.dt)
            score += min(2.0, 0.5 * max(1.0, self.dt / max(attitude_substep_s, 1e-9)))
        if active_ids:
            controlled = sum(
                self.agents[aid].orbit_controller is not None
                or self.agents[aid].attitude_controller is not None
                for aid in active_ids
            )
            knowledge = sum(self.agents[aid].knowledge_base is not None for aid in active_ids)
            mission = sum(
                self.agents[aid].mission_execution is not None
                or self.agents[aid].mission_strategy is not None
                or bool(self.agents[aid].mission_modules)
                for aid in active_ids
            )
            scale = float(len(active_ids))
            score += 1.0 * controlled / scale
            score += 0.5 * knowledge / scale
            score += 0.5 * mission / scale
        return float(score)

    @staticmethod
    def _object_worker_allocation(object_ids: list[str], workers: int) -> dict[str, list[str]]:
        count = max(1, int(workers))
        allocation = {f"worker_{index + 1}": [] for index in range(count)}
        for index, object_id in enumerate(object_ids):
            allocation[f"worker_{index % count + 1}"].append(str(object_id))
        return allocation

    def _resolve_reentry_object_ids(self) -> list[str]:
        if not bool(self.reentry_cfg.enabled):
            return []
        configured = [str(item).strip() for item in self.reentry_cfg.object_ids if str(item).strip()]
        if configured and not any(item in {"*", "all"} for item in configured):
            return [aid for aid in configured if aid in self.agents]
        return [aid for aid, agent in self.agents.items() if agent.kind == "satellite"]

    def _reentry_properties_for_agent(self, aid: str, agent: AgentRuntime, truth: StateTruth) -> ReentryObjectProperties:
        agent_cfg = self.object_configs.get(aid)
        specs = dict(getattr(agent_cfg, "specs", {}) or {}) if agent_cfg is not None else {}
        dynamics = getattr(agent, "dynamics", None)
        aero = resolve_vehicle_aero_properties(
            specs,
            default_reference_area_m2=1.0,
            default_cd=2.2,
            default_cl=0.0,
            default_nose_radius_m=self.reentry_cfg.default_nose_radius_m,
        )
        area_m2 = float(getattr(dynamics, "area_m2", aero.reference_area_m2) or aero.reference_area_m2)
        drag_area_m2 = getattr(dynamics, "drag_area_m2", None)
        if drag_area_m2 is None:
            drag_area_m2 = aero.drag_area_m2 if aero.drag_area_m2 is not None else area_m2
        cd = float(getattr(dynamics, "cd", aero.cd) or aero.cd)
        lift_area_m2 = getattr(dynamics, "lift_area_m2", None)
        if lift_area_m2 is None:
            lift_area_m2 = aero.lift_area_m2 if aero.lift_area_m2 is not None else drag_area_m2
        cl = float(getattr(dynamics, "lift_coefficient", aero.cl) or 0.0)
        return ReentryObjectProperties(
            mass_kg=float(max(float(truth.mass_kg), 1e-12)),
            drag_area_m2=float(max(float(drag_area_m2), 0.0)),
            cd=cd,
            nose_radius_m=float(aero.nose_radius_m),
            lift_area_m2=None if lift_area_m2 is None else float(max(float(lift_area_m2), 0.0)),
            cl=cl,
        )

    def _record_reentry_metrics(self, *, aid: str, truth: StateTruth, sample_index: int, dt_s: float) -> None:
        if aid not in self.reentry_metric_hists:
            return
        altitude_km = altitude_km_from_eci(truth.position_eci_km, truth.t_s, env=self.base_environment)
        active = bool(altitude_km <= self.reentry_cfg.begin_altitude_km)
        self.reentry_active_by_object[aid] = bool(self.reentry_active_by_object.get(aid, False) or active)
        prev_heat = 0.0
        if sample_index > 0:
            prev_heat = float(self.reentry_metric_hists[aid]["heat_load_j_m2"][sample_index - 1])
        metrics = reentry_metrics_for_state(
            r_eci_km=truth.position_eci_km,
            v_eci_km_s=truth.velocity_eci_km_s,
            t_s=truth.t_s,
            dt_s=dt_s,
            cfg=self.reentry_cfg,
            props=self._reentry_properties_for_agent(aid, self.agents[aid], truth),
            env=self.base_environment,
            active=active,
            previous_heat_load_j_m2=prev_heat,
        )
        for key, value in metrics.items():
            if key in self.reentry_metric_hists[aid]:
                self.reentry_metric_hists[aid][key][sample_index] = float(value)

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
            float_columns += 17  # rocket GNC/navigation metric histories
        float_columns += len(getattr(self, "reentry_object_ids", []) or []) * len(REENTRY_METRIC_KEYS)

        knowledge_pairs = 0
        for agent in self.agents.values():
            if agent.knowledge_base is None:
                continue
            targets = list(agent.knowledge_base.target_ids())
            knowledge_pairs += len(targets)
            float_columns += 12 * len(targets)  # estimated state and raw measurement histories

        retained_python_bytes_per_sample = 0
        for agent in self.agents.values():
            if agent.kind != "rocket" and self.controller_debug_enabled:
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
        return max(self.planned_samples - 1, 0)

    @property
    def retained_start_step(self) -> int:
        return int(self.sample_offset)

    @property
    def retained_end_step(self) -> int:
        return int(self.sample_offset + int(self.current_index))

    @property
    def retained_sample_count(self) -> int:
        return int(self.current_index + 1)

    @property
    def allocated_history_samples(self) -> int:
        return int(self.n)

    @property
    def done(self) -> bool:
        current_time_s = 0.0
        if hasattr(self, "t_s") and self.t_s.size:
            current_time_s = float(self.t_s[min(int(self.current_index), self.t_s.size - 1)])
        return bool(
            self.terminated_early
            or current_time_s >= float(self.cfg.simulator.duration_s) - max(1.0e-9, 1.0e-9 * abs(float(self.cfg.simulator.duration_s)))
        )

    def _emit_step_callback(self, step: int) -> None:
        if self.active_step_callback is None:
            return
        try:
            self.active_step_callback(int(self.sample_offset + int(step)), self.total_steps)
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

    def _grow_axis0(self, arr: np.ndarray | None, rows: int, *, fill: float = np.nan) -> np.ndarray | None:
        if arr is None:
            return None
        if arr.shape[0] >= int(rows):
            return arr
        shape = (int(rows), *arr.shape[1:])
        expanded = np.full(shape, fill, dtype=arr.dtype)
        expanded[: arr.shape[0], ...] = arr
        return expanded

    def _compact_axis0_latest(
        self,
        arr: np.ndarray | None,
        *,
        start: int,
        count: int,
        fill: float = np.nan,
    ) -> np.ndarray | None:
        if arr is None:
            return None
        retained = arr[int(start) : int(start) + int(count), ...].copy()
        arr[...] = fill
        arr[: int(count), ...] = retained
        return arr

    def _compact_event_history_latest(self, rows: list[dict[str, Any]], *, retained_start_time_s: float) -> list[dict[str, Any]]:
        threshold = float(retained_start_time_s) - 1.0e-9
        retained: list[dict[str, Any]] = []
        for row in rows:
            event_t = row.get("interval_end_t_s", row.get("t_s")) if isinstance(row, dict) else None
            try:
                t_s = float(event_t)
            except (TypeError, ValueError):
                retained.append(row)
                continue
            if t_s >= threshold:
                retained.append(row)
        return retained

    def _compact_dynamic_history_if_needed(self, *, keep_latest: int | None = None) -> None:
        if self.history_mode != "dynamic" or self.current_index < self.n - 1:
            return
        if self.n < self.max_history_samples:
            return
        if keep_latest is None:
            keep_latest = max(1, (int(self.n) * 3) // 4)
        keep = int(max(1, min(int(keep_latest), self.current_index + 1)))
        start = int(self.current_index - keep + 1)
        retained_start_time_s = float(self.t_s[start])
        self.t_s = self._compact_axis0_latest(self.t_s, start=start, count=keep)
        self.target_reference_orbit_hist = self._compact_axis0_latest(
            self.target_reference_orbit_hist,
            start=start,
            count=keep,
        )
        self.truth_hist = {
            aid: self._compact_axis0_latest(hist, start=start, count=keep) for aid, hist in self.truth_hist.items()
        }
        self.belief_hist = {
            aid: self._compact_axis0_latest(hist, start=start, count=keep) for aid, hist in self.belief_hist.items()
        }
        self.thrust_hist = {
            aid: self._compact_axis0_latest(hist, start=start, count=keep) for aid, hist in self.thrust_hist.items()
        }
        self.torque_hist = {
            aid: self._compact_axis0_latest(hist, start=start, count=keep) for aid, hist in self.torque_hist.items()
        }
        self.desired_attitude_hist = {
            aid: self._compact_axis0_latest(hist, start=start, count=keep)
            for aid, hist in self.desired_attitude_hist.items()
        }
        self.throttle_hist = {
            aid: self._compact_axis0_latest(hist, start=start, count=keep) for aid, hist in self.throttle_hist.items()
        }
        self.rocket_stage_hist = self._compact_axis0_latest(self.rocket_stage_hist, start=start, count=keep)
        self.rocket_q_dyn_hist = self._compact_axis0_latest(self.rocket_q_dyn_hist, start=start, count=keep)
        self.rocket_mach_hist = self._compact_axis0_latest(self.rocket_mach_hist, start=start, count=keep)
        self.rocket_metric_hists = {
            key: self._compact_axis0_latest(hist, start=start, count=keep)
            for key, hist in self.rocket_metric_hists.items()
        }
        self.reentry_metric_hists = {
            aid: {key: self._compact_axis0_latest(hist, start=start, count=keep) for key, hist in metrics.items()}
            for aid, metrics in self.reentry_metric_hists.items()
        }
        self.knowledge_hist = {
            obs: {tgt: self._compact_axis0_latest(hist, start=start, count=keep) for tgt, hist in by_tgt.items()}
            for obs, by_tgt in self.knowledge_hist.items()
        }
        self.knowledge_measurement_hist = {
            obs: {tgt: self._compact_axis0_latest(hist, start=start, count=keep) for tgt, hist in by_tgt.items()}
            for obs, by_tgt in self.knowledge_measurement_hist.items()
        }
        self.controller_debug_hist = {
            aid: self._compact_event_history_latest(rows, retained_start_time_s=retained_start_time_s)
            for aid, rows in self.controller_debug_hist.items()
        }
        self.bridge_hist = {
            aid: self._compact_event_history_latest(rows, retained_start_time_s=retained_start_time_s)
            for aid, rows in self.bridge_hist.items()
        }
        self.sample_offset += start
        self.current_index = keep - 1

    def _ensure_sample_capacity(self, sample_index: int) -> None:
        needed = int(sample_index) + 1
        if needed <= self.n:
            return
        grow_to = max(needed, int(max(self.n * 2, self.n + 1)))
        if self.history_mode == "dynamic":
            grow_to = min(grow_to, self.max_history_samples)
            if needed > grow_to:
                raise RuntimeError("dynamic history compaction did not free space for the next sample.")
        self.n = grow_to
        self.t_s = self._grow_axis0(self.t_s, grow_to)
        self.target_reference_orbit_hist = self._grow_axis0(self.target_reference_orbit_hist, grow_to)
        self.truth_hist = {aid: self._grow_axis0(hist, grow_to) for aid, hist in self.truth_hist.items()}
        self.belief_hist = {aid: self._grow_axis0(hist, grow_to) for aid, hist in self.belief_hist.items()}
        self.thrust_hist = {aid: self._grow_axis0(hist, grow_to) for aid, hist in self.thrust_hist.items()}
        self.torque_hist = {aid: self._grow_axis0(hist, grow_to) for aid, hist in self.torque_hist.items()}
        self.desired_attitude_hist = {
            aid: self._grow_axis0(hist, grow_to) for aid, hist in self.desired_attitude_hist.items()
        }
        self.throttle_hist = {aid: self._grow_axis0(hist, grow_to) for aid, hist in self.throttle_hist.items()}
        self.rocket_stage_hist = self._grow_axis0(self.rocket_stage_hist, grow_to)
        self.rocket_q_dyn_hist = self._grow_axis0(self.rocket_q_dyn_hist, grow_to)
        self.rocket_mach_hist = self._grow_axis0(self.rocket_mach_hist, grow_to)
        self.rocket_metric_hists = {
            key: self._grow_axis0(hist, grow_to) for key, hist in self.rocket_metric_hists.items()
        }
        self.reentry_metric_hists = {
            aid: {key: self._grow_axis0(hist, grow_to) for key, hist in metrics.items()}
            for aid, metrics in self.reentry_metric_hists.items()
        }
        self.knowledge_hist = {
            obs: {tgt: self._grow_axis0(hist, grow_to) for tgt, hist in by_tgt.items()}
            for obs, by_tgt in self.knowledge_hist.items()
        }
        self.knowledge_measurement_hist = {
            obs: {tgt: self._grow_axis0(hist, grow_to) for tgt, hist in by_tgt.items()}
            for obs, by_tgt in self.knowledge_measurement_hist.items()
        }
        self.history_memory_estimate = self._estimate_history_memory()
        enforce_history_memory_budget(self.history_memory_estimate)

    def snapshot(self, step_index: int | None = None) -> dict[str, Any]:
        if step_index is None:
            idx = self.current_index
        elif self.history_mode == "dynamic":
            idx = int(step_index) - int(self.sample_offset)
        else:
            idx = int(step_index)
        if idx < 0 or idx > int(self.current_index) or idx >= self.n:
            raise IndexError(
                f"step_index {step_index} is outside retained steps "
                f"{self.retained_start_step}..{self.retained_end_step}."
            )
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
            "step_index": int(self.sample_offset + idx),
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
        if isinstance(self.object_step_executor, ProcessPoolObjectStepExecutor):
            plan = dict(getattr(self, "object_execution_plan", {}) or {})
            if plan.get("policy") == "auto":
                self.object_step_executor.shutdown()
                self.object_step_executor = SerialObjectStepExecutor(self)
                plan.update(
                    {
                        "selected_backend": "serial",
                        "selected_workers": 1,
                        "eligible": False,
                        "reason": (
                            "auto planner switched to serial because an external intent "
                            f"provider was attached for object {oid!r}"
                        ),
                        "allocation": self._object_worker_allocation(
                            [aid for aid, agent in self.agents.items() if agent.active],
                            1,
                        ),
                    }
                )
                self.object_execution_plan = plan
            else:
                raise RuntimeError(
                    "External intent providers are not supported by object-parallel execution. "
                    "Use simulator.execution.policy=auto or serial."
                )
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
        orbit_proxy = (
            None
            if ctx.orbit_controller is None
            else _BudgetedControllerProxy(
                ctx.orbit_controller,
                budget_ms=self.orbit_controller_budget_ms,
                deadline_policy=self.controller_deadline_policy,
            )
        )
        attitude_proxy = (
            None
            if ctx.attitude_controller is None
            else _BudgetedControllerProxy(
                ctx.attitude_controller,
                budget_ms=self.attitude_controller_budget_ms,
                deadline_policy=self.controller_deadline_policy,
            )
        )
        mission_out = _run_mission_modules(
            agent=agent,
            t_s=ctx.t_s,
            dt_s=ctx.dt_s,
            env=ctx.env,
            orbit_controller=orbit_proxy,
            attitude_controller=attitude_proxy,
            orb_belief=ctx.orb_belief,
            att_belief=ctx.att_belief,
        )
        mission_out.update(
            _run_mission_strategy(
                agent=agent,
                t_s=ctx.t_s,
                dt_s=ctx.dt_s,
                env=ctx.env,
                orbit_controller=orbit_proxy,
                attitude_controller=attitude_proxy,
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
                t_s=ctx.t_s,
                dt_s=ctx.dt_s,
                env=ctx.env,
                orbit_controller=orbit_proxy,
                attitude_controller=attitude_proxy,
                orb_belief=ctx.orb_belief,
                att_belief=ctx.att_belief,
            )
        )
        if orbit_proxy is not None and orbit_proxy.call_count:
            mission_out["_integrated_orbit_runtime_ms"] = float(orbit_proxy.runtime_ms)
            mission_out["_integrated_orbit_deadline_missed"] = bool(orbit_proxy.deadline_missed)
        if attitude_proxy is not None and attitude_proxy.call_count:
            mission_out["_integrated_attitude_runtime_ms"] = float(attitude_proxy.runtime_ms)
            mission_out["_integrated_attitude_deadline_missed"] = bool(attitude_proxy.deadline_missed)
        return mission_out

    def _build_object_step_inputs(
        self,
        *,
        world_truth_start: dict[str, StateTruth],
        t_s: float,
        t_next: float,
        sample_index: int,
    ) -> list[ObjectStepInput]:
        inputs: list[ObjectStepInput] = []
        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            inputs.append(
                ObjectStepInput(
                    object_id=aid,
                    agent=agent,
                    initial_truth=world_truth_start[aid],
                    # The step uses Jacobi-style immutable start-of-interval
                    # truth. Share one mapping across inputs so process chunks
                    # serialize it once rather than once per object.
                    world_truth_decision=world_truth_start,
                    t_s=float(t_s),
                    t_next=float(t_next),
                    sample_index=int(sample_index),
                )
            )
        return inputs

    def _step_object_serial(self, item: ObjectStepInput) -> ObjectStepResult:
        aid = item.object_id
        agent = item.agent
        object_t0 = perf_counter()

        if agent.kind == "rocket":
            rocket_result = self.rocket_stepper.step(
                agent=agent,
                world_truth_decision=item.world_truth_decision,
                t_s=item.t_s,
                t_next=item.t_next,
            )
            agent.truth = rocket_result.truth
            result = ObjectStepResult(
                object_id=aid,
                stage="rocket_step",
                elapsed_s=perf_counter() - object_t0,
                truth=rocket_result.truth,
                thrust_eci_km_s2=np.array(rocket_result.thrust_eci_km_s2, dtype=float),
                torque_body_nm=np.array(rocket_result.torque_body_nm, dtype=float),
                delta_v_m_s=float(rocket_result.delta_v_m_s),
                max_accel_km_s2=float(rocket_result.max_accel_km_s2),
                burned=bool(rocket_result.burned),
                throttle=rocket_result.throttle,
                stage_index=rocket_result.stage_index,
                q_dyn_pa=rocket_result.q_dyn_pa,
                mach=rocket_result.mach,
                metrics=dict(rocket_result.metrics or {}),
            )
        else:
            provider = self.general_propagation.get(aid)
            if provider is not None:
                agent.truth = provider.canonical_state_at(item.t_next)
                if agent.belief is not None and agent.belief.state.size >= 6:
                    agent.belief.state[:6] = np.hstack((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s))
                    agent.belief.last_update_t_s = item.t_next
                result = ObjectStepResult(
                    object_id=aid,
                    stage="general_propagation_step",
                    elapsed_s=perf_counter() - object_t0,
                    truth=agent.truth,
                    thrust_eci_km_s2=self.zero3.copy(),
                    torque_body_nm=self.zero3.copy(),
                )
            else:
                sat_result = self.satellite_stepper.step(
                    aid=aid,
                    agent=agent,
                    initial_truth=item.initial_truth,
                    world_truth_decision=item.world_truth_decision,
                    t_s=item.t_s,
                    t_next=item.t_next,
                    sample_index=item.sample_index,
                )
                agent.truth = sat_result.truth
                result = ObjectStepResult(
                    object_id=aid,
                    stage="satellite_step",
                    elapsed_s=perf_counter() - object_t0,
                    truth=sat_result.truth,
                    thrust_eci_km_s2=np.array(sat_result.average_thrust_eci_km_s2, dtype=float),
                    torque_body_nm=np.array(sat_result.average_torque_body_nm, dtype=float),
                    delta_v_m_s=float(sat_result.delta_v_m_s),
                    max_accel_km_s2=float(sat_result.max_accel_km_s2),
                    burned=bool(sat_result.burned),
                )

        bridge_events: list[dict[str, Any]] = []
        bridge_elapsed = 0.0
        if agent.bridge is not None:
            bridge_t0 = perf_counter()
            evt = {"t_s": float(item.t_next), "object_id": aid}
            if hasattr(agent.bridge, "step"):
                try:
                    ret = agent.bridge.step(evt)
                    if ret is not None:
                        evt["bridge"] = ret
                except Exception as ex:
                    if bool(getattr(self.cfg.simulator.plugin_validation, "strict_runtime", False)):
                        raise RuntimeError(f"{aid} bridge.step failed at t={float(item.t_next):.6g} s") from ex
                    evt["bridge_error"] = str(ex)
            bridge_events.append(evt)
            bridge_elapsed = perf_counter() - bridge_t0

        return replace(result, bridge_events=bridge_events, bridge_elapsed_s=bridge_elapsed)

    def _apply_object_step_result(self, result: ObjectStepResult, *, sample_index: int) -> None:
        aid = result.object_id
        if result.updated_agent is not None:
            result.updated_agent.knowledge_base = self.agents[aid].knowledge_base
            self.agents[aid] = result.updated_agent
            self._refresh_agent_role_pointers(aid, result.updated_agent)
        agent = self.agents[aid]
        agent.truth = result.truth
        for name, delta in result.attitude_guardrail_count_deltas.items():
            if hasattr(self.attitude_guardrail_stats, name):
                setattr(
                    self.attitude_guardrail_stats,
                    name,
                    int(getattr(self.attitude_guardrail_stats, name)) + int(delta),
                )
        if result.belief_state is not None:
            agent.belief = StateBelief(
                state=np.array(result.belief_state, dtype=float),
                covariance=(
                    self.eye6.copy()
                    if result.belief_covariance is None
                    else np.array(result.belief_covariance, dtype=float)
                ),
                last_update_t_s=(
                    float(result.belief_last_update_t_s)
                    if result.belief_last_update_t_s is not None
                    else float(result.truth.t_s)
                ),
            )
        k = int(sample_index)

        if result.stage == "rocket_step":
            if aid in self.throttle_hist and result.throttle is not None:
                self.throttle_hist[aid][k] = float(result.throttle)
            if self.rocket_stage_hist is not None and result.stage_index is not None:
                self.rocket_stage_hist[k + 1] = float(result.stage_index)
            if self.rocket_q_dyn_hist is not None and result.q_dyn_pa is not None:
                self.rocket_q_dyn_hist[k + 1] = float(result.q_dyn_pa)
            if self.rocket_mach_hist is not None and result.mach is not None:
                self.rocket_mach_hist[k + 1] = float(result.mach)
            for metric_key, metric_value in dict(result.metrics or {}).items():
                if metric_key in self.rocket_metric_hists:
                    self.rocket_metric_hists[metric_key][k + 1] = float(metric_value)

        self.thrust_hist[aid][k + 1, :] = result.thrust_eci_km_s2
        self.torque_hist[aid][k + 1, :] = result.torque_body_nm
        if result.desired_attitude_quat_bn is not None and aid in self.desired_attitude_hist:
            self.desired_attitude_hist[aid][k + 1, :] = np.array(result.desired_attitude_quat_bn, dtype=float)
        self.total_dv_m_s_by_object[aid] += float(result.delta_v_m_s)
        self.max_accel_km_s2_by_object[aid] = max(
            self.max_accel_km_s2_by_object[aid],
            float(result.max_accel_km_s2),
        )
        if result.burned:
            self.burn_samples_by_object[aid] += 1

        self.runtime_profiler.record_object(aid, result.stage, float(result.elapsed_s))
        self.runtime_profiler.record_stage("object_step", float(result.elapsed_s))
        if self.runtime_profiler.enabled:
            for stage, elapsed_s in dict(result.profile_stage_seconds).items():
                count = int(dict(result.profile_stage_counts).get(stage, 0))
                if count <= 0:
                    continue
                self.runtime_profiler.object_stage_seconds.setdefault(aid, {})[str(stage)] = float(
                    self.runtime_profiler.object_stage_seconds.setdefault(aid, {}).get(str(stage), 0.0)
                    + float(elapsed_s)
                )
                self.runtime_profiler.object_stage_counts.setdefault(aid, {})[str(stage)] = int(
                    self.runtime_profiler.object_stage_counts.setdefault(aid, {}).get(str(stage), 0) + count
                )
        if result.controller_debug_events:
            self.controller_debug_hist[aid].extend(result.controller_debug_events)
        if result.updated_agent is not None or result.last_orbital_command_eval_t_s is not None:
            self._last_orbital_command_eval_t_s[aid] = result.last_orbital_command_eval_t_s
        if result.latched_orbital_thrust_cmd is not None:
            self._latched_orbital_thrust_cmd_by_object[aid] = np.array(
                result.latched_orbital_thrust_cmd,
                dtype=float,
            )
        if result.bridge_events:
            self.bridge_hist[aid].extend(result.bridge_events)
            self.runtime_profiler.record_object(aid, "bridge_step", float(result.bridge_elapsed_s))
            self.runtime_profiler.record_stage("bridge_step", float(result.bridge_elapsed_s))

    def _refresh_agent_role_pointers(self, aid: str, agent: AgentRuntime) -> None:
        if aid == "rocket" or agent.kind == "rocket":
            self.rocket = agent
        if aid == "chaser":
            self.chaser = agent
        if aid == "target":
            self.target = agent

    def _apply_worker_knowledge_sync_results(
        self,
        results: list[ObjectKnowledgeSyncResult],
        *,
        sample_index: int,
    ) -> None:
        detection: dict[str, Any] = {}
        consistency: dict[str, Any] = {}
        for result in results:
            aid = str(result.object_id)
            if result.detection_summary:
                detection[aid] = dict(result.detection_summary)
            if result.consistency_summary:
                consistency[aid] = dict(result.consistency_summary)
            snapshot = dict(result.knowledge_snapshot)
            for tid, hist in self.knowledge_hist.get(aid, {}).items():
                if tid not in snapshot and sample_index > 0:
                    hist[sample_index, :] = hist[sample_index - 1, :]
            for tid, belief in dict(result.knowledge_snapshot).items():
                hist = self.knowledge_hist.get(aid, {}).get(tid)
                if hist is None:
                    continue
                hist[sample_index, :] = np.array(belief.state, dtype=float).reshape(-1)[:6]
            for tid, meas in dict(result.measurement_snapshot).items():
                measurement_hist = self.knowledge_measurement_hist.get(aid, {}).get(tid)
                if measurement_hist is None:
                    continue
                arr = np.array(meas, dtype=float).reshape(-1)
                n = min(int(arr.size), int(measurement_hist.shape[1]))
                measurement_hist[sample_index, :n] = arr[:n]
        if detection:
            self.worker_knowledge_detection_by_observer = detection
        if consistency:
            self.worker_knowledge_consistency_by_observer = consistency

    def _process_worker_engine_snapshot(self, item: ObjectStepInput) -> _SingleRunEngine:
        aid = item.object_id
        worker = copy(self)
        worker.active_step_callback = None
        worker.agents = {aid: item.agent}
        worker.rocket = worker.agents.get(aid) if item.agent.kind == "rocket" else None
        worker.chaser = worker.agents.get(aid) if aid == "chaser" else None
        worker.target = worker.agents.get(aid) if aid == "target" else None
        worker.general_propagation = {
            oid: provider for oid, provider in self.general_propagation.items() if oid == aid
        }
        worker.controller_debug_hist = {aid: []}
        worker.bridge_hist = {aid: []}
        worker.runtime_profiler = _RuntimeProfiler(
            object_ids=[aid],
            enabled=bool(getattr(self.runtime_profiler, "enabled", True)),
        )
        worker.object_step_executor = SerialObjectStepExecutor(worker)
        worker.rocket_stepper = _RocketStepper(worker)
        worker.satellite_stepper = _SatelliteStepper(worker)
        worker.termination_monitor = _TerminationMonitor(worker)
        worker.knowledge_sync = None
        worker.truth_hist = {}
        worker.belief_hist = {}
        worker.thrust_hist = {}
        worker.torque_hist = {}
        worker.desired_attitude_hist = {aid: np.full((int(item.sample_index) + 2, 4), np.nan)}
        worker.knowledge_hist = {}
        worker.knowledge_measurement_hist = {}
        worker._last_orbital_command_eval_t_s = {
            aid: self._last_orbital_command_eval_t_s.get(aid)
        }
        worker._latched_orbital_thrust_cmd_by_object = {
            aid: np.array(self._latched_orbital_thrust_cmd_by_object.get(aid, self.zero3), dtype=float)
        }
        return worker

    @staticmethod
    def _validate_object_step_results(
        inputs: list[ObjectStepInput],
        results: list[ObjectStepResult],
    ) -> None:
        input_ids = [item.object_id for item in inputs]
        result_ids = [item.object_id for item in results]
        if result_ids != input_ids:
            raise RuntimeError(
                "Object step executor must return exactly one result per input in input order. "
                f"expected={input_ids!r}, received={result_ids!r}"
            )

    def step(self, dt_s: float | None = None) -> dict[str, Any]:
        activate_attitude_guardrail_stats(self.attitude_guardrail_stats)
        if not bool(getattr(self, "_acceleration_context_active", False)):
            with acceleration_context_from_config(self.cfg):
                self._acceleration_context_active = True
                try:
                    return self.step(dt_s=dt_s)
                finally:
                    self._acceleration_context_active = False
        if self.done:
            return self.snapshot()

        step_wall_t0 = perf_counter()
        compact_t0 = perf_counter()
        self._compact_dynamic_history_if_needed()
        self.runtime_profiler.record_stage("dynamic_history_compaction", perf_counter() - compact_t0)
        k = int(self.current_index)
        t = float(self.t_s[k])
        step_dt = self.dt if dt_s is None else float(dt_s)
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("step dt_s must be positive.")
        remaining_s = max(float(self.cfg.simulator.duration_s) - t, 0.0)
        step_dt = min(step_dt, remaining_s)
        if step_dt <= 0.0:
            return self.snapshot()
        capacity_t0 = perf_counter()
        self._ensure_sample_capacity(k + 1)
        self.runtime_profiler.record_stage("history_capacity", perf_counter() - capacity_t0)
        t_next = float(t + step_dt)
        self.t_s[k + 1] = t_next
        self._time_dependent_env_cache = {}

        if self.rocket is not None:
            for agent in self.agents.values():
                if agent.kind == "satellite" and not agent.active and agent.deploy_source == "rocket_deployment":
                    if agent.deploy_time_s is not None and t_next >= float(agent.deploy_time_s):
                        _deploy_from_rocket(agent, self.rocket, t_next)

        snapshot_t0 = perf_counter()
        world_truth_start = {
            aid: (agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state))
            for aid, agent in self.agents.items()
            if agent.active
        }
        if self.target_reference_truth is not None:
            world_truth_start["target_reference"] = self.target_reference_truth.copy()
        world_truth_live = dict(world_truth_start)
        self.runtime_profiler.record_stage("world_snapshot", perf_counter() - snapshot_t0)

        if self.target_reference_truth is not None and self.target_reference_dynamics is not None:
            target_ref_t0 = perf_counter()
            env_ref = {
                **self.base_environment,
                "world_truth": world_truth_live,
                "attitude_disabled": True,
                TIME_DEPENDENT_ENV_CACHE_KEY: self._time_dependent_env_cache,
            }
            self.target_reference_truth = self.target_reference_dynamics.step(
                state=self.target_reference_truth,
                command=Command.zero(),
                env=env_ref,
                dt_s=step_dt,
            )
            assert self.target_reference_orbit_hist is not None
            self.target_reference_orbit_hist[k + 1, 0:3] = self.target_reference_truth.position_eci_km
            self.target_reference_orbit_hist[k + 1, 3:6] = self.target_reference_truth.velocity_eci_km_s
            world_truth_live["target_reference"] = self.target_reference_truth.copy()
            self.runtime_profiler.record_stage("target_reference_step", perf_counter() - target_ref_t0)

        object_inputs = self._build_object_step_inputs(
            world_truth_start=world_truth_start,
            t_s=t,
            t_next=t_next,
            sample_index=k,
        )
        try:
            object_results = self.object_step_executor.step_objects(object_inputs)
        except ObjectStepBackendUnavailable as exc:
            policy = str(dict(self.object_execution_plan or {}).get("policy", "configured"))
            if policy != "auto":
                raise
            self.object_step_executor.shutdown()
            self.object_step_executor = SerialObjectStepExecutor(self)
            self.object_execution_plan["initial_selected_backend"] = self.object_execution_plan.get(
                "selected_backend",
                "process_pool",
            )
            self.object_execution_plan["selected_backend"] = "serial"
            self.object_execution_plan["selected_workers"] = 1
            self.object_execution_plan["runtime_fallback_reason"] = str(exc)
            self.object_execution_plan["reason"] = (
                "auto planner fell back to serial after object-worker transport became unavailable"
            )
            object_results = self.object_step_executor.step_objects(object_inputs)
        self._validate_object_step_results(object_inputs, object_results)
        for result in object_results:
            self._apply_object_step_result(result, sample_index=k)
            agent = self.agents[result.object_id]
            world_truth_live[result.object_id] = (
                agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            )

        termination_update_t0 = perf_counter()
        self.termination_monitor.update_rocket_insertion(t_s=t_next)
        if self.rocket is not None and self.rocket_inserted:
            for aid, agent in self.agents.items():
                if agent.kind == "satellite" and not agent.active and agent.deploy_source == "rocket_insertion":
                    _deploy_from_rocket(agent, self.rocket, t_next)
                    if agent.active and agent.truth is not None:
                        world_truth_live[aid] = agent.truth
        self.runtime_profiler.record_stage("termination_update", perf_counter() - termination_update_t0)

        knowledge_t0 = perf_counter()
        worker_knowledge_results = self.object_step_executor.sync_after_step(
            world_truth=world_truth_live,
            sample_index=k + 1,
            t_s=t_next,
        )
        if worker_knowledge_results is None:
            self.knowledge_sync.update_after_step(
                world_truth=world_truth_live,
                sample_index=k + 1,
                t_s=t_next,
            )
        else:
            self._apply_worker_knowledge_sync_results(
                worker_knowledge_results,
                sample_index=k + 1,
            )
        self.runtime_profiler.record_stage("knowledge_sync", perf_counter() - knowledge_t0)

        history_t0 = perf_counter()
        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            self.truth_hist[aid][k + 1, :] = _state_truth_to_array(truth)
            if agent.belief is not None:
                self._ensure_belief_hist_width(aid, agent.belief.state.size)
                self.belief_hist[aid][k + 1, : agent.belief.state.size] = agent.belief.state
            self._record_reentry_metrics(aid=aid, truth=truth, sample_index=k + 1, dt_s=step_dt)
        self.runtime_profiler.record_stage("history_write", perf_counter() - history_t0)

        self.current_index = k + 1
        self._emit_step_callback(self.current_index)

        termination_check_t0 = perf_counter()
        if self.termination_monitor.check_reentry(t_s=t_next):
            self.runtime_profiler.record_stage("termination_check", perf_counter() - termination_check_t0)
            self.runtime_profiler.record_stage("step_wall", perf_counter() - step_wall_t0)
            return self.snapshot()
        if self.termination_monitor.check_earth_impact(t_s=t_next):
            self.runtime_profiler.record_stage("termination_check", perf_counter() - termination_check_t0)
            self.runtime_profiler.record_stage("step_wall", perf_counter() - step_wall_t0)
            return self.snapshot()
        self.runtime_profiler.record_stage("termination_check", perf_counter() - termination_check_t0)
        self.runtime_profiler.record_stage("step_wall", perf_counter() - step_wall_t0)
        return self.snapshot()

    def run(self) -> dict[str, Any]:
        self._ensure_full_history_payload_allowed()
        if not bool(getattr(self, "_acceleration_context_active", False)):
            with acceleration_context_from_config(self.cfg):
                self._acceleration_context_active = True
                try:
                    return self.run()
                finally:
                    self._acceleration_context_active = False
        try:
            while not self.done:
                self.step()
            return self.build_payload()
        finally:
            self.object_step_executor.shutdown()

    def _build_payload_parts(self) -> _SingleRunPayloadParts:
        n_used = self.current_index + 1
        t_out = self.t_s[:n_used]
        truth_out = {k: v[:n_used, :] for k, v in self.truth_hist.items()}
        target_reference_orbit_out = (
            None if self.target_reference_orbit_hist is None else self.target_reference_orbit_hist[:n_used, :]
        )
        belief_out = {k: v[:n_used, :] for k, v in self.belief_hist.items()}
        thrust_out = {k: v[:n_used, :] for k, v in self.thrust_hist.items()}
        torque_out = {k: v[:n_used, :] for k, v in self.torque_hist.items()}
        desired_attitude_out = {k: v[:n_used, :] for k, v in self.desired_attitude_hist.items()}
        knowledge_out = {
            obs: {tgt: arr[:n_used, :] for tgt, arr in by_tgt.items()}
            for obs, by_tgt in self.knowledge_hist.items()
        }
        knowledge_measurements_out = {
            obs: {tgt: arr[:n_used, :] for tgt, arr in by_tgt.items()}
            for obs, by_tgt in getattr(self, "knowledge_measurement_hist", {}).items()
        }
        rocket_metrics_out: dict[str, np.ndarray] = {}
        if self.rocket is not None:
            rocket_object_id = str(getattr(self.rocket, "object_id", "rocket") or "rocket")
            if self.rocket_stage_hist is not None:
                rocket_metrics_out["stage_index"] = self.rocket_stage_hist[:n_used]
            if self.rocket_q_dyn_hist is not None:
                rocket_metrics_out["q_dyn_pa"] = self.rocket_q_dyn_hist[:n_used]
            if self.rocket_mach_hist is not None:
                rocket_metrics_out["mach"] = self.rocket_mach_hist[:n_used]
            if rocket_object_id in self.throttle_hist:
                rocket_metrics_out["throttle_cmd"] = self.throttle_hist[rocket_object_id][:n_used]
            for metric_key, metric_hist in getattr(self, "rocket_metric_hists", {}).items():
                rocket_metrics_out[metric_key] = metric_hist[:n_used]
        reentry_metrics_out = {
            oid: {key: hist[:n_used] for key, hist in metrics.items()}
            for oid, metrics in getattr(self, "reentry_metric_hists", {}).items()
        }

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
            knowledge_measurement_hist=knowledge_measurements_out,
            rocket_metrics=rocket_metrics_out,
            reentry_metrics=reentry_metrics_out,
            thrust_stats=thrust_stats,
        )

    def _payload_from_parts(self, parts: _SingleRunPayloadParts) -> dict[str, Any]:
        runtime_profile = self.runtime_profiler.payload(
            completed_steps=int(max(parts.n_used - 1, 0)),
            object_count=len(self.agents),
        )
        runtime_profile["executor"] = {
            "object_step_backend": str(getattr(self.object_step_executor, "backend_name", "unknown")),
            "object_step_workers": int(getattr(self.object_step_executor, "max_workers", 1) or 1),
            "planner": dict(getattr(self, "object_execution_plan", {}) or {}),
        }
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
                knowledge_measurement_hist=parts.knowledge_measurement_hist,
                bridge_hist=self.bridge_hist,
                controller_debug_hist=self.controller_debug_hist,
                rocket_throttle_cmd=self._primary_rocket_throttle_history(),
                rocket_metrics=parts.rocket_metrics,
                reentry_metrics=parts.reentry_metrics,
                thrust_stats=parts.thrust_stats,
                runtime_profile=runtime_profile,
                object_initialization=_object_initialization_metadata(self.cfg, self.object_configs),
                object_propagation={
                    oid: asdict(provider.metadata()) for oid, provider in self.general_propagation.items()
                },
                attitude_guardrail_stats=get_attitude_guardrail_stats(self.attitude_guardrail_stats),
                knowledge_detection_by_observer={
                    aid: agent.knowledge_base.detection_summary()
                    for aid, agent in self.agents.items()
                    if agent.knowledge_base is not None
                }
                if self.worker_knowledge_detection_by_observer is None
                else dict(self.worker_knowledge_detection_by_observer),
                knowledge_consistency_by_observer={
                    aid: agent.knowledge_base.consistency_summary()
                    for aid, agent in self.agents.items()
                    if agent.knowledge_base is not None
                }
                if self.worker_knowledge_consistency_by_observer is None
                else dict(self.worker_knowledge_consistency_by_observer),
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

    def _ensure_full_history_payload_allowed(self) -> None:
        if self.history_mode == "dynamic":
            raise RuntimeError("Dynamic history mode is only supported for step-driven game sessions.")

    def build_run_payload(self) -> dict[str, Any]:
        """Build the in-memory run payload without rendering or writing artifacts."""

        self._ensure_full_history_payload_allowed()
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
                torque_hist=parts.torque_hist,
                desired_attitude_hist=parts.desired_attitude_hist,
                knowledge_hist=parts.knowledge_hist,
                knowledge_measurement_hist=parts.knowledge_measurement_hist,
                rocket_metrics=parts.rocket_metrics,
                reentry_metrics=parts.reentry_metrics,
                bridge_hist=self.bridge_hist,
            ),
        )

    def build_payload(self) -> dict[str, Any]:
        self._ensure_full_history_payload_allowed()
        if not bool(getattr(self, "_acceleration_context_active", False)):
            with acceleration_context_from_config(self.cfg):
                self._acceleration_context_active = True
                try:
                    return self.build_payload()
                finally:
                    self._acceleration_context_active = False
        parts = self._build_payload_parts()
        payload = self._payload_from_parts(parts)
        return self._write_artifacts(payload, parts)


def _run_single_config(
    cfg: SimulationScenarioConfig,
    step_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    with acceleration_context_from_config(cfg):
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
    return scenario_config_from_dict(root, source_path=getattr(cfg, "source_path", None))
