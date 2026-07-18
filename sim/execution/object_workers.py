"""Process-worker transport and object-step result records."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from traceback import format_exc
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from sim.acceleration.settings import acceleration_context_from_config
from sim.core.models import StateBelief, StateTruth
from sim.dynamics.attitude.rigid_body import (
    activate_attitude_guardrail_stats,
    get_attitude_guardrail_stats,
)
from sim.execution.runtime_profile import _RuntimeProfiler
from sim.runtime_support import AgentRuntime
from sim.utils.parallel import restore_env_vars, set_parallel_worker_thread_limits

if TYPE_CHECKING:
    from sim.single_run import _SingleRunEngine


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
    agent.knowledge_base = message.knowledge_base
    activate_attitude_guardrail_stats(engine.attitude_guardrail_stats)
    guardrail_counts_before = get_attitude_guardrail_stats(engine.attitude_guardrail_stats)
    profiler_enabled = bool(getattr(engine.runtime_profiler, "enabled", True))
    engine.runtime_profiler = _RuntimeProfiler(object_ids=[aid], enabled=profiler_enabled)
    engine.controller_debug_hist = {aid: []}
    desired_attitude_hist = engine.desired_attitude_hist.get(aid)
    if desired_attitude_hist is None or desired_attitude_hist.shape != (1, 4):
        desired_attitude_hist = np.full((1, 4), np.nan)
    else:
        desired_attitude_hist[0, :] = np.nan
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
            engine.desired_attitude_hist[aid][0, :],
            dtype=float,
        )
        if aid in engine.desired_attitude_hist
        else None,
        belief_state=(None if belief is None else np.array(belief.state, dtype=float)),
        belief_covariance=(None if belief is None else np.array(belief.covariance, dtype=float)),
        belief_last_update_t_s=(None if belief is None else float(belief.last_update_t_s)),
        attitude_guardrail_count_deltas=guardrail_count_deltas,
    )
