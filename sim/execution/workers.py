from __future__ import annotations

import os
from typing import Any

from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.execution.metrics import closest_approach_from_run_payload, relative_range_series_from_run_payload
from sim.single_run import _run_single_config

_PARALLEL_WORKER_THREAD_ENV_VARS = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def set_parallel_worker_thread_limits(default_threads: str = "1") -> dict[str, str | None]:
    """Limit native math library threads for spawned workers unless the user already set them."""
    previous: dict[str, str | None] = {}
    for name in _PARALLEL_WORKER_THREAD_ENV_VARS:
        previous[name] = os.environ.get(name)
        if previous[name] is None:
            os.environ[name] = str(default_threads)
    return previous


def restore_env_vars(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def run_mc_iteration_from_dict(task: dict[str, Any]) -> dict[str, Any]:
    iteration = int(task.get("iteration", 0))
    cdict = dict(task.get("config_dict", {}) or {})
    strict_plugins = bool(task.get("strict_plugins", True))
    progress_queue = task.get("progress_queue")
    emit_every = int(task.get("progress_emit_every", 20) or 20)
    emit_every = max(1, emit_every)
    ci = scenario_config_from_dict(cdict)
    if strict_plugins:
        errs = validate_scenario_plugins(ci)
        if errs:
            msg = f"Plugin validation failed in Monte Carlo iteration {iteration}:\n- " + "\n- ".join(errs)
            raise ValueError(msg)

    last_emit = -(10**9)

    def _on_step(step: int, total: int) -> None:
        nonlocal last_emit
        if progress_queue is None:
            return
        s = max(int(step), 0)
        t = max(int(total), 0)
        should_emit = (s == 0) or (t > 0 and s >= t) or (s - last_emit >= emit_every)
        if not should_emit:
            return
        last_emit = s
        try:
            progress_queue.put(
                {
                    "event": "step",
                    "pid": int(os.getpid()),
                    "iteration": int(iteration),
                    "step": int(s),
                    "total": int(t),
                }
            )
        except Exception:
            pass

    ro = _run_single_config(ci, step_callback=_on_step if progress_queue is not None else None)
    if progress_queue is not None:
        try:
            progress_queue.put(
                {
                    "event": "done",
                    "pid": int(os.getpid()),
                    "iteration": int(iteration),
                }
            )
        except Exception:
            pass
    return {
        "iteration": iteration,
        "summary": ro["summary"],
        "closest_approach_km": closest_approach_from_run_payload(ro),
        "relative_range_series": relative_range_series_from_run_payload(ro),
    }


_set_parallel_worker_thread_limits = set_parallel_worker_thread_limits
_restore_env_vars = restore_env_vars
_run_mc_iteration_from_dict = run_mc_iteration_from_dict
