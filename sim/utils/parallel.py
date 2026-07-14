from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Any

PARALLEL_WORKER_THREAD_ENV_VARS = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)

_WORKER_PROGRESS_QUEUE: Any | None = None


def set_parallel_worker_thread_limits(default_threads: str = "1") -> dict[str, str | None]:
    """Limit native math threads inherited by newly spawned worker processes."""
    previous: dict[str, str | None] = {}
    for name in PARALLEL_WORKER_THREAD_ENV_VARS:
        previous[name] = os.environ.get(name)
        if previous[name] is None:
            os.environ[name] = str(default_threads)
    return previous


def restore_env_vars(previous: dict[str, str | None]) -> None:
    """Restore environment variables captured by a worker-launch policy."""
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def initialize_worker_progress_queue(progress_queue: Any) -> None:
    """Install a launch-time queue in a process-pool worker."""
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue


def worker_progress_queue() -> Any | None:
    """Return the process-local progress queue installed by the pool initializer."""
    return _WORKER_PROGRESS_QUEUE


def iter_bounded_futures(
    executor: Any,
    worker: Callable[[Any], Any],
    tasks: Iterable[Any],
    *,
    max_in_flight: int,
    poll_interval_s: float = 0.1,
) -> Iterator[tuple[Any | None, Any | None]]:
    """Submit only a bounded number of tasks while yielding completed futures."""
    task_iter = iter(tasks)
    pending: dict[Any, Any] = {}
    limit = max(1, int(max_in_flight))

    def _fill() -> None:
        while len(pending) < limit:
            try:
                task = next(task_iter)
            except StopIteration:
                return
            pending[executor.submit(worker, task)] = task

    _fill()
    while pending:
        done_now, _ = wait(
            set(pending),
            timeout=max(float(poll_interval_s), 0.0),
            return_when=FIRST_COMPLETED,
        )
        if not done_now:
            yield None, None
            continue
        for future in done_now:
            yield future, pending.pop(future)
        _fill()


def format_parallel_fallback_reason(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return f"{type(exc).__name__}: worker process exited without an error message"
