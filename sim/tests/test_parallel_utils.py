from __future__ import annotations

import os
from concurrent.futures import Future

from sim.utils.parallel import (
    format_parallel_fallback_reason,
    initialize_worker_progress_queue,
    iter_bounded_futures,
    restore_env_vars,
    set_parallel_worker_thread_limits,
    worker_progress_queue,
)


def test_format_parallel_fallback_reason_preserves_detail() -> None:
    assert format_parallel_fallback_reason(PermissionError("sandbox denied")) == "PermissionError: sandbox denied"


def test_format_parallel_fallback_reason_explains_empty_exception() -> None:
    assert (
        format_parallel_fallback_reason(EOFError())
        == "EOFError: worker process exited without an error message"
    )


def test_worker_thread_limits_preserve_explicit_user_values(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    previous = set_parallel_worker_thread_limits()

    assert previous["OMP_NUM_THREADS"] == "3"
    assert previous["OPENBLAS_NUM_THREADS"] is None
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"

    restore_env_vars(previous)
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_bounded_future_iterator_does_not_submit_entire_campaign_eagerly() -> None:
    class _ImmediateExecutor:
        def __init__(self) -> None:
            self.submitted: list[int] = []

        def submit(self, worker, task):
            self.submitted.append(task)
            future = Future()
            future.set_result(worker(task))
            return future

    executor = _ImmediateExecutor()
    iterator = iter_bounded_futures(executor, lambda value: value * 2, range(20), max_in_flight=4)

    first_future, first_task = next(iterator)

    assert len(executor.submitted) == 4
    results = [(first_task, first_future.result())]
    results.extend((task, future.result()) for future, task in iterator if future is not None)
    assert sorted(results) == [(value, value * 2) for value in range(20)]


def test_worker_progress_queue_is_installed_by_pool_initializer() -> None:
    marker = object()
    initialize_worker_progress_queue(marker)
    assert worker_progress_queue() is marker
