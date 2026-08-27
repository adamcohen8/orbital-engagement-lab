from __future__ import annotations

import os
import time
from typing import Callable

from .models import (
    AwaitResult,
    ExecutionOwner,
    InvalidTransitionError,
    MalformedRunStateError,
    ReconcileResult,
    RunIdentityMismatchError,
    RunInspection,
    RunNotFoundError,
    RunPhase,
    RunState,
    local_host_sha256,
)
from .store import LifecycleStore

MAX_AWAIT_TIMEOUT_S = 3600.0
DEFAULT_POLL_INTERVAL_S = 0.25


def _error_payload(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _owner_observation(
    store: LifecycleStore,
    run_id: str,
    *,
    process_alive: Callable[[int], bool] = _process_alive,
) -> tuple[str, ExecutionOwner | None]:
    owner = store.read_owner(run_id)
    if owner is None:
        return "no_owner", None
    if owner.host_sha256 != local_host_sha256():
        return "owner_unverifiable", owner
    return ("owner_active", owner) if process_alive(owner.pid) else ("owner_lost", owner)


def inspect_run(
    store: LifecycleStore,
    run_id: str,
    *,
    expected_normalized_config_sha256: str | None = None,
    expected_manifest_ref: str | None = None,
) -> RunInspection:
    try:
        manifest = store.read_manifest(run_id)
        if (
            expected_normalized_config_sha256 is not None
            and manifest.identity.normalized_config_sha256 != expected_normalized_config_sha256
        ):
            raise RunIdentityMismatchError("Run normalized-config identity does not match the caller expectation.")
        if expected_manifest_ref is not None and manifest.identity.manifest_ref != expected_manifest_ref:
            raise RunIdentityMismatchError("Run manifest reference does not match the caller expectation.")
        return RunInspection("found", run_id, state=manifest.state_dict())
    except RunNotFoundError:
        return RunInspection("not_found", run_id, error=_error_payload("not_found", "Run identity was not found."))
    except RunIdentityMismatchError as exc:
        return RunInspection("identity_mismatch", run_id, error=_error_payload("identity_mismatch", str(exc)))
    except MalformedRunStateError:
        return RunInspection(
            "malformed_state",
            run_id,
            error=_error_payload("malformed_state", "Durable lifecycle state could not be verified."),
        )
    except OSError:
        return RunInspection(
            "observer_error",
            run_id,
            error=_error_payload("observer_error", "Lifecycle state could not be observed."),
        )


def await_run(
    store: LifecycleStore,
    run_id: str,
    *,
    timeout_s: float,
    wake_condition: str = "terminal",
    after_sequence: int = 0,
    expected_normalized_config_sha256: str | None = None,
    expected_manifest_ref: str | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    process_alive: Callable[[int], bool] = _process_alive,
) -> AwaitResult:
    requested = float(timeout_s)
    if requested < 0.0 or requested > MAX_AWAIT_TIMEOUT_S:
        raise ValueError(f"timeout_s must be between 0 and {MAX_AWAIT_TIMEOUT_S:.0f} seconds.")
    if wake_condition not in {"terminal", "any_event"}:
        raise ValueError("wake_condition must be 'terminal' or 'any_event'.")
    if int(after_sequence) < 0:
        raise ValueError("after_sequence must be non-negative.")
    interval = min(max(float(poll_interval_s), 0.05), 5.0)
    started = monotonic()
    deadline = started + requested

    while True:
        inspection = inspect_run(
            store,
            run_id,
            expected_normalized_config_sha256=expected_normalized_config_sha256,
            expected_manifest_ref=expected_manifest_ref,
        )
        observed = max(0.0, monotonic() - started)
        if inspection.outcome != "found" or inspection.state is None:
            return AwaitResult(
                inspection.outcome,
                run_id,
                requested,
                observed,
                error=inspection.error,
            )
        state = inspection.state
        terminal = bool(state.get("terminal", False))
        sequence = int(state.get("sequence", 0))
        if terminal or (wake_condition == "any_event" and sequence > int(after_sequence)):
            event = None
            try:
                matching = store.read_events(run_id, after_sequence=max(sequence - 1, 0), limit=1)
                if matching:
                    event = matching[0].to_dict()
            except MalformedRunStateError:
                return AwaitResult(
                    "malformed_state",
                    run_id,
                    requested,
                    observed,
                    error=_error_payload("malformed_state", "Durable lifecycle events could not be verified."),
                )
            return AwaitResult(
                "terminal" if terminal else "event",
                run_id,
                requested,
                observed,
                state=state,
                event=event,
            )
        try:
            owner_outcome, owner = _owner_observation(store, run_id, process_alive=process_alive)
        except MalformedRunStateError:
            return AwaitResult(
                "malformed_state",
                run_id,
                requested,
                observed,
                state=state,
                error=_error_payload("malformed_state", "Execution-owner state could not be verified."),
            )
        if owner_outcome == "owner_lost" and owner is not None:
            return AwaitResult(
                "owner_lost",
                run_id,
                requested,
                observed,
                state=state,
                owner=owner.to_dict(),
                error=_error_payload(
                    "owner_lost",
                    "The local foreground execution owner is no longer running.",
                ),
            )
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            return AwaitResult("still_running", run_id, requested, observed, state=state)
        sleep(min(interval, remaining))


def reconcile_stale_run(
    store: LifecycleStore,
    run_id: str,
    *,
    expected_normalized_config_sha256: str | None = None,
    expected_manifest_ref: str | None = None,
    process_alive: Callable[[int], bool] = _process_alive,
) -> ReconcileResult:
    inspection = inspect_run(
        store,
        run_id,
        expected_normalized_config_sha256=expected_normalized_config_sha256,
        expected_manifest_ref=expected_manifest_ref,
    )
    if inspection.outcome != "found" or inspection.state is None:
        return ReconcileResult(
            inspection.outcome,
            run_id,
            state=inspection.state,
            error=inspection.error,
        )
    if bool(inspection.state.get("terminal", False)):
        return ReconcileResult("already_terminal", run_id, state=inspection.state)
    try:
        owner_outcome, owner = _owner_observation(store, run_id, process_alive=process_alive)
    except MalformedRunStateError:
        return ReconcileResult(
            "malformed_state",
            run_id,
            state=inspection.state,
            error=_error_payload("malformed_state", "Execution-owner state could not be verified."),
        )
    if owner_outcome != "owner_lost" or owner is None:
        return ReconcileResult(
            owner_outcome,
            run_id,
            state=inspection.state,
            owner=None if owner is None else owner.to_dict(),
        )
    try:
        terminal = store.transition(
            run_id,
            RunState.INTERRUPTED,
            phase=RunPhase.COMMITTING_TERMINAL_STATE,
            event="interrupted",
            execution={"completed": False, "status": "interrupted"},
            error={
                "code": "execution_owner_lost",
                "type": "ExecutionOwnerLost",
                "message": "The local foreground execution owner exited before terminal commit.",
            },
        )
    except InvalidTransitionError:
        terminal = store.read_manifest(run_id)
        return ReconcileResult(
            "already_terminal" if terminal.is_terminal else "transition_race",
            run_id,
            state=terminal.state_dict(),
            owner=owner.to_dict(),
        )
    return ReconcileResult(
        "interrupted",
        run_id,
        state=terminal.state_dict(),
        owner=owner.to_dict(),
        run_state_changed=True,
    )
