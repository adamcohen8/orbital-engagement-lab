from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .models import (
    EVENTS_RESULT_SCHEMA,
    LifecycleError,
    MalformedRunStateError,
    RunNotFoundError,
    RunPolicyError,
)
from .runner import prepare_foreground_run, run_foreground
from .service import await_run, inspect_run, reconcile_stale_run
from .store import LifecycleStore

EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_POLICY = 4
EXIT_MALFORMED = 5
EXIT_RUN_FAILED = 6


def _print_json(payload: dict[str, Any], *, compact: bool = False) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":") if compact else None), flush=True)


def _roots(args: argparse.Namespace) -> tuple[Path, Path]:
    output_raw = args.output_root or os.environ.get("OEL_OUTPUT_ROOT") or (Path.cwd() / "outputs")
    output_root = Path(output_raw).expanduser().resolve()
    managed = os.environ.get("OEL_MANAGED_DATA_ROOT", "").strip()
    state_raw = args.state_root or os.environ.get("OEL_RUN_STATE_ROOT")
    if state_raw:
        state_root = Path(state_raw).expanduser().resolve()
    elif managed:
        state_root = Path(managed).expanduser().resolve() / "state" / "run-lifecycle"
    else:
        state_root = output_root.parent / ".oel" / "run-lifecycle"
    return output_root, state_root


def _store(args: argparse.Namespace) -> LifecycleStore:
    output_root, state_root = _roots(args)
    return LifecycleStore(state_root=state_root, allowed_output_roots=(output_root,))


def _common_root_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, help="Authorized root containing lifecycle outputs.")
    parser.add_argument("--state-root", type=Path, help="Local root containing opaque run locators.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oel runs",
        description="Start, await, and inspect transport-neutral OEL foreground runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Validate and execute one trusted OEL scenario.")
    start.add_argument("--config", type=Path, required=True)
    start.add_argument("--output-dir", type=Path, required=True)
    start.add_argument("--resource-profile", default="laptop-safe")
    start.add_argument("--jsonl", action="store_true", help="Emit an accepted handle and terminal result as JSONL.")
    _common_root_options(start)

    inspect = subparsers.add_parser("inspect", help="Read and verify durable run state.")
    inspect.add_argument("run_id")
    inspect.add_argument("--expect-config-sha256")
    inspect.add_argument("--expect-manifest-ref")
    _common_root_options(inspect)

    wait = subparsers.add_parser("await", help="Wait for a bounded duration without changing run state.")
    wait.add_argument("run_id")
    wait.add_argument("--timeout", type=float, required=True)
    wait.add_argument("--wake-condition", choices=("terminal", "any_event"), default="terminal")
    wait.add_argument("--after-sequence", type=int, default=0)
    wait.add_argument("--expect-config-sha256")
    wait.add_argument("--expect-manifest-ref")
    _common_root_options(wait)

    events = subparsers.add_parser("events", help="Read the ordered at-least-once lifecycle event log.")
    events.add_argument("run_id")
    events.add_argument("--after-sequence", type=int, default=0)
    events.add_argument("--limit", type=int, default=100)
    _common_root_options(events)

    reconcile = subparsers.add_parser(
        "reconcile",
        help="Commit interrupted when a verified local execution owner has exited.",
    )
    reconcile.add_argument("run_id")
    reconcile.add_argument("--expect-config-sha256")
    reconcile.add_argument("--expect-manifest-ref")
    _common_root_options(reconcile)
    return parser


def _outcome_exit(outcome: str) -> int:
    if outcome == "not_found":
        return EXIT_NOT_FOUND
    if outcome in {"malformed_state", "observer_error", "identity_mismatch"}:
        return EXIT_MALFORMED
    return EXIT_SUCCESS


def _start(args: argparse.Namespace) -> int:
    output_root, _ = _roots(args)
    prepared = prepare_foreground_run(
        config_path=args.config,
        output_dir=args.output_dir,
        output_root=output_root,
        workspace_root=os.environ.get("OEL_WORKSPACE_ROOT") or Path.cwd(),
        resource_profile=args.resource_profile,
    )
    emitted_handle = False

    def emit(handle: Any) -> None:
        nonlocal emitted_handle
        emitted_handle = True
        if args.jsonl:
            _print_json(handle.to_dict(), compact=True)

    result = run_foreground(
        prepared,
        store=_store(args),
        on_handle=emit,
        capture_execution_output=bool(args.jsonl),
    )
    if args.jsonl:
        _print_json(result.to_dict(), compact=True)
    else:
        _print_json(result.to_dict())
    if not emitted_handle:
        return EXIT_RUN_FAILED
    return EXIT_SUCCESS if result.succeeded else EXIT_RUN_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            return _start(args)
        store = _store(args)
        if args.command == "inspect":
            result = inspect_run(
                store,
                args.run_id,
                expected_normalized_config_sha256=args.expect_config_sha256,
                expected_manifest_ref=args.expect_manifest_ref,
            )
            _print_json(result.to_dict())
            return _outcome_exit(result.outcome)
        if args.command == "await":
            result = await_run(
                store,
                args.run_id,
                timeout_s=args.timeout,
                wake_condition=args.wake_condition,
                after_sequence=args.after_sequence,
                expected_normalized_config_sha256=args.expect_config_sha256,
                expected_manifest_ref=args.expect_manifest_ref,
            )
            _print_json(result.to_dict())
            return _outcome_exit(result.outcome)
        if args.command == "events":
            events = store.read_events(
                args.run_id,
                after_sequence=args.after_sequence,
                limit=args.limit,
            )
            _print_json(
                {
                    "schema_version": EVENTS_RESULT_SCHEMA,
                    "run_id": args.run_id,
                    "events": [event.to_dict() for event in events],
                    "run_state_changed": False,
                }
            )
            return EXIT_SUCCESS
        if args.command == "reconcile":
            result = reconcile_stale_run(
                store,
                args.run_id,
                expected_normalized_config_sha256=args.expect_config_sha256,
                expected_manifest_ref=args.expect_manifest_ref,
            )
            _print_json(result.to_dict())
            return _outcome_exit(result.outcome)
    except RunPolicyError as exc:
        _print_json({"status": "policy_error", "error": str(exc)})
        return EXIT_POLICY
    except RunNotFoundError:
        _print_json({"status": "not_found", "error": "Run identity was not found."})
        return EXIT_NOT_FOUND
    except MalformedRunStateError:
        _print_json({"status": "malformed_state", "error": "Durable lifecycle state could not be verified."})
        return EXIT_MALFORMED
    except LifecycleError:
        _print_json({"status": "lifecycle_error", "error": "The lifecycle operation did not complete."})
        return EXIT_MALFORMED
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        _print_json({"status": "error", "error": str(exc)})
        return EXIT_POLICY
    parser.error("unsupported lifecycle command")
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
