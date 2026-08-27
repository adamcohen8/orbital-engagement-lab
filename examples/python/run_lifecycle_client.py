"""Provider-neutral reference client for OEL's local run lifecycle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence


def _read_first_record(
    path: Path,
    *,
    process: subprocess.Popen[str],
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and value.get("schema_version") == "oel.run-handle.v1":
                    return value
            if process.poll() is not None:
                raise RuntimeError(f"OEL start exited before accepting a run:\n{text[-2000:]}")
        time.sleep(0.05)
    raise TimeoutError("OEL did not emit an accepted run handle within 30 seconds.")


def _run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if not completed.stdout.strip():
        raise RuntimeError(completed.stderr.strip() or "OEL lifecycle command returned no JSON.")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("OEL lifecycle command returned a non-object JSON value.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.state_root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.state_root / f"start-{uuid.uuid4()}.jsonl"
    start_command = [
        sys.executable,
        "-m",
        "sim.execution.run_lifecycle",
        "start",
        "--config",
        str(args.config),
        "--output-dir",
        str(args.output_dir),
        "--output-root",
        str(args.output_root),
        "--state-root",
        str(args.state_root),
        "--jsonl",
    ]
    with protocol_path.open("w", encoding="utf-8") as protocol:
        process = subprocess.Popen(start_command, stdout=protocol, stderr=protocol, text=True)
    handle = _read_first_record(protocol_path, process=process)
    if handle.get("schema_version") != "oel.run-handle.v1":
        raise RuntimeError("OEL start did not emit a valid run-handle record.")
    print(json.dumps({"event": "accepted", "handle": handle}, sort_keys=True), flush=True)

    run_id = str(handle["run_id"])
    await_result = _run_json(
        [
            sys.executable,
            "-m",
            "sim.execution.run_lifecycle",
            "await",
            run_id,
            "--timeout",
            str(args.timeout),
            "--expect-config-sha256",
            str(handle["normalized_config_sha256"]),
            "--expect-manifest-ref",
            str(handle["manifest_ref"]),
            "--output-root",
            str(args.output_root),
            "--state-root",
            str(args.state_root),
        ]
    )
    print(json.dumps({"event": "wake", "result": await_result}, sort_keys=True), flush=True)
    if await_result.get("outcome") == "owner_lost":
        reconciled = _run_json(
            [
                sys.executable,
                "-m",
                "sim.execution.run_lifecycle",
                "reconcile",
                run_id,
                "--expect-config-sha256",
                str(handle["normalized_config_sha256"]),
                "--expect-manifest-ref",
                str(handle["manifest_ref"]),
                "--output-root",
                str(args.output_root),
                "--state-root",
                str(args.state_root),
            ]
        )
        print(json.dumps({"event": "reconcile", "result": reconciled}, sort_keys=True), flush=True)
        process.wait(timeout=30.0)
        protocol_path.unlink(missing_ok=True)
        return 1
    if await_result.get("outcome") != "terminal":
        return 2
    process.wait(timeout=30.0)
    protocol_path.unlink(missing_ok=True)
    state = dict(await_result.get("state", {}) or {})
    return 0 if state.get("state") == "completed" and process.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
