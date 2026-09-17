"""Content-bound local transaction ledger for Hosted OEL client actions."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sim.study_planning import canonical_sha256

LEDGER_EVENT_SCHEMA = "oel.hosted_transaction_event.v1"


def append_event(
    path: str | Path,
    event_type: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(target.suffix + ".lock")
    descriptor = _acquire_lock(lock)
    try:
        events = read_ledger(target)
        previous = None if not events else events[-1]["event_sha256"]
        event = {
            "schema": LEDGER_EVENT_SCHEMA,
            "sequence": len(events) + 1,
            "event_type": str(event_type),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "previous_event_sha256": previous,
            "details": dict(details),
        }
        event["event_sha256"] = canonical_sha256(event)
        _atomic_write_events(target, [*events, event])
        return event
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return []
    if source.is_symlink() or not source.is_file():
        raise ValueError("Hosted transaction ledger must be a regular file.")
    events: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("schema") != LEDGER_EVENT_SCHEMA:
            raise ValueError("Hosted transaction ledger contains an unsupported event.")
        expected = canonical_sha256(
            {key: item for key, item in value.items() if key != "event_sha256"}
        )
        if (
            value.get("sequence") != sequence
            or value.get("previous_event_sha256") != previous
            or value.get("event_sha256") != expected
        ):
            raise ValueError("Hosted transaction ledger failed chain verification.")
        events.append(value)
        previous = expected
    return events


def _acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Another Hosted OEL ledger update is active.") from exc


def _atomic_write_events(path: Path, events: list[Mapping[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            for event in events:
                stream.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["LEDGER_EVENT_SCHEMA", "append_event", "read_ledger"]
