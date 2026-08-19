"""Atomic state and local transaction locking for OEL installations."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .contracts import INSTALLATION_STATE_SCHEMA


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return destination


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return destination


def read_state(path: str | Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return dict(default or {})
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"State file must contain a JSON object: {source}")
    return value


def empty_installation_state() -> dict[str, Any]:
    return {"schema_version": INSTALLATION_STATE_SCHEMA, "installations": {}, "history": []}


class StateLock(AbstractContextManager["StateLock"]):
    def __init__(self, path: str | Path, *, operation: str, stale_after_s: float = 3600.0) -> None:
        self.path = Path(path)
        self.operation = operation
        self.stale_after_s = float(stale_after_s)
        self.transaction_id = str(uuid.uuid4())
        self._owned = False

    def __enter__(self) -> StateLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "oel.update-lock.v1",
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "pid": os.getpid(),
            "created_unix_s": time.time(),
        }
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            existing = self._existing()
            age = time.time() - float(existing.get("created_unix_s", time.time()))
            hint = " Run `oel update status` and use the documented recovery flow; do not delete an active lock."
            if age > self.stale_after_s:
                hint = " The lock appears stale; run `oel update recover-lock` after confirming no update process is active."
            raise RuntimeError(f"Another OEL state transaction is active: {existing}.{hint}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
        self._owned = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._owned:
            try:
                current = self._existing()
                if current.get("transaction_id") == self.transaction_id:
                    self.path.unlink()
            finally:
                self._owned = False

    def _existing(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"path": str(self.path), "status": "unreadable"}
        return value if isinstance(value, dict) else {"path": str(self.path), "status": "invalid"}


def recover_stale_lock(path: str | Path, *, stale_after_s: float = 3600.0) -> dict[str, Any]:
    lock = Path(path)
    if not lock.exists():
        return {"status": "ready", "removed": False, "path": str(lock)}
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot safely recover unreadable update lock: {lock}") from exc
    age = time.time() - float(payload.get("created_unix_s", time.time()))
    if age <= float(stale_after_s):
        raise RuntimeError(f"Update lock is not stale ({age:.1f} seconds old): {lock}")
    lock.unlink()
    return {"status": "ready", "removed": True, "path": str(lock), "previous": payload}
