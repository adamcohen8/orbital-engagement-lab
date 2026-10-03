"""Content-bound canonical ECI history exported from a completed review store."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator

from sim.analysis.history_adapters import AnalysisHistory, history_from_review_store
from sim.analysis.spacecraft_power import power_history_from_mapping, power_history_to_dict
from sim.utils.io import read_regular_file_nofollow

ORBIT_HISTORY_PRODUCT_SCHEMA = "oel.orbit_history_product.v1"
_MAX_JSON_BYTES = 64 * 1024 * 1024
_SQLITE_BUSY_TIMEOUT_MS = 1000
_SQLITE_BACKUP_TIMEOUT_S = 30.0


class OrbitHistoryProductError(ValueError):
    """Raised when a retained orbit history fails its contract."""


def orbit_history_semantic_sha256(history: AnalysisHistory) -> str:
    payload = power_history_to_dict(history)
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_file_sha256(path: Path) -> str | None:
    if path.is_symlink():
        raise OrbitHistoryProductError("Completed-run SQLite WAL must not be symbolic.")
    if not path.exists():
        return None
    if not path.is_file():
        raise OrbitHistoryProductError("Completed-run SQLite WAL is not a regular file.")
    return _file_sha256(path)


@contextmanager
def _consistent_review_database_snapshot(database: Path) -> Iterator[tuple[Path, str]]:
    """Yield one readable database snapshot and its matching source digest."""

    wal_path = database.with_name(database.name + "-wal")
    try:
        with tempfile.TemporaryDirectory(prefix="oel-orbit-history-snapshot-") as temporary:
            snapshot = Path(temporary) / database.name
            uri = f"{database.resolve().as_uri()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000.0)) as source:
                source.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
                source.execute("PRAGMA query_only = ON")
                source.execute("BEGIN")
                source.execute("SELECT rootpage FROM sqlite_schema LIMIT 1").fetchone()
                journal_mode = str(source.execute("PRAGMA journal_mode").fetchone()[0] or "").lower()
                database_sha_before = _file_sha256(database)
                wal_sha_before = _optional_file_sha256(wal_path) if journal_mode == "wal" else None
                backup_started = time.monotonic()

                def check_backup_timeout(_status: int, _remaining: int, _total: int) -> None:
                    if time.monotonic() - backup_started > _SQLITE_BACKUP_TIMEOUT_S:
                        raise OrbitHistoryProductError(
                            "Timed out creating a consistent snapshot of the completed-run review database."
                        )

                with closing(sqlite3.connect(snapshot, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000.0)) as target:
                    target.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
                    source.backup(
                        target,
                        pages=256,
                        sleep=0.01,
                        progress=check_backup_timeout,
                    )
                snapshot_sha = _file_sha256(snapshot)
                # Match the established review-evidence identity convention:
                # raw database bytes without WAL, backup snapshot bytes with WAL.
                source_review_sha256 = snapshot_sha if journal_mode == "wal" else database_sha_before
                yield snapshot, source_review_sha256
                if (
                    _file_sha256(database) != database_sha_before
                    or (
                        journal_mode == "wal"
                        and _optional_file_sha256(wal_path) != wal_sha_before
                    )
                ):
                    raise OrbitHistoryProductError(
                        "Completed-run review database changed during orbit-history export."
                    )
    except OrbitHistoryProductError:
        raise
    except sqlite3.Error as exc:
        raise OrbitHistoryProductError(
            "Completed-run review database could not be snapshotted consistently."
        ) from exc


def verify_orbit_history_product(output_dir: str | Path) -> tuple[dict[str, Any], AnalysisHistory]:
    """Check exact retained inventory, canonical state content, and receipts."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise OrbitHistoryProductError("Orbit-history product must not be a symbolic link.")
    root = requested.resolve()
    if not root.is_dir() or {item.name for item in root.iterdir()} != {
        "normalized_history.json", "orbit_history_manifest.json"
    }:
        raise OrbitHistoryProductError("Orbit-history product has an unexpected artifact inventory.")
    history_bytes = read_regular_file_nofollow(
        root / "normalized_history.json", min_bytes=1, max_bytes=_MAX_JSON_BYTES
    )
    manifest_bytes = read_regular_file_nofollow(
        root / "orbit_history_manifest.json", min_bytes=1, max_bytes=_MAX_JSON_BYTES
    )
    payload = json.loads(history_bytes)
    manifest = json.loads(manifest_bytes)
    if not isinstance(payload, dict) or not isinstance(manifest, dict):
        raise OrbitHistoryProductError("Orbit-history artifacts must be JSON objects.")
    history = power_history_from_mapping(payload)
    if payload != power_history_to_dict(history):
        raise OrbitHistoryProductError("Orbit history is not canonically normalized.")
    if set(manifest) != {
        "schema_version", "status", "asset_id", "epoch_jd_utc", "frame", "sample_count",
        "history_semantic_sha256", "history_file_sha256", "source_review_sha256",
        "source_config_sha256",
    } or manifest["schema_version"] != ORBIT_HISTORY_PRODUCT_SCHEMA:
        raise OrbitHistoryProductError("Unsupported orbit-history manifest contract.")
    if (
        manifest["status"] != "verified"
        or manifest["asset_id"] != history.object_id
        or manifest["epoch_jd_utc"] != history.initial_jd_utc
        or manifest["frame"] != "eci"
        or manifest["sample_count"] != int(history.times_s.size)
        or manifest["history_semantic_sha256"] != orbit_history_semantic_sha256(history)
        or manifest["history_file_sha256"] != hashlib.sha256(history_bytes).hexdigest()
    ):
        raise OrbitHistoryProductError("Orbit-history manifest differs from retained state evidence.")
    for key in ("source_review_sha256", "source_config_sha256"):
        value = manifest[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise OrbitHistoryProductError(f"Invalid {key} in orbit-history manifest.")
    return manifest, history


def export_orbit_history_product(
    completed_run: str | Path, *, object_id: str, output_dir: str | Path,
) -> dict[str, Any]:
    """Export one completed-run object's exact ECI review history."""

    source = Path(completed_run).expanduser().resolve()
    database = source / "review" / "run.sqlite" if source.is_dir() else source
    if database.is_symlink() or not database.is_file():
        raise OrbitHistoryProductError("Completed-run review database is missing or symbolic.")
    config = (source / "effective_config.json") if source.is_dir() else database.parent.parent / "effective_config.json"
    if config.is_symlink() or not config.is_file():
        raise OrbitHistoryProductError("Completed run lacks effective configuration provenance.")
    with _consistent_review_database_snapshot(database) as (snapshot, source_review_sha256):
        history = history_from_review_store(snapshot, object_id=object_id)
        config_bytes = read_regular_file_nofollow(config, min_bytes=1, max_bytes=_MAX_JSON_BYTES)
        source_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        with closing(sqlite3.connect(snapshot, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000.0)) as connection:
            connection.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
            row = connection.execute("SELECT config_json FROM run_metadata LIMIT 1").fetchone()
        if (row is None or not row[0] or
                json.loads(config_bytes) != json.loads(str(row[0]))):
            raise OrbitHistoryProductError("Effective config differs from review-store provenance.")
        if _file_sha256(config) != source_config_sha256:
            raise OrbitHistoryProductError("Completed-run evidence changed during orbit-history export.")
    destination_input = Path(output_dir).expanduser()
    if destination_input.is_symlink():
        raise OrbitHistoryProductError("Orbit-history destination must not be a symbolic link.")
    destination = destination_input.resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise OrbitHistoryProductError("Orbit-history destination must be absent and outside the completed run.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        payload = power_history_to_dict(history)
        history_path = staging / "normalized_history.json"
        history_path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        manifest = {
            "schema_version": ORBIT_HISTORY_PRODUCT_SCHEMA,
            "status": "verified",
            "asset_id": history.object_id,
            "epoch_jd_utc": history.initial_jd_utc,
            "frame": "eci",
            "sample_count": int(history.times_s.size),
            "history_semantic_sha256": orbit_history_semantic_sha256(history),
            "history_file_sha256": _file_sha256(history_path),
            "source_review_sha256": source_review_sha256,
            "source_config_sha256": source_config_sha256,
        }
        (staging / "orbit_history_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        verify_orbit_history_product(staging)
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_orbit_history_product(destination)[0]


__all__ = [
    "ORBIT_HISTORY_PRODUCT_SCHEMA", "OrbitHistoryProductError",
    "export_orbit_history_product", "orbit_history_semantic_sha256",
    "verify_orbit_history_product",
]
