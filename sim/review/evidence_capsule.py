"""Content-bound compression and temporary hydration for derived evidence files."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

EVIDENCE_CAPSULE_SCHEMA = "oel.evidence_capsule.v1"
EVIDENCE_CAPSULE_MANIFEST = "evidence_capsule.json"
_CHUNK_BYTES = 1024 * 1024


class EvidenceCapsuleError(ValueError):
    """Raised when compressed evidence cannot be trusted or hydrated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sqlite(path: Path) -> bool:
    return path.name.endswith(".sqlite")


def _sqlite_verification(path: Path) -> dict[str, Any]:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise EvidenceCapsuleError(f"SQLite quick_check failed for {path}: {quick_check!r}")
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        row_counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table.replace(chr(34), chr(34) * 2)}"').fetchone()[0])
            for table in tables
        }
        run_metadata: dict[str, Any] = {}
        if "run_metadata" in tables:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(run_metadata)").fetchall()
            }
            selected = [
                name
                for name in ("oel_version", "review_schema_version", "config_source_path", "config_sha256")
                if name in columns
            ]
            if selected:
                row = connection.execute(f"SELECT {', '.join(selected)} FROM run_metadata LIMIT 1").fetchone()
                if row is not None:
                    run_metadata = dict(zip(selected, row, strict=True))
    return {"quick_check": "ok", "row_counts": row_counts, "run_metadata": run_metadata}


def _manifest_path(logical_path: Path) -> Path:
    return logical_path.parent / EVIDENCE_CAPSULE_MANIFEST


def compressed_evidence_path(logical_path: str | Path) -> Path:
    path = Path(logical_path)
    return path.with_name(f"{path.name}.gz")


def _load_manifest(logical_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = _manifest_path(logical_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceCapsuleError(f"Evidence capsule manifest is unreadable: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != EVIDENCE_CAPSULE_SCHEMA:
        raise EvidenceCapsuleError(f"Evidence capsule manifest has an unsupported schema: {manifest_path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise EvidenceCapsuleError(f"Evidence capsule manifest has no artifact list: {manifest_path}")
    matches = [item for item in artifacts if isinstance(item, dict) and item.get("logical_path") == logical_path.name]
    if len(matches) != 1:
        raise EvidenceCapsuleError(
            f"Evidence capsule manifest must contain exactly one entry for {logical_path.name}: {manifest_path}"
        )
    entry = dict(matches[0])
    compressed_name = str(entry.get("compressed_path", "") or "")
    candidate = Path(compressed_name)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != compressed_name:
        raise EvidenceCapsuleError(f"Unsafe compressed evidence path in {manifest_path}: {compressed_name!r}")
    return manifest_path, entry


def _compressed_entry(logical_path: Path, *, verify_compressed: bool) -> tuple[Path, dict[str, Any]]:
    _manifest, entry = _load_manifest(logical_path)
    compressed = logical_path.parent / str(entry["compressed_path"])
    if not compressed.is_file() or compressed.is_symlink():
        raise EvidenceCapsuleError(f"Compressed evidence is missing or unsafe: {compressed}")
    if int(entry.get("compressed_bytes", -1)) != compressed.stat().st_size:
        raise EvidenceCapsuleError(f"Compressed evidence size does not match its manifest: {compressed}")
    if verify_compressed and _sha256(compressed) != str(entry.get("compressed_sha256", "")):
        raise EvidenceCapsuleError(f"Compressed evidence digest does not match its manifest: {compressed}")
    return compressed, entry


def evidence_file_exists(logical_path: str | Path) -> bool:
    path = Path(logical_path)
    if path.is_file() and not path.is_symlink():
        return True
    try:
        _compressed_entry(path, verify_compressed=False)
    except EvidenceCapsuleError:
        return False
    return True


def evidence_file_sha256(logical_path: str | Path) -> str:
    path = Path(logical_path)
    if path.is_file() and not path.is_symlink():
        return _sha256(path)
    _compressed, entry = _compressed_entry(path, verify_compressed=True)
    digest = str(entry.get("original_sha256", "") or "")
    if len(digest) != 64:
        raise EvidenceCapsuleError(f"Evidence capsule has an invalid original digest: {_manifest_path(path)}")
    materialized = materialize_evidence(path, prefer_capsule=True)
    try:
        if _sha256(materialized.path) != digest:
            raise EvidenceCapsuleError(f"Evidence capsule original digest verification failed: {path}")
    finally:
        materialized.close()
    return digest


def evidence_file_mtime_ns(logical_path: str | Path) -> int:
    path = Path(logical_path)
    if path.is_file() and not path.is_symlink():
        return path.stat().st_mtime_ns
    _compressed, entry = _compressed_entry(path, verify_compressed=True)
    value = entry.get("original_mtime_ns")
    if not isinstance(value, int) or value < 0:
        raise EvidenceCapsuleError(f"Evidence capsule has no valid original mtime: {_manifest_path(path)}")
    return value


@dataclass
class MaterializedEvidence:
    logical_path: Path
    path: Path
    _temporary: tempfile.TemporaryDirectory[str] | None = None

    @property
    def hydrated(self) -> bool:
        return self._temporary is not None

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def materialize_evidence(
    logical_path: str | Path, *, prefer_capsule: bool = False
) -> MaterializedEvidence:
    path = Path(logical_path).expanduser().resolve()
    if path.is_file() and not path.is_symlink() and not prefer_capsule:
        return MaterializedEvidence(logical_path=path, path=path)
    compressed, entry = _compressed_entry(path, verify_compressed=True)
    temporary = tempfile.TemporaryDirectory(prefix="oel-evidence-")
    hydrated = Path(temporary.name) / path.name
    try:
        with gzip.open(compressed, "rb") as source, hydrated.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=_CHUNK_BYTES)
        if hydrated.stat().st_size != int(entry.get("original_bytes", -1)):
            raise EvidenceCapsuleError(f"Hydrated evidence size does not match its manifest: {path}")
        if _sha256(hydrated) != str(entry.get("original_sha256", "")):
            raise EvidenceCapsuleError(f"Hydrated evidence digest does not match its manifest: {path}")
        if str(entry.get("kind")) == "sqlite":
            verification = _sqlite_verification(hydrated)
            expected = dict(entry.get("verification", {}) or {})
            if isinstance(expected.get("queries"), list):
                query_results: list[dict[str, Any]] = []
                uri = f"{hydrated.resolve().as_uri()}?mode=ro"
                with sqlite3.connect(uri, uri=True) as connection:
                    connection.execute("PRAGMA query_only = ON")
                    for item in expected["queries"]:
                        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
                            raise EvidenceCapsuleError(f"Invalid verification query in capsule manifest: {path}")
                        query_results.append(
                            {
                                "query": item["query"],
                                "rows": [list(row) for row in connection.execute(item["query"]).fetchall()],
                            }
                        )
                verification["queries"] = query_results
            if verification != expected:
                raise EvidenceCapsuleError(f"Hydrated SQLite verification does not match its manifest: {path}")
    except Exception:
        temporary.cleanup()
        raise
    return MaterializedEvidence(logical_path=path, path=hydrated, _temporary=temporary)


@contextmanager
def materialized_evidence_file(logical_path: str | Path) -> Iterator[Path]:
    materialized = materialize_evidence(logical_path)
    try:
        yield materialized.path
    finally:
        materialized.close()


def create_evidence_capsule(
    logical_path: str | Path,
    *,
    remove_original: bool = False,
    compression_level: int = 1,
    verification_queries: Sequence[str] = (),
    provenance_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Create and verify a gzip capsule; optionally remove the exact source file."""

    path = Path(logical_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise EvidenceCapsuleError(f"Evidence source is missing or unsafe: {path}")
    if path.name.endswith(("-wal", "-shm")):
        raise EvidenceCapsuleError(f"SQLite sidecars cannot be capsule sources: {path}")
    if _is_sqlite(path):
        for suffix in ("-wal", "-shm"):
            if path.with_name(path.name + suffix).exists():
                raise EvidenceCapsuleError(f"SQLite evidence has a WAL/SHM sidecar: {path.name + suffix}")
    if not 0 <= int(compression_level) <= 9:
        raise EvidenceCapsuleError("gzip compression level must be between 0 and 9")

    manifest_path = _manifest_path(path)
    if manifest_path.is_file():
        try:
            existing_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceCapsuleError(f"Refusing to replace an unreadable manifest: {manifest_path}") from exc
        if not isinstance(existing_payload, dict) or existing_payload.get("schema") != EVIDENCE_CAPSULE_SCHEMA:
            raise EvidenceCapsuleError(f"Refusing to replace an unrelated manifest: {manifest_path}")
    else:
        existing_payload = None

    original_sha256 = _sha256(path)
    verification = _sqlite_verification(path) if _is_sqlite(path) else {"byte_round_trip": True}
    query_results: list[dict[str, Any]] = []
    if verification_queries:
        if not _is_sqlite(path):
            raise EvidenceCapsuleError("verification queries are supported only for SQLite evidence")
        uri = f"{path.as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only = ON")
            for query in verification_queries:
                normalized = str(query).strip().lower()
                if not normalized.startswith(("select ", "with ")) or ";" in str(query).rstrip(";"):
                    raise EvidenceCapsuleError("verification queries must be one read-only SELECT or WITH statement")
                query_results.append(
                    {"query": str(query), "rows": [list(row) for row in connection.execute(str(query)).fetchall()]}
                )
        verification["queries"] = query_results
    provenance: list[dict[str, Any]] = []
    for raw_provenance_path in provenance_paths:
        provenance_path = Path(raw_provenance_path).expanduser().resolve()
        if not provenance_path.is_file() or provenance_path.is_symlink():
            raise EvidenceCapsuleError(f"Evidence provenance input is missing or unsafe: {provenance_path}")
        provenance.append(
            {
                "path": provenance_path.as_posix(),
                "bytes": provenance_path.stat().st_size,
                "sha256": _sha256(provenance_path),
            }
        )

    compressed = compressed_evidence_path(path)
    staged = compressed.with_name(f".{compressed.name}.tmp-{os.getpid()}")
    with path.open("rb") as source, staged.open("wb") as raw_target:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_target, compresslevel=int(compression_level), mtime=0
        ) as target:
            shutil.copyfileobj(source, target, length=_CHUNK_BYTES)

    generated_utc = _now_utc()
    entry = {
        "logical_path": path.name,
        "compressed_path": compressed.name,
        "kind": "sqlite" if _is_sqlite(path) else "file",
        "compression": {"algorithm": "gzip", "level": int(compression_level)},
        "original_bytes": path.stat().st_size,
        "original_sha256": original_sha256,
        "original_mtime_ns": path.stat().st_mtime_ns,
        "compressed_bytes": staged.stat().st_size,
        "compressed_sha256": _sha256(staged),
        "verified_utc": generated_utc,
        "verification": verification,
        "provenance": provenance,
        "restore_command": f"python tools/evidence_capsules.py restore {path.as_posix()}",
    }
    staged.replace(compressed)

    if existing_payload is not None:
        payload = existing_payload
        artifacts = [
            item for item in list(payload.get("artifacts", []) or [])
            if isinstance(item, dict) and item.get("logical_path") != path.name
        ]
    else:
        artifacts = []
    artifacts.append(entry)
    payload = {"schema": EVIDENCE_CAPSULE_SCHEMA, "generated_utc": generated_utc, "artifacts": artifacts}
    staged_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    staged_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staged_manifest.replace(manifest_path)

    materialized = materialize_evidence(path, prefer_capsule=True)
    try:
        hydrated = materialized.path
        if _sha256(hydrated) != original_sha256:
            raise EvidenceCapsuleError(f"Post-commit capsule verification failed: {path}")
    finally:
        materialized.close()
    if remove_original:
        path.unlink()
    return entry


def restore_evidence_capsule(logical_path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(logical_path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise EvidenceCapsuleError(f"Evidence destination already exists: {path}")
    _compressed, entry = _compressed_entry(path, verify_compressed=True)
    expected_sha256 = str(entry.get("original_sha256", ""))
    backup = path.with_name(f".{path.name}.backup-{os.getpid()}")
    if path.exists():
        path.replace(backup)
    try:
        materialized = materialize_evidence(path, prefer_capsule=True)
        try:
            staged = path.with_name(f".{path.name}.restore-{os.getpid()}")
            shutil.copyfile(materialized.path, staged)
            staged.replace(path)
        finally:
            materialized.close()
        if _sha256(path) != expected_sha256:
            raise EvidenceCapsuleError(f"Restored evidence digest check failed: {path}")
    except Exception:
        path.unlink(missing_ok=True)
        if backup.exists():
            backup.replace(path)
        raise
    backup.unlink(missing_ok=True)
    if _sha256(path) != expected_sha256:
        raise EvidenceCapsuleError(f"Restored evidence digest check failed: {path}")
    return path
