"""Public verified import of transport-neutral Hosted OEL result transfers."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping

from sim.study_planning import canonical_sha256

RESULT_TRANSFER_SCHEMA = "oel.hosted_result_transfer.v1"
RESULT_IMPORT_RECEIPT_SCHEMA = "oel.hosted_result_import_receipt.v1"


def import_result_transfer(
    transfer: Mapping[str, Any],
    destination: str | Path,
    *,
    read_file: Callable[[str, str], bytes],
) -> dict[str, Any]:
    """Import a declared transfer into a new directory and verify every byte."""

    document = dict(transfer)
    if document.get("schema") != RESULT_TRANSFER_SCHEMA:
        raise ValueError("Unsupported Hosted OEL result-transfer schema.")
    expected_transfer_sha256 = canonical_sha256(
        {key: value for key, value in document.items() if key != "transfer_sha256"}
    )
    if document.get("transfer_sha256") != expected_transfer_sha256:
        raise ValueError("Hosted OEL result transfer failed digest verification.")

    target = Path(destination).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Result-import destination must not already exist.")
    target.parent.mkdir(parents=True, exist_ok=True)

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Hosted OEL result transfer declares no artifacts.")

    imported: list[dict[str, Any]] = []
    seen_artifact_ids: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as temporary:
        temporary_root = Path(temporary) / "payload"
        temporary_root.mkdir()
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                raise ValueError("Hosted OEL result transfer contains an invalid artifact.")
            artifact_id = str(artifact.get("artifact_id", ""))
            if not artifact_id or artifact_id in seen_artifact_ids:
                raise ValueError("Hosted OEL result artifact ids must be non-empty and unique.")
            seen_artifact_ids.add(artifact_id)
            artifact_root = temporary_root / f"{index:03d}-{_safe_component(str(artifact.get('kind', 'artifact')))}"
            artifact_root.mkdir()
            files = artifact.get("files")
            if not isinstance(files, list) or not files:
                raise ValueError("Hosted OEL result artifact declares no files.")
            imported_files: list[dict[str, Any]] = []
            for relative_path, row in _validated_file_rows(files):
                payload = read_file(artifact_id, relative_path)
                if not isinstance(payload, bytes):
                    raise TypeError("Hosted result transport must return file bytes.")
                observed = hashlib.sha256(payload).hexdigest()
                if observed != row.get("content_sha256") or len(payload) != row.get("bytes"):
                    raise ValueError("Hosted result file did not match its transfer declaration.")
                output = artifact_root / PurePosixPath(relative_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as stream:
                    stream.write(payload)
                imported_files.append(
                    {
                        "relative_path": relative_path,
                        "bytes": len(payload),
                        "content_sha256": observed,
                    }
                )
            imported.append(
                {
                    "artifact_id": artifact_id,
                    "kind": str(artifact.get("kind", "")),
                    "local_directory": artifact_root.relative_to(temporary_root).as_posix(),
                    "files": imported_files,
                }
            )

        # Verify the materialized bytes, not only each transport response.
        for artifact in imported:
            for row in artifact["files"]:
                output = temporary_root / artifact["local_directory"] / row["relative_path"]
                payload = output.read_bytes()
                if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["content_sha256"]:
                    raise ValueError("Materialized Hosted result did not match its receipt.")

        receipt = {
            "schema": RESULT_IMPORT_RECEIPT_SCHEMA,
            "job_id": document["job_id"],
            "tenant_id": document["tenant_id"],
            "capsule_sha256": document["capsule_sha256"],
            "worker_image_sha256": document["worker_image_sha256"],
            "artifact_bundle_sha256": document["artifact_bundle_sha256"],
            "transfer_sha256": expected_transfer_sha256,
            "artifacts": imported,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        receipt_path = temporary_root / "result_import_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_root.rename(target)
    return receipt


def _validated_file_rows(files: list[Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Reject portable file and directory aliases before reading any payloads."""
    nodes: dict[tuple[str, ...], tuple[tuple[str, ...], bool]] = {}
    rows = []
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("Hosted OEL result transfer contains an invalid file row.")
        relative = _safe_relative_path(str(row.get("relative_path", "")))
        parts = PurePosixPath(relative).parts
        for length in range(1, len(parts) + 1):
            prefix = parts[:length]
            key = tuple(unicodedata.normalize("NFC", part).casefold() for part in prefix)
            is_file = length == len(parts)
            previous = nodes.get(key)
            if previous is not None and (previous[0] != prefix or previous[1] or is_file):
                raise ValueError("Hosted OEL result artifact contains a duplicate or aliasing file path.")
            nodes[key] = (prefix, is_file)
        rows.append((relative, row))
    return rows


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or ":" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
        or any(part.endswith((".", " ")) for part in path.parts)
    ):
        raise ValueError("Unsafe path in Hosted OEL result transfer.")
    return path.as_posix()


def _safe_component(value: str) -> str:
    text = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return text.strip("-")[:64] or "artifact"


__all__ = ["RESULT_IMPORT_RECEIPT_SCHEMA", "RESULT_TRANSFER_SCHEMA", "import_result_transfer"]
