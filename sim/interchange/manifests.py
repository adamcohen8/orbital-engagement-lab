from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .provenance import compute_manifest_id


def finalize_handoff_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(document)
    manifest["manifest_id"] = "oel.handoff_manifest:" + "0" * 64
    manifest["manifest_id"] = compute_manifest_id(manifest)
    return manifest


def write_handoff_manifest(document: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(dict(document), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(target)
    return target


__all__ = ["finalize_handoff_manifest", "write_handoff_manifest"]
