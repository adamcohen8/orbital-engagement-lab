from __future__ import annotations

import json
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
    target.write_text(json.dumps(dict(document), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


__all__ = ["finalize_handoff_manifest", "write_handoff_manifest"]
