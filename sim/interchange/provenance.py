from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import HANDOFF_MANIFEST_SCHEMA_ID

_NON_IDENTITY_KEYS = {
    "created_utc",
    "evaluated_utc",
    "later_verification_results",
    "manifest_id",
    "presentation_description",
    "product_id",
    "verification_results",
}


def _canonical_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string dictionary key")
            result[key] = _canonical_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON deterministically for identity and evidence."""

    normalized = _canonical_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_nonidentity_fields(value: Any, *, parent_path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if (
                key_text in _NON_IDENTITY_KEYS
                or _is_nonidentity_location(parent_path, key_text, value)
            ):
                continue
            result[key_text] = _without_nonidentity_fields(
                item,
                parent_path=(*parent_path, key_text),
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _without_nonidentity_fields(item, parent_path=(*parent_path, "[]"))
            for item in value
        ]
    return value


def _is_nonidentity_location(
    parent_path: tuple[str, ...],
    key: str,
    container: Mapping[str, Any],
) -> bool:
    """Identify presentation/provenance locations without erasing semantic paths.

    Scenario-patch operation ``path`` values select the configuration field to
    mutate and therefore participate in product identity. Other path-shaped
    fields in interchange envelopes describe where source or output artifacts
    happened to be stored and remain location-independent.
    """

    if key in {"parameter_path", "metric_path"} or (key == "path" and "metrics" in parent_path):
        return False
    is_patch_operation = (
        key == "path"
        and "operations" in parent_path
        and {"op", "kind", "value"}.issubset(container)
    )
    if is_patch_operation:
        return False
    filesystem_keys = {
        "source_path", "output_path", "scenario_path", "manifest_path",
        "packet_path", "database_path", "review_db_path", "config_path",
        "summary_json_path", "run_log_json_path", "output_dir", "output_directory",
    }
    if key in filesystem_keys or key.endswith("_file_path"):
        return True
    return key == "path" and any(
        token in parent_path
        for token in ("source", "sources", "output", "artifact", "artifacts", "provenance", "materialization")
    )


def product_identity_document(document: Mapping[str, Any]) -> dict[str, Any]:
    identity = _without_nonidentity_fields(document)
    identity.pop("product_id", None)
    identity.pop("created_utc", None)
    freshness = identity.get("freshness")
    if isinstance(freshness, dict):
        freshness.pop("evaluated_utc", None)
    return identity


def compute_product_id(document: Mapping[str, Any]) -> str:
    kind = str(document.get("product_kind", "") or "")
    if not kind:
        raise ValueError("product_kind is required before product identity can be computed")
    digest = hashlib.sha256(canonical_json_bytes(product_identity_document(document))).hexdigest()
    return f"{kind}:{digest}"


def manifest_identity_document(document: Mapping[str, Any]) -> dict[str, Any]:
    identity = _without_nonidentity_fields(document)
    identity.pop("manifest_id", None)
    identity.pop("created_utc", None)
    identity.pop("warnings", None)
    identity.pop("failures", None)
    identity.pop("recommended_next_action", None)
    identity.pop("execution_occurred", None)
    validation = identity.get("validation")
    if isinstance(validation, dict):
        validation.pop("safe_validation_result", None)
        validation.pop("ordinary_validation_result", None)
        validation.pop("validated_utc", None)
    return identity


def compute_manifest_id(document: Mapping[str, Any]) -> str:
    if document.get("schema_id") != HANDOFF_MANIFEST_SCHEMA_ID:
        raise ValueError(f"manifest schema_id must be {HANDOFF_MANIFEST_SCHEMA_ID!r}")
    digest = hashlib.sha256(canonical_json_bytes(manifest_identity_document(document))).hexdigest()
    return f"oel.handoff_manifest:{digest}"
