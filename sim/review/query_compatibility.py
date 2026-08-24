"""Compatibility routing for qualification queries across review schemas."""

from __future__ import annotations

import re
import sqlite3

_FIELD_NAME = re.compile(
    r"json_extract\(field\.value,\s*'\$\.name'\)\s*=\s*'([^']+)'", re.IGNORECASE
)
_OBJECT_ID = re.compile(r"d\.object_id\s*=\s*'([^']+)'", re.IGNORECASE)
_AFTER_TIME = re.compile(r"d\.generated_time_ns\s*>\s*([0-9]+)", re.IGNORECASE)
_TEXT_VALUE = re.compile(
    r"json_extract\(field\.value,\s*'\$\.value'\)\s*=\s*'([^']*)'", re.IGNORECASE
)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=? LIMIT 1", (table,)
    ).fetchone() is not None


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def compatible_review_query(connection: sqlite3.Connection, query: str) -> tuple[str, str]:
    """Prefer normalized fields while retaining read compatibility with v0.9 stores."""

    selected = str(query)
    has_normalized_diagnostics = _table_exists(connection, "fsw_diagnostic_fields")
    if has_normalized_diagnostics and _table_exists(connection, "actuator_command_receipts"):
        selected = re.sub(
            r"json_extract\(detail_json,\s*'\$\.disposition'\)",
            "disposition",
            selected,
            flags=re.IGNORECASE,
        )

    if not has_normalized_diagnostics or "json_each" not in selected.lower():
        return selected, "legacy" if selected == query else "normalized_receipt"
    field_match = _FIELD_NAME.search(selected)
    if field_match is None:
        return selected, "legacy"

    conditions = [f"field.field_name = {_literal(field_match.group(1))}"]
    object_match = _OBJECT_ID.search(selected)
    if object_match is not None:
        conditions.append(f"field.object_id = {_literal(object_match.group(1))}")
    time_match = _AFTER_TIME.search(selected)
    if time_match is not None:
        conditions.append(f"field.generated_time_ns > {int(time_match.group(1))}")

    lowered = selected.strip().lower()
    numeric_value = "COALESCE(field.value_real, CAST(field.value_integer AS REAL))"
    if lowered.startswith("select cast("):
        rewritten = (
            f"SELECT {numeric_value} FROM fsw_diagnostic_fields AS field "
            f"WHERE {' AND '.join(conditions)} ORDER BY field.generated_time_ns DESC LIMIT 1"
        )
        return rewritten, "normalized_diagnostic"
    if not lowered.startswith("select count(*)"):
        return selected, "legacy"

    text_match = _TEXT_VALUE.search(selected)
    if text_match is not None:
        conditions.append(f"field.value_text = {_literal(text_match.group(1))}")
    elif re.search(r"json_extract\(field\.value,\s*'\$\.value'\)\s*=\s*1", selected, re.IGNORECASE):
        conditions.append("field.value_integer = 1")
    elif re.search(
        r"cast\(json_extract\(field\.value,\s*'\$\.value'\)\s+as\s+integer\)\s*>\s*0",
        selected,
        re.IGNORECASE,
    ):
        conditions.append("COALESCE(field.value_integer, CAST(field.value_real AS INTEGER)) > 0")
    else:
        return selected, "legacy"
    return (
        "SELECT COUNT(*) FROM fsw_diagnostic_fields AS field WHERE " + " AND ".join(conditions),
        "normalized_diagnostic",
    )
