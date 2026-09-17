"""Small dependency-free validator for OEL study contract schemas.

The supported subset intentionally matches the closed schemas in this package.
It is not a general JSON Schema implementation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SchemaIssue:
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


def validate_schema(value: Any, schema: Mapping[str, Any], *, path: str = "$") -> tuple[SchemaIssue, ...]:
    issues: list[SchemaIssue] = []
    _validate(value, schema, path=path, issues=issues)
    return tuple(issues)


def _validate(value: Any, schema: Mapping[str, Any], *, path: str, issues: list[SchemaIssue]) -> None:
    if "const" in schema and value != schema["const"]:
        issues.append(SchemaIssue(path, f"must equal {schema['const']!r}"))
        return
    if "enum" in schema and value not in schema["enum"]:
        issues.append(SchemaIssue(path, f"must be one of {list(schema['enum'])!r}"))
        return

    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_matches_type(value, item) for item in expected):
            issues.append(SchemaIssue(path, f"must have one of the types {expected!r}"))
            return
    elif isinstance(expected, str) and not _matches_type(value, expected):
        issues.append(SchemaIssue(path, f"must be of type {expected}"))
        return

    if value is None:
        return
    if isinstance(value, float) and not math.isfinite(value):
        issues.append(SchemaIssue(path, "must be finite"))
        return

    if isinstance(value, Mapping):
        properties = dict(schema.get("properties", {}) or {})
        required = tuple(schema.get("required", ()) or ())
        for key in required:
            if key not in value:
                issues.append(SchemaIssue(f"{path}.{key}", "is required"))
        if schema.get("additionalProperties") is False:
            for key in sorted(set(value) - set(properties)):
                issues.append(SchemaIssue(f"{path}.{key}", "is not an allowed field"))
        for key, item in value.items():
            child = properties.get(key)
            if child is not None:
                _validate(item, child, path=f"{path}.{key}", issues=issues)

    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < int(minimum):
            issues.append(SchemaIssue(path, f"must contain at least {minimum} item(s)"))
        if maximum is not None and len(value) > int(maximum):
            issues.append(SchemaIssue(path, f"must contain no more than {maximum} item(s)"))
        if schema.get("uniqueItems") and len({_freeze(item) for item in value}) != len(value):
            issues.append(SchemaIssue(path, "must contain unique items"))
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate(item, item_schema, path=f"{path}[{index}]", issues=issues)

    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if minimum is not None and len(value) < int(minimum):
            issues.append(SchemaIssue(path, f"must contain at least {minimum} character(s)"))
        if maximum is not None and len(value) > int(maximum):
            issues.append(SchemaIssue(path, f"must contain no more than {maximum} character(s)"))
        pattern = schema.get("pattern")
        if pattern and re.fullmatch(str(pattern), value) is None:
            issues.append(SchemaIssue(path, "has an invalid format"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            issues.append(SchemaIssue(path, f"must be greater than or equal to {minimum}"))
        if maximum is not None and value > maximum:
            issues.append(SchemaIssue(path, f"must be less than or equal to {maximum}"))


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


__all__ = ["SchemaIssue", "validate_schema"]
