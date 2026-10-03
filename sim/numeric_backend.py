"""Shared strict selection of the native and reference numeric implementations."""
from __future__ import annotations

from typing import Any


def normalize_numeric_backend(value: Any, *, field_name: str = "numeric_backend", error_message: str | None = None) -> str:
    message = error_message or f"{field_name} must be python or rust."
    if value is None:
        return "rust"
    if not isinstance(value, str):
        raise ValueError(message)
    backend = value.strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(message)
    return backend
