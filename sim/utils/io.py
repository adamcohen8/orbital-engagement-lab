from __future__ import annotations

import hashlib
import math
from json.encoder import encode_basestring_ascii
from numbers import Integral, Real
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = value
        if scalar is not value:
            return json_safe(scalar)
    return value


def write_json(path: str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            # Preserve the historical json_safe + json.dump byte contract while
            # sanitizing values as the encoder visits them. Large run logs no
            # longer require a second full-size Python list/dict tree.
            for chunk in _iter_json_safe(payload, indent=2):
                handle.write(chunk)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without allocating a full-size bytes object."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_json_safe(value: Any, *, indent: int | str | None = None):
    """Yield the exact encoding of ``json.dump(json_safe(value))`` lazily."""

    indent_text = None if indent is None else indent if isinstance(indent, str) else " " * int(indent)
    markers: dict[int, Any] = {}

    def scalar(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int, float, list, tuple, dict)):
            return item
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Integral):
            return int(item)
        if isinstance(item, Real):
            numeric = float(item)
            return numeric if math.isfinite(numeric) else None
        convert = getattr(item, "item", None)
        if callable(convert):
            try:
                converted = convert()
            except Exception:
                converted = item
            if converted is not item:
                return scalar(converted)
        return item

    def float_text(item: float) -> str:
        return float.__repr__(item) if math.isfinite(item) else "null"

    def encode(item: Any, level: int):
        item = scalar(item)
        if isinstance(item, str):
            yield encode_basestring_ascii(item)
        elif item is None:
            yield "null"
        elif item is True:
            yield "true"
        elif item is False:
            yield "false"
        elif isinstance(item, int):
            yield int.__repr__(item)
        elif isinstance(item, float):
            yield float_text(item)
        elif isinstance(item, (list, tuple)):
            marker = id(item)
            if marker in markers:
                raise ValueError("Circular reference detected")
            markers[marker] = item
            try:
                if not item:
                    yield "[]"
                    return
                yield "["
                child_level = level + 1
                separator = "," if indent_text is None else ",\n" + indent_text * child_level
                if indent_text is not None:
                    yield "\n" + indent_text * child_level
                for index, child in enumerate(item):
                    if index:
                        yield separator
                    yield from encode(child, child_level)
                if indent_text is not None:
                    yield "\n" + indent_text * level
                yield "]"
            finally:
                markers.pop(marker, None)
        elif isinstance(item, dict):
            marker = id(item)
            if marker in markers:
                raise ValueError("Circular reference detected")
            markers[marker] = item
            try:
                # json_safe historically constructs a new dict with string
                # keys. Preserve its last-value-wins collision behavior while
                # retaining the first key's insertion position.
                normalized_items: dict[str, Any] = {}
                for key, child in item.items():
                    normalized_items[str(key)] = child
                if not normalized_items:
                    yield "{}"
                    return
                yield "{"
                child_level = level + 1
                separator = "," if indent_text is None else ",\n" + indent_text * child_level
                if indent_text is not None:
                    yield "\n" + indent_text * child_level
                for index, (key, child) in enumerate(normalized_items.items()):
                    if index:
                        yield separator
                    yield encode_basestring_ascii(key)
                    yield ": " if indent_text is not None else ":"
                    yield from encode(child, child_level)
                if indent_text is not None:
                    yield "\n" + indent_text * level
                yield "}"
            finally:
                markers.pop(marker, None)
        else:
            raise TypeError(f"Object of type {item.__class__.__name__} is not JSON serializable")

    yield from encode(value, 0)
