from __future__ import annotations

import json
import math
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
            # Stream the encoder directly to disk.  Large run logs no longer
            # require a second, full-size Unicode string (and its encoded byte
            # buffer) in addition to the sanitized payload already in memory.
            json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
