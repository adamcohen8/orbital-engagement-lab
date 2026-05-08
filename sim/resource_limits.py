from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class SimulationMemoryBudgetError(MemoryError):
    """Raised before a run allocates more history memory than the active budget allows."""


@dataclass(frozen=True)
class HistoryMemoryEstimate:
    samples: int
    active_objects: int
    knowledge_pairs: int
    array_bytes: int
    estimated_peak_bytes: int
    limit_bytes: int

    @property
    def estimated_peak_mb(self) -> float:
        return float(self.estimated_peak_bytes) / (1024.0 * 1024.0)

    @property
    def limit_mb(self) -> float:
        return float(self.limit_bytes) / (1024.0 * 1024.0)


DEFAULT_MAX_HISTORY_MEMORY_MB = 1024.0
_ENV_MAX_HISTORY_MEMORY_MB = "OEL_MAX_HISTORY_MEMORY_MB"


def _positive_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    out = float(value)
    if out <= 0:
        raise ValueError("max history memory must be positive.")
    return out


def configured_history_memory_limit_mb(cfg: Any) -> float:
    external_limit = _positive_float_or_none(os.environ.get(_ENV_MAX_HISTORY_MEMORY_MB))
    if external_limit is None:
        external_limit = DEFAULT_MAX_HISTORY_MEMORY_MB

    resource_limits = dict(getattr(getattr(cfg, "outputs", None), "resource_limits", {}) or {})
    config_limit = _positive_float_or_none(resource_limits.get("max_history_memory_mb"))
    if config_limit is None:
        return float(external_limit)
    return float(min(external_limit, config_limit))


def bytes_from_mb(value_mb: float) -> int:
    return int(float(value_mb) * 1024.0 * 1024.0)


def format_bytes_mb(value: int) -> str:
    return f"{float(value) / (1024.0 * 1024.0):.2f} MB"


def enforce_history_memory_budget(estimate: HistoryMemoryEstimate) -> None:
    if estimate.estimated_peak_bytes <= estimate.limit_bytes:
        return
    raise SimulationMemoryBudgetError(
        "Estimated simulation history memory exceeds the active budget: "
        f"estimated_peak={format_bytes_mb(estimate.estimated_peak_bytes)}, "
        f"limit={format_bytes_mb(estimate.limit_bytes)}, "
        f"samples={estimate.samples}, active_objects={estimate.active_objects}, "
        f"knowledge_pairs={estimate.knowledge_pairs}. "
        f"Raise the caller-controlled cap with {_ENV_MAX_HISTORY_MEMORY_MB} or "
        "--max-history-memory-mb, or reduce duration/dt/object count."
    )
