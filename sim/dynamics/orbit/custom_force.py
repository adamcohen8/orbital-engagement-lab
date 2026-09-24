"""Adapter for opt-in, object-scoped ONP acceleration models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True, eq=False)
class CustomForceModel:
    """Adapt a user model to ONP's internal stage callback.

    The user callable receives ``(state_eci_km_km_s, epoch_jd_utc)`` and
    returns an ECI acceleration in km/s². The six state components are position
    in km followed by velocity in km/s. Each stage receives its own read-only
    state snapshot and absolute UTC epoch.
    """

    acceleration: Callable[[np.ndarray, float], Any]
    initial_jd_utc: float
    name: str

    def __call__(self, t_s: float, state: np.ndarray, env: dict, ctx: Any) -> np.ndarray:
        state_snapshot = np.array(state, dtype=float, copy=True)
        state_snapshot.setflags(write=False)
        epoch_jd_utc = self.initial_jd_utc + float(t_s) / 86400.0
        result = self.acceleration(state_snapshot, epoch_jd_utc)
        try:
            vector = np.asarray(result, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"force model {self.name} must return a numeric ECI acceleration in km/s²") from exc
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"force model {self.name} must return three finite ECI acceleration components in km/s²")
        return vector
