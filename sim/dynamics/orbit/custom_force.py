"""Adapter for opt-in, object-scoped ONP acceleration models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from inspect import signature
from types import MappingProxyType
from typing import Any, Callable

import numpy as np


def _read_only_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


@dataclass(frozen=True)
class ForceObjectSnapshot:
    """Committed truth supplied to an object force as a time-labelled value."""

    state_eci_km_km_s: np.ndarray
    attitude_quat_bn: np.ndarray
    angular_rate_body_rad_s: np.ndarray
    mass_kg: float
    time_s: float


@dataclass(frozen=True)
class ForceEvaluationContext:
    object_id: str
    other_objects: Mapping[str, ForceObjectSnapshot]
    snapshot_basis: str = "start_of_step"


@dataclass(frozen=True, eq=False)
class CustomForceModel:
    """Adapt a user model to ONP's internal stage callback.

    The user callable receives ``(state_eci_km_km_s, epoch_jd_utc)`` and
    returns an ECI acceleration in km/s². The six state components are position
    in km followed by velocity in km/s. Each stage receives its own read-only
    state snapshot and absolute UTC epoch.
    """

    acceleration: Callable[..., Any]
    initial_jd_utc: float
    name: str

    def __post_init__(self) -> None:
        try:
            signature(self.acceleration).bind(None, None, None)
            supports_context = True
        except (TypeError, ValueError):
            supports_context = False
        object.__setattr__(self, "_supports_context", supports_context)

    def __call__(self, t_s: float, state: np.ndarray, env: dict, ctx: Any) -> np.ndarray:
        state_snapshot = _read_only_array(state)
        epoch_jd_utc = self.initial_jd_utc + float(t_s) / 86400.0
        if self._supports_context:
            other_objects = {}
            for object_id, truth in dict(env.get("world_truth", {}) or {}).items():
                if object_id == env.get("object_id"):
                    continue
                other_objects[str(object_id)] = ForceObjectSnapshot(
                    state_eci_km_km_s=_read_only_array(np.hstack((truth.position_eci_km, truth.velocity_eci_km_s))),
                    attitude_quat_bn=_read_only_array(truth.attitude_quat_bn),
                    angular_rate_body_rad_s=_read_only_array(truth.angular_rate_body_rad_s),
                    mass_kg=float(truth.mass_kg),
                    time_s=float(truth.t_s),
                )
            context = ForceEvaluationContext(
                object_id=str(env.get("object_id", "")),
                other_objects=MappingProxyType(other_objects),
            )
            result = self.acceleration(state_snapshot, epoch_jd_utc, context)
        else:
            result = self.acceleration(state_snapshot, epoch_jd_utc)
        try:
            vector = np.asarray(result, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"force model {self.name} must return a numeric ECI acceleration in km/s²") from exc
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"force model {self.name} must return three finite ECI acceleration components in km/s²")
        return vector
