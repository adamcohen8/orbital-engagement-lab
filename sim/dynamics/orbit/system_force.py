"""Stage-synchronized ONP integration for reciprocal two-object forces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from sim.dynamics.orbit.accelerations import accel_two_body


@dataclass(frozen=True)
class SystemForceContext:
    """One time-aligned ECI RK stage; arrays are immutable copies."""

    time_s: float
    epoch_jd_utc: float
    states_eci_km_km_s: Mapping[str, np.ndarray]
    masses_kg: Mapping[str, float]


def _immutable_state(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(6)
    return np.frombuffer(array.tobytes(), dtype=float)


def propagate_pair_rk4(
    *,
    states: Mapping[str, np.ndarray],
    masses_kg: Mapping[str, float],
    mu_km3_s2: Mapping[str, float],
    models: list[tuple[str, Callable[[SystemForceContext], Any]]],
    initial_jd_utc: float,
    t_s: float,
    t_next: float,
    substep_s: float,
    numeric_backend: str = "rust",
) -> dict[str, np.ndarray]:
    """Propagate both bodies from the same derivative stage at each RK4 call."""

    ids = tuple(states)
    if len(ids) != 2 or set(masses_kg) != set(ids) or set(mu_km3_s2) != set(ids):
        raise ValueError("joint RK4 requires two states with matching mass and gravity keys")
    if not np.isfinite(substep_s) or substep_s <= 0:
        raise ValueError("joint RK4 substep must be positive and finite")
    if numeric_backend not in {"python", "rust"}:
        raise ValueError("joint RK4 numeric backend must be python or rust")
    current_states = {oid: np.asarray(states[oid], dtype=float).reshape(6).copy() for oid in ids}
    masses = MappingProxyType({oid: float(masses_kg[oid]) for oid in ids})

    def derivative(stage_t_s: float, stage_states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        context = SystemForceContext(
            time_s=float(stage_t_s),
            epoch_jd_utc=float(initial_jd_utc) + float(stage_t_s) / 86400.0,
            states_eci_km_km_s=MappingProxyType({
                oid: _immutable_state(stage_states[oid]) for oid in ids
            }),
            masses_kg=masses,
        )
        accelerations = {
            oid: accel_two_body(stage_states[oid][:3], float(mu_km3_s2[oid])) for oid in ids
        }
        for name, function in models:
            result = function(context)
            if not isinstance(result, Mapping) or set(result) != set(ids):
                raise ValueError(f"system force model {name} must return one acceleration for each object: {ids!r}")
            for oid in ids:
                vector = np.asarray(result[oid], dtype=float)
                if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                    raise ValueError(f"system force model {name} returned invalid ECI acceleration for {oid!r}")
                accelerations[oid] += vector
        return {oid: np.hstack((stage_states[oid][3:], accelerations[oid])) for oid in ids}

    current = float(t_s)
    while current < float(t_next) - 1.0e-12:
        h = min(float(substep_s), float(t_next) - current)
        if numeric_backend == "rust":
            from sim.rust_orbit_backend import rk4_pair_callback_eci

            def pair_derivative(stage_time: float, stage_flat: list[float]) -> list[float]:
                stage_states = {
                    oid: np.asarray(stage_flat[index * 6:(index + 1) * 6], dtype=float)
                    for index, oid in enumerate(ids)
                }
                slopes = derivative(stage_time, stage_states)
                return np.hstack([slopes[oid] for oid in ids]).tolist()

            flat = np.hstack([current_states[oid] for oid in ids])
            next_flat = rk4_pair_callback_eci(flat, current, h, pair_derivative)
            current_states = {
                oid: next_flat[index * 6:(index + 1) * 6].copy()
                for index, oid in enumerate(ids)
            }
        else:
            k1 = derivative(current, current_states)
            k2 = derivative(current + h / 2, {oid: current_states[oid] + h * k1[oid] / 2 for oid in ids})
            k3 = derivative(current + h / 2, {oid: current_states[oid] + h * k2[oid] / 2 for oid in ids})
            k4 = derivative(current + h, {oid: current_states[oid] + h * k3[oid] for oid in ids})
            current_states = {
                oid: current_states[oid] + h * (k1[oid] + 2 * k2[oid] + 2 * k3[oid] + k4[oid]) / 6
                for oid in ids
            }
        current += h
    return current_states
