"""Explicit native analysis kernels; Python owns resources and workflow policy."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

import numpy as np

from sim.dynamics.orbit.epoch import sun_position_eci_km_enhanced, sun_position_eci_km_simple
from sim.dynamics.orbit.frames import _load_nut80_table


def numeric_backend(value: str) -> str:
    if value not in ("python", "rust"):
        raise ValueError("numeric_backend must be 'python' or 'rust'.")
    return value


def require_native(*symbols: str):
    try:
        native = import_module("oel_rust_orbit")
    except ImportError as exc:
        raise ValueError("Rust analysis requested; install a compatible oel_rust_orbit wheel.") from exc
    for symbol in symbols:
        owner = native
        for component in symbol.split("."):
            owner = getattr(owner, component, None)
        if not callable(owner):
            raise ValueError(f"Rust analysis requires native symbol {symbol!r}; upgrade oel_rust_orbit.")
    return native


class AnalyticSunEvaluator:
    """Copy resource tables once; retain at most 128 exact epochs per assessment.

    Returned arrays are copies. No state, callbacks, or cross-assessment resource
    identity is cached. Python chooses the same analytic model and NUT80 table.
    """

    def __init__(self, backend: str, model: str) -> None:
        numeric_backend(backend)
        if model not in ("analytic_simple", "analytic_enhanced"):
            raise ValueError("Analytic Sun model must be analytic_simple or analytic_enhanced.")
        if backend == "rust":
            native = require_native("ONPEnvironmentContext.analytic_sun_km")
            coefficients, terms = _load_nut80_table()
            context = native.ONPEnvironmentContext(coefficients.reshape(-1).tolist(), terms.reshape(-1).tolist())
            def evaluate(jd):
                return np.asarray(context.analytic_sun_km(jd, model == "analytic_enhanced"), dtype=float)
        else:
            evaluate = sun_position_eci_km_enhanced if model == "analytic_enhanced" else sun_position_eci_km_simple
        self._evaluate = lru_cache(maxsize=128)(evaluate)

    def at_jd(self, jd_utc: float) -> np.ndarray:
        jd = float(jd_utc)
        if not np.isfinite(jd):
            raise ValueError("Sun epoch must be finite.")
        result = self._evaluate(jd)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError("Analytic Sun returned an invalid ECI position.")
        return result.copy()


class OpticalPointingEvaluator:
    def __init__(self) -> None:
        self._native = require_native("optical_pointing", "optical_gimbal_vector")

    def pointing(self, state, target, *, pointing_mode):
        values = self._native.optical_pointing(
            np.asarray(state, dtype=float).tolist(), np.asarray(target, dtype=float).tolist(), pointing_mode
        )
        return np.asarray(values[0], dtype=float).reshape(3, 3), np.asarray(values[1], dtype=float), float(values[2])

    def gimbal(self, state, target, *, pointing_mode):
        return np.asarray(self._native.optical_gimbal_vector(
            np.asarray(state, dtype=float).tolist(), np.asarray(target, dtype=float).tolist(), pointing_mode
        ), dtype=float)
