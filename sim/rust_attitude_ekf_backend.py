"""Explicit optional Euler EKF arithmetic; estimator policy remains in Python."""

from functools import lru_cache

import numpy as np

from sim.rust_orbit_backend import _extension


def _factory():
    cls = getattr(_extension(), "AttitudeEKFContext", None)
    if cls is None:
        raise RuntimeError("Rust attitude EKF requires an oel_rust_orbit wheel with AttitudeEKFContext")
    return cls


@lru_cache(maxsize=8)
def _context(factory, inertia: bytes):
    try:
        return factory(inertia)
    except ValueError as exc:
        if str(exc) == "Singular matrix":
            raise np.linalg.LinAlgError("Singular matrix") from exc
        raise


def predict(state, inertia, dt_s, *, base=None):
    # Resolve availability on every crossing and content-bind mutable inertia.
    context = _context(_factory(), np.asarray(inertia, dtype="<f8").reshape(3, 3).tobytes())
    raw = context.predict(
        np.asarray(state, dtype="<f8").reshape(7).tobytes(),
        float(dt_s),
        None if base is None else np.asarray(base, dtype="<f8").reshape(7).tobytes(),
    )
    values = np.frombuffer(raw, dtype="<f8").copy()
    if values.size != 56:
        raise RuntimeError("Rust attitude EKF returned an invalid prediction/Jacobian")
    return values[:7], values[7:].reshape(7, 7)


def propagate(state, inertia, dt_s):
    context = _context(_factory(), np.asarray(inertia, dtype="<f8").reshape(3, 3).tobytes())
    return (
        np.frombuffer(context.propagate(np.asarray(state, dtype="<f8").reshape(7).tobytes(), float(dt_s)), dtype="<f8")
        .copy()
        .reshape(7)
    )
