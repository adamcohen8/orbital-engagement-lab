"""Prepared immutable geometry batches; NumPy roots and selection stay in Python."""

import numpy as np

from sim.rust_orbit_backend import _extension


def _kernels():
    native = _extension()
    factory = getattr(native, "ConjunctionHistoryContext", None)
    hermite = getattr(native, "conjunction_hermite_batch", None)
    if factory is None or hermite is None:
        raise RuntimeError("Rust conjunction geometry requires an oel_rust_orbit wheel with conjunction batches")
    return factory, hermite


def history_context(times, states, incoming):
    factory, _ = _kernels()
    return factory(*(np.asarray(value, dtype="<f8").tobytes() for value in (times, states, incoming)))


def interpolate(context, queries, *, side):
    queries = np.asarray(queries, dtype="<f8").reshape(-1)
    raw = context.interpolate(queries.tobytes(), side == "left")
    return np.frombuffer(raw, dtype="<f8").copy().reshape(queries.size, 6)


def hermite_batch(cases):
    _, function = _kernels()
    values = np.asarray(cases, dtype="<f8").reshape(-1, 14)
    raw = function(values.tobytes())
    return np.frombuffer(raw, dtype="<f8").copy().reshape(values.shape[0], 6)
