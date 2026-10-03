"""Native stateless integrand; SciPy retains adaptive integration and errors."""

from functools import lru_cache

from scipy import LowLevelCallable

from sim.rust_orbit_backend import _extension


def _factory():
    factory = getattr(_extension(), "conjunction_probability_integrand_capsule", None)
    if not callable(factory):
        raise RuntimeError("Rust collision probability requires a wheel with the native Pc callback")
    return factory


@lru_cache(maxsize=2)
def _callback(factory):
    return LowLevelCallable(factory())


def integrand():
    # Resolve availability on every request, even when the capsule is cached.
    return _callback(_factory())
