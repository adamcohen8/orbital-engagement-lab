"""Optional Rust OGP-SGP4/SDP4 binding on OEL's numeric mean-element contract."""

from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.tle import TLEElements
from sim.rust_orbit_backend import _extension


def _numeric_elements(elements: TLEElements) -> list[float]:
    return [
        float(elements.epoch_jd_utc),
        float(elements.mean_motion_rev_per_day),
        float(elements.eccentricity),
        float(elements.inclination_deg),
        float(elements.raan_deg),
        float(elements.argp_deg),
        float(elements.mean_anomaly_deg),
        float(elements.bstar),
    ]


class RustOGPContext:
    """One initialized Rust OGP record; accepts arbitrary time-query order."""

    def __init__(self, elements: TLEElements) -> None:
        self._native = _extension().OGPContext(_numeric_elements(elements))

    def propagate(self, tsince_min: float) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self._native, "propagate_bytes"):
            values = np.frombuffer(self._native.propagate_bytes(float(tsince_min)), dtype="<f8").copy()
            return values[:3], values[3:]
        position, velocity = self._native.propagate(float(tsince_min))
        return np.asarray(position, dtype=np.float64), np.asarray(velocity, dtype=np.float64)

    def propagate_many(self, tsince_min: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.asarray(tsince_min, dtype=np.float64).reshape(-1)
        if hasattr(self._native, "propagate_many_bytes"):
            positions, velocities, sparse_errors = self._native.propagate_many_bytes(np.asarray(times, dtype="<f8").tobytes())
            errors = np.full(times.shape, "", dtype=object)
            for index, error in sparse_errors:
                errors[index] = error
            return (np.frombuffer(positions, dtype="<f8").reshape(-1, 3).copy(),
                    np.frombuffer(velocities, dtype="<f8").reshape(-1, 3).copy(), errors)
        positions, velocities, errors = self._native.propagate_many(times.tolist())
        return (
            np.asarray(positions, dtype=np.float64).reshape(-1, 3),
            np.asarray(velocities, dtype=np.float64).reshape(-1, 3),
            np.asarray(errors, dtype=object),
        )


def propagate_batch(elements: list[TLEElements], time_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One binary batch boundary, with the older wheel's row API as fallback."""

    native = _extension()
    shape = (*time_grid.shape, 3)
    errors = np.full(time_grid.shape, "", dtype=object)
    if hasattr(native, "ogp_propagate_batch_bytes"):
        numeric = np.asarray([_numeric_elements(element) for element in elements], dtype="<f8")
        positions, velocities, sparse_errors = native.ogp_propagate_batch_bytes(
            numeric.tobytes(), np.asarray(time_grid, dtype="<f8").tobytes(), time_grid.shape[1],
        )
        for index, error in sparse_errors:
            errors.flat[index] = error
        return (
            np.frombuffer(positions, dtype="<f8").reshape(shape).copy(),
            np.frombuffer(velocities, dtype="<f8").reshape(shape).copy(),
            errors,
        )
    positions = np.zeros(shape, dtype=float)
    velocities = np.zeros(shape, dtype=float)
    for index, element in enumerate(elements):
        try:
            row_pos, row_vel, row_errors = RustOGPContext(element).propagate_many(time_grid[index])
        except (RuntimeError, ValueError) as exc:
            errors[index, :] = str(exc)
            continue
        positions[index], velocities[index], errors[index] = row_pos, row_vel, row_errors
    return positions, velocities, errors


class RustOGPObservationBatch:
    """Immutable observation and whitening arrays held natively for one fit."""

    def __init__(self, times_jd_utc, positions, velocities, factors) -> None:
        native = _extension()
        if not hasattr(native, "OGPObservationBatch"):
            raise RuntimeError("Fused Rust OGP residuals require an oel_rust_orbit wheel with OGPObservationBatch.")
        observations = np.column_stack((positions, velocities))
        if hasattr(native.OGPObservationBatch, "from_bytes"):
            self._native = native.OGPObservationBatch.from_bytes(
                np.asarray(times_jd_utc, dtype="<f8").tobytes(),
                np.asarray(observations, dtype="<f8").tobytes(),
                np.asarray(factors, dtype="<f8").tobytes(),
            )
            return
        self._native = native.OGPObservationBatch(
            np.asarray(times_jd_utc, dtype=float).tolist(),
            observations.ravel().tolist(),
            np.asarray(factors, dtype=float).ravel().tolist(),
        )

    def residual(self, elements: TLEElements) -> np.ndarray:
        values = self._native.residual_bytes(_numeric_elements(elements))
        return np.frombuffer(values, dtype="<f8").copy()
