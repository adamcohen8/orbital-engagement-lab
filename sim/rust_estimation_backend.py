"""Explicit opt-in bindings for the Rust estimation and measurement kernels.

The Python estimators own input normalization and policy.  This adapter only
packs validated NumPy arrays, calls the optional extension, and reconstructs
arrays with the caller's shapes.  The default backend remains Python.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Sequence

import numpy as np


@lru_cache(maxsize=1)
def _extension():
    try:
        return import_module("oel_rust_orbit")
    except ImportError as exc:  # pragma: no cover - exercised by install checks
        raise RuntimeError(
            "Rust estimation backend requested but oel_rust_orbit is unavailable; "
            "install the optional oel_rust_orbit wheel first"
        ) from exc


def _flat(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.reshape(-1)


def _f64_bytes(value: np.ndarray | Sequence[float]) -> bytes:
    """Pack validated numeric arrays as explicit little endian float64 bytes."""

    return np.ascontiguousarray(np.asarray(value, dtype="<f8")).tobytes()


def _f64_from_bytes(value: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Decode native little endian float64 bytes into writable NumPy storage."""

    return np.frombuffer(value, dtype="<f8").copy().reshape(shape)


def _require(function_name: str):
    extension = _extension()
    function = getattr(extension, function_name, None)
    if function is None:
        raise RuntimeError(
            f"installed oel_rust_orbit wheel does not provide {function_name}; "
            "rebuild the wheel with the estimation kernels"
        )
    return function


def try_two_body_observation_states(
    initial_state: Sequence[float] | np.ndarray,
    offsets_s: Sequence[float] | np.ndarray,
    *,
    mu_km3_s2: float,
    max_step_s: float = 30.0,
) -> np.ndarray | None:
    """Propagate independent IOD observation epochs in one native call.

    An older installed wheel retains the established Python propagation path.
    """
    native = getattr(_extension(), "iod_two_body_states_bytes", None)
    if native is None:
        return None
    state = _flat(initial_state, "initial_state")
    offsets = _flat(offsets_s, "offsets_s")
    if state.size != 6:
        raise ValueError("initial_state must have six elements")
    raw = native(_f64_bytes(state), _f64_bytes(offsets), float(mu_km3_s2), float(max_step_s))
    return _f64_from_bytes(raw, (offsets.size, 6))


def ekf_update(
    state: Sequence[float] | np.ndarray,
    covariance: Sequence[Sequence[float]] | np.ndarray,
    measurement: Sequence[float] | np.ndarray,
    measurement_matrix: Sequence[Sequence[float]] | np.ndarray,
    measurement_covariance: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run one Joseph-form EKF update in Rust."""

    x = _flat(state, "state")
    p = np.ascontiguousarray(np.asarray(covariance, dtype=np.float64))
    z = _flat(measurement, "measurement")
    h = np.ascontiguousarray(np.asarray(measurement_matrix, dtype=np.float64))
    r = np.ascontiguousarray(np.asarray(measurement_covariance, dtype=np.float64))
    if p.shape != (x.size, x.size):
        raise ValueError("covariance must be square and match state dimension")
    if h.shape != (z.size, x.size):
        raise ValueError("measurement_matrix must have shape (measurement_dimension, state_dimension)")
    if r.shape != (z.size, z.size):
        raise ValueError("measurement_covariance must be square and match measurement dimension")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(h)) or np.any(~np.isfinite(r)):
        raise ValueError("EKF inputs must contain only finite values")
    byte_function = getattr(_extension(), "estimation_ekf_update_bytes", None)
    if byte_function is not None:
        raw = byte_function(_f64_bytes(x), _f64_bytes(p), _f64_bytes(z), _f64_bytes(h), _f64_bytes(r))
        return (_f64_from_bytes(raw[0], x.shape), _f64_from_bytes(raw[1], p.shape),
                _f64_from_bytes(raw[2], z.shape), _f64_from_bytes(raw[3], r.shape), float(raw[4]))
    raw = _require("estimation_ekf_update")(
        x.tolist(),
        p.reshape(-1).tolist(),
        z.tolist(),
        h.reshape(-1).tolist(),
        r.reshape(-1).tolist(),
    )
    return (
        np.asarray(raw[0], dtype=np.float64),
        np.asarray(raw[1], dtype=np.float64).reshape(p.shape),
        np.asarray(raw[2], dtype=np.float64),
        np.asarray(raw[3], dtype=np.float64).reshape(r.shape),
        float(raw[4]),
    )


def ekf_update_innovation(
    state: np.ndarray, covariance: np.ndarray, innovation: np.ndarray,
    measurement_matrix: np.ndarray, measurement_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply one nonlinear owner's already wrapped innovation in Rust."""

    x = _flat(state, "state")
    p = np.ascontiguousarray(np.asarray(covariance, dtype=np.float64))
    y = _flat(innovation, "innovation")
    h = np.ascontiguousarray(np.asarray(measurement_matrix, dtype=np.float64))
    r = np.ascontiguousarray(np.asarray(measurement_covariance, dtype=np.float64))
    if p.shape != (x.size, x.size) or h.shape != (y.size, x.size) or r.shape != (y.size, y.size):
        raise ValueError("EKF innovation dimensions do not match state and measurement matrices")
    native = getattr(_extension(), "estimation_ekf_update_innovation_bytes", None)
    if native is None:
        raise RuntimeError("Rust estimation backend requires estimation_ekf_update_innovation_bytes")
    raw = native(_f64_bytes(x), _f64_bytes(p), _f64_bytes(y), _f64_bytes(h), _f64_bytes(r))
    return (_f64_from_bytes(raw[0], x.shape), _f64_from_bytes(raw[1], p.shape),
            _f64_from_bytes(raw[2], y.shape), _f64_from_bytes(raw[3], r.shape), float(raw[4]))


def ukf_sigma_points(
    state: Sequence[float] | np.ndarray,
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened sigma points and mean/covariance weights."""

    x = _flat(state, "state")
    p = np.ascontiguousarray(np.asarray(covariance, dtype=np.float64))
    if p.shape != (x.size, x.size):
        raise ValueError("covariance must be square and match state dimension")
    byte_function = getattr(_extension(), "estimation_ukf_sigma_points_bytes", None)
    if byte_function is not None:
        raw = byte_function(_f64_bytes(x), _f64_bytes(p), float(alpha), float(beta), float(kappa))
        count = 2 * x.size + 1
        return (_f64_from_bytes(raw[0], (count, x.size)), _f64_from_bytes(raw[1], (count,)),
                _f64_from_bytes(raw[2], (count,)))
    raw = _require("estimation_ukf_sigma_points")(
        x.tolist(), p.reshape(-1).tolist(), float(alpha), float(beta), float(kappa)
    )
    count = 2 * x.size + 1
    return (
        np.asarray(raw[0], dtype=np.float64).reshape(count, x.size),
        np.asarray(raw[1], dtype=np.float64),
        np.asarray(raw[2], dtype=np.float64),
    )


def ukf_recombine(
    sigma_points: Sequence[Sequence[float]] | np.ndarray,
    mean_weights: Sequence[float] | np.ndarray,
    covariance_weights: Sequence[float] | np.ndarray,
    process_covariance: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recombine propagated sigma points and add process covariance."""

    sigma = np.ascontiguousarray(np.asarray(sigma_points, dtype=np.float64))
    wm = _flat(mean_weights, "mean_weights")
    wc = _flat(covariance_weights, "covariance_weights")
    q = np.ascontiguousarray(np.asarray(process_covariance, dtype=np.float64))
    if sigma.ndim != 2 or sigma.shape[0] != wm.size or wc.size != wm.size:
        raise ValueError("sigma points and weights have incompatible shapes")
    if q.shape != (sigma.shape[1], sigma.shape[1]):
        raise ValueError("process_covariance must match sigma-point dimension")
    byte_function = getattr(_extension(), "estimation_ukf_recombine_bytes", None)
    if byte_function is not None:
        raw = byte_function(_f64_bytes(sigma), _f64_bytes(wm), _f64_bytes(wc), _f64_bytes(q))
        return _f64_from_bytes(raw[0], (sigma.shape[1],)), _f64_from_bytes(raw[1], q.shape)
    raw = _require("estimation_ukf_recombine")(
        sigma.reshape(-1).tolist(), wm.tolist(), wc.tolist(), q.reshape(-1).tolist()
    )
    return np.asarray(raw[0], dtype=np.float64), np.asarray(raw[1], dtype=np.float64).reshape(q.shape)


def two_body_predict_and_jacobian(state, dt_s: float, mu_km3_s2: float, epsilon: float = 1.0e-6):
    """Evaluate the existing seven RK4 states behind one native boundary."""

    values = _flat(state, "state")
    if values.shape != (6,):
        raise ValueError("state must have shape (6,)")
    raw = _require("estimation_two_body_predict_and_jacobian_bytes")(
        _f64_bytes(values), float(dt_s), float(mu_km3_s2), float(epsilon),
    )
    return _f64_from_bytes(raw[0], (6,)), _f64_from_bytes(raw[1], (6, 6))


def jacobian_from_perturbed(
    base: Sequence[float] | np.ndarray,
    perturbed: Sequence[Sequence[float]] | np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Assemble a forward-difference Jacobian from propagated perturbations."""

    base_array = _flat(base, "base")
    perturbed_array = np.ascontiguousarray(np.asarray(perturbed, dtype=np.float64))
    if perturbed_array.shape != (base_array.size, base_array.size):
        raise ValueError("perturbed must have shape (state_dimension, state_dimension)")
    byte_function = getattr(_extension(), "estimation_jacobian_from_perturbed_bytes", None)
    if byte_function is not None:
        raw = byte_function(_f64_bytes(base_array), _f64_bytes(perturbed_array), float(epsilon))
        return _f64_from_bytes(raw, (base_array.size, base_array.size))
    raw = _require("estimation_jacobian_from_perturbed")(
        base_array.tolist(), perturbed_array.reshape(-1).tolist(), float(epsilon)
    )
    return np.asarray(raw, dtype=np.float64).reshape(base_array.size, base_array.size)



def _uniform_whitening_blocks(residual_blocks, factor_blocks):
    """Pack a rectangular block sequence in bulk, or use the ragged path."""

    try:
        residuals = np.asarray(residual_blocks, dtype=np.float64)
        factors = np.asarray(factor_blocks, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if residuals.ndim != 2 or factors.shape != (residuals.shape[0], residuals.shape[1], residuals.shape[1]):
        return None
    if not np.isfinite(residuals).all() or not np.isfinite(factors).all():
        return None
    return residuals, factors, [residuals.shape[1]] * residuals.shape[0]

def whiten_residual_blocks(
    residual_blocks: Sequence[Sequence[float] | np.ndarray],
    factor_blocks: Sequence[Sequence[Sequence[float]] | np.ndarray],
) -> np.ndarray:
    """Whiten variable-width residual blocks with precomputed factors."""

    if len(residual_blocks) != len(factor_blocks):
        raise ValueError("residual_blocks and factor_blocks must have the same length")
    uniform = _uniform_whitening_blocks(residual_blocks, factor_blocks)
    if uniform is not None:
        residual_values, factor_values, dimensions = uniform
        byte_function = getattr(_extension(), "estimation_whiten_residual_blocks_bytes", None)
        if byte_function is not None:
            raw = byte_function(_f64_bytes(residual_values), _f64_bytes(factor_values), dimensions)
            return _f64_from_bytes(raw, (residual_values.size,))
    residual_arrays = [_flat(block, "residual block") for block in residual_blocks]
    factors = [np.ascontiguousarray(np.asarray(block, dtype=np.float64)) for block in factor_blocks]
    dimensions = [int(block.size) for block in residual_arrays]
    for factor, dimension in zip(factors, dimensions, strict=True):
        if factor.shape != (dimension, dimension):
            raise ValueError("each factor must be square and match its residual block")
        if np.any(~np.isfinite(factor)):
            raise ValueError("residual factors must contain only finite values")
    byte_function = getattr(_extension(), "estimation_whiten_residual_blocks_bytes", None)
    if byte_function is not None:
        residual_values = np.concatenate(residual_arrays) if residual_arrays else np.empty(0, dtype=np.float64)
        factor_values = (
            np.concatenate([factor.reshape(-1) for factor in factors])
            if factors
            else np.empty(0, dtype=np.float64)
        )
        raw = byte_function(_f64_bytes(residual_values), _f64_bytes(factor_values), dimensions)
        return _f64_from_bytes(raw, (sum(dimensions),))
    raw = _require("estimation_whiten_residual_blocks")(
        np.concatenate(residual_arrays).tolist() if residual_arrays else [],
        np.concatenate([factor.reshape(-1) for factor in factors]).tolist() if factors else [],
        dimensions,
    )
    return np.asarray(raw, dtype=np.float64)


def measurement_whiten_rows(
    residuals: Sequence[Sequence[float]] | np.ndarray,
    factors: Sequence[Sequence[Sequence[float]]] | np.ndarray,
) -> np.ndarray:
    """Whiten fixed-width measurement rows with one native call."""

    residual_array = np.ascontiguousarray(np.asarray(residuals, dtype=np.float64))
    factor_array = np.ascontiguousarray(np.asarray(factors, dtype=np.float64))
    if residual_array.ndim != 2:
        raise ValueError("residuals must be a 2D array")
    rows, dimension = residual_array.shape
    if factor_array.shape != (rows, dimension, dimension):
        raise ValueError("factors must have shape (rows, dimension, dimension)")
    byte_function = getattr(_extension(), "measurement_whiten_rows_bytes", None)
    if byte_function is not None:
        raw = byte_function(
            _f64_bytes(residual_array),
            _f64_bytes(factor_array),
            rows,
            dimension,
        )
        return _f64_from_bytes(raw, (rows, dimension))
    raw = _require("measurement_whiten_rows")(
        residual_array.reshape(-1).tolist(), factor_array.reshape(-1).tolist(), rows, dimension
    )
    return np.asarray(raw, dtype=np.float64).reshape(rows, dimension)


def measurement_whiten_variable_rows(
    residual_blocks: Sequence[Sequence[float] | np.ndarray],
    factor_blocks: Sequence[Sequence[Sequence[float]] | np.ndarray],
) -> np.ndarray:
    """Whiten variable-width ground/radar/optical/SLR residual blocks."""

    if len(residual_blocks) != len(factor_blocks):
        raise ValueError("residual_blocks and factor_blocks must have the same length")
    uniform = _uniform_whitening_blocks(residual_blocks, factor_blocks)
    if uniform is not None:
        residual_values, factor_values, dimensions = uniform
        byte_function = getattr(_extension(), "measurement_whiten_variable_rows_bytes", None)
        if byte_function is not None:
            raw = byte_function(_f64_bytes(residual_values), _f64_bytes(factor_values), dimensions)
            return _f64_from_bytes(raw, (residual_values.size,))
    residual_arrays = [_flat(block, "residual block") for block in residual_blocks]
    factors = [np.ascontiguousarray(np.asarray(block, dtype=np.float64)) for block in factor_blocks]
    dimensions = [int(block.size) for block in residual_arrays]
    for factor, dimension in zip(factors, dimensions, strict=True):
        if factor.shape != (dimension, dimension):
            raise ValueError("each factor must be square and match its residual block")
        if np.any(~np.isfinite(factor)):
            raise ValueError("residual factors must contain only finite values")
    byte_function = getattr(_extension(), "measurement_whiten_variable_rows_bytes", None)
    if byte_function is not None:
        residual_values = np.concatenate(residual_arrays) if residual_arrays else np.empty(0, dtype=np.float64)
        factor_values = (
            np.concatenate([factor.reshape(-1) for factor in factors])
            if factors
            else np.empty(0, dtype=np.float64)
        )
        raw = byte_function(_f64_bytes(residual_values), _f64_bytes(factor_values), dimensions)
        return _f64_from_bytes(raw, (sum(dimensions),))
    raw = _require("measurement_whiten_variable_rows")(
        np.concatenate(residual_arrays).tolist() if residual_arrays else [],
        np.concatenate([factor.reshape(-1) for factor in factors]).tolist() if factors else [],
        dimensions,
    )
    return np.asarray(raw, dtype=np.float64)


def ground_station_predictions(
    states_eci: Sequence[Sequence[float]] | np.ndarray,
    target_ecef: Sequence[Sequence[float]] | np.ndarray,
    station_ecef: Sequence[Sequence[float]] | np.ndarray,
    station_eci: Sequence[Sequence[float]] | np.ndarray,
    station_velocity_eci: Sequence[Sequence[float]] | np.ndarray,
    enu_rotations: Sequence[Sequence[Sequence[float]]] | np.ndarray,
) -> np.ndarray:
    """Predict batched ground-station azimuth/elevation/range/range-rate."""

    arrays = [
        np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        for value in (states_eci, target_ecef, station_ecef, station_eci, station_velocity_eci, enu_rotations)
    ]
    state_array, target_array, station_array, station_eci_array, velocity_array, rotation_array = arrays
    if state_array.ndim != 2 or state_array.shape[1] != 6:
        raise ValueError("states_eci must have shape (rows, 6)")
    rows = state_array.shape[0]
    if target_array.shape != (rows, 3) or station_array.shape != (rows, 3):
        raise ValueError("station and target ECEF arrays must have shape (rows, 3)")
    if station_eci_array.shape != (rows, 3) or velocity_array.shape != (rows, 3):
        raise ValueError("station ECI arrays must have shape (rows, 3)")
    if rotation_array.shape != (rows, 3, 3):
        raise ValueError("enu_rotations must have shape (rows, 3, 3)")
    byte_function = getattr(_extension(), "measurement_ground_station_batch_bytes", None)
    if byte_function is not None:
        raw = byte_function(
            _f64_bytes(state_array),
            _f64_bytes(target_array),
            _f64_bytes(station_array),
            _f64_bytes(station_eci_array),
            _f64_bytes(velocity_array),
            _f64_bytes(rotation_array),
        )
        return _f64_from_bytes(raw, (rows, 4))
    raw = _require("measurement_ground_station_batch")(
        state_array.reshape(-1).tolist(),
        target_array.reshape(-1).tolist(),
        station_array.reshape(-1).tolist(),
        station_eci_array.reshape(-1).tolist(),
        velocity_array.reshape(-1).tolist(),
        rotation_array.reshape(-1).tolist(),
    )
    return np.asarray(raw, dtype=np.float64).reshape(rows, 4)


def optical_radec_predictions(relative_positions: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    positions = np.ascontiguousarray(np.asarray(relative_positions, dtype=np.float64))
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("relative_positions must have shape (rows, 3)")
    byte_function = getattr(_extension(), "measurement_optical_radec_batch_bytes", None)
    if byte_function is not None:
        raw = byte_function(_f64_bytes(positions))
        return _f64_from_bytes(raw, (positions.shape[0], 2))
    raw = _require("measurement_optical_radec_batch")(positions.reshape(-1).tolist())
    return np.asarray(raw, dtype=np.float64).reshape(positions.shape[0], 2)


def range_predictions(relative_positions: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    positions = np.ascontiguousarray(np.asarray(relative_positions, dtype=np.float64))
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("relative_positions must have shape (rows, 3)")
    byte_function = getattr(_extension(), "measurement_range_batch_bytes", None)
    if byte_function is not None:
        raw = byte_function(_f64_bytes(positions))
        return _f64_from_bytes(raw, (positions.shape[0],))
    raw = _require("measurement_range_batch")(positions.reshape(-1).tolist())
    return np.asarray(raw, dtype=np.float64)
