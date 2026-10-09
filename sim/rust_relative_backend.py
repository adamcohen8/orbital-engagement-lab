"""Rust kernels for relative dynamics and covariance arithmetic.

Rust is the primary numerical backend; Python reference selection is explicit.  This module is deliberately
small: it validates OEL's public array shapes, performs the native call, and
returns NumPy arrays with the same units and ordering as the reference paths.
Nonlinear ONP force evaluation remains in the supplied Python propagator except
for the explicitly eligible two-body/J2 covariance finite-difference trials.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from math import isfinite
from typing import Iterable

import numpy as np


@lru_cache(maxsize=1)
def _extension():
    try:
        return import_module("oel_rust_orbit")
    except ImportError as exc:  # pragma: no cover - depends on optional wheel
        raise RuntimeError(
            "Rust relative backend requested but oel_rust_orbit is unavailable; "
            "install the optional oel_rust_orbit wheel first"
        ) from exc


def try_covariance_orbit_trials(
    nominal_state: np.ndarray,
    differences: np.ndarray,
    substeps_s: np.ndarray,
    *,
    mu_km3_s2: float,
    include_j2: bool,
    command_accel_eci_km_s2: np.ndarray,
) -> np.ndarray | None:
    """Batch twelve eligible two-body/J2 finite-difference trajectories."""
    native = getattr(_extension(), "covariance_orbit_trials_bytes", None)
    if native is None:
        return None
    nominal = np.asarray(nominal_state, dtype="<f8").reshape(6)
    steps = np.asarray(differences, dtype="<f8").reshape(6)
    substeps = np.asarray(substeps_s, dtype="<f8").reshape(-1)
    command = np.asarray(command_accel_eci_km_s2, dtype="<f8").reshape(3)
    raw = native(
        nominal.tobytes(), steps.tobytes(), substeps.tobytes(), float(mu_km3_s2),
        bool(include_j2), command.tobytes(),
    )
    return np.frombuffer(raw, dtype="<f8").copy().reshape(6, 12)


def try_orbit_trial_batch(propagator, states, widths_s, context, command=None):
    """Batch eligible native RK4 flows; unsupported owners keep their scalar path."""
    from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin, j3_plugin, j4_plugin

    codes = {j2_plugin: 2, j3_plugin: 3, j4_plugin: 4}
    if (type(propagator) is not OrbitPropagator or propagator.numeric_backend != "rust"
            or propagator.integrator != "rk4" or propagator.model != "two_body"
            or propagator.state_frame != "eci"
            or any(not any(plugin is known for known in codes) for plugin in propagator.plugins)):
        return None
    function = getattr(_extension(), "orbit_trial_batch_bytes", None)
    if function is None:
        return None
    values = np.asarray(states, dtype="<f8")
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("orbit trial states must have shape (N, 6)")
    widths = np.asarray(widths_s, dtype="<f8").reshape(-1)
    acceleration = np.zeros(3) if command is None else np.asarray(command, dtype=float).reshape(3)
    raw = function(values.tobytes(), widths.tobytes(), float(context.mu_km3_s2),
                   [codes[plugin] for plugin in propagator.plugins], acceleration.tolist())
    propagator.last_numeric_path = "rust_native_orbit_trial_batch"
    return np.frombuffer(raw, dtype="<f8").copy().reshape(values.shape)


def _state(value: np.ndarray, name: str) -> list[float]:
    state = np.asarray(value, dtype=np.float64).reshape(-1)
    if state.shape != (6,):
        raise ValueError(f"{name} must have shape (6,)")
    values = state.tolist()
    if not all(map(isfinite, values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def _matrix(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 36:
        raise ValueError(f"{name} must contain 36 values")
    return array.reshape(6, 6).copy()


def _plane_mean(value: np.ndarray, name: str) -> list[float]:
    mean = np.asarray(value, dtype=np.float64).reshape(-1)
    if mean.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    values = mean.tolist()
    if not all(map(isfinite, values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def _plane_covariance(value: np.ndarray, name: str) -> list[float]:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2)")
    values = covariance.reshape(-1).tolist()
    if not all(map(isfinite, values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def _backend_error(name: str, exc: Exception) -> RuntimeError:
    return RuntimeError(f"Rust relative kernel {name!r} is unavailable in the installed wheel")


def try_chief_state_propagation(
    chief_state_eci_km_s: np.ndarray,
    dt_s: float,
    *,
    mu_km3_s2: float,
    max_step_s: float,
) -> np.ndarray | None:
    """Use the native relative-model RK4 owner when the installed wheel has it."""
    function = getattr(_extension(), "relative_chief_propagate_bytes", None)
    if function is None:
        return None
    raw = function(
        _state(chief_state_eci_km_s, "chief_state_eci_km_s"),
        float(dt_s), float(mu_km3_s2), float(max_step_s),
    )
    return np.frombuffer(raw, dtype="<f8").copy().reshape(6)


def hcw_state_transition_matrix(mean_motion_rad_s: float, dt_s: float) -> np.ndarray:
    native = _extension()
    byte_function = getattr(native, "relative_hcw_stm_bytes", None)
    if byte_function is not None:
        return np.frombuffer(byte_function(float(mean_motion_rad_s), float(dt_s)), dtype="<f8").copy().reshape(6, 6)
    return _matrix(
        native.relative_hcw_stm(float(mean_motion_rad_s), float(dt_s)),
        "HCW STM",
    )


def ss_j2_state_transition_matrix(
    mean_motion_rad_s: float,
    reference_radius_km: float,
    reference_inclination_rad: float,
    *,
    j2: float,
    earth_radius_km: float,
    dt_s: float,
) -> np.ndarray:
    native = _extension()
    byte_function = getattr(native, "relative_ss_j2_stm_bytes", None)
    if byte_function is not None:
        raw = byte_function(float(mean_motion_rad_s), float(reference_radius_km), float(reference_inclination_rad),
                            float(j2), float(earth_radius_km), float(dt_s))
        return np.frombuffer(raw, dtype="<f8").copy().reshape(6, 6)
    return _matrix(
        native.relative_ss_j2_stm(
            float(mean_motion_rad_s),
            float(reference_radius_km),
            float(reference_inclination_rad),
            float(j2),
            float(earth_radius_km),
            float(dt_s),
        ),
        "SS-J2 STM",
    )


def th_propagate_relative_state(
    relative_state_ric: np.ndarray,
    dt_s: float,
    chief_state_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float,
    max_step_s: float,
) -> np.ndarray:
    result = _extension().relative_th_propagate(
        _state(relative_state_ric, "relative_state_ric"),
        float(dt_s),
        _state(chief_state_eci_km_s, "chief_state_eci_km_s"),
        float(mu_km3_s2),
        float(max_step_s),
    )
    return np.asarray(result, dtype=np.float64).reshape(6)


def th_finite_difference_transition_matrix(
    relative_state_ric: np.ndarray,
    dt_s: float,
    chief_state_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float,
    max_step_s: float,
) -> np.ndarray:
    return _matrix(
        _extension().relative_th_finite_difference(
            _state(relative_state_ric, "relative_state_ric"),
            float(dt_s),
            _state(chief_state_eci_km_s, "chief_state_eci_km_s"),
            float(mu_km3_s2),
            float(max_step_s),
        ),
        "TH finite-difference STM",
    )


def th_variational_propagate_relative_state_and_stm(
    relative_state_ric: np.ndarray,
    dt_s: float,
    chief_state_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float,
    max_step_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    native = _extension()
    byte_function = getattr(native, "relative_th_variational_bytes", None)
    state, phi = (byte_function or native.relative_th_variational)(
        _state(relative_state_ric, "relative_state_ric"),
        float(dt_s),
        _state(chief_state_eci_km_s, "chief_state_eci_km_s"),
        float(mu_km3_s2),
        float(max_step_s),
    )
    if byte_function is not None:
        return np.frombuffer(state, dtype="<f8").copy(), np.frombuffer(phi, dtype="<f8").copy().reshape(6, 6)
    return np.asarray(state, dtype=np.float64).reshape(6), _matrix(phi, "TH variational STM")


def ya_state_transition_matrix(
    dt_s: float,
    chief_start_eci_km_s: np.ndarray,
    chief_end_eci_km_s: np.ndarray | None = None,
    *,
    mu_km3_s2: float,
    max_step_s: float,
) -> np.ndarray:
    end = None if chief_end_eci_km_s is None else _state(chief_end_eci_km_s, "chief_end_eci_km_s")
    return _matrix(
        _extension().relative_ya_stm(
            float(dt_s),
            _state(chief_start_eci_km_s, "chief_start_eci_km_s"),
            end,
            float(mu_km3_s2),
            float(max_step_s),
        ),
        "YA STM",
    )


def ya_propagate_relative_state_and_stm(
    relative_state_ric: np.ndarray,
    dt_s: float,
    chief_start_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float,
    max_step_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    native = _extension()
    byte_function = getattr(native, "relative_ya_propagate_bytes", None)
    state, phi = (byte_function or native.relative_ya_propagate)(
        _state(relative_state_ric, "relative_state_ric"),
        float(dt_s),
        _state(chief_start_eci_km_s, "chief_start_eci_km_s"),
        float(mu_km3_s2),
        float(max_step_s),
    )
    if byte_function is not None:
        return np.frombuffer(state, dtype="<f8").copy(), np.frombuffer(phi, dtype="<f8").copy().reshape(6, 6)
    return np.asarray(state, dtype=np.float64).reshape(6), _matrix(phi, "YA STM")


def eci_relative_to_ric_batch(
    deputy_states_eci_km_s: np.ndarray,
    chief_states_eci_km_s: np.ndarray,
) -> np.ndarray:
    deputy = np.asarray(deputy_states_eci_km_s, dtype=np.float64)
    chief = np.asarray(chief_states_eci_km_s, dtype=np.float64)
    if deputy.ndim != 2 or chief.ndim != 2 or deputy.shape != chief.shape or deputy.shape[1] != 6:
        raise ValueError("deputy and chief histories must have equal shape (sample_count, 6)")
    native = _extension()
    byte_function = getattr(native, "relative_eci_to_ric_batch_bytes", None)
    if byte_function is not None:
        out = byte_function(np.asarray(deputy, dtype="<f8").tobytes(), np.asarray(chief, dtype="<f8").tobytes())
        return np.frombuffer(out, dtype="<f8").copy().reshape(deputy.shape)
    out = native.relative_eci_to_ric_batch(deputy.reshape(-1).tolist(), chief.reshape(-1).tolist())
    return np.asarray(out, dtype=np.float64).reshape(deputy.shape)


def ric_relative_to_eci_batch(
    relative_states_ric: np.ndarray,
    chief_states_eci_km_s: np.ndarray,
) -> np.ndarray:
    relative = np.asarray(relative_states_ric, dtype=np.float64)
    chief = np.asarray(chief_states_eci_km_s, dtype=np.float64)
    if relative.ndim != 2 or chief.ndim != 2 or relative.shape != chief.shape or relative.shape[1] != 6:
        raise ValueError("relative and chief histories must have equal shape (sample_count, 6)")
    native = _extension()
    byte_function = getattr(native, "relative_ric_to_eci_batch_bytes", None)
    if byte_function is not None:
        out = byte_function(np.asarray(relative, dtype="<f8").tobytes(), np.asarray(chief, dtype="<f8").tobytes())
        return np.frombuffer(out, dtype="<f8").copy().reshape(relative.shape)
    out = native.relative_ric_to_eci_batch(relative.reshape(-1).tolist(), chief.reshape(-1).tolist())
    return np.asarray(out, dtype=np.float64).reshape(relative.shape)


def finite_difference_stm(plus_minus_states: np.ndarray, steps: np.ndarray) -> np.ndarray:
    values = np.asarray(plus_minus_states, dtype=np.float64)
    step_values = np.asarray(steps, dtype=np.float64).reshape(-1)
    if values.shape != (6, 12):
        raise ValueError("plus_minus_states must have shape (6, 12)")
    if step_values.shape != (6,):
        raise ValueError("steps must have shape (6,)")
    out = _extension().covariance_finite_difference_stm(values.reshape(-1).tolist(), step_values.tolist())
    return _matrix(out, "finite-difference STM")


def finite_difference_stm_batch(plus_minus_states: np.ndarray, steps: np.ndarray) -> np.ndarray:
    values = np.asarray(plus_minus_states, dtype=np.float64)
    step_values = np.asarray(steps, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (6, 12):
        raise ValueError("plus_minus_states must have shape (sample_count, 6, 12)")
    if step_values.shape != (values.shape[0], 6):
        raise ValueError("steps must have shape (sample_count, 6)")
    native = _extension()
    byte_function = getattr(native, "covariance_finite_difference_stm_batch_bytes", None)
    if byte_function is not None:
        out = byte_function(np.asarray(values, dtype="<f8").tobytes(), np.asarray(step_values, dtype="<f8").tobytes())
        return np.frombuffer(out, dtype="<f8").copy().reshape(values.shape[0], 6, 6)
    out = native.covariance_finite_difference_stm_batch(
        values.reshape(-1).tolist(), step_values.reshape(-1).tolist()
    )
    return np.asarray(out, dtype=np.float64).reshape(values.shape[0], 6, 6)


def propagate_covariance_history(
    initial_covariance: np.ndarray,
    transition_history: np.ndarray,
    process_noise_history: np.ndarray,
) -> np.ndarray:
    initial = np.asarray(initial_covariance, dtype=np.float64)
    transitions = np.asarray(transition_history, dtype=np.float64)
    process_noise = np.asarray(process_noise_history, dtype=np.float64)
    if initial.shape != (6, 6):
        raise ValueError("initial_covariance must have shape (6, 6)")
    if transitions.ndim != 3 or transitions.shape[1:] != (6, 6):
        raise ValueError("transition_history must have shape (interval_count, 6, 6)")
    if process_noise.shape not in {(6, 6), transitions.shape}:
        raise ValueError("process_noise_history must have shape (6, 6) or match transition_history")
    native = _extension()
    byte_function = getattr(native, "covariance_propagate_history_bytes", None)
    if byte_function is not None:
        out = byte_function(np.asarray(initial, dtype="<f8").tobytes(),
                            np.asarray(transitions, dtype="<f8").tobytes(),
                            np.asarray(process_noise, dtype="<f8").tobytes())
        return np.frombuffer(out, dtype="<f8").copy().reshape(transitions.shape[0] + 1, 6, 6)
    out = native.covariance_propagate_history(
        initial.reshape(-1).tolist(),
        transitions.reshape(-1).tolist(),
        process_noise.reshape(-1).tolist(),
    )
    return np.asarray(out, dtype=np.float64).reshape(transitions.shape[0] + 1, 6, 6)


def small_object_collision_probability(
    mean_plane_km: np.ndarray,
    covariance_plane_km2: np.ndarray,
    hard_body_radius_km: float,
) -> float | None:
    """Evaluate OEL's small-hard-body encounter-plane screening estimate."""

    result = _extension().covariance_small_object_collision_probability(
        _plane_mean(mean_plane_km, "mean_plane_km"),
        _plane_covariance(covariance_plane_km2, "covariance_plane_km2"),
        float(hard_body_radius_km),
    )
    return None if result is None else float(result)
