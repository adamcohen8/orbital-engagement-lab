"""Frame and invariant utilities for ideal CR3BP research (not J2000 alignment).

States are [..., 6], physical km and km/s. Inertial axes coincide with rotating
axes at reference_time_s when reference_angle_rad is zero. Primary-centered
inertial axes are nonrotating but their origins accelerate with the primary.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from sim.dynamics.orbit.cr3bp import EARTH_MOON_CR3BP, CR3BPSystem


def _system(system: CR3BPSystem) -> None:
    if not (
        np.isfinite([system.mu, system.distance_km, system.mean_motion_rad_s]).all()
        and 0 < system.mu < 1
        and system.distance_km > 0
        and system.mean_motion_rad_s > 0
    ):
        raise ValueError("CR3BP requires 0 < mu < 1 and positive finite distance and mean motion.")


def _states(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=float)
    if value.ndim < 1 or value.shape[-1] != 6 or not np.isfinite(value).all():
        raise ValueError("States must be finite arrays with last dimension 6.")
    return value


def _origin(origin: str, system: CR3BPSystem) -> np.ndarray:
    if origin not in {"barycenter", "p1", "p2"}:
        raise ValueError("Origin must be barycenter, p1, or p2.")
    x = {"barycenter": 0.0, "p1": -system.mu, "p2": 1 - system.mu}[origin]
    return np.array([x * system.distance_km, 0.0, 0.0])


def transform_cr3bp_state(
    state: np.ndarray,
    time_s: float | np.ndarray,
    *,
    source_axes: str = "rotating",
    target_axes: str = "inertial",
    source_origin: str = "barycenter",
    target_origin: str = "barycenter",
    reference_time_s: float = 0.0,
    reference_angle_rad: float = 0.0,
    system: CR3BPSystem = EARTH_MOON_CR3BP,
) -> np.ndarray:
    """Convert axes and origins, including transport velocity; scalar or batched.

    time_s broadcasts against state batch dimensions. Velocities are derivatives
    of coordinates in the selected axes relative to the selected moving origin.
    """
    _system(system)
    state = _states(state)
    if source_axes not in {"rotating", "inertial"} or target_axes not in {"rotating", "inertial"}:
        raise ValueError("Axes must be rotating or inertial.")
    times = np.asarray(time_s, dtype=float)
    if not np.isfinite(times).all() or not np.isfinite([reference_time_s, reference_angle_rad]).all():
        raise ValueError("Times and reference angle must be finite.")
    theta = reference_angle_rad + system.mean_motion_rad_s * (times - reference_time_s)
    shape = np.broadcast_shapes(state.shape[:-1], theta.shape)
    state = np.broadcast_to(state, shape + (6,))
    theta = np.broadcast_to(theta, shape)

    def rotate(v: np.ndarray, angle: np.ndarray) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), axis=-1)

    omega = np.array([0.0, 0.0, system.mean_motion_rad_s])
    r, v = state[..., :3], state[..., 3:]
    if source_axes == "inertial":
        r = rotate(r, -theta)
        v = rotate(v, -theta) - np.cross(omega, r)
    r = r + _origin(source_origin, system) - _origin(target_origin, system)
    if target_axes == "inertial":
        v = rotate(v + np.cross(omega, r), theta)
        r = rotate(r, theta)
    return np.concatenate((r, v), axis=-1)


def cr3bp_effective_potential(position_nd: np.ndarray, *, system: CR3BPSystem = EARTH_MOON_CR3BP) -> np.ndarray:
    """Dimensionless Omega; singular primary locations return +infinity."""
    _system(system)
    p = np.asarray(position_nd, dtype=float)
    if p.shape[-1:] != (3,) or not np.isfinite(p).all():
        raise ValueError("Positions must be finite arrays with last dimension 3.")
    r1 = np.linalg.norm(p + np.array([system.mu, 0.0, 0.0]), axis=-1)
    r2 = np.linalg.norm(p - np.array([1 - system.mu, 0.0, 0.0]), axis=-1)
    with np.errstate(divide="ignore"):
        return 0.5 * (p[..., 0] ** 2 + p[..., 1] ** 2) + (1 - system.mu) / r1 + system.mu / r2


def cr3bp_jacobi_constant(state_km_s: np.ndarray, *, system: CR3BPSystem = EARTH_MOON_CR3BP) -> np.ndarray:
    """Dimensionless Jacobi C = 2 Omega - |v_rotating|^2, barycentric input."""
    _system(system)
    state = _states(state_km_s)
    r = state[..., :3] / system.distance_km
    v = state[..., 3:] / (system.distance_km * system.mean_motion_rad_s)
    return 2 * cr3bp_effective_potential(r, system=system) - np.sum(v * v, axis=-1)


def cr3bp_jacobi_diagnostics(states_km_s: np.ndarray, *, system: CR3BPSystem = EARTH_MOON_CR3BP) -> dict:
    values = np.atleast_1d(cr3bp_jacobi_constant(states_km_s, system=system))
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Jacobi diagnostics require a nonempty nonsingular state history.")
    delta = values - values[0]
    return {
        "initial": float(values[0]),
        "final": float(values[-1]),
        "max_absolute_change": float(np.max(np.abs(delta))),
        "relative_change_scale": float(np.max(np.abs(delta)) / max(abs(values[0]), np.finfo(float).eps)),
        "interpretation": "Conservation applies only to unforced ideal CR3BP; thrust changes C.",
    }


def cr3bp_libration_points(*, system: CR3BPSystem = EARTH_MOON_CR3BP) -> dict[str, np.ndarray]:
    """All five equilibria in physical barycentric rotating km (zero velocity)."""
    _system(system)
    mu = system.mu

    def f(x: float) -> float:
        return x - (1 - mu) * (x + mu) / abs(x + mu) ** 3 - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3

    eps = min(mu, 1 - mu) * 1e-8
    roots = {
        "L1": brentq(f, -mu + eps, 1 - mu - eps),
        "L2": brentq(f, 1 - mu + eps, 3.0),
        "L3": brentq(f, -3.0, -mu - eps),
    }
    points = {key: np.array([x, 0.0, 0.0]) * system.distance_km for key, x in roots.items()}
    points.update(
        {
            "L4": np.array([0.5 - mu, np.sqrt(3) / 2, 0.0]) * system.distance_km,
            "L5": np.array([0.5 - mu, -np.sqrt(3) / 2, 0.0]) * system.distance_km,
        }
    )
    return points


def cr3bp_zero_velocity_grid(
    jacobi: float,
    *,
    plane: str = "xy",
    slice_km: float = 0.0,
    bounds_km: tuple[float, float, float, float] | None = None,
    resolution: int = 301,
    system: CR3BPSystem = EARTH_MOON_CR3BP,
) -> tuple:
    """Return mesh u,v and 2 Omega-C; negative values are forbidden on this slice."""
    _system(system)
    if plane not in {"xy", "xz", "yz"} or not np.isfinite([jacobi, slice_km]).all():
        raise ValueError("Use xy/xz/yz and finite Jacobi constant and slice.")
    if not isinstance(resolution, int) or not 16 <= resolution <= 1000:
        raise ValueError("Grid resolution must be an integer between 16 and 1000.")
    bounds = np.asarray(bounds_km if bounds_km is not None else np.array([-1.5, 1.5, -1.5, 1.5]) * system.distance_km)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("Bounds must be finite increasing (umin, umax, vmin, vmax).")
    u, v = np.meshgrid(np.linspace(*bounds[:2], resolution), np.linspace(*bounds[2:], resolution))
    p = np.full(u.shape + (3,), float(slice_km))
    i, j = ["xyz".index(axis) for axis in plane]
    p[..., i], p[..., j] = u, v
    return u, v, 2 * cr3bp_effective_potential(p / system.distance_km, system=system) - jacobi
