from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, MOON_MU_KM3_S2
from sim.dynamics.orbit.integrators import rk4_step_state

EARTH_MOON_DISTANCE_KM = 384400.0
EARTH_MOON_MU = MOON_MU_KM3_S2 / (EARTH_MU_KM3_S2 + MOON_MU_KM3_S2)
EARTH_MOON_MEAN_MOTION_RAD_S = float(
    np.sqrt((EARTH_MU_KM3_S2 + MOON_MU_KM3_S2) / (EARTH_MOON_DISTANCE_KM**3))
)


@dataclass(frozen=True)
class CR3BPSystem:
    name: str = "earth_moon"
    distance_km: float = EARTH_MOON_DISTANCE_KM
    mu: float = EARTH_MOON_MU
    mean_motion_rad_s: float = EARTH_MOON_MEAN_MOTION_RAD_S


EARTH_MOON_CR3BP = CR3BPSystem()


def cr3bp_system(name: str = "earth_moon") -> CR3BPSystem:
    key = str(name or "earth_moon").strip().lower().replace("-", "_")
    if key in {"earth_moon", "earthmoon", "em"}:
        return EARTH_MOON_CR3BP
    raise ValueError(f"Unsupported CR3BP system '{name}'.")


def cr3bp_l1_position_km(system: CR3BPSystem | None = None) -> float:
    sys = EARTH_MOON_CR3BP if system is None else system
    return float(_collinear_libration_x(sys.mu, point="l1") * sys.distance_km)


def cr3bp_l1_state_km_s(system: CR3BPSystem | None = None) -> np.ndarray:
    state = np.zeros(6, dtype=float)
    state[0] = cr3bp_l1_position_km(system)
    return state


def cr3bp_moon_position_km(system: CR3BPSystem | None = None) -> float:
    sys = EARTH_MOON_CR3BP if system is None else system
    return float((1.0 - float(sys.mu)) * float(sys.distance_km))


def cr3bp_moon_state_km_s(system: CR3BPSystem | None = None) -> np.ndarray:
    state = np.zeros(6, dtype=float)
    state[0] = cr3bp_moon_position_km(system)
    return state


def cr3bp_halo_seed_state_km_s(
    *,
    system: CR3BPSystem | None = None,
    family: str = "l1_northern",
) -> np.ndarray:
    """Return a deterministic Earth-Moon CR3BP orbit seed in rotating coordinates.

    The larger L1 family is a corrected periodic northern halo orbit. The NRHO
    family is a corrected southern L2 near-rectilinear halo orbit. The default
    family remains the original small training seed for backward compatibility
    with existing scenarios.
    """

    sys = EARTH_MOON_CR3BP if system is None else system
    key = str(family or "l1_northern").strip().lower().replace("-", "_")
    if key in {"l1_northern_large", "l1_large", "large", "large_northern", "l1_northern_corrected_large"}:
        state_nd = np.array(
            [
                0.825758855860872,
                0.0,
                0.08,
                0.0,
                0.193697258505661,
                0.0,
            ],
            dtype=float,
        )
        return cr3bp_dimensional_state(state_nd, system=sys)
    if key in {"l2_nrho_southern", "nrho", "southern_nrho", "l2_southern_nrho", "gateway_nrho"}:
        state_nd = np.array(
            [
                1.0213444229227893,
                0.0,
                -0.181626,
                0.0,
                -0.10177667502757823,
                0.0,
            ],
            dtype=float,
        )
        return cr3bp_dimensional_state(state_nd, system=sys)
    if key not in {"l1_northern", "northern", "north"}:
        raise ValueError(f"Unsupported CR3BP orbit seed family '{family}'.")
    state_nd = np.array(
        [
            0.823385182067467,
            0.0,
            0.022277556,
            0.0,
            0.134184,
            0.0,
        ],
        dtype=float,
    )
    return cr3bp_dimensional_state(state_nd, system=sys)


def cr3bp_dimensional_state(state_nd: np.ndarray, *, system: CR3BPSystem | None = None) -> np.ndarray:
    sys = EARTH_MOON_CR3BP if system is None else system
    state = np.array(state_nd, dtype=float).reshape(6).copy()
    state[:3] *= float(sys.distance_km)
    state[3:] *= float(sys.distance_km) * float(sys.mean_motion_rad_s)
    return state


def cr3bp_nondimensional_state(state_km_s: np.ndarray, *, system: CR3BPSystem | None = None) -> np.ndarray:
    sys = EARTH_MOON_CR3BP if system is None else system
    state = np.array(state_km_s, dtype=float).reshape(6).copy()
    state[:3] /= float(sys.distance_km)
    state[3:] /= float(sys.distance_km) * float(sys.mean_motion_rad_s)
    return state


def cr3bp_relative_state(deputy_state: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
    return np.array(deputy_state, dtype=float).reshape(6) - np.array(reference_state, dtype=float).reshape(6)


def propagate_cr3bp_state(
    state_km_s: np.ndarray,
    dt_s: float,
    t_s: float,
    command_accel_km_s2: np.ndarray | None = None,
    *,
    system: CR3BPSystem | None = None,
) -> np.ndarray:
    sys = EARTH_MOON_CR3BP if system is None else system
    command = np.zeros(3, dtype=float) if command_accel_km_s2 is None else np.array(command_accel_km_s2, dtype=float).reshape(3)

    def deriv(t_local: float, x_local: np.ndarray) -> np.ndarray:
        return cr3bp_derivative_physical(x_local, command_accel_km_s2=command, system=sys)

    return rk4_step_state(deriv_fn=deriv, t_s=float(t_s), x=np.array(state_km_s, dtype=float).reshape(6), dt_s=float(dt_s))


def propagate_cr3bp_reference_stm(
    reference_state_km_s: np.ndarray,
    stm: np.ndarray,
    dt_s: float,
    t_s: float,
    *,
    system: CR3BPSystem | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sys = EARTH_MOON_CR3BP if system is None else system
    augmented = np.hstack(
        (
            np.array(reference_state_km_s, dtype=float).reshape(6),
            np.array(stm, dtype=float).reshape(6, 6).reshape(36),
        )
    )

    def deriv(t_local: float, x_local: np.ndarray) -> np.ndarray:
        reference = np.array(x_local[:6], dtype=float)
        phi = np.array(x_local[6:], dtype=float).reshape(6, 6)
        a = cr3bp_jacobian_physical(reference, system=sys)
        return np.hstack((cr3bp_derivative_physical(reference, system=sys), (a @ phi).reshape(36)))

    out = rk4_step_state(deriv_fn=deriv, t_s=float(t_s), x=augmented, dt_s=float(dt_s))
    return out[:6].copy(), out[6:].reshape(6, 6).copy()


def cr3bp_derivative_physical(
    state_km_s: np.ndarray,
    *,
    command_accel_km_s2: np.ndarray | None = None,
    system: CR3BPSystem | None = None,
) -> np.ndarray:
    sys = EARTH_MOON_CR3BP if system is None else system
    state_nd = cr3bp_nondimensional_state(state_km_s, system=sys)
    deriv_nd = cr3bp_derivative_nondimensional(state_nd, mu=sys.mu)
    deriv = np.empty(6, dtype=float)
    deriv[:3] = np.array(state_km_s, dtype=float).reshape(6)[3:]
    deriv[3:] = deriv_nd[3:] * float(sys.distance_km) * float(sys.mean_motion_rad_s) ** 2
    if command_accel_km_s2 is not None:
        deriv[3:] += np.array(command_accel_km_s2, dtype=float).reshape(3)
    return deriv


def cr3bp_jacobian_physical(state_km_s: np.ndarray, *, system: CR3BPSystem | None = None) -> np.ndarray:
    sys = EARTH_MOON_CR3BP if system is None else system
    state_nd = cr3bp_nondimensional_state(state_km_s, system=sys)
    u_rr_nd = cr3bp_potential_hessian_nondimensional(state_nd, mu=sys.mu)
    n = float(sys.mean_motion_rad_s)
    a = np.zeros((6, 6), dtype=float)
    a[:3, 3:] = np.eye(3, dtype=float)
    a[3:, :3] = (n**2) * u_rr_nd
    a[3:, 3:] = n * np.array(
        [
            [0.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return a


def cr3bp_derivative_nondimensional(state_nd: np.ndarray, *, mu: float = EARTH_MOON_MU) -> np.ndarray:
    x, y, z, vx, vy, vz = np.array(state_nd, dtype=float).reshape(6)
    mu2 = float(mu)
    mu1 = 1.0 - mu2
    r1 = float(np.sqrt((x + mu2) ** 2 + y * y + z * z))
    r2 = float(np.sqrt((x - mu1) ** 2 + y * y + z * z))
    r1 = max(r1, 1.0e-12)
    r2 = max(r2, 1.0e-12)
    d_omega_dx = x - mu1 * (x + mu2) / (r1**3) - mu2 * (x - mu1) / (r2**3)
    d_omega_dy = y - mu1 * y / (r1**3) - mu2 * y / (r2**3)
    d_omega_dz = -mu1 * z / (r1**3) - mu2 * z / (r2**3)
    return np.array(
        [
            vx,
            vy,
            vz,
            2.0 * vy + d_omega_dx,
            -2.0 * vx + d_omega_dy,
            d_omega_dz,
        ],
        dtype=float,
    )


def cr3bp_potential_hessian_nondimensional(state_nd: np.ndarray, *, mu: float = EARTH_MOON_MU) -> np.ndarray:
    x, y, z = np.array(state_nd, dtype=float).reshape(6)[:3]
    mu2 = float(mu)
    mu1 = 1.0 - mu2
    x1 = x + mu2
    x2 = x - mu1
    r1_sq = x1 * x1 + y * y + z * z
    r2_sq = x2 * x2 + y * y + z * z
    r1 = max(float(np.sqrt(r1_sq)), 1.0e-12)
    r2 = max(float(np.sqrt(r2_sq)), 1.0e-12)
    r1_3 = r1**3
    r2_3 = r2**3
    r1_5 = r1**5
    r2_5 = r2**5

    u_xx = 1.0 - mu1 * (1.0 / r1_3 - 3.0 * x1 * x1 / r1_5) - mu2 * (
        1.0 / r2_3 - 3.0 * x2 * x2 / r2_5
    )
    u_yy = 1.0 - mu1 * (1.0 / r1_3 - 3.0 * y * y / r1_5) - mu2 * (
        1.0 / r2_3 - 3.0 * y * y / r2_5
    )
    u_zz = -mu1 * (1.0 / r1_3 - 3.0 * z * z / r1_5) - mu2 * (
        1.0 / r2_3 - 3.0 * z * z / r2_5
    )
    u_xy = 3.0 * mu1 * x1 * y / r1_5 + 3.0 * mu2 * x2 * y / r2_5
    u_xz = 3.0 * mu1 * x1 * z / r1_5 + 3.0 * mu2 * x2 * z / r2_5
    u_yz = 3.0 * mu1 * y * z / r1_5 + 3.0 * mu2 * y * z / r2_5
    return np.array(
        [
            [u_xx, u_xy, u_xz],
            [u_xy, u_yy, u_yz],
            [u_xz, u_yz, u_zz],
        ],
        dtype=float,
    )


def _collinear_libration_x(mu: float, *, point: str) -> float:
    key = str(point or "l1").strip().lower()
    if key != "l1":
        raise ValueError(f"Unsupported collinear libration point '{point}'.")
    x = 1.0 - float(mu) - (float(mu) / 3.0) ** (1.0 / 3.0)
    for _ in range(50):
        f = _collinear_equilibrium_f(x, float(mu))
        h = 1.0e-7
        df = (_collinear_equilibrium_f(x + h, float(mu)) - _collinear_equilibrium_f(x - h, float(mu))) / (2.0 * h)
        if abs(df) <= 1.0e-14:
            break
        step = f / df
        x -= step
        if abs(step) <= 1.0e-14:
            break
    return float(x)


def _collinear_equilibrium_f(x: float, mu: float) -> float:
    mu2 = float(mu)
    mu1 = 1.0 - mu2
    r1 = abs(float(x) + mu2)
    r2 = abs(float(x) - mu1)
    return float(x) - mu1 * (float(x) + mu2) / (r1**3) - mu2 * (float(x) - mu1) / (r2**3)
