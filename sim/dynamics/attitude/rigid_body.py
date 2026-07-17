from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass

import numpy as np

from sim.acceleration.settings import acceleration_enabled_from_mode
from sim.utils.quaternion import (
    normalize_quaternion,
    omega_matrix,
    quaternion_delta_from_body_rate,
    quaternion_multiply,
)

_MAX_ABS_RATE_RAD_S = 1e6
_MAX_ABS_TORQUE_NM = 1e12
propagate_attitude_exponential_map_kernel = None


@dataclass
class AttitudeGuardrailStats:
    non_finite_input_events: int = 0
    rate_clamp_events: int = 0
    torque_clamp_events: int = 0
    non_finite_coriolis_events: int = 0
    singular_inertia_events: int = 0
    non_finite_output_events: int = 0
    policy: str = "sanitize"


_ATTITUDE_GUARDRAIL_CONTEXT: ContextVar[AttitudeGuardrailStats | None] = ContextVar(
    "oel_attitude_guardrail_stats",
    default=None,
)


def _current_attitude_guardrail_stats() -> AttitudeGuardrailStats:
    stats = _ATTITUDE_GUARDRAIL_CONTEXT.get()
    if stats is None:
        stats = AttitudeGuardrailStats()
        _ATTITUDE_GUARDRAIL_CONTEXT.set(stats)
    return stats


class _GuardrailStatsProxy:
    def __getattr__(self, name: str):
        return getattr(_current_attitude_guardrail_stats(), name)

    def __setattr__(self, name: str, value) -> None:
        stats = _current_attitude_guardrail_stats()
        previous = getattr(stats, name)
        setattr(stats, name, value)
        if name != "policy" and int(value) > int(previous) and str(stats.policy) == "error":
            raise FloatingPointError(f"Attitude numerical guardrail triggered: {name}.")


_ATTITUDE_GUARDRAIL_STATS = _GuardrailStatsProxy()


def new_attitude_guardrail_stats(*, policy: str = "error") -> AttitudeGuardrailStats:
    resolved = str(policy or "error").strip().lower()
    if resolved not in {"error", "sanitize"}:
        raise ValueError("Attitude guardrail policy must be 'error' or 'sanitize'.")
    return AttitudeGuardrailStats(policy=resolved)


def activate_attitude_guardrail_stats(stats: AttitudeGuardrailStats) -> None:
    _ATTITUDE_GUARDRAIL_CONTEXT.set(stats)


def reset_attitude_guardrail_stats(stats: AttitudeGuardrailStats | None = None) -> None:
    if stats is None:
        stats = new_attitude_guardrail_stats(policy="sanitize")
    activate_attitude_guardrail_stats(stats)
    _ATTITUDE_GUARDRAIL_STATS.non_finite_input_events = 0
    _ATTITUDE_GUARDRAIL_STATS.rate_clamp_events = 0
    _ATTITUDE_GUARDRAIL_STATS.torque_clamp_events = 0
    _ATTITUDE_GUARDRAIL_STATS.non_finite_coriolis_events = 0
    _ATTITUDE_GUARDRAIL_STATS.singular_inertia_events = 0
    _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events = 0


def get_attitude_guardrail_stats(stats: AttitudeGuardrailStats | None = None) -> dict[str, int | str]:
    current = _current_attitude_guardrail_stats() if stats is None else stats
    data = asdict(current)
    data.pop("policy", None)
    return data


def _add_guardrail_counts(counts: np.ndarray) -> None:
    values = np.asarray(counts, dtype=int).reshape(-1)
    if values.size < 6:
        return
    if not np.any(values[:6]):
        return
    _ATTITUDE_GUARDRAIL_STATS.non_finite_input_events += int(values[0])
    _ATTITUDE_GUARDRAIL_STATS.rate_clamp_events += int(values[1])
    _ATTITUDE_GUARDRAIL_STATS.torque_clamp_events += int(values[2])
    _ATTITUDE_GUARDRAIL_STATS.non_finite_coriolis_events += int(values[3])
    _ATTITUDE_GUARDRAIL_STATS.singular_inertia_events += int(values[4])
    _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += int(values[5])


def rigid_body_derivatives(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_raw = np.asarray(quat_bn, dtype=float).reshape(-1)
    if q_raw.size != 4 or not np.all(np.isfinite(q_raw)) or float(np.linalg.norm(q_raw)) <= 0.0:
        _ATTITUDE_GUARDRAIL_STATS.non_finite_input_events += 1
    q = normalize_quaternion(q_raw)
    w = np.asarray(omega_body_rad_s, dtype=float).reshape(3)
    inertia = np.asarray(inertia_kg_m2, dtype=float).reshape(3, 3)
    tau = np.asarray(torque_body_nm, dtype=float).reshape(3)

    # Clamp/sanitize extremes to keep attitude propagation numerically stable.
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(tau)) and np.all(np.isfinite(inertia))):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_input_events += 1
    w_nonfinite = ~np.isfinite(w)
    tau_nonfinite = ~np.isfinite(tau)
    w = np.nan_to_num(w, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    tau = np.nan_to_num(tau, nan=0.0, posinf=_MAX_ABS_TORQUE_NM, neginf=-_MAX_ABS_TORQUE_NM)
    _ATTITUDE_GUARDRAIL_STATS.rate_clamp_events += int(np.sum(w_nonfinite))
    _ATTITUDE_GUARDRAIL_STATS.torque_clamp_events += int(np.sum(tau_nonfinite))
    w_pre_clip = w.copy()
    tau_pre_clip = tau.copy()
    w = np.clip(w, -_MAX_ABS_RATE_RAD_S, _MAX_ABS_RATE_RAD_S)
    tau = np.clip(tau, -_MAX_ABS_TORQUE_NM, _MAX_ABS_TORQUE_NM)
    _ATTITUDE_GUARDRAIL_STATS.rate_clamp_events += int(np.sum(w != w_pre_clip))
    _ATTITUDE_GUARDRAIL_STATS.torque_clamp_events += int(np.sum(tau != tau_pre_clip))

    q_dot = 0.5 * omega_matrix(w) @ q
    Iw = inertia @ w
    coriolis = np.array(
        [
            w[1] * Iw[2] - w[2] * Iw[1],
            w[2] * Iw[0] - w[0] * Iw[2],
            w[0] * Iw[1] - w[1] * Iw[0],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(coriolis)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_coriolis_events += 1
    rhs = tau - np.nan_to_num(coriolis, nan=0.0, posinf=_MAX_ABS_TORQUE_NM, neginf=-_MAX_ABS_TORQUE_NM)
    try:
        omega_dot = np.linalg.solve(inertia, rhs)
    except np.linalg.LinAlgError:
        _ATTITUDE_GUARDRAIL_STATS.singular_inertia_events += 1
        omega_dot = np.zeros(3, dtype=float)
    if not np.all(np.isfinite(omega_dot)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    omega_dot = np.nan_to_num(omega_dot, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    return q_dot, omega_dot


def propagate_attitude_euler(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_dot, omega_dot = rigid_body_derivatives(quat_bn, omega_body_rad_s, inertia_kg_m2, torque_body_nm)
    dt = float(max(dt_s, 0.0))
    q_candidate = np.array(quat_bn, dtype=float).reshape(4) + dt * q_dot
    if not np.all(np.isfinite(q_candidate)) or float(np.linalg.norm(q_candidate)) <= 0.0:
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    q_next = normalize_quaternion(q_candidate)
    omega_next = np.array(omega_body_rad_s, dtype=float).reshape(3) + dt * omega_dot
    if not (np.all(np.isfinite(q_next)) and np.all(np.isfinite(omega_next))):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    omega_next = np.nan_to_num(omega_next, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    omega_next = np.clip(omega_next, -_MAX_ABS_RATE_RAD_S, _MAX_ABS_RATE_RAD_S)
    return q_next, omega_next


def propagate_attitude_exponential_map(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
    dt_s: float,
    acceleration_mode: str = "off",
) -> tuple[np.ndarray, np.ndarray]:
    global propagate_attitude_exponential_map_kernel
    if acceleration_enabled_from_mode(acceleration_mode):
        if propagate_attitude_exponential_map_kernel is None:
            from sim.acceleration.kernels.attitude import (
                propagate_attitude_exponential_map_kernel as accelerated_propagate,
            )

            propagate_attitude_exponential_map_kernel = accelerated_propagate
        q_next, omega_next, counts = propagate_attitude_exponential_map_kernel(
            np.asarray(quat_bn, dtype=float).reshape(4),
            np.asarray(omega_body_rad_s, dtype=float).reshape(3),
            np.asarray(inertia_kg_m2, dtype=float).reshape(3, 3),
            np.asarray(torque_body_nm, dtype=float).reshape(3),
            float(dt_s),
        )
        _add_guardrail_counts(counts)
        return q_next, omega_next

    # Integrate angular-rate dynamics with first-order step.
    _, omega_dot = rigid_body_derivatives(quat_bn, omega_body_rad_s, inertia_kg_m2, torque_body_nm)
    dt = float(max(dt_s, 0.0))
    omega_now = np.asarray(omega_body_rad_s, dtype=float).reshape(3)
    omega_next = omega_now + dt * omega_dot
    if not np.all(np.isfinite(omega_next)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    omega_next = np.nan_to_num(omega_next, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    omega_next = np.clip(omega_next, -_MAX_ABS_RATE_RAD_S, _MAX_ABS_RATE_RAD_S)

    # Use midpoint body rate to build quaternion delta via exponential map.
    omega_mid = omega_now + 0.5 * dt * omega_dot
    if not np.all(np.isfinite(omega_mid)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    omega_mid = np.nan_to_num(omega_mid, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    omega_mid = np.clip(omega_mid, -_MAX_ABS_RATE_RAD_S, _MAX_ABS_RATE_RAD_S)
    dq = quaternion_delta_from_body_rate(omega_mid, dt)
    # q_dot uses Omega(w) @ q with the convention equivalent to q ⊗ [0, w],
    # so the finite update must right-multiply by dq.
    q_next = normalize_quaternion(quaternion_multiply(quat_bn, dq))
    if not np.all(np.isfinite(q_next)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    return q_next, omega_next
