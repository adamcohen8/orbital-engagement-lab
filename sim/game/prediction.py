# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *

def _cw_coast_state(x0: np.ndarray, t_s: float, mean_motion_rad_s: float) -> np.ndarray:
    x, y, z, xd, yd, zd = np.array(x0, dtype=float).reshape(6)
    n = float(mean_motion_rad_s)
    t = float(t_s)
    nt = n * t
    c = float(np.cos(nt))
    s = float(np.sin(nt))
    if abs(n) <= 1.0e-12:
        return np.array([x + xd * t, y + yd * t, z + zd * t, xd, yd, zd], dtype=float)

    xp = (4.0 - 3.0 * c) * x + (s / n) * xd + (2.0 * (1.0 - c) / n) * yd
    yp = 6.0 * (s - nt) * x + y - (2.0 * (1.0 - c) / n) * xd + ((4.0 * s - 3.0 * nt) / n) * yd
    zp = c * z + (s / n) * zd
    xdp = 3.0 * n * s * x + c * xd + 2.0 * s * yd
    ydp = -6.0 * n * (1.0 - c) * x - 2.0 * s * xd + (4.0 * c - 3.0) * yd
    zdp = -n * s * z + c * zd
    return np.array([xp, yp, zp, xdp, ydp, zdp], dtype=float)


def _cw_coast_states(x0: np.ndarray, times_s: np.ndarray, mean_motion_rad_s: float) -> np.ndarray:
    x, y, z, xd, yd, zd = np.array(x0, dtype=float).reshape(6)
    times = np.array(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float)
    n = float(mean_motion_rad_s)
    if abs(n) <= 1.0e-12:
        out = np.empty((times.size, 6), dtype=float)
        out[:, 0] = x + xd * times
        out[:, 1] = y + yd * times
        out[:, 2] = z + zd * times
        out[:, 3] = xd
        out[:, 4] = yd
        out[:, 5] = zd
        return out
    nt = n * times
    c = np.cos(nt)
    s = np.sin(nt)
    one_minus_c = 1.0 - c
    out = np.empty((times.size, 6), dtype=float)
    out[:, 0] = (4.0 - 3.0 * c) * x + (s / n) * xd + (2.0 * one_minus_c / n) * yd
    out[:, 1] = 6.0 * (s - nt) * x + y - (2.0 * one_minus_c / n) * xd + ((4.0 * s - 3.0 * nt) / n) * yd
    out[:, 2] = c * z + (s / n) * zd
    out[:, 3] = 3.0 * n * s * x + c * xd + 2.0 * s * yd
    out[:, 4] = -6.0 * n * one_minus_c * x - 2.0 * s * xd + (4.0 * c - 3.0) * yd
    out[:, 5] = -n * s * z + c * zd
    return out


def _satellite_marker_size_px(
    scale_x_px_per_km: float,
    scale_y_px_per_km: float,
    *,
    diameter_km: float = SATELLITE_SPRITE_DIAMETER_KM,
) -> int:
    raw_px = float(max(abs(float(scale_x_px_per_km)), abs(float(scale_y_px_per_km)))) * float(max(diameter_km, 0.0))
    if not np.isfinite(raw_px) or raw_px <= 0.0:
        return 0
    return max(int(round(raw_px)), 1)


def _satellite_marker_reticle_radii_px(sprite_size_px: int) -> tuple[int, int]:
    size = int(max(sprite_size_px, 0))
    if size <= 0:
        return 0, 0
    if size < 24:
        return 2, 4
    dot_radius = max(2, min(3, int(round(size * 0.08))))
    ring_radius = max(dot_radius + 2, min(6, int(round(size * 0.18))))
    return dot_radius, ring_radius


def _cw_forced_state(
    x0: np.ndarray,
    accel_ric_km_s2: np.ndarray,
    t_s: float,
    mean_motion_rad_s: float,
    *,
    substep_s: float = 0.1,
) -> np.ndarray:
    state = np.array(x0, dtype=float).reshape(6).copy()
    accel = np.array(accel_ric_km_s2, dtype=float).reshape(3)
    duration = float(max(t_s, 0.0))
    n = float(mean_motion_rad_s)
    if duration <= 0.0 or not np.all(np.isfinite(state)) or not np.all(np.isfinite(accel)):
        return state
    if not np.isfinite(n) or n <= 0.0:
        state[:3] += state[3:6] * duration + 0.5 * accel * duration * duration
        state[3:6] += accel * duration
        return state
    step = float(max(substep_s, 1.0e-6))
    elapsed = 0.0
    while elapsed < duration - 1.0e-12:
        dt = min(step, duration - elapsed)
        r, i, c, rd, idot, cd = state
        rdd = 3.0 * n * n * r + 2.0 * n * idot + accel[0]
        idd = -2.0 * n * rd + accel[1]
        cdd = -n * n * c + accel[2]
        state[3] = rd + rdd * dt
        state[4] = idot + idd * dt
        state[5] = cd + cdd * dt
        state[0] = r + state[3] * dt
        state[1] = i + state[4] * dt
        state[2] = c + state[5] * dt
        elapsed += dt
    return state


def _coast_prediction_model_key(value: str) -> str:
    key = str(value or "hcw").strip().lower().replace("-", "_")
    aliases = {
        "cw": "hcw",
        "cislunar": "cislunar",
        "cislunar_l1": "cislunar_l1",
        "cr3bp": "cr3bp",
        "cr3bp_rotating": "cr3bp",
        "tschauner_hempel": "tschauner_hempel",
        "th": "tschauner_hempel",
        "ts": "ts",
        "elliptic": "elliptic_linear",
        "elliptical": "elliptic_linear",
        "elliptic_linear": "elliptic_linear",
    }
    return aliases.get(key, key or "hcw")


def _cr3bp_projection_mode_key(value: str) -> str:
    key = str(value or "nonlinear").strip().lower().replace("-", "_")
    if key in {"linear", "linearized", "stm", "variational"}:
        return "linearized"
    return "nonlinear"


def _cr3bp_coast_prediction_horizon_mode_key(value: str) -> str:
    key = str(value or "default").strip().lower().replace("-", "_")
    if key in {"time_remaining", "remaining_time", "mission_remaining", "mission_time_remaining"}:
        return "time_remaining"
    return "default"


def _relative_frame_key(value: str) -> str:
    key = str(value or "ric").strip().lower().replace("-", "_")
    if key in {"cislunar", "cislunar_l1", "earth_moon_rotating", "cr3bp", "cr3bp_rotating"}:
        return "cislunar_l1"
    if key in {"moon_ric", "lunar_ric", "target_moon_ric", "target_lunar_ric"}:
        return "moon_ric"
    return "ric"


def _cr3bp_state_to_moon_ric_rect(deputy_state: np.ndarray, chief_state: np.ndarray) -> np.ndarray:
    moon = cr3bp_moon_state_km_s()
    deputy = np.array(deputy_state, dtype=float).reshape(6) - moon
    chief = np.array(chief_state, dtype=float).reshape(6) - moon
    return eci_relative_to_ric_rect(deputy, chief)


def _moon_ric_basis_rows(chief_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moon = cr3bp_moon_state_km_s()
    chief_rows = np.array(chief_states, dtype=float).reshape(-1, 6) - moon
    r = chief_rows[:, :3]
    v = chief_rows[:, 3:]
    r_norm = np.maximum(np.linalg.norm(r, axis=1), 1.0e-12)
    r_hat = r / r_norm[:, None]
    h = np.cross(r, v)
    h_norm = np.maximum(np.linalg.norm(h, axis=1), 1.0e-12)
    c_hat = h / h_norm[:, None]
    i_hat = np.cross(c_hat, r_hat)
    i_norm = np.maximum(np.linalg.norm(i_hat, axis=1), 1.0e-12)
    i_hat = i_hat / i_norm[:, None]
    axes = np.stack((r_hat, i_hat, c_hat), axis=2)
    omega = h / np.maximum(np.sum(r * r, axis=1), 1.0e-12)[:, None]
    return axes, omega


def _cr3bp_states_to_moon_ric_rect_rows(
    deputy_states: np.ndarray,
    chief_states: np.ndarray,
    *,
    basis_axes: np.ndarray | None = None,
    basis_omega: np.ndarray | None = None,
) -> np.ndarray:
    deputy_rows = np.array(deputy_states, dtype=float).reshape(-1, 6)
    chief_rows = np.array(chief_states, dtype=float).reshape(-1, 6)
    if deputy_rows.shape[0] != chief_rows.shape[0]:
        raise ValueError("deputy_states and chief_states must have matching row counts")
    if deputy_rows.size == 0:
        return np.empty((0, 6), dtype=float)
    axes = None if basis_axes is None else np.array(basis_axes, dtype=float).reshape(-1, 3, 3)
    omega = None if basis_omega is None else np.array(basis_omega, dtype=float).reshape(-1, 3)
    if axes is None or omega is None or axes.shape[0] != chief_rows.shape[0] or omega.shape[0] != chief_rows.shape[0]:
        axes, omega = _moon_ric_basis_rows(chief_rows)
    dr_eci = deputy_rows[:, :3] - chief_rows[:, :3]
    dv_eci = deputy_rows[:, 3:] - chief_rows[:, 3:]
    omega_cross_dr = np.cross(omega, dr_eci)
    dr_ric = np.einsum("nji,nj->ni", axes, dr_eci)
    dv_ric = np.einsum("nji,nj->ni", axes, dv_eci - omega_cross_dr)
    return np.hstack((dr_ric, dv_ric))


def _moon_ric_rect_state_to_cr3bp(rel_moon_ric: np.ndarray, chief_state: np.ndarray) -> np.ndarray:
    moon = cr3bp_moon_state_km_s()
    chief_abs = np.array(chief_state, dtype=float).reshape(6)
    chief_moon = chief_abs - moon
    deputy_moon = ric_rect_state_to_eci(
        np.array(rel_moon_ric, dtype=float).reshape(6),
        chief_moon[:3],
        chief_moon[3:],
    )
    return deputy_moon + moon


def _nonlinear_cr3bp_moon_ric_coast_prediction(
    rel0: np.ndarray,
    *,
    target_state: np.ndarray,
    times: np.ndarray,
    current_t_s: float,
) -> np.ndarray:
    reference = np.array(target_state, dtype=float).reshape(6)
    deputy = _moon_ric_rect_state_to_cr3bp(rel0, reference)
    rows: list[np.ndarray] = []
    current_t = float(current_t_s)
    previous_t = 0.0
    for target_t in np.array(times, dtype=float).reshape(-1):
        step_s = float(target_t - previous_t)
        if step_s > 0.0:
            deputy = _dashboard_dep("propagate_cr3bp_state", propagate_cr3bp_state)(deputy, step_s, current_t)
            reference = _dashboard_dep("propagate_cr3bp_state", propagate_cr3bp_state)(reference, step_s, current_t)
            current_t += step_s
        rows.append(_cr3bp_state_to_moon_ric_rect(deputy, reference))
        previous_t = float(target_t)
    return np.vstack(rows) if rows else np.empty((0, 6), dtype=float)


def _linearized_cr3bp_moon_ric_coast_prediction(
    rel0: np.ndarray,
    *,
    target_state: np.ndarray,
    times: np.ndarray,
    current_t_s: float,
) -> np.ndarray:
    reference = np.array(target_state, dtype=float).reshape(6)
    deputy0 = _moon_ric_rect_state_to_cr3bp(rel0, reference)
    delta0 = deputy0 - reference
    stm = np.eye(6, dtype=float)
    rows: list[np.ndarray] = []
    current_t = float(current_t_s)
    previous_t = 0.0
    for target_t in np.array(times, dtype=float).reshape(-1):
        step_s = float(target_t - previous_t)
        if step_s > 0.0:
            reference, stm = _dashboard_dep("propagate_cr3bp_reference_stm", propagate_cr3bp_reference_stm)(reference, stm, step_s, current_t)
            current_t += step_s
        deputy_linear = reference + stm @ delta0
        rows.append(_cr3bp_state_to_moon_ric_rect(deputy_linear, reference))
        previous_t = float(target_t)
    return np.vstack(rows) if rows else np.empty((0, 6), dtype=float)


def _linearized_cr3bp_moon_ric_stm_table(
    *,
    target_state: np.ndarray,
    times: np.ndarray,
    current_t_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array(target_state, dtype=float).reshape(6)
    stm = np.eye(6, dtype=float)
    references: list[np.ndarray] = []
    stms: list[np.ndarray] = []
    current_t = float(current_t_s)
    previous_t = 0.0
    for target_t in np.array(times, dtype=float).reshape(-1):
        step_s = float(target_t - previous_t)
        if step_s > 0.0:
            reference, stm = _dashboard_dep("propagate_cr3bp_reference_stm", propagate_cr3bp_reference_stm)(reference, stm, step_s, current_t)
            current_t += step_s
        references.append(reference.copy())
        stms.append(stm.copy())
        previous_t = float(target_t)
    if not references:
        return np.empty((0, 6), dtype=float), np.empty((0, 6, 6), dtype=float)
    return np.vstack(references), np.stack(stms, axis=0)


def _linearized_cr3bp_moon_ric_projection_from_stm_table(
    rel0: np.ndarray,
    *,
    target_state: np.ndarray,
    references: np.ndarray,
    stms: np.ndarray,
    basis_axes: np.ndarray | None = None,
    basis_omega: np.ndarray | None = None,
) -> np.ndarray:
    reference0 = np.array(target_state, dtype=float).reshape(6)
    reference_rows = np.array(references, dtype=float).reshape(-1, 6)
    stm_rows = np.array(stms, dtype=float).reshape(-1, 6, 6)
    if reference_rows.size == 0 or stm_rows.size == 0:
        return np.empty((0, 6), dtype=float)
    deputy0 = _moon_ric_rect_state_to_cr3bp(rel0, reference0)
    delta0 = deputy0 - reference0
    deputy_rows = reference_rows + np.einsum("nij,j->ni", stm_rows, delta0)
    return _cr3bp_states_to_moon_ric_rect_rows(
        deputy_rows,
        reference_rows,
        basis_axes=basis_axes,
        basis_omega=basis_omega,
    )

def _elliptic_linear_coast_states(
    rel0_ric: np.ndarray,
    times_s: np.ndarray,
    chief_state_eci: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> np.ndarray:
    """Propagate linearized RIC relative motion along a two-body elliptic chief.

    This is the numerical form of the Tschauner-Hempel idea used for teaching
    overlays: the relative state remains linearized, but the chief radius and
    angular rate vary along the orbit instead of being frozen as in HCW.
    """

    rel = np.array(rel0_ric, dtype=float).reshape(6)
    chief = np.array(chief_state_eci, dtype=float).reshape(6)
    times = np.array(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float)
    order = np.argsort(times)
    sorted_times = times[order]
    state = np.hstack((chief, rel)).astype(float)
    rows = np.zeros((times.size, 6), dtype=float)
    current_t = 0.0
    max_step_s = max(float(np.max(np.diff(sorted_times))) if sorted_times.size > 1 else 0.0, 1.0)
    max_step_s = min(max(max_step_s, 1.0), 60.0)
    for sorted_idx, target_t in enumerate(sorted_times):
        target = float(max(target_t, current_t))
        while current_t < target:
            h = min(max_step_s, target - current_t)
            state = _rk4_step(_elliptic_linear_derivative, state, h, float(mu_km3_s2))
            current_t += h
        rows[order[sorted_idx]] = state[6:12]
    return rows


def _elliptic_ya_coast_states(
    rel0_ric: np.ndarray,
    times_s: np.ndarray,
    chief_state_eci: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> np.ndarray:
    """Propagate elliptic-chief RIC relative motion with the closed-form YA STM."""

    rel = np.array(rel0_ric, dtype=float).reshape(6)
    chief0 = np.array(chief_state_eci, dtype=float).reshape(6)
    times = np.array(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float)
    order = np.argsort(times)
    sorted_times = times[order]
    chief = chief0.copy()
    rows = np.zeros((times.size, 6), dtype=float)
    current_t = 0.0
    for sorted_idx, target_t in enumerate(sorted_times):
        target = float(max(float(target_t), current_t))
        duration_s = target - current_t
        if duration_s > 0.0:
            chief = _two_body_coast_state(chief, duration_s, mu_km3_s2=float(mu_km3_s2))
            current_t = target
        phi = _dashboard_dep("ya_closed_form_transition_matrix", ya_closed_form_transition_matrix)(target, chief0, chief, mu_km3_s2=float(mu_km3_s2))
        rows[order[sorted_idx]] = phi @ rel
    return rows


def _elliptic_reference_cache_valid(
    cached_reference_eci: Any,
    current_reference_eci: Any,
    *,
    elapsed_s: float,
) -> bool:
    if cached_reference_eci is None or current_reference_eci is None:
        return cached_reference_eci is None and current_reference_eci is None
    try:
        cached = np.array(cached_reference_eci, dtype=float).reshape(6)
        current = np.array(current_reference_eci, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(cached)) or not np.all(np.isfinite(current)):
        return False
    if float(elapsed_s) <= 0.0:
        expected = cached
    else:
        expected = _two_body_coast_state(cached, float(elapsed_s))
    pos_error_km = float(np.linalg.norm(current[:3] - expected[:3]))
    vel_error_km_s = float(np.linalg.norm(current[3:6] - expected[3:6]))
    return bool(
        pos_error_km <= ELLIPTIC_REFERENCE_CACHE_POSITION_TOL_KM
        and vel_error_km_s <= ELLIPTIC_REFERENCE_CACHE_VELOCITY_TOL_KM_S
    )


def _cr3bp_reference_cache_valid(cached_reference: Any, current_reference: Any, *, elapsed_s: float = 0.0) -> bool:
    if cached_reference is None or current_reference is None:
        return cached_reference is None and current_reference is None
    try:
        cached = np.array(cached_reference, dtype=float).reshape(6)
        current = np.array(current_reference, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(cached)) or not np.all(np.isfinite(current)):
        return False
    expected = cached
    if float(elapsed_s) > 0.0:
        try:
            expected = _dashboard_dep("propagate_cr3bp_state", propagate_cr3bp_state)(cached, float(elapsed_s), 0.0)
        except Exception:
            expected = cached
    pos_error_km = float(np.linalg.norm(current[:3] - expected[:3]))
    vel_error_km_s = float(np.linalg.norm(current[3:6] - expected[3:6]))
    return bool(
        pos_error_km <= CR3BP_REFERENCE_CACHE_POSITION_TOL_KM
        and vel_error_km_s <= CR3BP_REFERENCE_CACHE_VELOCITY_TOL_KM_S
    )


def _cr3bp_relative_cache_valid(cached_rel0: Any, current_rel0: np.ndarray) -> bool:
    try:
        cached = np.array(cached_rel0, dtype=float).reshape(6)
        current = np.array(current_rel0, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(cached)) or not np.all(np.isfinite(current)):
        return False
    pos_error_km = float(np.linalg.norm(current[:3] - cached[:3]))
    vel_error_km_s = float(np.linalg.norm(current[3:6] - cached[3:6]))
    return bool(
        pos_error_km <= CR3BP_RELATIVE_CACHE_POSITION_TOL_KM
        and vel_error_km_s <= CR3BP_RELATIVE_CACHE_VELOCITY_TOL_KM_S
    )


def _two_body_coast_state(
    state_eci: np.ndarray,
    duration_s: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> np.ndarray:
    state = np.array(state_eci, dtype=float).reshape(6)
    duration = float(max(duration_s, 0.0))
    if duration <= 0.0:
        return state.copy()
    current_t = 0.0
    step_s = min(max(duration / 4.0, 1.0), 10.0)
    out = state.astype(float)
    while current_t < duration:
        h = min(step_s, duration - current_t)
        out = _rk4_step(_two_body_derivative, out, h, float(mu_km3_s2))
        current_t += h
    return out


def _two_body_derivative(state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    r = np.array(state[:3], dtype=float)
    v = np.array(state[3:6], dtype=float)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 1.0e-9 or not np.isfinite(r_norm):
        return np.zeros_like(state)
    acc = -float(mu_km3_s2) * r / (r_norm**3)
    return np.hstack((v, acc))


def _rk4_step(func: Any, state: np.ndarray, step_s: float, mu_km3_s2: float) -> np.ndarray:
    h = float(step_s)
    k1 = func(state, mu_km3_s2)
    k2 = func(state + 0.5 * h * k1, mu_km3_s2)
    k3 = func(state + 0.5 * h * k2, mu_km3_s2)
    k4 = func(state + h * k3, mu_km3_s2)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _elliptic_linear_derivative(state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    chief_r = np.array(state[:3], dtype=float)
    chief_v = np.array(state[3:6], dtype=float)
    rho = np.array(state[6:9], dtype=float)
    rho_dot = np.array(state[9:12], dtype=float)
    r_norm = float(np.linalg.norm(chief_r))
    if r_norm <= 1.0e-9 or not np.isfinite(r_norm):
        return np.zeros_like(state)
    h_vec = np.cross(chief_r, chief_v)
    h_norm = float(np.linalg.norm(h_vec))
    theta_dot = h_norm / max(r_norm * r_norm, 1.0e-12)
    radial_rate = float(np.dot(chief_r, chief_v)) / r_norm
    theta_ddot = -2.0 * theta_dot * radial_rate / r_norm
    omega = np.array([0.0, 0.0, theta_dot], dtype=float)
    omega_dot = np.array([0.0, 0.0, theta_ddot], dtype=float)
    gravity_gradient = (float(mu_km3_s2) / (r_norm**3)) * np.array([2.0 * rho[0], -rho[1], -rho[2]])
    rho_ddot = (
        gravity_gradient
        - 2.0 * np.cross(omega, rho_dot)
        - np.cross(omega_dot, rho)
        - np.cross(omega, np.cross(omega, rho))
    )
    chief_acc = -float(mu_km3_s2) * chief_r / (r_norm**3)
    return np.hstack((chief_v, chief_acc, rho_dot, rho_ddot))

__all__ = [name for name in globals() if not name.startswith("__")]
