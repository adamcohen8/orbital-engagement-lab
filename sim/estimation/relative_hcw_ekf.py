from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief

HCW_MEASUREMENT_MODELS = {
    "relative_state",
    "relative_range",
    "relative_range_rate",
    "relative_angles",
    "relative_angles_range",
    "relative_angles_range_rate",
}


@dataclass(frozen=True)
class HCWRelativeEKFUpdateDiagnostics:
    measurement_available: bool
    update_applied: bool
    measurement_model: str = "relative_state"
    innovation: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))
    innovation_covariance: np.ndarray = field(default_factory=lambda: np.full((6, 6), np.nan))
    nis: float = float("nan")
    predicted_cov_trace: float = float("nan")
    posterior_cov_trace: float = float("nan")


@dataclass
class HCWRelativeEKFEstimator(Estimator):
    """EKF over rectangular RIC relative state using analytic HCW propagation.

    The native state is deputy relative to chief in the chief-centered RIC frame:
    [R, I, C, Rdot, Idot, Cdot], using km and km/s.
    """

    mean_motion_rad_s: float
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    measurement_model: str = "relative_state"
    measurement_origin: str = "chief"
    last_update_diagnostics: HCWRelativeEKFUpdateDiagnostics | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mean_motion_rad_s <= 0.0:
            raise ValueError("mean_motion_rad_s must be positive.")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        self.process_noise_diag = _diag6(self.process_noise_diag, "process_noise_diag")
        self.measurement_model = normalize_hcw_measurement_model(self.measurement_model)
        self.measurement_origin = _normalize_measurement_origin(self.measurement_origin)
        meas_dim = hcw_measurement_dimension(self.measurement_model)
        meas_noise = np.array(self.meas_noise_diag, dtype=float).reshape(-1)
        if meas_noise.size == 6 and meas_dim != 6:
            meas_noise = _default_measurement_noise_for_model(self.measurement_model, meas_noise)
        if meas_noise.size != meas_dim:
            raise ValueError(
                f"meas_noise_diag must be length-{meas_dim} for measurement_model={self.measurement_model!r}."
            )
        if np.any(meas_noise < 0.0):
            raise ValueError("meas_noise_diag must be non-negative.")
        self.meas_noise_diag = meas_noise

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        output_t_s = float(t_s)
        meas_t_s = output_t_s
        if measurement is not None:
            meas_t_s = float(np.clip(float(measurement.t_s), float(belief.last_update_t_s), output_t_s))

        x_pred, p_pred = self._predict(belief.state, belief.covariance, from_t_s=belief.last_update_t_s, to_t_s=meas_t_s)

        if measurement is None:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            self.last_update_diagnostics = HCWRelativeEKFUpdateDiagnostics(
                measurement_available=False,
                update_applied=False,
                measurement_model=self.measurement_model,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)

        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        h_pred = hcw_measurement_vector(
            self.measurement_model,
            x_pred,
            measurement_origin=self.measurement_origin,
        )
        if z.size < h_pred.size:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            self.last_update_diagnostics = HCWRelativeEKFUpdateDiagnostics(
                measurement_available=True,
                update_applied=False,
                measurement_model=self.measurement_model,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)
        z = z[: h_pred.size]
        h_jac = hcw_measurement_jacobian(
            self.measurement_model,
            x_pred,
            measurement_origin=self.measurement_origin,
        )
        r = np.diag(self.meas_noise_diag)
        innovation = _measurement_innovation(self.measurement_model, z, h_pred)
        s_mat = h_jac @ p_pred @ h_jac.T + r
        hp_t = p_pred @ h_jac.T
        try:
            k_gain = np.linalg.solve(s_mat.T, hp_t.T).T
            s_y = np.linalg.solve(s_mat, innovation)
        except np.linalg.LinAlgError:
            s_pinv = np.linalg.pinv(s_mat)
            k_gain = hp_t @ s_pinv
            s_y = s_pinv @ innovation
        x_upd = x_pred + k_gain @ innovation
        i_kh = np.eye(6) - k_gain @ h_jac
        p_upd = i_kh @ p_pred @ i_kh.T + k_gain @ r @ k_gain.T
        p_upd = 0.5 * (p_upd + p_upd.T)
        self.last_update_diagnostics = HCWRelativeEKFUpdateDiagnostics(
            measurement_available=True,
            update_applied=True,
            measurement_model=self.measurement_model,
            innovation=np.array(innovation, dtype=float),
            innovation_covariance=np.array(s_mat, dtype=float),
            nis=float(innovation.T @ s_y),
            predicted_cov_trace=float(np.trace(p_pred)),
            posterior_cov_trace=float(np.trace(p_upd)),
        )
        if meas_t_s < output_t_s:
            x_upd, p_upd = self._predict(x_upd, p_upd, from_t_s=meas_t_s, to_t_s=output_t_s)
        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=output_t_s)

    def _predict(
        self,
        x_prev: np.ndarray,
        p_prev: np.ndarray,
        *,
        from_t_s: float,
        to_t_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt_s = max(float(to_t_s) - float(from_t_s), 0.0)
        phi = hcw_state_transition_matrix(float(self.mean_motion_rad_s), dt_s)
        x = np.asarray(x_prev, dtype=float).reshape(6)
        p = np.asarray(p_prev, dtype=float).reshape(6, 6)
        q_scale = dt_s / self.dt_s if self.dt_s > 0.0 else 1.0
        p_pred = phi @ p @ phi.T + np.diag(self.process_noise_diag) * max(q_scale, 0.0)
        return phi @ x, 0.5 * (p_pred + p_pred.T)


def hcw_state_transition_matrix(mean_motion_rad_s: float, dt_s: float) -> np.ndarray:
    n = float(mean_motion_rad_s)
    if n <= 0.0:
        raise ValueError("mean_motion_rad_s must be positive.")
    t = float(dt_s)
    nt = n * t
    c = float(np.cos(nt))
    s = float(np.sin(nt))
    return np.array(
        [
            [4.0 - 3.0 * c, 0.0, 0.0, s / n, 2.0 * (1.0 - c) / n, 0.0],
            [6.0 * (s - nt), 1.0, 0.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, c, 0.0, 0.0, s / n],
            [3.0 * n * s, 0.0, 0.0, c, 2.0 * s, 0.0],
            [6.0 * n * (c - 1.0), 0.0, 0.0, -2.0 * s, 4.0 * c - 3.0, 0.0],
            [0.0, 0.0, -n * s, 0.0, 0.0, c],
        ],
        dtype=float,
    )


def normalize_hcw_measurement_model(model: str) -> str:
    raw = str(model or "relative_state").strip().lower().replace("-", "_")
    aliases = {
        "state": "relative_state",
        "full_state": "relative_state",
        "ric_state": "relative_state",
        "range": "relative_range",
        "range_rate": "relative_range_rate",
        "angles": "relative_angles",
        "angles_range": "relative_angles_range",
        "angles_range_rate": "relative_angles_range_rate",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in HCW_MEASUREMENT_MODELS:
        valid = ", ".join(sorted(HCW_MEASUREMENT_MODELS))
        raise ValueError(f"Unsupported HCW measurement_model '{model}'. Valid options: {valid}")
    return normalized


def hcw_measurement_dimension(model: str) -> int:
    normalized = normalize_hcw_measurement_model(model)
    if normalized == "relative_state":
        return 6
    if normalized == "relative_range":
        return 1
    if normalized in {"relative_range_rate", "relative_angles"}:
        return 2
    if normalized == "relative_angles_range":
        return 3
    if normalized == "relative_angles_range_rate":
        return 4
    raise ValueError(f"Unsupported HCW measurement_model '{model}'.")


def hcw_measurement_vector(model: str, state: np.ndarray, *, measurement_origin: str = "chief") -> np.ndarray:
    normalized = normalize_hcw_measurement_model(model)
    origin = _normalize_measurement_origin(measurement_origin)
    x = np.asarray(state, dtype=float).reshape(6)
    sign = 1.0 if origin == "chief" else -1.0
    rel_r = sign * x[:3]
    rel_v = sign * x[3:]
    rng_km = float(np.linalg.norm(rel_r))
    if rng_km <= 0.0:
        los = np.zeros(3, dtype=float)
        range_rate = 0.0
    else:
        los = rel_r / rng_km
        range_rate = float(np.dot(rel_v, los))
    az = float(np.arctan2(los[1], los[0])) if rng_km > 0.0 else 0.0
    el = float(np.arcsin(np.clip(los[2], -1.0, 1.0))) if rng_km > 0.0 else 0.0
    if normalized == "relative_state":
        return sign * x
    if normalized == "relative_range":
        return np.array([rng_km], dtype=float)
    if normalized == "relative_range_rate":
        return np.array([rng_km, range_rate], dtype=float)
    if normalized == "relative_angles":
        return np.array([az, el], dtype=float)
    if normalized == "relative_angles_range":
        return np.array([az, el, rng_km], dtype=float)
    if normalized == "relative_angles_range_rate":
        return np.array([az, el, rng_km, range_rate], dtype=float)
    raise ValueError(f"Unsupported HCW measurement_model '{model}'.")


def hcw_measurement_jacobian(model: str, state: np.ndarray, *, measurement_origin: str = "chief") -> np.ndarray:
    normalized = normalize_hcw_measurement_model(model)
    if normalized == "relative_state":
        sign = 1.0 if _normalize_measurement_origin(measurement_origin) == "chief" else -1.0
        return sign * np.eye(6)
    x = np.asarray(state, dtype=float).reshape(6)
    h0 = hcw_measurement_vector(normalized, x, measurement_origin=measurement_origin)
    jac = np.zeros((h0.size, 6), dtype=float)
    eps = np.array([1e-5, 1e-5, 1e-5, 1e-8, 1e-8, 1e-8], dtype=float)
    for i in range(6):
        xp = x.copy()
        xp[i] += eps[i]
        hp = hcw_measurement_vector(normalized, xp, measurement_origin=measurement_origin)
        jac[:, i] = _measurement_innovation(normalized, hp, h0) / eps[i]
    return jac


def _diag6(value: np.ndarray, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"{field_name} must be length-6.")
    if np.any(arr < 0.0):
        raise ValueError(f"{field_name} must be non-negative.")
    return arr


def _normalize_measurement_origin(value: str) -> str:
    raw = str(value or "chief").strip().lower().replace("-", "_")
    aliases = {
        "chief": "chief",
        "target": "chief",
        "deputy_from_chief": "chief",
        "observer": "deputy",
        "deputy": "deputy",
        "chaser": "deputy",
        "chief_from_deputy": "deputy",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"chief", "deputy"}:
        raise ValueError("measurement_origin must be 'chief' or 'deputy'.")
    return normalized


def _default_measurement_noise_for_model(model: str, full_state_diag: np.ndarray) -> np.ndarray:
    pos_var = float(np.mean(np.asarray(full_state_diag[:3], dtype=float)))
    vel_var = float(np.mean(np.asarray(full_state_diag[3:6], dtype=float)))
    angle_var = max(pos_var, 1e-18)
    normalized = normalize_hcw_measurement_model(model)
    if normalized == "relative_range":
        return np.array([pos_var], dtype=float)
    if normalized == "relative_range_rate":
        return np.array([pos_var, vel_var], dtype=float)
    if normalized == "relative_angles":
        return np.array([angle_var, angle_var], dtype=float)
    if normalized == "relative_angles_range":
        return np.array([angle_var, angle_var, pos_var], dtype=float)
    if normalized == "relative_angles_range_rate":
        return np.array([angle_var, angle_var, pos_var, vel_var], dtype=float)
    return np.asarray(full_state_diag, dtype=float).reshape(6)


def _measurement_innovation(model: str, z: np.ndarray, h: np.ndarray) -> np.ndarray:
    innovation = np.asarray(z, dtype=float).reshape(-1) - np.asarray(h, dtype=float).reshape(-1)
    normalized = normalize_hcw_measurement_model(model)
    if normalized in {"relative_angles", "relative_angles_range", "relative_angles_range_rate"}:
        innovation[0] = _wrap_angle_rad(float(innovation[0]))
        innovation[1] = _wrap_angle_rad(float(innovation[1]))
    return innovation


def _wrap_angle_rad(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)
