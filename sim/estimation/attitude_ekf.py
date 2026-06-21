from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.acceleration.kernels.estimation import (
    attitude_ekf_numerical_jacobian_kernel,
    attitude_ekf_propagate_state_kernel,
)
from sim.acceleration.settings import acceleration_settings_from_mode
from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.utils.quaternion import normalize_quaternion, omega_matrix


@dataclass
class AttitudeEKFEstimator(Estimator):
    dt_s: float
    inertia_kg_m2: np.ndarray
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    acceleration_mode: str = "off"
    _acceleration_enabled_value: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.inertia_kg_m2 = np.asarray(self.inertia_kg_m2, dtype=float).reshape(3, 3)
        self.process_noise_diag = np.asarray(self.process_noise_diag, dtype=float).reshape(7)
        self.meas_noise_diag = np.asarray(self.meas_noise_diag, dtype=float).reshape(7)
        self._acceleration_enabled_value = bool(acceleration_settings_from_mode(self.acceleration_mode).enabled)

    def _acceleration_enabled(self) -> bool:
        return self._acceleration_enabled_value

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        output_t_s = float(t_s)
        meas_t_s = output_t_s
        if measurement is not None:
            meas_t_s = float(np.clip(float(measurement.t_s), float(belief.last_update_t_s), output_t_s))

        x_pred, p_pred = self._predict(belief.state, belief.covariance, from_t_s=belief.last_update_t_s, to_t_s=meas_t_s)

        if measurement is None:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)

        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        if z.size < 7:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)
        z = z[:7].copy()
        z[:4] = normalize_quaternion(z[:4])
        if np.dot(z[:4], x_pred[:4]) < 0.0:
            z[:4] *= -1.0

        h = np.eye(7)
        r = np.diag(self.meas_noise_diag)
        y = z - h @ x_pred
        s = h @ p_pred @ h.T + r
        hp_t = p_pred @ h.T
        try:
            k = np.linalg.solve(s.T, hp_t.T).T
        except np.linalg.LinAlgError:
            k = hp_t @ np.linalg.pinv(s)
        x_upd = x_pred + k @ y
        x_upd[:4] = normalize_quaternion(x_upd[:4])
        i_kh = np.eye(7) - k @ h
        p_upd = i_kh @ p_pred @ i_kh.T + k @ r @ k.T
        p_upd = 0.5 * (p_upd + p_upd.T)
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
        x_pred = self._propagate_state(x_prev, dt_s=dt_s)
        f = self._numerical_jacobian(x_prev, dt_s=dt_s)
        q_scale = dt_s / self.dt_s if self.dt_s > 0.0 else 1.0
        q = np.diag(self.process_noise_diag) * max(q_scale, 0.0)
        p_pred = f @ p_prev @ f.T + q
        return x_pred, p_pred

    def _propagate_state(self, x: np.ndarray, *, dt_s: float | None = None) -> np.ndarray:
        step_dt_s = self.dt_s if dt_s is None else float(dt_s)
        if self._acceleration_enabled():
            return attitude_ekf_propagate_state_kernel(
                np.asarray(x, dtype=float).reshape(7),
                step_dt_s,
                np.asarray(self.inertia_kg_m2, dtype=float).reshape(3, 3),
            )
        q = normalize_quaternion(x[:4])
        w = x[4:7]

        q_dot = 0.5 * (omega_matrix(w) @ q)
        q_next = normalize_quaternion(q + step_dt_s * q_dot)

        iw = self.inertia_kg_m2 @ w
        w_dot = np.linalg.solve(self.inertia_kg_m2, -np.cross(w, iw))
        w_next = w + step_dt_s * w_dot
        return np.hstack((q_next, w_next))

    def _numerical_jacobian(self, x: np.ndarray, *, dt_s: float | None = None) -> np.ndarray:
        eps = 1e-6
        base = self._propagate_state(x, dt_s=dt_s)
        if self._acceleration_enabled():
            return attitude_ekf_numerical_jacobian_kernel(
                np.asarray(x, dtype=float).reshape(7),
                np.asarray(base, dtype=float).reshape(7),
                self.dt_s if dt_s is None else float(dt_s),
                np.asarray(self.inertia_kg_m2, dtype=float).reshape(3, 3),
            )
        j = np.zeros((7, 7))
        for i in range(7):
            xp = x.copy()
            xp[i] += eps
            yp = self._propagate_state(xp, dt_s=dt_s)
            j[:, i] = (yp - base) / eps
        return j
