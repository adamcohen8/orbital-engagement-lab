from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.acceleration.settings import acceleration_settings_from_mode
from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.two_body import propagate_two_body_rk4

orbit_ekf_numerical_jacobian_kernel = None
propagate_two_body_rk4_kernel = None


def _load_acceleration_kernels() -> None:
    global orbit_ekf_numerical_jacobian_kernel, propagate_two_body_rk4_kernel
    if propagate_two_body_rk4_kernel is not None:
        return
    from sim.acceleration.kernels.estimation import (
        orbit_ekf_numerical_jacobian_kernel as accelerated_jacobian,
    )
    from sim.acceleration.kernels.estimation import (
        propagate_two_body_rk4_kernel as accelerated_propagate,
    )

    orbit_ekf_numerical_jacobian_kernel = accelerated_jacobian
    propagate_two_body_rk4_kernel = accelerated_propagate


@dataclass(frozen=True)
class OrbitEKFUpdateDiagnostics:
    measurement_available: bool
    update_applied: bool
    innovation: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))
    innovation_covariance: np.ndarray = field(default_factory=lambda: np.full((6, 6), np.nan))
    nis: float = float("nan")
    predicted_cov_trace: float = float("nan")
    posterior_cov_trace: float = float("nan")


@dataclass
class OrbitEKFEstimator(Estimator):
    mu_km3_s2: float
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    last_update_diagnostics: OrbitEKFUpdateDiagnostics | None = field(default=None, init=False, repr=False)
    acceleration_mode: str = "off"
    _q: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)), init=False, repr=False)
    _r: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)), init=False, repr=False)
    _h: np.ndarray = field(default_factory=lambda: np.eye(6), init=False, repr=False)
    _i6: np.ndarray = field(default_factory=lambda: np.eye(6), init=False, repr=False)
    _zero_accel: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _acceleration_enabled_value: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.process_noise_diag = np.array(self.process_noise_diag, dtype=float)
        self.meas_noise_diag = np.array(self.meas_noise_diag, dtype=float)
        self._q = np.diag(self.process_noise_diag)
        self._r = np.diag(self.meas_noise_diag)
        self._acceleration_enabled_value = bool(acceleration_settings_from_mode(self.acceleration_mode).enabled)
        if self._acceleration_enabled_value:
            _load_acceleration_kernels()

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        output_t_s = float(t_s)
        meas_t_s = output_t_s
        if measurement is not None:
            meas_t_s = float(np.clip(float(measurement.t_s), float(belief.last_update_t_s), output_t_s))

        x_pred, p_pred = self._predict(belief.state, belief.covariance, from_t_s=belief.last_update_t_s, to_t_s=meas_t_s)

        if measurement is None:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            self.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
                measurement_available=False,
                update_applied=False,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)

        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        if z.size < 6:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            self.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
                measurement_available=True,
                update_applied=False,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)
        z = z[:6]
        y = z - x_pred
        s = self._h @ p_pred @ self._h.T + self._r
        hp_t = p_pred @ self._h.T
        try:
            k = np.linalg.solve(s.T, hp_t.T).T
            s_y = np.linalg.solve(s, y)
        except np.linalg.LinAlgError:
            s_pinv = np.linalg.pinv(s)
            k = hp_t @ s_pinv
            s_y = s_pinv @ y
        x_upd = x_pred + k @ y
        i_kh = self._i6 - k @ self._h
        p_upd = i_kh @ p_pred @ i_kh.T + k @ self._r @ k.T
        p_upd = 0.5 * (p_upd + p_upd.T)
        nis = float(y.T @ s_y)
        self.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
            measurement_available=True,
            update_applied=True,
            innovation=np.array(y, dtype=float),
            innovation_covariance=np.array(s, dtype=float),
            nis=nis,
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
        x_pred = self._propagate_state(x_prev, dt_s=dt_s)
        f = self._numerical_jacobian(x_prev, base=x_pred, dt_s=dt_s)
        q_scale = dt_s / self.dt_s if self.dt_s > 0.0 else 1.0
        p_pred = f @ p_prev @ f.T + self._q * max(q_scale, 0.0)
        return x_pred, 0.5 * (p_pred + p_pred.T)

    def _acceleration_enabled(self) -> bool:
        return self._acceleration_enabled_value

    def _propagate_state(self, x: np.ndarray, *, dt_s: float) -> np.ndarray:
        step_dt_s = float(dt_s)
        if self._acceleration_enabled():
            _load_acceleration_kernels()
            return propagate_two_body_rk4_kernel(np.asarray(x, dtype=float).reshape(6), step_dt_s, float(self.mu_km3_s2))
        return propagate_two_body_rk4(
            x_eci=x,
            dt_s=step_dt_s,
            mu_km3_s2=self.mu_km3_s2,
            accel_cmd_eci_km_s2=self._zero_accel,
        )

    def _numerical_jacobian(
        self, x: np.ndarray, *, base: np.ndarray | None = None, dt_s: float | None = None
    ) -> np.ndarray:
        step_dt_s = self.dt_s if dt_s is None else float(dt_s)
        eps = 1e-6
        base_eval = base
        if base_eval is None:
            base_eval = self._propagate_state(x, dt_s=step_dt_s)
        if self._acceleration_enabled():
            _load_acceleration_kernels()
            return orbit_ekf_numerical_jacobian_kernel(
                np.asarray(x, dtype=float).reshape(6),
                np.asarray(base_eval, dtype=float).reshape(6),
                step_dt_s,
                float(self.mu_km3_s2),
            )
        j = np.zeros((6, 6))
        for i in range(6):
            xp = x.copy()
            xp[i] += eps
            yp = propagate_two_body_rk4(
                x_eci=xp,
                dt_s=step_dt_s,
                mu_km3_s2=self.mu_km3_s2,
                accel_cmd_eci_km_s2=self._zero_accel,
            )
            j[:, i] = (yp - base_eval) / eps
        return j
