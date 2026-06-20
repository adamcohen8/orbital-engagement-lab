from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator


@dataclass
class OrbitUKFEstimator(Estimator):
    propagator: OrbitPropagator
    context: OrbitContext
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        output_t_s = float(t_s)
        meas_t_s = output_t_s
        if measurement is not None:
            meas_t_s = float(np.clip(float(measurement.t_s), float(belief.last_update_t_s), output_t_s))
        x_pred, p_pred, sigma_pred, wm, wc = self._predict(
            belief.state,
            belief.covariance,
            from_t_s=float(belief.last_update_t_s),
            to_t_s=meas_t_s,
        )

        if measurement is None:
            if meas_t_s < output_t_s:
                x_pred, p_pred, _, _, _ = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)
        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        if z.size < x_pred.size:
            if meas_t_s < output_t_s:
                x_pred, p_pred, _, _, _ = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)

        n = x_pred.size
        h_sigma = sigma_pred
        z_pred = np.sum(wm[:, None] * h_sigma, axis=0)
        r = np.diag(self.meas_noise_diag)
        s_mat = r.copy()
        pxz = np.zeros((n, n))
        for i in range(2 * n + 1):
            dz = h_sigma[i] - z_pred
            dx = sigma_pred[i] - x_pred
            s_mat += wc[i] * np.outer(dz, dz)
            pxz += wc[i] * np.outer(dx, dz)

        try:
            k_gain = np.linalg.solve(s_mat.T, pxz.T).T
        except np.linalg.LinAlgError:
            k_gain = pxz @ np.linalg.pinv(s_mat)
        innovation = z[:n] - z_pred
        x_upd = x_pred + k_gain @ innovation
        i_kh = np.eye(n) - k_gain
        p_upd = i_kh @ p_pred @ i_kh.T + k_gain @ r @ k_gain.T
        p_upd = 0.5 * (p_upd + p_upd.T)
        if meas_t_s < output_t_s:
            x_upd, p_upd, _, _, _ = self._predict(x_upd, p_upd, from_t_s=meas_t_s, to_t_s=output_t_s)
        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=output_t_s)

    def _predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        *,
        from_t_s: float,
        to_t_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = np.asarray(state, dtype=float).reshape(-1).size
        lam = self.alpha**2 * (n + self.kappa) - n
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)

        sigma = self._sigma_points(state, covariance, lam)
        dt_s = max(float(to_t_s) - float(from_t_s), 0.0)
        sigma_pred = np.array(
            [
                self.propagator.propagate(
                    x_eci=s,
                    dt_s=dt_s,
                    t_s=float(from_t_s),
                    command_accel_eci_km_s2=np.zeros(3),
                    env={},
                    ctx=self.context,
                )
                for s in sigma
            ]
        )

        x_pred = np.sum(wm[:, None] * sigma_pred, axis=0)
        q_scale = dt_s / self.dt_s if self.dt_s > 0.0 else 1.0
        p_pred = np.diag(self.process_noise_diag) * max(q_scale, 0.0)
        for i in range(2 * n + 1):
            dx = sigma_pred[i] - x_pred
            p_pred += wc[i] * np.outer(dx, dx)
        return x_pred, 0.5 * (p_pred + p_pred.T), sigma_pred, wm, wc

    def _sigma_points(self, x: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
        n = x.size
        c = np.linalg.cholesky((n + lam) * p + 1e-12 * np.eye(n))
        points = [x]
        for i in range(n):
            points.append(x + c[:, i])
            points.append(x - c[:, i])
        return np.array(points)
