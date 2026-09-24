"""Ground tracking EKF. Truth is intentionally absent from this interface."""

import numpy as np

from sim.core.models import StateBelief
from sim.estimation.ground_station_od import predict_ground_station_measurement
from sim.estimation.orbit_ekf import OrbitEKFEstimator

from .config import MAX_PREDICTION_STEPS


class GroundOrbitEstimator:
    def __init__(self, *, mu_km3_s2, process_noise_diag, prediction_step_s, jd_utc_start=None, frame_context=None):
        self.predictor = OrbitEKFEstimator(mu_km3_s2, 1.0, np.array(process_noise_diag), np.zeros(6))
        self.step = prediction_step_s
        self.jd = jd_utc_start
        self.frames = frame_context

    def predict(self, belief, epoch):
        if epoch < belief.last_update_t_s:
            raise ValueError("ground prediction cannot run backwards")
        if not np.isfinite(self.step) or self.step <= 0 or not np.isfinite(epoch):
            raise ValueError("ground prediction requires a finite epoch and positive step")
        if (epoch - belief.last_update_t_s) / self.step > MAX_PREDICTION_STEPS:
            raise ValueError("ground prediction exceeds the bounded step count")
        result = StateBelief(
            np.array(belief.state, copy=True), np.array(belief.covariance, copy=True), belief.last_update_t_s
        )
        for _ in range(MAX_PREDICTION_STEPS):
            if result.last_update_t_s >= epoch:
                return result
            next_epoch = min(epoch, result.last_update_t_s + self.step)
            if next_epoch <= result.last_update_t_s:
                raise ValueError("ground prediction step cannot advance floating-point time")
            result = self.predictor.update(result, None, next_epoch)
        if result.last_update_t_s < epoch:
            raise ValueError("ground prediction exceeds the bounded step count")
        return result

    def update(self, belief, row, station):
        epoch = row["time_s"]
        predicted = self.predict(belief, epoch)
        components = row["components"]

        def observe(x):
            values = predict_ground_station_measurement(
                target_state_eci=x, station=station, t_s=epoch, jd_utc_start=self.jd, frame_context=self.frames
            )
            return np.array([values[c] for c in components])

        def difference(a, b):
            d = a - b
            for i, c in enumerate(components):
                if c == "azimuth_deg":
                    d[i] = (d[i] + 180.0) % 360.0 - 180.0
            return d

        x, p = predicted.state, predicted.covariance
        h = np.empty((len(components), 6))
        for j in range(6):
            eps = 1e-3 if j < 3 else 1e-6
            dx = np.zeros(6)
            dx[j] = eps
            h[:, j] = difference(observe(x + dx), observe(x - dx)) / (2 * eps)
        innovation = difference(np.array(row["vector"]), observe(x))
        r = np.diag(np.square(row["sigma"]))
        s = h @ p @ h.T + r
        cross = p @ h.T
        try:
            gain = np.linalg.solve(s, cross.T).T
        except np.linalg.LinAlgError:
            gain = cross @ np.linalg.pinv(s)
        a = np.eye(6) - gain @ h
        covariance = a @ p @ a.T + gain @ r @ gain.T
        return StateBelief(x + gain @ innovation, (covariance + covariance.T) / 2.0, epoch)
