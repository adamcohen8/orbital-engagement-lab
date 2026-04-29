from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.estimation.attitude_ekf import AttitudeEKFEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.utils.quaternion import normalize_quaternion


@dataclass
class JointStateEstimator(Estimator):
    orbit_estimator: OrbitEKFEstimator
    dt_s: float
    inertia_kg_m2: np.ndarray | None = None
    attitude_process_var: float = 1e-8
    attitude_meas_var: float = 1e-6
    attitude_estimator: AttitudeEKFEstimator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        inertia = np.eye(3) if self.inertia_kg_m2 is None else np.array(self.inertia_kg_m2, dtype=float).reshape(3, 3)
        self.inertia_kg_m2 = inertia
        self.attitude_estimator = AttitudeEKFEstimator(
            dt_s=float(self.dt_s),
            inertia_kg_m2=inertia,
            process_noise_diag=np.ones(7, dtype=float) * float(self.attitude_process_var),
            meas_noise_diag=np.ones(7, dtype=float) * float(self.attitude_meas_var),
        )

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        if belief.state.size < 13:
            raise ValueError("JointStateEstimator expects 13-state belief [r,v,q,w].")

        x = belief.state
        p = belief.covariance

        orb_belief = StateBelief(state=x[:6], covariance=p[:6, :6], last_update_t_s=belief.last_update_t_s)
        orb_meas = None if measurement is None else Measurement(vector=measurement.vector[:6], t_s=measurement.t_s)
        orb_upd = self.orbit_estimator.update(orb_belief, orb_meas, t_s)

        att_state = np.hstack((normalize_quaternion(x[6:10]), np.array(x[10:13], dtype=float)))
        att_cov = p[6:13, 6:13] if p.shape == (13, 13) else np.eye(7) * self.attitude_meas_var
        att_belief = StateBelief(state=att_state, covariance=att_cov, last_update_t_s=belief.last_update_t_s)
        att_meas = None
        if measurement is not None and measurement.vector.size >= 13:
            att_meas_vec = np.hstack((normalize_quaternion(measurement.vector[6:10]), np.array(measurement.vector[10:13], dtype=float)))
            att_meas = Measurement(vector=att_meas_vec, t_s=measurement.t_s)
        assert self.attitude_estimator is not None
        att_upd = self.attitude_estimator.update(att_belief, att_meas, t_s)

        x_upd = np.hstack((orb_upd.state, normalize_quaternion(att_upd.state[:4]), att_upd.state[4:7]))

        p_upd = np.zeros((13, 13))
        p_upd[:6, :6] = orb_upd.covariance
        p_upd[6:13, 6:13] = att_upd.covariance

        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=t_s)
