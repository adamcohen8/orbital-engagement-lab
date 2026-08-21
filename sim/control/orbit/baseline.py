from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.dynamics.orbit.elements import orbital_element_feedback_accel, rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2


@dataclass
class StationkeepingController(Controller):
    target_state: np.ndarray
    kp_pos: float = 1e-5
    kd_vel: float = 5e-4
    max_accel_km_s2: float = 5e-5

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        target_state = np.array(self.target_state, dtype=float).reshape(6)
        state = np.array(belief.state[:6], dtype=float).reshape(6)
        pos_err = target_state[:3] - state[:3]
        vel_err = target_state[3:6] - state[3:6]
        a_cmd = self.kp_pos * pos_err + self.kd_vel * vel_err
        n = np.linalg.norm(a_cmd)
        if n > self.max_accel_km_s2 and n > 0.0:
            a_cmd *= self.max_accel_km_s2 / n
        return Command(thrust_eci_km_s2=a_cmd, torque_body_nm=np.zeros(3), mode_flags={"mode": "stationkeeping"})


@dataclass
class SemiMajorAxisEccentricityController(Controller):
    target_a_km: float
    target_ecc: float = 0.0
    energy_gain_per_s: float = 1.0e-3
    eccentricity_gain_per_s: float = 5.0e-4
    max_accel_km_s2: float = 5.0e-5
    a_tolerance_km: float = 0.5
    ecc_tolerance: float = 1.0e-4
    mu_km3_s2: float = EARTH_MU_KM3_S2

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        state = np.array(belief.state[:6], dtype=float).reshape(6)
        r = state[:3]
        v = state[3:6]
        mu = float(self.mu_km3_s2)
        r_norm = float(np.linalg.norm(r))
        v_norm = float(np.linalg.norm(v))
        h = np.cross(r, v)
        coes = rv_to_coe_eci(r, v, mu_km3_s2=mu)
        a_err = float(self.target_a_km) - float(coes.a_km)
        e_vec = np.cross(v, h) / mu - r / max(r_norm, 1e-12)
        e_err = float(coes.ecc) - float(self.target_ecc)

        energy = 0.5 * v_norm * v_norm - mu / max(r_norm, 1e-12)
        target_energy = -mu / (2.0 * float(self.target_a_km))
        energy_rate_cmd = 0.0
        if abs(a_err) > float(max(self.a_tolerance_km, 0.0)):
            energy_rate_cmd = float(self.energy_gain_per_s) * (target_energy - energy)

        ecc_rate_cmd = np.zeros(3, dtype=float)
        if abs(e_err) > float(max(self.ecc_tolerance, 0.0)):
            e_norm = float(np.linalg.norm(e_vec))
            direction = e_vec / e_norm if e_norm > 1.0e-12 else r / max(r_norm, 1.0e-12)
            target_e_vec = float(self.target_ecc) * direction
            ecc_rate_cmd = float(self.eccentricity_gain_per_s) * (target_e_vec - e_vec)

        def ecc_rate_for_accel(accel_eci: np.ndarray) -> np.ndarray:
            return (np.cross(accel_eci, h) + np.cross(v, np.cross(r, accel_eci))) / mu

        basis = np.eye(3, dtype=float)
        ecc_jac = np.column_stack([ecc_rate_for_accel(basis[:, i]) for i in range(3)])
        lhs = np.vstack((v.reshape(1, 3), ecc_jac))
        rhs = np.hstack(([energy_rate_cmd], ecc_rate_cmd))
        accel_eci, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

        n = float(np.linalg.norm(accel_eci))
        amax = float(max(self.max_accel_km_s2, 0.0))
        if amax <= 0.0:
            accel_eci = np.zeros(3, dtype=float)
        elif n > amax:
            accel_eci *= amax / n
        return Command(
            thrust_eci_km_s2=accel_eci,
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "sma_ecc_feedback",
                "a_km": float(coes.a_km),
                "ecc": float(coes.ecc),
                "a_error_km": float(a_err),
                "ecc_error": float(e_err),
                "energy_error_km2_s2": float(target_energy - energy),
            },
        )


@dataclass
class OrbitalElementsFeedbackController(Controller):
    target_coes: dict
    controlled_elements: tuple[str, ...] | list[str] | str = ("a", "ecc", "inc", "raan", "argp")
    energy_gain_per_s: float = 1.0e-3
    eccentricity_gain_per_s: float = 5.0e-4
    plane_gain_per_s: float = 5.0e-4
    max_accel_km_s2: float = 5.0e-5
    mu_km3_s2: float = EARTH_MU_KM3_S2

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        result = orbital_element_feedback_accel(
            np.array(belief.state[:6], dtype=float).reshape(6),
            dict(self.target_coes or {}),
            controlled_elements=self.controlled_elements,
            energy_gain_per_s=float(self.energy_gain_per_s),
            eccentricity_gain_per_s=float(self.eccentricity_gain_per_s),
            plane_gain_per_s=float(self.plane_gain_per_s),
            max_accel_km_s2=float(self.max_accel_km_s2),
            mu_km3_s2=float(self.mu_km3_s2),
        )
        coes = result.current_coes
        return Command(
            thrust_eci_km_s2=np.array(result.accel_eci_km_s2, dtype=float),
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "orbital_elements_feedback",
                "a_km": float(coes.a_km),
                "ecc": float(coes.ecc),
                "inc_deg": float(coes.inc_deg),
                "raan_deg": float(coes.raan_deg),
                "argp_deg": float(coes.argp_deg),
                "energy_error_km2_s2": float(result.energy_error_km2_s2),
                "eccentricity_vector_error_norm": float(np.linalg.norm(result.eccentricity_vector_error)),
                "hhat_error_norm": float(np.linalg.norm(result.hhat_error)),
            },
        )


@dataclass
class SafetyBarrierController(Controller):
    keep_out_radius_km: float
    kp_barrier: float = 5e-5
    max_accel_km_s2: float = 1e-4

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        r = belief.state[:3]
        norm_r = np.linalg.norm(r)
        if norm_r >= self.keep_out_radius_km:
            return Command.zero()
        direction = r / max(norm_r, 1e-9)
        a_cmd = self.kp_barrier * (self.keep_out_radius_km - norm_r) * direction
        n = np.linalg.norm(a_cmd)
        if n > self.max_accel_km_s2 and n > 0.0:
            a_cmd *= self.max_accel_km_s2 / n
        return Command(thrust_eci_km_s2=a_cmd, torque_body_nm=np.zeros(3), mode_flags={"mode": "barrier"})


@dataclass
class RiskThresholdController(Controller):
    risk_fn: callable
    nominal: Controller
    evasive: Controller
    threshold: float = 0.5

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        risk = float(self.risk_fn(belief, t_s))
        if risk >= self.threshold:
            return self.evasive.act(belief, t_s, budget_ms)
        return self.nominal.act(belief, t_s, budget_ms)
