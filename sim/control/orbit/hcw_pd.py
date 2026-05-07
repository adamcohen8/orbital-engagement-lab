from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.orbit.lqr import HCWLQRController
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv


def _as_gain_matrix(value: np.ndarray | float | list[float], name: str) -> np.ndarray:
    raw = np.array(value, dtype=float)
    flat = raw.reshape(-1)
    if flat.size == 1:
        out = np.eye(3, dtype=float) * float(flat[0])
    elif flat.size == 3:
        out = np.diag(flat)
    elif raw.shape == (3, 3):
        out = raw.reshape(3, 3)
    else:
        raise ValueError(f"{name} must be a scalar, length-3 vector, or 3x3 matrix.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite numbers.")
    return out


def _as_state(value: np.ndarray | list[float], name: str) -> np.ndarray:
    out = np.array(value, dtype=float).reshape(-1)
    if out.size != 6:
        raise ValueError(f"{name} must be length 6.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite numbers.")
    return out


@dataclass
class HCWPDController(Controller):
    """PD orbit controller using rectangular RIC state feedback for HCW-style rendezvous."""

    max_accel_km_s2: float
    mean_motion_rad_s: float = 0.0
    kp: np.ndarray = field(default_factory=lambda: np.eye(3) * 4.0e-6)
    kd: np.ndarray = field(default_factory=lambda: np.eye(3) * 4.0e-3)
    desired_state_ric: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)
    state_signs: np.ndarray = field(default_factory=lambda: np.ones(6))

    def __post_init__(self) -> None:
        if self.max_accel_km_s2 < 0.0:
            raise ValueError("max_accel_km_s2 must be non-negative.")
        if self.mean_motion_rad_s < 0.0:
            raise ValueError("mean_motion_rad_s must be non-negative.")
        if self.ric_curv_state_slice[1] - self.ric_curv_state_slice[0] != 6:
            raise ValueError("ric_curv_state_slice must select exactly 6 elements.")
        if self.chief_eci_state_slice[1] - self.chief_eci_state_slice[0] != 6:
            raise ValueError("chief_eci_state_slice must select exactly 6 elements.")
        self.kp = _as_gain_matrix(self.kp, "kp")
        self.kd = _as_gain_matrix(self.kd, "kd")
        self.desired_state_ric = _as_state(self.desired_state_ric, "desired_state_ric")
        signs = np.array(self.state_signs, dtype=float).reshape(-1)
        if signs.size != 6:
            raise ValueError("state_signs must be length 6.")
        signs[signs == 0.0] = 1.0
        self.state_signs = np.sign(signs)

    def linear_system_summary(self) -> dict[str, object]:
        k_gain = np.hstack((self.kp, self.kd))
        return {
            "system_type": "hcw_pd_feedback",
            "law_label": HCWLQRController._control_law_label(self.state_signs),
            "control_axes": ["R", "I", "C"],
            "state_labels": ["R", "I", "C", "dR", "dI", "dC"],
            "gain_matrix": k_gain.tolist(),
        }

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        i0, i1 = self.ric_curv_state_slice
        j0, j1 = self.chief_eci_state_slice
        if belief.state.size < max(i1, j1):
            return Command.zero()

        x_curv = np.array(belief.state[i0:i1], dtype=float)
        chief_eci = np.array(belief.state[j0:j1], dtype=float)
        r_chief = chief_eci[:3]
        v_chief = chief_eci[3:]
        r0 = float(np.linalg.norm(r_chief))
        if r0 <= 0.0:
            return Command.zero()

        x_rect = ric_curv_to_rect(x_curv, r0_km=r0)
        err = x_rect - self.desired_state_ric
        x_effective = self.state_signs * err
        k_gain = np.hstack((self.kp, self.kd))
        accel_ric_pre_limit = -k_gain @ x_effective
        if self.mean_motion_rad_s > 0.0:
            n = float(self.mean_motion_rad_s)
            x, _y, z, xdot, ydot, _zdot = x_effective
            accel_ric_pre_limit += np.array(
                [
                    -3.0 * n * n * x - 2.0 * n * ydot,
                    2.0 * n * xdot,
                    n * n * z,
                ],
                dtype=float,
            )
        accel_ric = np.array(accel_ric_pre_limit, dtype=float)
        nrm = float(np.linalg.norm(accel_ric_pre_limit))
        limit_scale = 1.0
        if self.max_accel_km_s2 == 0.0:
            limit_scale = 0.0
            accel_ric[:] = 0.0
        elif nrm > self.max_accel_km_s2:
            limit_scale = float(self.max_accel_km_s2 / nrm)
            accel_ric *= limit_scale

        c_ir = ric_dcm_ir_from_rv(r_chief, v_chief)
        accel_eci = c_ir @ accel_ric
        return Command(
            thrust_eci_km_s2=accel_eci,
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "hcw_pd",
                "ric_curv_state_slice": [i0, i1],
                "chief_eci_state_slice": [j0, j1],
                "desired_state_ric": self.desired_state_ric.tolist(),
                "accel_ric_km_s2": accel_ric.tolist(),
                "linear_feedback_debug": HCWLQRController._linear_feedback_debug_payload(
                    control_axes=["R", "I", "C"],
                    k_gain=k_gain,
                    x_rect=x_rect,
                    x_effective=x_effective,
                    control_pre_limit=accel_ric_pre_limit,
                    control_post_limit=accel_ric,
                    limit_scale=limit_scale,
                    state_signs=self.state_signs,
                ),
            },
        )
