from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.orbit.lqr import HCWLQRController
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_dcm_ir_from_rv


def _as_length(arr: np.ndarray | float | list[float], length: int, name: str) -> np.ndarray:
    out = np.array(arr, dtype=float).reshape(-1)
    if out.size == 1:
        out = np.full(length, float(out[0]))
    if out.size != length:
        raise ValueError(f"{name} must be a scalar or length-{length} vector.")
    return out


def _as_gain_matrix(arr: np.ndarray | float | list[float], name: str) -> np.ndarray:
    raw = np.array(arr, dtype=float)
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


def _curv_position_to_rect(position_curv_km: np.ndarray, r0_km: float, eps: float = 1e-12) -> np.ndarray:
    x_r_curv, x_i_curv, x_c_curv = np.array(position_curv_km, dtype=float).reshape(3)
    r0 = max(float(r0_km), eps)
    r = max(r0 + x_r_curv, eps)
    theta_i = x_i_curv / r0
    theta_c = x_c_curv / r0

    c_i = np.cos(theta_i)
    s_i = np.sin(theta_i)
    c_c = np.cos(theta_c)
    s_c = np.sin(theta_c)

    return np.array(
        [
            r * c_c * c_i - r0,
            r * c_c * s_i,
            r * s_c,
        ],
        dtype=float,
    )


def curv_accel_to_rect(
    accel_curv_km_s2: np.ndarray,
    position_curv_km: np.ndarray,
    r0_km: float,
    delta_km: float = 1e-3,
) -> np.ndarray:
    """Map a curvilinear RIC burn vector into the local rectangular RIC basis."""

    u_curv = np.array(accel_curv_km_s2, dtype=float).reshape(3)
    q = np.array(position_curv_km, dtype=float).reshape(3)
    h_base = max(float(delta_km), 1e-9)
    jac = np.zeros((3, 3), dtype=float)
    for idx in range(3):
        h = h_base * max(1.0, abs(float(q[idx])))
        dq = np.zeros(3, dtype=float)
        dq[idx] = h
        jac[:, idx] = (
            _curv_position_to_rect(q + dq, r0_km=r0_km)
            - _curv_position_to_rect(q - dq, r0_km=r0_km)
        ) / (2.0 * h)
    return jac @ u_curv


@dataclass
class CurvilinearRICPDController(Controller):
    """PD orbit controller whose feedback law is designed in curvilinear RIC coordinates."""

    max_accel_km_s2: float
    kp: np.ndarray = field(default_factory=lambda: np.ones(3) * 5.0e-7)
    kd: np.ndarray = field(default_factory=lambda: np.ones(3) * 1.5e-3)
    desired_state_curv: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)
    conversion_delta_km: float = 1e-3

    def __post_init__(self) -> None:
        if self.max_accel_km_s2 < 0.0:
            raise ValueError("max_accel_km_s2 must be non-negative.")
        if self.ric_curv_state_slice[1] - self.ric_curv_state_slice[0] != 6:
            raise ValueError("ric_curv_state_slice must select exactly 6 elements.")
        if self.chief_eci_state_slice[1] - self.chief_eci_state_slice[0] != 6:
            raise ValueError("chief_eci_state_slice must select exactly 6 elements.")
        if self.conversion_delta_km <= 0.0:
            raise ValueError("conversion_delta_km must be positive.")

        self.kp = _as_gain_matrix(self.kp, "kp")
        self.kd = _as_gain_matrix(self.kd, "kd")
        self.desired_state_curv = _as_length(self.desired_state_curv, 6, "desired_state_curv")

    def linear_system_summary(self) -> dict[str, object]:
        k_gain = np.hstack((self.kp, self.kd))
        return {
            "system_type": "curvilinear_pd_feedback",
            "law_label": "-Kp*x_curv - Kd*xdot_curv",
            "control_axes": ["R_curv", "I_curv", "C_curv"],
            "state_labels": ["R_curv", "I_curv", "C_curv", "dR_curv", "dI_curv", "dC_curv"],
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

        err = x_curv - self.desired_state_curv
        accel_curv_pre_limit = -(self.kp @ err[:3] + self.kd @ err[3:])
        accel_rect_pre_limit = curv_accel_to_rect(
            accel_curv_pre_limit,
            position_curv_km=x_curv[:3],
            r0_km=r0,
            delta_km=self.conversion_delta_km,
        )

        accel_curv = np.array(accel_curv_pre_limit, dtype=float)
        accel_rect = np.array(accel_rect_pre_limit, dtype=float)
        nrm = float(np.linalg.norm(accel_rect_pre_limit))
        limit_scale = 1.0
        if self.max_accel_km_s2 == 0.0:
            limit_scale = 0.0
            accel_curv[:] = 0.0
            accel_rect[:] = 0.0
        elif nrm > self.max_accel_km_s2:
            limit_scale = float(self.max_accel_km_s2 / nrm)
            accel_curv *= limit_scale
            accel_rect *= limit_scale

        c_ir = ric_dcm_ir_from_rv(r_chief, v_chief)
        accel_eci = c_ir @ accel_rect
        k_gain = np.hstack((self.kp, self.kd))
        debug = HCWLQRController._linear_feedback_debug_payload(
            control_axes=["R_curv", "I_curv", "C_curv"],
            k_gain=k_gain,
            x_rect=x_curv,
            x_effective=err,
            control_pre_limit=accel_curv_pre_limit,
            control_post_limit=accel_curv,
            limit_scale=limit_scale,
            state_signs=np.ones(6),
        )
        debug["frame"] = "curvilinear_RIC"
        debug["state_labels"] = ["R_curv", "I_curv", "C_curv", "dR_curv", "dI_curv", "dC_curv"]
        debug["law_label"] = "-Kp*x_curv - Kd*xdot_curv"
        debug["control_rect_ric_pre_limit"] = accel_rect_pre_limit.tolist()
        debug["control_rect_ric_post_limit"] = accel_rect.tolist()

        return Command(
            thrust_eci_km_s2=accel_eci,
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "curvilinear_ric_pd",
                "ric_curv_state_slice": [i0, i1],
                "chief_eci_state_slice": [j0, j1],
                "desired_state_curv": self.desired_state_curv.tolist(),
                "accel_curv_ric_km_s2": accel_curv.tolist(),
                "accel_rect_ric_km_s2": accel_rect.tolist(),
                "accel_eci_km_s2": accel_eci.tolist(),
                "linear_feedback_debug": debug,
            },
        )
