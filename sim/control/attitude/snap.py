from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.quaternion import dcm_to_quaternion_bn


def _dcm_from_euler_321(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, sr], [0.0, -sr, cr]])
    rot_y = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    rot_z = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


@dataclass
class SnapAttitudeController(Controller):
    desired_state6: np.ndarray
    one_shot: bool = True
    _done: bool = field(default=False, init=False)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if self.one_shot and self._done:
            return Command.zero()
        x = np.array(self.desired_state6, dtype=float).reshape(-1)
        if x.size != 6:
            raise ValueError("desired_state6 must be [roll,pitch,yaw,wx,wy,wz].")
        if not np.all(np.isfinite(x)):
            raise ValueError("desired_state6 must contain finite values.")
        q_next_bn = dcm_to_quaternion_bn(_dcm_from_euler_321(*x[:3]))
        self._done = True
        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "snap",
                "snap_attitude_state6": x.tolist(),
                "attitude_state_override": {
                    "q_next_bn": q_next_bn.tolist(),
                    "w_next_body_rad_s": x[3:].tolist(),
                    "phase": "snap",
                },
            },
        )
