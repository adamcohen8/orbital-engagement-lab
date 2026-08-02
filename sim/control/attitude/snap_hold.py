from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.control.attitude.snap import _dcm_from_euler_321
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn


@dataclass
class SnapAndHoldRICAttitudeController(Controller):
    desired_state6_ric: np.ndarray

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        x = np.array(self.desired_state6_ric, dtype=float).reshape(-1)
        if x.size != 6:
            raise ValueError("desired_state6_ric must be [yaw_R,roll_I,pitch_C,wx,wy,wz].")
        if not np.all(np.isfinite(x)):
            raise ValueError("desired_state6_ric must contain finite values.")
        if belief.state.size < 6:
            return Command.zero()
        r_eci = np.array(belief.state[:3], dtype=float)
        v_eci = np.array(belief.state[3:6], dtype=float)
        if not np.all(np.isfinite(r_eci)) or not np.all(np.isfinite(v_eci)) or np.linalg.norm(r_eci) <= 0.0:
            return Command.zero()
        yaw_r, roll_i, pitch_c = x[:3]
        c_br = _dcm_from_euler_321(yaw_r, roll_i, pitch_c)
        c_ir = ric_dcm_ir_from_rv(r_eci, v_eci)
        q_next_bn = dcm_to_quaternion_bn(c_br @ c_ir.T)
        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "snap_hold_ric",
                "snap_hold_ric_state6": x.tolist(),
                "attitude_state_override": {
                    "q_next_bn": q_next_bn.tolist(),
                    "w_next_body_rad_s": x[3:].tolist(),
                    "phase": "hold",
                },
            },
        )
