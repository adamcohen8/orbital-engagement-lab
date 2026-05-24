from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.quaternion import quaternion_to_dcm_bn


def _construct_controller(spec: Any) -> Any:
    if hasattr(spec, "act"):
        return spec
    if isinstance(spec, dict):
        module = spec.get("module")
        class_name = spec.get("class_name")
        params = dict(spec.get("params", {}) or {})
        if not module or not class_name:
            raise ValueError("controller spec dict must include 'module' and 'class_name'.")
        mod = importlib.import_module(str(module))
        return getattr(mod, str(class_name))(**params)
    raise TypeError("base_controller must be a controller object or constructor dict.")


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.array(vec, dtype=float).reshape(3)
    mag = float(np.linalg.norm(arr))
    if mag <= eps:
        return np.zeros(3, dtype=float)
    return arr / mag


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    ua = _unit(a)
    ub = _unit(b)
    if float(np.linalg.norm(ua)) <= 0.0 or float(np.linalg.norm(ub)) <= 0.0:
        return 0.0
    return float(np.arccos(np.clip(float(np.dot(ua, ub)), -1.0, 1.0)))


@dataclass
class GimbaledThrusterController(Controller):
    base_controller: Any
    neutral_direction_body: np.ndarray
    max_gimbal_angle_rad: float = 0.0
    attitude_quat_slice: tuple[int, int] = (6, 10)

    def __post_init__(self) -> None:
        self.base_controller = _construct_controller(self.base_controller)
        self.neutral_direction_body = _unit(np.array(self.neutral_direction_body, dtype=float))

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        cmd = self.base_controller.act(belief, t_s, budget_ms)
        accel = np.array(cmd.thrust_eci_km_s2, dtype=float).reshape(3)
        i0, i1 = self.attitude_quat_slice
        gimbal_angle = 0.0
        if belief.state.size >= i1 and float(np.linalg.norm(accel)) > 0.0:
            q = np.array(belief.state[i0:i1], dtype=float).reshape(4)
            qn = float(np.linalg.norm(q))
            if qn > 0.0:
                c_bn = quaternion_to_dcm_bn(q / qn)
                desired_plume_body = -_unit(c_bn @ accel)
                gimbal_angle = _angle(self.neutral_direction_body, desired_plume_body)
                max_angle = float(max(self.max_gimbal_angle_rad, 0.0))
                if gimbal_angle > max_angle >= 0.0:
                    accel = np.zeros(3, dtype=float)
        mode_flags = dict(cmd.mode_flags or {})
        mode_flags.update(
            {
                "mode": "gimbaled_thruster_guidance",
                "gimbaled_base_mode": mode_flags.get("mode"),
                "gimbal_angle_request_rad": float(gimbal_angle),
            }
        )
        return Command(
            thrust_eci_km_s2=accel,
            torque_body_nm=np.array(cmd.torque_body_nm, dtype=float),
            mode_flags=mode_flags,
        )
