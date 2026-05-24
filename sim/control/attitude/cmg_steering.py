from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


def _construct_controller(spec: Any) -> Controller:
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


@dataclass
class CMGSteeringController(Controller):
    base_controller: Any
    max_torque_nm: np.ndarray | float = 0.2
    momentum_nms: np.ndarray | float = 1.0
    gimbal_rate_limit_rad_s: np.ndarray | float = 0.1

    def __post_init__(self) -> None:
        self.base_controller = _construct_controller(self.base_controller)

    @staticmethod
    def _vec(value: np.ndarray | float, default: float = 0.0) -> np.ndarray:
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(3, float(arr[0]), dtype=float)
        if arr.size != 3:
            return np.full(3, float(default), dtype=float)
        return arr

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        cmd = self.base_controller.act(belief, t_s, budget_ms)
        torque = np.array(cmd.torque_body_nm, dtype=float).reshape(3)
        cap = np.minimum(np.abs(self._vec(self.max_torque_nm)), np.abs(self._vec(self.momentum_nms) * self._vec(self.gimbal_rate_limit_rad_s)))
        torque = np.clip(torque, -cap, cap)
        mode_flags = dict(cmd.mode_flags or {})
        mode_flags.update(
            {
                "mode": "cmg_steering",
                "cmg_base_mode": mode_flags.get("mode"),
                "cmg_torque_cap_nm": cap.tolist(),
            }
        )
        return Command(
            thrust_eci_km_s2=np.array(cmd.thrust_eci_km_s2, dtype=float),
            torque_body_nm=torque,
            mode_flags=mode_flags,
        )
