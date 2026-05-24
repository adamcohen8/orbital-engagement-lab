from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


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


@dataclass
class ElectricPropulsionController(Controller):
    base_controller: Any
    mass_kg: float = 100.0
    max_thrust_n: float = 0.5
    duty_cycle: float = 1.0
    max_power_w: float | None = None
    power_per_newton_w: float | None = None

    def __post_init__(self) -> None:
        self.base_controller = _construct_controller(self.base_controller)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        cmd = self.base_controller.act(belief, t_s, budget_ms)
        accel = np.array(cmd.thrust_eci_km_s2, dtype=float).reshape(3)
        thrust_cap = float(max(self.max_thrust_n, 0.0))
        if self.max_power_w is not None and self.power_per_newton_w is not None and float(self.power_per_newton_w) > 0.0:
            thrust_cap = min(thrust_cap, float(max(self.max_power_w, 0.0)) / float(self.power_per_newton_w))
        thrust_cap *= float(np.clip(self.duty_cycle, 0.0, 1.0))
        max_accel = 0.0 if self.mass_kg <= 0.0 else thrust_cap / float(self.mass_kg) / 1e3
        n = float(np.linalg.norm(accel))
        if n > max_accel > 0.0:
            accel *= max_accel / n
        elif max_accel <= 0.0:
            accel = np.zeros(3, dtype=float)
        mode_flags = dict(cmd.mode_flags or {})
        mode_flags.update(
            {
                "mode": "electric_propulsion_guidance",
                "electric_base_mode": mode_flags.get("mode"),
                "electric_propulsion_max_thrust_n": float(thrust_cap),
            }
        )
        return Command(
            thrust_eci_km_s2=accel,
            torque_body_nm=np.array(cmd.torque_body_nm, dtype=float),
            mode_flags=mode_flags,
        )
