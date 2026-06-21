from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.quaternion import quaternion_to_dcm_bn


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


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.array(vec, dtype=float).reshape(3)
    mag = float(np.linalg.norm(arr))
    if mag <= eps:
        return np.zeros(3, dtype=float)
    return arr / mag


def _bounded_nonnegative_lstsq(a: np.ndarray, b: np.ndarray, upper: np.ndarray) -> np.ndarray:
    matrix = np.array(a, dtype=float)
    target = np.array(b, dtype=float).reshape(matrix.shape[0])
    upper = np.array(upper, dtype=float).reshape(matrix.shape[1])
    free = np.ones(matrix.shape[1], dtype=bool)
    x = np.zeros(matrix.shape[1], dtype=float)
    residual = target.copy()
    for _ in range(matrix.shape[1] + 1):
        if not np.any(free):
            break
        sol, *_ = np.linalg.lstsq(matrix[:, free], residual, rcond=None)
        trial = np.zeros_like(x)
        trial[free] = sol
        too_low = trial < 0.0
        too_high = trial > upper
        if not np.any(too_low | too_high):
            x[free] = trial[free]
            break
        fixed = free & (too_low | too_high)
        x[fixed & too_low] = 0.0
        x[fixed & too_high] = upper[fixed & too_high]
        free[fixed] = False
        residual = target - matrix @ x
    return np.clip(x, 0.0, upper)


@dataclass
class RCSAllocationAwareController(Controller):
    base_controller: Any
    thrusters: tuple[dict[str, Any], ...] | list[dict[str, Any]]
    mass_kg: float = 100.0
    allocation_mode: Literal["force_only", "torque_only", "force_torque"] = "force_only"
    torque_body_nm: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.base_controller = _construct_controller(self.base_controller)
        self.thrusters = tuple(dict(row or {}) for row in self.thrusters)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        base = self.base_controller.act(belief, t_s, budget_ms)
        desired_force_n = np.array(base.thrust_eci_km_s2, dtype=float).reshape(3) * float(max(self.mass_kg, 0.0)) * 1e3
        c_bn = np.eye(3)
        if belief.state.size >= 10:
            c_bn = quaternion_to_dcm_bn(np.array(belief.state[6:10], dtype=float).reshape(4))
        desired_force_body_n = c_bn @ desired_force_n
        desired_torque = (
            np.array(base.torque_body_nm, dtype=float).reshape(3)
            if self.torque_body_nm is None
            else np.array(self.torque_body_nm, dtype=float).reshape(3)
        )
        force_dirs = []
        torque_dirs = []
        max_forces = []
        names = []
        for idx, row in enumerate(self.thrusters):
            force_dir = _unit(np.array(row.get("force_direction_body", [1.0, 0.0, 0.0]), dtype=float))
            pos = np.array(row.get("position_body_m", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            force_dirs.append(force_dir)
            torque_dirs.append(np.cross(pos, force_dir))
            max_forces.append(float(max(row.get("max_thrust_n", 0.0), 0.0)))
            names.append(str(row.get("name", f"rcs_{idx}")))
        if not force_dirs:
            return base
        if self.allocation_mode == "torque_only":
            allocation = np.column_stack(torque_dirs)
            target = desired_torque
        elif self.allocation_mode == "force_torque":
            allocation = np.vstack((np.column_stack(force_dirs), np.column_stack(torque_dirs)))
            target = np.hstack((desired_force_body_n, desired_torque))
        else:
            allocation = np.column_stack(force_dirs)
            target = desired_force_body_n
        forces = _bounded_nonnegative_lstsq(allocation, target, np.array(max_forces, dtype=float))
        achieved_force = np.sum(np.array(force_dirs).T * forces.reshape(1, -1), axis=1)
        achieved_torque = np.sum(np.array(torque_dirs).T * forces.reshape(1, -1), axis=1)
        achieved_force_eci = c_bn.T @ achieved_force
        accel = achieved_force_eci / max(float(self.mass_kg), 1e-12) / 1e3
        mode_flags = dict(base.mode_flags or {})
        mode_flags.update(
            {
                "mode": "rcs_allocation_aware",
                "rcs_base_mode": mode_flags.get("mode"),
                "rcs_thruster_names": names,
                "rcs_thruster_forces_n": forces.tolist(),
                "rcs_force_body_n": achieved_force.tolist(),
                "rcs_force_eci_n": achieved_force_eci.tolist(),
                "rcs_force_error_n": (desired_force_body_n - achieved_force).tolist(),
                "rcs_torque_error_nm": (desired_torque - achieved_torque).tolist(),
            }
        )
        return Command(thrust_eci_km_s2=accel, torque_body_nm=achieved_torque, mode_flags=mode_flags)
