from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Actuator
from sim.core.models import Command


@dataclass(frozen=True)
class ActuatorFaultConfig:
    stuck_off: bool = False
    thrust_scale: float = 1.0
    torque_scale: float = 1.0
    thrust_bias_eci_km_s2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_bias_body_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class FaultedActuator(Actuator):
    base: Actuator
    faults: ActuatorFaultConfig = field(default_factory=ActuatorFaultConfig)

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        applied = self.base.apply(command, limits, dt_s)
        return apply_actuator_faults(applied, self.faults)


def apply_actuator_faults(command: Command, faults: ActuatorFaultConfig) -> Command:
    mode_flags = dict(command.mode_flags)
    if bool(faults.stuck_off):
        mode_flags["actuator_fault_stuck_off"] = True
        return Command(
            thrust_eci_km_s2=np.zeros(3, dtype=float),
            torque_body_nm=np.zeros(3, dtype=float),
            mode_flags=mode_flags,
        )
    thrust = np.array(command.thrust_eci_km_s2, dtype=float).reshape(3)
    torque = np.array(command.torque_body_nm, dtype=float).reshape(3)
    thrust = thrust * float(faults.thrust_scale) + np.array(faults.thrust_bias_eci_km_s2, dtype=float).reshape(3)
    torque = torque * float(faults.torque_scale) + np.array(faults.torque_bias_body_nm, dtype=float).reshape(3)
    mode_flags["actuator_fault_thrust_scale"] = float(faults.thrust_scale)
    mode_flags["actuator_fault_torque_scale"] = float(faults.torque_scale)
    return Command(thrust_eci_km_s2=thrust, torque_body_nm=torque, mode_flags=mode_flags)
