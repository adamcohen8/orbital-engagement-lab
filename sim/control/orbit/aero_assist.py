from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM


@dataclass
class AtmosphericPassController(Controller):
    """Coast through an atmospheric pass, then apply a simple prograde raise burn."""

    raise_start_s: float = 180.0
    raise_end_s: float = 260.0
    prograde_accel_km_s2: float = 1.0e-5
    min_raise_altitude_km: float | None = None

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if belief.state.size < 6:
            return Command.zero()
        r_eci = np.array(belief.state[:3], dtype=float)
        v_eci = np.array(belief.state[3:6], dtype=float)
        altitude_km = float(np.linalg.norm(r_eci) - EARTH_RADIUS_KM)
        phase = "atmospheric_pass"
        accel = np.zeros(3, dtype=float)
        in_raise_window = float(self.raise_start_s) <= float(t_s) <= float(self.raise_end_s)
        altitude_ok = self.min_raise_altitude_km is None or altitude_km >= float(self.min_raise_altitude_km)
        speed = float(np.linalg.norm(v_eci))
        if in_raise_window and altitude_ok and speed > 0.0 and float(self.prograde_accel_km_s2) > 0.0:
            accel = v_eci / speed * float(self.prograde_accel_km_s2)
            phase = "raise_burn"
        return Command(
            thrust_eci_km_s2=accel,
            torque_body_nm=np.zeros(3, dtype=float),
            mode_flags={
                "mode": "atmospheric_pass",
                "phase": phase,
                "altitude_km": altitude_km,
                "raise_window_active": bool(in_raise_window),
            },
        )
