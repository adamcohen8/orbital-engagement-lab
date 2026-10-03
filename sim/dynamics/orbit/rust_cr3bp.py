"""Optional native CR3BP binding; physical rotating barycentric km/km/s."""

from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.integrators import AdaptiveStepInfo
from sim.rust_orbit_backend import _extension


def integrate(
    state: np.ndarray,
    dt_s: float,
    t_s: float,
    *,
    system,
    command: np.ndarray | None = None,
    stm: bool = False,
    integrator: str = "rk4",
    adaptive_atol: float = 1e-9,
    adaptive_rtol: float = 1e-7,
    h_init: float | None = None,
    collect_info: bool = True,
) -> tuple[np.ndarray, AdaptiveStepInfo | None]:
    name = "cr3bp_propagate_stm" if stm else "cr3bp_propagate"
    extension = _extension()
    native = getattr(extension, "cr3bp_propagate_stm_buffer", None) if stm else None
    buffered = native is not None
    if not buffered:
        native = getattr(extension, name, None)
    if native is None:
        raise RuntimeError("Rust CR3BP requires a CR3BP-enabled oel_rust_orbit wheel (0.6.0 or newer)")
    values = np.asarray(state, dtype=np.float64).reshape(42 if stm else 6)
    method = str(integrator).strip().lower()
    if buffered:
        out, raw = native(
            values.tolist(), float(t_s), float(dt_s), float(system.distance_km),
            float(system.mu), float(system.mean_motion_rad_s), method,
            float(adaptive_atol), float(adaptive_rtol),
            None if h_init is None else float(h_init), bool(collect_info),
        )
    elif stm:
        out, raw = native(
            values.tolist(), float(t_s), float(dt_s), float(system.distance_km),
            float(system.mu), float(system.mean_motion_rad_s), method,
            float(adaptive_atol), float(adaptive_rtol),
            None if h_init is None else float(h_init),
        )
    else:
        acceleration = [0.0, 0.0, 0.0] if command is None else np.asarray(command, dtype=np.float64).reshape(3).tolist()
        out, raw = native(
            values.tolist(), float(t_s), float(dt_s), float(system.distance_km),
            float(system.mu), float(system.mean_motion_rad_s), acceleration, method,
            float(adaptive_atol), float(adaptive_rtol),
            None if h_init is None else float(h_init),
        )
    info = None if raw is None or not collect_info else AdaptiveStepInfo(
        method="rkf78" if method == "adaptive" else method,
        accepted_steps=raw[0], rejected_steps=raw[1], attempted_steps=raw[2],
        min_step_s=raw[3], max_step_s=raw[4], final_step_s=raw[5],
        suggested_next_step_s=raw[6], max_error_ratio=raw[7],
    )
    values = np.frombuffer(out, dtype="<f8") if buffered else np.asarray(out, dtype=np.float64)
    return values, info
