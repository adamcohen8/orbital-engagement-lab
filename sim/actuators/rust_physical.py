"""Opt-in native interval calculations behind the physical hardware owner."""

from __future__ import annotations

import numpy as np

from sim.flight_software.contracts import TelemetryField
from sim.flight_software.rust_actuator_backend import extension


def enable_hardware(hardware):
    from sim.actuators.physical import (
        CmgHardware,
        ContinuousEngineHardware,
        IdealWrenchHardware,
        MagnetorquerHardware,
        ReactionWheelHardware,
    )

    if type(hardware) in (
        IdealWrenchHardware,
        ReactionWheelHardware,
        MagnetorquerHardware,
        CmgHardware,
        ContinuousEngineHardware,
    ):
        extension()
        hardware._native_actuator_intervals = True


def advance_hardware(hardware, demand, start_time_ns, end_time_ns):
    """Called after the existing hardware owner's interval/payload checks."""
    from sim.actuators.physical import (
        ActuatorRealization,
        CmgHardware,
        ContinuousEngineHardware,
        IdealWrenchHardware,
        MagnetorquerHardware,
        ReactionWheelHardware,
    )

    payload = demand.payload
    axes = None
    state = []
    if type(hardware) is IdealWrenchHardware:
        kind = 0
        params = [
            hardware.max_force_n,
            hardware.max_torque_n_m,
            hardware.response_time_constant_s,
            hardware.specific_impulse_s or 0.0,
        ]
        request = [0.0] * 6 if payload is None else list(payload.force_n) + list(payload.torque_n_m)
        state = list(hardware.realized_force_n) + list(hardware.realized_torque_n_m)
    elif type(hardware) is ReactionWheelHardware:
        kind = 1
        params = hardware.max_torque_n_m.tolist() + hardware.max_momentum_n_m_s.tolist()
        axes = hardware.axes_body.T
        request = [0.0] * len(hardware.momentum_n_m_s) if payload is None else list(payload.torque_n_m)
        if len(request) != hardware.axes_body.shape[0]:
            raise ValueError("reaction-wheel command count must match configured wheels")
        state = hardware.momentum_n_m_s.tolist()
    elif type(hardware) is MagnetorquerHardware:
        kind = 2
        params = hardware.max_dipole_a_m2.tolist() + hardware.magnetic_field_body_t.tolist()
        request = [0.0] * 3 if payload is None else list(payload.dipole_a_m2)
    elif type(hardware) is CmgHardware:
        kind = 3
        params = hardware.momentum_n_m_s.tolist() + hardware.max_gimbal_rate_rad_s.tolist()
        request = [0.0] * 3 if payload is None else list(payload.gimbal_rate_rad_s)
        state = hardware.gimbal_angle_rad.tolist()
    elif type(hardware) is ContinuousEngineHardware:
        kind = 4
        params = [hardware.max_thrust_n, hardware.specific_impulse_s or 0.0]
        angles = () if payload is None else payload.gimbal_angles_rad
        request = [
            0.0 if payload is None else float(payload.throttle_0_1),
            float(angles[0]) if len(angles) > 0 else 0.0,
            float(angles[1]) if len(angles) > 1 else 0.0,
        ]
    else:
        raise TypeError("unsupported native actuator hardware")
    key = (kind, tuple(params), None if axes is None else (axes.shape, axes.tobytes()))
    cached = getattr(hardware, "_native_interval_context", None)
    if cached is None or cached[0] != key:
        cached = (key, extension().ActuatorIntervalContext(kind, params, axes, np.matmul))
        hardware._native_interval_context = cached
    requested_force, requested_torque, force, torque, next_state, saturated, flow, channels = cached[1].advance(
        request, state, (end_time_ns - start_time_ns) / 1.0e9
    )
    telemetry = ()
    if kind == 0:
        hardware.realized_force_n = tuple(force)
        hardware.realized_torque_n_m = tuple(torque)
    elif kind == 1:
        hardware.momentum_n_m_s = np.asarray(next_state)
        telemetry = tuple(
            TelemetryField(f"wheel_{i}_momentum_n_m_s", float(v), "N*m*s") for i, v in enumerate(next_state)
        )
        if hardware.inertia_kg_m2 is not None:
            telemetry += tuple(
                TelemetryField(f"wheel_{i}_speed_rad_s", float(v), "rad/s")
                for i, v in enumerate(hardware.momentum_n_m_s / hardware.inertia_kg_m2)
            )
    elif kind == 2:
        telemetry = tuple(TelemetryField(f"dipole_{axis}_a_m2", float(v), "A*m^2") for axis, v in zip("xyz", channels))
    elif kind == 3:
        hardware.gimbal_angle_rad = np.asarray(next_state)
        telemetry = tuple(
            TelemetryField(f"gimbal_{axis}_angle_rad", float(v), "rad") for axis, v in zip("xyz", next_state)
        )
    else:
        telemetry = tuple(
            TelemetryField(name, float(v), unit)
            for name, v, unit in zip(("throttle", "gimbal_yaw", "gimbal_pitch"), channels, ("fraction", "rad", "rad"))
        )
    return ActuatorRealization(
        hardware.actuator_id,
        start_time_ns,
        end_time_ns,
        None if demand.source_command is None else demand.source_command.command_id,
        demand.mode,
        tuple(requested_force),
        tuple(requested_torque),
        tuple(force),
        tuple(torque),
        mass_flow_kg_s=flow,
        device_state=telemetry,
        saturated=saturated,
    )
