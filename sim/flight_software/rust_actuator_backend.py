"""Optional native numerical contexts for the maintained actuator pipeline.

Python owns policy, boundary records and checkpoint state. NumPy reductions
and matrix products retain their existing arithmetic to preserve trajectories.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np

from sim.flight_software.contracts import (
    ActuatorCommand,
    CmgGimbalRateCommand,
    IdealWrenchCommand,
    MagnetorquerDipoleCommand,
    ReactionWheelTorqueCommand,
)
from sim.gnc.attitude_v2 import AttitudeAllocator, AttitudeAllocatorKind, QuaternionTorqueController
from sim.gnc.contracts import AllocationResult, AllocationStatus, RequestedEffort, RequestedEffortKind


@lru_cache(maxsize=1)
def extension():
    from sim.rust_orbit_backend import _extension

    native = _extension()
    if not all(hasattr(native, name) for name in ("AttitudeActuationContext", "ActuatorIntervalContext")):
        raise RuntimeError("Rust actuator pipeline requires oel-rust-orbit >=0.12.0")
    return native


@lru_cache(maxsize=1)
def implementation_digest():
    from sim.flight_software.rust_game_backend import extension as packets_extension

    digest = sha256(Path(__file__).read_bytes())
    for native in (extension(), packets_extension()):
        binary = getattr(native, "oel_rust_orbit", getattr(native, "oel_rust_game", native))
        digest.update(Path(binary.__file__).read_bytes())
    for owner in (
        "actuators/rust_physical.py",
        "actuators/physical.py",
        "actuators/command_bus.py",
        "runtime/satellites/flight_software_runtime.py",
    ):
        digest.update((Path(__file__).parents[1] / owner).read_bytes())
    for name in ("reference_stacks.py", "rust_game_packets.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


class NativeAttitudeAllocator(AttitudeAllocator):
    def __init__(self, config):
        super().__init__(config)
        self._native_kind = {
            AttitudeAllocatorKind.IDEAL_WRENCH: 0,
            AttitudeAllocatorKind.REACTION_WHEEL: 1,
            AttitudeAllocatorKind.MAGNETORQUER: 2,
            AttitudeAllocatorKind.CMG: 3,
        }[config.kind]
        axes = np.asarray(config.axes_body, dtype=float).T
        self._native_axes = axes
        self._native_inverse = np.linalg.pinv(axes) if self._native_kind == 1 else None
        count = axes.shape[1] if self._native_kind == 1 else 3
        self._native_limits = list(config.limits) * count if len(config.limits) == 1 else list(config.limits)
        self._context_key = None
        self._context = None
        self._get_context(None)

    def _get_context(self, controller):
        key = (
            None
            if controller is None
            else (controller.kp, controller.kd, controller.max_torque_n_m, controller.detumble_rate_threshold_rad_s)
        )
        if self._context is None or key != self._context_key:
            kp, kd, maximum, detumble = ((0.0,) * 3, (0.0,) * 3, 1.0, 0.0) if key is None else key
            self._context = extension().AttitudeActuationContext(
                self._native_kind,
                kp,
                kd,
                maximum,
                detumble,
                self._native_limits,
                list(self.config.cmg_momentum_n_m_s),
                self._native_inverse,
                self._native_axes,
                np.dot,
                np.matmul,
            )
            self._context_key = key
        return self._context

    def _result(self, effort, solution, command_id, row):
        channels, residual, status = row
        if status == 3:
            return AllocationResult(effort.effort_id, effort.generated_at, AllocationStatus.INFEASIBLE)
        values = tuple(channels)
        payload = (
            IdealWrenchCommand((0.0, 0.0, 0.0), values)
            if self._native_kind == 0
            else ReactionWheelTorqueCommand(values)
            if self._native_kind == 1
            else MagnetorquerDipoleCommand(values)
            if self._native_kind == 2
            else CmgGimbalRateCommand(values)
        )
        command = ActuatorCommand(
            command_id,
            self.config.satellite_id,
            self.config.actuator_id,
            solution.generated_at,
            effort.validity,
            self.config.actuator_frame,
            payload,
        )
        return AllocationResult(
            effort.effort_id,
            solution.generated_at,
            (AllocationStatus.EXACT, AllocationStatus.SATURATED, AllocationStatus.RESIDUAL)[status],
            (command,),
            residual_torque_n_m=tuple(residual),
        )

    def allocate(self, effort, solution, *, command_id):
        if effort.frame != solution.frame or effort.torque_n_m is None:
            return super().allocate(effort, solution, command_id=command_id)
        row = self._get_context(None).allocate(effort.torque_n_m, solution.magnetic_field_body_t)
        return self._result(effort, solution, command_id, row)

    def fused_control(self, solution, reference, controller, next_command_id):
        if type(controller) is not QuaternionTorqueController:
            effort = controller.control(solution, reference)
            return effort, None if effort is None else self.allocate(effort, solution, command_id=next_command_id())
        if not solution.valid_for_control or reference.attitude_quat_from_frame is None:
            return None, None
        torque, channels, residual, status = self._get_context(controller).control_allocate(
            solution.attitude_quat_bn,
            reference.attitude_quat_from_frame,
            solution.angular_rate_body_rad_s,
            solution.magnetic_field_body_t,
        )
        effort = RequestedEffort(
            "attitude-torque",
            RequestedEffortKind.TORQUE,
            solution.generated_at,
            solution.frame,
            reference.validity,
            torque_n_m=tuple(torque),
        )
        return effort, self._result(effort, solution, next_command_id(), (channels, residual, status))
