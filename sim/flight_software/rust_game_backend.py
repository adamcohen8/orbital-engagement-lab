"""Native numerical components for the default Trainer stacks.

Typed events, qualification, mission policy, hardware, and scoring stay with
OEL's existing owners. Rust selection fails closed; it never falls
back to Python when an extension is missing.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from importlib import import_module
from pathlib import Path

import numpy as np

from sim.flight_software.contracts import IdealWrenchCommand, TelemetryField
from sim.gnc.attitude_v2 import QuaternionTorqueController
from sim.gnc.contracts import AllocationResult, AllocationStatus, RequestedEffort, RequestedEffortKind
from sim.gnc.navigation_v2 import OrbitNavigator
from sim.gnc.orbit_v2 import TranslationAllocator, TranslationAllocatorKind


@lru_cache(maxsize=1)
def extension():
    try:
        native = import_module("oel_rust_game")
    except ImportError as exc:
        raise RuntimeError(
            "Rust game backend requires the oel_rust_game wheel; install OEL with the game profile "
            "or launch with --backend python"
        ) from exc
    required = (
        "TrustedPacketEncoder",
        "BoundaryValidator",
        "FrameConverter",
        "preview_two_body",
        "preview_th_history",
        "score_sample",
        "pilot_force",
        "ric_to_eci",
        "ideal_wrench",
        "advance_attitude",
        "attitude_torque",
        "pd_feedback",
        "retreat",
        "predict_batch",
        "transfer_control",
        "effector_positions",
        "relative_observation",
    )
    if getattr(native, "BACKEND_ABI", None) != 1 or any(not callable(getattr(native, key, None)) for key in required):
        raise RuntimeError("Rust game backend requires a compatible oel_rust_game wheel (ABI 1)")
    return native


@lru_cache(maxsize=1)
def implementation_digest() -> str:
    native = extension()
    # Bind snapshot and qualification identity to the actual installed binary.
    binary = getattr(native, "oel_rust_game", native)
    digest = sha256(Path(binary.__file__).read_bytes())
    for name in ("rust_game_backend.py", "rust_game_control.py", "rust_game_stacks.py", "rust_game_packets.py", "rust_game_frames.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


class RustOrbitNavigator(OrbitNavigator):
    numeric_backend = "rust"

    def _relative_observation_state(self, los, transform, range_m, range_rate, angular_rate):
        state = extension().relative_observation(
            tuple(los),
            transform.ravel().tolist(),
            range_m,
            range_rate,
            None if angular_rate is None else tuple(angular_rate),
        )
        state = np.asarray(state)
        return state[:3], state[3:]

    @classmethod
    def from_navigator(cls, navigator: OrbitNavigator):
        native = cls.__new__(cls)
        native.__dict__.update(navigator.__dict__)
        return native


class RustQuaternionTorqueController(QuaternionTorqueController):
    def control(self, solution, reference):
        if not solution.valid_for_control or reference.attitude_quat_from_frame is None:
            return None
        torque = extension().attitude_torque(
            solution.attitude_quat_bn,
            reference.attitude_quat_from_frame,
            solution.angular_rate_body_rad_s,
            self.kp,
            self.kd,
            self.max_torque_n_m,
            self.detumble_rate_threshold_rad_s,
        )
        return RequestedEffort(
            "attitude-torque",
            RequestedEffortKind.TORQUE,
            solution.generated_at,
            solution.frame,
            reference.validity,
            torque_n_m=tuple(torque),
        )


def native_torque_controller(controller):
    if type(controller) is not QuaternionTorqueController:
        raise ValueError("Rust game backend supports the built-in quaternion torque controller")
    return RustQuaternionTorqueController(
        controller.kp, controller.kd, controller.max_torque_n_m, controller.detumble_rate_threshold_rad_s
    )


class RustTranslationAllocator(TranslationAllocator):
    def __init__(self, config):
        if config.kind is not TranslationAllocatorKind.IDEAL_WRENCH:
            raise ValueError("Rust game backend supports ideal-wrench translation allocation")
        super().__init__(config)

    def allocate(self, effort, solution, *, next_command_id, unavailable_actuators=frozenset()):
        if effort.force_n is None:
            return AllocationResult(effort.effort_id, effort.generated_at, AllocationStatus.INVALID)
        if self.config.actuator_id in unavailable_actuators or solution.attitude.attitude_quat_bn is None:
            return AllocationResult(effort.effort_id, effort.generated_at, AllocationStatus.INFEASIBLE)
        body, achieved, scale = extension().ideal_wrench(
            effort.force_n,
            solution.attitude.attitude_quat_bn,
            self.config.max_force_n,
        )
        residual = np.asarray(effort.force_n) - np.asarray(achieved)
        command = self._command(next_command_id(), effort, IdealWrenchCommand(tuple(body), (0.0, 0.0, 0.0)))
        return AllocationResult(
            effort.effort_id,
            effort.generated_at,
            AllocationStatus.SATURATED if scale < 1.0 else AllocationStatus.EXACT,
            (command,),
            residual_force_n=tuple(float(x) for x in residual),
            status_details=(
                TelemetryField("allocator_requested_force_n", float(np.linalg.norm(effort.force_n)), "N"),
                TelemetryField("residual_force_n", float(np.linalg.norm(residual)), "N"),
            ),
        )
