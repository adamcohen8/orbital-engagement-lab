"""Attitude-controller exports, loaded only when requested."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AtmosphericLiftAxisController": "sim.control.attitude.aero_assist",
    "QuaternionPDController": "sim.control.attitude.baseline",
    "ReactionWheelPDController": "sim.control.attitude.baseline",
    "ReactionWheelPIDController": "sim.control.attitude.baseline",
    "SmallAngleLQRController": "sim.control.attitude.baseline",
    "MagnetorquerBdotController": "sim.control.attitude.bdot_magnetorquer",
    "CMGSteeringController": "sim.control.attitude.cmg_steering",
    "ECIDetumblePDController": "sim.control.attitude.detumble_pd",
    "RICDetumblePDController": "sim.control.attitude.detumble_pd",
    "PoseCommandGenerator": "sim.control.attitude.pose_commands",
    "RICFrameLQRController": "sim.control.attitude.ric_lqr",
    "RICFramePDController": "sim.control.attitude.ric_pd",
    "RICFramePIDController": "sim.control.attitude.ric_pid",
    "SnapAttitudeController": "sim.control.attitude.snap",
    "SnapAndHoldRICAttitudeController": "sim.control.attitude.snap_hold",
    "SurrogateSnapECIController": "sim.control.attitude.surrogate_snap",
    "SurrogateSnapRICController": "sim.control.attitude.surrogate_snap",
    "DetumbleThenSlewController": "sim.control.attitude.switching",
    "WheelDesaturationController": "sim.control.attitude.wheel_desaturation",
    "ZeroTorqueController": "sim.control.attitude.zero_torque",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
