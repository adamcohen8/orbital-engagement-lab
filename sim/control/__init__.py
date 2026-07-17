"""Controller exports, loaded only when requested."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ATTITUDE_EXPORTS = {
    "AtmosphericLiftAxisController",
    "CMGSteeringController",
    "DetumbleThenSlewController",
    "ECIDetumblePDController",
    "MagnetorquerBdotController",
    "PoseCommandGenerator",
    "QuaternionPDController",
    "ReactionWheelPDController",
    "ReactionWheelPIDController",
    "RICDetumblePDController",
    "RICFrameLQRController",
    "RICFramePDController",
    "RICFramePIDController",
    "SmallAngleLQRController",
    "SnapAndHoldRICAttitudeController",
    "SnapAttitudeController",
    "SurrogateSnapECIController",
    "SurrogateSnapRICController",
    "WheelDesaturationController",
    "ZeroTorqueController",
}

_ORBIT_EXPORTS = {
    "AtmosphericPassController",
    "AttitudeAgnosticImpulsiveManeuverer",
    "DeltaVManeuver",
    "ElectricPropulsionController",
    "GimbaledThrusterController",
    "HCWCurvInputRectOutputController",
    "HCWInTrackCrossTrackMPCController",
    "HCWLQRController",
    "HCWNoRadialLQRController",
    "HCWNoRadialManualController",
    "HCWRelativeOrbitMPCController",
    "ImpulsiveManeuver",
    "ImpulsiveManeuverResult",
    "IntegratedManeuverCommand",
    "IntegratedManeuverDecision",
    "OrbitalAttitudeManeuverCoordinator",
    "OrbitalElementsFeedbackController",
    "PredictiveBurnConfig",
    "PredictiveBurnScheduler",
    "RCSAllocationAwareController",
    "RelativeOrbitMPCController",
    "RiskThresholdController",
    "RobustMPCController",
    "SafetyBarrierController",
    "ScheduledImpulseController",
    "SemiMajorAxisEccentricityController",
    "StationkeepingController",
    "StochasticPolicyController",
    "ThrustLimitedDeltaVManeuver",
    "ThrustLimitedDeltaVManeuverResult",
    "ZeroController",
}

_EXPORTS = {
    **{name: "sim.control.attitude" for name in _ATTITUDE_EXPORTS},
    **{name: "sim.control.orbit" for name in _ORBIT_EXPORTS},
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
