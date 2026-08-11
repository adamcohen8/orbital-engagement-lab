"""Orbit-controller exports, loaded only when requested."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {}


def _export(module: str, *names: str) -> None:
    _EXPORTS.update((name, module) for name in names)


_export("sim.control.orbit.advanced", "RobustMPCController", "StochasticPolicyController")
_export("sim.control.orbit.aero_assist", "AtmosphericPassController")
_export(
    "sim.control.orbit.baseline",
    "OrbitalElementsFeedbackController",
    "RiskThresholdController",
    "SafetyBarrierController",
    "SemiMajorAxisEccentricityController",
    "StationkeepingController",
)
_export("sim.control.orbit.curv_pd", "CurvilinearRICPDController", "curv_accel_to_rect")
_export("sim.control.orbit.electric_propulsion", "ElectricPropulsionController")
_export("sim.control.orbit.gimbaled_thruster", "GimbaledThrusterController")
_export(
    "sim.control.orbit.hcw_mpc",
    "HCWInTrackCrossTrackMPCController",
    "HCWRelativeOrbitMPCController",
    "RelativeLinearOrbitMPCController",
    "SSJ2RelativeOrbitMPCController",
)
_export("sim.control.orbit.hcw_pd", "HCWPDController", "RelativeLinearPDController", "SSJ2PDController")
_export(
    "sim.control.orbit.hcw_transfer",
    "HCWEvasionOptimizationResult",
    "HCWPositionTransferSolution",
    "hcw_phi_rv",
    "hcw_state_transition_blocks",
    "hcw_state_transition_matrix",
    "optimize_hcw_evasion_burn_direction",
    "propagate_hcw_relative_state",
    "propagate_linear_relative_state",
    "relative_state_transition_blocks",
    "solve_hcw_position_rendezvous",
    "solve_linear_position_rendezvous",
)
_export(
    "sim.control.orbit.impulsive",
    "AttitudeAgnosticImpulsiveManeuverer",
    "DeltaVManeuver",
    "ImpulsiveManeuver",
    "ImpulsiveManeuverResult",
    "ThrustLimitedDeltaVManeuver",
    "ThrustLimitedDeltaVManeuverResult",
)
_export(
    "sim.control.orbit.integrated",
    "IntegratedManeuverCommand",
    "IntegratedManeuverDecision",
    "OrbitalAttitudeManeuverCoordinator",
)
_export("sim.control.orbit.lqr", "HCWLQRController", "RelativeLinearLQRController", "SSJ2LQRController")
_export("sim.control.orbit.lqr_curv_variant", "HCWCurvInputRectOutputController")
_export(
    "sim.control.orbit.lqr_no_radial",
    "HCWNoRadialLQRController",
    "HCWNoRadialManualController",
    "SSJ2NoRadialLQRController",
)
_export("sim.control.orbit.predictive_burn", "PredictiveBurnConfig", "PredictiveBurnScheduler")
_export("sim.control.orbit.rcs_allocator", "RCSAllocationAwareController")
_export("sim.control.orbit.relative_mpc", "RelativeOrbitMPCController")
_export("sim.control.orbit.ric_pd", "RICPDTransferController")
_export("sim.control.orbit.rmoe", "RMOEIfThenController", "estimate_rmoes_from_rect_ric")
_export(
    "sim.control.orbit.reference_rpo",
    "CBarApproachController",
    "KeepOutStandoffController",
    "HCWRendezvousPlannerController",
    "LowThrustPhasingController",
    "PlaneChangeTrimController",
    "ProportionalNavigationController",
    "PassiveSafeRetreatController",
    "RBarApproachController",
    "RICApproachController",
    "RICRelativeHoldController",
    "RICFlyaroundController",
    "RICWaypointController",
    "TerminalBrakingController",
    "VBarApproachController",
)
_export("sim.control.orbit.scheduled_impulse", "ScheduledImpulseController")
_export("sim.control.orbit.zero_controller", "ZeroController")

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
