"""Authoritative identities and product metadata for built-in OEL GNC surfaces."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from typing import Any


@dataclass(frozen=True)
class GNCDescriptor:
    builtin_id: str
    display_name: str
    category: str
    module: str
    class_name: str
    maturity: str
    packaging: str
    summary: str
    aliases: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    known_limits: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    parameter_units: tuple[tuple[str, str], ...] = ()

    def target(self) -> tuple[str, str]:
        return self.module, self.class_name

    def parameter_schema(self) -> dict[str, dict[str, Any]]:
        """Return an inspectable constructor schema for this trusted built-in."""
        cls = getattr(importlib.import_module(self.module), self.class_name)
        units = dict(self.parameter_units)
        schema: dict[str, dict[str, Any]] = {}
        if is_dataclass(cls):
            for item in fields(cls):
                if not item.init:
                    continue
                required = item.default is MISSING and item.default_factory is MISSING
                default: Any = None
                if item.default is not MISSING:
                    default = item.default
                elif item.default_factory is not MISSING:
                    try:
                        default = item.default_factory()
                    except Exception:
                        default = "<factory>"
                schema[item.name] = {
                    "required": required,
                    "default": _jsonable(default),
                    "annotation": _annotation_name(item.type),
                    "units": units.get(item.name, _infer_units(item.name)),
                }
            return schema
        signature = inspect.signature(cls)
        for name, item in signature.parameters.items():
            if name == "self" or item.kind in (item.VAR_POSITIONAL, item.VAR_KEYWORD):
                continue
            schema[name] = {
                "required": item.default is inspect.Signature.empty,
                "default": None if item.default is inspect.Signature.empty else _jsonable(item.default),
                "annotation": _annotation_name(item.annotation),
                "units": units.get(name, _infer_units(name)),
            }
        return schema

    def to_dict(self, *, include_parameters: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameter_units"] = dict(self.parameter_units)
        if include_parameters:
            payload["parameters"] = self.parameter_schema()
        return payload


def _annotation_name(value: Any) -> str:
    if value is inspect.Signature.empty:
        return "Any"
    return getattr(value, "__name__", str(value))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def _infer_units(name: str) -> str:
    suffixes = (
        ("_km_s2", "km/s^2"),
        ("_km_s", "km/s"),
        ("_rad_s", "rad/s"),
        ("_rad", "rad"),
        ("_deg", "deg"),
        ("_km", "km"),
        ("_m_s", "m/s"),
        ("_nm", "N*m"),
        ("_n", "N"),
        ("_kg", "kg"),
        ("_w", "W"),
        ("_s", "s"),
    )
    for suffix, units in suffixes:
        if name.endswith(suffix):
            return units
    return "dimensionless_or_structured"


def _entry(
    builtin_id: str,
    display_name: str,
    category: str,
    module: str,
    class_name: str,
    maturity: str,
    summary: str,
    *,
    packaging: str = "public",
    aliases: tuple[str, ...] = (),
    assumptions: tuple[str, ...] = (),
    known_limits: tuple[str, ...] = (),
    examples: tuple[str, ...] = (),
    tests: tuple[str, ...] = (),
    parameter_units: tuple[tuple[str, str], ...] = (),
) -> GNCDescriptor:
    return GNCDescriptor(
        builtin_id=builtin_id,
        display_name=display_name,
        category=category,
        module=module,
        class_name=class_name,
        maturity=maturity,
        packaging=packaging,
        summary=summary,
        aliases=aliases,
        assumptions=assumptions,
        known_limits=known_limits,
        examples=examples,
        tests=tests,
        parameter_units=parameter_units,
    )


_ENTRIES: tuple[GNCDescriptor, ...] = (
    # Orbit controllers and actuator-aware wrappers.
    _entry("orbit.zero", "Zero Controller", "orbit_controller", "sim.control.orbit.zero_controller", "ZeroController", "reference", "Coast/no-thrust baseline."),
    _entry("orbit.ric_pd_transfer", "RIC PD Transfer", "orbit_controller", "sim.control.orbit.ric_pd", "RICPDTransferController", "flagship", "Guided rectangular-RIC transfer with coast, correction, braking, and terminal PD phases.", assumptions=("Near-circular chief for HCW/SS-J2 transfer model use.",), examples=("configs/ric_pd_10km_experiment.yaml",), tests=("sim/tests/test_ric_pd_transfer.py",)),
    _entry("orbit.ric_pd_hold", "RIC PD Hold", "orbit_controller", "sim.control.orbit.hcw_pd", "HCWPDController", "workbench", "Rectangular-RIC PD hold with optional HCW or SS-J2 feedforward.", aliases=("orbit.hcw_pd",), known_limits=("Compatibility class name remains HCWPDController during product rename.",)),
    _entry("orbit.hcw_lqr", "HCW LQR", "orbit_controller", "sim.control.orbit.lqr", "HCWLQRController", "reference", "Discrete LQR for circular-chief HCW relative motion.", assumptions=("Circular chief and small relative separation.",), examples=("examples/configs/public_closed_loop_rendezvous_lqr.yaml",), tests=("sim/tests/test_orbit_hcw_lqr.py",)),
    _entry("orbit.ss_j2_lqr", "SS-J2 LQR", "orbit_controller", "sim.control.orbit.lqr", "SSJ2LQRController", "workbench", "LQR using homogeneous Schweighart-Sedwick averaged-J2 relative dynamics.", assumptions=("Near-circular Earth chief within the configured SS-J2 envelope.",)),
    _entry("orbit.hcw_lqr_no_radial", "HCW LQR No Radial", "orbit_controller", "sim.control.orbit.lqr_no_radial", "HCWNoRadialLQRController", "workbench", "HCW LQR constrained to in-track and cross-track acceleration.", tests=("sim/tests/test_orbit_hcw_lqr_no_radial.py",)),
    _entry("orbit.hcw_manual_no_radial", "HCW Manual Gain No Radial", "orbit_controller", "sim.control.orbit.lqr_no_radial", "HCWNoRadialManualController", "workbench", "Manual-gain no-radial relative controller."),
    _entry("orbit.ss_j2_lqr_no_radial", "SS-J2 LQR No Radial", "orbit_controller", "sim.control.orbit.lqr_no_radial", "SSJ2NoRadialLQRController", "workbench", "SS-J2 LQR constrained to in-track and cross-track acceleration."),
    _entry("orbit.hcw_curvilinear_input", "HCW Curvilinear Input Variant", "orbit_controller", "sim.control.orbit.lqr_curv_variant", "HCWCurvInputRectOutputController", "workbench", "Compatibility wrapper for curvilinear-RIC input and rectangular feedback output."),
    _entry("orbit.curvilinear_ric_pd", "Curvilinear RIC PD", "orbit_controller", "sim.control.orbit.curv_pd", "CurvilinearRICPDController", "workbench", "PD feedback in curvilinear RIC coordinates.", tests=("sim/tests/test_orbit_curv_pd.py",)),
    _entry("orbit.relative_mpc", "Relative Orbit MPC", "orbit_controller", "sim.control.orbit.relative_mpc", "RelativeOrbitMPCController", "experimental", "Two-body nonlinear relative-orbit MPC workbench.", known_limits=("Convergence and tuning envelope are not a Reference claim.",)),
    _entry("orbit.hcw_mpc", "HCW Relative MPC", "orbit_controller", "sim.control.orbit.hcw_mpc", "HCWRelativeOrbitMPCController", "experimental", "Finite-horizon MPC using HCW prediction."),
    _entry("orbit.hcw_mpc_in_cross", "HCW MPC In/Cross Track", "orbit_controller", "sim.control.orbit.hcw_mpc", "HCWInTrackCrossTrackMPCController", "experimental", "HCW MPC constrained to in-track and cross-track acceleration."),
    _entry("orbit.ss_j2_mpc", "SS-J2 Relative MPC", "orbit_controller", "sim.control.orbit.hcw_mpc", "SSJ2RelativeOrbitMPCController", "experimental", "Finite-horizon MPC using SS-J2 prediction."),
    _entry("orbit.ss_j2_pd", "SS-J2 PD", "orbit_controller", "sim.control.orbit.hcw_pd", "SSJ2PDController", "workbench", "Rectangular-RIC PD with SS-J2 feedforward."),
    _entry("orbit.rmoe_if_then", "RMOE If-Then", "orbit_controller", "sim.control.orbit.rmoe", "RMOEIfThenController", "workbench", "Rule-based relative mean orbital-element/NMC targeting.", examples=("configs/rmoe_if_then_nmc_demo.yaml",)),
    _entry("orbit.scheduled_impulse", "Scheduled Impulse", "orbit_controller", "sim.control.orbit.scheduled_impulse", "ScheduledImpulseController", "reference", "Deterministic interval-aware scheduled acceleration command."),
    _entry("orbit.atmospheric_pass", "Atmospheric Pass", "orbit_controller", "sim.control.orbit.aero_assist", "AtmosphericPassController", "workbench", "Timed coast and raise-burn control for bounded aero-assist studies."),
    _entry("orbit.stationkeeping", "ECI Stationkeeping", "orbit_controller", "sim.control.orbit.baseline", "StationkeepingController", "reference", "Simple absolute ECI state feedback."),
    _entry("orbit.sma_ecc_feedback", "SMA/Eccentricity Feedback", "orbit_controller", "sim.control.orbit.baseline", "SemiMajorAxisEccentricityController", "reference", "Low-thrust semi-major-axis and eccentricity regulation."),
    _entry("orbit.coe_feedback", "COE Feedback", "orbit_controller", "sim.control.orbit.baseline", "OrbitalElementsFeedbackController", "workbench", "Selected classical-orbital-element feedback."),
    _entry("orbit.safety_barrier", "Safety Barrier", "orbit_controller", "sim.control.orbit.baseline", "SafetyBarrierController", "experimental", "Repulsive keep-out prototype."),
    _entry("orbit.risk_threshold", "Risk Threshold", "orbit_controller", "sim.control.orbit.baseline", "RiskThresholdController", "internal", "Callable-based nominal/evasive adapter, retained as an extension hook."),
    _entry("orbit.ric_relative_hold", "RIC Relative Hold", "orbit_controller", "sim.control.orbit.reference_rpo", "RICRelativeHoldController", "reference", "Reference rectangular-RIC offset and rate hold.", assumptions=("Uses OEL's relative-orbit belief with curvilinear RIC plus chief ECI state.",), tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.vbar_approach", "V-Bar Approach", "orbit_controller", "sim.control.orbit.reference_rpo", "VBarApproachController", "reference", "Rate-limited in-track approach with terminal slowdown.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.rbar_approach", "R-Bar Approach", "orbit_controller", "sim.control.orbit.reference_rpo", "RBarApproachController", "reference", "Rate-limited radial approach with terminal slowdown.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.cbar_approach", "C-Bar Approach", "orbit_controller", "sim.control.orbit.reference_rpo", "CBarApproachController", "reference", "Rate-limited cross-track approach with terminal slowdown.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.ric_waypoints", "RIC Waypoint Path", "orbit_controller", "sim.control.orbit.reference_rpo", "RICWaypointController", "reference", "Tolerance-gated rectangular-RIC waypoint sequencer.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.keep_out_standoff", "Keep-Out Standoff", "orbit_controller", "sim.control.orbit.reference_rpo", "KeepOutStandoffController", "reference", "Outward protective command inside a configured RIC keep-out sphere.", known_limits=("Spherical geometric barrier; not a probabilistic conjunction model.",), tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.passive_safe_retreat", "Passive-Safe Retreat", "orbit_controller", "sim.control.orbit.reference_rpo", "PassiveSafeRetreatController", "reference", "Acquire an outward drift rate, then coast.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.terminal_braking", "Terminal Braking", "orbit_controller", "sim.control.orbit.reference_rpo", "TerminalBrakingController", "reference", "Closing-rate-limited terminal relative-state controller.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.ric_flyaround", "RIC Flyaround", "orbit_controller", "sim.control.orbit.reference_rpo", "RICFlyaroundController", "reference", "Tolerance-gated polygonal inspection flyaround in an RIC plane.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.low_thrust_phasing", "Low-Thrust Phasing", "orbit_controller", "sim.control.orbit.reference_rpo", "LowThrustPhasingController", "workbench", "Low-authority along-track relative phasing baseline.", known_limits=("Relative V-bar phasing baseline, not a general long-arc optimizer.",), tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.plane_change_trim", "Plane-Change Trim", "orbit_controller", "sim.control.orbit.reference_rpo", "PlaneChangeTrimController", "workbench", "Cross-track relative position/rate trim baseline.", known_limits=("Local relative trim, not an impulsive global plane-change optimizer.",), tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.hcw_rendezvous_planner", "HCW Rendezvous Planner", "orbit_controller", "sim.control.orbit.reference_rpo", "HCWRendezvousPlannerController", "reference", "Closed-form HCW/SS-J2 velocity acquisition with finite-burn realization.", assumptions=("Near-circular chief and configured linear relative-dynamics envelope.",), tests=("sim/tests/test_ric_pd_transfer.py",)),
    _entry("orbit.proportional_navigation", "Proportional Navigation", "orbit_controller", "sim.control.orbit.reference_rpo", "ProportionalNavigationController", "workbench", "Simple target-directed or target-opposed RIC proportional navigation.", known_limits=("Educational local relative guidance, not operational intercept guidance.",), tests=("sim/tests/test_reference_gnc.py",)),
    _entry("orbit.rcs_allocation", "RCS Allocation Aware", "orbit_wrapper", "sim.control.orbit.rcs_allocator", "RCSAllocationAwareController", "workbench", "Wrap and cap desired acceleration against an RCS cluster.", known_limits=("Component-bench adapter; complete v2 stacks use the typed RCS allocator and hardware profile.",), tests=("sim/tests/test_actuator_aware_controllers.py",)),
    _entry("orbit.electric_propulsion", "Electric Propulsion Wrapper", "orbit_wrapper", "sim.control.orbit.electric_propulsion", "ElectricPropulsionController", "workbench", "Apply electric-propulsion thrust, duty-cycle, and power limits.", known_limits=("Component-bench adapter; complete v2 stacks use the typed continuous-engine allocator and hardware profile.",), tests=("sim/tests/test_actuator_aware_controllers.py",)),
    _entry("orbit.gimbaled_thruster", "Gimbaled Thruster Wrapper", "orbit_wrapper", "sim.control.orbit.gimbaled_thruster", "GimbaledThrusterController", "workbench", "Suppress commands outside configured gimbal authority.", known_limits=("Component-bench adapter; complete v2 stacks use the typed continuous-engine allocator and hardware profile.",), tests=("sim/tests/test_actuator_aware_controllers.py",)),
    _entry("orbit.robust_mpc_hook", "Robust MPC Hook", "orbit_controller", "sim.control.orbit.advanced", "RobustMPCController", "internal", "Fallback adapter reserved for a future robust MPC implementation."),
    _entry("orbit.stochastic_policy_hook", "Stochastic Policy Hook", "orbit_controller", "sim.control.orbit.advanced", "StochasticPolicyController", "internal", "Adapter for an externally supplied stochastic policy."),
    # Attitude controllers and helpers.
    _entry("attitude.zero_torque", "Zero Torque", "attitude_controller", "sim.control.attitude.zero_torque", "ZeroTorqueController", "reference", "No-torque baseline."),
    _entry("attitude.quaternion_pd", "Quaternion PD", "attitude_controller", "sim.control.attitude.baseline", "QuaternionPDController", "reference", "Quaternion attitude hold and recovery with torque saturation."),
    _entry("attitude.reaction_wheel_pd", "Reaction Wheel PD", "attitude_controller", "sim.control.attitude.baseline", "ReactionWheelPDController", "reference", "Quaternion PD with simplified reaction-wheel allocation."),
    _entry("attitude.reaction_wheel_pid", "Reaction Wheel PID", "attitude_controller", "sim.control.attitude.baseline", "ReactionWheelPIDController", "workbench", "Reaction-wheel PD with integral correction."),
    _entry("attitude.small_angle_lqr", "Small-Angle LQR", "attitude_controller", "sim.control.attitude.baseline", "SmallAngleLQRController", "reference", "Linearized attitude LQR baseline."),
    _entry("attitude.ric_pd", "RIC Frame PD", "attitude_controller", "sim.control.attitude.ric_pd", "RICFramePDController", "reference", "RIC-frame attitude PD wrapper."),
    _entry("attitude.ric_pid", "RIC Frame PID", "attitude_controller", "sim.control.attitude.ric_pid", "RICFramePIDController", "workbench", "RIC-frame attitude PID wrapper."),
    _entry("attitude.ric_lqr", "RIC Frame LQR", "attitude_controller", "sim.control.attitude.ric_lqr", "RICFrameLQRController", "reference", "RIC-frame small-angle LQR wrapper."),
    _entry("attitude.eci_detumble_pd", "ECI Detumble PD", "attitude_controller", "sim.control.attitude.detumble_pd", "ECIDetumblePDController", "workbench", "ECI rate damping followed by reference hold."),
    _entry("attitude.ric_detumble_pd", "RIC Detumble PD", "attitude_controller", "sim.control.attitude.detumble_pd", "RICDetumblePDController", "workbench", "RIC rate damping followed by reference hold."),
    _entry("attitude.atmospheric_lift_axis", "Atmospheric Lift Axis", "attitude_controller", "sim.control.attitude.aero_assist", "AtmosphericLiftAxisController", "workbench", "Point a body lift axis along a requested RIC lift direction."),
    _entry("attitude.surrogate_snap_eci", "Surrogate Snap ECI", "attitude_controller", "sim.control.attitude.surrogate_snap", "SurrogateSnapECIController", "workbench", "Rate-limited surrogate attitude response in ECI.", known_limits=("Surrogate response is not a physical actuator model.",)),
    _entry("attitude.surrogate_snap_ric", "Surrogate Snap RIC", "attitude_controller", "sim.control.attitude.surrogate_snap", "SurrogateSnapRICController", "workbench", "Rate-limited surrogate attitude response in RIC.", known_limits=("Surrogate response is not a physical actuator model.",)),
    _entry("attitude.snap", "Snap Attitude", "attitude_controller", "sim.control.attitude.snap", "SnapAttitudeController", "internal", "Direct state-override helper."),
    _entry("attitude.snap_hold_ric", "Snap And Hold RIC", "attitude_controller", "sim.control.attitude.snap_hold", "SnapAndHoldRICAttitudeController", "internal", "Direct snap-and-hold state-override helper."),
    _entry("attitude.detumble_then_slew", "Detumble Then Slew", "attitude_controller", "sim.control.attitude.switching", "DetumbleThenSlewController", "workbench", "Switch between detumble and nominal slew/hold controllers."),
    _entry("attitude.magnetorquer_bdot", "Magnetorquer B-dot", "attitude_controller", "sim.control.attitude.bdot_magnetorquer", "MagnetorquerBdotController", "workbench", "B-field-aware simplified magnetorquer detumble.", known_limits=("Component-bench only until a physical v2 magnetorquer profile is promoted.",), tests=("sim/tests/test_actuator_aware_controllers.py",)),
    _entry("attitude.wheel_desaturation", "Wheel Desaturation", "attitude_controller", "sim.control.attitude.wheel_desaturation", "WheelDesaturationController", "workbench", "Momentum-unload torque request from wheel momentum.", known_limits=("Component-bench only until v2 wheel-momentum telemetry and unloading hardware are promoted.",), tests=("sim/tests/test_actuator_aware_controllers.py",)),
    _entry("attitude.cmg_steering", "CMG Steering", "attitude_controller", "sim.control.attitude.cmg_steering", "CMGSteeringController", "workbench", "Cap a base controller by simplified CMG authority.", known_limits=("Component-bench only until a physical v2 CMG profile is promoted.",), tests=("sim/tests/test_actuator_aware_controllers.py",)),
    _entry("attitude.replay", "Attitude Replay", "attitude_controller", "sim.control.attitude.replay", "AttitudeReplayController", "internal", "Replay a validated attitude command history."),
    _entry("attitude.thrust_align", "Thrust Align", "attitude_controller", "sim.control.attitude.reference_pointing", "ThrustAlignController", "reference", "Point a configured body thrust axis at a supplied ECI burn direction.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("attitude.target_track", "Target Track", "attitude_controller", "sim.control.attitude.reference_pointing", "TargetTrackController", "reference", "Point a configured boresight at a supplied target position.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("attitude.nadir_pointing", "Nadir Pointing", "attitude_controller", "sim.control.attitude.reference_pointing", "NadirPointingController", "reference", "Point a configured body axis toward Earth center.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("attitude.velocity_pointing", "Velocity Pointing", "attitude_controller", "sim.control.attitude.reference_pointing", "VelocityPointingController", "reference", "Point a configured body axis prograde or retrograde.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("attitude.sun_pointing", "Sun Pointing", "attitude_controller", "sim.control.attitude.reference_pointing", "SunPointingController", "reference", "Point a configured body axis at a supplied ECI Sun direction.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("attitude.ric_axis_pointing", "RIC Axis Pointing", "attitude_controller", "sim.control.attitude.reference_pointing", "RICAxisPointingController", "reference", "Point a configured body axis along a requested RIC direction.", tests=("sim/tests/test_reference_gnc.py",)),
    # Mission strategies.
    _entry("mission.pursuit", "Pursuit", "mission_strategy", "sim.mission.modules", "PursuitMissionStrategy", "reference", "Generate target-directed pursuit intent."),
    _entry("mission.evade", "Evade", "mission_strategy", "sim.mission.modules", "EvadeMissionStrategy", "workbench", "Generate target-opposed evasive intent."),
    _entry("mission.hold", "Hold", "mission_strategy", "sim.mission.modules", "HoldMissionStrategy", "reference", "Hold ECI/RIC attitude or request sun, spotlight, sensing, or target pointing."),
    _entry("mission.desired_state", "Desired State", "mission_strategy", "sim.mission.modules", "DesiredStateMissionStrategy", "reference", "Request an explicit or target-derived ECI state."),
    _entry("mission.stationkeep", "Relative Station Keep", "mission_strategy", "sim.mission.modules", "StationKeepMissionStrategy", "reference", "Maintain a desired target-relative rectangular-RIC state."),
    _entry("mission.coe_stationkeep", "COE Station Keep", "mission_strategy", "sim.mission.modules", "OrbitalElementsStationKeepMissionStrategy", "reference", "Maintain a target COE set at current true anomaly."),
    _entry("mission.coe_tracking", "COE Tracking", "mission_strategy", "sim.mission.modules", "OrbitalElementsTrackingMissionStrategy", "reference", "Track selected orbital elements with acceleration intent."),
    _entry("mission.inspect", "Inspect", "mission_strategy", "sim.mission.modules", "InspectMissionStrategy", "workbench", "Maintain an inspection offset while pointing at a target."),
    _entry("mission.defensive", "Defensive", "mission_strategy", "sim.mission.modules", "DefensiveMissionStrategy", "workbench", "Request a fixed-axis or away-from-chaser defensive burn."),
    _entry("mission.safe_hold", "Safe Hold", "mission_strategy", "sim.mission.modules", "SafeHoldMissionStrategy", "reference", "Request zero thrust and a safe attitude target."),
    _entry("mission.executive", "Mission Executive", "mission_strategy", "sim.mission.modules", "MissionExecutiveStrategy", "workbench", "Switch between strategy/execution modes using deterministic triggers."),
    _entry("mission.rocket_pursuit", "Rocket Pursuit", "mission_strategy", "sim.mission.modules", "RocketPursuitMissionStrategy", "workbench", "Select a rocket pursuit orbital goal."),
    _entry("mission.rocket_predefined_orbit", "Rocket Predefined Orbit", "mission_strategy", "sim.mission.modules", "RocketPredefinedOrbitMissionStrategy", "workbench", "Select a predefined rocket target orbit."),
    _entry("mission.rocket_legacy", "Rocket Mission Strategy", "mission_strategy", "sim.mission.modules", "RocketMissionStrategy", "compatibility", "Legacy combined rocket goal and launch-timing strategy."),
    # Execution modules.
    _entry("execution.controller_pointing", "Controller Pointing", "mission_execution", "sim.mission.modules", "ControllerPointingExecution", "reference", "Run orbit and attitude controllers, align the thruster, and gate burns."),
    _entry("execution.predictive_burn", "Predictive Burn", "mission_execution", "sim.mission.modules", "PredictiveBurnExecution", "workbench", "Predict, slew, and gate a future controller burn."),
    _entry("execution.integrated_command", "Integrated Command", "mission_execution", "sim.mission.modules", "IntegratedCommandExecution", "reference", "Execute an orbit-controller burn when attitude alignment permits."),
    _entry("execution.direct_integrated", "Direct Integrated", "mission_execution", "sim.mission.modules", "DirectIntegratedExecution", "workbench", "Pass direct or fallback intent through the integrated command path."),
    _entry("execution.impulsive", "Impulsive", "mission_execution", "sim.mission.modules", "ImpulsiveExecution", "workbench", "Interval-aware pulse or impulse execution."),
    _entry("execution.budgeted_end_state", "Budgeted End State", "mission_execution", "sim.mission.modules", "BudgetedEndStateExecution", "workbench", "Convert desired end-state velocity error into a budgeted maneuver."),
    _entry("execution.safe_hold", "Safe Hold", "mission_execution", "sim.mission.modules", "SafeHoldExecution", "reference", "Force zero thrust while allowing attitude hold torque."),
    _entry("execution.timed_finite_burn", "Timed Finite Burn", "mission_execution", "sim.mission.modules", "TimedFiniteBurnExecution", "reference", "Apply a configured ECI or RIC acceleration over a bounded time interval.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.one_shot_impulse", "One-Shot Impulse", "mission_execution", "sim.mission.modules", "OneShotImpulseExecution", "reference", "Apply a delta-v as an explicit equivalent finite burn in the containing interval.", known_limits=("Does not bypass the deterministic dynamics engine with an instantaneous state edit.",), tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.pulse_train", "Pulse Train", "mission_execution", "sim.mission.modules", "PulseTrainExecution", "reference", "Deterministic periodic duty-cycle burn command.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.slew_then_burn", "Slew Then Burn", "mission_execution", "sim.mission.modules", "SlewThenBurnExecution", "reference", "Command burn attitude, gate on alignment, then apply thrust.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.burn_until", "Burn Until Condition", "mission_execution", "sim.mission.modules", "BurnUntilConditionExecution", "reference", "Apply acceleration until a deterministic time, duration, or speed condition.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.coast_until", "Coast Until Condition", "mission_execution", "sim.mission.modules", "CoastUntilConditionExecution", "reference", "Suppress thrust until a deterministic time or observed-range condition.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.waypoint_sequence", "Command Phase Sequencer", "mission_execution", "sim.mission.modules", "WaypointSequencerExecution", "reference", "Advance through named, duration-gated command phases.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.abort_safe_hold_retreat", "Abort Safe Hold / Retreat", "mission_execution", "sim.mission.modules", "AbortSafeHoldRetreatExecution", "reference", "Convert abort intent into safe hold or a configured RIC retreat command.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.fuel_budget_gate", "Fuel Budget Gate", "mission_execution", "sim.mission.modules", "FuelBudgetGateExecution", "reference", "Suppress direct thrust below a configured fuel margin.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.keep_out_gate", "Keep-Out Gate", "mission_execution", "sim.mission.modules", "KeepOutGateExecution", "reference", "Override direct intent with an outward command inside a geometric keep-out sphere.", known_limits=("Uses configured object knowledge and a spherical deterministic boundary.",), tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.command_replay", "Command Replay", "mission_execution", "sim.mission.modules", "CommandReplayExecution", "workbench", "Replay a checked command timeline for regression and training.", tests=("sim/tests/test_reference_gnc.py",)),
    _entry("execution.rocket_go_now", "Rocket Go Now", "mission_execution", "sim.mission.modules", "RocketGoNowExecution", "reference", "Authorize launch immediately."),
    _entry("execution.rocket_go_when_possible", "Rocket Go When Possible", "mission_execution", "sim.mission.modules", "RocketGoWhenPossibleExecution", "workbench", "Authorize launch when delta-v margin is sufficient."),
    _entry("execution.rocket_wait_optimal", "Rocket Wait Optimal", "mission_execution", "sim.mission.modules", "RocketWaitOptimalExecution", "workbench", "Authorize launch inside a periodic window."),
    # Compact and legacy modules.
    _entry("module.detumble_gate", "Attitude Detumble Gate", "mission_module", "sim.mission.modules", "AttitudeDetumbleGateMissionModule", "workbench", "Switch an attitude controller using rate thresholds."),
    _entry("module.satellite_legacy", "Satellite Mission Module", "mission_module", "sim.mission.modules", "SatelliteMissionModule", "compatibility", "Legacy compact satellite behavior wrapper."),
    _entry("module.defensive_ric_axis", "Defensive RIC Axis Burn", "mission_module", "sim.mission.modules", "DefensiveRICAxisBurnMissionModule", "workbench", "Legacy defensive fixed-axis burn with knowledge and attitude gates."),
    _entry("module.single_ric_axis_burn", "Single RIC Axis Burn", "mission_module", "sim.mission.modules", "SingleRICAxisBurnMissionModule", "workbench", "One-shot RIC-axis slew and burn."),
    _entry("module.multi_ric_axis_burn", "Multi RIC Axis Burn", "mission_module", "sim.mission.modules", "MultiRICAxisBurnMissionModule", "workbench", "Ordered multi-axis RIC burn sequence."),
    _entry("module.scheduled_vector_burn", "Scheduled Vector Burn", "mission_module", "sim.mission.modules", "ScheduledVectorBurnMissionModule", "workbench", "Duration-bound ECI or RIC vector burn."),
    _entry("module.rocket_legacy", "Rocket Mission Module", "mission_module", "sim.mission.modules", "RocketMissionModule", "compatibility", "Legacy compact rocket module."),
    _entry("module.end_state_legacy", "End State Maneuver", "mission_module", "sim.mission.modules", "EndStateManeuverMissionModule", "compatibility", "Legacy combined desired-state and burn execution module."),
    _entry("module.integrated_command_legacy", "Integrated Command Mission", "mission_module", "sim.mission.modules", "IntegratedCommandMissionModule", "compatibility", "Legacy compact integrated-command module."),
    _entry("module.predictive_command_legacy", "Predictive Integrated Command", "mission_module", "sim.mission.modules", "PredictiveIntegratedCommandMissionModule", "compatibility", "Legacy compact predictive-burn module."),
)

_BY_ID = {entry.builtin_id: entry for entry in _ENTRIES}
_ALIASES = {alias: entry.builtin_id for entry in _ENTRIES for alias in entry.aliases}


def catalog_entries(*, include_internal: bool = True) -> tuple[GNCDescriptor, ...]:
    if include_internal:
        return _ENTRIES
    return tuple(entry for entry in _ENTRIES if entry.maturity != "internal")


def catalog_entry(builtin_id: str) -> GNCDescriptor | None:
    key = str(builtin_id or "").strip().lower()
    key = _ALIASES.get(key, key)
    return _BY_ID.get(key)


def resolve_builtin_target(builtin_id: str) -> tuple[str, str]:
    entry = catalog_entry(builtin_id)
    if entry is None:
        valid = ", ".join(sorted(_BY_ID))
        raise ValueError(f"Unknown built-in GNC id {builtin_id!r}. Valid ids: {valid}")
    return entry.target()


def validate_catalog() -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    allowed_categories = {"orbit_controller", "orbit_wrapper", "attitude_controller", "mission_strategy", "mission_execution", "mission_module"}
    allowed_maturity = {"flagship", "reference", "workbench", "experimental", "compatibility", "internal"}
    for entry in _ENTRIES:
        if entry.builtin_id in seen:
            errors.append(f"duplicate built-in id: {entry.builtin_id}")
        seen.add(entry.builtin_id)
        if entry.category not in allowed_categories:
            errors.append(f"{entry.builtin_id}: invalid category {entry.category!r}")
        if entry.maturity not in allowed_maturity:
            errors.append(f"{entry.builtin_id}: invalid maturity {entry.maturity!r}")
        try:
            cls = getattr(importlib.import_module(entry.module), entry.class_name)
        except Exception as exc:
            errors.append(f"{entry.builtin_id}: cannot import {entry.module}.{entry.class_name}: {exc}")
            continue
        if not inspect.isclass(cls):
            errors.append(f"{entry.builtin_id}: target is not a class")
    for alias, target in _ALIASES.items():
        if alias in _BY_ID:
            errors.append(f"alias collides with built-in id: {alias}")
        if target not in _BY_ID:
            errors.append(f"alias {alias} has unknown target {target}")
    return errors
