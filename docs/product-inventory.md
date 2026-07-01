# Product Inventory

This inventory summarizes the built-in controller and mission surfaces that
ship with Orbital Engagement Lab. It is meant to answer "what comes with the
product?" before a user starts reading Python modules or reverse-engineering
scenario YAML.

Use this page with:

- [Scenario YAML](scenario-yaml.md) for pointer syntax.
- [Controller Bench](controller-bench.md) for comparative evaluation workflows.
- [Controller Naming Conventions](project/controller_naming_conventions.md) for
  naming rules when adding or renaming controllers.
- [Reference GNC Library Roadmap](project/reference_gnc_library_roadmap.md) for
  controller and command-module promotion planning.
- [Public Core And Pro Boundary](public-vs-pro.md) for public/pro packaging
  posture.

Inventory labels:

- **Flagship**: curated, product-facing path for demonstrations or evidence.
- **Reference**: supported baseline or reusable building block.
- **Workbench**: useful for engineering studies, tuning, wrapping, or custom
  scenarios.
- **Experimental**: included for exploration; validate before relying on it.
- **Compatibility**: retained for older configs or migration paths.

## Mission And Dynamics Capability Inventory

These surfaces are not all "controllers", but they are user-facing product
capabilities that ship with the simulator and appear in scenario YAML, plots,
payloads, or validation workflows.

| Product surface | Config surface | Status | Primary use |
| --- | --- | --- | --- |
| Shared Vehicle Aero Contract | `objects.<id>.specs.aero` | Reference | Canonical vehicle aero properties shared by satellites, re-entry diagnostics, aero-assist studies, and rocket ascent defaults. |
| Orbit Drag Force Model | `simulator.dynamics.orbit.drag` | Reference | Enables atmosphere-relative drag acceleration using configured vehicle area/coefficient and environment density. |
| Atmospheric Lift Force Model | `objects.<id>.specs.aero.cl`, `lift_area_m2`, `lift_axis_body` plus `orbit.drag` | Workbench | First-pass attitude-coupled lift for atmospheric steering and plane-change exploration. |
| Re-entry Diagnostics | `simulator.dynamics.reentry` | Reference | Tracks entry episodes, dynamic pressure, drag g-load, heat rate, heat load, and object-level termination limits. |
| Aero-Assisted Atmospheric Pass | `configs/aero_assisted_plane_change_demo.yaml` | Workbench | Demonstrates a satellite dipping into the atmosphere, using lift-axis steering, then burning to raise altitude. |
| Rocket Ascent Aero | `objects.rocket.specs.aero` plus `simulator.dynamics.rocket.aero` | Workbench | Shared object geometry/defaults with detailed rocket coefficient refinements, max-Q/TVC diagnostics, and insertion studies. |
| Object-Based Aero/Re-entry Termination | `simulator.dynamics.reentry.termination.by_object` and `simulator.termination.by_object` | Reference | Lets rockets, disposed stages, satellites, and atmospheric test articles coexist with different stop conditions. |

Canonical vehicle aero fields live under `objects.<id>.specs.aero`:
`reference_area_m2`, `drag_area_m2`, `lift_area_m2`, `cd`, `cl`,
`nose_radius_m`, `reference_length_m`, `lift_axis_body`/`lift_vector_body`, and
`cp_offset_body_m`. Use `simulator.dynamics.orbit.drag` to enable aerodynamic
forces, `simulator.dynamics.reentry` for diagnostics and limits, and
`simulator.dynamics.rocket.aero` only for detailed rocket coefficient
refinements. Flat aliases such as `specs.drag_area_m2`, `specs.cd`, and
`specs.nose_radius_m` remain compatibility inputs.

## Estimation And Knowledge

These surfaces feed closed-loop object knowledge during simulation or support
estimation-oriented validation fixtures. Batch OD workflows are covered in the
OD contract and may be Pro/private depending on packaging.

| Product surface | Config/API surface | Status | Primary use |
| --- | --- | --- | --- |
| ECI State EKF Knowledge | `knowledge.estimation.type: ekf` | Reference | Live target-state belief from noisy ECI state or relative measurements. |
| EKF Maneuver Detection | `knowledge.estimation.maneuver_detection` / `EKFManeuverDetector` | Reference | Innovation/NIS persistence gate for EKF knowledge tracks; reports suspect/confirmed maneuver evidence in knowledge consistency summaries. |
| Measured State Knowledge | `knowledge.estimation.type: measured_state` | Reference | Trust the latest full-state measurement while publishing a `StateBelief`. |
| HCW Relative EKF Knowledge | `knowledge.estimation.type: relative_hcw_ekf` / `HCWRelativeEKFEstimator` | Reference | Live rectangular-RIC relative-state estimation for circular-chief, small-separation RPO scenarios; public validation covers full relative-state and az/el/range/range-rate measurement cases. |
| TH Relative EKF Knowledge | `knowledge.estimation.type: relative_th_ekf` / `THRelativeEKFEstimator` | Reference | Live rectangular-RIC relative-state estimation for eccentric two-body-chief, small-separation RPO scenarios using numerically integrated TH linear dynamics; public validation covers full relative-state and az/el/range/range-rate measurement cases. |
| YA STM Relative EKF Knowledge | `knowledge.estimation.type: relative_ya_ekf` / `YARelativeEKFEstimator` | Reference | Live rectangular-RIC relative-state estimation for eccentric two-body-chief RPO using the closed-form Yamanaka-Ankersen anomaly-domain STM mapped into OEL's km/km/s RIC state; public validation compares it against HCW, TH-integrated, and ECI EKF rows. |

## Orbital Controllers

Orbital controllers are configured through `orbit_control` pointers in scenario
YAML. They emit translational commands and may be combined with mission
execution modules that handle pointing, burn gating, or integrated
orbit/attitude command execution.

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Zero Controller | `ZeroController` | `sim.control.orbit.zero_controller` | Reference | Coast/no-thrust baseline and config smoke tests. |
| RIC_PD Transfer | `RICPDTransferController` | `sim.control.orbit.ric_pd` | Flagship | 10 km-class RIC-frame RPO transfer with guided coast, correction burns, and terminal PD cleanup. |
| HCW LQR | `HCWLQRController` | `sim.control.orbit.lqr` | Reference | Linear HCW rendezvous/control baseline for closed-loop RPO examples and benchmarks. |
| HCW LQR, No Radial Burn | `HCWNoRadialLQRController` | `sim.control.orbit.lqr_no_radial` | Workbench | HCW LQR variant that suppresses radial-axis burns for constrained RPO studies. |
| HCW Manual Gain, No Radial Burn | `HCWNoRadialManualController` | `sim.control.orbit.lqr_no_radial` | Workbench | Manually supplied gain matrix for no-radial-burn tuning experiments. |
| HCW Curvilinear Input Variant | `HCWCurvInputRectOutputController` | `sim.control.orbit.lqr_curv_variant` | Workbench | HCW controller variant for curvilinear-RIC inputs with rectangular-RIC output behavior. |
| Curvilinear RIC PD | `CurvilinearRICPDController` | `sim.control.orbit.curv_pd` | Workbench | PD control in curvilinear RIC coordinates, with conversion to rectangular RIC/ECI commands. |
| Relative Orbit MPC | `RelativeOrbitMPCController` | `sim.control.orbit.relative_mpc` | Experimental | General relative-orbit model-predictive control experiments. |
| HCW Relative MPC | `HCWRelativeOrbitMPCController` | `sim.control.orbit.hcw_mpc` | Experimental | HCW-based MPC studies over a finite horizon. |
| HCW Relative MPC, In/Cross Track | `HCWInTrackCrossTrackMPCController` | `sim.control.orbit.hcw_mpc` | Experimental | MPC variant constrained to in-track and cross-track control axes. |
| RMOE If-Then | `RMOEIfThenController` | `sim.control.orbit.rmoe` | Workbench | Rule-based relative mean orbital-element/NMC targeting with priority logic and drift limiting. |
| Scheduled Impulse | `ScheduledImpulseController` | `sim.control.orbit.scheduled_impulse` | Reference | Deterministic delayed finite-duration impulse for validation and maneuver-detection proof scenarios. |
| Atmospheric Pass | `AtmosphericPassController` | `sim.control.orbit.aero_assist` | Workbench | Timed coast/raise-burn controller for aero-assisted atmospheric-pass demos. |
| Stationkeeping | `StationkeepingController` | `sim.control.orbit.baseline` | Reference | ECI state-hold or simple target-state feedback. |
| SMA/Ecc Feedback | `SemiMajorAxisEccentricityController` | `sim.control.orbit.baseline` | Reference | Low-thrust semi-major-axis and eccentricity regulation. |
| COE Feedback | `OrbitalElementsFeedbackController` | `sim.control.orbit.baseline` | Reference | Classical-orbital-element tracking across selected elements. |
| Safety Barrier | `SafetyBarrierController` | `sim.control.orbit.baseline` | Experimental | Repulsive keep-out style safety behavior inside a configured radius. |
| Risk Threshold | `RiskThresholdController` | `sim.control.orbit.baseline` | Experimental | Switches between nominal and evasive controllers using a user-supplied risk function. |

Actuator-aware orbital wrappers are covered in more detail in
[Actuators](actuators.md), but they are part of the controller inventory:

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| RCS Allocation Aware | `RCSAllocationAwareController` | `sim.control.orbit.rcs_allocator` | Workbench | Previews/caps desired commands against configured RCS cluster authority. |
| Electric Propulsion | `ElectricPropulsionController` | `sim.control.orbit.electric_propulsion` | Workbench | Caps a base controller by electric-propulsion thrust, duty-cycle, and power limits. |
| Gimbaled Thruster | `GimbaledThrusterController` | `sim.control.orbit.gimbaled_thruster` | Workbench | Suppresses thrust directions outside configured gimbal authority. |

Additional orbital-control helpers, such as `PredictiveBurnScheduler`,
`OrbitalAttitudeManeuverCoordinator`, and the HCW transfer solvers, are library
building blocks rather than top-level user-selectable controllers.

## Attitude Controllers

Attitude controllers are configured through `attitude_control` pointers. They
emit body torque commands or actuator-aware torque requests and can receive
targets from mission strategies and execution modules.

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Zero Torque | `ZeroTorqueController` | `sim.control.attitude.zero_torque` | Reference | No-torque baseline and attitude-disabled smoke paths. |
| Quaternion PD | `QuaternionPDController` | `sim.control.attitude.baseline` | Reference | General quaternion attitude hold/recovery with torque saturation. |
| Reaction Wheel PD | `ReactionWheelPDController` | `sim.control.attitude.baseline` | Reference | Reaction-wheel allocated quaternion PD control. |
| Reaction Wheel PID | `ReactionWheelPIDController` | `sim.control.attitude.baseline` | Workbench | Reaction-wheel PD with integral correction for persistent errors. |
| Small-Angle LQR | `SmallAngleLQRController` | `sim.control.attitude.baseline` | Reference | Linearized attitude-control baseline. |
| RIC Frame PD | `RICFramePDController` | `sim.control.attitude.ric_pd` | Reference | RIC-frame attitude hold using PD feedback. |
| RIC Frame PID | `RICFramePIDController` | `sim.control.attitude.ric_pid` | Workbench | RIC-frame attitude hold with integral action. |
| RIC Frame LQR | `RICFrameLQRController` | `sim.control.attitude.ric_lqr` | Reference | RIC-frame linear attitude-control baseline. |
| ECI Detumble PD | `ECIDetumblePDController` | `sim.control.attitude.detumble_pd` | Reference | Rate damping and detumble behavior in an ECI-oriented attitude path. |
| RIC Detumble PD | `RICDetumblePDController` | `sim.control.attitude.detumble_pd` | Reference | RIC-oriented detumble behavior for proximity operations. |
| Atmospheric Lift Axis | `AtmosphericLiftAxisController` | `sim.control.attitude.aero_assist` | Workbench | Points a configured body lift axis along a requested RIC lift direction for aero-assisted passes. |
| Snap Attitude | `SnapAttitudeController` | `sim.control.attitude.snap` | Workbench | Direct/snap-style attitude target behavior for fast scenario construction. |
| Snap And Hold RIC | `SnapAndHoldRICAttitudeController` | `sim.control.attitude.snap_hold` | Workbench | Snap to a RIC attitude and then hold it. |
| Surrogate Snap ECI | `SurrogateSnapECIController` | `sim.control.attitude.surrogate_snap` | Workbench | Fast surrogate attitude slew/recovery model in ECI. |
| Surrogate Snap RIC | `SurrogateSnapRICController` | `sim.control.attitude.surrogate_snap` | Workbench | Fast surrogate attitude slew/recovery model in RIC. |
| Detumble Then Slew | `DetumbleThenSlewController` | `sim.control.attitude.switching` | Workbench | Switches between detumble and nominal slew/hold behavior. |
| Magnetorquer B-dot | `MagnetorquerBdotController` | `sim.control.attitude.bdot_magnetorquer` | Workbench | B-field-aware detumble controller for magnetorquer studies. |
| Wheel Desaturation | `WheelDesaturationController` | `sim.control.attitude.wheel_desaturation` | Workbench | Momentum unload torque request from wheel momentum state. |
| CMG Steering | `CMGSteeringController` | `sim.control.attitude.cmg_steering` | Workbench | Caps/wraps a base attitude controller by simplified CMG authority. |

`PoseCommandGenerator` is an attitude command helper used by mission logic for
sun-track, spotlight, RIC-pointing, and target-facing commands. It is not a
standalone closed-loop controller.

## Mission Modules

Mission logic is split into three related surfaces:

- **Mission strategies** decide intent: pursue, evade, hold, inspect,
  stationkeep, defend, or select a rocket orbital goal.
- **Mission execution modules** convert intent into integrated commands:
  pointing, predictive burns, gated burns, impulses, direct commands, or safe
  hold.
- **Mission modules** are compact or legacy-style modules that combine a
  behavior into one plugin. New scenarios should usually prefer explicit
  `mission_strategy` plus `mission_execution` blocks when that makes intent
  easier to audit.

### Mission Strategies

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Pursuit | `PursuitMissionStrategy` | `sim.mission.modules` | Reference | Generate target-directed pursuit intent from object knowledge or blind fallback direction. |
| Evade | `EvadeMissionStrategy` | `sim.mission.modules` | Reference | Generate target-opposed evasive intent from object knowledge or blind fallback direction. |
| Hold | `HoldMissionStrategy` | `sim.mission.modules` | Reference | Hold ECI/RIC attitude, sun-track, spotlight, or sensing-oriented pointing intent. |
| Desired State | `DesiredStateMissionStrategy` | `sim.mission.modules` | Reference | Request an explicit or target-derived ECI desired state. |
| Station Keep | `StationKeepMissionStrategy` | `sim.mission.modules` | Reference | Maintain a desired relative RIC state around a target. |
| COE Station Keep | `OrbitalElementsStationKeepMissionStrategy` | `sim.mission.modules` | Reference | Maintain a target set of classical orbital elements at current true anomaly. |
| COE Tracking | `OrbitalElementsTrackingMissionStrategy` | `sim.mission.modules` | Reference | Track selected orbital elements with feedback acceleration intent. |
| Inspect | `InspectMissionStrategy` | `sim.mission.modules` | Workbench | Hold a relative inspection geometry while pointing at the target. |
| Defensive | `DefensiveMissionStrategy` | `sim.mission.modules` | Workbench | Burn along a fixed RIC axis or away from a known chaser. |
| Safe Hold | `SafeHoldMissionStrategy` | `sim.mission.modules` | Reference | Zero-thrust safe attitude hold or sun-track fallback. |
| Mission Executive | `MissionExecutiveStrategy` | `sim.mission.modules` | Workbench | Mode machine that switches between strategy/execution pairs using range or fuel triggers. |
| Rocket Pursuit | `RocketPursuitMissionStrategy` | `sim.mission.modules` | Workbench | Select a rocket pursuit orbital goal. |
| Rocket Predefined Orbit | `RocketPredefinedOrbitMissionStrategy` | `sim.mission.modules` | Workbench | Select a predefined rocket target altitude/eccentricity goal. |
| Rocket Mission Strategy | `RocketMissionStrategy` | `sim.mission.modules` | Compatibility | Older combined rocket goal and launch-timing wrapper. |

### Mission Execution Modules

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Controller Pointing | `ControllerPointingExecution` | `sim.mission.modules` | Reference | Run orbit/attitude controllers, set attitude target for thrust alignment, and gate burns by pointing error. |
| Predictive Burn | `PredictiveBurnExecution` | `sim.mission.modules` | Workbench | Plan burns ahead of time, slew toward required attitude, then fire when alignment and timing allow. |
| Integrated Command | `IntegratedCommandExecution` | `sim.mission.modules` | Reference | Execute orbit-controller burns only when attitude alignment is satisfied. |
| Direct Integrated | `DirectIntegratedExecution` | `sim.mission.modules` | Reference | Pass direct thrust/torque intent through the integrated command path. |
| Impulsive | `ImpulsiveExecution` | `sim.mission.modules` | Workbench | Pulse burn commands on a configured cadence with optional attitude gating. |
| Budgeted End State | `BudgetedEndStateExecution` | `sim.mission.modules` | Workbench | Convert desired end-state velocity error into budgeted thrust-limited or impulsive maneuvers. |
| Safe Hold | `SafeHoldExecution` | `sim.mission.modules` | Reference | Force zero thrust while allowing attitude hold control. |
| Rocket Go Now | `RocketGoNowExecution` | `sim.mission.modules` | Reference | Immediately authorize launch. |
| Rocket Go When Possible | `RocketGoWhenPossibleExecution` | `sim.mission.modules` | Workbench | Authorize launch only when estimated delta-v margin is sufficient. |
| Rocket Wait Optimal | `RocketWaitOptimalExecution` | `sim.mission.modules` | Workbench | Authorize launch inside a periodic launch window. |

### Compact Mission Modules

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Attitude Detumble Gate | `AttitudeDetumbleGateMissionModule` | `sim.mission.modules` | Workbench | Switch an attitude controller between detumble and nominal modes using rate thresholds. |
| Satellite Mission Module | `SatelliteMissionModule` | `sim.mission.modules` | Compatibility | Compact satellite behavior wrapper for coast, pursuit/evade, and attitude pointing modes. |
| Defensive RIC Axis Burn | `DefensiveRICAxisBurnMissionModule` | `sim.mission.modules` | Workbench | Defensive fixed-axis RIC burn with attitude alignment and knowledge gating. |
| Single RIC Axis Burn | `SingleRICAxisBurnMissionModule` | `sim.mission.modules` | Workbench | One-shot RIC-frame burn/slew behavior for scripted scenarios. |
| Rocket Mission Module | `RocketMissionModule` | `sim.mission.modules` | Compatibility | Compact rocket launch authorization and goal-selection module. |
| End State Maneuver | `EndStateManeuverMissionModule` | `sim.mission.modules` | Workbench | Combined desired-state maneuver planner with integrated attitude/burn gating. |
| Integrated Command Mission | `IntegratedCommandMissionModule` | `sim.mission.modules` | Workbench | Compact module for integrated orbit/attitude command execution. |
| Predictive Integrated Command | `PredictiveIntegratedCommandMissionModule` | `sim.mission.modules` | Workbench | Compact predictive-burn variant of the integrated command path. |

## Maintainer Notes

Runtime exports in `sim/control/orbit/__init__.py`,
`sim/control/attitude/__init__.py`, and `sim/mission/__init__.py` may include
additional workbench or compatibility items beyond the product-facing inventory.

When adding a new controller or mission behavior:

1. Follow [Controller Naming Conventions](project/controller_naming_conventions.md).
2. Check the [Reference GNC Library Roadmap](project/reference_gnc_library_roadmap.md)
   to decide whether the behavior is Reference, Workbench, Experimental,
   Compatibility, or Internal/Hook.
3. Add or update at least one runnable YAML example when the behavior is
   product-facing.
4. Update this inventory if users should know the feature ships with the
   product.
5. Add focused tests for any promoted user-facing behavior.
