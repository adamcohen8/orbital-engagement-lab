# Reference GNC Library Roadmap

## Purpose

Orbital Engagement Lab has enough controller, attitude, actuator, and mission
execution pieces to support a deliberate built-in GNC library. This roadmap
turns the current organic inventory into a product plan: which surfaces are
ready to present as dependable off-the-shelf choices, which need polish, and
which new primitives should be added next.

The goal is not to prevent custom controllers. Customization is part of the
product. The goal is to give users a trusted shelf of reference building
blocks before they write their own mission-specific logic.

Use this roadmap with:

- [Product Inventory](../product-inventory.md) for what currently ships.
- [Controller Naming Conventions](controller_naming_conventions.md) for
  naming and migration rules.
- the private Controller Bench guide for comparative evaluation;
- [Scenario YAML](../scenario-yaml.md) for plugin pointer syntax.

## Library Model

Keep the mental model simple:

- **Controllers** decide what acceleration or torque should be requested.
- **Mission strategies** decide intent: pursue, hold, inspect, evade, defend,
  stationkeep, or select a rocket orbital goal.
- **Command modules** decide when and how intent is allowed to become a burn,
  impulse, torque, mode transition, safe hold, or abort.
- **The deterministic engine** decides what actually happens.

This split should stay visible in docs, YAML examples, local UI labels, benchmark
suites, and reports. It lets users replace one layer without rewriting the
whole stack.

## Readiness Labels

Use these labels when promoting, documenting, or hiding GNC surfaces.

| Label | Meaning | Minimum evidence |
| --- | --- | --- |
| Flagship | Curated demonstration or evidence path. | Runnable scenario, docs, focused tests, review artifacts, and validation story. |
| Reference | Stable reusable baseline or primitive. | Focused tests, at least one runnable config/example, documented parameters, and clear output flags. |
| Workbench | Useful engineering study component. | Importable, validated by at least one smoke/config path or focused unit test, with known limits called out. |
| Experimental | Real implementation, but not yet a product promise. | Tests or examples exist, but coverage, tuning guidance, or convergence evidence is incomplete. |
| Compatibility | Retained for old configs or migration. | Should have migration guidance and should not be the preferred path for new scenarios. |
| Internal/Hook | Helper, adapter, replay, or custom-extension hook. | May be documented for developers, but should not be marketed as an off-the-shelf controller. |

## Promotion Gates

A controller or command module should not become Reference unless it has:

1. A product-facing name that describes domain, frame, method, and role.
2. A stable Python module/class and scenario YAML pointer.
3. Focused tests for nominal behavior, bad inputs, saturation/authority, and
   at least one edge case.
4. At least one runnable config or example that validates with
   `run_simulation.py --validate-only`.
5. Meaningful `mode_flags` or `mission_mode` fields for review-store and plot
   inspection.
6. Parameter guidance: expected units, rough tuning range, and model limits.
7. A controller-bench case when performance comparison or tuning is part of
   the user story.
8. local UI catalog exposure when non-developer users should select it.

Flagship status additionally requires a maintained scenario, review artifacts,
and a validation/evidence note.

## Current Assessment

This assessment is based on source inspection, docs/config usage, local UI exposure,
and focused controller/mission regression tests. It is a product-readiness
assessment, not a full mission qualification claim.

Focused checks run during this audit:

```bash
.venv/bin/python -m pytest sim/tests/test_controller_lab_bench.py sim/tests/test_ric_pd_transfer.py sim/tests/test_orbit_hcw_lqr.py sim/tests/test_orbit_hcw_lqr_no_radial.py sim/tests/test_orbit_hcw_lqr_curv_variant.py sim/tests/test_orbit_hcw_lqr_convergence.py sim/tests/test_actuator_aware_controllers.py sim/tests/test_attitude_rw_pd.py sim/tests/test_attitude_lqr.py sim/tests/test_attitude_ric_pd.py sim/tests/test_attitude_ric_pid.py sim/tests/test_attitude_ric_lqr.py sim/tests/test_surrogate_snap_controller.py sim/tests/test_orbital_attitude_coordinator.py sim/tests/test_impulsive_attitude_target.py sim/tests/test_mission_executive.py sim/tests/test_mission_new_architecture_migrations.py sim/tests/test_rocket_mission_split.py sim/tests/test_mission_defensive_ric_axis.py sim/tests/test_orbital_elements_stationkeep.py
```

Result: 80 passed.

Representative configs validated:

```bash
.venv/bin/python run_simulation.py --config configs/quickstart_5min.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/aero_assisted_plane_change_demo.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/actuator_lab_presets_smoke.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/orbital_elements_tracking_general.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/rmoe_if_then_nmc_demo.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/controller_bench_hcw_pd_10km_verify.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/mission_reconstitution_planner_demo.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/controller_rcs_allocation_smoke.yaml --validate-only
```

Result: all parsed and reported `Plugins: OK` and `Result: OK`.

### Works Well Today

These are good candidates for the curated library with little or moderate
documentation polish.

| Surface | Proposed label | Rationale |
| --- | --- | --- |
| `ZeroController` | Reference | Reliable no-thrust baseline, heavily used in configs and tests. |
| `RICPDTransferController` | Flagship | Best curated RPO transfer surface; tested, documented, and scenario-backed. |
| `HCWLQRController` | Reference | Strong baseline for linear HCW rendezvous and benchmarks. |
| `HCWNoRadialLQRController` | Workbench/Reference | Useful constrained-control variant with tests and configs. |
| `HCWNoRadialManualController` | Workbench | Good gain-tuning surface for no-radial studies. |
| `StationkeepingController` | Reference | Simple ECI state feedback primitive. |
| `SemiMajorAxisEccentricityController` | Reference | Useful low-thrust semi-major-axis/eccentricity feedback. |
| `OrbitalElementsFeedbackController` | Reference | General COE feedback primitive, especially with mission strategies. |
| `RMOEIfThenController` | Workbench | Real rule-based RMOE controller with tests and scenario coverage. |
| Actuator-aware wrappers | Workbench | RCS, electric propulsion, and gimbaled-thruster wrappers validate and compose well. |
| `ReactionWheelPDController` | Reference | Strong attitude-control baseline with broad config/test use. |
| `ReactionWheelPIDController` | Workbench/Reference | Adds integral action with focused tests. |
| `SmallAngleLQRController` | Reference | Good LQR attitude baseline with capture mode. |
| RIC attitude PD/PID/LQR | Reference/Workbench | Useful RIC-frame attitude primitives with direct tests. |
| `SurrogateSnapECIController` | Workbench | Very useful training/scenario surrogate with tests and configs. |
| `ControllerPointingExecution` | Reference | Central command module for controller plus attitude pointing/gating. |
| `PredictiveBurnExecution` | Workbench | Important integrated burn planner; validated by quickstart and tests. |
| `IntegratedCommandExecution` | Reference | Clean burn-when-aligned command path. |
| `BudgetedEndStateExecution` | Workbench | Useful delta-v-budgeted end-state execution primitive. |
| `SafeHoldExecution` | Reference | Clear zero-thrust safety execution surface. |
| `MissionExecutiveStrategy` | Workbench | Useful mode machine with range/fuel transitions and tests. |
| Rocket strategy/execution split | Workbench | Better than older combined rocket wrappers; tested. |
| `SingleRICAxisBurnMissionModule` | Workbench | Useful scripted burn primitive with direct tests and configs. |

### Needs Improvement Before Promotion

These can remain available, but should not be presented as polished Reference
surfaces until the listed gaps close.

| Surface | Current issue | Recommended action |
| --- | --- | --- |
| `HCWPDController` | Works, but naming conflicts with product-facing RIC_PD guidance. | Reframe as `RICPDHoldController` or keep as compatibility/workbench. |
| `CurvilinearRICPDController` | Useful, but not exposed in local UI and needs stronger examples. | Add scenario, docs, and bench coverage if promoted. |
| `RelativeOrbitMPCController` | Real nonlinear MPC, but convergence/tuning evidence is thin. | Keep Experimental; add benchmark envelopes and budget behavior tests. |
| `HCWRelativeOrbitMPCController` | Real implementation, but still specialist-tuned. | Keep Experimental; add maintained benchmark suite and tuning notes. |
| `HCWInTrackCrossTrackMPCController` | Good constrained variant, but lacks config/example footprint. | Add example and compare against no-radial LQR. |
| `SafetyBarrierController` | Valuable concept, no direct tests/config use. | Productize as a keep-out/standoff controller or hide as internal. |
| `RiskThresholdController` | Powerful wrapper, but depends on user-supplied callable and has no evidence. | Recast as a typed risk-gate command module with documented risk inputs. |
| `QuaternionPDController` | Useful simple controller, but lacks direct tests. | Add direct unit tests and a small config. |
| `ECIDetumblePDController` / `RICDetumblePDController` | Important but under-tested. | Add detumble configs and tests for rate-only and reference-lock modes. |
| `SurrogateSnapRICController` | Useful, but no direct tests/config use. | Add RIC snap smoke config and test. |
| `MagnetorquerBdotController` | Workbench surface with simplified field assumptions. | Keep Workbench; document B-field expectations and add scenario evidence. |
| `AtmosphericPassController` / `AtmosphericLiftAxisController` | First-pass aero-assist, not high-fidelity guidance. | Keep Workbench; document model limits and compare against no-aero baseline. |
| `DirectIntegratedExecution` | Useful primitive, but low test/config footprint. | Add direct-command tests and a small scenario. |
| `ImpulsiveExecution` | Conceptually important, but minimal scenario evidence. | Promote after timed impulse and pulse-train examples. |
| `EvadeMissionStrategy` | Simple but almost no direct evidence. | Add direct tests and an evasive-target example. |
| `InspectMissionStrategy` | Product-relevant, but under-tested. | Add inspection geometry scenario and review-store metrics. |
| `DefensiveRICAxisBurnMissionModule` | Overlaps with split strategy/execution approach. | Keep Workbench or migrate to `DefensiveMissionStrategy` plus command module. |

### Keep Hidden Or Compatibility-Only

These are useful hooks or legacy forms, but should not be marketed as
off-the-shelf GNC choices.

| Surface | Recommended label | Reason |
| --- | --- | --- |
| `RobustMPCController` | Internal/Hook | Currently delegates to fallback; not a real robust MPC implementation yet. |
| `StochasticPolicyController` | Internal/Hook | Useful policy adapter, not a built-in controller. |
| `SnapAttitudeController` | Internal/Hook | State-override tool, not physical control. |
| `SnapAndHoldRICAttitudeController` | Internal/Hook | State-override tool, not physical control. |
| `AttitudeReplayController` | Internal/Hook | Useful replay mechanism, but not productized as a controller. |
| `SatelliteMissionModule` | Compatibility | Older compact wrapper; prefer split strategy/execution. |
| `RocketMissionModule` | Compatibility | Older compact wrapper; prefer split rocket strategy/execution. |
| `RocketMissionStrategy` | Compatibility | Older combined rocket goal and launch-timing wrapper. |
| `EndStateManeuverMissionModule` | Compatibility/Workbench | Prefer `DesiredStateMissionStrategy` plus `BudgetedEndStateExecution`. |
| `IntegratedCommandMissionModule` | Compatibility/Workbench | Prefer split mission strategy plus `IntegratedCommandExecution`. |
| `PredictiveIntegratedCommandMissionModule` | Compatibility/Workbench | Prefer split mission strategy plus `PredictiveBurnExecution`. |
| `AttitudeDetumbleGateMissionModule` | Workbench/Internal | Good idea, but no config/test footprint yet. |

## Target Reference Library

The next product milestone should be a compact, reliable GNC shelf rather than
a long list of loosely related classes.

### Orbit Controllers

| Candidate | Role | Priority |
| --- | --- | --- |
| RIC relative hold | Hold a target RIC offset and rate. | P0 |
| V-bar approach | Approach along in-track line with rate limits. | P0 |
| R-bar approach | Approach along radial line with rate limits. | P0 |
| Cross-track approach | Approach or depart along C-bar/cross-track geometry. | P1 |
| Waypoint RIC path | Sequence relative waypoints with tolerance gates. | P0 |
| Flyaround/inspection geometry | Maintain or cycle around inspection viewpoints. | P1 |
| Keep-out/standoff | Repel or tangent-burn near protected zones. | P0 |
| Passive-safe retreat | Drift-away or abort to safer relative motion. | P0 |
| Terminal braking | Enforce closing-rate and terminal box limits. | P0 |
| Low-thrust phasing | Semi-major-axis/phase targeting for slow rendezvous. | P1 |
| Plane-change/inclination trim | Low-thrust or impulsive cross-track correction. | P1 |
| Impulsive HCW rendezvous planner | Two-impulse or N-impulse closed-form planner. | P0 |
| Pursuit/evasion proportional guidance | Simple target-directed or target-opposed guidance. | P1 |

### Attitude Controllers

| Candidate | Role | Priority |
| --- | --- | --- |
| Thrust-align controller | Point a configured thruster axis at a requested burn vector. | P0 |
| Target-track controller | Point body boresight at another object. | P0 |
| Nadir pointing | Point a body axis toward Earth center/local nadir. | P0 |
| Velocity pointing | Point along prograde/retrograde velocity. | P0 |
| Sun pointing | Point solar panel or body axis at sun vector. | P0 |
| RIC-axis pointing | Point body axis along R/I/C directions. | P0 |
| Slew-rate-limited tracker | Physical-ish attitude target tracking with rate limits. | P1 |
| Detumble profile set | ECI/RIC detumble presets with clear thresholds. | P1 |

### Command Modules

| Candidate | Role | Priority |
| --- | --- | --- |
| Timed finite burn | Burn from start time for duration. | P0 |
| One-shot impulse | Apply one impulse or equivalent finite burn. | P0 |
| Pulse train | Duty-cycle burn with period/width/phase. | P0 |
| Slew-then-burn | Set attitude target, wait for alignment, then fire. | P0 |
| Burn-until-condition | Fire until range, speed, state, or time condition is met. | P0 |
| Coast-until-condition | Hold/coast until a transition condition is met. | P0 |
| Waypoint sequencer | Advance through mission phases by tolerance gates. | P0 |
| Abort/safe-hold/retreat | Switch to zero thrust, safe attitude, or passive retreat. | P0 |
| Fuel-budget gate | Permit command only when budget/margin is sufficient. | P1 |
| Keep-out gate | Override nominal command when keep-out rules trigger. | P0 |
| Event/range/fuel/time executive templates | Prebuilt `MissionExecutiveStrategy` patterns. | P1 |
| Command replay | Replay validated command history for regression/training. | P1 |

## Recommended Implementation Sequence

### Phase 1: Curate And Clean Up

1. Mark compatibility/internal surfaces in docs and local UI catalogs.
2. Add direct tests for simple but under-tested controllers:
   `QuaternionPDController`, detumble controllers, `SurrogateSnapRICController`,
   `EvadeMissionStrategy`, and `InspectMissionStrategy`.
3. Add one small validation config for each promoted Reference primitive.
4. Align naming for `HCWPDController`; either migrate to a RIC-facing name or
   explicitly mark it compatibility/workbench.
5. Add a table of promoted Reference GNC surfaces to the local UI capability catalog.

### Phase 2: Build Missing Reference Primitives

1. Implement RIC hold, V-bar approach, R-bar approach, waypoint RIC path,
   keep-out/standoff, passive-safe retreat, and terminal braking.
2. Implement thrust-align, target-track, nadir, velocity, sun, and RIC-axis
   pointing attitude controllers.
3. Implement timed finite burn, one-shot impulse, slew-then-burn,
   burn-until-condition, coast-until-condition, and waypoint sequencer command
   modules.
4. Add a small `configs/reference_gnc_*` example family with review enabled.

### Phase 3: Benchmark And Promote

1. Add controller-bench suites for RIC approach, terminal box, keep-out,
   passive-safe abort, attitude pointing, and command-module sequencing.
2. Promote stable items from Workbench to Reference only after benchmark
   evidence is available.
3. Keep MPC, aero-assist, and advanced defensive logic in Workbench or
   Experimental until they have documented envelopes and benchmark results.

## Public/Pro Boundary Guidance

Public core should include enough Reference GNC primitives to make OEL feel
complete and educational:

- RIC hold, V-bar/R-bar approach, waypoint path, terminal braking,
  basic keep-out, safe hold, simple attitude pointing, and simple timed burns.

Pro/private should emphasize high-leverage workflow acceleration:

- controller-bench envelopes,
- tuned variants,
- advanced defensive mode libraries,
- robust/stochastic MPC once real,
- optimized command sequencing,
- mission-specific curated packs,
- evidence/report automation,
- customer/integration-specific adapters.

## Maintenance Rules

When adding a controller or command module:

1. Choose the product name first.
2. Add the class and config pointer.
3. Add a focused test.
4. Add one runnable config if users should select it.
5. Add docs and expected mode flags.
6. Add local UI exposure only after the docs and config exist.
7. Add controller-bench coverage before using performance claims.

Avoid expanding the inventory with clever one-offs. New built-ins should either
be broadly reusable primitives or explicitly marked Workbench/Experimental.
