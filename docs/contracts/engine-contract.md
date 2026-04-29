# Engine Contract

This document defines the current execution contract for Orbital Engagement Lab
single-run simulation workflows. It is intentionally narrower than the full
implementation: it describes behavior users, tests, docs, and future
integrations may rely on, and it calls out areas that are still being
consolidated.

The contract applies to:

- public CLI single-run execution,
- `SimulationSession` and public API single-run execution,
- GUI-backed single-run execution,
- deterministic single-run scenarios used by validation and examples.

Batch workflows such as Monte Carlo, sensitivity, and controller benchmarking
reuse the same scenario model and single-run machinery where possible, but their
orchestration and artifact contracts are still Pro/private surfaces and are not
fully covered here.


## Stability Level

This is a 0.1 contract. It is meant to document intended behavior, not freeze
every implementation detail permanently. Changes that alter the observable
contract should update this document, tests, and release notes together.

Stable enough to rely on:

- deterministic step indexing and time grid,
- single-run object lifecycle,
- top-level step ordering,
- snapshot shape and timing,
- termination summary semantics,
- public API entrypoints for loading, running, and stepping scenarios.

Still maturing:

- exact payload field stability beyond documented summary fields,
- batch-analysis orchestration semantics,
- controller-benchmark and campaign artifact schemas,
- GUI coverage of every advanced YAML feature.


## Canonical Entrypoints

Preferred public entrypoints:

- CLI: `python run_simulation.py --config <path>`
- API: `SimulationConfig.from_yaml(...)` and `SimulationSession`
- Execution service: `sim.execution.run_simulation_config_file(...)`

For single-run scenarios, these routes should converge on the same conceptual
execution model. A user should not need to understand private orchestration
paths to reason about normal single-run behavior.

Private or transitional internals:

- `sim.master_simulator` remains important for legacy and batch workflows.
- Batch analysis may delegate to campaign or legacy master paths.
- Internal classes prefixed with `_`, including `_SingleRunEngine`, are not
  public extension APIs even when documented here for behavior.
- The legacy lower-level `SimulationKernel` loop has been removed; single-run
  behavior should be reasoned about through the canonical engine contract here.


## Time Grid

A single run uses a fixed outer time step:

- `dt_s` is the outer simulation step.
- `duration_s` defines the nominal run duration.
- The nominal time history includes the initial sample at `t = 0`.
- For standard single-run execution, the intended sample count is
  `floor(duration_s / dt_s) + 1`.

Substepping:

- Orbit and attitude propagation may use `orbit_substep_s` and
  `attitude_substep_s`.
- The effective internal satellite substep is the smaller of the active orbit
  and attitude substeps.
- Orbital command sampling may be slower than attitude/dynamics substeps through
  `orbit_command_period_s`.

Config validation should reject timing grids that cannot be represented cleanly.
Scenario authors should treat non-divisible timing combinations as invalid
unless an explicit migration or compatibility note says otherwise.


## Object Lifecycle

Supported object roles:

- `rocket`
- `chaser`
- `target`

Object creation:

- Enabled config sections create runtime objects.
- Disabled config sections do not participate in truth, belief, control,
  sensing, or output histories.

Initial sample:

- At `t = 0`, active objects write initial truth history.
- Initial belief history is written when a belief state exists.
- Knowledge bases, when configured, write an initial snapshot for known targets.

Activation:

- Most enabled objects are active at initialization.
- A chaser configured for rocket deployment may exist but remain inactive until
  deployment time.
- Inactive objects are skipped for propagation/control until activated.

Rocket-specific lifecycle:

- Rocket guidance and rocket simulation are handled by the rocket runtime path.
- Rocket waiting-for-launch state may hold the vehicle without thrusting.
- Rocket insertion can terminate `rocket_ascent` scenarios when configured
  insertion criteria are satisfied.


## Step Order

For each outer step from `t_k` to `t_{k+1}`, single-run execution follows this
conceptual order.

1. Resolve time and environment context.
2. Activate any time-gated objects, such as rocket-deployed chasers.
3. Build an internal current-time world-truth snapshot from active objects at
   `t_k`.
4. Propagate optional reference trajectories, such as target reference orbit.
5. For each active object, execute its runtime path:
   - rocket path: mission modules, mission strategy, mission execution,
     guidance, rocket propagation, belief update, thrust/mass/stage metrics;
   - satellite path: internal substep loop with current-time mission modules,
     mission strategy, external intent, mission execution, controller
     evaluation, actuator limiting, dynamics propagation, post-propagation
     sensing/estimation, and debug logging.
6. Update bridges for objects with enabled bridge integrations.
7. Update object knowledge bases from the post-step world truth.
8. Write truth, belief, knowledge, applied thrust, applied torque, desired
   attitude, and runtime/debug histories for `t_{k+1}`.
9. Emit the step callback, if one is registered.
10. Evaluate termination conditions.
11. Return a snapshot for the current index when stepping interactively.

The key invariant is that agents decide from current estimated information. At
an internal satellite substep beginning at `t_i`, mission logic and controllers
observe belief-derived own state and observer-owned knowledge at `t_i`, then
produce the command applied over `[t_i, t_{i+1}]`. Sensors and estimators then
update belief to `t_{i+1}` after dynamics propagation. Raw world truth is not
exposed to agent decision logic, either as a direct `world_truth` argument or
inside the decision-facing environment; perfect information should be modeled
with zero-error sensors/knowledge, not by reading simulator truth.


## Truth, Belief, And Knowledge Timing

Truth:

- Truth at index `0` is the initial condition.
- Truth at index `k + 1` represents propagated state at `t_{k+1}`.
- Single-run decision logic sees belief-derived own state and observer-owned
  knowledge. It should not see raw simulator truth for other objects.
- Mission target/reference resolution does not fall back to raw `world_truth`;
  if a target is not present in observer-owned knowledge, target-dependent
  decisions must hold, coast, or use an explicitly configured blind/explicit
  mode.
- Dynamics may receive an object-local world-truth context for perturbations,
  sensors, knowledge generation, and integration support, but controller and
  mission decisions are based on estimated current-time state.

Belief:

- Belief at index `0` is the initial belief.
- Estimator updates occur after propagation using measurements associated with
  the post-propagation evaluation time.
- Estimators propagate by elapsed time from `belief.last_update_t_s` to the
  requested update time; they must not advance by a fixed outer `dt_s` for each
  internal substep update.
- If no estimator/belief is configured for a satellite path, a truth-derived
  fallback belief may be created for control continuity.

Knowledge:

- Knowledge bases are observer-owned.
- Knowledge updates occur after active objects have reached the post-step world
  truth for `t_{k+1}`.
- If a target belief is unavailable at a sample, the prior knowledge sample may
  be carried forward when appropriate.

Controllers and mission modules should not assume that truth, belief, and
knowledge are the same thing. Truth is simulation state; belief is the
controller/estimator state; knowledge is the observer's tracked state about
other objects.


## Controller And Actuator Timing

Satellite control:

- Orbit and attitude controllers may be evaluated during internal substeps.
- Controllers act on belief and observer-owned knowledge corresponding to the
  start of the interval they command.
- Mission modules, mission strategy, external intent providers, and mission
  execution can modify or replace controller commands.
- Integrated mission commands may bypass separate orbit/attitude command
  combination when `mission_use_integrated_command` is set.
- Orbital thrust commands may be latched and reused until the orbital command
  period elapses.

Actuator limiting:

- Commands are constrained by available mass, dry mass, max thrust, Isp-derived
  propellant use, attitude-coupled thruster direction, and torque logic when
  configured.
- Applied thrust and torque histories represent the command actually applied to
  dynamics, not merely the raw controller request.

Runtime budgets:

- The lower-level kernel can evaluate controller runtime against a budget and,
  in realtime mode, replace overrun commands with zero command.
- The current high-level single-run path records controller runtime/debug data
  but should not be treated as a hard realtime scheduler.


## Snapshots

`SimulationSession.step()` and engine snapshots expose:

- `step_index`
- `time_s`
- `truth`
- `belief`
- `applied_thrust`
- `applied_torque`

Snapshot semantics:

- A reset single-run session returns a snapshot at index `0`.
- A step advances at most one outer time index unless the run is already done.
- Calling step after completion returns the final snapshot.
- Snapshots are only available for single-run scenarios, not batch analysis.


## Termination

Nominal termination:

- A run completes when the current index reaches the final time-grid index.

Early termination:

- Earth-impact termination can stop active object scenarios.
- Rocket insertion can stop `rocket_ascent` scenarios when insertion criteria
  are met.

Termination metadata:

- `terminated_early`
- `termination_reason`
- `termination_time_s`
- `termination_object_id`

Downstream tools should use these fields instead of inferring early termination
only from sample count.


## Payload And Artifact Expectations

Single-run payloads should include:

- time history,
- truth histories by object,
- belief histories by object,
- knowledge histories by observer/target when configured,
- applied thrust histories,
- applied torque histories,
- desired attitude histories when configured,
- controller debug history where available,
- rocket metrics when a rocket is present,
- summary metadata,
- plot and animation artifact paths when enabled.

The summary is the primary stable review surface for 0.1. It should include:

- scenario name and description,
- object IDs,
- sample count,
- `dt_s` and duration,
- termination status,
- thrust statistics,
- attitude guardrail statistics,
- plot and animation output manifests.

Detailed payload arrays and debug structures are useful but still maturing.
If a downstream tool depends on a detailed field, add or update tests that
codify that dependency.


## Public Extension Points

Supported extension surfaces:

- scenario YAML configuration,
- object presets,
- plugin pointers validated by `validate_scenario_plugins`,
- controller/mission classes that implement the expected methods,
- `SimulationSession` and public API wrappers,
- external intent providers registered on a `SimulationSession`.

Internal surfaces:

- private helper functions,
- `_SingleRunEngine` implementation details,
- exact controller debug record contents,
- legacy master-simulator orchestration,
- Pro/private batch-analysis internals.

When in doubt, prefer extending through config, presets, validated plugin
pointers, or the public API rather than importing private helpers.


## Compatibility Rules

Changes should update this document when they alter:

- step order,
- object activation semantics,
- truth/belief/knowledge timing,
- controller or actuator timing,
- termination semantics,
- stable summary fields,
- public API behavior,
- supported extension points.

For pre-1.0 releases, changes may still be made, but they should be explicit in
release notes and accompanied by focused tests.


## Known Gaps

- Batch workflow contracts are not yet fully documented.
- Scenario YAML contract and payload/artifact contract need dedicated documents.
- Public GUI workflows do not yet expose every advanced YAML capability.
- Release-grade validation packages are still private maturity work, not a
  completed public-core contract.
