# Payload And Artifact Contract

This document defines the 0.1 contract for simulation payloads, summaries, and
output artifacts. It focuses on single-run payloads and the public review
surface. Batch/campaign payloads are included only at a high level because their
full schemas remain Pro/private maturity work.


## Stability Level

Stable enough to rely on:

- single-run `summary` presence,
- core summary fields listed below,
- time/truth/belief/thrust/torque history keys,
- plot and animation artifact maps in summary,
- public API convenience properties that wrap these fields.

Still maturing:

- full-log debug structure,
- controller debug record contents,
- campaign and sensitivity report schemas,
- AI report metadata schemas,
- exact artifact filenames beyond published manifest mappings.


## Single-Run Payload Shape

A single-run payload should include these top-level keys when applicable:

- `summary`
- `time_s`
- `truth_by_object`
- `target_reference_orbit_truth`
- `belief_by_object`
- `applied_thrust_by_object`
- `applied_torque_by_object`
- `desired_attitude_by_object`
- `knowledge_by_observer`
- `knowledge_detection_by_observer`
- `knowledge_consistency_by_observer`
- `bridge_events_by_object`
- `controller_debug_by_object`
- `rocket_throttle_cmd`
- `rocket_metrics`

Consumers should treat `summary` as the most stable review surface. Detailed
histories and debug maps are available for analysis, but fields not documented
here should be considered lower stability until tests and docs promote them.


## Summary Contract

The single-run `summary` should include:

- `scenario_name`
- `scenario_description`
- `objects`
- `samples`
- `dt_s`
- `duration_s`
- `terminated_early`
- `termination_reason`
- `termination_time_s`
- `termination_object_id`
- `rocket_insertion_achieved`
- `rocket_insertion_time_s`
- `target_reference_orbit_enabled`
- `thrust_stats`
- `attitude_guardrail_stats`
- `knowledge_detection_by_observer`
- `knowledge_consistency_by_observer`
- `plot_outputs`
- `animation_outputs`

Compatibility expectations:

- `scenario_name` is a string.
- `objects` is a sorted list of participating object IDs.
- `samples` is the number of retained time samples.
- `duration_s` is the final retained simulation time, not necessarily the
  requested duration when a run terminates early.
- `terminated_early` is the authoritative early-termination flag.
- Termination detail fields may be `null` for nominal completion.
- `thrust_stats` is keyed by object ID.
- `plot_outputs` and `animation_outputs` are maps from artifact/figure IDs to
  paths or artifact descriptors.


## History Arrays

The following histories are keyed by object ID unless otherwise noted:

- `truth_by_object`
- `belief_by_object`
- `applied_thrust_by_object`
- `applied_torque_by_object`
- `desired_attitude_by_object`

Expected conventions:

- `time_s` is a one-dimensional list of sample times in seconds.
- Object histories should have the same first dimension as `time_s`.
- Truth state arrays use the project state convention documented by code and
  tests: position, velocity, attitude quaternion, angular rate, and mass where
  available.
- Applied thrust is in `km/s^2`.
- Applied torque is in `N*m`.
- Missing or inactive values may be represented by empty arrays, omitted maps,
  or `NaN` samples depending on the object lifecycle.

Consumers should not infer object participation solely from a history key; use
`summary.objects` for the stable list of participating objects.


## Knowledge And Estimation Artifacts

Knowledge fields:

- `knowledge_by_observer`
- `knowledge_detection_by_observer`
- `knowledge_consistency_by_observer`

Conventions:

- Knowledge is observer-owned and target-indexed.
- Detection summaries describe access/update behavior.
- Consistency summaries may include metrics such as update rate, NIS/NEES-style
  values, and position-error statistics when configured.

These fields are important validation surfaces, but exact nested metric sets may
grow. Validation harness checks should use explicit metric paths so schema
expectations are testable.


## Rocket Metrics

Rocket-specific fields may include:

- `rocket_throttle_cmd`
- `rocket_metrics.stage_index`
- `rocket_metrics.q_dyn_pa`
- `rocket_metrics.mach`
- `rocket_metrics.throttle_cmd`

Rocket metrics are present only when a rocket object participates. Consumers
should handle absent or empty rocket metrics for satellite-only scenarios.


## Artifacts On Disk

Stats artifacts:

- `index.md` is the human-readable start-here file for an output directory.
  It summarizes the workflow, scenario, key results, suggested review order,
  and artifact inventory.
- `master_run_summary.json` is written when `outputs.stats.save_json` is true.
- `master_run_log.json` is written when `outputs.stats.save_full_log` is true.

Plot and animation artifacts:

- Plot artifacts are represented in `summary.plot_outputs`.
- Animation artifacts are represented in `summary.animation_outputs`.
- Consumers should read artifact maps instead of constructing filenames from
  assumptions.

Output directories:

- `outputs.output_dir` controls where artifacts are written.
- Tests and validation workflows should prefer temporary directories unless
  intentionally checking repo-level example outputs.


## Public API Wrappers

`SimulationResult` provides convenience accessors over payload fields:

- `summary`
- `time_s`
- `truth`
- `belief`
- `applied_thrust`
- `applied_torque`
- `knowledge`
- `artifacts`
- `metrics`
- `snapshot(step_index)`

These wrappers are part of the intended public API. If payload keys change,
wrappers should either preserve compatibility or receive release-note coverage
and focused tests.


## Batch And Campaign Payloads

Batch payloads may include:

- `monte_carlo`
- `analysis`
- `runs`
- `aggregate_stats`
- `commander_brief`
- `artifacts`
- baseline comparisons,
- sensitivity rankings,
- AI report artifacts.

These are Pro/private workflow surfaces in 0.1. They are real and tested, but
their complete schema is not yet a public-core contract. Campaign and benchmark
schema stabilization should happen in a dedicated benchmark/campaign contract.


## Compatibility Rules

Changes should update this contract when they alter:

- stable summary field names or meanings,
- top-level single-run payload keys,
- history units,
- artifact-map semantics,
- public API wrapper behavior,
- termination metadata semantics.

For new analysis tools, prefer adding fields over changing existing meanings.
If an existing field must change, add migration notes and tests.


## Known Gaps

- No generated JSON schema exists for payloads.
- Campaign, sensitivity, benchmark, and AI report payloads need dedicated
  contracts.
- Debug records are intentionally lower stability than summaries.
- Some artifact maps still need stronger descriptor normalization.
