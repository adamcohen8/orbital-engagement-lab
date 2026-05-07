# Changelog

All notable changes to Orbital Engagement Lab will be tracked in this file.

This project uses semantic versioning while it is pre-1.0: minor versions may
still introduce API or workflow changes, and release notes should call out
migration-sensitive behavior explicitly.

## 0.5.0 - 2026-05-07

Flagship public workflow and Python API release.

### Added

- Added the public flagship `configs/hcw_pd_10km_experiment.yaml` review
  scenario with saved report artifacts, refreshed plot gallery images, and a
  companion custom-analysis script.
- Added public Python API helpers for single-run workspace validation, relative
  state/range analysis, event checks, records/dataframes, and callback metrics.
- Added private Pro Python API documentation for controller-bench, campaign,
  sweep, and AI workflow helpers while keeping those docs out of the public
  export.

### Fixed

- Fixed thrust-alignment error plots to use the object-resolved thruster axis
  instead of a plot-level default, eliminating false 90-degree alignment errors.
- Fixed `SimulationResult.first_crossing()` so time-window filtering is applied
  consistently to both timestamps and values.
- Fixed workspace validation so generated Monte Carlo/sensitivity configs are
  still checked when plugin validation is non-strict and plugin issues are
  warnings.
- Tightened the public export boundary so public users cannot access Pro API
  workflow helpers through the exported package.

## 0.4.2 - 2026-05-07

HCW PD controller and validation workflow release.

### Added

- Added an HCW PD orbital controller with 10 km rendezvous experiment,
  Monte Carlo, and controller-bench tuning configs.
- Added campaign-level Monte Carlo range-timeseries plotting and Gemini-backed
  post-run AI report settings for the HCW PD experiment.
- Added validation evidence packaging, release workflow tests, pytest marker
  tiers, and split validation harness suites for orbit, attitude, actuator,
  sensor, integrated RPO, and HPOP reference checks.

### Fixed

- Kept relative-initialization belief state synchronized with adjusted truth
  state so controllers and logs start from the same state.
- Fixed Monte Carlo sampled-parameter reporting for canonical
  `objects.<object_id>` paths in initial relative-state plots.
- Limited attitude-coupled thrust application to attitude-enabled runs.

## 0.4.1 - 2026-05-05

Documentation, public-export, and hygiene release.

### Added

- Added private controller-bench and campaign-analysis documentation plus public
  Python API and examples-matrix guides.
- Added Ruff lint/format configuration and a local artifact cleanup helper.

### Fixed

- Fixed correctness-oriented Ruff findings, including mutable defaults,
  closure-captured loop variables, stale imports, and an undefined mission
  targeting knowledge reference.
- Tightened public export generation and checks so generated public releases
  remain Ruff-clean and exclude private-only documentation and local records.
- Limited default pytest collection to the supported `sim/tests` suite while
  ignoring scratch experiments.

## 0.4.0 - 2026-05-05

Named-object architecture and public workflow release.

### Configuration

- Added canonical `objects.<object_id>` scenario configuration for named
  satellites, rockets, and role-specific objects while keeping legacy
  `rocket`/`chaser`/`target` aliases as a compatibility layer.
- Updated curated public and private configs to use named objects and clearer
  comments around the parameters users are expected to vary.
- Added focused public use-case configs for TLE propagation, ground-station
  access, closed-loop rendezvous, attitude hold, and manual RPO training.

### Execution

- Moved single-run, Monte Carlo, sensitivity, and validation dispatch behind
  the `sim.execution` service boundary.
- Split core run payload assembly from artifact writing so the engine can be
  used more cleanly as a library.
- Kept generated Monte Carlo and sensitivity configs consistent when legacy
  object parameter paths are used with canonical `objects` configs.

### Reporting And Public Export

- Refreshed public/private documentation, README use-case guidance, and public
  export checks for the new example set.
- Preserved payload compatibility for existing consumers, including legacy
  rocket throttle fields.

### Migration Notes

- New scenario YAML should prefer `objects.<object_id>` paths. Legacy
  `rocket`, `chaser`, and `target` sections remain supported for compatibility.
- Batch parameter paths may still use legacy object aliases, but release 0.4.0
  synchronizes those changes into the canonical `objects` map before execution.

## 0.3.1 - 2026-05-01

### Governance

- Added a public README personal-capacity/no-endorsement notice.
- Softened public/commercial wording that named specific government audiences
  where broader aerospace training language was sufficient.
- Strengthened public/private documentation boundaries and export guardrails for
  local-only governance, compliance, and provenance records.

## 0.3.0 - 2026-04-29

Guided configuration and TLE initialization release.

### Configuration

- Added a headless-safe `configs/quickstart_5min.yaml` first-run scenario for
  new public users.
- Added `run_simulation.py --quickstart`, `--doctor`, start-here console output,
  `--open-output`, and a first-five-minutes guide for the public onboarding
  path.
- Added TLE-based satellite initialization support for scenario YAML configs.
- Added GUI support for a guided config-building workflow before running a
  simulation.
- Expanded config adapter coverage for generated YAML workflows.

### Reporting

- Added start-here output indexes for generated run artifacts.
- Fixed output indexes so non-path artifact values render as text instead of
  bogus links.
- Fixed default next-step links so they only point at artifacts that were
  actually saved.
- Included saved Monte Carlo histogram images in aggregate artifact inventories.

### Governance

- Strengthened public export checks for private-only assistant tooling and
  smoke configs.
- Moved public-facing safety, trusted-scenario, compatibility, and known-limitations
  guidance into the launch documentation.
- Reframed the private README as a buyer-facing Pro overview and moved the
  command-heavy workspace guide to private technical documentation.

## 0.1.2 - 2026-04-27

Validation maturity release.

### Validation

- Added estimation/knowledge validation harness coverage with evidence summary
  artifacts for RO/Kalman knowledge performance.
- Added attitude/disturbance validation scenarios covering nominal control,
  rate recovery, and disturbance-torque exposure.
- Added evidence manifests for validation harness runs, including git,
  dependency, benchmark, and artifact metadata.
- Hardened HPOP reference-data discovery through a reusable path resolver that
  supports `OEL_HPOP_ROOT` and local ignored HPOP checkouts.

### Governance

- Added validation setup and governance documentation for choosing suites,
  interpreting evidence artifacts, and maintaining private/public boundaries.
- Added a validation-plan helper that recommends focused checks from changed
  files.

### Fixes

- Preserved Monte Carlo run-summary payloads in run details so knowledge
  validation summaries can aggregate estimator evidence.

## 0.1.1 - 2026-04-27

Video game trainer release.

### Game Mode

- Added the optional Pygame RPO trainer backend and launcher with selectable
  training levels.
- Added six RPO learning levels covering natural relative motion, V-bar and
  R-bar approaches, close rendezvous, keepout recovery, and a defensive target
  demonstration.
- Added fullscreen gameplay controls, keyboard grab/release handling, speed
  multipliers, pass/fail scoring, live mission metrics, and on-screen debriefs.
- Added RIC translation controls, relative-motion projection difficulty
  settings, target-reference display support, burn markers, goal/keepout
  overlays, and close-range rendezvous zoom behavior.

### Packaging

- Added `pygame` as an optional `game` extra.
- Included built-in game YAML configs in package data so installed builds can
  open the level selector.

### Fixes

- Fixed terminal pass/fail states so the game window stays open long enough for
  the debrief to be read.
- Fixed Level 6 restarts so defensive target state and delta-v budget reset for
  each attempt.
- Fixed first defensive target pulses so they count against the target delta-v
  cap.

### Verification

- Added focused game-mode regression coverage for launcher discovery, controls,
  speed multipliers, scoring/debriefs, NMT goals, defensive target behavior,
  and package-data coverage.

## 0.1.0 - 2026-04-24

Initial public-core maturity release.

### Public Core

- Added a curated public-core workflow around deterministic single-run
  scenarios, the CLI, the Python API, the desktop GUI, plotting, object presets,
  and YAML-backed scenario configuration.
- Added generated public export tooling and boundary checks so the public
  repository contains the intended open-core surface.
- Curated public examples under `examples/configs/` for rendezvous,
  high-fidelity orbit/environment propagation, and manual engagement wiring.
- Added public documentation for quickstart, scenario YAML, plotting, plot
  gallery, public-vs-Pro boundaries, and product maturity direction.

### Private/Product Workspace

- Added Pro/private workflows for controller benchmarking, optimization,
  Monte Carlo and sensitivity campaigns, AI-assisted reports, validation
  harnesses, and early cFS/SIL integration patterns.
- Added a validation maturity plan covering current confidence level, HPOP/MATLAB
  evidence, remaining decision-grade gaps, and validation investment priorities.
- Added a product maturity roadmap focused on workflow curation, validation,
  contracts, public/private boundaries, and repeatable engineering use.

### Verification

- Expanded regression coverage across simulation, dynamics, controls,
  estimation, mission behavior, app/GUI services, public export checks,
  controller benchmarking, campaign reporting, and validation harness behavior.
- Aligned private CI with default pytest collection rather than a narrow
  hand-picked subset.
- Added generated-public CI checks for public export integrity, public package
  installation, public tests, curated config validation, representative example
  execution, and GUI startup smoke testing where Qt is available.

### Validation

- Added automated validation harness support for plugin/config validation,
  single-run benchmarks, Monte Carlo benchmarks, HPOP comparisons, MATLAB HPOP
  bridge runs, tolerance gates, and JSON/Markdown reports.
- Preserved historical HPOP/MATLAB parity evidence as private validation context.

### Scope

- This release is intended for research, prototyping, pre-flight engineering
  analysis, and software-in-the-loop experimentation.
- It is not flight-qualified software and should not be treated as operational
  decision-grade without independent mission-specific validation.
