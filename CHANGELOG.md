# Changelog

All notable changes to Orbital Engagement Lab will be tracked in this file.

This project uses semantic versioning while it is pre-1.0: minor versions may
still introduce API or workflow changes, and release notes should call out
migration-sensitive behavior explicitly.

## Unreleased

## 0.8.0 - 2026-05-24

### Added

- Added a Pro-only GNC workbench scaffold helper for custom orbit and attitude
  controller plugins, including generated controller code,
  strict-plugin-validation smoke scenarios, pytest smoke tests, local review
  instructions, and public-export exclusions.
- Added public actuator models for RCS thruster clusters, electric propulsion,
  spacecraft gimbaled thrusters, physical magnetorquers, simplified CMGs, wheel
  desaturation assist, and actuator fault/degradation wrappers.
- Wired public actuator-stack diagnostics into configured satellite runtime
  report resource data sources.
- Added public actuator-aware controller scaffolds for magnetorquer B-dot,
  wheel desaturation, CMG steering, RCS allocation preview, electric propulsion,
  and gimbaled-thruster reachability.
- Added local smoke configs for each actuator-aware controller family.
- Added public actuator presets for RCS, electric propulsion, magnetorquers,
  CMGs, and gimbaled thrusters; added strict actuator config validation, a
  multi-family actuator lab smoke scenario, and Pro GNC workbench
  `--actuator-preset` targeting.
- Added satellite presets for `CUBESAT_6U`, `SMALLSAT_RPO`,
  `TARGET_BUS_PASSIVE`, `ELECTRIC_PROP_SMALLSAT`, and `ADCS_DEMO_SAT`, with
  matching scenario YAML object presets.
- Added a public interactive Orbital Calculator with category menus for quick
  two-body, GEO, apogee/perigee, plane-change, sun-synchronous, phasing,
  atmospheric-drag/deorbit-lifetime range, rocket-equation, and robust
  state-vector/orbital-element conversion estimates, plus Hohmann rendezvous
  phase/wait-time helpers, first-order J2 secular-rate estimates, HCW
  relative-motion drift, eclipse, ground-track, and entry-interface estimates.

### Fixed

- Fixed `BASIC_RCS_6DOF` so the preset guarantees full six-axis force and torque
  allocation authority, including body-X roll torque.
- Fixed strict actuator validation so scalar-or-vector actuator fields accept
  scalar values when the runtime actuator stack supports them.
## 0.7.4 - 2026-05-23

RPO trainer debrief reports and Pursuit Arcade balance update.

### Added

- Added per-attempt Markdown game debrief reports for structured training
  levels, saved under `outputs/game_debriefs/<scenario_id>/attempt_.../` with
  `summary.json` and matplotlib plot artifacts.
- Added debrief plots for mission timeline, 2D RIC trajectory planes, relative
  range, relative velocity, cumulative delta-v, and RIC control commands.
- Added a terminal-screen `D` shortcut that closes the game and opens the
  debrief attempt folder for levels that generate debriefs.

### Changed

- Scoped debrief generation to structured training levels only; Sandbox and
  Pursuit Arcade do not generate debrief reports, and the tutorial report
  covers only the final free-maneuver phase.
- Replaced the debrief text event timeline with a timeline figure and rendered
  burn activity as filled start-to-stop intervals.
- Retuned Pursuit Arcade to a 3 m/s chaser delta-v cap, no regular-round flat
  time bonus, 1000 seconds per unused m/s of chaser delta-v, and a 5000 second
  boss-round bonus.
- Made Pursuit Arcade boss target eccentricity ramp from 0.05 upward by 0.05
  per boss round, capped at 0.20.
- Added a Pursuit Arcade target defensive delta-v ramp that holds at 0.1 m/s
  through round 20, then increases by 0.01 m/s per round.

### Fixed

- Fixed debrief 2D RIC plot conventions so radial position is vertical in RI
  and RC plots, cross-track position is vertical in IC plots, and the legend
  stays below the figure on one row.
- Fixed cumulative delta-v plotting to match the sampled acceleration
  integration used by the game training tracker.

## 0.7.3 - 2026-05-22

RPO trainer tutorial, Sandbox, and precision-thrust update.

### Added

- Added a guided Level 0 tutorial flow with staged +/-I, +/-R, and +/-C burn
  demonstrations, green target orbit paths, axis explanations, wrong-key
  feedback, and a required 10x speed-multiplier step after the first burn.
- Added an open-ended Sandbox mode after Pursuit Arcade with editable pre-launch
  RIC state and target orbit inputs, unlimited chaser delta-v, delta-v-used UI,
  target-centered RI/RC views, and success on the 20,000 second timeout.
- Added a Sandbox camera-rule toggle on `C` for switching between full
  trajectory/projection framing and satellites-only close-up framing.

### Changed

- Changed Sandbox timing to a 20,000 second scenario with a 1 second simulator
  step for finer manual maneuvering.
- Reworked manual game thrust timing so player key duration contributes
  fractional-step thrust duty cycle across tutorial, levels, Arcade, and
  Sandbox.
- Lowered the game dashboard minimum zoom span to 0.005 km and cached sampled
  dashboard trail/projection rows and burn markers for smoother close-range
  Sandbox views.
- Refreshed the product maturity roadmap against the current game/tutorial,
  Sandbox, recording, release, and RMOE-controller posture.

### Fixed

- Fixed Sandbox setup launch so the setup-applied runtime preserves the intended
  20,000 second duration and 1 second simulator step.
- Fixed Sandbox setup scrolling, RI/RC target-centering, and timeout completion
  behavior.
- Hid tutorial progress/high-score selector text and capitalized game UI
  indicators consistently.
- Fixed guided tutorial burn completion so the requested 0.25 m/s burn clears
  without needing excess delta-v.

## 0.7.2 - 2026-05-21

RPO trainer music packaging, recording polish, Level 5 tuning, and RMOE controller
support.

### Added

- Added default public game music assets for the RPO trainer, with a lean
  no-music public export option for smaller downloads.
- Added looped level music muxing for saved game recordings, while preserving a
  silent-recording fallback if audio processing fails.
- Added a rule-based RMOE if-then orbit controller, GUI controller registration,
  and a demo config/test path for RMOE-driven natural-motion targeting.
- Added initial game-module architecture boundaries for input polling, audio,
  recording, mission phase state, and tuning constants.

### Changed

- Mapped Level 1 to the existing `07_starfield_attract_mode.wav` track and
  tightened public packaging so only runtime-referenced WAV files are tracked.
- Retuned Level 5 around a 3 km in-track forbidden cylinder, four 0.25 km
  inspection cubes, and a 3 km-ahead chaser start for a safer passive RC-circle
  drift.
- Added `200x` game speed support and lowered the default recording FPS so very
  high-speed recordings are less expensive.
- Added public README instructions for users who only want to clone and launch
  the video game.

### Fixed

- Hardened game recording startup, frame capture, finalization, and cleanup so
  recording failures do not crash an active game.
- Cleared live burn axes on Pygame focus loss/minimize/hide events to avoid
  stale thrust input.
- Made opposing RIC translation key behavior explicit: opposite inputs on the
  same axis cancel while multi-axis burns still combine normally.
- Updated public-export scanning so binary music files do not trigger text-only
  sensitive-pattern checks.

## 0.7.1 - 2026-05-20

Pygame RPO trainer polish, elliptical-orbit lessons, and Pursuit Arcade tuning.

### Added

- Added Levels 6-8 for elliptical-target RPO: burn familiarization plus
  approach, elliptical NMC entry, and terminal rendezvous in the eccentric-orbit
  setup.
- Added Tschauner-Hempel-style elliptical coast projections, target true anomaly
  display, NMC boundary overlays, and level-specific RI/RC camera rules for the
  elliptical lessons.
- Added Pursuit Arcade boss rounds with elliptical target orbits, randomized
  target true anomaly, boss music, bonus scoring, and randomized energy-matched
  starts after round 1.
- Added new optional synthetic game music cues for Level 8 and arcade boss
  rounds.

### Changed

- Renumbered the former defensive/evasion levels to Levels 9 and 10 and updated
  launcher, roadmap, and public docs to match the current progression.
- Tuned Pursuit Arcade to tighten the goal radius by 5 meters per cleared round,
  cap the player at 5 m/s of delta-v, and award 3000 seconds plus 1000 seconds
  per unused m/s of chaser delta-v after each cleared round.
- Standardized game UI engineering notation and expanded level title/status
  text so active missions show the level number and name.

### Fixed

- Improved Pygame performance for high-speed play by throttling expensive
  projection refreshes, caching elliptical prediction work, and refreshing more
  aggressively while the player is burning.
- Fixed Escape handling so active levels return to the level selector instead
  of the start screen.
- Fixed long in-level instruction cards by allowing scrollable briefing text.
- Fixed camera framing for selected levels, including fixed Level 7 RI framing
  and Level 8 scaling from the two current satellites.

## 0.7.0 - 2026-05-17

Interactive RPO trainer expansion and public game export update.

### Added

- Added a branded game start screen, level-selector progress tracking, optional
  music controls, per-attempt debrief JSON, and in-game objective checklists.
- Added Level 0 tutorial training, a passive cross-track inspection level, an
  evasive-target survival level, and Pursuit Arcade mode with round scoring,
  random target evasion directions, and time bonuses.
- Added optional synthetic arcade music cues and round-clear sound effects, kept
  outside the source distribution as separately managed WAV assets.

### Changed

- Reworked game camera and plot scaling behavior for R/I and R/C training views,
  including level-specific zoom, fixed-axis support, target/chaser pair framing,
  and cleaner overlay/proximity-ring controls.
- Updated RPO training objectives, scoring, and mission feedback for burn-axis
  requirements, cross-track phasing, speed-multiplier practice, target delta-v
  limits, and arcade carry-forward timing.
- Replaced the old keepout-recovery Level 5 with a passive cross-track approach
  level and updated the game-mode roadmap to match the current level set.
- Kept optional game music WAVs out of the public export while packaging the
  start-screen PNG needed by the public game.

### Fixed

- Fixed arcade round transitions, music restart behavior, delta-v formatting, and
  missing-audio handling so the game can run without optional music assets.
- Fixed Level 2 and Level 3 plot framing and overlays so their RI/RC views match
  the intended training geometry.
- Fixed Level 1 cross-track phase-burn tracking so radial burns no longer satisfy
  the in-track or phase objective by accident.

## 0.6.2 - 2026-05-16

Rocket insertion, access reporting, and release-readiness update.

### Added

- Added a private current-architecture rocket insertion engagement scenario
  with regression coverage for insertion-triggered deployment and
  initialization-delay coasting.
- Added an interactive 30-degree inclined geosynchronous ground-track review
  config initialized over 90 degrees west longitude.
- Added satellite- and ground-station-oriented Markdown access reports with UTC
  AOS/LOS windows for runs that include ground-station access data.
- Added satellite physical-spec propagation for drag/SRP area and coefficient
  settings.
- Added commercial-readiness gates and expanded product maturity subcategory
  scoring for release, validation, GUI, game, RL/ML, rocket GNC, simulator, and
  public/private workflow planning.

### Changed

- Removed the legacy `sim.scenarios` package and its phased-ASAT/Monte Carlo
  helper APIs. Current scenario work should use YAML configs and the
  `sim.execution`/`SimulationSession` paths instead.
- Tightened mission decision APIs so agent-facing mission logic uses
  observer-owned knowledge instead of raw `world_truth`.
- Updated public/private export rules so private rocket insertion configs and
  their private regression tests stay out of generated public releases.

### Fixed

- Fixed ML/Gym mission helper calls after decision-facing `world_truth` removal.
- Hardened validation evidence dependency-version collection against malformed
  or missing package metadata.
- Removed excluded Pro rocket configs from the public examples matrix.
- Made public plotting tests skip cleanly when the optional Matplotlib stack is
  installed but unusable in the local environment.
- Made the versioned public release gate run release evidence commands with
  the invoking Python interpreter instead of a hardcoded `.venv` path.
- Restored the inclined GEO ground-track review config to interactive mode with
  the requested 30-degree inclination and 90-degree west longitude setup.

## 0.6.1 - 2026-05-15

Public workflow, plotting, and game-mode update.

### Added

- Added `config_help.py` plus public/private-scoped YAML config help topics so
  users can discover valid config fields and inspect current YAML values without
  running a simulation.
- Added orbital-elements conversion helpers, feedback controllers, mission
  strategies, stationkeeping/tracking example configs, and public plots for
  classical orbital-element histories.
- Added ground-station access plotting, map-backed ground-track support,
  attitude-control summary plots, and expanded rocket story diagnostics.
- Added in-level game-mode MP4 recording with level-selector controls, restart
  discard behavior, and 100x runtime speed support.

### Changed

- Moved requirements compatibility shims under `requirements/` while keeping the
  root `requirements.txt` shim for base installs.
- Moved private release/export operations docs under `docs/operations/` and
  project roadmap material under `docs/project/`.

### Fixed

- Fixed public manual RPO training to use RIC translation controls and fixed the
  defensive-target demo RIC command frame to use the target reference state.
- Kept generated public exports scoped to public config-help topics while
  preserving private Pro help topics in the private tree.
- Fixed optional ML dependency install hints after the requirements shim move.

## 0.6.0 - 2026-05-12

Rocket GNC, controller-bench, and public/private boundary release.

### Added

- Added private rocket ascent, launch-to-orbit, TVC tracking, and controller-bench
  GNC configs, including nominal, heavy-payload, wind, and PSO tuning cases.
- Added rocket navigation, orbital insertion scoring, ascent summary metrics,
  and controller-bench reporting for rocket GNC design workflows.
- Added GUI and output-index support for rocket-aware run summaries, artifacts,
  and controller-bench result inspection.

### Fixed

- Tightened the public/private boundary so pro rocket GNC workflows, configs,
  and contracts stay out of generated public releases.
- Added public export checks and release rehearsal artifacts for the new
  boundary rules.
- Guarded public release dry-run cleanup with a generated-workspace marker so
  existing ordinary directories are never deleted by accident.

## 0.5.1 - 2026-05-08

Security, resource, and public-boundary hardening release.

### Added

- Added caller-controlled config path policies for scenario presets, output
  directories, external dynamics data, and AI prompt/question files.
- Added AI endpoint trust gates so hosted provider requests use built-in
  endpoints unless callers explicitly opt into custom endpoints for trusted
  configs.
- Added single-run history memory budgeting with CLI/API caps before dense
  history arrays are allocated.
- Added bounded Monte Carlo relative-range plotting that avoids retaining every
  run's full time history in campaign memory.

### Fixed

- Verified downloaded real gravity model cache files by size and SHA-256 before
  using or replacing managed cache files.
- Hardened public export checks to scan the full generated tree for sensitive
  text, local-only files, and cache artifacts.
- Added Ruff checks to private and generated-public CI gates.

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
