# Orbital Engagement Lab Product Maturity Roadmap

## Purpose

This document is a current-state product maturity assessment for Orbital
Engagement Lab. The project is no longer best described as a young simulation
framework. It is now a simulation product and engineering platform with a
public-core surface, Pro/private analysis workflows, validation tooling,
interactive workflows, and early external-integration patterns.

This update is based on the current repository state, the existing roadmap,
the 0.1.1 public release, the contract documents now present under
`docs/contracts`, the curated config and experiment surfaces, recent review
fixes around API plugin validation and keepout-time metrics, and the public
release of the optional Pygame RPO trainer.

The project already has:

- a deterministic multi-object simulator,
- orbit and attitude dynamics,
- sensing, estimation, mission behaviors, and control stacks,
- campaign tooling for Monte Carlo and sensitivity,
- controller benchmark and optimization workflows,
- validation harnesses and HPOP comparison utilities,
- AI-assisted reporting workflows,
- a desktop GUI,
- RL-oriented interfaces,
- a public optional video-game/manual-control training mode, and
- early cFS/SIL integration patterns.

The highest-value work now is productization, not feature sprawl:
workflow clarity, validation discipline, curated examples, operational
reliability, and confidence that users can do repeatable engineering work.


## Current Product Readiness

Scoring rubric:

- 1/10: scaffolding exists
- 5/10: real feature, but still experimental or specialist-only
- 8/10: users can do meaningful work with it today
- 10/10: mature, dependable, discoverable, and validated for repeated use

Current scores:

- Core Simulator Engine: 8.7/10
- Scenario YAML and Config System: 8.4/10
- Validation Harness and Fidelity Evidence: 7.4/10
- Controller Bench: 8.1/10
- Monte Carlo Simulation: 8.0/10
- Sensitivity Analysis: 8.2/10
- AI Reports: 7.2/10
- RL/ML: 7.0/10
- GUI: 6.6/10
- Video Game Mode: 7.1/10
- cFS/SIL Integration: 5.2/10

Interpretation:

- The engine is strong enough for meaningful work today. It has also crossed an
  important trust threshold: API single-run execution now validates strict
  plugin pointers before creating an engine, so programmatic users are less
  likely to get plausible-but-wrong fallback behavior.
- The scenario/config system is one of the more mature surfaces. Strict boolean
  parsing, timing-grid validation, preset resolution, plugin pointer validation,
  and scenario contracts are now real product assets.
- Analysis tooling is substantial and usable, but campaign/benchmark
  reproducibility and baseline semantics still need hardening before it feels
  like a release-grade analysis product.
- Validation is no longer merely a background activity. The repository has
  validation harnesses, reference configs, MATLAB/HPOP comparison scripts, and
  contract docs, but validation evidence is not yet packaged into a simple,
  continuously enforced product-readiness story.
- GUI, AI reporting, and RL/ML are useful but still power-user oriented.
- Game mode has crossed from prototype/manual-control demo into a public
  optional training surface. It is pilot-playable and public-demo-ready, but
  still needs onboarding, difficulty calibration, replay/review tools, and
  instructor-facing workflows before it is classroom-ready.
- cFS/SIL is a credible capability, but not yet a "ready by default" product
  surface.


## Current-State Highlights

What has improved since the prior roadmap:

- The project has shipped a 0.1.1 public release centered on the optional
  Pygame RPO trainer, proving that the public export process can carry a real
  feature release rather than only a static public-core snapshot.
- Contract artifacts exist for engine behavior, scenario YAML, payload/artifact
  shape, sensitivity analysis, and AI reports.
- API, CLI, and GUI single-run paths are closer to one conceptual model.
- Programmatic API execution now uses strict plugin validation rather than
  silently accepting broken plugin pointers when strict validation is enabled.
- Keepout dwell-time metrics now count elapsed intervals instead of samples in
  both engagement metrics and controller-benchmark metrics.
- Sensitivity analysis has grown into a real workflow with OAAT, LHS,
  two-parameter grid studies, generated-run preflight, failure-tolerant
  execution, report artifacts, and demo configs.
- Video game mode now has a selectable six-level RPO progression, packaged game
  configs, fullscreen Pygame gameplay, RIC translation controls, live metrics,
  pass/fail scoring, on-screen debriefs, difficulty-scaled trajectory
  projection, target-reference display support, and focused regression tests.
- Public/private boundary docs and export tooling exist, which is a strong
  signal that the project is thinking like a product rather than a code dump.

What still holds maturity back:

- The example/experiment surface is still larger than the verified product
  surface. Config-first examples are curated, but `examples/experiments` remains
  broad and historically accumulated.
- Generated outputs and local artifacts still appear in the working tree,
  reinforcing the need for stricter artifact hygiene and clearer user guidance.
- Test volume is broad, but test classification is still informal. The project
  needs an explicit smoke/product/regression/validation/slow taxonomy that CI
  can enforce.
- Batch workflows still rely on a mixture of newer execution-service boundaries
  and legacy master-simulator orchestration.
- Baselines, campaign manifests, and release-readiness checks are not yet
  systematic enough for the analysis features to feel fully operational.
- The project now has public releases, but release discipline is still light.
  A repeatable release checklist, test matrix, export checklist, and tag/publish
  expectations should become first-class project artifacts.


## North-Star Direction

Orbital Engagement Lab should become a simulation platform that is:

1) Architecturally coherent
2) Easy to configure correctly
3) Validated at multiple fidelity tiers
4) Useful for repeatable mission and autonomy studies
5) Capable of dependable campaign-scale analysis
6) Clear about what is public-core versus Pro/private
7) Stable enough that new work does not create silent regressions

Guiding principle:

Prefer depth over breadth.

That means:

- tighter interfaces over more one-off workflows,
- fewer but better examples,
- benchmarked and tested features over speculative expansion,
- and explicit product surfaces over "everything in the repo is a feature."


## Contract and Baseline Artifacts

The project has moved from "contracts needed" to "0.1 contracts exist and must
now be maintained." These contracts should be treated as product artifacts. A
behavior that is not documented, tested, and represented in an endorsed workflow
should be considered experimental even if working code exists.

Current contract artifacts:

- `docs/contracts/engine-contract.md`: step order, object lifecycle,
  truth/belief/knowledge timing, controller timing, snapshots, termination, and
  extension boundaries.
- `docs/contracts/scenario-yaml-contract.md`: top-level sections, strict types,
  timing grid, object presets, algorithm pointers, simulator/output sections,
  and migration expectations.
- `docs/contracts/payload-artifact-contract.md`: single-run payload shape,
  summary fields, history arrays, artifacts, and public API wrapper semantics.
- `docs/contracts/sensitivity-analysis-contract.md`: sensitivity study behavior
  and artifact expectations.
- `docs/contracts/ai-report-contract.md`: AI report packet and workflow
  expectations.

Still-needed contract/baseline work:

- benchmark contract: baseline storage, pass/fail gates, metric definitions, and
  tolerance review process,
- campaign contract: Monte Carlo manifests, seed/provenance rules, resumability,
  and partial-run semantics,
- validation evidence contract: what evidence is enough to call a physics or
  workflow surface validated,
- release-readiness contract: version bump, changelog update, private-main
  source commit, public export regeneration, public export check, public test
  subset, public push, and release tag expectations,
- public/private contract enforcement: the docs exist, but export checks should
  behave like an allowlist in practice,
- release-readiness checklist: the minimum tests, examples, contracts, and
  validation artifacts required before a feature graduates.


## Feature-by-Feature Path To 10/10

### 1. Core Simulator Engine

Current score: 8.7/10

Why it is already strong:

- deterministic kernel and multi-object execution model exist,
- the major dynamics/control/estimation pieces are wired together,
- API/CLI/GUI all rely on the same underlying stack for single-run workflows,
- public engine and payload contracts now exist,
- strict plugin validation is enforced for programmatic API engine creation,
- and the engine has broad test coverage.

Why it is not yet 9+/10:

- batch orchestration still has visible legacy/new boundaries,
- some payload/debug structures remain lower-stability than the public summary,
- and release-grade guarantees around API compatibility are still informal.

What would make it 10/10:

- lock down one canonical execution model across API, CLI, GUI, and batch flows,
- retire or isolate remaining legacy orchestration paths,
- formalize compatibility/stability guarantees for the public Python API,
- make run payloads and artifact metadata fully standardized,
- and maintain the engine contract with tests that protect step order, object
  lifecycles, truth/knowledge timing, plugin validation, and extension
  boundaries.

### 2. Scenario YAML and Config System

Current score: 8.4/10

Why it is strong:

- scenario YAML loading is central to the product,
- strict boolean parsing prevents a common class of YAML mistakes,
- timing-grid validation catches invalid simulation grids early,
- preset resolution and merge semantics are documented,
- plugin pointer shape is constrained and validated,
- config-first examples are becoming the primary supported workflow,
- packaged game configs now prove that non-core YAML workflows can be shipped as
  installed product assets,
- and the scenario YAML contract now exists.

What would make it 10/10:

- add an explicit schema version and migration workflow,
- publish a compact machine-checkable schema for editor/GUI validation,
- further improve invalid-combination diagnostics,
- keep one canonical config template per supported workflow,
- ensure all generated configs round-trip through validation cleanly,
- and align GUI controls exactly with the scenario contract.

### 3. Validation Harness and Fidelity Evidence

Current score: 7.4/10

Why it is meaningful today:

- validation harness code exists,
- HPOP comparison utilities and MATLAB bridge scripts exist,
- validation configs cover multiple perturbation/fidelity combinations,
- high-fidelity orbit and integrated-controller validation scenarios exist,
- estimation/knowledge and attitude/disturbance suites now package
  domain-specific evidence summaries,
- validation evidence manifests capture reproducibility metadata for harness
  runs,
- validation setup/governance docs and a validation planner helper exist, and
- validation appears as both documentation and test surface.

Why it is not yet 9+/10:

- validation evidence is now packaged at the harness/domain-summary level, but
  not yet as a full release-readiness package that a non-author can interpret
  quickly,
- reference-data provenance and tolerance rationale need clearer ownership,
- and validation suites are improving but still need clean CI/release tiers for
  smoke, regression, slow/reference, and external-tool checks.

What would make it 10/10:

- define validation tiers and required pass criteria,
- publish known-good baselines with provenance and tolerance rationale,
- make validation reports communicate pass/fail without plot forensics,
- wire representative validation suites into CI/release readiness,
- and maintain a validation evidence contract alongside the engine and payload
  contracts.

### 4. Controller Bench

Current score: 8.1/10

Why it is usable now:

- dedicated controller-benchmark subsystem exists,
- config-driven variant/case comparison works,
- reporting and plots are real,
- metric evaluation is improving, including corrected keepout dwell-time
  semantics,
- and the feature supports meaningful engineering comparison today.

What would make it 10/10:

- define canonical benchmark suites with stored baselines and pass/fail gates,
- stabilize the metric taxonomy and objective naming scheme,
- add stronger workflow docs for "build a new bench suite" and "interpret
  results",
- make benchmark outputs comparable across runs and versions,
- and integrate benchmark baselines directly into CI and release readiness
  checks.

### 5. Monte Carlo Simulation

Current score: 8.0/10

Why it is usable now:

- campaign execution plumbing is real,
- reporting and dashboard artifacts exist,
- parallel execution exists,
- AI report integration exists for campaign outputs,
- and the workflow supports meaningful analysis work.

What would make it 10/10:

- add resumable campaign execution and partial-run recovery,
- improve reproducibility guarantees around parallel workers and seeds,
- store campaign manifests and stronger provenance metadata by default,
- publish benchmark runtime envelopes and expected artifact sizes,
- define a compact, repeatable "campaign review" workflow for users,
- and split campaign orchestration contracts from legacy implementation detail.

### 6. Sensitivity Analysis

Current score: 8.2/10

Why it is usable now:

- one-at-a-time, LHS, and two-parameter grid paths exist,
- the feature is wired into API/CLI/GUI-facing services,
- generated-run preflight and failure-tolerant execution are implemented,
- JSON, CSV, Markdown, and PNG artifacts are produced,
- curated demo configs cover the main study modes,
- and sensitivity-specific AI report prompting and contracts are implemented.

What would make it 10/10:

- broaden standard sensitivity study templates tied to mission use cases,
- add stronger statistical interpretation for LHS and interaction studies,
- harden edge-case validation around distributions, timing grids, and
  high-cardinality studies,
- add more benchmark-backed examples of good sensitivity study design,
- and align the GUI/editor UX so sensitivity setup feels as polished as
  single-run scenario setup.

### 7. AI Reports

Current score: 7.2/10

Why it is usable now:

- provider adapters exist,
- dry-run support, packet generation, cost estimation, saved prompt/input
  artifacts, and CLI workflows exist,
- report contracts are now present,
- and report prompting is specialized for campaign, sensitivity, controller
  bench, and validation-style outputs.

Why it is not yet trustable by default:

- report quality is not yet benchmarked against known-good campaign/report pairs,
- the "analysis aid versus decision evidence" boundary needs stronger user
  guidance,
- and provider failures/retries/fallbacks need a more product-like behavior
  story.

What would make it 10/10:

- define stable report templates by use case, not just provider/model,
- strengthen error handling and retry/report fallback behavior,
- add report quality checks and output validation,
- make the "estimate -> inspect packet -> generate report" workflow more
  discoverable and standardized,
- and create benchmark campaign/report pairs that prove the reports add
  practical engineering value.

### 8. RL/ML

Current score: 7.0/10

Why it is usable now:

- Gymnasium-style wrappers exist,
- attitude and rendezvous environments exist,
- PPO training helpers exist,
- vectorized rollouts and self-play support exist,
- and tests cover core environment behavior.

What would make it 10/10:

- publish a small set of benchmark tasks with expected reward curves,
- provide stable observation/action schemas and version them,
- ship reproducible training recipes with runtime and hardware guidance,
- add artifact conventions for checkpoints, evaluation rollups, and replay,
- remove dependence on ad hoc local-output assumptions for policy demos,
- and define one blessed training workflow instead of several overlapping ones.

### 9. GUI

Current score: 6.6/10

Why it is usable now:

- it is more than a thin launcher,
- scenario editing and analysis controls are present,
- backend capability catalogs are wired through,
- and tests cover key service and IO flows.

What would make it 10/10:

- reduce the gap between what the GUI exposes and what advanced configs support,
- make analysis, outputs, and object relationships easier to understand in-app,
- improve validation/error feedback and migration guidance in the editor,
- tighten the conceptual model so GUI terminology matches docs and CLI exactly,
- add a few polished end-to-end GUI workflows that new users can complete
  without reading source or YAML first,
- and make the GUI read from the same schema/contract source as CLI validation.

### 10. Video Game Mode

Current score: 7.1/10

Why it is meaningful today:

- it is now a public optional training surface, not just an internal manual
  control demo,
- `pygame` is available through the `game` extra,
- `run_game.py` can launch a level selector without requiring a config path,
- six packaged RPO levels provide a progression from natural relative motion to
  rendezvous, keepout recovery, and a defensive target demonstration,
- RIC translation controls, speed multipliers, fullscreen gameplay, keyboard
  grab/release behavior, trajectory projection difficulty, burn markers,
  target-reference display support, keepout/goal overlays, live metrics,
  pass/fail scoring, and on-screen debriefs are implemented,
- focused tests cover launcher discovery, controls, speed behavior, scoring,
  NMT goals, defensive target behavior, terminal debrief handling, and package
  data coverage,
- and the mode is credible for public demos and supervised pilot usability
  testing.

Why it is not yet classroom-ready:

- first-time users still need stronger onboarding for RIC axes, NMT goals,
  keepout logic, and relative-motion intuition,
- difficulty has not yet been calibrated with novice cadets or instructors,
- pass/fail debriefs are useful but not yet a replay or trace-review teaching
  workflow,
- the level selector and HUD are functional, but still more engineering-tool
  than polished courseware,
- and instructor-facing knobs, lesson plans, and assessment exports do not yet
  exist.

What would make it 10/10:

- make the product purpose explicit: RPO intuition training for cadets and new
  Space Force officers,
- add a short in-game onboarding path with visual explanations of radial,
  in-track, cross-track, keepout, relative speed, and natural motion,
- add a replay/trace-review mode that marks burns, closest approach, goal
  achievement, and failure triggers,
- calibrate level tolerances and budgets through observed playtests,
- add instructor-facing level tuning and assessment export workflows,
- define classroom-ready lesson plans and pass/fail expectations,
- and verify full-screen behavior, keyboard capture, and rendering across the
  intended Windows/macOS/Linux classroom machines.

### 11. cFS/SIL Integration

Current score: 5.2/10

Why it is not just scaffolding:

- bridge code exists,
- an ICD exists,
- a mock endpoint exists,
- a starter cFS app/stub pattern exists,
- and there are scripts showing how the loop can be run.

What would make it 10/10:

- define simulator-time-master and rate-match behavior precisely,
- add repeatable fault injection for latency, jitter, packet loss, and replay,
- provide supported SIL benchmark scenarios with known-good timing,
- add transport diagnostics and interface conformance checks,
- package the startup/integration workflow so a non-author can run it,
- and publish a real SIL-readiness checklist with acceptance criteria.


## Roadmap Phases

### Phase 1 - Workflow Curation and Surface Ownership

Goal:
Make the product surface explicit.

Priority work:

- keep examples config-first and curated,
- add a lightweight release checklist covering version bumps, changelog entries,
  private/public push order, export checks, public test scope, and release tags,
- keep broad Python experiments out of the supported example surface,
- define endorsed workflows for:
  - public single-run scenarios,
  - public game-mode trainer release and demo flows,
  - private campaign analysis,
  - controller benchmark runs,
  - validation evidence runs,
  - AI-assisted review/reporting,
  - GUI-driven configuration,
  - SIL loop demos,
- align docs, examples, presets, and CI around the same workflows,
- and tighten public/private boundaries so public export reflects intentional
  product ownership.

Exit criteria:

- examples are few, curated, and verified,
- releases follow a repeatable checklist,
- public export contains only intentional public examples,
- docs and CI reference the same endorsed workflows,
- and no local-artifact-dependent demos remain in the product-facing example set.

### Phase 2 - Test Suite Rationalization and Validation Discipline

Goal:
Turn the test corpus into a product-confidence system.

Priority work:

- classify tests by value:
  - smoke
  - product/regression
  - validation/reference
  - research/experimental
  - slow
- make CI reflect those classes explicitly instead of hand-picking files,
- move all kept tests to temp-directory output handling,
- archive or delete historical tests that only protect abandoned surfaces,
- tie validation artifacts to benchmark scenarios and stored baselines,
- and make local default pytest, private CI, and generated-public CI differ only
  where the public/private boundary requires it.

Exit criteria:

- CI test scope is intentional and documented,
- tests do not write into repo-relative output dirs,
- benchmark regressions are automatically detectable,
- public and private CI both exercise their intended supported surfaces,
- and validation reports communicate pass/fail without plot forensics.

### Phase 3 - Platform Consolidation

Goal:
Make the framework feel like one system rather than a collection of related
systems.

Priority work:

- consolidate execution paths around the public API/session model,
- reduce overlap between legacy orchestration and newer execution service paths,
- standardize summaries, payloads, and artifact metadata,
- maintain the engine and payload contracts as code changes,
- clarify extension points versus internal implementation detail,
- and make single-run, stepped, Monte Carlo, sensitivity, and controller-bench
  workflows feel structurally related rather than separate inventions.

Exit criteria:

- API/CLI/GUI share the same conceptual execution model,
- internal legacy paths are clearly isolated or retired,
- core payload fields and artifact manifests have compatibility expectations,
- and users can understand the main execution flow from one source of truth.

### Phase 4 - Scenario and Config Maturity

Goal:
Make YAML authoring dependable and evolvable.

Priority work:

- simplify and document the schema around the current mission/objective model,
- add explicit versioning and migration guidance,
- improve invalid-combination detection,
- keep templates aligned with the scenario contract,
- keep a small set of canonical templates per workflow,
- and document supported combinations of dynamics, controllers, objectives,
  outputs, and analysis modes.

Exit criteria:

- users can start from a canonical config for each supported workflow,
- most authoring errors fail before runtime,
- and schema evolution is managed rather than tribal.

### Phase 5 - Analysis Feature Hardening

Goal:
Push Monte Carlo, sensitivity, controller bench, and AI reports from
"powerful" to "dependable."

Priority work:

- define canonical campaign and benchmark suites,
- add artifact provenance and reproducibility metadata,
- improve resumability and parallel robustness,
- store baseline manifests and make metric definitions explicit,
- standardize report formats and benchmark result interpretation,
- create known-good AI report benchmark pairs,
- and make AI report generation a clearly staged workflow with pre-call review
  packets, quality checks, usage/cost reconciliation, and report-payload
  adapters for each major product workflow, including validation harness runs.

Exit criteria:

- campaign and benchmark results are reproducible and reviewable,
- controller-bench and campaign outputs have stable semantics,
- and AI reporting is useful enough to trust as part of engineering review,
  including auditable prompt scope and provider usage records.

### Phase 6 - Validation and Fidelity Evidence

Goal:
Make validation an explicit product-readiness system rather than a collection
of useful scripts.

Priority work:

- define validation tiers and acceptance criteria,
- separate smoke validation from slow/reference validation,
- document reference-data provenance and tolerance rationale,
- produce concise validation evidence summaries,
- connect validation outcomes to release readiness,
- and make validation failures explainable without manual plot inspection.

Exit criteria:

- validation evidence can be reviewed by a non-author,
- representative fidelity checks are repeatable,
- and validated claims are traceable to configs, baselines, and tolerances.

### Phase 7 - Mission and Autonomy Maturity

Goal:
Make mission behavior, objectives, and autonomy use cases a first-class platform
strength.

Priority work:

- harden mission executive transitions,
- improve observability of mode changes and controller intent,
- define archetypal mission patterns and canonical scenarios,
- and add mission-level regression suites for representative objective flows.

Exit criteria:

- mission scenarios are explainable and debuggable,
- and objective-level regressions are automatically detectable.

### Phase 8 - GUI and Interactive Workflow Polish

Goal:
Raise GUI and game mode from useful tools to polished workflows.

Priority work:

- improve in-app guidance and workflow discoverability,
- align GUI concepts with docs and YAML terminology,
- polish interactive/manual-control scenarios and dashboard outputs,
- add RPO trainer onboarding, replay/trace review, and instructor-facing
  assessment/tuning workflows,
- calibrate game-mode level difficulty with novice users,
- and define when the GUI versus game mode should be used.

Exit criteria:

- a new user can complete at least one meaningful GUI workflow and one
  interactive workflow without reading source code,
- a novice can complete the first game level with only in-app guidance,
- an instructor can inspect or export pass/fail evidence from a training run,
- and the interactive surfaces feel intentional rather than incidental.

### Phase 9 - SIL and Real-Time Hardening

Goal:
Move cFS/SIL from starter kit to repeatable co-simulation capability.

Priority work:

- add transport diagnostics, timing contracts, replay support, and fault models,
- define benchmark SIL scenarios,
- package repeatable setup and run instructions,
- and produce formal readiness criteria.

Exit criteria:

- SIL workflows are repeatable,
- timing behavior is documented and testable,
- and communication faults can be injected and analyzed intentionally.


## Priority Order

Recommended sequence:

1) Workflow curation and surface ownership
2) Test suite rationalization and validation discipline
3) Platform consolidation
4) Scenario and config maturity
5) Analysis feature hardening
6) Validation and fidelity evidence
7) Mission and autonomy maturity
8) GUI and interactive workflow polish
9) SIL and real-time hardening

Why this order:

- product-surface clarity prevents future rework,
- better tests and validation increase confidence before more expansion,
- consolidation and config maturity reduce user and maintainer friction,
- analysis features become more trustworthy once the foundation is cleaner,
- validation evidence should become a product artifact before broader claims are
  made,
- and SIL hardening should happen after interfaces and validation norms are
  stabilized.


## Near-Term Recommendations

The next practical maturity jump should focus on six concrete moves:

1) Establish test classes and CI targets.
Define smoke, product-regression, validation-reference, slow, and experimental
test groups. Make the default local command and CI command intentional.

2) Create a release-readiness checklist.
Codify the 0.1.1 release flow: version bump, changelog entry, focused tests,
public export regeneration, public export integrity check, public checkout
tests, private push, public push, and release tag.

3) Turn contracts into living gates.
For engine, scenario YAML, payloads, sensitivity, and AI reports, add a small
contract-test set that fails when behavior drifts from the document.

4) Shrink the supported example surface.
Keep `examples/configs` as the user-facing example front door. Move broad
Python experiments behind clear local/experimental labeling and keep public
export conservative.

5) Create baseline-backed analysis demos.
Pick one Monte Carlo campaign, one sensitivity study, and one controller bench
suite as canonical baselines with expected summary artifacts and pass/fail
interpretation.

6) Produce a validation evidence summary.
Create one short, generated or maintained summary that says which fidelity
claims are currently supported, which are partial, and which remain exploratory.


## End Goal

A modular, validated, and operationally coherent orbital engagement simulation
platform where:

- the core engine is trusted,
- public and Pro/private surfaces are intentional,
- examples are curated and verified,
- tests reflect real product risk,
- validation evidence is easy to inspect,
- advanced analysis workflows are repeatable,
- and every major feature has a clear, credible path from current usefulness to
  10/10 product readiness.
