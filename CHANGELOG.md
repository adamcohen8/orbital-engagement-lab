# Changelog

All notable changes to Orbital Engagement Lab will be tracked in this file.

This project uses semantic versioning while it is pre-1.0: minor versions may
still introduce API or workflow changes, and release notes should call out
migration-sensitive behavior explicitly.

## Unreleased

## 0.21.3 - 2026-07-18

Release thesis: `v0.21.3` makes OEL's regression and performance feedback
faster and more memory-efficient while preserving deterministic simulation
outputs and the full private/public release gates.

Private/Pro scope: `v0.21.3` decomposes the private Scale and shared game test
God files into focused owners, adds measured fast feedback and exact clean-
commit evidence reuse, and retains all private validation and generated-public
release boundaries. The public release receives the compatible test workflow
and estimator runtime improvements.

### Changed

- Decomposed the game and Scale test God files into owner-aligned suites,
  replaced file-wide slow classification with measured test-level markers and
  a one-minute fast-feedback lane, shared isolated public-export fixtures, and
  added exact-clean-commit validation-evidence reuse without weakening the full
  private or generated-public release gates.
- Reduced estimator hot-path work by reusing invariant EKF matrices, removing
  identity observation products, and caching the latest HCW transition matrix.
  The maintained standard suite measured the full serial satellite case 4.1%
  faster and the sensing/relative-EKF case 11.3% faster with exact baseline
  physics hashes across every runnable benchmark case.
- Reduced sensing-run peak memory by 21.5% in the isolated benchmark by using
  the lower-level SciPy inverse-gamma primitive instead of importing
  `scipy.stats`, bounding maneuver-screen history, caching invariant thresholds,
  and packing long-lived knowledge statistics as doubles.

## 0.21.2 - 2026-07-17

Release thesis: `v0.21.2` decomposes OEL's largest implementation modules into
focused, discoverable owners while retaining stable façades and deterministic
simulation, configuration, artifact, and command-line contracts.

Private/Pro scope: `v0.21.2` gives Scale, campaign, controller-benchmark,
orbit-determination, AI-reporting, and validation workflows explicit ownership
maps and focused implementation families. The public release receives the
corresponding configuration, Python API, runtime, plotting, game, mission,
ingestion, observation, and public-export architecture improvements.

### Changed

- Decomposed plotting/output generation, runtime construction, object-step
  execution, history storage, and run-payload assembly behind their stable
  façades. Capability registries and architecture guides now route maintainers
  to focused owners while preserving figure IDs, artifacts, execution
  planning, histories, and deterministic simulation results.
- Decomposed the scenario configuration and public Python API God files behind
  their existing compatibility façades. Focused implementation packages and
  static ownership maps now cover schema/loading and API/session/workspace
  responsibilities without changing normalized configs, validation messages,
  public imports, class identities, serialized forms, or simulation outputs.
- Decomposed the private Scale SQLite store, command-line dispatcher, and Monte
  Carlo workflow behind their existing façades. Focused modules now own schema
  and migrations, catalog/measurement/OD/propagation/screening/refinement data,
  workflow command registries, deterministic campaign planning, execution,
  batch parity, quality gates, and reporting without changing database or CLI
  contracts.
- Decomposed the game training/dashboard/launcher/runner cluster, mission
  strategies and executions, controller benchmarks, orbit determination,
  campaigns, ingestion, observations, AI reports, validation harness, and
  public-export pipeline into focused implementation families. Existing import
  paths, CLI entry points, class identities, packets, reports, validation
  verdicts, and export behavior remain compatibility contracts.
- Updated the root, public, and private agent playbooks with façade-to-owner
  navigation, single-authoritative-implementation rules, compatibility checks,
  and explicit public-export boundary guidance.

## 0.21.1 - 2026-07-16

Release thesis: `v0.21.1` makes OEL performance work reproducible across the
full simulator, reduces time and memory overhead without changing deterministic
outputs, and promotes Scale ONP refinement from handoff packets to resumable,
review-store-backed execution.

Private/Pro scope: `v0.21.1` adds the durable Scale ONP execution queue and
saved MATLAB HPOP and Orekit zonal-reference validation lanes. The public
release retains the public-safe benchmark harness, deterministic runtime and
memory improvements, and less conservative laptop-safe resource guidance.

### Added

- Added the first executable OEL Scale Phase 3 boundary: canonical
  `handoff-onp` and `materialize-onp-refinement` commands now turn current
  handoff packets into deterministic passive two-object ONP scenario YAML,
  derive case-start ECI states through OGP plus the explicit Vallado IAU-80
  TEME-to-ECI reduction, enable standard review output, run safe-first and
  trusted canonical validation, and persist config/validation/source lineage in
  the additive Scale `0.6` `onp_refinement_cases` store table. Legacy
  `handoff-hpop` remains available as a compatibility alias; materialization
  does not execute ONP.
- Added the resumable Scale ONP execution queue. The new
  `run-onp-refinement-queue` command atomically claims validated cases,
  rechecks config digests and canonical validation, runs serially or through a
  bounded local process pool, records durable attempts in
  `onp_refinement_runs`, skips completed matching configs, and supports
  explicit recovery/retry. Each batch writes manifest and summary artifacts
  plus a review-store-derived OGP-vs-ONP sampled comparison CSV for TCA, miss
  distance, and relative speed.
- Added a maintained full-path performance suite covering the Basilisk common
  denominator, fixed and adaptive ONP propagation, the complete serial and
  parallel satellite loop, sensing and relative estimation, modern actuators,
  CR3BP, OGP-SGP4, rocket ascent, re-entry, campaign orchestration, and output
  artifacts. It records exact repeated-run physics hashes separately from
  timing evidence.
- Added saved external-reference workflows for MATLAB HPOP full-force cases
  and cumulative Orekit J2/J3/J4 cases, including provenance, refresh commands,
  release-plan integration, and focused regression tests.

### Changed

- Reduced acceleration-off startup and steady-state memory by loading optional
  dynamics, control, estimation, atmosphere, SciPy, and Numba paths on demand;
  eliminating unnecessary integrator diagnostics and history temporaries; and
  streaming large benchmark hashes and JSON artifacts. Repeated benchmark
  outputs remain exact.
- Improved deterministic throughput through cached signature compatibility
  plans, force-model and frame calculations, lower-allocation state/history
  updates, and bounded parallel campaign submission while preserving serial
  fallback and output parity.
- Made laptop-safe preflight use the single-run history estimate plus worker and
  plotting overhead, distinguish advisory from refusal headroom, honor the
  configured load threshold, prefer macOS pressure-adjusted memory, and avoid
  self-throttling serial campaigns on their own trailing load average.

### Fixed

- Preserved Scale ONP materialization identity across store migrations so
  changed refinement settings rematerialize cases instead of reusing stale
  generated configs.
- Kept private external-validation tests and reference tooling outside the
  generated public surface while retaining public-safe validation guidance.

## 0.21.0 - 2026-07-14

Release thesis: `v0.21.0` turns OEL orbit determination into a coherent,
evidence-producing capability from observation normalization and initial-orbit
determination through batch/sequential estimation, maneuver recovery, relative
OD, OGP mean-element fitting, covariance studies, and agent-facing pilot
workflows.

Private/Pro scope: `v0.21.0` adds the governed external OD validation campaigns,
Scale integrations, productization packets, and campaign-scale execution work.
The public release retains public-safe deterministic physics, relative-motion
models, observation contracts, documentation, and reproducible fallback
workflows without exposing private orchestration, external raw evidence, or Pro
report packets.

### Added

- Added a shared HCW/Schweighart-Sedwick J2 relative-dynamics contract across
  batch and integrated OD, live relative EKF knowledge, LQR/MPC/PD control,
  and linear rendezvous targeting, plus an ONP-derived nonlinear relative STM
  reference and explicit near-circular/mean-element/forcing metadata.
- Added OD Phase 6 regime-bounded OGP mean-element fitting across named
  OGP-SGP4 near-Earth and OGP-SDP4 synchronous, half-day-resonant, and
  high-eccentricity deep-space families, with native TEME PV packets,
  multi-day holdout, and queryable review evidence.
- Added automatic nonsingular/classical parameterization, separate
  state-space and element-space metrics, and prior/arc/postfit-identifiability
  gates that safely remove unsupported B* or epoch-offset trials.
- Added OD Phase 5 initial-orbit determination with geometry-gated Gibbs and
  Herrick-Gibbs position-triplet methods, checked-in Lambert candidates, Gauss
  angles-only roots, constrained-Lambert range hypotheses, and common batch
  refinement.
- Added the versioned optical RA/Dec observation contract with observer, epoch,
  frame, covariance, bias, and quality metadata; explicit unique, ambiguous,
  and insufficient result states; mission-input handoff; and queryable Phase 5
  validation evidence.
- Added the OD Phase 4 external campaign with checksum-pinned Orekit 13.1.7,
  GMAT R2026a, IGS final SP3, ILRS CRD v2, station metadata, and governed raw
  input provenance.
- Added independent Orekit batch-estimation fit/holdout references for circular
  LEO, eccentric HEO, and GEO, plus an executed GMAT DSN range/Doppler
  estimation and sigma-editing reference.
- Added GPS-time-aware SP3 ingestion and an IGS G01 MEO model-mismatch case.
- Added CRD v2 normal-point parsing, iterative two-way SLR light time, and an
  actual six-station LAGEOS-1 OGP-SDP4 fit/holdout workflow with queryable
  review evidence and explicit calibration caveats.
- Added Phase 3 ONP native ground-sensor OD systematics: station/shared range,
  range-rate, angular, and clock parameters with partial priors and
  identifiability evidence; elevation weighting; Bennett apparent-elevation
  refraction; station exclusion/holdout screening; queryable review tables;
  and a bounded versioned CCSDS TDM-compatible tracking adapter route.
- Added safe-first `SimulationWorkspace.validate_candidate_config()` for
  agent-authored YAML, with explicit trusted-plugin validation and no implied
  execution permission.
- Added provider-neutral agent report packet preparation and report audit
  workflows through the Pro CLI and Python API.

### Changed

- Cached exact HCW/SS observation-epoch STMs in batch relative OD and made all
  migrated controllers use the same exact ZOH discretization. SS-J2 reduces to
  HCW at J2=0 and remains an opt-in candidate judged against nonlinear ONP
  truth; it does not silently replace HCW or inherit HCW covariance claims.
- Cached TH variational and YA closed-form observation-epoch STMs, made the TH
  variational STM the default, and decoupled nonlinear truth plus sequential
  ECI RK4 substeps from observation cadence.
- Corrected optical RA/Dec covariance whitening, frame-consistent ground-station
  velocity and range-rate prediction, preliminary-fit covariance scaling,
  integrated relative-OD maneuver epoch semantics, mixed-dimension maneuver
  gating, and UKF fixed-interval smoothing cross covariance.
- Preserved full correlated, epoch-specific effective covariance in integrated
  relative sequential filters; made exported relative residual whitening match
  the solver; and recorded out-of-envelope SS-J2 rows as inapplicable instead of
  weakening the eccentricity guard or aborting applicable comparisons.
- Replaced relative angle/range/range-rate finite-difference Jacobians with
  analytic forms, removed six redundant transition propagations per UKF step,
  and added exact repeated-point caching to deterministic batch residuals.
- Reframed direct AI report provider calls as optional headless automation and
  direct AI config generation as a legacy compatibility adapter. Existing
  provider workflows remain available.
- Updated the Mendicant and MCP roadmaps so agents own configuration and report
  authorship while OEL owns deterministic validation, evidence packets, figure
  rendering, and report audit.
- Applied the CLI native-math thread policy to programmatic object and Scale
  workers, added duration-aware object-parallel crossover planning, bounded
  campaign submissions to twice the active worker count, and cached parsed
  Scale campaign configs per worker process.
- Made optional progress transport independent of parallel physics execution,
  retained already completed Monte Carlo results across backend fallback, and
  removed a duplicate config-campaign summary serialization.

### Fixed

- Fixed Python-API object-worker native-thread oversubscription that could make
  the process backend several times slower than serial despite exact physics
  parity.
- Fixed automatic object execution so short high-cost runs account for process
  startup and so transport failure can fall back before any failed step is
  applied; forced parallel execution continues to fail explicitly.
- Fixed campaign infrastructure failures so unavailable progress transport or a
  broken process pool does not discard completed deterministic work or
  unnecessarily disable otherwise available parallel execution.

### Performance Evidence

- On the local six-object high-fidelity ONP profile, the 120-second Python-API
  process run improved from 0.23x to 1.30x serial throughput after worker thread
  policy parity. The post-fix 300-second run measured 1.38x serial throughput.
- Serial and process histories were exactly equal for truth, belief, applied
  thrust, applied torque, and synchronized object knowledge. A six-second
  forced-process run remained startup-bound at 0.25x serial throughput and is
  now selected as serial by automatic planning.

### Migration Notes

- HCW remains the default circular relative-motion model. Schweighart-Sedwick
  J2, TH, and YA remain explicit model selections with their documented
  envelopes; no SS-J2 covariance claim is inherited from HCW.
- OGP remains the catalog-style SGP4/SDP4 mean-element family and ONP remains
  OEL's numerical propagator. HPOP names remain external-validation or legacy
  surfaces.
- Existing scenario, CLI, Python, review-store, and MCP tool contracts remain
  available. Runtime profiles gain duration-aware object-planner evidence, and
  asynchronous campaign callback completion order remains non-contractual.
- External Orekit, GMAT, IGS, and ILRS evidence is bounded to its named cases,
  checksums, epochs, frames, and licenses. It is validation evidence, not a
  flight-dynamics certification or operational accuracy claim.
- The MCP prototype remains source-checkout-only and at `prototype` maturity;
  supported SDK packaging and cross-host interoperability still wait for the
  stable official Python MCP SDK v2 release.

## 0.20.5 - 2026-07-13

Release thesis: `v0.20.5` hardens the experimental pre-v2 MCP boundary without
adding new tools or promoting the prototype to general agent-host support.

Private/Pro scope: `v0.20.5` adds complete normalized public/Pro golden
snapshots and a review-only structural diff tool while keeping those combined
fixtures and Pro contract details outside the public export.

### Added

- Added a public local-stdio MCP threat model covering trusted operator
  configuration, protected assets, deployment-profile interpretation,
  external-host disclosure, and the stop-line before remote/authenticated use.
- Added an actual wheel-build regression test proving that the source-checkout
  MCP prototype and MCP dependencies remain absent from the installed OEL
  wheel until a deliberate optional SDK v2 profile is introduced.
- Added full normalized golden snapshots for public and Pro tool definitions,
  capability responses, representative success/failure envelopes, and protocol
  errors.
- Added a private structural golden-diff command that reports exact normalized
  field changes and never rewrites the approved fixtures.

### Changed

- Direct-frontier results now replace local paths under authorized roots with
  opaque `oel-local-ref:<digest>` identifiers, including paths appearing in
  projected query rows and inspection metadata.
- Unexpected handler and protocol failures now return generic errors without
  local diagnostic details; reviewed policy, validation, and query errors
  retain their actionable safe messages.
- Public documentation now explicitly states that deployment-profile selection
  is trusted operator configuration rather than authentication, entitlement,
  classification, or release approval.

### Security

- Prevented authorized local filesystem roots, workspace names, customer path
  components, and unexpected backend diagnostics from crossing the
  direct-frontier result boundary.
- Added regression coverage for frontier path projection, query-row redaction,
  unexpected internal errors, and absence of Pro golden/tooling material from
  the generated public export.

### Migration Notes

- No MCP tool IDs, request schemas, ordinary public/Pro-local result semantics,
  OEL physics, or existing CLI/Python workflows change in this release.
- Direct-frontier callers must treat `oel-local-ref` values as opaque local
  provenance references rather than filesystem paths.
- The MCP prototype remains source-checkout-only and at `prototype` maturity;
  supported packaging and cross-host interoperability still wait for the
  stable official Python MCP SDK v2 release.

## 0.20.4 - 2026-07-13

Release thesis: `v0.20.4` establishes the public-safe contract, security, and
export groundwork for future OEL MCP interoperability without presenting the
pre-v2 prototype as a generally supported agent-host integration.

Private/Pro scope: `v0.20.4` adds a one-way composed Pro registry for bounded
IHE inspection and validation, plus deployment profiles for local Mendicant
and restricted frontier-agent use. These profiles do not authorize external
transmission or expose hidden evaluation truth.

### Added

- Added an optional, dependency-free, local stdio MCP prototype for capability
  discovery, completed-run inspection, and bounded read-only review queries.
- Added frozen transport-independent request and response schemas, explicit
  result projections, evidence-completeness states, effect metadata, resource
  limits, and payload-free audit records for five Phase 0 tools.
- Added separate public and Pro registries, with public, Pro-local, Mendicant
  sealed/tandem, and direct-frontier-restricted deployment views.
- Added Pro tools for inspecting visible IHE evaluation manifests and
  validating IHE studies without exposing hidden truth or executing physics.
- Added host-neutral conformance checks, adversarial path/SQL/size/handling
  tests, semantic CLI/Python parity fixtures, and golden compatibility hashes
  for discovery, success, and failure envelopes.
- Added an MCP roadmap and official Python SDK v2 adoption, supply-chain, and
  rollback checklist.

### Changed

- Extended the generated public-export allowlist to include only the
  public-safe MCP registry, handlers, protocol adapter, conformance helper, and
  documentation.
- Hardened the public-export checker so Pro MCP registrations, the composed Pro
  server, private contracts, and combined public/Pro golden fixtures remain
  absent.
- Separated configured MCP read roots from write roots and required explicit
  data marking and release scope for data-bearing calls.

### Security

- MCP path resolution fails closed on traversal, symlink escape, unauthorized
  roots, missing handling metadata, oversized inputs/results, unsafe SQL, and
  deployment-ineligible tools.
- The direct-frontier view rejects local-only data, and MCP discovery or
  transport is explicitly not treated as entitlement, release, or
  accreditation authority.
- Public MCP modules do not import, probe for, advertise, or leak Pro/IHE
  registrations or metadata.

### Migration Notes

- The MCP surface remains `prototype` maturity and is not installed as a core
  dependency. Existing CLI, Python, scenario, review-store, and agent-task
  workflows are unchanged and remain usable without MCP.
- Do not advertise this release as general Codex, Claude, or other MCP-host
  support. Official SDK-backed and cross-host support waits for the stable
  Python MCP SDK v2 release and the documented interoperability gates.
- The supported first SDK migration should replace the protocol adapter while
  preserving the frozen handlers, schemas, registries, security policy, and
  golden fixtures.

## 0.20.3 - 2026-07-12

Release thesis: `v0.20.3` improves simulation, plotting, and RPO Trainer
performance without changing physics or gameplay, restores exact external

Private/Pro scope: `v0.20.3` promotes exact-parity within-scenario ONP object
execution to an entitled automatic planner and coordinates it with campaign
parallelism through one hierarchical process budget.

### Added

- Added reproducibility manifests with canonical packet-evidence digests and
  ordering, decoding, timestamp, and timeout qualification counts.
- Added Pro `serial`, `parallel`, `auto`, and backward-compatible `configured`
  object-execution policies with eligibility, cost, resource, and
  serializability checks plus runtime planner evidence.
- Added a global hierarchical process planner for Monte Carlo, sensitivity,
  config queues, controller benchmarks, optimization particles, and child
  object workers.
- Added regression coverage for exact serial/process truth, belief, command,
  and knowledge parity; stable object allocation; inherited worker caps; and
  forced-parallel failure behavior.

### Changed

- Reduced ONP runtime overhead by synchronizing persistent worker state through
  pipes, avoiding redundant serialization and scratch allocation, and keeping
  the parent process authoritative for timeline and knowledge ordering.
- Reused the attitude EKF nominal propagation during numerical Jacobian
  construction and shared run-scoped orbital-element histories across plots.
- Reduced RPO Trainer rendering and scoring overhead through scalar dashed-path
  drawing and exact cached score geometry and delta-v terms.
- Updated public and private agent guidance for safe validation, TLE propagation
  contracts, output freshness, performance evidence, execution planning, and
  public/private workflow boundaries.

### Fixed

- Restored the MATLAB SGP4 bridge's current analysis schema and native-TEME
  comparison contract while retaining a canonical ECI cross-check.
- Fixed stale worker belief and controller state that previously caused
  process-execution output discrepancies in knowledge-coupled scenarios.
- Prevented nested campaign and object process pools from independently
  consuming the same host worker allowance.
- Kept public exports serial-only by excluding private execution-planner
  implementation and requiring the `object_parallelism` entitlement.

### Performance Evidence

- The six-spacecraft high-fidelity ONP profile improved from 254.56 s to
  61.99 s after the core performance pass, with exact non-timing summaries and
  exact review rows after timestamp normalization.
- The five-chaser, 300-second ONP case measured about 9.4 s serial versus
  4.8 s with six object workers; the one-orbit case measured 192.83 s serial
  versus 94.24 s with six workers. These are workload- and hardware-specific
  measurements, not universal speedup claims.
- The 120-frame RPO Trainer dashboard benchmark improved from 0.269 s to
  0.158 s, and the 5,000-sample training-score benchmark improved from 1.426 s
  to 0.539 s, with unchanged gameplay results.

### Migration Notes

- Existing object-execution configurations retain `configured` compatibility.
  Use `simulator.execution.policy: auto` for resource-aware Pro selection,
  `serial` to force parent-process stepping, and `parallel` only when an
  unsupported or under-resourced run should fail rather than fall back.
- Public-core scenarios remain deterministic and serial at the object layer;
  automatic object execution and hierarchical campaign/object planning require
  the private `object_parallelism` entitlement.
- Performance changes do not alter force models, integration contracts,
  controller behavior, game scoring, or gameplay rules.

## 0.20.2 - 2026-07-11

Release thesis: `v0.20.2` makes re-entry evidence physically consistent by
requiring diagnostic heating and load calculations to accompany a trajectory
propagated with atmospheric drag.

Private/Pro scope: `v0.20.2` also adds a rocket orbital-delivery accuracy workflow for downstream RPO feasibility studies, with deterministic dispersion sampling and review-ready evidence.

### Added

- Added validation coverage for drag-coupled re-entry trajectories and shared atmosphere-model configuration.
- Added Pro orbital-delivery targets in ECI, classical elements, or relative RIC; single-run delivery errors; Monte Carlo covariance and percentiles; a bounded correction-delta-v proxy; and RPO feasibility rates.
- Added positive rocket-stage dry-mass, propellant, thrust, and specific-impulse scale factors for deterministic campaign variation.
- Added a Pro orbital-delivery accuracy contract and an illustrative 24-run rocket-to-payload campaign fixture with explicit non-claim language.

### Changed

- Re-entry diagnostics now require `simulator.dynamics.orbit.drag: true` and use the atmosphere configured under `simulator.environment`.
- Monte Carlo summaries and output indexes now include orbital-delivery evidence when that private analysis is configured.

### Fixed

- Fixed configurations that could report atmospheric heating and loads for a trajectory propagated without the corresponding drag force.
- Fixed mismatched re-entry and environment atmosphere selections by rejecting them during scenario validation.
- Fixed orbital-delivery campaigns with no successful deployment so they report zero delivery and feasibility rates instead of omitting aggregate evidence.
- Fixed enabled orbital-delivery sections with missing object or target definitions so they fail during validation rather than at runtime.

### Migration Notes

- Scenarios with re-entry diagnostics enabled must explicitly enable orbit drag and configure their atmosphere under `simulator.environment`; a legacy re-entry atmosphere alias is accepted only when it matches.
- Orbital-delivery dispersion outputs are scenario-conditioned OEL evidence, not launch-provider accuracy claims, unless distributions are calibrated to documented mission data.

## 0.20.1 - 2026-07-11

Release thesis: `v0.20.1` consolidates overlapping public documentation into
fewer canonical guides and strengthens generated-export checks.


### Added

- Added canonical public capability-routing, quickstart, product-boundary, flagship-validation, and validation-operations guidance.

### Changed

- Consolidated overlapping onboarding, validation, public/private-boundary,
  and product documentation into fewer canonical guides with repaired links
  and export-manifest ownership.

### Fixed

- Fixed public-export checks and agent-documentation tests after documentation
  consolidation removed superseded pages.

### Migration Notes

- Public APIs and scenario contracts are unchanged in this patch release; moved documentation now resolves through the canonical index and guides.

## 0.20.0 - 2026-07-10

Release thesis: `v0.20.0` turns OEL's expanding propagation, planning, and
evidence foundations into clearer public workflows while hardening the private
Scale, intent-hypothesis, controller-lab, and release-governance surfaces. It
adds a bounded public Lambert transfer planner, strengthens OGP/ONP contracts
and provenance, and makes the public/private export boundary allowlist-first.

### Added

- Added a public two-body Lambert solver and grid-based Orbit Transfer Planner
  with zero-, one-, and two-impulse candidate classification, time/delta-v
  budgets, verification residuals, scenario configuration, review-store
  tables, plots, and regression coverage.
- Added a versioned workflow-evidence envelope and schema for reproducible
  workflow inputs, derived products, artifacts, and provenance.
- Added private OEL Scale hardening for catalog screening, propagation,
  recomputation, refinement, sensitivity, validation, operational-store
  provenance, and CLI workflows.
- Added private intent-hypothesis evaluation contracts, observational and
  synthetic evaluation packs, corpus generation, scoring workflows, and Scale
  handoff support.
- Added controller execution-policy support for deterministic deadline and
  failure behavior, plus private controller-lab catalog, agent, reporting, and
  command-line surfaces.
- Added scheduled-impulse interval delivery and impact-crossing coverage so
  off-grid events remain stable across integration step partitions.

### Changed

- Clarified OGP as the passive catalog-style SGP4/SDP4 family and ONP as the
  numerical propagation path across agent guidance, examples, configuration,
  API behavior, review metadata, and validation material.
- Strengthened scenario validation around mutually exclusive orbital-state
  forms, complete Cartesian states, plugin specifications, path policy,
  execution limits, and normalized configuration contracts.
- Refactored mission-recovery analysis into a public analysis module while
  retaining compatibility through the reporting surface.
- Improved attitude/orbit actuator handling, controller telemetry, frame
  provenance, OGP batch propagation, review-store metadata, and single-run
  artifact construction.
- Expanded RPO Trainer launcher, training, operator, debrief, and playback
  behavior while keeping synthetic video-game configurations outside the
  sensitive operational-scenario review policy.
- Consolidated installation metadata around `pyproject.toml`, added SciPy as a
  core numerical dependency, and added a clean wheel-build shim to prevent
  stale build artifacts from entering packages.

### Fixed

- Fixed public help text that referenced a private controller-bench scenario.
- Fixed scheduled impulses that could be missed when their delivery interval
  fell between integration-grid endpoints.
- Fixed frame, TLE, review-store, campaign, covariance, and validation paths
  that could lose provenance or accept incomplete configuration state.
- Fixed public export drift by switching to positive manifest ownership,
  explicitly keeping all IHE and Scale product surfaces private, and narrowing
  high-risk analysis and evidence promotion to reviewed files.

### Migration Notes

- Scenario files with unknown fields, multiple orbital-state forms, or a
  partial Cartesian position/velocity state now fail validation instead of
  being silently normalized.
- General optimization, campaign, Scale, IHE, controller-lab, and private
  validation workflows remain Pro surfaces. The public Lambert planner is a
  bounded two-body trade-space tool and is not an operational maneuver plan or
  a general optimization API.
- The former standalone GUI requirements file has been removed; install the
  `game` or `full` optional dependency profile for the downloadable trainer.

### Security And Supply Chain

- Regenerated the local CycloneDX SBOM for the release environment. The local
  `pip-audit` run reports CVE-2025-3000 against installed optional dependency
  PyTorch 2.12.0 because the advisory provides no fixed-version range; the NVD
  affected-version record is limited to PyTorch 2.6.0, so the finding is
  documented as non-applicable to the evaluated environment. Recheck the
  advisory in CI before release publication in case its affected range changes.

## 0.19.2 - 2026-07-05

Release thesis: `v0.19.2` tightens the RPO Trainer web-preview and Operator
Mode polish after the `v0.19.1` recording/playback release. It focuses on
mobile selector parity, script-screen ergonomics, and matching the browser
operator sandbox more closely to the downloadable trainer.

### Added

- Added mobile web-preview frame-convention controls so phone/tablet players
  can switch between OEL and Space Force display conventions from the level
  selector.
- Added script-screen burn insertion from a selected trajectory probe point in
  both the downloadable trainer and web preview, pre-populating the new burn
  time from the cyan trajectory marker.

### Fixed

- Fixed mobile web-preview selector layout so the OEL GitHub button remains
  inline with the other header controls in landscape mode and the frame button
  stacks correctly in portrait mode.
- Fixed web-preview operator sandbox playback to keep the in-level HCW
  projection one orbital period ahead after launch while preserving
  target-centered camera toggles.
- Fixed web-preview operator burn playback to use the downloadable game's burn
  slowdown and visual animation behavior during scripted execution.

## 0.19.1 - 2026-07-05

Release thesis: `v0.19.1` is a focused RPO Trainer polish release for the
Operator Mode and web-preview paths introduced in `v0.19.0`. It keeps the
release narrow: more reliable operator playback recording, more predictable
high-speed burn visualization, and a closer browser preview of the downloadable
script-planning experience.

### Added

- Added a three-second static Operator Mode script-screen lead-in to full
  attempt MP4 recordings when recording is enabled before launching the level.
- Added web-preview operator script plot overlays for planned trajectories,
  orange burn markers, post-burn velocity vectors, trajectory probing, and
  cyan time/state readouts.

### Fixed

- Fixed high-speed Operator Mode burn slowdown arming so 500x/1000x playback
  accounts for the full sim time a rendered frame may consume before crossing
  a scripted burn.
- Fixed web-preview operator sandbox script plots to remain target-centered and
  fixed burn-table editing so a clicked input stays focused while typing.
- Fixed stale web-preview shell mode classes after returning from gameplay to
  the level selector.

## 0.19.0 - 2026-07-05

Release thesis: `v0.19.0` prepares the downloadable RPO Trainer for a more
classroom-ready USAFA/OTC review by adding Operator Mode, persistent frame
convention settings, and a deeper level-selector/script-planning workflow. It
keeps the changes focused on training-game usability, astrodynamics teaching
clarity, and performance polish for the `v0.19.0` private/public release line.

### Added

- Added RPO Trainer Operator Mode, where players script time-tagged impulsive
  R/I/C burns before launch and then watch the spacecraft execute the plan.
- Added separate Pilot/Operator progress, persistent last-selected game mode,
  saved per-level operator burn scripts, and an operator-specific tutorial flow.
- Added operator script-screen mission briefs, numeric objectives, an equation
  sheet, full RI/RC preview plots, burn velocity vectors, trajectory probing
  with time/state readout, and a 10 second minimum spacing rule between scripted
  burns.
- Added frame-convention settings for OEL Default and Space Force-style display
  presets, including a first-run dialog, persistent local settings, and a
  selector settings button.
- Added signed RIC axis labels and frame-convention-aware plot display mapping
  while preserving the physical RIC dynamics behind the plots.

### Changed

- Reworked the downloadable trainer level selector for Pilot/Operator mode
  switching, wrapped level-description text, mode-colored selector controls,
  and circular keyboard navigation through levels and difficulty settings.
- Updated Operator Mode difficulty semantics so difficulty controls actuator
  execution error while operator playback always shows the full coast
  projection.
- Replaced the live elliptical coast projection path with a YA closed-form STM
  projection for both Pilot and Operator gameplay, with the previous numerical
  TH-style path retained as fallback.
- Refined operator burn visualization so projection transitions scale from
  1.0 to 2.0 seconds with burn magnitude and temporarily cap playback speed
  during the burn animation.
- Moved Pursuit Arcade out of the downloadable level list so leaderboard-style
  arcade play remains web-preview-oriented, while keeping the implementation
  available for future reintroduction.

### Fixed

- Fixed operator script-screen RI/RC previews so they match the first in-level
  game frame, keep initial RIC state readouts, and preserve target-centered
  camera behavior across levels.
- Fixed Level 0 naming so Pilot and Operator tutorials present distinct mode
  labels, with the operator tutorial using the same RIC primer animations before
  scripted burn demonstrations.
- Fixed operator playback inefficiencies by skipping guided-tutorial path
  updates, live pilot-burn prediction syncing, and manual maneuver bookkeeping
  during view-only operator levels.
- Fixed cislunar operator scripted burns to use the Moon-RIC frame path.

## 0.18.0 - 2026-07-01

Release thesis: `v0.18.0` improves the public RPO Trainer classroom-readiness
surface while expanding public frame, TLE/OGP, RPO estimation, and review-store
foundations. It keeps batch relative orbit determination, advanced OD, and
other Pro workflow layers private, with export checks tightened around that
boundary.

### Added

- Added public RPO estimation comparison material for HCW, numerically
  integrated TH, closed-form YA STM, and ECI EKF tracks, including validation
  configs and tests.
- Added EKF maneuver-detection support and validation coverage for delayed
  impulse detection.
- Added public frame-context, EOP-aware orbit-frame, and review-store support
  needed for stronger propagation, plotting, and evidence workflows.
- Added RPO Trainer UI support for an in-level pause reference screen,
  shared game fonts, an ISS-style target sprite, and additional public game
  training polish.

### Changed

- Updated public OGP/TLE initialization and ground-access examples, docs, and
  scenario help for the current catalog-style propagation posture.
- Improved RPO Trainer downloadable-game text rendering, debrief plot layout,
  RIC plot presentation, level geometry, and target-scaling behavior.
- Refined Level 3/4 training entries and RPO approach tuning while preserving
  public-safe game assets and configs.
- Updated supply-chain/security release posture to `v0.18.0`.

### Fixed

- Fixed debrief report presentation issues including RIC legend contrast,
  excess plot whitespace, timeline plotting, and watermark/axis-label overlap.
- Fixed target-reference curvilinear animation handling for generated output
  tests.
- Fixed public export drift so relative OD and its docs/tests/configs remain
  private while public relative EKF estimation remains exported.
- Fixed private fallback batch least-squares convergence when SciPy is absent
  by scaling the fallback residual-evaluation budget for finite-difference
  Jacobians.

## 0.17.0 - 2026-06-28

Release thesis: `v0.17.0` expands the public OGP/ONP propagation and evidence
surface while keeping orbit-determination and OEL Scale operational workflows
private/Pro-only. It updates public agent/docs language around OGP and ONP,
adds passive catalog-propagation and validation maturity work, introduces
opt-in object-step process-pool parity support, and tightens the public export
boundary for local development tools, configs, and Pro OD workflows.

### Added

- Added OGP/ONP-facing propagation, validation, and documentation updates for
  public catalog-style propagation workflows.
- Added scenario-YAML support for ground-station measurement metadata and
  opt-in object-level execution settings used for process-pool parity and
  profiling runs.
- Added private/Pro Scale and orbit-determination workflow scaffolding,
  synthetic measurement/OD paths, and related operational-store contract
  updates behind the public export boundary.

### Changed

- Renamed public-facing propagation language so OGP refers to the OEL General
  Propagator family and ONP refers to the OEL Numerical Propagator, while
  reserving HPOP for external reference and validation workflows.
- Updated validation governance, evidence-matrix, and model-validation docs
  for the current propagation and reference-comparison posture.
- Tightened public export rules so local development assets and OD-related
  implementation paths remain private/Pro-only.

### Fixed

- Fixed persistent `process_pool` object stepping so worker-mutated agent
  runtime state is returned to the parent simulator process before the next
  timeline step.
- Fixed public/pro documentation drift around ground-station observations and
  orbit-determination availability.

## 0.16.1 - 2026-06-27

Release thesis: `v0.16.1` is a narrow generated-public release. The public
product change is limited to a mobile RPO Trainer preview selector improvement,
with release metadata and boundary tooling updated to keep private/prototype
surfaces out of the generated public repository.

### Added

- No new public runtime APIs or scenario workflows.

### Changed

- Updated the RPO Trainer web preview so mobile users select a level first and
  launch it with an explicit `Play Level` button.
- Tightened public export rules so new private/prototype source, docs, configs,
  and tests stay out of the generated public repository.

### Fixed

- Fixed public export drift by stripping private helper APIs and tests from
  generated public-owned files when those helpers support private-only
  workflows.

## 0.16.0 - 2026-06-26

Release thesis: `v0.16.0` expands the public RPO trainer and public-export
governance while keeping the new catalog-scale screening work private/Pro-only.
It adds a Sun-angle inspection training level, hardens live-game runtime
behavior for long and high-speed attempts, introduces allowlist-first public
surface checks, and seeds the private OEL Scale workflow.

### Added

- Added the Level 6 Sun-angle inspection scenario, with Sun-angle constraint
  parsing, scoring, dashboard overlays, mission hints, debrief metrics, a new
  procedural music cue, and renumbered later RPO training levels.
- Added dynamic history mode for step-driven game sessions so long live
  attempts retain a bounded sample window while offline runs continue to use
  full-history payloads.
- Added measured-state object tracking as a lightweight estimator mode for
  state-measurement workflows that should trust the latest sensor state.
- Added public-surface manifest governance, including controlled namespace
  ownership checks, public config metadata requirements, and a private
  promotion checklist.
- Added private OEL Scale catalog-store, TLE ingest, SGP4 propagation,
  sampled pair-screening, refinement, handoff, CLI, docs, and tests; these
  paths are explicitly excluded from the generated public export.

### Changed

- Updated public and example configs with explicit public ownership,
  public-surface, and support-level metadata for release/export checks.
- Updated the RPO trainer input loop, two-rail speed behavior, cislunar CR3BP
  prediction sampling/cache behavior, mission terminal banner scrolling, and
  physical spacecraft marker scaling.
- Updated the web RPO trainer preview and arcade replay engine to preserve
  same-tick tap burns, show effective burn speed separately from coast speed,
  and use physical plot-scale spacecraft markers.
- Updated public/private boundary docs to describe the allowlist-governed
  public export model.

### Fixed

- Fixed scale pair screening so it only consumes current propagation products
  matching the active config's object set, propagation windows, model, backend,
  and output frame.
- Fixed ignored RPO trainer sprite max-size configuration by removing the stale
  parameter path and relying on physical marker sizing.
- Fixed manual game command latching for live player burns and preserved
  tracker replay evidence from game-owned history streams.

## 0.15.0 - 2026-06-23

Release thesis: this release consolidates OEL's validation and trust posture
after `v0.14.0`: it expands external-reference attitude and SGP4 evidence,
hardens agent/release evidence packets, makes canonical scenario YAML the
primary authored form, and removes unfinished desktop review surfaces from the
supported local workflow.

### Added

- Added first-pass attitude external-reference validation workflows, including
  analytic attitude reference cases, optional Basilisk comparison and
  step-size sweep runners, attitude-reference configs, harness integration,
  and private spike notes that document current limits and evidence.
- Added SGP4 reference-vector validation and SGP4 orbit-determination evidence
  suites with checked-in reference cases, harness specs, validation-plan
  routing, and model-validation documentation for what the evidence does and
  does not prove.
- Added model documentation under `docs/models/` plus a top-level physics
  model index so orbit, relative-motion, attitude, actuator, and environment
  assumptions are easier to inspect.
- Added public examples for actuator presets, mission recovery planning,
  orbital-elements stationkeeping, rocket launch-to-orbit, and explicit
  SGP4 passive TEME/ECI transform behavior.
- Added commercial-readiness and public-push guard tooling to make release,
  export, and public-repo operations harder to perform out of order.

### Changed

- Canonical scenario YAML now centers authored object definitions under
  `objects` and campaign configuration under `analysis`; legacy top-level
  `rocket`, `chaser`, `target`, `simulator.scenario_type`, and
  `monte_carlo` YAML aliases are no longer accepted by the strict scenario
  parser.
- Updated curated configs, docs, examples, prompts, and tests to use canonical
  `objects` / `analysis` YAML, while keeping Python API helpers able to
  normalize legacy dict conveniences where appropriate.
- Expanded review-store and agent-task evidence packets with schema
  compatibility metadata, config hashes, semantic metric request audits,
  saved-query completeness summaries, artifact/plot path status, and top-level
  `evidence_summary` readiness signals.
- Strengthened public export and release tooling with required public-surface
  checks, stronger private/pro boundary scans, release-readiness reports,
  retention guidance, source-clean status, and generated-public test patching
  for private-only agent-task recipes.
- Reframed the supported local review path around `python -m sim.review` and
  custom review plotting APIs instead of unfinished desktop workbenches.
- Updated package data and public export rules for SGP4 nutation data, public
  docs, and private validation/release artifacts.

### Fixed

- Fixed stale docs, prompts, and tests that still taught or exercised legacy
  YAML shapes after the parser moved to canonical `objects` / `analysis`
  authoring.
- Fixed public export drift around private SGP4 orbit-determination validation,
  Basilisk spike material, commercial-readiness tooling, and private-only
  agent-task test expectations.
- Fixed review evidence ambiguity by marking unknown or empty saved-query
  results, truncated rows, missing artifacts, and missing plots explicitly in
  machine-readable packets.
- Fixed generated public agent-task tests so public plot-summary coverage is
  preserved while private `dynamics_od_smoke` recipe assertions are removed.

### Removed

- Removed the unfinished PySide GUI, local app service layer, Evidence Studio,
  and legacy `run_gui.py`, `run_orw.py`, and `run_evidence_studio.py`
  launchers from the active codebase.
- Removed stale GUI/app/Evidence Studio tests and optional PySide dependency
  declarations that no longer matched the supported workflow.

## 0.14.0 - 2026-06-21

Release thesis: `v0.14.0` adds a public passive SGP4/general-perturbations
propagation path, tightens agent-facing config validation and release evidence,
and keeps orbit-determination evidence work behind the Pro/public boundary.

### Added

- Added passive catalog-object propagation with object-level
  `propagation_method: general` and `general.model: sgp4`, including TLE
  parsing fields, SGP4 propagation metadata in review stores, a public ISS
  example, and MATLAB SGP4 comparison harness wiring for local validation.
- Added Pro/private orbit-determination evidence paths for structured
  mission-input fitting and precise-orbit validation comparison, while keeping
  those workflows excluded or stubbed from the public core.
- Added stronger config-help and agent-repair guidance for observations,
  ground access, covariance, force-model/runtime settings, and strict plugin
  validation errors.

### Changed

- Updated public agent docs, examples, and release/export checks so TLE
  ingestion distinguishes numerical OEL propagation from explicit passive SGP4
  propagation.
- Expanded validation-plan recommendations, private merge evidence, and public
  export checks to include Ruff, config schema/help, controller-bench, reentry,
  and MATLAB SGP4 boundary coverage.
- Tightened physics/runtime behavior around attitude actuator telemetry,
  reaction-wheel sign conventions, RCS body-frame allocation, mass-property
  center-of-mass use, atmosphere local-solar-time handling, rocket aero/stage
  mass timing, and reentry g-load accounting.

### Fixed

- Fixed the agent capability-routing release coverage for
  SGP4/general-perturbations propagation so the public agent docs and tests
  agree.
- Fixed several strict-validation gaps for unsupported YAML aliases,
  object-level central-body overrides, ground-station schema drift, covariance
  pair coverage, and passive SGP4 object constraints.

## 0.13.1 - 2026-06-20

Release thesis: `v0.13.1` is a physics-correctness and release-readiness patch
that tightens atmosphere frame handling, sensor timestamp semantics, actuator
telemetry, rocket aero telemetry, and mass-property validation
ahead of the public export.

### Added

- Added regression coverage for attitude-dependent drag/SRP geometry,
  orbit-determination attitude/Cd fitting, mass-property validation, composite
  sensor timing, rocket aero telemetry, reaction-wheel telemetry, and re-entry
  frame plumbing.

### Changed

- Updated orbit-determination and atmosphere utilities to carry elapsed
  `jd_utc_start` timing, HPOP-like drag/density frame settings, attitude-aware
  geometry profiles, holdout metrics, and covariance diagnostics more
  consistently through validation and review evidence.
- Updated RPO trainer tuning, dashboard behavior, cislunar level config, and
  game-mode tests for the current training experience.

### Fixed

- Fixed spherical harmonics/zonal force double counting, attitude-coupled
  drag/SRP area resolution, reaction-wheel torque sign/telemetry, magnetic
  dipole scaling, MPC sign conventions, keepout crossing checks, delayed
  sensor timestamps, EKF/UKF elapsed-time prediction, stale measurement
  handling, RCS force-only torque passthrough, rocket q/Mach/aero telemetry,
  re-entry altitude/frame consistency, and validation of nonphysical mass or
  ECI initial-state inputs.
- Restored density-frame fallback to configured drag-frame settings so HPOP-like
  validation configs evaluate density and drag in the same Earth-fixed frame
  unless a separate density frame is explicitly requested.

## 0.13.0 - 2026-06-19

Release thesis: `v0.13.0` adds the private mission-input and orbit-determination
wedge, strengthens high-fidelity atmosphere/ephemeris validation paths, and
polishes the RPO trainer's cislunar and web-preview music experience while
preserving the public-core boundary.

### Added

- Added Pro/private mission-input ingestion, observation normalization,
  dynamics orbit-determination, estimated-parameter, fit/holdout evidence, and
  synthetic OD smoke workflows with contracts, docs, tests, and agent-task
  packet support.
- Added additional atmosphere model backends and data support for
  Harris-Priester, Jacchia 70, MSIS-86, and NRLMSISE-00 style validation and
  comparison work, plus precise-orbit and DE440-light validation utilities.
- Added a reproducible procedural lunar-mission music generator and wired the
  selected `30_far_side_navigation_demo.wav` cue to the downloadable cislunar
  rendezvous bonus level and the default public game-music export allowlist.

### Changed

- Updated the public/private export boundary so Pro ingestion, observation,
  batch OD, validation, and pilot-evidence materials stay excluded or stubbed
  from the public core, with stronger public-export rule checks.
- Updated the web RPO Trainer Preview sandbox music to use
  `06_casting_the_orbit_line.wav`, matching the broader heroic rendezvous tone
  intended for the v0.13.0 release path.
- Omitted optional WAV music assets from Python wheels while keeping the default
  public source/export distribution fully playable with runtime music.

### Fixed

- Updated resource-profile validate-only handling so validation can apply a
  requested profile without writing temporary profiled configs into scenario
  output trees.

## 0.12.4 - 2026-06-17

Release thesis: `v0.12.4` tunes the web Pursuit Arcade scoring loop and fixes
round-clear reporting so the browser preview is fairer for leaderboard play.

### Changed

- Updated web Pursuit Arcade round transitions so clears award 75% of the
  target orbital period, unused chaser delta-v is worth 100 seconds per m/s,
  boss clears add a 2000 second bonus, and new rounds restart at 1x playback
  speed with full-trajectory camera framing.
- Added end-of-run submission copy explaining that optional email verification
  reserves the username for future Pursuit Arcade scores.

### Fixed

- Fixed web arcade round-clear reporting so transition and HUD range values use
  the exact pass tick, while RI/RC goal rings remain stable projected range
  references.

## 0.12.3 - 2026-06-17

Release thesis: `v0.12.3` graduates the cislunar rendezvous trainer from beta
copy, improves high-speed game smoothness with a shared variable-step game
engine path, and adds orbit-plane plot swaps for richer spatial intuition.

### Added

- Added downloadable-game O/P plot swaps: `O` swaps the RI panel into an
  orbit-plane view and `P` swaps the RC panel into an orbit-plane view. In the
  cislunar Moon-RIC level, the swapped panel shows the Moon-centered target
  NRHO and the chaser's instantaneous position.
- Added shared speed-dependent game tick sizing for the Pygame trainer and the
  web preview helper, allowing large-base-step levels to remain visually
  smoother at lower playback speeds while preserving existing smaller-step
  behavior.
- Added optional Vercel Web Analytics support for Vercel-hosted preview
  deployments while keeping Plausible support and local/file analytics
  suppression.

### Changed

- Removed the beta tag from `Bonus Level - Cislunar Rendezvous` and moved it
  ahead of Pursuit Arcade in the downloadable level selector.
- Updated the cislunar trainer to use the corrected NRHO target near perilune,
  Moon-centered RIC controls, a 0.1 m/s close-approach speed limit, a
  100x high-speed maneuver cap, and a 30 FPS dashboard cap for better thermal
  behavior.
- Updated the cislunar target-orbit display to use a pre-propagated target
  orbit for the Moon view instead of a live projection tied to the current
  target state.

### Fixed

- Fixed cislunar CR3BP projection caching so reference-state cache validation
  accounts for the propagated reference motion over elapsed time.
- Fixed variable-step game stepping so thrust, torque, delta-v limiter timing,
  and saved histories use the actual per-step interval.

## 0.12.2 - 2026-06-17

Release thesis: `v0.12.2` is a web-preview polish release for the mobile-first
Pursuit Arcade and hosted leaderboard experience. It keeps the browser preview
lightweight while making the selector, HUD, camera controls, and release docs
clearer for social-media and classroom users.

### Added

- Added an `OEL GitHub` link to the web preview level selector so mobile,
  computer, and landscape users can reach the public repository before
  launching a level.
- Added a live `Time Remaining` readout and a combined `dV Remaining` arcade
  HUD line that reports target and chaser budgets from the validated arcade
  session state.

### Changed

- Updated the mobile web preview controls so the game action button becomes an
  explicit `Toggle Camera` control on mobile, while computer layouts keep the
  view selector.
- Reworked mobile landscape play into a wider plot-first layout with a lower
  control dock: status/actions/speed controls on the left and RIC burn controls
  on the right.
- Standardized mobile top bars for Sandbox and Pursuit Arcade with mission
  info on the left and objectives on the right.
- Updated selector copy from `Desktop` to `Computer`, removed the unused
  `Assists Easy` selector badge, and tightened selector button alignment across
  computer, portrait mobile, and landscape mobile views.
- Updated web-preview docs to describe the unified tutorial, sandbox, Pursuit
  Arcade, and hosted leaderboard paths.

### Fixed

- Fixed mobile plot taps changing the camera only while a finger was held down;
  mobile camera changes now use the explicit button instead.
- Fixed duplicate arcade score text beside relative speed and removed duplicate
  total-score text from the objective box.
- Fixed mobile long-press selection/callout behavior on burn and control
  buttons.
- Fixed landscape mobile selector and game layout issues where controls could
  overlap text or consume too much plot height.

## 0.12.1 - 2026-06-16

Release thesis: `v0.12.1` is a web-preview patch that makes Pursuit Arcade
more usable as a public, mobile-first first impression and prepares the hosted
leaderboard flow for small classroom or outreach competitions.

### Added

- Added a unified desktop/mobile web preview route with automatic view
  detection, explicit view toggles, and a compatibility redirect from the old
  mobile arcade page.
- Added mobile Pursuit Arcade controls for RIC burns, speed-multiple buttons,
  tap-to-toggle camera behavior on the plot panels, and a dedicated landscape
  mobile layout.
- Added leaderboard helper modules for username reservation, email verification
  ownership checks, and promotion of verified attempts onto the leaderboard.

### Changed

- Updated the web preview HUD, level selector, debriefs, and arcade text to use
  engineering-style distance and speed units that match the downloadable game.
- Updated the mobile web preview so Tutorial, Sandbox, Pursuit Arcade, music,
  and level selection share the same implementation path as the desktop
  preview.
- Updated leaderboard docs and deployment notes for optional email ownership
  and username reservation behavior.

### Fixed

- Fixed mobile landscape layouts where the plots could collapse to only a few
  pixels tall.
- Fixed mobile long-press behavior so holding burn controls no longer triggers
  browser text selection or callouts.
- Fixed browser compatibility hazards around blocked `localStorage` and older
  Safari media-query listeners.
- Fixed mobile arcade HUD copy so the music button no longer shows redundant
  keyboard-prefix text.

## 0.12.0 - 2026-06-16

Release thesis: `v0.12.0` expands the RPO Trainer into a more complete
classroom and web-preview experience. It adds the local Pursuit Arcade browser
prototype, cislunar rendezvous training, custom review evidence plotting,
and clearer local review exploration for completed-run inspection.

### Added

- Added a browser-native Pursuit Arcade prototype to the web RPO Trainer
  Preview, including deterministic two-body replay validation, multi-round
  local play, boss-round elliptic projections, static replay plot generation,
  and local competition fixtures/tests.
- Added the `Bonus Level - Cislunar Rendezvous` trainer mission with an
  opt-in Earth-Moon CR3BP propagator, an L2 NRHO target seed, Moon-centered
  target RIC controls, linearized CR3BP trajectory projection, custom
  Artemis-inspired sprites, and high-speed cislunar time scaling.
- Added CR3BP support for the game/training path, including physical
  Earth-Moon rotating-frame propagation, deterministic halo/NRHO seed states,
  Moon-RIC transforms, and STM-based projection support.
- Added `sim.review.EvidencePlotter` and `python -m sim.review plot` support
  for generating OEL-styled custom figures from completed review-store
  evidence.

### Changed

- Kept scripted review inspection centered on `python -m sim.review` while
  continuing local exploration of interactive review tooling.
- Updated the downloadable and web RPO Trainer arcade flow with round
  transitions, camera toggles, defensive target behavior, arcade/boss music,
  HUD layout fixes, and pause restrictions during arcade attempts.
- Updated agent and review docs to prefer the custom review plotting API for
  brief-ready figures and to reserve interactive review tooling for local
  exploration.

### Fixed

- Fixed `run_game.py --speed-multiple` so omitted values fall back to each
  level's configured default instead of raising on `None`.
- Fixed cislunar dashboard handling for full truth arrays, target-centered
  Moon-RIC plotting, larger level-specific sprites, and bounded cached CR3BP
  projections.
- Tightened the public web-preview deployment smoke checks so Pursuit Arcade
  module and music assets are included in the static Pages artifact.

## 0.11.0 - 2026-06-13

Release thesis: `v0.11.0` turns review evidence into the common inspection
surface for both humans and agents. It adds a repeatable agent-task runner,
golden-path adoption workflows, workflow-level review manifests and SQLite
tables for major analysis outputs, and a clearer post-run plotting role for
the experimental local review plotting tools. The release also polishes the RPO
Trainer experience for public education demos.

### Added

- Added `python -m sim.agent_task`, a machine-readable agent workflow runner
  with bundled public recipes, semantic review metric definitions, standard
  review plot recipes, config comparison packets, structured failure hints, and
  `agent_evidence_packet.json` output for repeatable agent handoffs.
- Added `docs/agent-golden-paths.md`, a public-safe first-run guide with exact
  validate/run/query loops for minimal propagation, closed-loop rendezvous, and
  mission recovery/reconstitution workflows.
- Added a review-store plotting service and experimental dynamic plot creation
  for completed runs, with saved-query/table loading,
  x/y/group column mapping, line/scatter/bar plots, OEL light/dark styling,
  PNG/SVG/PDF export, and provenance recorded in
  `review/generated_artifacts.json`.
- Added a common workflow review evidence layer for Monte Carlo, controller
  bench, sensitivity, and validation outputs. Supported workflow reporters now
  write `review/workflow_manifest.json`, table-backed `review/run.sqlite`
  evidence, schema/saved-view metadata, and workflow-aware `sim.review` query
  recipes.
- Added reproducible RPO Trainer visual assets and screenshots, including
  red/yellow satellite sprites and an instructor-facing one-pager with landing,
  level-selector, level 3, and level 5 imagery.

### Changed

- Linked the golden paths from the agent docs, public agent playbook, agent
  task-card index, review-query recipes, and documentation index so first-time
  agent users can start from reproducible adoption workflows.
- Extended OEL Agent regression coverage so golden-path configs, review output
  settings, saved-query names, and non-empty evidence rows stay aligned.
- Overhauled the public README source around a single fast proof path, tighter
  workflow routing, shorter AI-agent and RPO trainer sections, and consolidated
  trust/safety guidance.
- Clarified the public product posture: CLI/YAML/Python API plus the review
  query API are the primary simulation and review surfaces, the RPO trainer is
  the polished interactive surface, and local desktop tools remain outside the
  first-run path.
- Taught `python -m sim.review` to inspect workflow review manifests and
  artifact inventories while preferring workflow tables when opening
  table-backed review outputs.
- Updated the downloadable and web RPO Trainer UI around sprite-based vehicle
  markers, smoother live HCW projection during burns, compact command legends,
  matched start-screen panel styling, and cleaner start-screen scaling.

### Fixed

- Removed the paused single-step shortcut from the RPO Trainer controls and UI
  so the downloadable and web experiences match the intended start/pause flow.
- Fixed a sandboxed multiprocessing test so the private merge gate can verify
  parallel-campaign fallback behavior without relying on host IPC availability.

## 0.10.4 - 2026-06-12

Release thesis: `v0.10.4` strengthens OEL Agents as a public, evidence-backed
workflow by adding artifact-first Python scenario authoring, richer review-store
queries, and mission-recovery/reconstitution examples that remain tied to the
deterministic simulator and documented review evidence.

### Added

- Added `ScenarioArtifact`, `ScenarioBuilder`, structured validation reports,
  saved-config helpers, review accessors, and evidence manifests to the public
  Python API so agents, notebooks, and apps can create durable scenario YAML
  artifacts before validation and execution.
- Added public mission-recovery analysis through `analysis.mission_recovery`,
  including final-vs-initial orbital-element comparison, delta-v/propellant
  estimates, optional same-apsis slot search, deterministic planner trade-space
  candidates, and the `mission_recovery_trade_space` plot.
- Added review-store tables and saved queries for mission recovery and mission
  reconstitution evidence: `mission_recovery_summary`,
  `mission_recovery_elements`, `mission_recovery_candidates`,
  `mission_recovery_burns`, and `mission_recovery_candidate_elements`.
- Added public agent task cards, answer examples, and YAML fixtures for Python
  API minimal propagation, mission recovery after a +C burn, and mission
  reconstitution trade-space comparison.
- Added `outputs.stats.save_history_npz` to write compressed
  `master_run_history.npz` files with a manifest for scalable downstream Python
  analysis of long histories.

### Changed

- Updated OEL Agent instructions and public docs to prefer `.venv/bin/python`
  commands, inspect review-store schemas before custom SQL, and cite saved
  review queries or SQL when summarizing run evidence.
- Expanded scenario YAML, review-store, Python API, payload-artifact, and
  orbital-calculator docs for scenario artifacts, mission recovery, mission
  reconstitution evidence, and binary history outputs.
- Updated the validation harness to preserve the latest generated benchmark
  config, record benchmark runtime artifacts, and prune stale Monte Carlo run
  directories from prior harness runs.
- Bumped the review-store schema version to `0.3` for the new mission-recovery
  tables and review metadata.

### Fixed

- Fixed disabled plot and animation output paths so single-run artifact writing
  returns early when those output families are off.
- Tightened JSON output writing so NumPy/scalar values and non-finite floats are
  converted to JSON-safe representations before serialization.
- Added strict runtime plugin-validation behavior for bridge step failures when
  `simulator.plugin_validation.strict_runtime` is enabled.
- Updated public export ignore rules to keep temporary `.venv_temp/` runtime
  folders out of generated public exports.

## 0.10.3 - 2026-06-05

Release thesis: `v0.10.3` improves the RPO Trainer onboarding path by adding a
RIC frame primer to the downloaded tutorial, making the web preview open on the
existing level selector experience, and tightening web/full-game control
parity.

### Added

- Added a three-step RIC frame primer to the local Pygame tutorial, with
  synchronized RI/RC and ECI-style animations before the guided burn stages.
- Added the existing level selector-style landing screen to the web RPO Trainer
  Preview, limited to Tutorial and Sandbox entries for the hosted preview.
- Added a web preview level-selector music track and an in-game Level Select
  return control so browser users are not dependent on Esc.

### Changed

- Updated the web RPO Trainer Preview tutorial flow to start with the RIC frame
  primer before entering the existing guided Level 0 tutorial.
- Matched the web selector and sandbox metadata more closely to the downloaded
  game, including hiding progress/high-score fields for Sandbox.
- Improved primer and gameplay HUD layout so objective text and control lines
  avoid clipping across the preview and local game surfaces.

### Fixed

- Fixed web Esc handling so it returns to the level selector even when focus is
  inside a sandbox form control.
- Fixed local primer input handling so speed, camera, clip, and maneuver
  inputs do not leak into the guided tutorial while primer screens are active.
- Fixed RIC primer visual issues, including impossible ECI orbit sketches,
  cross-track inclination depiction, and an artifact inside the Earth render.

## 0.10.2 - 2026-06-04

Release thesis: `v0.10.2` polishes the web-hosted RPO Trainer Preview for
public launch by bringing sandbox camera behavior, tutorial/sandbox music, and
static-site smoke coverage closer to the downloaded game experience.

### Added

- Added browser-started tutorial and sandbox music to the web RPO Trainer
  Preview, with `M` key and HUD button controls.
- Added a public GitHub Pages smoke step that checks the static preview shell,
  JavaScript, CSS, and bundled music assets before deployment.

### Changed

- Updated the web preview HUD so mode/start/reset controls sit with the music
  control and the objectives area no longer repeats the coach text.
- Tuned sandbox camera framing and zoom behavior to better match the local
  Pygame trainer, with a sandbox-only camera rule toggle.
- Hid sandbox setup controls while a level is running and restored them when
  paused.
- Excluded web-preview WAV assets from lean public exports alongside the local
  game music files.

### Fixed

- Fixed web key-release handling so thrust inputs do not remain stuck after
  focus moves through native controls.
- Added a request-animation-frame fallback for browser shells that do not expose
  the API during static smoke checks.

## 0.10.1 - 2026-06-04

Release thesis: `v0.10.1` adds the first web-hosted RPO Trainer Preview path
and tightens the public/private release workflow around Pro-only covariance
analysis tests.

### Added

- Added `web/rpo-trainer-preview/`, a static browser-native RPO Trainer Preview
  with Level 0-style tutorial gates, sandbox presets, RI/RC plots, HCW coast
  projection, keyboard/touch controls, and a full-OEL download call to action.
- Added a GitHub Pages Actions workflow that publishes
  `web/rpo-trainer-preview/` as a static site when changes land on `main`.
- Added curvilinear rendezvous summary plotting and RIC 2D target-burn marker
  plot variants for richer public-safe rendezvous visualizations.

### Fixed

- Excluded the Pro covariance analysis regression test from generated public
  exports so public-export CI does not run a Pro-only workflow against public
  stubs.
- Fixed plotting test import ordering for Ruff.
- Improved dark-style Earth rendering and run-dashboard trajectory framing for
  public plotting artifacts.

## 0.10.0 - 2026-06-02

Release thesis: `v0.10.0` introduces **OEL Agents v0**, a tested public
workflow for AI coding agents to generate scenario YAML, validate configs, run
deterministic simulations, query review evidence, summarize artifacts, and
submit opt-in public-safe feedback without bypassing the checked-in physics
engine.

### Added

- Added the OEL Agents v0 task-card evaluation set with public task cards for
  passive propagation, closed-loop rendezvous, TLE ground access, attitude hold,
  and one-variable comparison. Each card includes prompts, assumptions,
  validate/run commands, review-store SQL, expected answer shape, pass criteria,
  red flags, and a golden answer example.
- Added `docs/agent-evaluation-packet.md`,
  `docs/agent-capability-routing.md`, `docs/agent-task-cards.md`,
  `docs/agent-review-queries.md`, and `docs/agent-feedback-loop.md` so users
  can route broad agent requests, evaluate agent behavior, query run evidence,
  and submit opt-in public-safe feedback.
- Added built-in named review queries exposed through
  `python -m sim.review --list-saved-queries` and
  `python -m sim.review --saved-query <name>`.
- Added public GitHub issue forms for bug reports, documentation issues,
  feature/workflow requests, and agent feedback.
- Added `tools/prepare_agent_feedback.py`, a local helper that drafts
  public-safe Agent Feedback issue text without submitting anything.

### Changed

- Updated public agent guidance so OEL Agents follow the evidence loop:
  natural-language request -> scenario YAML -> validation -> deterministic run
  -> review-store query or artifact inspection -> evidence-backed answer.
- Refocused public agent guidance on general user-intent handling: examples and
  task cards are onboarding rails and evaluation fixtures, not the boundary of
  supported agent workflows.
- Updated public agent examples to enable standard review output so agents can
  practice querying `review/run.sqlite` after a successful run.
- Updated public README and security/contribution docs to separate normal
  public bug reports from private vulnerability or sensitive-data reporting.

### Fixed

- Tightened public-agent preflight validation so malformed nested
  `relative_to_target_ric.state` blocks fail during `--validate-only` instead
  of surfacing later at runtime.
- Clarified `--doctor` Python failures by showing the detected interpreter
  version and the required Python `>=3.10` baseline.
- Added an attitude-specific saved review query for first/final quaternion and
  body-rate evidence, and clarified that orbital `thrust` rows are not
  reaction-wheel telemetry.
- Clarified the compare-one-change task card so scenario name and output
  directory edits are bookkeeping, while the one-change rule applies to
  physical scenario parameters.
- Added regression coverage that validates the public agent task cards,
  executes each card's review-store SQL against generated outputs, checks named
  saved review queries, verifies feedback draft generation, and confirms agent
  materials survive generated public export.

## 0.9.3 - 2026-05-28

### Added

- Added in-level RPO trainer clip recording for short demo/social captures:
  press G during gameplay to start a clip, press G again to discard it, and
  press Enter/Return to save it under `outputs/game_recordings/clips/`. F9
  remains an alternate when the operating system forwards function keys to the
  game.

### Changed

- Full-attempt game recordings now include three seconds of level-brief context
  at the beginning and three seconds of pass/fail screen context before
  finalizing.

### Fixed

- Prevented manual game clip recording from starting from briefing or terminal
  pass/fail screens; active clips can still be saved or discarded after a level
  ends.
- Avoided overwriting existing manual clip MP4s by appending a numeric suffix
  when a generated clip path already exists.

## 0.9.2 - 2026-05-27

### Added

- Added sealed mode for restricted/shared environments, with explicit gates for
  untrusted scenario plugin imports, hosted AI, custom AI endpoints,
- Added supply-chain and procurement baseline documentation, SBOM generation,
  dependency-audit evidence workflow, data-handling/export/CUI boundary notes,
  incident response guidance, and release-checklist gates for procurement
  review.
- Added a validation evidence matrix generator and catalog for current
  validation posture, including scenario, claim, validity envelope, truth source,
  pass/fail gates, last run date, commit/version, known limits, and status.
- Added numerical-methods convergence evidence for RK4/RKF78 behavior, adaptive
  step accounting, and resource-aware batch-study execution improvements.
- Added ML/RL policy-card helpers and docs that distinguish observer-owned
  belief/knowledge observations from truth-state oracle baselines.

### Changed

- Raised the supported Python baseline to Python 3.10 through 3.12 and updated
  first-run docs, doctor checks, package metadata, and generated public stubs to
  avoid accidentally creating Python 3.9 virtual environments.
- Clarified TLE workflows at the README, cookbook, example, and scenario points
  of use: TLEs initialize ECI state only, then OEL numerically propagates the
  configured force model rather than running SGP4/general perturbations.
- Tightened public README/CLI alignment around supported public flags,
  `.venv/bin/python` command paths, validation/export checks, and public/private
  docs boundaries.
- Updated HPOP/MATLAB reference handling and validation docs to distinguish
  available current evidence from external reference data that must be supplied
  locally.
- Expanded release checklist expectations for sealed-mode exceptions,
  dependency audit artifacts, validation evidence matrix regeneration, and
  public/private push discipline.

### Fixed

- Fixed sealed-mode validation for `outputs.ai_config` so it matches execution
  semantics and treats AI config creation as enabled by default unless disabled
  or dry-run.
- Fixed resource preflight and benchmark guidance for systems where plain
  `python` or `python3` resolves to an unsupported interpreter.
- Fixed public-export drift around private/pro paths, unsupported public
  workflows, and release-facing command examples.

## 0.9.1 - 2026-05-26

### Added

- Added the OEL plot style layer with dark and light Matplotlib themes,
  role-color conventions, artifact footer metadata, and shared helpers for
  static plots and animations.
- Added public README navigation and a more prominent OEL Agents section with
  example prompts for AI coding assistants.
- Added an opt-in SQLite review store writer for single-run outputs, including
  metadata, object-state, relative-state, thrust, ground-access, event, metric,
  and artifact tables for future review/query workflows.
- Added the `sim.review` SELECT-only query API and `python -m sim.review` smoke
  CLI as the recommended review path for agents/scripts, plus local-only
  experimental review plotting work that is not recommended for routine review
  workflows.

### Changed

- Unified built-in plot, animation, game-debrief, campaign, sensitivity,
  validation, and custom-analysis save paths around the OEL artifact style
  helpers.
- Refreshed the checked-in plot gallery images with OEL-styled artifacts and
  updated plotting docs to describe `oel_dark`, `oel_light`, and custom figure
  usage.
- Updated the public TLE propagation example to use OEL numerical
  special-perturbations propagation with J2, J3, J4, drag, SRP, and sun/moon
  third-body effects enabled, and clarified that the run is not SGP4/general
  perturbations propagation.

### Fixed

- Made OEL figure saving create missing output directories consistently with
  animation saving.
- Preserved scenario metadata in streaming Monte Carlo relative-range plot
  footers.
- Made plot artifact version discovery fall back cleanly in uninstalled or
  exported source trees with incomplete package metadata.

## 0.9.0 - 2026-05-26

### Added

- Added OEL Agents, a public-safe agent workflow layer for using AI coding
  agents with Orbital Engagement Lab through documented YAML, CLI validation,
  simulation execution, output inspection, and evaluation workflows.
- Added root `AGENTS.md`, `agents/public/AGENTS.md`, and `docs/oel-agents.md`
  so agents such as Codex, Cursor, Claude Code, and Gemini CLI can generate
  scenarios, validate configs, run simulations, inspect artifacts, and explain
  results without bypassing the deterministic physics engine.
- Added public agent-generated scenario examples for passive orbit propagation,
  closed-loop rendezvous, TLE ground-station access, and attitude hold.
- Added `agents/public/evaluation-rubric.md` to help agents distinguish config
  validity, execution evidence, physical interpretation, goal fit, and model
  limitations.
- Added public-agent tests that validate and run the agent-generated examples
  headlessly, plus private-only coverage for controlled internal agent
  guidance.
- Added an opt-in acceleration subsystem with optional Numba/JIT kernels, warmup
  tooling, local benchmark tooling, CLI/config controls, and an RK4 fast path
  for supported two-body/J2/J3/J4 orbit propagation plus accelerated
  exponential-map attitude propagation, RIC frame transforms, and orbit/attitude
  EKF finite-difference Jacobian paths.
- Added the product-facing RIC_PD 10 km flagship scenario, validation harness
  suite, validation package documentation, and public/private docs links.
- Added validation resource profiles, resource preflight reporting,
  checkpoint/resume for Monte Carlo validation, checkpoint cleanup commands,
  and runtime pressure gates for long local runs.
- Added `simulator.resource_profile` as the canonical scenario-level resource
  profile field and made unsafe resource preflight block validation/simulation
  starts unless explicitly overridden.
- Added raw knowledge-measurement histories, `knowledge_filtering` plots, and
  validation harness truth/measurement/estimate metrics for sensor distribution
  and estimator noise-filtering evidence.
- Added a dedicated noisy `knowledge_filtering_single` validation benchmark
  with normalized residual and filter-improvement gates.
- Forced game-mode runs to disable optional acceleration so gameplay never pays
  first-run JIT compilation latency.
- Added generated-config hashes to Monte Carlo checkpoints so stale or legacy
  checkpoints are ignored after scenario/controller changes.
- Added a product inventory covering orbital controllers, attitude
  controllers, and mission modules.
- Added campaign study metrics/gates, campaign plot/report extensions, and
  re-entry diagnostics/examples for private/pro analysis workflows.
- Added first-pass atmospheric steering for satellites with object-level lift
  coefficient, lift area, and attitude-coupled lift-axis specs.
- Added atmospheric-pass controllers, demo scenario, validation hooks, and
  plotting for aero-assisted plane-change/raise-burn studies.

### Changed

- Clarified that users should be able to ask agents for OEL workflows in
  ordinary language rather than copy-pasting structured prompts.
- Added rendezvous-specific agent guidance distinguishing "validated and ran"
  from rendezvous success, with notes on when to enable full logs, CSV, and
  plots for trajectory-quality review.
- Updated public export exclusions so private-only agent guidance and tests stay
  out of the public export.
- Renamed the flagship product-facing controller/scenario/docs from HCW PD to
  RIC_PD where the controller identity is RIC-frame PD rather than HCW as the
  user-facing method.
- Documented controller naming conventions for future controller modules,
  classes, configs, reports, and product-facing labels.
- Re-entry metric `active` now represents current threshold state, while
  summaries preserve ever-entered state, episode count, latest exit time, and
  cumulative heat load.

### Fixed

- Suppressed incomplete relative-range time-series plot artifacts when Monte
  Carlo resumes from checkpoints that do not contain the bounded plot series.
- Made resource preflight count canonical enabled objects instead of double
  counting normalized legacy/object aliases.

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
- Added a rule-based RMOE if-then orbit controller and a demo config/test path
  for RMOE-driven natural-motion targeting.
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
  scoring for release, validation, game, RL/ML, rocket GNC, simulator, and
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
- Added output-index support for rocket-aware run summaries, artifacts, and
  controller-bench result inspection.

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

- Added the public flagship `configs/ric_pd_10km_experiment.yaml` review
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

Predecessor RIC-frame PD controller and validation workflow release.

### Added

- Added a RIC-frame PD orbital controller with 10 km rendezvous experiment,
  Monte Carlo, and controller-bench tuning configs.
- Added campaign-level Monte Carlo range-timeseries plotting and Gemini-backed
  post-run AI report settings for the RIC-frame PD experiment.
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
- Explored local guided config-building workflows before simulation execution.
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
  scenarios, the CLI, the Python API, plotting, object presets, and YAML-backed
  scenario configuration.
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
  estimation, mission behavior, public export checks,
  controller benchmarking, campaign reporting, and validation harness behavior.
- Aligned private CI with default pytest collection rather than a narrow
  hand-picked subset.
- Added generated-public CI checks for public export integrity, public package
  installation, public tests, curated config validation, and representative
  example execution.

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
