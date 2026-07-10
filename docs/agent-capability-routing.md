# Agent Capability Routing

Use this routing map when a user asks for OEL help that is broader than the
checked-in agent examples. The goal is to choose the smallest documented OEL
workflow that can answer the user's question, then validate, run, inspect
evidence, and state limits.

Examples and task cards are useful rails. They are not the full capability map.

Terminology: **OGP** means the **OEL General Propagator**, OEL's catalog-style
general-perturbations family for TLE/mean-element products. **OGP-SGP4** is the
supported near-Earth SGP4 path; **OGP-SDP4** is the supported deep-space
SDP4/resonance path. **ONP** means the **OEL Numerical Propagator**, OEL's configurable
numerical propagation path for two-body and special-perturbation force-model
studies. **HPOP** should be reserved for external reference/validation
workflows, not used as the name of OEL's native propagator.
When older agent checks say **SGP4/general-perturbations propagation**, route
that to OGP-SGP4 unless the request clearly needs deep-space OGP-SDP4.

## How To Route A Request

1. Identify the user's intent: propagation, rendezvous, access, attitude,
   plotting, manual training, comparison, validation, security review, or
   workflow automation.
2. Pick the smallest public workflow that can answer it.
3. Decide whether a checked-in example is close enough to copy. If not, create
   a scoped new YAML scenario with a distinct scenario name and output
   directory.
4. Ask only for details that change the study.
5. Validate before execution.
6. Run deterministic OEL commands only.
7. Inspect `index.md`, `master_run_summary.json`, review-store tables, CSV, and
   plots as appropriate.
8. Say what the evidence supports, what is missing, and what public-core limits
   apply.

## Routing Table

| User intent | Public workflow | Start here | Evidence to inspect | Ask before proceeding when | Do not claim |
| --- | --- | --- | --- | --- | --- |
| Propagate a satellite | Single-run scenario YAML with simple dynamics first | `configs/quickstart_5min.yaml`, `agents/examples/public_agent_single_satellite.yaml`, `docs/scenario-yaml.md` | `run_metadata`, `objects`, `object_state`, `master_run_summary.json` | Duration, initial orbit/TLE/altitude, or fidelity changes the answer | Operational ephemeris accuracy or mission validation |
| Use a TLE | OGP-backed TLE initialization recovers an ECI-compatible state, then ONP propagates configured dynamics; explicit continuous OGP propagation uses passive `propagation_method: general` | `examples/configs/public_tle_2hr_propagation.yaml`, `examples/configs/public_sgp4_passive_propagation.yaml`, `examples/configs/public_ground_station_access_from_tle.yaml` | `object_initialization`, force-model config or `object_propagation`, `object_state`, plots or CSV when enabled | User expects catalog-scale screening, current catalog freshness, covariance, or realistic force modeling | Continuous OGP behavior unless the scenario explicitly uses `propagation_method: general` and `general.model: sgp4` |
| Rendezvous or relative motion | Single-run RPO scenario with public controllers or passive relative motion | `examples/configs/public_closed_loop_rendezvous_lqr.yaml`, `agents/examples/public_agent_rendezvous_lqr.yaml` | `relative_state`, `metrics`, `thrust`, `events`, range/range-rate plots when enabled | Terminal success threshold, safety constraints, control posture, sensing/estimation assumptions | Terminal rendezvous success without thresholds and evidence |
| Time-fuel orbit change or slot acquisition | Use the Orbit Transfer Planner through `analysis.mission_recovery.planner.sources: [orbit_transfer]` for two-body Lambert trade-space candidates classified as zero-, one-, or two-impulse transfers | `configs/orbit_transfer_planner_demo.yaml`, `docs/scenario-yaml.md` | `mission_recovery_candidates`, `mission_recovery_burns`, transfer timing columns, delta-v/time budgets, verification residuals | Initial state, desired orbit/slot/state, time budget, delta-v budget, or fidelity is unspecified | Global optimality outside the configured search grid or operational maneuver readiness |
| Mission recovery from a simple burn | Use `analysis.mission_recovery` with `planner.sources: [analytic_reconstitution]` for the Original-Orbit Recovery Estimate and simulator-backed final-vs-initial comparison; use the orbital calculator for quick standalone estimates | `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml`, `agents/examples/public_agent_mission_reconstitution_trade_space.yaml`, `docs/scenario-yaml.md`, `docs/orbital-calculator.md`, `sim.orbital_calculator.mission_recovery_from_intrack_impulse` | `mission_recovery_summary`, `mission_recovery_elements`, `mission_recovery_candidates`, `mission_recovery_burns`, recovery delta-v, propellant estimate, slot-recovery time/tolerance | User needs a different target orbit, finite-burn optimization, non-impulsive planning, realistic ops constraints, covariance, or validated mission planning | That a two-body recovery estimate is an operational recovery plan |
| Ground-station access | Geometric access workflow from propagated states | `agents/examples/public_agent_ground_access.yaml`, `examples/configs/public_ground_station_access_from_tle.yaml` | `ground_access`, access samples, range/elevation histories, no-access reasons | Station, min elevation, duration, TLE, or force model is unspecified | RF link availability, scheduling, weather, or comms success |
| Attitude hold or pointing | Single-run attitude dynamics/control workflow | `agents/examples/public_agent_attitude_hold.yaml`, `examples/configs/public_attitude_hold_disturbance.yaml` | `object_state` quaternion/body rates, `attitude_state_first_last`, attitude plots when enabled | Target attitude, initial error, disturbance model, actuator assumptions, or settling metric is needed | Flight-qualified ADCS behavior or actuator margin evidence not recorded |
| Rich plots or visual review | Output artifact workflow with requested figure IDs | `docs/plotting.md`, `docs/plot-gallery.md`, relevant scenario YAML outputs | `index.md`, PNG files, plot metadata, source config, underlying review/CSV data | User needs a specific figure, axis, event overlay, or publication format | That a plot alone proves validation or correctness |
| Manual RPO training or game use | RPO trainer workflow | `run_game.py`, `docs/game-mode-roadmap.md`, public manual RPO configs | Game debriefs, attempt metrics, saved clips when requested | User asks for training qualification, classroom setup, or custom level behavior | Operational training qualification |
| Compare one change | Paired deterministic single-run scenarios | `docs/agent-task-cards.md`, nearby public config | Same review queries for both runs, output dirs, changed parameter, final metrics | More than one physical parameter changes, or statistical confidence is needed | Robustness, sensitivity, or causality beyond the controlled comparison |
| Monte Carlo, sensitivity, or campaign analysis | Public fallback is one-change or small manual comparison; full campaign workflows are outside public core | `docs/oel-agents.md`, public one-change task card | Paired-run evidence in public core; campaign artifacts only when available in the source tree under review | User asks for uncertainty distributions, rankings, batch studies, or optimizer behavior | Statistical robustness from one or two deterministic runs |
| Controller benchmarking | Public fallback is a focused deterministic comparison; benchmark suites are outside routine public-agent workflow | Public rendezvous configs, `agents/public/evaluation-rubric.md` | Metrics, final state, burn activity, plots, exact controller config | User asks for rankings, leaderboard, optimization, or suite-level claims | Controller superiority without benchmark evidence |
| Validation evidence | Run documented validation or release evidence commands when present | `docs/operations/RELEASE_CHECKLIST.md`, `docs/validation-claims.md`, `validation/` docs | Evidence manifests, validation matrix, pytest output, tolerance rows | User asks for mission qualification, HPOP parity, or procurement evidence | Validation beyond the specific evidence package |
| Security, sealed mode, or untrusted configs | Safe validation, sealed-mode validation, and public security docs | `SECURITY.md`, `docs/security/`, `run_simulation.py --safe-validate`, `--sealed-mode --validate-only` | Validation output, blocked surfaces, security docs, path-policy messages | Config is untrusted, uses plugins, external paths, hosted AI, custom endpoints, or non-loopback SIL | That validation makes untrusted code safe to execute |
| Custom analysis of a completed run | Review-store query or artifact analysis workflow | `docs/agent-review-queries.md`, `docs/custom-analysis.md` | `review/run.sqlite`, `index.md`, JSON/CSV artifacts, SQL query used | Review store is missing or requested metric was not recorded | Metrics that are not present in artifacts |
| AI-generated reports | Public core treats deterministic artifacts as report-like outputs; hosted AI reporting is not default public workflow | `index.md`, `master_run_summary.json`, plots, `docs/agent-feedback-loop.md` for feedback | Generated deterministic artifacts and provenance | User asks for hosted AI provider use, proprietary report packets, or private data | Private AI-report capability in the public core |

## Public Core Boundary

When a request depends on capability outside the public core, do not pretend an
example covers it. Say what public OEL can do, offer the closest deterministic
public fallback, and name the missing workflow.

Useful public fallbacks:

- Replace Monte Carlo or sensitivity requests with one controlled comparison.
- Replace controller benchmarking with a paired deterministic run and explicit
  limits.
- Replace AI-report requests with deterministic `index.md`, JSON, CSV, plot,
  and review-query summaries.
- Replace operational access/comms requests with geometric access evidence and
  stated exclusions.

## Evidence By Workflow

| Workflow | Primary evidence | Helpful saved queries |
| --- | --- | --- |
| Propagation | `object_state`, `run_metadata`, `objects`, summary JSON | `run_metadata`, `objects`, `passive_final_state` |
| Rendezvous | `relative_state`, `metrics`, `thrust`, `events` | `rendezvous_metrics`, `rendezvous_closest_approach`, `relative_final_state`, `burn_activity`, `burn_events` |
| Mission recovery | `mission_recovery_summary`, `mission_recovery_elements`, `object_state`, `metrics` | `mission_recovery_summary`, `mission_recovery_elements` |
| Ground access | `ground_access`, access/no-access rows | `ground_access_summary`, `ground_access_no_access_reasons` |
| Attitude | quaternion and body-rate columns in `object_state` | `attitude_state_first_last`, `attitude_rates_first_last` |
| Artifacts | `index.md`, `artifacts`, PNG/CSV/JSON files | `artifacts` |

## Clarifying Question Triggers

Ask before running when the missing detail changes the study:

- "realistic", "operational", "high fidelity", or "validated";
- "deorbit", "decay", "lifetime", or atmospheric effects;
- "access", "contact", or "comms" when station, elevation, RF, or scheduling
  assumptions matter;
- "rendezvous success" without range, time, delta-v, or safety thresholds;
- "uncertainty", "Monte Carlo", "sensitivity", "optimize", or "benchmark";
- untrusted scenario YAML, external paths, custom plugin modules, hosted AI, or
  networked integrations.

Default quietly when the detail is incidental: headless execution, plots off
unless requested, review output on for agent analysis, simple dynamics first,
and no campaign/report machinery unless the user asks for it.
