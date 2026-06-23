# Public Core And Pro Boundary

Orbital Engagement Lab is organized as an open-core project.

This is the public-facing boundary summary. In the private source-of-truth
workspace, `docs/public-private-boundary.md` carries the operational export
rules and private-only path guidance.

The public repository contains the simulation foundation: dynamics, controllers,
estimators, actuators, sensors, mission modules, scenario YAML, API workflows,
examples, validation helpers, and lightweight extension surfaces.

Orbital Engagement Pro builds on the same foundation with higher-level analysis
workflows for teams that need repeatability, search, campaign management, and
review-ready outputs.

For the evaluator-facing overview of what Pro adds, see
[Orbital Engagement Pro](pro.md).

## Public Core

The public core includes:

- deterministic single-run simulation
- orbit and attitude dynamics
- reference orbit and attitude controllers
- sensing and runtime state-estimation primitives used by closed-loop
  simulations
- passive ground-station access tracking
- actuator models and mass depletion
- YAML scenario loading
- adversarial and engagement-style simulation primitives, including generic
  chaser/target knowledge, pursuit, evade, and defensive behaviors
- reusable object preset YAML files
- primary CLI, scenario YAML, Python API, review query, and custom review
  plotting workflows
- examples and starter validation workflows
- public-safe external-reference validation claims, commands, summaries, and
  artifacts for selected HPOP/MATLAB orbit and Basilisk attitude comparisons
  when the underlying reference material is redistributable
- public use-case configs under `examples/configs/public_*.yaml`
- rocket/ascent simulation primitives, educational launch-to-orbit scenarios,
  TVC/ascent diagnostics, and public rocket GNC contracts

The public core should be useful for research, education, prototyping, and
inspectable engineering experiments.

## Pro Layer

The pro layer includes:

- controller-benchmark suites and comparison reports
- optimization and gain tuning
- Monte Carlo campaign orchestration
- sensitivity studies
- covariance propagation and encounter uncertainty screening
- orbit determination against external observations or precise-orbit products
- SGP4 mean-element OD, residual-based maneuver screening, and burn
  investigation workflows
- batch nonlinear least squares and estimated-parameter workflows
- data ingestion, observation normalization, and mission-input packet creation
- campaign dashboards and baselines
- AI-assisted campaign reports from Monte Carlo and sensitivity outputs
- report cost estimation before hosted LLM calls
- curated validation and mission-assurance scenario packs beyond the public
  trust baseline
- validation automation, release evidence packaging, and customer-specific
  comparison reports
- rocket insertion engagement scenarios and deeper adversarial campaign packs
- custom and program-specific flight-software integration workflows
- Pro workflow configs under `examples/configs/pro_*.yaml`
- rocket/ascent benchmarking, optimization, payload-margin campaigns, and
  rocket guidance comparison workflows

Those workflow accelerators are intentionally not part of the public export.
Public modules that would otherwise expose those surfaces raise clear import
errors explaining the boundary.

AI report provider adapters, prompt templates, cost-estimation helpers, hosted
LLM smoke configs, and generated AI report artifacts belong to the Pro layer.
The public core may mention that Pro can add AI-assisted reporting, but public
examples should not require API keys or hosted model accounts.

## Design Principle

The public repo should feel complete as a simulation core. The pro repo should
feel like workflow acceleration around that core, not like the place where the
basic simulator lives.
