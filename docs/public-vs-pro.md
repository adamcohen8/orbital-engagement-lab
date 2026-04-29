# Public Core And Pro Boundary

Orbital Engagement Lab is organized as an open-core project.

The public repository contains the simulation foundation: dynamics, controllers,
estimators, actuators, sensors, mission modules, scenario YAML, API workflows,
examples, validation helpers, and lightweight extension surfaces.

Orbital Engagement Pro builds on the same foundation with higher-level analysis
workflows for teams that need repeatability, search, campaign management, and
review-ready outputs.

## Public Core

The public core includes:

- deterministic single-run simulation
- orbit and attitude dynamics
- reference orbit and attitude controllers
- sensing and estimation primitives
- passive ground-station access tracking
- actuator models and mass depletion
- YAML scenario loading
- reusable object preset YAML files
- Python API, CLI, and GUI entrypoints
- examples and starter validation workflows

The public core should be useful for research, education, prototyping, and
inspectable engineering experiments.

## Pro Layer

The pro layer includes:

- controller-benchmark suites and comparison reports
- optimization and gain tuning
- Monte Carlo campaign orchestration
- sensitivity studies
- campaign dashboards and baselines
- AI-assisted campaign reports from Monte Carlo and sensitivity outputs
- report cost estimation before hosted LLM calls
- curated validation and mission-assurance scenario packs
- cFS/SIL and program-specific flight-software integration workflows

Those workflows are intentionally not part of the public export. Public modules
that would otherwise expose those surfaces raise clear import errors explaining
the boundary.

AI report provider adapters, prompt templates, cost-estimation helpers, hosted
LLM smoke configs, and generated AI report artifacts belong to the Pro layer.
The public core may mention that Pro can add AI-assisted reporting, but public
examples should not require API keys or hosted model accounts.

## Design Principle

The public repo should feel complete as a simulation core. The pro repo should
feel like workflow acceleration around that core, not like the place where the
basic simulator lives.
