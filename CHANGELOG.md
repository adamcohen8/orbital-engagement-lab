# Changelog

All notable changes to Orbital Engagement Lab will be tracked in this file.

This project uses semantic versioning while it is pre-1.0: minor versions may
still introduce API or workflow changes, and release notes should call out
migration-sensitive behavior explicitly.

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
