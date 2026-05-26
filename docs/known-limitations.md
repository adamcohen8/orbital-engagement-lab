# Known Limitations

Orbital Engagement Lab is a public beta simulation core. It is useful for
research, education, prototyping, and pre-flight engineering analysis, but it is
not flight-qualified or operational decision-grade software.

## Public Beta Scope

- The public core is optimized for deterministic single-run simulation.
- Monte Carlo, sensitivity studies, controller benchmarking, optimization,
  campaign dashboards, AI-assisted reporting, and custom flight-software workflows are Pro
  surfaces and are not included in the public export.
- The GUI is intended for scenario editing, single-run execution, and artifact
  inspection. It does not expose every lower-level config option.
- Payload and artifact shapes are documented, but some non-contract fields may
  evolve while the project is pre-1.0.

## Validation Status

- The public repo includes unit and regression tests plus curated public
  scenarios.
- The engine has contract docs for scenario YAML, payload artifacts, and
  single-run behavior.
- External high-fidelity reference data and validation harness evidence are
  private/product surfaces and are not bundled in the public repo.
- Users should independently validate behavior for their mission envelope,
  force models, time spans, controller assumptions, and numerical tolerances.
- Public scenarios such as `configs/ric_pd_10km_experiment.yaml` are review
  workflows and examples. They are not mission qualification evidence by
  themselves; see [Validation Claims](validation-claims.md).

## Scenario Safety

Only run scenario YAML files from sources you trust. Scenario configs can point
at importable Python modules/classes for controllers, guidance, mission
strategies, and mission execution modules. That extension model is powerful, but
loading an untrusted scenario can run untrusted Python code.

## Compatibility

The package currently declares Python `>=3.9`. Public CI exercises the default
public test suite on Python 3.11. Broader Python-version CI is planned; if you
depend on another supported Python version, run the public test suite locally in
that environment before relying on it.

## Modeling Limits

- TLE initialization uses a dependency-free Keplerian/two-body approximation;
  it does not perform full SGP4 propagation.
- Ground-station access is passive and geometric. It tracks line of sight,
  elevation, and range; it does not model RF link budgets, weather, scheduling,
  or command/telemetry behavior.
- Spherical-harmonic gravity can use inline terms or user-provided coefficient
  files. HPOP/GGM03 reference data is not distributed with the public core.
- Atmospheric re-entry diagnostics are first-pass aero/thermal estimates. They
  use atmosphere-relative speed, configured drag area/coefficient, and
  Sutton-Graves heat rate. Optional satellite lift is coefficient/vector based
  and intended for first-pass atmospheric steering studies, with vehicle aero
  properties supplied through `objects.<id>.specs.aero`. Reported re-entry
  g-load is aerodynamic drag load, not total acceleration during thrusting.
  The model does not include ablation, plasma, breakup debris, plume heating,
  high-fidelity hypersonic aerodynamics, or detailed thermal protection
  response.
- The bundled RPO trainer is educational and interactive; it is not a
  certification, mission-assurance, or operational training system.
