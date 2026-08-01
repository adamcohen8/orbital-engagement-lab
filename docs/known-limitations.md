# Known Limitations

Orbital Engagement Lab is a public beta simulation core. It is useful for
research, education, prototyping, and pre-flight engineering analysis, but it is
not flight-qualified or operational decision-grade software.

## Public Beta Scope

- The public core is optimized for deterministic single-run simulation.
- The public core includes a bounded two-body Lambert solver and grid-based
  orbit-transfer trade-space example. General optimization, gain tuning, Monte
  Carlo, sensitivity studies, controller benchmarking, campaign dashboards,
  AI-assisted reporting, and custom flight-software workflows are Pro surfaces
  and are not included in the public export.
- The primary public surfaces are the CLI, scenario YAML, Python API, review
  query CLI/API, custom review plotting, and RPO trainer.
- Payload and artifact shapes are documented, but some non-contract fields may
  evolve while the project is pre-1.0.

## Validation Status

- The public repo includes unit and regression tests plus curated public
  scenarios.
- The engine has contract docs for scenario YAML, payload artifacts, and
  single-run behavior.
- The [Physics Model Reference](physics-models.md) documents model equations,
  assumptions, config knobs, implementation locations, evidence hooks, and
  limits so validation evidence can be interpreted in context.
- Public-safe external-reference validation evidence should be bundled when it
  is redistributable and tied to a specific claim. Proprietary reference data,
  customer-specific evidence, and large automated validation workflows remain
  private/product surfaces.
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

The package declares Python `>=3.10,<3.15`. Blocking public CI exercises the
default public test suite on Python 3.11. The wider release compatibility
program requires retained evidence for claimed Python/operating-system rows;
see [`compatibility.md`](compatibility.md). Installation and recovery commands
for Windows, macOS, and Linux are maintained in
[`installation.md`](installation.md). If you depend on a particular row, verify
that release evidence and run acceptance in the target environment.
Python 3.9 is no longer a supported procurement baseline because several
dependency vulnerability fixes require Python 3.10 or newer.

## Modeling Limits

- **OGP** means the **OEL General Propagator**: OEL's catalog-style
  general-perturbations family for TLE/mean-element products. The current OGP
  implementation supports **OGP-SGP4** for near-Earth SGP4 and **OGP-SDP4**
  for deep-space/resonance TLEs at or above the 225-minute period threshold.
- **ONP** means the **OEL Numerical Propagator**: OEL's configurable numerical
  propagation path for two-body and special-perturbation force-model studies.
  ONP is distinct from passive catalog-style OGP propagation. HPOP names
  external reference/validation workflows, not the native OEL propagator.
- TLE initialization samples OGP into an ECI-compatible initial state.
  Subsequent propagation uses the configured ONP force model; it does not
  continue as catalog-style OGP propagation unless the object explicitly uses
  `propagation_method: general` with `general.model: sgp4`. The legacy
  Keplerian mean-element initializer is available only as an explicit opt-in.
- OGP objects are passive catalog-style objects in the initial
  implementation. They do not accept thrust, orbit controllers, maneuvers, or
  covariance. TLEs with orbital period at or above 225 minutes dispatch to
  OGP-SDP4 instead of being treated as near-Earth SGP4. Continuous simulation
  histories are canonical ECI even when product metadata requests native TEME;
  direct OGP and Scale batch interfaces expose native TEME arrays. The engine's
  Vallado IAU-76/FK5 + IAU-80 TEME-to-ECI transform currently uses fixed time-scale
  defaults and zero EOP nutation corrections unless callers pass explicit
  corrections. Scenario-level Earth-fixed transforms can use
  `simulator.frames.model: iau76_80_eop` with an EOP file and are recorded in
  `frame_provenance`, but this remains a validation/parity surface rather than
  an operational frame service.
- Ground-station access is passive and geometric. It tracks line of sight,
  elevation, and range; it does not model RF link budgets, weather, scheduling,
  or command/telemetry behavior.
- Opt-in ground-station measurements are synthetic geometric rows. The v0
  public core preserves simulator-generated azimuth/elevation/range-style
  evidence for inspection, but calibrated sensor processing, association,
  bias estimation, orbit determination, and covariance-derived tracking
  evidence are Pro/private workflows.
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
