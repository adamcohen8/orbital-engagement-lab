# Coverage and Directed-Link Programmatic Acceptance

Status: **public experimental programmatic acceptance; coverage has a retained
independent reference comparison, and directed-link external validation is
deferred to a future release**.

This record covers the deterministic kernels introduced through Coverage
Phases 1-6. It is not a release-readiness or operational-authority claim.

Canonical conical coverage and directed object-to-object links now also have
evidence-only scenario YAML execution, completed ONP/review and ECI OGP history
normalization, primary review-store tables, and provider-backed event
refinement. These adapters have focused integration coverage. Rich coverage,
communications coverage, aggregation, tasking, causal ONP consumers, and
agent-facing execution still require separate adapters.

## Reproducible Gate

Run from an activated OEL source checkout:

```bash
MPLBACKEND=Agg python -m pytest \
  sim/tests/test_global_coverage_phase1.py \
  sim/tests/test_global_coverage_phase2.py \
  sim/tests/test_global_coverage_phase3.py \
  sim/tests/test_coverage_link_remaining_phases.py \
  sim/tests/test_orbital_analysis_adapters.py -q
```

The acceptance fixtures cover:

- frozen conical coverage parity, HEALPix identity, sparse intervals, dwell,
  revisit, censoring, regional/point queries, and deterministic artifacts;
- independently generated Astropy HEALPix 1.0.3 identity vectors and matched
  WGS84/cone end-to-end covered-cell counts for orders 5 through 8, retained in
  `validation-coverage-healpix-astropy-v1.json`;
- rectangular and pushbroom hard FOVs, WGS84 sampled boundaries, explicit Sun
  evidence, service constraints, chunk parity, and plot generation;
- an independently coded scalar hand ledger, exact SI constants, range-
  doubling loss, zero-margin closure, RF monotonicity, WGS84 clear/tangent/
  occulted/inside-Earth cases, attitude and non-identity mounting, fixed-site
  elevation, scalar/batch parity, and provider-refined/sample-bounded events;
- RF-qualified global coverage with an explicit Earth-terminal profile,
  non-closing margin fixtures, Phase 2 query compatibility, resource limits,
  chunk-stable scientific identity, and deterministic artifacts;
- exact synthetic constellation union, overlap, required multiplicity, failed
  cells, dwell, revisit, uniform domain/service identity, malformed-source
  rejection, and artifact fixtures;
- exact bounded task selection with source and asset bindings, slew/settling,
  duty, storage, horizon energy, downlink, stable tie behavior, and artifacts;
- authorized next-boundary runtime delivery with consumer isolation and exact
  link-configuration binding; and
- source-bound cadence sensitivity evidence with nested epochs, matched
  non-refinement assumptions, and explicit caller limits.

The margin review plot is also inspected for legibility, clipping, threshold
visibility, and legend placement. Automated plot generation alone is not
visual acceptance.

## Outstanding Scientific Gate

Coverage now retains one matched-assumption independent reference comparison in
`validation-coverage-healpix-astropy-v1.json`. It binds exact inputs, frame,
sampling, Earth model, HEALPix identity vectors, end-to-end covered-cell counts,
discrepancies, and tolerances. Directed Link Analysis still needs an independent
external comparison before decision-grade promotion. That comparison is
explicitly deferred beyond v0.27.0 and is not represented as complete by this
experimental release. The v0.27.0 claim remains limited to deterministic,
programmatic engineering analysis under the non-claims below.

Real studies must also retain their own finer-cadence and next-order sensitivity
packets. Passing the synthetic acceptance suite does not establish convergence
for a different orbit, attitude history, FOV, RF profile, or horizon.

## Claim Boundary

The accepted envelope is sampled deterministic engineering analysis. It does
not establish atmospheric or weather availability, interference performance,
hardware calibration, packet delivery, operational communications assurance,
exact footprint polygons, probabilistic availability, full attitude slew
dynamics, time-varying power/thermal behavior, or orbit-design optimality.
