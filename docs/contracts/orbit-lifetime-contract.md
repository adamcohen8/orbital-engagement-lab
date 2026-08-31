# Orbit Lifetime Analysis Contract

This contract defines OEL's bounded public deterministic orbit-decay and
lifetime workflow. It propagates one supplied ECI Cartesian state through the
authoritative OEL Numerical Propagator (ONP), retains declared spacecraft and
atmosphere inputs, refines altitude-threshold crossings, and produces
replayable evidence. Machine-readable records are defined by
[`oel-orbit-lifetime-v1.schema.json`](schemas/oel-orbit-lifetime-v1.schema.json).

## Input contracts

`oel.orbit_lifetime_problem.v1` declares one asset, an absolute Julian UTC
epoch, one ECI Cartesian state, a horizon no longer than 90 days, integration
and output cadence, event-refinement controls, mass, drag area and coefficient,
optional J2, one atmosphere record, and warning, disposal, and reentry
altitudes. At most 500,000 integration steps and 200,000 output samples are
accepted. Unknown, missing, non-finite, unbounded, Earth-intersecting, or
internally inconsistent inputs fail validation.

The epoch must lie between Julian UTC `1721425.5` and `5373393.5`, leaving the
full 90-day horizon representable by the supported UTC conversion runtime.

The public atmosphere contract deliberately requires explicit inputs:

- `constant` retains one positive density;
- `exponential` retains reference density, reference altitude, scale height,
  and ceiling altitude;
- `ussa1976` has no weather input;
- `nrlmsise00` retains F10.7, 81-day F10.7, daily Ap, and seven Ap values; and
- `harris_priester` selects and retains one supported coefficient-table F10.7:
  65, 75, 100, 125, 150, 175, 200, 225, 250, or 275.

The normalized result also retains the effective atmosphere altitude domain.
Constant density is valid from the Earth surface upward; exponential density is
bounded by its declared ceiling; USSA1976 and NRLMSISE-00 are bounded at 1000
km; and Harris-Priester is strictly bounded to 110-2000 km. Initial osculating
apogee and the declared reentry threshold must remain inside the selected
model's domain.

The workflow never silently retrieves current or historical space weather.
Those scalar inputs are modeling assumptions bound into the normalized problem
and its semantic digest.

`oel.orbit_lifetime_comparison_problem.v1` supplies one complete base problem
and two through eight atmosphere cases. Every case replaces only the
atmosphere record. All state, spacecraft, horizon, force-model, cadence, and
threshold inputs remain identical. The base problem's placeholder atmosphere is
not an effective case input and is excluded from comparison semantic identity.

## Deterministic model and outcomes

The analyzer uses ONP's existing RK4 propagator and existing drag and optional
J2 acceleration plugins. `drag_enabled: false` exists for conservation and
control studies; it does not change the retained spacecraft or atmosphere
record. Thresholds are crossings of instantaneous geocentric altitude,
defined as ECI radius minus OEL's Earth equatorial radius. They are not
perigee-altitude, mean-element, legal-compliance, or surviving-debris tests.

Each crossing is bracketed by authoritative ONP states and refined by repeated
ONP propagation until the declared time tolerance is met. When integration-step
endpoints are both above a threshold, the analyzer also brackets an interior
perigee from the radial-velocity sign change and tests that minimum for a hidden
downward crossing. Analysts must still demonstrate integration-step convergence
for a consequential case.

A successful analysis has `status: completed` and exactly one outcome:

- `reentry_threshold_reached` means the declared reentry altitude was crossed
  and `stop_at_reentry` stopped the propagation at the refined event; or
- `atmosphere_domain_limit_reached` means a non-stopping run reached the lower
  validity limit of its selected atmosphere model;
- `earth_surface_reached` means a non-stopping run reached the Earth surface;
  or
- `horizon_complete` means propagation reached the declared horizon.

An unreached threshold is reported with `reached: false` and null timing. It is
not extrapolated. Invalid input, propagation failure, or failed replay returns
an error and does not create completed scientific evidence.

## Evidence and replay

A single-case output directory contains exactly:

```text
normalized_problem.json
orbit_lifetime_summary.json
orbit_lifetime_timeseries.csv
orbit_lifetime_events.csv
orbit_lifetime_manifest.json
```

The timeseries retains ECI state, elapsed time, altitude, density, osculating
semi-major axis, eccentricity, inclination, perigee and apogee altitude,
specific energy, angular momentum, drag acceleration, and semi-major-axis
change. The event table retains threshold kind, refined time and state,
original bracket, iteration count, and disposition. Summary extrema are over
the retained output samples. The result also exposes drag-work versus Kepler
energy closure when J2 is disabled.

A comparison output directory contains exactly:

```text
normalized_comparison.json
orbit_lifetime_comparison_summary.json
orbit_lifetime_comparison.csv
orbit_lifetime_comparison_manifest.json
```

The comparison summary retains each complete single-case evidence summary and
states that non-atmosphere inputs are identical. Model spread is sensitivity
evidence, not a probability distribution or calibrated uncertainty.

Both writers require an absent destination and atomically promote a sibling
temporary directory. Authoritative replay rejects symbolic links, unexpected
or oversized artifacts, malformed or noncanonical JSON, receipt drift, unknown
manifest fields, and forged derived tables. It reconstructs the normalized
input, reruns the authoritative analysis, and byte-compares every derived JSON
and CSV artifact.
The summary and manifest also bind the lifetime implementation source files and
selected atmosphere coefficient assets. Replay fails closed when that provider
identity differs from the retained evidence.

## Validation and claim limits

Public acceptance includes two-body conservation with drag disabled, a
constant-density short-arc drag-energy limit, event ordering and refinement,
integration-step convergence, horizon-complete negative cases, every supported
atmosphere adapter, identical-input comparisons, JSON Schema validation,
deterministic replay, tamper rejection, resource bounds, and CLI failures.
OEL's retained fixed-state atmosphere/drag diagnostics compare NRLMSISE-00 and
Harris-Priester against checksum-bound Orekit 13.1.7 reference evidence
generated by checked-in Java source. Those fixed-state component comparisons do
not independently validate a propagated lifetime trajectory,
threshold-crossing time, or end-to-end lifetime workflow. No STK, ODTK, paid
service, live external runtime, or network request is required for routine
replay.

This v1 workflow is deterministic engineering evidence for one declared case.
It does not predict space weather, calibrate density or ballistic coefficient,
model uncertainty, establish disposal compliance, estimate reentry casualty
risk, maintain orbit custody, qualify software or hardware, or authorize an
operation.
