# Global Communications Coverage Contract

Status: **frozen and implemented for the Phase 4 programmatic core v0.1**.

Contract identifier: `oel.global-communications-coverage.v0.1`.

## Product Boundary

Global Communications Coverage evaluates a declared one-way RF service from
one spacecraft terminal to, or from, every canonical HEALPix Earth cell. Every
cell is bound to the same explicit `EarthTerminalProfile`, including pattern,
gain, elevation mask, provenance, direction, noise, rate, and threshold
assumptions. A result without that profile is geometric or sensor coverage,
not communications coverage.

The evaluator consumes deterministic state and, for a directional spacecraft
terminal, achieved, replay, or explicitly analytic-ideal attitude evidence
after propagation. A constant-gain source terminal declares
`attitude_source_kind: not_required`, omits the attitude provider and attitude
history, and remains attitude independent. It cannot alter a simulation. It
uses the same WGS84, HEALPix NESTED, center-of-cell, sampled time, sparse
interval, dwell, revisit, gap, and censoring semantics as Global Earth Coverage
v0.1.

## Supported v0.1 Record

The normalized programmatic configuration declares:

- analysis, service, source-asset, state-provider, and source-terminal
  identities, plus an attitude-provider identity when attitude is required;
- direction: `spacecraft_to_earth` or `earth_to_spacecraft`;
- source-terminal constant or axisymmetric hard-cone gain and full mounting;
- a named and provenance-bound Earth-terminal constant or zenith-centered
  axisymmetric hard-cone profile;
- minimum Earth-terminal elevation and optional maximum range;
- fixed frequency, transmit power, information rate, system-noise
  temperature, required `Eb/N0`, and independently named fixed losses;
- HEALPix order, explicit epochs, resource limits, and execution chunk size.

Unknown units or fields belong in adapters and must fail before this record is
constructed. v0.1 uses radians, kilometres, watts, hertz, kelvin, bits per
second, dBi, and dB.

## Geometry and RF Semantics

Each Earth cell represents a fixed zero-height WGS84 terminal whose `+Z` axis
is the ellipsoid outward normal. Physical access requires convex-ellipsoid
line of sight, inclusive minimum elevation, optional inclusive range, and both
terminal-pattern gates.

RF terms are evaluated only through the authoritative Directed Link Analysis
free-space ledger. Gain direction switches with link direction; configuration
does not silently reinterpret transmit and receive quantities. Service
qualification requires every geometry and pattern gate plus margin greater
than or equal to zero within the frozen `1e-10 dB` comparison tolerance.

Primary disposition order is:

1. `earth_blocked`;
2. `below_elevation_mask`;
3. `beyond_max_range`;
4. `source_outside_pattern`;
5. `earth_terminal_outside_pattern`;
6. `negative_margin`;
7. `available`.

All reason counts are retained even though the sparse interval product stores
only final service qualification.

## Products and Identity

Stable artifacts are:

- `communications_coverage_manifest.json`;
- `communications_coverage_summary.json`;
- `communications_coverage_samples.csv`;
- `communications_coverage_cells.csv`; and
- `communications_coverage_intervals.npz`.

The manifest binds normalized assumptions, input evidence, frame provenance,
resource estimates, semantic identity, and artifact hashes. Execution chunk
size remains provenance but is excluded from scientific identity. RF terms are
normalized to their declared `1e-10 dB` comparison envelope before semantic
hashing so sub-tolerance vector-shape roundoff cannot create a false scientific
difference.

The result preserves the structural Phase 2 query surface. Region and point
queries therefore consume RF-qualified intervals without rerunning geometry or
the link ledger.

## Acceptance and Non-Claims

Acceptance includes RF-closing and deliberately non-closing services,
direction and pattern fixtures, all-cell query equivalence, chunk parity,
deterministic artifacts, resource rejection, and constellation aggregation
compatibility. Decision-grade use still requires cadence/order sensitivity and
an independent matched-assumption external comparison.

v0.1 does not model terrain, atmosphere, rain, weather, interference,
polarization, protocols, hardware reliability, contention, scheduling,
probability, or actual terminal deployment. A notional profile is an explicit
engineering assumption, not a claim that matching stations exist everywhere.
