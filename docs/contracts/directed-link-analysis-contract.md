# Directed Link Analysis Contract

Status: **frozen for Directed Link Analysis v0.1**.

Implementation status as of 2026-08-20: the additive programmatic kernel,
sample/window artifacts, evidence packet, and margin review plot are
implemented in `sim.analysis.directed_link`. A separate authorized causal
task-boundary monitor adapter is implemented. Directed spacecraft-to-spacecraft
and spacecraft-to-fixed-site links now also have evidence-only Scenario YAML,
completed ONP/review and ECI OGP
history adapters, primary review-store tables, and provider-identified
transition refinement. Concrete ONP-consumer integration and an independent
external matched-assumption comparison remain pending. A behavior change that
contradicts this document requires an explicit contract revision; additive
fields must remain backward compatible.

## Product Boundary

Directed Link Analysis v0.1 evaluates a one-way RF link from one transmitting
terminal to one receiving terminal. A link identity is:

```text
(transmitting asset, transmitting terminal,
 receiving asset, receiving terminal, link configuration)
```

Reversing the endpoints creates a different link. Assets provide time-indexed
state; terminals provide mounting, pointing, gain, and RF properties. A fixed
Earth site is an asset type, not a special meaning attached to every link.

The same deterministic evaluator supports:

- scalar evaluation for an authorized runtime monitor; and
- vectorized or chunked evaluation for post-processing.

The two adapters must preserve the same equations, thresholds, reason order,
and term ledger. The link calculation remains outside orbit-integrator
internals.

v0.1 is an engineering free-space calculation. It is not operational
communications assurance and does not establish spectrum compatibility,
availability in weather, packet delivery, or mission success.

## Supported v0.1 Envelope

The frozen envelope supports:

- Earth-orbiting spacecraft assets with OEL ECI position and velocity;
- fixed WGS84 Earth-site assets with an absolute epoch and OEL's governed
  Earth-fixed-to-ECI path;
- spacecraft-to-spacecraft and spacecraft-to-fixed-site links;
- same-epoch, instantaneous, one-way geometry;
- WGS84 ellipsoid occultation and a fixed-site elevation mask;
- an optional maximum slant range;
- fixed carrier frequency and information data rate;
- attitude-independent constant gain; or
- an axisymmetric hard-cone terminal with constant in-cone gain;
- achieved attitude, explicit replay attitude, or an explicit analytic ideal
  attitude law;
- fixed terminal mounting relative to the asset body or local site frame;
- free-space path loss and explicit nonnegative fixed losses;
- receiver system-noise temperature, required `Eb/N0`, and link margin;
- retained-sample post-processing and bounded transition refinement when the
  state and attitude providers can evaluate the requested epoch; and
- scalar/vectorized parity with deterministic ordering.

The two mandatory acceptance fixtures are one spacecraft-to-spacecraft link
and one spacecraft-to-fixed-site link. At least one fixture must exercise a
directional terminal and non-identity mounting.

## Normalized Study Record v0.1

Scenario YAML, Python, and future agent adapters normalize into this closed
logical record before execution. This is the contract shape, not authorization
for arbitrary provider imports:

```yaml
contract_version: oel.directed-link-analysis.v0.1
analysis_id: unique-study-id
epoch:
  time_system: utc
  jd_utc: 2460000.5
evaluation:
  mode: postprocess              # postprocess | runtime_monitor
  times_s: [0.0, 10.0, 20.0]
  transition_refinement:
    enabled: true
    transition_time_tolerance_s: 0.001
    max_iterations: 64
assets:
  - asset_id: spacecraft_a
    state_provider:
      kind: scenario_object
      object_id: spacecraft_a
    attitude_source:
      kind: achieved             # achieved | replay | analytic_ideal | not_required
      provider_id: spacecraft_a.attitude_truth
  - asset_id: site_a
    state_provider:
      kind: fixed_wgs84_site
      longitude_deg: -104.526
      geodetic_latitude_deg: 38.803
      ellipsoidal_height_km: 1.9
terminals:
  - terminal_id: spacecraft_a.tx
    asset_id: spacecraft_a
    parent_frame: body
    quat_parent_from_terminal: [1.0, 0.0, 0.0, 0.0]
    pattern:
      kind: axisymmetric_hard_cone
      gain_dbi: 6.0
      half_angle_deg: 45.0
  - terminal_id: site_a.rx
    asset_id: site_a
    parent_frame: enu
    quat_parent_from_terminal: [1.0, 0.0, 0.0, 0.0]
    pattern:
      kind: constant
      gain_dbi: 12.0
links:
  - link_id: spacecraft_a_to_site_a
    tx_terminal_id: spacecraft_a.tx
    rx_terminal_id: site_a.rx
    carrier_frequency_hz: 2200000000.0
    tx_power_w: 10.0
    data_rate_bps: 1000000.0
    system_noise_temperature_k: 500.0
    required_eb_n0_db: 9.6
    tx_line_loss_db: 0.0
    rx_line_loss_db: 0.0
    misc_loss_db: 0.0
    geometry_constraints:
      min_fixed_site_elevation_deg: 10.0
      max_range_km: 2500.0
```

Post-processing requires explicit `times_s`; convenience adapters may expand a
start/stop/cadence request before validation. The sequence is strictly
increasing and its first and final values define the evaluated horizon.
`runtime_monitor` instead requires a positive `task_period_s`, omits
`times_s`, disables transition refinement, contains exactly one directed link,
and names a separately authorized causal consumer. Unknown fields fail
validation. A transport adapter may not silently add a loss, attitude law,
terminal, or RF assumption.

## Identities and Typed Inputs

Every study has a stable `analysis_id`. Every asset, terminal, and link has a
non-empty, study-unique string ID. Results preserve all five directed-link
identity components; display labels are not identities.

### Asset state

At every evaluated epoch, an endpoint state contains:

- `time_s`, measured from the study epoch;
- an absolute study epoch and declared time system;
- `position_eci_km`, a finite three-vector;
- `velocity_eci_km_s`, a finite three-vector when range rate is retained; and
- the state-provider identity and provenance.

Both endpoint states are evaluated at the same epoch. v0.1 does not apply
light-time correction and does not mix transmit and receive epochs.

A fixed Earth site is declared with WGS84 geodetic longitude, latitude, and
ellipsoidal height. It uses OEL's governed Earth rotation/frame path at each
epoch. A study without the epoch/frame information needed for that transform
fails validation.

### Attitude and mounting

OEL's achieved spacecraft attitude is the scalar-first Hamilton quaternion
`q_bn`. Its DCM `C_bn` maps ECI vectors into body axes:

```text
v_body = C_bn v_eci
```

A terminal mounting is the scalar-first unit quaternion
`quat_body_from_terminal`; its DCM `C_bt` maps terminal-frame vectors into
body-frame vectors. A terminal-frame vector is therefore mapped to ECI by:

```text
v_eci = C_bn^T C_bt v_terminal
```

The terminal boresight is `+Z` in terminal coordinates. The full mounting
quaternion is required even though the v0.1 directional pattern is
axisymmetric. Quaternions must be finite, normalized within the frozen
validation tolerance, and accompanied by source provenance.

The fixed-site local frame is right-handed East-North-Up:

```text
+X = east, +Y = north, +Z = ellipsoid normal/up
```

Its terminal mounting uses `quat_enu_from_terminal`, with the same
terminal-to-parent direction as the spacecraft mounting.

An attitude-independent terminal declares that fact explicitly and does not
require asset attitude. A directional terminal fails closed when achieved or
explicitly assumed attitude is unavailable. Commanded attitude alone is not
physical availability evidence.

### Terminal pattern

Exactly one v0.1 pattern is selected:

1. `constant`: gain is a finite scalar `gain_dbi` in all directions and the
   terminal is explicitly attitude independent.
2. `axisymmetric_hard_cone`: gain is `gain_dbi` when the peer direction is at
   or inside `half_angle_deg` from `+Z`, and the terminal is unavailable
   outside it. The half-angle must satisfy `0 < half_angle_deg <= 180`.

At the transmitter, peer direction points from transmitter to receiver. At
the receiver, peer direction points from receiver to transmitter. Boundary
membership is inclusive within the documented angular numerical tolerance.

No interpolation, sidelobe, polarization, roll-dependent, azimuth/elevation,
or imported pattern semantics are implied by the hard cone.

### RF quantities

The authoritative configuration uses names with units; untyped numeric maps
are not accepted. Required values are:

- `carrier_frequency_hz > 0`;
- `tx_power_w > 0`;
- transmitting and receiving `gain_dbi` values;
- `system_noise_temperature_k > 0`;
- `data_rate_bps > 0`; and
- finite `required_eb_n0_db`.

Optional fixed losses are independently named and finite, nonnegative dB
quantities:

- `tx_line_loss_db`;
- `rx_line_loss_db`; and
- `misc_loss_db`.

All optional losses default to exactly `0 dB`. OEL does not infer dBm, linear
gain, bandwidth, coding gain, implementation loss, or atmospheric loss from a
generic value.

## Geometry and Constraint Semantics

Range is the Euclidean distance between same-epoch ECI endpoint positions.
Zero or non-finite range is invalid.

Earth occultation uses the closed WGS84 reference ellipsoid. A
spacecraft-to-spacecraft line is geometrically clear only when the open line
segment between endpoints does not intersect the ellipsoid. Endpoint contact
with an explicitly fixed surface site is permitted. A spacecraft state inside
the ellipsoid is invalid rather than visible.

For a fixed site, elevation is measured in its WGS84 East-North-Up frame.
Elevation equal to the configured minimum passes. An optional maximum range is
also inclusive. v0.1 does not include terrain, buildings, atmospheric
refraction, or an apparent-horizon correction.

Range rate may be retained as the same-epoch geometric derivative

```text
range_rate_km_s = dot(v_rx - v_tx, r_rx - r_tx) / range_km
```

but it is evidence only. v0.1 does not infer Doppler because transmit/receive
time, sign, and signal-reference semantics are not part of this contract.

## Authoritative Link Equations

The constants are exact SI definitions:

```text
c = 299792458 m/s
k = 1.380649e-23 J/K
```

For same-epoch range `R_m` and carrier frequency `f_hz`:

```text
free_space_path_loss_db = 20 log10(4 pi R_m f_hz / c)
tx_power_dbw            = 10 log10(tx_power_w)
eirp_dbw                = tx_power_dbw + tx_gain_dbi - tx_line_loss_db
received_power_dbw      = eirp_dbw + rx_gain_dbi
                          - free_space_path_loss_db
                          - rx_line_loss_db - misc_loss_db
noise_density_dbw_hz    = 10 log10(k system_noise_temperature_k)
cn0_db_hz               = received_power_dbw - noise_density_dbw_hz
eb_n0_db                = cn0_db_hz - 10 log10(data_rate_bps)
margin_db                = eb_n0_db - required_eb_n0_db
```

Every named term is preserved in the result. A term cannot be omitted because
its value is zero. RF-qualified availability requires all geometry and
pointing gates to pass and `margin_db >= 0`.

This is the only authoritative v0.1 equation path. Convenience summaries may
not recompute or round intermediate terms.

## Time and Execution Semantics

Post-processing is the default for OGP and analytical ONP studies. Evaluations
occur at the explicitly declared analysis epochs. The evaluator does not
silently substitute the propagator step, output cadence, or wall-clock time.

An availability interval is left-closed and right-open, `[start, end)`, except
that a point result at the final study epoch is still retained. Adjacent
passing samples alone do not prove a continuous interval unless the state and
attitude provider contract supports evaluation between them.

Transition refinement is optional and declared. When enabled, the evaluator
brackets a change in the controlling signed constraint and re-evaluates the
authoritative state, attitude, geometry, and link equations until both the
declared time tolerance and iteration limit are satisfied. If a provider
cannot evaluate the requested epoch, the output remains `sample_bounded`; it
must not present an interpolated crossing as exact-provider evidence.

Quaternion replay interpolation, when used, is shortest-arc SLERP after sign
continuity normalization. Position interpolation is never silently invented;
it belongs to the declared state provider.

A runtime monitor is allowed only when a user-authorized simulated consumer
depends on this specific link. It evaluates at a declared runtime task
boundary after the required achieved state is committed. Its result may affect
only later task boundaries, preventing a zero-time algebraic loop. A refined
post-processing event cannot retroactively change a simulation.

### Frozen numerical behavior

The normalized v0.1 configuration records these numerical tolerances:

- quaternion norm validation: `1e-10` absolute;
- angular boundary comparison: `1e-12 rad` absolute;
- range boundary comparison: `1e-9 km` absolute; and
- RF-ledger comparison: `1e-10 dB` absolute.

An inclusive angular or range constraint passes when it is no more than its
limit plus the corresponding tolerance. A tangent line to the WGS84 ellipsoid
is occulted. Scalar/vectorized classification and primary reason must be exact;
finite range and velocity terms must agree with `rtol=1e-12` and
`atol=1e-12` in their declared units, and RF terms must meet the absolute dB
tolerance.

When transition refinement is enabled, the configuration must declare a
positive `transition_time_tolerance_s` and a positive finite iteration limit.
There is no hidden event-time default.

## Result and Reason Contract

Every sample preserves the inputs, geometry gates, pattern gates, complete RF
ledger, final availability, and provenance needed to reproduce it.

Each gate is recorded independently. `primary_reason` is the first applicable
item in this frozen order:

1. `invalid_input`
2. `state_unavailable`
3. `attitude_unavailable`
4. `earth_occulted`
5. `below_elevation_mask`
6. `beyond_max_range`
7. `tx_outside_pattern`
8. `rx_outside_pattern`
9. `negative_margin`
10. `available`

At exactly zero margin the reason is `available`. A result may retain multiple
failed-gate booleans, but the primary reason remains deterministic.

The stable logical products are:

- `link_analysis_manifest.json`: version, identities, normalized
  configuration, epoch/frame/Earth model, evaluator version, input hashes,
  execution mode, cadence/refinement, exclusions, and artifact hashes;
- `link_samples`: one row per directed link and evaluation epoch, including
  term ledger and gate evidence;
- `link_intervals`: start, end, duration, start/end censoring, transition
  disposition, and controlling acquisition/loss reasons; and
- `link_summary`: evaluated horizon, available duration/fraction, interval
  count, margin extrema at evaluated epochs, censoring, and claim limits.

The review database may store these tables directly when bounded. A larger
sample artifact may be content addressed, with identities and summaries left
queryable in the review store. Artifact row order is link identity then time,
using stable UTF-8 byte ordering for IDs.

Partial or cancelled runs receive an incomplete disposition and may not
produce an ordinary complete-study summary.

## Validation and Failure Behavior

Configuration validation fails closed for ambiguous units, duplicate IDs,
missing epochs, unsupported frames/providers, non-finite values, invalid
quaternions, directional patterns without physical/assumed attitude,
nonpositive required quantities, negative loss values, or unsupported pattern
types.

Acceptance requires, at minimum:

- hand-calculated golden ledgers with every term compared;
- approximately `6.0206 dB` additional path loss when range doubles;
- expected frequency, power, gain, loss, temperature, and data-rate
  monotonicity;
- exact zero-margin boundary behavior;
- WGS84 clear, tangent, occulted, and invalid-inside-Earth geometry cases;
- known achieved-attitude and non-identity-mounting cases;
- fixed-site elevation-mask boundary cases;
- spacecraft-to-spacecraft and spacecraft-to-fixed-site fixtures;
- scalar/vectorized parity at the contract tolerance;
- parity across chunk sizes and supported worker counts;
- event-refinement convergence and sample-bounded fallback tests; and
- one independent matched-assumption comparison whose reference, inputs,
  tolerances, and discrepancies are retained.

Scientific tolerances and performance budgets are separate acceptance
records. Passing software tests alone does not make the model operationally
authoritative.

## Explicit Non-Claims and Deferred Work

v0.1 does not model or claim:

- light time, relativistic signal effects, Doppler, or coherent turnaround;
- atmosphere, ionosphere, rain, cloud, scintillation, weather, or refraction;
- terrain, buildings, or local horizon masks beyond fixed minimum elevation;
- polarization, general antenna patterns, sidelobes, gimbals, or scan loss;
- interference, spectrum coordination, adjacent-channel effects, or jamming;
- bandwidth-dependent receiver details beyond the declared system
  temperature and information data rate;
- adaptive coding/modulation, availability statistics, or calibrated hardware;
- half-duplex rules, contention, routing, scheduling, latency, or protocols;
- packet delivery, data volume, storage, power, thermal, or resource coupling;
- uncertain-state probability or Monte Carlo availability; or
- operational communications assurance.

Agent-facing execution tools and public/Pro packaging are not frozen by this
scientific contract. They may be added only after the deterministic evaluator,
evidence products, and acceptance fixtures satisfy this boundary.
