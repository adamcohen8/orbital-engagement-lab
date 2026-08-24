# Coverage and Link Budget Analysis

Status: Living design and implementation plan. Programmatic cores now exist,
while the separately named adapters and independent validation gates below
remain pending. Normative behavior is governed by the linked contracts rather
than this planning note.

## Purpose

Coverage and link-budget analysis are natural next capabilities for Orbital
Engagement Lab (OEL). They build directly on propagation, frames, attitude,
ground stations, sensors, multi-object execution, review stores, plotting, and
evidence provenance that OEL already has.

The goal should not be to add two isolated calculators. OEL should develop a
coherent visibility, coverage, and communications analysis family that supports
both fast trajectory analysis and causally coupled mission simulation.

The intended conceptual pipeline is:

```text
Propagated states + frames + attitude
                  |
                  v
       Observer/target geometry
                  |
                  v
 LOS + elevation + FOV + pointing + range
                  |
                  v
   Coverage constraints or RF constraints
                  |
                  v
 Intervals + margins + revisit + dwell + data volume
                  |
                  v
      Review tables + plots + evidence receipt
```

## Current OEL Baseline

OEL already provides several important ingredients:

- Deterministic ONP and OGP propagation.
- Frame transformations and absolute-epoch provenance.
- Sampled ground-station line-of-sight, elevation, and range calculations.
- Minimum-elevation and maximum-range access constraints.
- Ground-access histories, summaries, plots, review-store rows, and saved
  queries.
- Synthetic ground-station azimuth/elevation/range measurements.
- Sensor field-of-view, body-frame boresight, and object-tracking concepts.
- Attitude propagation and nadir/target-pointing machinery.
- Multi-object simulation and OGP batch propagation.
- Review-store, plotting, comparison, and evidence-provenance infrastructure.

The existing ground-station capability is passive geometric access. It does
not currently establish:

- RF link closure.
- Sensor footprints or regional coverage.
- Revisit, dwell, or coverage-gap metrics.
- Weather or atmospheric RF losses.
- Contact scheduling.
- Command, telemetry, or data-delivery success.
- Operational communications or sensor performance.

Existing access durations are inferred from sampled histories. Product-grade
coverage and link-window claims will require explicit event-time accuracy and
convergence semantics rather than silently inheriting the simulation output
timestep.

## Coverage Vocabulary

The word `coverage` is overloaded. OEL should distinguish at least the
following study types.

### Geometric access

Can observer or asset A see target B under declared geometric constraints?

Typical constraints include line of sight, occultation, minimum elevation,
maximum range, field of view, and pointing.

### Sensor coverage

Is a point, target, or area inside the sensor footprint while all declared
payload constraints are satisfied?

Potential constraints include field-of-view shape, boresight, pointing error,
off-nadir angle, incidence angle, illumination, range, payload duty cycle, and
slew or settling state.

### Communications coverage

For each represented location on Earth, can a declared communications service
close a link with sufficient margin under the applicable geometric, RF, and
operational constraints? A communications-coverage service must include a
declared notional or actual Earth-terminal profile; geometric visibility alone
is not communications coverage.

### Network or constellation coverage

What access, sensor coverage, or communications availability is produced by a
set of assets rather than a single asset?

OEL and its agents must not silently infer that geometric visibility means
sensor coverage, or that sensor/geometric coverage means communications
availability.

## Assets, Terminals, Links, and Coverage Services

The core ontology should distinguish assets, terminals or sensors, directed
links, and Earth coverage services.

### Assets and state providers

An asset is an entity capable of hosting a terminal or sensor. Its position,
velocity, attitude, and mounting-frame state may come from a state provider
such as:

- An ONP-propagated object.
- An OGP-propagated object or batch ephemeris.
- A fixed Earth site.
- A supplied ephemeris.
- A future moving ground, airborne, or other platform.

This abstraction does not require every fixed ground station to become an OEL
simulation `object`. It provides a common endpoint contract while allowing the
current fixed-ground-site model to remain distinct from propagated objects.

### Terminals and sensors

An asset may host multiple terminals or sensors with independent hardware,
mounting, pointing, and operational characteristics. Link and coverage
identities should therefore refer to the terminal or sensor, not only to its
parent asset.

### Directed links

A link is a directed terminal-to-terminal relationship:

```text
Transmitting asset/terminal -> Receiving asset/terminal
```

Supported relationship types should eventually include spacecraft-to-
spacecraft, spacecraft-to-ground, ground-to-spacecraft, and links involving
other explicitly supported state providers. Uplink and downlink between the
same assets remain different links because their terminals, frequencies,
powers, gains, receivers, rates, and thresholds may differ.

The canonical identity of a link should include at least:

```text
tx_asset_id
tx_terminal_id
rx_asset_id
rx_terminal_id
link_model_id
```

### Global Earth coverage services

Coverage is a property mapped over a declared representation of the complete
Earth surface. A coverage service binds:

- A source asset and sensor or communications terminal.
- A service definition and thresholds.
- An Earth-surface model and global tessellation.
- Time, frame, attitude, and fidelity contracts.

Examples include geometric visibility, sensor-FOV coverage, imaging coverage
below a maximum incidence angle, communications coverage above a link-margin
threshold, coverage by at least two assets, or service capable of delivering
a declared data volume.

Regions and point targets should normally be queries or aggregations over the
global product. Explicit region-only and point-only runs may exist as faster
partial studies, but must not make global coverage claims.

## Primary Architecture Decision: Runtime or Post-Processing

Coverage and link calculations should support both runtime evaluation and
post-processing. The choice should be governed by causality, not solely by the
selected propagator.

The governing rule is:

> If coverage or link state can change subsequent simulated behavior, evaluate
> it during runtime. If it only describes a completed trajectory, calculate it
> afterward.

### Runtime does not mean integrator-internal

Coverage and link budgets are algebraic functions of state, attitude, time,
and configuration. They should not be evaluated inside ONP numerical
integrator stages or treated as force-model calculations.

Runtime evaluation should occur at deterministic synchronization or event
boundaries:

```text
Propagate to synchronization boundary
                  |
                  v
Evaluate geometry, coverage, or link state
                  |
                  v
Publish a typed runtime event or observation
                  |
                  v
Mission logic or flight software reacts
                  |
                  v
Propagate the next interval
```

### Recommended defaults

| Study | Default evaluation posture |
| --- | --- |
| OGP catalog, access, coverage, or link study | Batched post-processing |
| ONP passive trajectory with analytical coverage outputs | Batched post-processing |
| ONP operations whose behavior depends on link state | Runtime |
| ONP payload operations whose behavior depends on coverage | Runtime |
| Qualification, regression, or debugging | Runtime plus post-run replay/parity |
| Future OGP-driven discrete mission operations | Runtime event layer consuming OGP ephemeris, not logic embedded in OGP |

OGP is naturally suited to rapid batched analysis over satellites, stations,
targets, time grids, frequencies, or terminal configurations. ONP must support
causal coupling, but ONP should still use post-processing by default when the
result is analytical only.

Using ONP should not automatically impose per-step coverage or link overhead.
Likewise, OGP should remain a passive propagation product even if a future
discrete-event mission layer consumes its ephemeris.

## One Authoritative Model, Two Execution Adapters

OEL should implement one authoritative deterministic calculation contract for
each domain:

```text
GeometryState + CoverageModel -> CoverageResult
GeometryState + LinkModel     -> LinkResult
```

Each model should have two adapters:

1. A scalar or incremental runtime adapter.
2. A vectorized, chunked post-processing adapter.

The two adapters must share:

- Equations and constants.
- Units and frame semantics.
- Constraint ordering and reason codes.
- Threshold behavior.
- Model and schema versions.
- Null, invalid, and unavailable-state handling.

A parity contract should verify that the runtime and batch adapters produce
equivalent results for the same saved states within declared tolerances. OEL
should not develop separate runtime and analytical link-budget implementations
that can silently disagree.

An optional audit mode should replay runtime calculations over the saved
trajectory, compare the results with recorded runtime events, and explain any
difference caused by cadence, interpolation, knowledge, or latency.

## Runtime Use Cases

Runtime evaluation is necessary when coverage or link results affect:

- Payload activation or image collection.
- Downlink start and stop.
- Data-recorder occupancy.
- Transmitter or payload power consumption.
- Antenna, gimbal, or spacecraft slews.
- Acquisition or loss-of-signal events.
- Command or telemetry delivery.
- Measurement availability.
- Mission-mode transitions.
- Termination after sufficient data delivery.
- Maneuvers intended to improve contact or coverage.

For example, a spacecraft could slew toward a station when predicted link
margin becomes adequate, transmit until onboard storage falls below a
threshold, and return to nadir pointing after loss of signal. A post-run
calculation cannot reproduce that history because link state changes the
subsequent simulation.

### Truth and onboard knowledge

The physical channel or footprint may be evaluated from simulator truth, but
flight software should not automatically receive omniscient truth.

Runtime OEL should distinguish:

- Physical availability calculated by the environment from truth.
- Predicted availability based on onboard plans or ephemerides.
- Observed availability delivered through typed events such as carrier lock,
  received-signal measurement, acknowledgement, or packet receipt.

Latency, uncertainty, cadence, and failure behavior should be explicit when
they are modeled. A v1 workflow may use ideal availability, but it must label
that assumption rather than presenting it as realistic communications
behavior.

## Post-Processing Use Cases

Post-processing should remain the default when the analyst asks questions such
as:

- What access windows occurred?
- What would the link margin have been?
- Which stations support the requested data rate?
- What percentage of a region was covered?
- What was the dwell or revisit time?
- What was the maximum coverage gap?
- How do several terminal or antenna assumptions compare?
- What does a constellation or catalog cover?

The post-processing path can vectorize or chunk over:

- Time.
- Spacecraft.
- Stations.
- Point targets or coverage cells.
- Frequencies.
- Terminal configurations.
- Alternative thresholds and data rates.

This is particularly valuable for OGP batch propagation, but completed ONP
state and attitude histories should use the same accelerated analysis path
whenever they contain the required inputs.

## Evaluation Cadence and Event Refinement

Runtime evaluation cadence should be independent of orbital integration
cadence. A small ONP force-integration step does not imply that a link budget
or footprint must be evaluated at the same rate.

The design should support:

- A declared coverage, payload, or link evaluation cadence.
- Detection of threshold crossings between evaluations.
- Deterministic acquisition/loss event-time refinement.
- Hard event boundaries when an event changes commands or modes.
- Recorded timing tolerance and refinement method.
- Convergence checks across analysis timesteps.

For causal runtime behavior, the contract must specify whether a threshold is
observed only at task boundaries or is treated as a continuous event requiring
refinement. Post-processing must not retroactively change runtime decisions.

Before product-grade coverage work, the existing access calculation should be
factored around a shared geometry and event-refinement layer that can be reused
by ground access, sensor footprints, links, eclipses, and later scheduling.

The geometry contract should record whether it uses a spherical or ellipsoidal
Earth, its frame model, absolute epoch, Earth-orientation inputs, and relevant
tolerances.

## Attitude and Terminal Pointing

Attitude is a first-class input to directional coverage and link calculations,
not an optional loss term added after geometric access has been declared.

The governing rule is:

> Every directional sensor or communications terminal must declare an attitude
> source. If OEL cannot establish that source, the directional calculation
> fails closed.

A quaternion appearing in a state history is not by itself sufficient
evidence of physical attitude. OEL currently permits an identity quaternion as
an initial/default state, including scenarios where attitude propagation may
be disabled. Coverage and link analysis must not silently interpret such a
placeholder as achieved pointing.

### Pose chain

For each spacecraft sensor or communications terminal, OEL should construct
the complete pose chain:

```text
Inertial frame
    |
    v  achieved spacecraft attitude
Body frame
    |
    v  fixed or gimbaled terminal mounting
Terminal or sensor frame
    |
    v
Boresight, FOV axes, and antenna-pattern coordinates
```

At each evaluation time, OEL should:

1. Compute the transmitter-to-receiver or observer-to-target line-of-sight
   vector.
2. Transform that vector into the transmitting or observing terminal frame.
3. For a link, transform the reverse vector into the receiving terminal frame.
4. Calculate off-boresight angle and pattern azimuth/elevation at the relevant
   endpoints.
5. Apply FOV, gimbal, pointing, or antenna-pattern constraints.
6. Preserve the intermediate geometry in review evidence.

For sensor coverage, attitude normally determines the physical footprint and
acts as a feasibility constraint. For a directional RF link, attitude should
normally affect gain continuously:

```text
Tx gain = G_tx(pattern azimuth, pattern elevation)
Rx gain = G_rx(pattern azimuth, pattern elevation)
```

A binary inside/outside antenna cone may be supported as a bounded model, but
must be labeled rather than presented as a general antenna pattern.

### Explicit attitude sources

Directional analysis should support explicitly declared attitude sources:

| Attitude source | Meaning |
| --- | --- |
| `propagated_truth` | Use achieved ONP attitude history, including tracking error and slew dynamics |
| `replay` | Use an explicitly supplied quaternion or attitude ephemeris |
| `analytic_assumption` | Generate an ideal nadir, velocity, target-track, Sun-pointing, or other declared law |
| `attitude_independent` | Valid only for a genuinely attitude-independent model such as an isotropic antenna |

Commanded attitude should not count as achieved physical attitude. It may be
used for planning or predicted-coverage analysis, but outputs must identify it
as commanded or predicted rather than actual.

An onboard controller may make decisions from estimated attitude or predicted
contact state. That belongs to the onboard decision side of the runtime
boundary; the physical channel or footprint should still be evaluated from
truth unless the study explicitly requests a different analytical product.

### ONP attitude behavior

For ONP studies:

- When attitude dynamics are enabled, physical coverage and link analysis
  should use the propagated truth quaternion.
- Slew lag, pointing error, actuator saturation, disturbances, and failed
  acquisition should affect physical results naturally.
- Post-processing should use the saved achieved quaternion history and match
  runtime geometry within declared tolerances.
- If attitude dynamics are disabled, a directional calculation must require an
  explicit analytic assumption or replay source.
- Commanded, estimated, and achieved attitudes should remain separately
  identifiable in evidence.

Using ONP does not automatically require runtime coverage evaluation. A
completed ONP trajectory with sufficient position and attitude history can be
analyzed efficiently in the batch path when coverage or link results do not
affect the simulation.

### OGP attitude behavior

OGP should not claim to produce physical attitude merely because a combined
state representation contains quaternion fields.

For OGP studies:

- Isotropic or otherwise attitude-independent links may be calculated without
  a spacecraft attitude model.
- Ideal nadir, velocity, target-track, or other analytic laws may be generated
  from the OGP trajectory when explicitly selected.
- Directional models may use a supplied attitude replay.
- Physical slew, tracking error, or actuator behavior is outside passive OGP
  propagation.
- If those physical behaviors matter, the selected case should be
  materialized into ONP or another attitude-capable mission simulation.

This permits fast idealized OGP screening while preventing ideal pointing
assumptions from leaking into claims about physically achieved coverage or
link closure.

### Terminal mounting

Spacecraft attitude and terminal orientation are distinct. A terminal model
should declare its mounting relative to the spacecraft body, including at
least:

- Position or phase-center offset in the body frame when material.
- Full mounting orientation.
- Terminal-axis and boresight convention.
- Fixed or gimbaled mounting behavior.
- Gimbal angle, rate, and travel limits when modeled.

A single `boresight_body` vector is sufficient for an axisymmetric conical FOV
or symmetric antenna pattern. It is not sufficient for:

- Rectangular or pushbroom sensors.
- Asymmetric antenna patterns.
- Polarization orientation.
- Roll-sensitive payloads.
- Gimbaled terminals.
- Multiple independently mounted terminals.

The longer-term contract should therefore use a full normalized mounting
orientation, conceptually similar to:

```yaml
terminal:
  mount:
    position_body_m: [0.0, 0.0, 0.0]
    quat_body_from_terminal: [1.0, 0.0, 0.0, 0.0]
  pattern:
    kind: axisymmetric
    boresight_axis: +z
```

This example is illustrative and does not establish final field names or axis
conventions.

Ground terminals need a corresponding Earth-fixed/local-frame mounting model,
such as:

- Fixed orientation in a declared local frame such as ENU.
- Ideal tracking mount.
- Azimuth/elevation gimbal with declared position and rate limits.
- Supplied pointing schedule or replay.

Both ends of a communications link must be respected.

### Roll and secondary-axis semantics

Aligning a single boresight vector leaves a roll degree of freedom. That is
irrelevant for an axisymmetric antenna or circular FOV, but material for
rectangular footprints, asymmetric gain patterns, pushbroom orientation, and
polarization.

Roll-sensitive models must therefore declare a secondary axis, clock angle,
full mounting/attitude reference, or another unambiguous orientation contract.
OEL must not silently choose arbitrary roll and then present the resulting
footprint or gain as physically determined.

### Runtime causal ordering

For a causally coupled ONP study, the intended ordering is:

```text
Propagate orbit and achieved attitude to time t
                    |
                    v
Evaluate physical footprint or link from truth
                    |
                    v
Publish acquisition, loss, or measurement event
                    |
                    v
Deliver the event at the permitted FSW boundary
                    |
                    v
FSW commands future attitude or transmission state
```

OEL should avoid a zero-time algebraic loop in which an availability event
changes attitude and that changed attitude retroactively changes the same
event. Commanding a slew toward a station does not establish a link; the link
becomes physically available only after achieved attitude and terminal
pointing satisfy the declared model.

### Post-processing and attitude interpolation

Post-run analysis must account for attitude motion when refining events
between stored samples. Its interpolation contract should include:

- Quaternion normalization.
- Sign equivalence because `q` and `-q` represent the same attitude.
- Shortest-path interpolation.
- A declared interpolation method.
- Appropriate use of angular-rate history when supported.
- Convergence evidence when event timing is part of the claim.

An access or link transition caused by a slew may be dominated by attitude
motion rather than orbital motion. Refining only the position history is not
sufficient.

### Attitude and terminal evidence

Each directional coverage or link sample should preserve enough evidence to
audit the calculation, including:

- Attitude source and model.
- Actual or assumed body attitude.
- Terminal mounting orientation and pointing state.
- Boresight in the relevant body, terminal, and inertial frames.
- Line-of-sight vector.
- Off-boresight angle.
- Pattern azimuth and elevation.
- Applied transmit or receive gain.
- FOV, pointing, and gimbal constraint results.
- Commanded-versus-achieved pointing error when available.
- Attitude validity and failure reason.
- Interpolation and event-refinement method.

Candidate reason codes include:

- `attitude_missing`.
- `attitude_invalid`.
- `outside_fov`.
- `tx_pointing`.
- `rx_pointing`.
- `gimbal_limit`.
- `slewing`.

### Attitude validation

Attitude-specific validation should include:

- Identity and known 90-degree quaternion transformations.
- Equivalent results for `q` and `-q`.
- Exact on-boresight and FOV-boundary cases.
- Roll-sensitive rectangular FOV and asymmetric pattern cases.
- Fixed, rotating, replayed, and gimbaled terminal fixtures.
- Commanded-versus-achieved slew-lag cases.
- Runtime-versus-batch parity on the same attitude history.
- Fail-closed cases for missing, placeholder, or invalid attitude provenance.

## Link Budget Capability

Link budgets should be generalized as directed terminal-to-terminal
calculations between assets. The first validation fixture may be a
spacecraft-to-ground downlink without narrowing the underlying model to ground
access.

### Recommended first flagship study

The first end-to-end capability should answer:

> Can this spacecraft downlink at the requested data rate to this ground
> network during the requested time horizon, and how much usable data can it
> deliver?

This question is narrow enough to validate, has immediate analyst value, and
uses the current ground-access and review infrastructure while establishing
most of the shared substrate needed for later coverage and scheduling.

### Initial inputs

A link model should declare:

- Transmitting asset and terminal identities.
- Receiving asset and terminal identities.
- Link direction.
- Carrier frequency.
- Transmit power.
- Transmit and receive losses.
- Transmit and receive antenna gains.
- Receiver system noise temperature.
- Data rate.
- Required `Eb/N0` or another explicitly named threshold.
- Miscellaneous fixed losses.
- Elevation and range constraints.
- Optional boresight and pointing constraints.
- Calculation model, constants, units, and version.

Longer-term terminal models may include antenna gain patterns, polarization,
gimbals, pointing loss, modulation and coding, bandwidth, atmospheric loss,
rain, interference, hardware availability, and other explicitly supported
effects.

### Core deterministic equations

For a simple v1 budget expressed in decibels:

```text
Received power = EIRP + receive gain - free-space path loss - other losses
C/N0           = received power - Boltzmann term - system-noise-temperature term
Eb/N0          = C/N0 - data-rate term
Link margin    = calculated Eb/N0 - required Eb/N0
```

Every term should be preserved in the evidence output. OEL should not emit
only a final margin that cannot be audited.

### Link outputs

The analysis should produce time-indexed evidence such as:

- Line of sight.
- Range, elevation, and azimuth.
- Transmit and receive pointing errors.
- Applied antenna gains.
- Free-space path loss.
- Other applied losses.
- Received power.
- `C/N0`.
- `Eb/N0`.
- Link margin.
- Usable-link boolean.
- Explicit reason when the link is unusable.

Window-level evidence should include:

- Acquisition and loss-of-signal times.
- Window duration.
- Minimum, maximum, and representative margin.
- Minimum range and maximum elevation.
- Usable-contact duration.
- Assumed or achieved data rate.
- Estimated delivered data volume.
- Constraints responsible for clipping the window.

Candidate review-store tables include:

- `link_samples`.
- `link_windows`.
- `link_summary`.

Rows should be keyed by transmitting asset/terminal, receiving
asset/terminal, link model, and time or window identity. Ground-relative
azimuth and elevation should be present only when meaningful for the selected
endpoint type; the generic link contract should use endpoint-frame geometry
and pointing angles.

### Initial claim boundary

The v1 claim should be deliberately narrow:

> Deterministic one-way satellite-to-ground RF feasibility using free-space
> path loss and explicitly declared fixed gains, losses, noise temperature,
> data rate, and threshold assumptions.

Unless separately implemented and validated, v1 should explicitly exclude:

- Weather and atmospheric variability.
- Interference and contested spectrum.
- Spectrum coordination.
- Hardware calibration and failure rates.
- Network scheduling or resource conflicts.
- Command and telemetry protocol success.
- Operational communications assurance.

## Global Earth Coverage Capability

The canonical coverage domain should be the complete Earth surface rather than
an arbitrary region selected before analysis. A region of interest should
normally be a query or aggregation over the same global evidence used for all
other regions.

This supports reusable questions such as:

- Global covered-area fraction.
- Worst-covered locations.
- Maximum and percentile revisit or gap by geography.
- Arctic, equatorial, national, or other region summaries without changing
  the underlying physical study.
- Comparison of geographic service inequality across designs.

### Earth-surface representation

The global coverage contract should declare:

- Earth shape and datum.
- Tessellation or cell-system identifier and version.
- Resolution or hierarchy level.
- Cell identity, representative location, and physical area.
- Treatment of coastlines, poles, and antimeridian boundaries.
- Area-weighting and aggregation rules.

A regular latitude/longitude grid should not be the foundational global
representation unless its unequal cell areas are explicitly corrected. The
design should evaluate hierarchical or equal-area alternatives before a cell
system is selected.

The Earth should be the canonical semantic domain without requiring wasteful
brute-force evaluation. Implementations may use footprint-first
intersection/rasterization, hierarchical cells, chunking, sparse access
intervals, multi-resolution products, or other deterministic accelerations.

Coverage provenance must distinguish:

- `global_earth`: every cell in the declared global tessellation was handled
  under the study contract.
- `region`: only a declared subset was evaluated; no global claim is allowed.
- `points`: only declared point targets were evaluated; this is access
  evidence rather than a global coverage product.

### Phase 1: Global sensor-footprint coverage

Generate the instantaneous footprint from achieved or explicitly assumed
attitude and intersect or rasterize it onto the global Earth tessellation.
Initial products should support a bounded sensor geometry such as a conical
FOV and report:

- Per-cell access intervals.
- Accumulated dwell.
- Revisit and gap metrics.
- Covered-area fraction.
- Footprint and cell-intersection provenance.

Point-target fixtures remain useful for validation and fast queries, but they
are not the product backbone.

Implementation status as of 2026-08-19: the additive programmatic Phase 1
kernel is implemented in `sim.analysis`. It includes dependency-free HEALPix
NESTED centers on the WGS84 authalic sphere, WGS84 surface visibility, achieved
or explicitly assumed attitude plus sensor mounting, vectorized/chunked
conical coverage, sampled sparse intervals, dwell/revisit/censoring summaries,
resource preflight, deterministic JSON/CSV evidence, and NPZ intervals with a
canonical semantic content hash. Focused
reference, attitude, interval, chunk-parity, fail-closed, and artifact tests are
checked in alongside it.

The programmatic API accepts explicit state and attitude evidence. The
canonical conical product is also implemented through Scenario YAML,
completed ONP/review and ECI OGP history normalization, review tables, native
plots, saved queries, and the supported `coverage_link_review` agent recipe.
Rich footprints, communications coverage, aggregation, tasking, and concrete
causal consumers remain separate adapter scopes.

### Phase 2: Regional and point queries

Aggregate the global coverage product over arbitrary declared regions or point
sets. Region metrics should use physical cell area or another explicit
weighting method. Naive counts of latitude/longitude samples should not be
presented as area coverage.

Implementation status as of 2026-08-19: the additive programmatic Phase 2
kernel is implemented in `sim.analysis.coverage_queries`. Regions are required
to be non-empty, versioned, provenance-bound, strictly sorted sets of canonical
cells at the source product's order. Their masks receive semantic hashes, and
their dwell, revisit, gap, censoring, covered-cell, equal-area fraction, and
physical-area results are derived directly from the Phase 1 sparse intervals
without rerunning propagation or footprint geometry. Point queries normalize a
WGS84 zero-height longitude/latitude, use the canonical HEALPix NESTED
point-to-cell rule, and report the containing cell's resolution-dependent
sample state and metrics without claiming exact subcell visibility.

The query evidence surface writes a source-bound manifest, complete region
definitions and point mappings, regional sample rows, and point-cell sample
rows. Query identity is deterministic regardless of caller ordering. Focused
tests cover all-cell regional equivalence, known subset metrics, point-cell
interval inheritance, antimeridian/polar mapping, validation, semantic binding,
and deterministic artifacts.

### Phase 3: Rich footprint and service constraints

The Phase 3 slice adds and inspects:

- Rectangular, pushbroom, or other validated FOV shapes.
- Ground-track and footprint overlays.
- Field-of-view boundary intersections with the Earth.
- Off-nadir and incidence-angle constraints.
- Illumination and other explicitly validated service constraints.
- Footprint provenance and geometry failures.

This layer must handle antimeridian crossings, poles, partial Earth
intersections, invalid pointing, and the declared Earth-shape model.

Implementation status as of 2026-08-20: the additive programmatic Phase 3
kernel is implemented in `sim.analysis.rich_coverage` under the separate
`oel.rich-earth-coverage-analysis.v0.1` contract. It supports axisymmetric hard
cones, rectangular hard FOVs, and semantically distinct pushbroom hard FOVs in
the mounted sensor frame. Optional inclusive gates cover maximum target
off-nadir angle, maximum incidence angle, and minimum/maximum target Sun
elevation. Illumination requires explicit, provider-identified Sun ECI evidence
at every analysis epoch.

Every epoch retains ordered sampled FOV-boundary-ray intersections with WGS84,
including complete, partial, and no-intersection dispositions, plus the WGS84
subsatellite track and achieved boresight off-nadir angle. A review plot overlays
selected sampled boundaries on the ground track with antimeridian splitting.
The boundary is expressly review geometry: HEALPix center membership remains
authoritative for coverage metrics and physical area.

The rich product preserves the Phase 2 structural query surface, so region and
point queries consume its sparse intervals without rerunning propagation or
geometry. Deterministic JSON/CSV/NPZ artifacts bind the scientific config,
input state/attitude/Sun evidence, reason ledger, intervals, ground track, and
sampled boundaries. Scenario YAML, completed-run extraction, review-store and
agent-tool adapters, and independent matched-assumption validation remain
follow-on work.

### Phase 4: Global communications coverage

Map a declared communications service across the global Earth tessellation.
Every evaluated Earth cell must bind a declared notional or actual ground
terminal profile, including the receiver or transmitter assumptions necessary
to compute link closure. Without that terminal profile, the product is
geometric access rather than communications coverage.

Implementation status as of 2026-08-20: the additive programmatic Phase 4
kernel is implemented in `sim.analysis.communications_coverage` under
`oel.global-communications-coverage.v0.1`. Each WGS84/HEALPix cell binds the
same named, provenance-bearing Earth-terminal profile. Both downlink and
uplink direction are explicit, and RF service qualification uses the single
Directed Link Analysis free-space ledger after physical line of sight,
elevation, range, source-pattern, and Earth-terminal-pattern gates. The result
preserves sparse intervals and the Phase 2 region/point query surface.

### Phase 5: Constellation aggregation

Combine multiple spacecraft and report:

- Mean and percentile revisit.
- Maximum coverage gap.
- Dwell distribution.
- Coverage multiplicity.
- Number of assets simultaneously available.
- Geographic cells that fail declared service criteria.

Implementation status as of 2026-08-20: the additive programmatic Phase 5
kernel is implemented in `sim.analysis.coverage_aggregation` under
`oel.constellation-coverage-aggregation.v0.2`. It combines two or more
content-bound global products with identical epochs, HEALPix order, canonical
cells, domain disposition, and explicit service-definition identity without
rerunning propagation or geometry. It retains exact
sample/cell multiplicity, union or required-multiplicity intervals, dwell,
finite complete-revisit statistics and percentiles, active-asset counts, and
never-qualified cells.

### Phase 6: Tasking and optimization

Only after the underlying evidence is trustworthy should OEL add:

- Slew and settling constraints.
- Payload duty cycles.
- Storage, power, and downlink coupling.
- Contact scheduling.
- Payload tasking.
- Constellation or orbit-design optimization.

Implementation status as of 2026-08-20: the first bounded programmatic Phase 6
kernel is implemented in `sim.analysis.coverage_tasking` under
`oel.coverage-tasking.v0.2`. It exactly selects from at most 24 source-bound,
asset-bound single-asset observation/downlink opportunities with non-overlap,
direct angular-rate slew plus settling, payload duty, storage, and horizon-
energy constraints. It does not create opportunities or substitute for attitude
dynamics. Multi-asset scheduling, time-varying power/thermal models,
constellation design, and orbit-design optimization remain separate contracts,
not implied capabilities of this bounded solver.

The reproducible internal acceptance scope and the still-open external
scientific gate are recorded in
[`Coverage and Directed-Link Programmatic Acceptance`](docs/validation-coverage-link-programmatic.md).

Candidate review-store or derived-product tables include:

- `earth_coverage_cells`.
- `coverage_intervals`.
- `coverage_summary`.
- `coverage_regions`.
- `coverage_service_definitions`.

High-volume global cell/time evidence may require a content-addressed derived
artifact rather than placing every dense sample directly in the primary
review database. The review store should retain queryable summaries,
identities, provenance, and artifact references.

## Configuration Direction

The exact schema remains an open design question, but model definition and
evaluation authority should be separate.

Named assets, terminals, sensors, coverage services, and directed links should
be defined once and then referenced by either an analytical study or an
authorized runtime monitor. Conceptually:

```yaml
links:
  leo_s_band_downlink:
    transmitter: satellite_a.s_band_tx
    receiver: colorado_springs.s_band_rx

analysis:
  links:
    evaluation: postprocess
    model: leo_s_band_downlink
  coverage:
    domain: global_earth
    service: satellite_a.imager_visibility
    tessellation: <to-be-selected>

runtime:
  link_monitors:
    - link: leo_s_band_downlink
      publishes: link_availability
      cadence_s: 1.0
```

The example is illustrative, not a proposed final schema.

An implicit `auto` mode is undesirable when a calculation can affect
execution. The scenario should explicitly distinguish evidence-only analysis
from runtime authority to publish events that mission logic or flight software
may consume.

## Agent-Native Study Behavior

Coverage and link analysis should become agent tools only after the underlying
YAML/API/review workflow has a stable model, canonical fixture, validation
envelope, and evidence packet.

An agent should clarify the details that materially change the study, such as:

- Coverage type: geometric, sensor, communications, or network.
- Assets, terminals, sensors, and link direction.
- Global Earth service definition or an explicitly partial region/point
  domain.
- Time horizon and epoch.
- Propagator and fidelity.
- Attitude and pointing assumptions.
- Sensor field of view and constraints.
- Frequency, direction, data rate, and terminal parameters.
- Runtime-causal behavior versus evidence-only analysis.
- Required success metric.

The durable study record should bind assumptions, frames, units, epochs,
fidelity, model versions, evaluation mode, tolerances, acceptance criteria,
claims, non-claims, and evidence citations. The agent may orchestrate and
explain the study, while deterministic OEL calculations remain authoritative.

## Validation Strategy

Software tests alone should not be treated as scientific validation. A useful
validation progression includes:

### Geometry and events

- Analytically tractable line-of-sight and elevation cases.
- Spherical and ellipsoidal Earth fixtures where supported.
- Frame and epoch checks.
- Threshold-crossing and event-time convergence tests.
- Antimeridian, pole, grazing, and no-intersection cases.
- Runtime-versus-batch parity on identical histories.

### Link budget

- Hand-calculated reference budgets with every term recorded.
- Range-doubling and frequency-scaling checks.
- Gain, loss, noise-temperature, and data-rate monotonicity.
- Exact threshold-crossing cases.
- Pointing-pattern test cases when patterns are introduced.
- Independent external comparison under matched assumptions before broader
  validation claims.

### Coverage

- Point-target cases with known access intervals.
- Conical-footprint geometry cases.
- Complete global tessellation identity and area-accounting checks.
- Global covered-area and area-weighting checks.
- Equivalence between regional queries and aggregation of the same global
  cells.
- Explicit rejection of global claims from region-only or point-only studies.
- Revisit, dwell, and gap calculations on synthetic schedules.
- Constellation aggregation cases with exactly known unions and overlaps.
- Independent external comparison under matched frames, geometry, sampling,
  and constraint assumptions.

Validation evidence should state exactly which models and envelopes it
supports. It should not promote a simple free-space budget into operational RF
assurance or a geometric footprint into calibrated sensor performance.

## Public and Pro Product Boundary

The transparent deterministic kernels are public core, but availability differs
by integration layer. The canonical conical/directed-link product has the
public Scenario/review/agent path; several richer families remain public
programmatic kernels only. A separate governed communications workflow is an
implemented private Pro, export-excluded v0.2 surface.

This promotion does not change the maturity or claim boundary: the current
stack remains experimental engineering analysis pending independent external
validation.

### Public core

- Transparent geometric access.
- Global Earth coverage for conical and implemented rich hard-field-of-view
  geometries, plus point and regional queries over those products.
- Simple, inspectable free-space link budgets.
- Programmatic communications coverage, constellation aggregation, bounded
  tasking, cadence sensitivity, and a standalone authorized single-link causal
  monitor protocol. The monitor has no concrete ONP/FSW consumer integration.
- ONP, completed-review, and ECI OGP history adapters.
- Documented equations, assumptions, examples, review tables, plots, tests,
  and bounded acceptance evidence.

### Implemented private Pro v0.2 workflow

The governed Pro communications workflow adds named ITU atmospheric models,
RF-qualified HEALPix coverage, constellation aggregation and sensitivity,
bounded downlink scheduling/adaptive-rate delivery, seeded campaigns,
ground-network/constellation trades, and governed equipment/customer adapters.
It is local, separately entitled, experimental, and excluded from public
export. Its declared profiles are illustrative; it does not claim measured RF,
current weather/interference, calibrated probability, packet assurance, or
operational availability.

### Still future or outside the current product

- Imported or calibrated antenna patterns.
- Current-weather assimilation and calibrated atmospheric availability.
- Measured interference and operational RF environments.
- Operational multi-terminal networks, routing, and availability assurance.
- Unbounded multi-asset scheduling and resource-conflict optimization.
- High-fidelity time-varying storage, power, thermal, payload, and downlink
  coupling.
- Constellation design and orbit optimization at operational scale.
- Operationally calibrated uncertainty campaigns.
- Distributed customer-catalog services beyond the governed local adapters.

Public kernels are not silently withheld, but the governed Pro workflow is an
intentional private product surface because it composes proprietary profiles,
campaign/scheduling automation, separate entitlement, and a private evidence
contract. Packaging does not change the scientific non-claims above.

| Family | Public kernel | Public Scenario/review | Public agent recipe | Private Pro workflow | External validation |
| --- | --- | --- | --- | --- | --- |
| Canonical conical coverage/direct link | yes | yes | yes | reusable input | coverage reference retained; directed-link comparison pending |
| Rich footprints | yes | no | no | reusable input | internal fixtures only |
| Communications coverage/aggregation/tasking | yes | no | no | yes, governed v0.2 | named seam comparisons only |
| Runtime causal monitor | standalone protocol | no concrete consumer | no | no concrete consumer | protocol fixtures only |

## Pre-Implementation Freeze Gates

Production implementation was gated on freezing two bounded contracts:

1. `Directed Link Analysis v0.1`.
2. `Global Earth Coverage Analysis v0.2`.

Each contract should define supported inputs, typed schemas, mathematical
semantics, time/frame/attitude behavior, outputs, failures, claim and non-claim
boundaries, acceptance fixtures, and performance/resource envelopes.

### Proposed Directed Link Analysis v0.1 envelope

The recommended initial link envelope is:

- Directed terminal-to-terminal evaluation.
- At least one spacecraft-to-spacecraft fixture and one spacecraft-to-fixed-
  Earth-site fixture to prove the endpoint abstraction is not ground-specific.
- Free-space path loss.
- Fixed carrier frequency and fixed data rate.
- Scalar gain with an attitude-independent declaration, or a bounded
  axisymmetric gain/pointing model.
- Achieved or explicitly assumed attitude with terminal mounting.
- Same-epoch instantaneous geometry.
- A complete, typed link-term ledger and deterministic link margin.
- Scalar runtime and vectorized/chunked batch parity.

The v0.1 link contract should exclude unless separately implemented and
validated:

- Light-time correction.
- Atmospheric, rain, and weather losses.
- Terrain and refraction.
- Interference and spectrum coordination.
- General imported antenna patterns.
- Polarization.
- Adaptive modulation or coding.
- Terminal contention, scheduling, and network protocols.
- Power, storage, and packet-delivery coupling.
- Operational communications assurance.

Range rate may be preserved as geometry evidence, but Doppler should not be
inferred until signal frequency, sign, reference epoch, and transmit/receive-
time semantics are frozen.

### Global Earth Coverage Analysis v0.2 envelope

The recommended initial coverage envelope is:

- Complete Earth-surface coverage under a declared global tessellation.
- One source spacecraft in the first physical fixture.
- One bounded conical sensor FOV.
- Achieved ONP attitude or an explicit OGP analytic/replay assumption.
- Deterministic footprint-to-cell evaluation.
- Sparse per-cell intervals where practical.
- Per-cell dwell, revisit, and gap evidence.
- Area-weighted global and regional summaries.
- Explicit global, region-only, and point-only domain dispositions.
- Resolution and time-step convergence evidence.

The v0.1 coverage contract should exclude unless separately implemented and
validated:

- Terrain masking.
- Clouds and weather.
- General rectangular, pushbroom, or imported sensor patterns.
- Payload tasking, slew scheduling, storage, and power coupling.
- Communications coverage without an explicit Earth-terminal profile.
- Population or economic weighting.
- Constellation optimization.
- Steady-state or repeating-coverage inference beyond the simulated horizon.

### Gate 1: Earth model and tessellation

Before coverage implementation, freeze:

- Spherical or ellipsoidal Earth.
- Datum and surface-altitude semantics.
- Canonical global cell system and version.
- Stable cell identity.
- Hierarchy or resolution semantics.
- Cell representative point and physical area.
- Polar and antimeridian handling.
- Footprint-to-cell intersection policy.
- Area aggregation rules.

The footprint-to-cell policy must say whether coverage means that the cell
center is inside the footprint, any portion intersects, a minimum fraction
intersects, or an overlap fraction is accumulated. Center sampling is fast but
resolution-sensitive; polygon overlap is more defensible for area claims but
more expensive.

A focused pre-implementation benchmark should compare viable cell systems for
deterministic identity, global completeness, area distortion, hierarchical
refinement, polygon operations, dependency and licensing burden, batch speed,
and portable artifact representation.

Coverage resolution must be part of every claim. Validation should measure how
key metrics change under at least one declared refinement.

### Gate 2: Time, interval, and censoring semantics

Freeze exact definitions for:

- State and attitude evaluation time.
- Position and quaternion interpolation.
- Instantaneous versus swept footprints.
- Acquisition and loss times.
- Link-margin threshold crossings.
- Runtime task-boundary versus continuous-event semantics.
- Coverage-interval merge tolerances.
- Dwell, revisit, and gap calculations.
- Study-start and study-end censoring.

If a study starts during a gap or coverage interval, OEL does not know when
that interval began. If it continues past the study end, its duration is also
censored. Such intervals must be labeled rather than reported as complete
maximum-gap or dwell observations.

A fast-moving narrow footprint may cross cells between stored samples. The
contract must state whether OEL evaluates independent instantaneous
footprints, interpolated footprints, or the swept footprint over an interval.

### Gate 3: Frame, epoch, occultation, and signal-time semantics

Freeze:

- Absolute-epoch requirements.
- Inertial and Earth-fixed frame paths.
- Earth-orientation provenance.
- Same-time versus transmit/receive-time geometry.
- Earth occultation model.
- Range-rate semantics.
- Supported central body.
- Treatment of terrain, refraction, and atmospheric bending.

The recommended v0.1 posture is Earth-only, same-epoch instantaneous geometry,
no light time, no terrain, and no refractive or atmospheric bending. Those
limits should be explicit in every receipt.

### Gate 4: Type-safe RF quantities

Freeze canonical types and units for:

- Watts, dBW, and dBm.
- Hz and allowed human-facing frequency units.
- dBi and linear gain.
- Linear and decibel losses.
- Kelvin and system noise temperature.
- `C/N0`, `Eb/N0`, SNR, and margin.
- Bits per second.
- Pattern and pointing angles.

The configuration should make it difficult to confuse dBW with dBm or apply a
loss with the wrong sign. Generic untyped dictionaries should not be the
authoritative RF contract.

Every link result must preserve the complete term ledger rather than only a
final margin.

### Gate 5: Terminal frame and pattern conventions

Freeze:

- Terminal-frame and axis conventions.
- Mounting-quaternion direction.
- Boresight axis.
- Pattern azimuth/elevation convention.
- Pattern interpolation and out-of-domain behavior.
- Fixed versus gimbaled state.
- Ground-terminal local frame.
- Supported v0.1 gain-pattern family.

The recommended v0.1 pattern family is attitude-independent scalar gain plus a
bounded axisymmetric directional option. Full mounting orientation should
still be preserved so later 2D patterns, roll, and polarization do not require
a schema replacement.

### Gate 6: Coverage metric definitions

Freeze:

- Whether revisit is measured from the end of one interval to the start of the
  next.
- Whether short interruptions are merged and at what tolerance.
- Continuous versus accumulated dwell.
- Per-cell, regional, and global gap definitions.
- Physical visibility versus service-qualified multiplicity.
- Union and overlap behavior across assets.
- Area weighting for regional and global metrics.
- Treatment of censored first and final intervals.
- Study-horizon provenance and prohibition on unsupported periodic or steady-
  state inference.

### Gate 7: Storage and scalability

A dense `assets x times x Earth cells` product will become too large quickly.
Before implementation, freeze the authoritative storage posture among:

- Dense samples.
- Sparse access intervals.
- Footprint polygons plus derived cells.
- Chunked cell artifacts.
- A deterministic hybrid.

The recommended posture is sparse per-cell intervals plus footprint artifacts
and aggregated summaries. The primary review database should retain queryable
identities, summaries, provenance, and artifact references rather than every
dense global sample when that would be excessive.

Also freeze:

- Chunk identity and ordering.
- Content hashes.
- Memory and runtime budgets.
- Resource estimation.
- Cancellation and incomplete-artifact behavior.
- Deterministic results across worker counts and chunk sizes.

### Gate 8: Runtime authority and event ordering

For causally coupled studies, freeze:

- Which runtime component computes physical availability.
- Typed event identities and payloads.
- Which mission or FSW consumers may receive them.
- Delivery timing and task-boundary rules.
- Truth, belief, predicted, and observed state boundaries.
- Same-time event ordering.
- Prevention of zero-time algebraic loops.
- Runtime behavior when terminal state is unavailable or invalid.

The recommended v0.1 runtime event should expose physical availability and
margin evidence only. Power, storage, scheduling, adaptive rate, and packet
delivery remain future consumers. Global Earth coverage remains postprocessed;
runtime monitors are limited to specific links, targets, or active regions.

### Gate 9: Backward compatibility

The new geometry layer overlaps existing `ground_access`. Before replacing any
path, freeze whether existing behavior becomes a compatibility view, remains a
legacy path temporarily, or follows a documented deprecation window.

The recommended posture is one authoritative geometry implementation with
compatibility outputs. Existing scenario behavior, table names, saved queries,
reason codes, ordering, and artifact names should remain stable until an
explicit migration is approved and documented.

### Gate 10: Validation and acceptance

Before implementation, define acceptance fixtures and tolerances for:

- Scalar runtime versus vectorized batch parity.
- Identical results across chunk sizes and worker counts.
- Existing ground-access output parity.
- Known quaternion, terminal mounting, and pointing cases.
- Hand-calculated link budgets with every term recorded.
- Approximately 6.02 dB additional free-space loss when range doubles.
- Frequency, gain, loss, noise-temperature, and data-rate monotonicity.
- Exact threshold and no-link cases.
- Global cell identity and Earth-area accounting.
- Known footprint/cell intersections.
- Regional-query equivalence with aggregation of the same global cells.
- Revisit, gap, merge-tolerance, and censoring cases.
- Time-step and coverage-resolution refinement.
- At least one independent matched-assumption comparison for each product.

Scientific tolerances and performance budgets should be recorded separately.
Passing software tests alone does not establish the broader claim envelope.

### Gate 11: Artifact and interoperability contract

Before high-volume outputs are created, freeze enough of the artifact contract
to support:

- Stable JSON summaries.
- Review-store identities and tables.
- Content-addressed large coverage products.
- Geographic coordinate-reference metadata.
- Coverage raster or vector export without redefining cell identity.
- Optional columnar storage for large products.
- Plot projections and antimeridian behavior.
- Deterministic manifests and provenance hashes.

The final external format need not be selected before the geometry kernel, but
the internal cell identity and manifest must allow later GIS-friendly export.

### Gate 12: Data governance and product boundary

Freeze handling for:

- User-provided antenna patterns, terminal catalogs, and ground-site data.
- Trusted file paths and safe validation.
- Source hashes and licensing/provenance metadata.
- Customer or sensitive inputs.
- Public release fixtures versus future proprietary/customer fixtures, models,
  and generated artifacts.
- Claim restrictions for synthetic, assumed, or uncalibrated equipment data.

The implemented transparent geometry, bounded free-space calculations, and
follow-on deterministic analysis cores are public. Future proprietary patterns,
managed environment services, customer catalogs, and operational-scale
workflow automation may belong in Pro after their value and validation burden
are understood.

### Explicitly deferred beyond v0.1

Unless a contract is deliberately expanded before freeze, defer:

- Atmospheric, rain, cloud, and weather effects.
- Terrain masking and refraction.
- General antenna-pattern files and polarization.
- Interference and contested spectrum.
- Adaptive modulation and coding.
- Network scheduling, half-duplex, and beam contention.
- Power, storage, packetization, and protocol coupling.
- Population or economic weighting.
- Constellation optimization.
- Uncertainty campaigns.
- Imported customer equipment catalogs.
- Agent or MCP exposure.

## Contract Freeze Decision

The original v0.1 contract freeze was completed on 2026-08-19 as design-only
work. The mixed checkout was then captured as an audited checkpoint before
production implementation began. Additive programmatic implementations now
exist for the frozen link and coverage contracts; adapter and validation
status remains explicit below. The 2026-08-20 implementation audit revised
aggregation, sensitivity, tasking, and runtime-monitor contracts to v0.2 where
their original shapes could not enforce the stated evidence boundary.

The frozen artifacts are:

- [`Directed Link Analysis v0.1`](docs/contracts/directed-link-analysis-contract.md)
- [`Global Earth Coverage Analysis v0.2`](docs/contracts/global-earth-coverage-analysis-contract.md)
- [`Rich Earth Coverage Analysis v0.1`](docs/contracts/rich-earth-coverage-analysis-contract.md)
- [`Global Communications Coverage v0.1`](docs/contracts/global-communications-coverage-contract.md)
- [`Constellation Coverage Aggregation v0.2`](docs/contracts/constellation-coverage-aggregation-contract.md)
- [`Coverage Tasking v0.2`](docs/contracts/coverage-tasking-contract.md)
- [`Directed Link Runtime Monitor v0.2`](docs/contracts/directed-link-runtime-monitor-contract.md)
- [`Coverage Sensitivity Evidence v0.2`](docs/contracts/coverage-sensitivity-evidence-contract.md)

The link contract freezes a directed terminal-to-terminal, same-epoch,
one-way free-space calculation. It defines the endpoint abstraction, achieved
and assumed attitude behavior, mounting direction, constant and axisymmetric
hard-cone gains, WGS84 occultation, typed SI/dB inputs, the complete RF term
ledger, threshold behavior, reason ordering, interval/refinement semantics,
artifacts, validation fixtures, and explicit non-claims.

The coverage contract freezes:

- WGS84 ellipsoid geometry for physical line of sight;
- HEALPix NESTED cell identity on the WGS84 authalic sphere for equal-area
  global indexing and aggregation;
- the stable identity
  `("healpix_nest_wgs84_authalic_v1", order, nested_pixel_index)`;
- an explicit order for every analysis, with order 6/order 7 as the first
  baseline/refinement fixture;
- a single body-mounted axisymmetric conical sensor;
- achieved ONP attitude or explicitly assumed OGP attitude;
- authoritative center-of-cell inclusion rather than polygon overlap;
- independent instantaneous samples with left-closed, right-open interval
  accumulation and no swept-footprint inference;
- exact sampled dwell, complete-revisit, boundary-censoring, global fraction,
  region-mask, and point-cell definitions;
- sparse per-cell interval artifacts with semantic content hashes; and
- mandatory cadence and next-order sensitivity evidence.

The tessellation decision compared H3, S2, and HEALPix against the first
product's needs. HEALPix was selected because exact equal-area aggregation,
global completeness, hierarchy, deterministic integer identity, and polar
coverage matter more for this product than conventional GIS polygon
operations. Cell identity is independent of any Python dependency; adding a
runtime package is a later governed implementation choice.

This freeze deliberately does not choose public/Pro packaging, add an agent or
MCP surface, install a dependency, or claim production readiness.

## Proposed Development Sequence

1. **Complete:** freeze `Directed Link Analysis v0.1` with supported schemas,
   equations, semantics, failures, claims, non-claims, fixtures, and resource
   envelope.
2. **Complete:** freeze `Global Earth Coverage Analysis v0.1`, then revise it
   additively to v0.2 for provider-identified transition refinement. Its Earth
   model, tessellation, cell semantics, time/interval definitions, metrics,
   artifacts, claims, non-claims, fixtures, and resource envelope.
3. **Complete for the v0.1 envelope:** resolve all pre-implementation gates
   that would change core identities, equations, storage, or scientific
   semantics.
4. **Complete for Phase 1 surface targets:** define the shared observer/target
   geometry contract. Directed-link occultation and its complete reason
   taxonomy remain part of the link slice.
5. **Complete in the Phase 1 normalized Python record:** define source asset,
   state provider, attitude provider, sensor, and global coverage identities.
   Scenario-YAML placement remains an adapter decision.
6. **Complete for Phase 1 surface targets:** implement the shared vectorized
   geometry evaluator without changing existing `ground_access` behavior.
7. **Complete for canonical coverage and directed links:** add deterministic
   event-window detection and optional arbitrary-epoch provider refinement with
   convergence, iteration-limit, sample-bounded, and censoring evidence.
   Scenario and completed-history adapters now bind the scalar kernels to
   arbitrary-epoch callbacks. Retained histories use declared cubic Hermite
   position/velocity interpolation and shortest-arc attitude SLERP; callers may
   instead supply a propagator-backed callback. Evidence retains the source,
   bracket, tolerance, iteration count, and disposition.
8. **Programmatic core complete:** deliver the generalized directed
   terminal-to-terminal free-space link kernel with spacecraft-to-spacecraft
   and spacecraft-to-fixed-site fixtures.
9. **Complete for the primary run-review adapter:** add typed
   link-term samples, intervals, summary, margin plot, and evidence packet.
   Scenario runs populate `link_summary`, `link_samples`, `link_windows`, and
   `link_transitions`. The supported aliases are `coverage_summary`,
   `coverage_transition_summary`, `directed_link_summary`, and
   `directed_link_windows`.
10. **Complete for conical scenario/completed-history adapters:** deliver global Earth
    conical-sensor coverage using achieved or explicitly assumed attitude and
    the frozen cell/intersection contract. Scenario runs populate
    `coverage_summary`, `coverage_samples`, `coverage_intervals`, and
    `coverage_transitions`.
11. **Programmatic core complete; adapters pending:** add region and point
    queries over the same global product, including revisit, dwell,
    covered-area, gap, and censoring metrics.
12. **Programmatic core complete; independent validation pending:** add rich
    hard-FOV footprints, explicit service constraints, sampled WGS84 boundary
    intersections, and ground-track/footprint review overlays.
13. **Evidence adapter partially complete:** chunk-size scientific parity is
    covered for the programmatic products, and
    `oel.coverage-sensitivity-evidence.v0.2` produces source-bound cadence or
    next-order comparison packets only after validating the source products,
    retaining the required epochs, and proving normalized non-refinement
    assumptions match, against caller-declared acceptance limits.
    Directed-link scalar runtime and vectorized batch samples use the same
    evaluator and have exact fixture parity. Each real study must still run and
    retain its sensitivity comparisons. Worker-count and concrete ONP-runtime
    integration evidence remain pending for their applicable adapters.
14. **Pending external evidence:** complete independent matched-assumption
    validation for the coverage and directed-link products. Internal
    hand-ledger, geometry, threshold, and aggregation fixtures do not replace
    this gate.
15. **Programmatic core complete; adapters and external validation pending:**
    add global communications coverage only with an explicitly declared Earth
    terminal profile.
16. **Programmatic core complete:** add deterministic constellation coverage
    aggregation with uniform domain/service identity, union, required
    multiplicity, revisit, overlap, active-asset, and failure evidence. Routed
    network aggregation remains separate.
17. **Bounded causal adapter complete; ONP consumer integration pending:** add
    authorized runtime monitors for one specific link and named consumer. An
    evaluation after committed state becomes deliverable only at the next task
    boundary, preventing a zero-time loop. Global-grid runtime evaluation
    remains prohibited.
18. **First bounded tasking core complete; broader models pending:** select
    source-bound, explicitly asset-bound single-asset opportunities under
    slew/settling, duty, storage, horizon-energy, and downlink constraints. The
    completed-run adapter must still dereference and verify each source window.
    Multi-asset scheduling,
    time-varying power/thermal behavior, advanced environments, constellation
    design, and orbit optimization remain separate follow-on contracts.
19. Expose stable workflows through agent-facing tools only after the
    deterministic contracts and evidence packets are proven.

This remains a working development sequence rather than a release commitment.
The programmatic directed-link, global sensor/region/point/rich footprint,
global communications, constellation aggregation, and bounded tasking cores
are complete. Canonical conical coverage and directed spacecraft-to-spacecraft
or spacecraft-to-fixed-site links (ground-to-ground is excluded)
also have evidence-only scenario adapters, normalized ONP/review/ECI-OGP
history adapters, primary review tables, and provider-backed transition
refinement. Rich footprints, communications coverage, aggregation, tasking,
and causal ONP consumers still need their own integration adapters. Release
readiness still requires cadence/order and runtime parity evidence and
independent external validation. Public packaging is resolved; agent exposure
remains gated behind the evidence appropriate to each workflow.

## Resolved v0.1 Design Questions

- The first link product is directed and one-way. Uplink and downlink are two
  separate directed configurations.
- The required endpoint fixtures are spacecraft-to-spacecraft and
  spacecraft-to-fixed-WGS84-site.
- v0.1 accepts explicit attitude-independent scalar gain and a simple
  axisymmetric hard-cone gain.
- Directional terminals use achieved ONP attitude, replay, or an explicitly
  idealized analytic law. OGP has no implicit physical attitude.
- OEL's scalar-first inertial-to-body `q_bn` convention is retained. Terminal
  mounting maps terminal coordinates into the parent body or ENU frame, and
  terminal/sensor boresight is `+Z`.
- The fixed-site local frame is right-handed East-North-Up.
- Shortest-arc SLERP is used for replay attitude. Link transition refinement
  uses an identified arbitrary-epoch evaluator or remains sample bounded;
  retained-history Hermite/SLERP evaluation is labeled as interpolation rather
  than exact propagation.
- Runtime v0.2 exposes physical link availability and margin only for a
  specifically authorized monitored link. Global coverage is postprocessed.
- Link v0.1 is same-epoch and coverage v0.2 retains independently sampled
  states plus optional bracketed transition refinement; neither
  claims light-time or swept-footprint behavior.
- Physical Earth geometry is WGS84. Global indexing is HEALPix NESTED on the
  WGS84 authalic sphere.
- No grid-order default is inferred. Order 6 and order 7 are the first
  baseline/refinement pair, and every result records order and cadence.
- Sparse cell intervals are content-addressed large artifacts; review storage
  retains queryable identities, summaries, provenance, and references.
- Coverage metrics and censoring rules are frozen in the coverage contract.
- Independent matched-assumption comparisons are required before either
  product is accepted.

## Questions Deliberately Deferred Beyond v0.1

- Whether terminals eventually move to a reusable governed equipment catalog.
- Gimbaled and roll-dependent patterns, ground-terminal tracking modes, and
  imported pattern interpolation.
- A truth/belief/observed onboard link-state model beyond physical availability.
- Reusable governed Earth-terminal catalogs and tracking modes beyond the
  explicit public terminal profiles.
- Time-varying data production, packetization, coding, storage, power, thermal,
  and delivery behavior beyond the bounded public tasking model.
- Multi-asset scheduling, constellation design, and orbit optimization beyond
  the public deterministic aggregation and single-asset tasking cores.
- Exact polygon-overlap or swept-footprint coverage.
- OGP-driven standalone discrete-event mission operations.
- Productized catalogs, managed services, and large-scale workflow packaging.

## Frozen v0.1 Working Recommendations

The contract freeze establishes the following v0.1 decisions:

1. Treat coverage and link budgets as one related product family with a shared
   geometry and event substrate.
2. Govern runtime versus post-processing by causality rather than propagator
   alone.
3. Keep coverage and link calculations outside numerical propagator internals.
4. Default OGP to batched post-processing.
5. Default analytical ONP studies to post-processing as well.
6. Permit runtime ONP evaluation when the result changes simulated behavior.
7. Use one authoritative model with scalar runtime and vectorized batch
   adapters.
8. Require runtime/batch parity and preserve both model and evaluation
   provenance.
9. Begin with usable satellite-to-ground contact and a bounded free-space link
   budget as the first validation fixture, while keeping the calculation model
   generalized to directed terminal-to-terminal links.
10. Build the deterministic YAML/API/review workflow before exposing a new
    agent tool surface.
11. Require an explicit attitude source for every directional model and fail
    closed when attitude provenance is missing or ambiguous.
12. Use achieved attitude, rather than commanded attitude, for physical
    availability.
13. Treat terminal mounting as distinct from spacecraft attitude and require
    full orientation for roll-sensitive models.
14. Give OGP no implied physical attitude; require an explicit ideal law,
    replay, or attitude-independent model.
15. Treat links as directed terminal-to-terminal relationships between assets,
    not merely undifferentiated object pairs.
16. Treat the complete Earth surface as the canonical coverage domain using a
    declared global tessellation and physical area weighting.
17. Treat regions and point targets as queries or explicitly partial studies,
    and prohibit global claims from partial domains.
18. Require a declared Earth-terminal profile before labeling a global surface
    product as communications coverage.
19. Keep global Earth coverage out of the simulation loop; runtime evaluation
    should monitor only the specific links, targets, or active regions that can
    affect simulated behavior.
20. Freeze separate `Directed Link Analysis v0.1` and `Global Earth Coverage
    Analysis v0.1` contracts before production implementation.
21. Treat Earth/tessellation, time/censoring, RF units, terminal frames,
    metrics, storage, runtime authority, compatibility, validation, artifacts,
    and data governance as implementation gates rather than incidental coding
    details.
22. Keep v0.1 deliberately narrow and defer weather, terrain, general patterns,
    interference, scheduling, resource coupling, optimization, uncertainty,
    customer catalogs, and agent exposure.
23. Use WGS84 ellipsoid geometry for physical visibility and HEALPix NESTED on
    the WGS84 authalic sphere for canonical equal-area global indexing.
24. Define v0.1 coverage as explicit-cadence, center-of-cell sampled coverage,
    not polygon-overlap or swept-footprint coverage.
25. Require every global study to declare HEALPix order and cadence, and retain
    both next-order and finer-cadence sensitivity evidence.
