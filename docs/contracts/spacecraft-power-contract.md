# Spacecraft Power Analysis Contract

This contract defines OEL's bounded public deterministic spacecraft-power
analysis. It couples one retained ECI state history to analytic Sun geometry,
an explicit shadow model, one solar array, a lumped battery, and a declared
load timeline. The machine-readable records are defined by
[`oel-spacecraft-power-v1.schema.json`](schemas/oel-spacecraft-power-v1.schema.json).

## Input contracts

`oel.spacecraft_power_problem.v1` declares one asset, an absolute Julian UTC
epoch, a horizon no longer than seven days, an integration step no larger than
60 seconds, transition-refinement settings, array and battery parameters, base
load, and at most 1,000 bounded activities. Unknown, missing, non-finite, or
out-of-horizon values fail validation.

`oel.spacecraft_power_history.v1` retains at most 200,000 strictly increasing
samples in ECI kilometres and kilometres per second. `sun_tracking_ideal`
requires no attitude. `history_body_fixed` requires a normalized body-from-ECI
quaternion at every state sample and applies the declared body-frame array
normal. The problem epoch and asset ID must match the history exactly.

The optional mission-schedule adapter first authoritatively verifies one
`oel.mission_scheduling_evidence.v1` directory. It converts selected
observation and downlink rows for the named asset into constant activity loads
from the verifier's recomputed authoritative activity records and binds every
converted activity to the schedule's semantic digest. It never rereads a
receipt-only ledger after verification and does not change or repair the
schedule.

## Deterministic model

The Sun comes from OEL's selected `analytic_simple` or `analytic_enhanced`
ephemeris. Illumination uses OEL's existing `none`, cylindrical, or conical
shadow provider. Sunlight, penumbra, and umbra boundaries are refined to the
declared time tolerance. Each transition retains both its original
pre-refinement analysis-grid bracket and its final narrowed refinement bracket.
Analysis fails if the iteration limit cannot meet the declared tolerance.

For each integration interval, the analyzer uses Simpson-average solar-array
generation and the midpoint declared load. Generation is limited by array
area, efficiency, inverse-square solar distance, incidence cosine, and maximum
generation power. Solar energy serves the load first. Surplus charges the
battery subject to charge rate, charge efficiency, and maximum state of charge;
remaining surplus is curtailed. A deficit discharges the battery subject to
discharge rate, discharge efficiency, and minimum state of charge; remaining
demand is unmet.

The result is `feasible` only when integrated unmet load is zero within the
contract tolerance. State-of-charge saturation, reserve depletion, unmet-load
onset, and illumination transitions are explicit events. Battery-storage,
power-bus, and load-service conservation residuals are retained separately.

## Evidence and replay

One output directory contains exactly:

```text
normalized_problem.json
normalized_history.json
spacecraft_power_summary.json
spacecraft_power_timeseries.csv
spacecraft_power_intervals.csv
spacecraft_power_events.csv
spacecraft_power_manifest.json
```

`oel.spacecraft_power_evidence.v1` is the summary consumed by the public study
lifecycle. `oel.spacecraft_power_manifest.v1` binds the exact bytes and
scientific semantic digests. Writes require an absent destination and are
atomically promoted from a sibling temporary directory.

Authoritative replay rejects symbolic links, unexpected artifacts, oversized
or malformed records, noncanonical normalized inputs, receipt changes, and
unknown manifest fields. It recomputes the analysis from the retained problem
and history and byte-compares the summary and all three derived CSV tables.

## Validation and claim limits

Public acceptance covers hand-computed sunlight charging and battery
depletion, exact reserve timing, closed-form circular cylindrical-eclipse
duration, conical sunlight/penumbra/umbra transitions, fixed-array incidence,
integration-step refinement, energy conservation, schedule digest binding,
schema validation, deterministic replay, tampering, malformed inputs, and CLI
failure behavior. These checks require no paid service.

The v1 result is deterministic engineering evidence for one supplied history
and declared load timeline. It excludes thermal state, temperature-dependent
performance, degradation, self-shadowing, regulator or bus topology,
uncertainty, probabilistic availability, hardware qualification, and
operational authorization.
