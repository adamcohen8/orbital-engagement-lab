# Spacecraft Power Analysis

OEL v0.29 can determine whether one declared spacecraft load timeline is
power-feasible along a retained orbit. The public workflow resolves eclipse,
solar-array generation, charge/discharge limits, battery reserve, curtailment,
and unmet load, then writes replayable JSON and CSV evidence.

## Run the canonical schedule-coupled example

```bash
.venv/bin/python examples/python/spacecraft_power_schedule_chain.py \
  --output-root outputs/public_spacecraft_power
```

The example solves the existing two-asset public schedule, selects SAT-A's
activities, generates a two-body ECI history spanning eclipse, assesses power,
authoritatively replays the result, and binds its summary into a verified study
lifecycle bundle. The terminal summary distinguishes power recomputation from
lifecycle identity replay.

## Analyze your own completed run

First export one object's ECI history from a completed run with a review store:

```bash
.venv/bin/python -m sim.spacecraft_power export-review-history \
  outputs/my_completed_run --object-id spacecraft \
  --output outputs/my_power_history.json
```

Copy the checked-in problem as a starting point, then set the epoch, asset ID,
horizon, array, battery, loads, and attitude posture deliberately:

```bash
.venv/bin/python -m sim.spacecraft_power validate \
  my_power_problem.json outputs/my_power_history.json
.venv/bin/python -m sim.spacecraft_power analyze \
  my_power_problem.json outputs/my_power_history.json \
  --output-dir outputs/my_power_evidence
.venv/bin/python -m sim.spacecraft_power replay \
  outputs/my_power_evidence
```

The same commands are available from the unified launcher as `oel power ...`.
Inputs and outputs use the versioned schema in the
[Spacecraft Power Analysis Contract](contracts/spacecraft-power-contract.md).

To add activities from a completed public mission schedule, supply all three
adapter arguments:

```bash
.venv/bin/python -m sim.spacecraft_power analyze \
  my_power_problem.json outputs/my_power_history.json \
  --mission-schedule outputs/my_schedule \
  --observation-load-w 180 --downlink-load-w 120 \
  --output-dir outputs/my_schedule_power_evidence
```

The adapter verifies and recomputes the schedule before conversion, consumes
the verifier's authoritative activity records, and records its semantic digest
on every added activity. It does not reinterpret the scheduler's horizon-level
energy budget as battery state.

## Read the result

Start with `spacecraft_power_summary.json`:

- `feasibility` is `feasible` only when no declared load is unmet;
- `battery.minimum_soc_margin_fraction` is margin above the declared reserve;
- `totals` separates generated, served, unmet, charged, discharged, and
  curtailed energy;
- `conservation_residuals_wh` exposes three independent accounting closures;
- `source_product_sha256s` binds any converted schedule; and
- `claim_limits` travels with the scientific result.

Use the timeseries for power and state-of-charge traces, intervals for refined
eclipse phases, and events for transitions, reserve depletion, saturation, and
unmet-load onset. Illumination events retain the source sample bracket in
`original_bracket_start_s` / `original_bracket_end_s` and the narrowed bracket
in `bracket_start_s` / `bracket_end_s`. Cite a result only after `replay`
returns `status: verified`.

## Public and Pro boundary

The strict contracts, deterministic local analysis, review-history and
schedule adapters, complete evidence, replay, schema, and small examples are
public. Pro or future work includes uncertainty and Monte Carlo, degradation,
thermal coupling, detailed EPS/network models, managed environmental data,
optimization and campaign trades, customer spacecraft models, dashboards,
and qualification evidence. Neither edition turns this v1 result into flight
or operational authorization.
