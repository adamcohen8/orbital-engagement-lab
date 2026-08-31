# Curated Examples

The product-facing examples are YAML scenario configs. They are meant to be
validated and run through the standard CLI:

```bash
.venv/bin/python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml --validate-only
.venv/bin/python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml
```

## Public Configs

- `public_tle_2hr_propagation.yaml`: initialize from TLE lines, then predict a two-hour state history with OEL numerical propagation.
- `public_sgp4_passive_propagation.yaml`: propagate a passive catalog-style object from TLE lines with `propagation_method: general` and `general.model: sgp4`.
- `public_ground_station_access_from_tle.yaml`: initialize from a TLE, then compute ground-station access windows with OEL numerical propagation.
- `public_coverage_and_link_analysis.yaml`: evaluate whole-Earth conical coverage and a directed free-space link after one deterministic propagation.
- `public_closed_loop_rendezvous_lqr.yaml`: run a closed-loop chaser/target rendezvous with HCW LQR.
- `public_orbit_environment_stack.yaml`: inspect perturbation/environment toggles in deterministic propagation.
- `public_attitude_hold_disturbance.yaml`: evaluate attitude hold with initial pointing error and disturbance torque.
- `public_manual_rpo_training.yaml`: launch a manual/game-style RPO scenario with editable player authority.
- `public_rendezvous_closed_loop.yaml`: broader closed-loop RPO artifact review with attitude pointing, sensing, EKF knowledge, and standard plots.
- `public_manual_engagement.yaml`: manual/game-mode engagement with stabilized attitude, object knowledge, and defensive target logic.

Public configs use the canonical `objects` map. Conventional object IDs such as
`chaser` and `target` are example names, not required engine slots.

The standard TLE examples are initializer-only workflows: OEL samples OGP to
recover an ECI-compatible initial state at `simulator.initial_jd_utc`, then
integrates the configured OEL force model. The explicit
`public_sgp4_passive_propagation.yaml` example is the continuous catalog-style
OGP exception and is passive by design.

## CCSDS TDM Tracking OD

The bounded public tracking-data example parses reduced-geometric CCSDS TDM
2.0 KVN AZEL/range observations, fits a declared arc, and repropagates the
solution against an untouched holdout:

```bash
.venv/bin/python -m sim.tracking_od inspect-tdm \
  examples/tracking_od/public_reduced_geometric_azel_range.tdm
.venv/bin/python -m sim.tracking_od fit \
  examples/tracking_od/public_reduced_geometric_azel_range.tdm \
  examples/tracking_od/public_tdm_fit_holdout_problem.json \
  --output-dir outputs/public_tdm_fit_holdout
```

See [CCSDS TDM Tracking Orbit Determination](../docs/tracking-od.md) for the
supported profile, evidence interpretation, and public/Pro boundary.

## Multi-Asset Mission Scheduling

Solve and authoritatively replay the bounded two-spacecraft collection and
downlink example:

```bash
.venv/bin/python -m sim.mission_scheduling solve \
  examples/mission_scheduling/public_two_asset_collection_problem.json \
  --output-dir outputs/public_two_asset_collection
.venv/bin/python -m sim.mission_scheduling replay \
  outputs/public_two_asset_collection
```

See [Bounded Multi-Asset Mission Scheduling](../docs/mission-scheduling.md) for
problem semantics, validation, evidence interpretation, and the Public/Pro
boundary.

To generate the observation and link products with OEL before building and
replaying their schedule:

```bash
.venv/bin/python examples/python/mission_scheduling_source_chain.py \
  --output-root outputs/public_mission_source_chain
```

## Spacecraft Power

Solve the public schedule, couple SAT-A's selected activities to a two-body
orbit spanning eclipse, assess solar-array and battery feasibility, replay the
scientific evidence, and retain it as a verified study:

```bash
.venv/bin/python examples/python/spacecraft_power_schedule_chain.py \
  --output-root outputs/public_spacecraft_power
```

See [Spacecraft Power Analysis](../docs/spacecraft-power.md) for direct CLI
usage, evidence interpretation, validation, and the Public/Pro boundary.

## Orbit Lifetime

Propagate one declared ONP drag case to refined altitude thresholds, compare
four atmosphere assumptions with identical non-atmosphere inputs, replay both
products, and retain the single case as a verified study:

```bash
.venv/bin/python examples/python/orbit_lifetime_workflow.py \
  --output-root outputs/public_orbit_lifetime
```

See [Deterministic Orbit Lifetime Analysis](../docs/orbit-lifetime.md) for
direct CLI usage, evidence interpretation, validation, and the Public/Pro
boundary.

## Integrated Study Lifecycle

Run real trajectory-targeting, conjunction-assessment, and mission-scheduling
examples, then retain each result as a verified content-bound study:

```bash
.venv/bin/python examples/python/study_lifecycle_three_domains.py \
  --output-root outputs/study_lifecycle_three_domains
```

The generated summaries distinguish lifecycle identity replay from each
domain's authoritative physics replay. See
[Integrated Study Lifecycle](../docs/study-lifecycle.md).

## Flagship Built-In Scenario

After the quickstart, the recommended public review path is the built-in RIC_PD
10 km scenario:

```bash
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
.venv/bin/python examples/python/flagship_analysis.py
```

Open `outputs/flagship_ric_pd_10km/index.md` first, then inspect the custom
metrics under `outputs/flagship_ric_pd_10km/custom_analysis/`.

Private/Pro examples use `pro_*.yaml` names in the full private workspace and
are not included in the public export. Workflow-shaped Pro examples live under
`examples/workflows/`.

Older exploratory Python demos live outside the supported public examples
surface.

For a task-oriented table of public examples, expected outputs, and recommended
starting points, see [Examples Matrix](../docs/examples-matrix.md).
