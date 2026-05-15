# Flagship HCW PD 10 km Scenario

`configs/hcw_pd_10km_experiment.yaml` is the recommended review scenario after
the five-minute quickstart. It is longer than the quickstart because it exercises
a tuned HCW PD relative-orbit controller, attitude dynamics, reaction-wheel
pointing, and thrust-alignment gating in one deterministic run.

## Run It

Validate first:

```bash
python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml --validate-only
```

Run the scenario:

```bash
python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml
```

Open the start-here file first:

```text
outputs/flagship_hcw_pd_10km/index.md
```

## Review Order

Start with the run index and summary:

- `outputs/flagship_hcw_pd_10km/index.md`
- `outputs/flagship_hcw_pd_10km/master_run_summary.json`

Then inspect the rendezvous and control plots:

- `outputs/flagship_hcw_pd_10km/rendezvous_summary.png`
- `outputs/flagship_hcw_pd_10km/control_effort.png`
- `outputs/flagship_hcw_pd_10km/relative_ranges.png`
- `outputs/flagship_hcw_pd_10km/trajectory_ric_curv_2d_multi.png`

For the integrated orbit-attitude behavior, review:

- `outputs/flagship_hcw_pd_10km/attitude_control_summary.png`
- `outputs/flagship_hcw_pd_10km/chaser_thrust_alignment_error.png`
- `outputs/flagship_hcw_pd_10km/chaser_attitude.png`
- `outputs/flagship_hcw_pd_10km/chaser_quaternion_error.png`

Selected snapshots from this visual set are checked into the public
[Plot Gallery](plot-gallery.md). The local `index.md` remains the authoritative
artifact inventory for the current config.

## Python Analysis

The companion API example runs the same config and writes a small custom metrics
package:

```bash
python examples/python/flagship_analysis.py
```

It writes:

```text
outputs/flagship_hcw_pd_10km/custom_analysis/flagship_metrics.json
outputs/flagship_hcw_pd_10km/custom_analysis/flagship_metrics.csv
```

Use this script as the starting point for custom notebooks or mission-analysis
scripts that need to compute scenario-specific metrics beyond the built-in
payload summary.

## What It Demonstrates

- A chaser initialized roughly 10 km in-track from a passive target.
- A tuned public `HCWPDController` operating on curvilinear RIC state.
- Two-body deterministic orbit dynamics with perturbations disabled.
- Chaser attitude dynamics and reaction-wheel PD stabilization.
- Integrated command execution that only applies orbital thrust after attitude
  alignment is within tolerance.
- Saved JSON, CSV, and PNG artifacts suitable for a lightweight engineering
  review.

## What It Does Not Claim

This scenario is a public golden path, not mission qualification evidence. It
does not claim high-fidelity environmental accuracy, flight-software readiness,
operational safety, or validated performance outside its configured assumptions.
For the current validation posture, see [Validation Claims](validation-claims.md)
and [Known Limitations](known-limitations.md).
