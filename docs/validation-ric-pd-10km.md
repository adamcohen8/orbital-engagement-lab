# Flagship RIC_PD 10 km Scenario And Validation

`configs/ric_pd_10km_experiment.yaml` is the recommended engineering review
scenario after the quickstart. It combines a tuned RIC_PD relative-orbit
transfer law inside the complete `fsw.rpo_reference` stack, two-body dynamics,
reaction-wheel attitude control, and thrust-alignment gating in one
deterministic run.

## Run And Review

```bash
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
.venv/bin/python examples/python/flagship_analysis.py
```

Open `outputs/flagship_ric_pd_10km/index.md` first. It is the authoritative
artifact inventory. Then inspect:

- `master_run_summary.json` for the run summary;
- `rendezvous_summary.png`, `relative_ranges.png`, and
  `trajectory_ric_curv_2d_multi.png` for relative motion;
- `control_effort.png` for applied control;
- `attitude_control_summary.png` and
  `chaser_thrust_alignment_error.png` for orbit-attitude coupling;
- `custom_analysis/flagship_metrics.json` for companion API metrics.

Selected examples also appear in the [Plot Gallery](plot-gallery.md).

## What The Scenario Demonstrates

- A chaser initialized approximately 10 km in-track from a passive target.
- Raw onboard own-state and relative measurements processed by
  `fsw.rpo_reference`, with no simulator-owned controller bypass.
- Stack-native `ric_pd_transfer` guidance: transfer acquisition, coast and
  correction, optional final braking, and terminal cleanup on curvilinear RIC
  state. The reusable `RICPDTransferController` is a subordinate guidance law;
  the v2 translation controller owns SI limits and typed requested effort.
- Deterministic two-body dynamics with perturbations disabled.
- Maintained rendezvous-goal semantics after acquisition, plus chaser attitude
  dynamics, reaction-wheel stabilization, configured +Z thrust-axis pointing,
  and thrust gating.
- JSON, CSV, SQLite, and PNG evidence suitable for a lightweight review.

## Validation Claim

Under the checked-in assumptions, the scenario should run deterministically to
completion, close to a sub-10 m terminal range, keep terminal relative speed
below 0.01 m/s, use less than 10 m/s total delta-v, avoid attitude guardrail
events, and preserve finite chaser-target knowledge.

This is a scenario-level regression claim. It is not flight qualification,
external truth-model validation, operational safety evidence, or proof of
performance outside the configured dynamics, sensing, actuator, and controller
envelope.

## Acceptance Gates

| Gate | Threshold |
| --- | ---: |
| Initial range | `>= 9.0 km` |
| Final range and closest approach | `<= 0.01 km` |
| Final relative speed | `<= 0.01 m/s` |
| Total delta-v | `<= 10.0 m/s` |
| Chaser burn samples | `1` to `9000` |
| Target burn samples | `0` |
| Chaser max acceleration | `<= 6.000001e-5 km/s^2` |
| Attitude guardrail events | `0` |
| Attitude and knowledge finite fractions | `1.0` |

The private release-blocking harness runs only plugin validation and this
deterministic scenario.

The private three-run perturbation envelope is an optional exploratory check.
It varies initial in-track separation
from `-12 km` to `-8 km`, radial velocity from `-0.25 m/s` to `+0.25 m/s`, and
cross-track velocity from `0.7 m/s` to `1.3 m/s`. Each run must finish within
`0.05 km` and `0.05 m/s`, remain below `12 m/s` delta-v and `11000` chaser
burn samples, keep the target passive, and record no attitude guardrail events.
It must be requested explicitly from the private validation workspace.

Three samples cannot support calibrated percentile or uncertainty claims, so
this Monte Carlo is not release blocking and is not run by the release plan.

## Evidence

The public path writes the run index, summary, review store, figures, and custom
metrics. The private deterministic harness additionally writes its report,
content-bound evidence manifest, attitude and estimation summaries,
truth/measurement/estimate plot, and single-run copy under
`outputs/validation_harness_ric_pd_10km/`.
An explicitly requested exploratory Monte Carlo run writes its checkpoints in
the same output area.

The knowledge chain in this scenario is primarily a traceability check because
the configured path is deterministic and near-perfect. Noise-distribution and
filter-rejection claims belong to the dedicated sensor and
`estimation_knowledge` suites.

## Interpretation And Limits

Passing means the current implementation satisfies this configured acceptance
envelope on the evaluated runtime. It protects the flagship path against
regression and supports release review.

It does not validate arbitrary RPO geometries, high-fidelity perturbations,
hardware timing or actuators, broad sensor-noise envelopes, operational safety,
or controller superiority. See [Validation Claims](validation-claims.md) and
[Known Limitations](known-limitations.md) before generalizing the result.
