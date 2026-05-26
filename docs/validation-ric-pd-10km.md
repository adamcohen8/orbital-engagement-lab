# RIC_PD 10 km Validation Package

This page defines the focused validation package for the flagship
`configs/ric_pd_10km_experiment.yaml` scenario.

## Validation Claim

Under the assumptions encoded in `configs/ric_pd_10km_experiment.yaml`, the
flagship RIC_PD 10 km RPO scenario should run deterministically to completion,
close from a 10 km-class initial separation to a sub-10 m terminal range, keep
terminal relative speed below 0.01 m/s, use less than 10 m/s total delta-v,
avoid attitude guardrail events, and preserve finite chaser-target knowledge
throughout the run.

This is a scenario-level regression and evidence package. It is not a
flight-qualification claim, external truth-model validation, or proof of
performance outside the configured initial condition, dynamics, sensor,
actuator, and controller assumptions.

## Run Command

```bash
python validation/automated_validation_harness.py --suite ric_pd_10km
```

Validation defaults to the `laptop-safe` resource profile. Simulation
benchmarks run one Monte Carlo case at a time, disable plots, checkpoint each
completed case, and refuse to start when current resource preflight is unsafe.
During batch execution, the resource governor also pauses before launching the
next case if local memory or load pressure is high. To inspect the expected load
without running:

```bash
python validation/automated_validation_harness.py --suite ric_pd_10km --estimate-resource-requirements
```

To intentionally discard saved Monte Carlo progress and start fresh:

```bash
python validation/automated_validation_harness.py --suite ric_pd_10km --clear-checkpoints
```

Checkpoints are reused only when their generated iteration config hash still
matches the current scenario and controller settings.
Use `--allow-unsafe-resource-start` only for intentional local overrides after
reviewing the preflight output.

The suite performs:

- plugin validation for the flagship config,
- one deterministic single-run simulation,
- a three-run Monte Carlo perturbation envelope around the flagship initial
  condition,
- derived relative-motion checks for final range and final relative speed,
- delta-v, burn-sample, attitude, and knowledge finite-history checks,
- evidence-manifest generation with git, platform, dependency, and input
  provenance.

The deterministic single run also preserves the knowledge evidence chain:
truth state, raw state measurements when present, and filtered knowledge
estimates. Because the flagship config uses a deterministic near-perfect
knowledge path, this artifact is primarily a traceability check for the
controller validation. Noisy sensor-distribution and filter-noise-rejection
claims belong to the `estimation_knowledge` and `sensor_measurements`
validation suites, where nonzero sensor noise is part of the configured
envelope.

## Acceptance Gates

The canonical gates live in `configs/validation_harness_ric_pd_10km.yaml`.
The headline checks are:

| Gate | Threshold |
| --- | ---: |
| Initial range | `>= 9.0 km` |
| Final range | `<= 0.01 km` |
| Closest approach | `<= 0.01 km` |
| Final relative speed | `<= 0.01 m/s` |
| Total delta-v | `<= 10.0 m/s` |
| Chaser burn samples | `1` to `9000` |
| Target burn samples | `0` |
| Chaser max acceleration | `<= 6.000001e-5 km/s^2` |
| Attitude guardrail events | `0` |
| Chaser attitude finite fraction | `1.0` |
| Chaser-target knowledge finite fraction | `1.0` |

The Monte Carlo envelope lives in `configs/ric_pd_10km_experiment_mc.yaml`.
It varies the initial in-track separation from `-12 km` to `-8 km`, initial
radial velocity from `-0.25 m/s` to `+0.25 m/s`, and initial cross-track
velocity from `0.7 m/s` to `1.3 m/s`. Each run must satisfy:

| Monte Carlo per-run gate | Threshold |
| --- | ---: |
| Final range | `<= 0.05 km` |
| Final relative speed | `<= 0.05 m/s` |
| Total delta-v | `<= 12.0 m/s` |
| Chaser burn samples | `<= 11000` |
| Target burn samples | `0` |
| Attitude guardrail events | `0` |

The aggregate harness gate requires all three runs to pass.

## Evidence Artifacts

The harness writes:

```text
outputs/validation_harness_ric_pd_10km/validation_harness_report.json
outputs/validation_harness_ric_pd_10km/validation_harness_report.md
outputs/validation_harness_ric_pd_10km/validation_evidence_manifest.json
outputs/validation_harness_ric_pd_10km/validation_attitude_summary.json
outputs/validation_harness_ric_pd_10km/validation_attitude_summary.md
outputs/validation_harness_ric_pd_10km/validation_estimation_knowledge_summary.json
outputs/validation_harness_ric_pd_10km/validation_estimation_knowledge_summary.md
outputs/validation_harness_ric_pd_10km/ric_pd_10km_single_run/truth_measurement_estimate_chain.png
```

The single-run benchmark also writes a copy of the scenario outputs under:

```text
outputs/validation_harness_ric_pd_10km/ric_pd_10km_single_run/
outputs/validation_harness_ric_pd_10km/ric_pd_10km_monte_carlo/
outputs/validation_harness_ric_pd_10km/ric_pd_10km_monte_carlo/mc_checkpoints/
```

## Interpretation

Passing this package means the current implementation still satisfies the
configured RIC_PD flagship acceptance envelope on the local runtime. It supports
product confidence, release review, and regression protection for the flagship
controller path.

It does not validate:

- arbitrary RPO geometries,
- high-fidelity perturbation environments,
- sensor-noise envelopes beyond the configured deterministic knowledge path,
- flight-software timing,
- actuator hardware behavior,
- operational safety or rules of engagement.

Use Monte Carlo, sensitivity, controller-bench, and HPOP-backed packages to
expand the evidence envelope after this deterministic package is stable.
