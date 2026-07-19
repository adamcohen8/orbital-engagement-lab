# Public-Core Performance Benchmarks

OEL's public-core performance suite measures distinct runtime workloads instead of treating one scenario as a proxy for
the entire public simulator. The suite records wall time, simulated-time throughput, work counts, runtime-profiler stage totals,
coverage assertions, and exact repeated-run physics hashes.

The benchmark is a timing workflow, not a replacement for physics validation. Continue to use the applicable external
validation harness for correctness claims.

## Run the suite

List the maintained cases without executing them:

```bash
.venv/bin/python -m sim.performance --list
```

Run the short integration profile:

```bash
.venv/bin/python -m sim.performance --profile smoke
```

Run the default measurement profile, which warms code and static data for each case and reports the median of three
measured repetitions:

```bash
.venv/bin/python -m sim.performance --profile standard
```

Use the `full` profile for longer sustained workloads. A case or category can be isolated when investigating a hotspot:

```bash
.venv/bin/python -m sim.performance --profile standard --case sensing_relative_ekf
.venv/bin/python -m sim.performance --profile standard --category propagation
```

Use `--json` for a machine-readable stdout payload, `--output-dir` to select the result directory, and `--fail-on-skip`
when optional or external coverage must be present in automation.

## Maintained workload matrix

| Case | Runtime path exercised |
|---|---|
| `zonal_rk4_accelerated` | Fixed-step ONP RK4 with explicit J2/J3/J4 and the accelerated kernel |
| `sensing_relative_ekf` | Geometric LOS/range/dropout access, angles/range/range-rate measurements, relative HCW EKF, maneuver screening, and ground access |
| `modern_actuator_stack` | RCS, electric propulsion, gimbals, magnetorquers, CMGs, reaction-wheel device dynamics, desaturation, pulse torque, and faults |
| `cr3bp_earth_moon` | Earth-Moon CR3BP propagation on the maintained NRHO seed |
| `ogp_sgp4` | Passive catalog-style OGP-SGP4 propagation |
| `rocket_ascent` | Rocket guidance, atmosphere, aerodynamics, staging/mass, and ascent diagnostics |
| `reentry_diagnostics` | Atmospheric re-entry dynamics, heating, dynamic pressure, and load diagnostics |
| `artifact_output_pipeline` | Simulation plus JSON/full log, review store, index, and plot generation |

The source-of-truth matrix is `configs/performance_benchmark_suite.yaml`. Purpose-built scenario fixtures live under
`configs/performance/`; maintained product scenarios are reused where they are already the right workload.

The eight `drag_*` rows intentionally share one spacecraft, orbit, epoch, RK4 step, force selection, duration, and output
policy. Only `simulator.environment.atmosphere_model` changes between them. Compare those rows directly to study relative
atmosphere-model runtime cost. Each measured atmospheric repetition starts with an empty trajectory-epoch cache while
retaining any imports, compiled acceleration kernels, and static input tables prepared by the selected profile's
warm-ups. Do not interpret equal runtime as physical agreement between the models' density predictions.

## Profiles and interpretation

- `smoke` uses one short measured execution and is intended to prove that every path is callable.
- `standard` uses one warm-up and three measured repetitions. Use this for normal before/after optimization work.
- `full` uses one warm-up, five repetitions, and longer durations to reduce startup noise.

When configured, warm-ups prepare imports, optional compiled kernels, and static model data. Before every measured
`drag_*` repetition, the suite clears Jacchia-70 and Harris-Priester caches keyed by individual trajectory epochs.
Consequently, `standard` and `full` atmosphere timings represent a fresh trajectory in an otherwise warmed process, not
repeated execution of an identical cached trajectory. `smoke` has no configured warm-up, so its measurement can include
startup work. The JSON result records the per-repetition policy as
`measurement_cache_policy: cold_trajectory_epoch_cache`.

Each case must pass its configured capability assertions. Repeated executions must also produce the same canonical
physics hash; runtime measurements, output paths, and artifact provenance are excluded from that hash. A one-repeat smoke
run confirms coverage but does not establish repeat parity by itself, so optimization evidence should use `standard`,
`full`, or an explicit `--repeats 2` or greater.

The public core uses deterministic serial object stepping. The JSON result records the actual runtime backend so benchmark reports remain explicit about what was measured.

The report writes `benchmark_results.json`, `benchmark_report.md`, effective configs, and per-case scratch outputs. Compare
results only when the manifest/profile, hardware, Python environment, acceleration mode, measurement cache policy,
resource policy, and output policy match. The report embeds the Git commit, dirty-worktree flag, platform, CPU count,
Python executable, and NumPy version for that reason.
