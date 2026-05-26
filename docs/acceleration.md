# Optional Acceleration

Orbital Engagement Lab includes an optional acceleration layer for hot numeric kernels. The first supported backend is
Numba/JIT, exposed as an opt-in feature so ordinary installs and validation runs remain reproducible on machines without
Numba.

## Install

```bash
pip install -e ".[accel]"
```

For compatibility installs, `.[full]` also includes the acceleration extra.

## Warmup

Warmup compiles the supported kernels and populates Numba's on-disk cache for the current Python, platform, and CPU.

```bash
python -m sim.acceleration.warmup
python -m sim.acceleration.warmup --profile validation
```

The command also works without Numba installed; in that case it exercises the Python fallback kernels and reports the
backend as `python`.

## Benchmark

Use the bundled benchmark to confirm the supported RK4/J2 orbit fast path on a local machine:

```bash
python -m sim.acceleration.benchmarks --iterations 10000
python -m sim.acceleration.benchmarks --iterations 10000 --json
```

The benchmark reports baseline Python propagator time, accelerated-path time, speedup, and final state delta norm.
Use `--kind attitude` to include the attitude exponential-map propagation benchmark, `--kind estimation` to time the
orbit/attitude EKF finite-difference Jacobian paths, or `--kind all` to run every local acceleration benchmark:

```bash
python -m sim.acceleration.benchmarks --kind all
python -m sim.acceleration.benchmarks --kind estimation --estimation-iterations 1000
```

## Runtime Controls

Acceleration is disabled by default. Enable it per run with:

```bash
python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --acceleration auto
```

or in YAML:

```yaml
simulator:
  acceleration:
    mode: auto   # off | auto | numba
    warmup: false
```

The environment variable `OEL_ACCELERATION=off|auto|numba` overrides config mode for the current process.
Game sessions force acceleration off through their runtime config so players do not pay first-run JIT cost.

## Current Coverage

The first accelerated path covers fixed-step RK4 orbital propagation when the dynamics use only:

- two-body gravity
- J2
- J3
- J4
- constant command acceleration for the integration step

Scenarios with drag, SRP, spherical harmonics, third-body plugins, adaptive integrators, or custom acceleration plugins
fall back to the existing Python implementation. RIC frame transforms are wired into the runtime acceleration path, and
re-entry scalar kernels are available for warmup and parity tests while they are staged for broader integration.

Attitude acceleration currently covers the exponential-map rigid-body attitude propagation path used by
`OrbitalAttitudeDynamics`. The accelerated kernel preserves quaternion normalization, angular-rate/torque clamps,
singular-inertia handling, and guardrail event accounting through the existing Python wrapper.

Estimator acceleration currently covers the orbit EKF two-body RK4 propagation/Jacobian path and the attitude EKF
propagation/Jacobian path used inside the joint-state estimator. This targets long RIC_PD-style runs where estimator
updates dominate runtime after the core orbit and attitude dynamics are accelerated.

## Resource Notes

Acceleration reduces runtime per supported numerical step; it does not replace resource profiles, checkpointing, or
thermal safeguards. For Monte Carlo validation, keep using `--resource-profile laptop-safe` or `standard` on local
machines. The first accelerated call may include compilation overhead unless kernels have already been warmed.
