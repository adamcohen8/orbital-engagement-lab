# Optional Acceleration

Orbital Engagement Lab includes an optional acceleration layer for hot numeric kernels. The first supported backend is
Numba/JIT, exposed as an opt-in feature so ordinary installs and validation runs remain reproducible on machines without
Numba.

## Install

```bash
.venv/bin/python -m pip install -e ".[accel]"
```

For compatibility installs, `.[full]` also includes the acceleration extra.
The acceleration extra installs Numba plus SciPy because Numba's linear algebra
lowering uses SciPy BLAS symbols on supported JIT paths.

## Warmup

Warmup compiles the supported kernels and populates Numba's on-disk cache for the current Python, platform, and CPU.

```bash
.venv/bin/python -m sim.acceleration.warmup
.venv/bin/python -m sim.acceleration.warmup --profile validation
```

The command also works without Numba installed; in that case it exercises the Python fallback kernels and reports the
backend as `python`.

## Benchmark

Use the bundled benchmark to confirm the supported RK4/J2 orbit fast path on a local machine:

```bash
.venv/bin/python -m sim.acceleration.benchmarks --iterations 10000
.venv/bin/python -m sim.acceleration.benchmarks --iterations 10000 --json
```

The benchmark reports baseline Python propagator time, accelerated-path time, speedup, and final state delta norm.
Use `--kind attitude` to include the attitude exponential-map propagation benchmark, `--kind estimation` to time the
orbit/attitude EKF finite-difference Jacobian paths, or `--kind all` to run every local acceleration benchmark:

```bash
.venv/bin/python -m sim.acceleration.benchmarks --kind all
.venv/bin/python -m sim.acceleration.benchmarks --kind estimation --estimation-iterations 1000
```

For end-to-end timing across propagation, the full satellite loop, sensing and estimation, actuators, lifecycle models,
campaign orchestration, and artifact generation, use the [full-path performance suite](performance-benchmarks.md).

## Runtime Controls

Acceleration is disabled by default. Enable it in YAML:

```yaml
simulator:
  acceleration:
    mode: auto   # off | auto | numba
    warmup: false
```

The environment variable `OEL_ACCELERATION=off|auto|numba` overrides config mode for the current process.
Game sessions force acceleration off through their runtime config so players do not pay first-run JIT cost.

## Current Coverage

The fixed-step RK4 orbit fast path covers propagation when the dynamics use only:

- two-body gravity
- J2
- J3
- J4
- constant command acceleration for the integration step

Normalized spherical-harmonic gravity fields also use an accelerated force-evaluation kernel with fixed-step and
adaptive integrators. The authoritative Python frame implementation still prepares each ECI-to-body-fixed rotation;
the optional backend compiles the normalized Legendre recurrence and degree/order summation. Unnormalized or mixed
coefficient sets retain the existing Python finite-difference implementation. Other force plugins continue through
their existing implementation even when they coexist with an accelerated normalized gravity field.
Repeated IAU-76/80/EOP frame inputs reuse an exact immutable cached rotation;
callers still receive independent arrays, and no time interpolation or frame
approximation is introduced.

NRLMSISE-00 uses an accelerated upper-atmosphere diffusion/spline kernel across
its full altitude and space-weather input domain. The kernel preserves the
reference coefficient set and operation ordering, runs with fast math disabled,
and falls back to the authoritative Python implementation whenever acceleration
is unavailable or disabled. A larger no-fast-math kernel covers the standard-
switch thermosphere at and above 300 km when the model's historical Ap inputs
select its quiet branch; disturbed conditions and lower altitudes retain the
complete authoritative path. WGS-84 atmosphere coordinates use the same exact
iterative conversion in one compiled boundary for every accelerated atmosphere
model. MSIS-86 similarly accelerates its complete
temperature/density profile calculation and adds a compiled fixed-switch globe
path when the model's Ap-history criterion selects its quiet branch. Disturbed-
Ap inputs retain the authoritative Python globe calculation, while all MSIS-86
paths retain the Python fallback when acceleration is unavailable or disabled.
Other atmosphere models retain their existing model-specific accelerated
kernels and Python fallbacks.

DE440 light ephemerides accelerate the mandatory Earth-Moon barycenter, Moon,
and Sun Chebyshev evaluations in one no-fast-math kernel when the compact NPZ
coefficient format is selected. The Sun/Moon-only path also performs UTC-to-TDB
conversion and geocentric reduction inside that boundary. Callers that need
only the geocentric Sun/Moon pair avoid constructing the complete position dictionary; optional planetary
bodies and MAT-file coefficient sources retain the authoritative Python paths.
Cannonball SRP similarly fuses spacecraft-to-Sun geometry, cylindrical or
conical eclipse evaluation, pressure scaling, and acceleration into one
no-fast-math kernel whenever the time-dependent environment has resolved a Sun
position. Acceleration-off mode retains the original Python implementations for
both capabilities.

Fixed-step RK4 scenarios in the exact quiet-thermosphere NRLMSISE-00 domain can
also execute the complete supported force plan behind one compiled boundary.
The plan preserves configured plugin order and may combine normalized spherical
harmonics, drag, cannonball SRP, and DE440 Sun/Moon third-body gravity beneath
the existing `OrbitPropagator.propagate` API. Frame/EOP and ephemeris
preparation remain owned by their authoritative implementations. Disturbed Ap,
altitudes below 300 km, and acceleration-off mode retain the authoritative
implementations.

A staged compiled-component tier covers richer plans outside that fused domain
for both fixed-step RK4 and adaptive RKF78. It supports explicit J2/J3/J4,
normalized spherical harmonics, drag and lift with constant density or any OEL
atmosphere family (exponential, USSA-1976, NRLMSISE-00, MSIS-86, Jacchia-70,
JB2006, JB2008, and Harris-Priester), cannonball SRP, Sun/Moon and selected
planetary third-body gravity, and plans interleaved with custom acceleration
callbacks. Authoritative Python owners prepare state-dependent density, frame,
ephemeris, and custom-plugin values; compiled kernels evaluate the supported
numeric force components; the propagator then accumulates all contributions in
the configured plugin order. Custom Python callbacks are preserved rather than
silently treated as nopython code. Small force plans continue through their
faster specialized compiled evaluators when staging would add overhead, and
mixed or unnormalized harmonic fields retain their authoritative evaluator.

RIC frame transforms are wired into the runtime acceleration path, and re-entry scalar kernels are available for
warmup and parity tests while they are staged for broader integration.

Attitude acceleration covers the exponential-map rigid-body path used by
`OrbitalAttitudeDynamics`. A staged numeric plan can evaluate the built-in
gravity-gradient, magnetic, scalar/facet drag, and scalar/facet SRP torques and
propagate all attitude substeps behind one compiled boundary. The plan preserves
the public disturbance accumulation order, quaternion normalization,
angular-rate/torque clamps, singular-inertia handling, and guardrail event
accounting. Custom disturbance objects, geometry lookup profiles, rectangular-
prism face models, acceleration-off runs, and unavailable acceleration backends
retain the authoritative Python path.
The coupled dynamics object's owned default two-body orbit propagator inherits
the same acceleration mode; explicitly supplied orbit propagators retain their
own configured mode.

Estimator acceleration currently covers the orbit EKF two-body RK4 propagation/Jacobian path and the attitude EKF
propagation/Jacobian path used inside the joint-state estimator. This targets long RIC_PD-style runs where estimator
updates dominate runtime after the core orbit and attitude dynamics are accelerated.

## Resource Notes

Acceleration reduces runtime per supported numerical step; it does not replace
resource planning, checkpointing, or thermal safeguards for long campaign
workflows. The first accelerated call may include compilation overhead unless
kernels have already been warmed.
