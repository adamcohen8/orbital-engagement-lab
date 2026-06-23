# Environment Perturbation Models

This page summarizes the environment perturbation models wired into OEL's
public scenario workflow. It is a model reference, not a validation claim. Use
the checked-in simulator, scenario YAML, tests, and validation harnesses as the
source of truth for a particular study.

OEL's default satellite propagator is an inertial Cartesian state
`[r_eci_km, v_eci_km_s]`. Standard orbit propagation uses two-body gravity as
the baseline acceleration and adds configured perturbation plugins. Force
models return acceleration in `km/s^2`; aerodynamic and SRP inputs use SI
vehicle properties such as `mass_kg`, `drag_area_m2`, `srp_area_m2`, `cd`, and
`cr`, with unit conversions at the force boundary.

## Core Frame And Unit Assumptions

- Primary orbit state: ECI position in kilometers and velocity in kilometers
  per second.
- Propagation time: seconds from scenario start.
- Epoch-dependent behavior: `simulator.initial_jd_utc` populates
  `jd_utc_start` so Sun/Moon ephemerides, Earth rotation, atmosphere local
  quantities, and ECEF conversions can be time dependent.
- Environment assembly: `simulator.environment` forms the base runtime
  environment. The nested `atmosphere_env` dictionary is flattened into that
  base, and `simulator.dynamics.orbit.atmosphere_model` may fill
  `atmosphere_model` when no shared environment value is supplied.
- Earth constants: `sim/dynamics/orbit/environment.py` defines the central
  body gravitational parameter, equatorial radius, zonal coefficients, Earth
  rotation rate, solar pressure constants, and third-body gravitational
  parameters used by the public implementation.
- Earth-fixed conversions: simple GMST rotation is the normal public path.
  HPOP-like frame paths can use EOP files when configured for validation
  comparisons.

## Additive Orbit Force Model

Implementation entry points:

- `sim/dynamics/orbit/propagator.py`: propagator and plugin wiring.
- `sim/runtime_support.py`: maps scenario YAML booleans to propagator plugins.
- `sim/dynamics/orbit/accelerations.py`: two-body, J2/J3/J4, drag, lift, SRP,
  and third-body acceleration formulas.
- `sim/dynamics/orbit/atmosphere.py`: atmosphere density and atmosphere state
  models.
- `sim/dynamics/orbit/spherical_harmonics.py`: generic spherical-harmonic
  gravity terms and coefficient loading.
- `sim/dynamics/orbit/epoch.py` and `sim/dynamics/orbit/eclipse.py`: Sun/Moon
  position resolution and SRP shadow geometry.

For `simulator.dynamics.orbit.model: two_body`, the derivative is:

```text
dr/dt = v
dv/dt = -mu r / |r|^3 + a_command + sum(a_plugins)
```

The CR3BP mode is a separate model path and rejects the ordinary two-body
perturbation flags.

## Gravity

### J2/J3/J4 Zonals

The explicit zonal toggles use closed-form perturbing accelerations in ECI for
J2, J3, and J4:

```yaml
simulator:
  dynamics:
    orbit:
      j2: true
      j3: true
      j4: true
```

The equations are the standard zonal spherical-harmonic acceleration families
using OEL's Earth `mu`, reference radius, and configured zonal constants. J2
uses the familiar oblateness term proportional to `J2 * mu * Re^2 / r^5`. J3
and J4 use the corresponding polynomial terms in `s = z/r`. See
`accel_j2`, `accel_j3`, and `accel_j4` in
`sim/dynamics/orbit/accelerations.py`.

When `spherical_harmonics.enabled` is true, runtime plugin wiring uses the
spherical-harmonics plugin instead of the explicit J2/J3/J4 plugins to avoid
double counting.

### Spherical Harmonics

OEL can also compute perturbations from configured harmonic terms:

```yaml
simulator:
  dynamics:
    orbit:
      spherical_harmonics:
        enabled: true
        degree: 8
        order: 8
        terms:
          - n: 2
            m: 0
            c_nm: -4.841693259705e-04
            s_nm: 0.0
            normalized: true
```

For inline terms, the perturbing potential follows:

```text
U_nm = mu/r * (Re/r)^n * P_nm(sin(phi)) *
       (C_nm cos(m lambda) + S_nm sin(m lambda))
```

The generic path computes the gradient of the perturbing potential in the
Earth-fixed frame with a configurable finite-difference step, then rotates the
acceleration back to ECI. Fully normalized HPOP/GGM03-style terms use the
analytic HPOP-like normalized Legendre path. Public distributions should
provide explicit `terms` or an explicit coefficient file path. HPOP/GGM03
reference files are not assumed to be present in the public core.

Key knobs:

- `spherical_harmonics.enabled`
- `degree`, `order`
- `terms`
- `source`, `coeff_path` or `source_path`
- `reference_radius_km`
- `frame_model`
- `eop_path`
- `fd_step_km`

## Atmospheric Density, Drag, And Lift

Drag is enabled by `simulator.dynamics.orbit.drag`. Re-entry diagnostics do not
enable drag by themselves.

```yaml
simulator:
  dynamics:
    orbit:
      drag: true
  environment:
    atmosphere_model: "ussa1976"
    atmosphere_env: {}
```

When no density override is supplied, the drag plugin calls
`density_from_model`. Supported model names include:

- `exponential`
- `ussa1976`
- `msis86`
- `nrlmsise00`
- `jacchia70`
- `jb2006`
- `jb2008`
- `harris_priester`

`density_kg_m3` in the environment overrides model lookup for drag and
re-entry calculations. The simple exponential model is a compact scale-height
model and returns zero above 1000 km. The USSA-1976 path uses standard
atmosphere layers through 86 km and log-space interpolation to 1000 km. The
other atmosphere paths are source-local or callable-backed engineering models
with solar/geomagnetic inputs supplied through `atmosphere_env`.

The drag acceleration is:

```text
a_drag = -0.5 * rho * Cd * A / m * |v_rel| * v_rel
```

where `v_rel` is the atmosphere-relative velocity. OEL supports simple
co-rotating atmosphere-relative velocity and HPOP-like frame handling via:

- `drag_frame_model`
- `drag_eop_path`
- `drag_earth_rotation_rad_s`
- `density_frame_model`
- `density_eop_path`

Satellite lift is optional and only participates when drag is enabled and
object aero specs provide lift information. Lift uses the same dynamic pressure
family, configured `cl`, a lift area, and a supplied lift direction projected
normal to relative wind. This is intended for first-pass atmospheric steering
studies, not detailed hypersonic vehicle aerodynamics.

Shared object-level aero properties live under `objects.<id>.specs.aero`,
including `reference_area_m2`, `drag_area_m2`, `lift_area_m2`, `cd`, `cl`,
`nose_radius_m`, `reference_length_m`, `lift_axis_body` or
`lift_vector_body`, and `cp_offset_body_m`.

## Solar Radiation Pressure

SRP is enabled with:

```yaml
simulator:
  dynamics:
    orbit:
      srp: true
```

The cannonball SRP acceleration is:

```text
a_srp = -(P_srp * distance_scale * Cr * A / m) *
        shadow_factor * sun_dir_sc_eci
```

`P_srp` defaults to solar irradiance divided by the speed of light and can be
overridden with `srp_pressure_n_m2` or `solar_irradiance_w_m2`. The area is
`srp_area_m2` when supplied, otherwise the object reference area. Sun geometry
comes from explicit environment vectors, configured ephemerides, or analytic
Sun/Moon models. Shadowing uses `srp_shadow_model`, with `conical` as the
normal model and `cylindrical` or `none` available for simpler cases.

## Third Bodies And Ephemerides

Sun and Moon third-body acceleration uses:

```text
a_3b = mu_b * ((r_b - r_sc)/|r_b - r_sc|^3 - r_b/|r_b|^3)
```

Enable the public runtime plugins with:

```yaml
simulator:
  dynamics:
    orbit:
      third_body_sun: true
      third_body_moon: true
```

The Sun/Moon resolver accepts explicit `sun_pos_eci_km` and
`moon_pos_eci_km`, sampled ephemeris arrays, an ephemeris callable, or an
epoch-driven `ephemeris_mode`. Supported modes include `analytic_simple`,
`analytic_enhanced`, `de440_hpop`/`de440`, and `spice`/`spiceypy`, depending on
available local data and dependencies.

`third_body_planets_plugin` exists for Mercury through Pluto when selected via
`third_body_planets`, but the standard scenario runtime currently wires only
the Sun and Moon booleans. Planetary third bodies require direct API/plugin
use or workflow-specific wiring.

## Re-Entry Diagnostics

Re-entry diagnostics are configured separately from the orbit force model:

```yaml
simulator:
  dynamics:
    reentry:
      enabled: true
      begin_altitude_km: 300.0
      atmosphere_model: "ussa1976"
      termination:
        enabled: false
```

An object is "active" for re-entry metrics while its current altitude is below
`begin_altitude_km`. Metrics are sampled from the propagated state and include
altitude, density, atmosphere-relative speed, dynamic pressure, drag
deceleration, lift acceleration, lift-to-drag ratio, g-load, Sutton-Graves heat
rate, and integrated heat load.

The diagnostic families are:

```text
q = 0.5 * rho * |v_rel|^2
a_drag_load = q * Cd * A_drag / m
a_lift_load = q * |Cl| * A_lift / m
g_load = hypot(a_drag_load, a_lift_load) / g0
heat_rate = k * sqrt(rho / nose_radius_m) * |v_rel|^3
heat_load += heat_rate * dt
```

Optional termination checks can stop a run on entry, minimum altitude, maximum
dynamic pressure, drag deceleration, g-load, heat rate, or heat load. These are
simulation stop criteria, not survivability or safety certification.

Evidence hooks:

- `summary.reentry_summary_by_object` in `master_run_summary.json`
- `reentry_metrics_by_object.<object_id>.*` in the full payload
- `master_run_history.npz` entries when history NPZ output is enabled
- re-entry plot IDs such as `reentry_summary`, `reentry_aero`, and
  `reentry_thermal`
- campaign metrics such as `reentry_peak_g_load`,
  `reentry_peak_heat_rate_w_m2`, `reentry_final_heat_load_j_m2`, and
  `reentry_min_altitude_km`

## Validation And Evidence Hooks

For ordinary scenario changes, validate YAML before running:

```bash
.venv/bin/python run_simulation.py --config <path> --validate-only
```

For orbit perturbation implementation or config changes, use focused pytest
coverage plus the private validation harnesses described in the governance
docs.

The `orbit_physics` suite covers integrators, epoch handling, eclipse geometry,
atmosphere, drag, SRP, third-body, J3/J4, spherical-harmonic, and DE440/HPOP
support through focused pytest targets. HPOP comparison workflows and MATLAB
bridge suites provide stronger external-reference checks when the required
local reference data and tooling are present; treat those as case-specific
engineering evidence.

For completed runs, prefer structured artifacts:

- `master_run_summary.json`
- `index.md`
- `master_run_history.npz` when enabled
- review-store queries when `outputs.review.enabled: true`
- validation harness reports and evidence manifests for validation runs

## Limitations

- OEL does not silently promote a scenario to a high-fidelity force stack.
  Enable each perturbation needed for the study.
- Public scenarios and tests provide reproducibility evidence for configured
  cases. They do not claim flight qualification, operational decision
  authority, or validated performance for arbitrary mission envelopes.
- TLE initialization and OEL numerical propagation are not SGP4/general
  perturbations unless an object explicitly uses the SGP4 general propagation
  path.
- Spherical harmonics need explicit terms or accessible coefficient files.
  Public distributions should not assume private HPOP/GGM03 files exist.
- Atmospheric models depend on the configured density backend, epoch,
  geodetic/frame assumptions, and solar/geomagnetic inputs. They are not a
  substitute for mission-specific atmosphere validation.
- SRP is a cannonball-style acceleration with simple area and reflectivity
  inputs unless a workflow provides attitude-aware projected areas. It does not
  model detailed material optical properties or articulating geometry by
  default.
- Re-entry diagnostics are first-pass aero/thermal estimates. They do not
  model ablation, breakup, plasma, debris, plume heating, detailed TPS
  response, or high-fidelity hypersonic aerodynamics.
- Planetary third-body support exists at the plugin/API level, but standard
  scenario booleans currently cover Sun and Moon only.
