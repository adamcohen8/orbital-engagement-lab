# Scenario YAML

Scenario YAML is the main user-facing interface for repeatable simulations. A
scenario file defines the objects, algorithms, dynamics settings, outputs, and
optional analysis settings for a run.

Only run scenario YAML files from sources you trust. Scenario configs can point
at importable Python modules/classes for controllers, guidance, mission
strategies, and mission execution modules. Loading an untrusted scenario can run
untrusted Python code.

For a first pass over an untrusted scenario, use safe validation:

```bash
.venv/bin/python run_simulation.py --config <path> --safe-validate
```

Safe validation parses the YAML, applies path-policy checks, and validates plugin
pointer shape without importing configured plugin modules. It is only an
inspection aid; run scenarios only after you trust the referenced code and
paths.

For a stronger restricted profile, add sealed mode:

```bash
.venv/bin/python run_simulation.py --config <path> --sealed-mode --validate-only
```

Sealed mode uses shape-only plugin validation and then rejects plugin modules
outside OEL's trusted built-in prefixes, hosted AI calls, custom AI endpoints,
non-loopback cFS/SIL networking, and high-detail output retention unless the
caller provides an explicit `--allow-*` opt-in.

When you are unsure what values a field accepts, use the config help CLI. It
accepts fuzzy topic names, prints valid options with descriptions, and can show
the value currently set in a scenario file:

```bash
.venv/bin/python config_help.py "ephemeris model"
.venv/bin/python config_help.py "ephemeris model" --config configs/automation_smoke.yaml
.venv/bin/python config_help.py "plot preset"
.venv/bin/python config_help.py --list
```

The `--config` form parses YAML as plain data only. It does not resolve object
presets, import configured plugin classes, or run the simulation.

## Top-Level Shape

```yaml
scenario_name: "my_scenario"

objects:
  target:
    kind: "satellite"
    enabled: true
    preset: "basic_satellite"
    initial_state:
      coes:
        a_km: 7000.0
        ecc: 0.0
        inc_deg: 45.0
        raan_deg: 0.0
        argp_deg: 0.0
        true_anomaly_deg: 0.0

simulator:
  duration_s: 120.0
  dt_s: 1.0

outputs:
  output_dir: "outputs/my_scenario"
  mode: "save"
```

To station-keep a single satellite against a desired orbit, use
`OrbitalElementsStationKeepMissionStrategy` with `target_coes` and pair it with
`ControllerPointingExecution`. The current implementation maintains the desired
COE shape/orientation at the satellite's current true anomaly; it does not
target a specific phase yet. See
`configs/orbital_elements_stationkeep_smoke.yaml` for a complete single-satellite
example.

New configs should define scene participants under `objects`, keyed by object
ID. The conventional IDs `rocket`, `chaser`, and `target` still work, but they
are names rather than fixed engine slots. Legacy top-level `rocket`, `chaser`,
and `target` sections remain accepted for compatibility. If both a canonical
`objects.<id>` entry and a matching legacy alias appear, the `objects.<id>`
entry is the source of truth.

Passive ground stations can be defined at the top level. They do not control or
estimate spacecraft state; they only record access to active scene objects.

```yaml
ground_stations:
  - id: "colorado_springs"
    lat_deg: 38.803
    lon_deg: -104.526
    alt_km: 1.9
    min_elevation_deg: 10.0
    max_range_km: 2500.0
```

## Object Presets

Agents can point to reusable object preset YAML files:

```yaml
objects:
  chaser:
    kind: "satellite"
    enabled: true
    preset: "../sim/presets/objects/basic_satellite.yaml"
    specs:
      dry_mass_kg: 180.0
      fuel_mass_kg: 20.0
```

Preset paths resolve in this order:

- relative to the scenario YAML file
- relative to the current working directory
- relative to the repository root
- by name inside `sim/presets/objects`

For untrusted YAML, preset paths must stay inside the approved config/project
roots. Absolute paths, `~`, and traversal outside those roots require the caller
to opt into trusted external config paths.

This means built-in names work directly:

```yaml
objects:
  target:
    kind: "satellite"
    enabled: true
    preset: "basic_satellite"
```

Built-in satellite object presets include `basic_satellite`, `cubesat_6u`,
`smallsat_rpo`, `target_bus_passive`, `electric_prop_smallsat`, and
`adcs_demo_sat`.

Scenario-local values override preset values. Nested dictionaries, such as
`specs.mass_properties`, merge recursively.

If a scenario overrides with `specs.mass_kg` and does not provide
`dry_mass_kg` or `fuel_mass_kg`, preset dry/fuel masses are ignored for that
agent so the explicit total mass is honored.

## Mass Properties

Satellite and rocket objects can define simulator-ready mass properties under
`specs.mass_properties`. OEL uses `inertia_kg_m2` for coupled attitude dynamics,
and retains source metadata for twin-building and review workflows:

```yaml
objects:
  chaser:
    kind: "satellite"
    enabled: true
    specs:
      mass_kg: 200.0
      mass_properties:
        mass_kg: 200.0
        center_of_mass_body_m: [0.0, 0.0, 0.0]
        inertia_kg_m2:
          - [12.0, 0.1, 0.0]
          - [0.1, 10.0, 0.0]
          - [0.0, 0.0, 8.0]
        inertia_reference_point: center_of_mass
        frame: body
        source: user_supplied
        confidence: high
```

Strict validation checks that inertia matrices are finite, symmetric,
positive-definite, and satisfy principal-moment triangle inequalities. It also
validates center-of-mass vectors and the source/confidence vocabulary. If no
inertia is supplied, OEL keeps the existing default inertia behavior for
backward compatibility; if an invalid inertia is supplied explicitly, validation
and runtime reject it instead of silently falling back.

To convert a simple CAD-exported mass-property JSON/YAML file into an OEL
snippet and audit report:

```bash
.venv/bin/python tools/import_mass_properties.py cad_mass_properties.json \
  --output assets/twins/my_sat/mass_properties.yaml \
  --report outputs/my_sat_mass_properties_report.md \
  --summary
```

For a reusable object-level bundle, collect geometry profiles, mass properties,
source evidence, assumptions, and generated validation output in a spacecraft
twin package. See `docs/spacecraft-twin-packages.md` and
`examples/twins/demo_sat/twin.yaml`.

## Actuator Presets

Satellite specs can opt into the public actuator stack with
`specs.actuator_preset` or `specs.actuators.preset`:

```yaml
objects:
  chaser:
    kind: "satellite"
    enabled: true
    specs:
      mass_kg: 250.0
      actuator_preset: BASIC_RCS_6DOF
```

Available actuator presets are `BASIC_RCS_6DOF`,
`BASIC_ELECTRIC_PROPULSION`, `BASIC_MAGNETORQUER_TRIAD`, `BASIC_CMG_TRIAD`,
and `BASIC_GIMBALED_THRUSTER`. `BASIC_RCS_6DOF` is the full six-axis RCS
cluster preset: its geometry is tested for independent force and torque
authority along body X, Y, and Z. When using `specs.actuators.preset`, nested
fields in the same `actuators` block override the preset:

```yaml
specs:
  mass_kg: 250.0
  actuators:
    preset: BASIC_ELECTRIC_PROPULSION
    orbital:
      electric_propulsion:
        max_thrust_n: 0.25
```

Strict plugin validation also validates actuator preset names and core actuator
schema fields before the simulation starts. Scalar-or-vector actuator fields can
use either a single scalar value or a three-element vector, matching the runtime
actuator stack.

## Satellite Initial State

Satellite objects can initialize their orbit from ECI position/velocity,
classical orbital elements, or a TLE.

TLE example:

```yaml
simulator:
  # Optional. When set, TLE mean anomaly is advanced from the TLE epoch to this
  # Julian date using two-body mean motion.
  initial_jd_utc: 2460310.75

objects:
  target:
    kind: "satellite"
    enabled: true
    initial_state:
      tle:
        line1: "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9005"
        line2: "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1000"
```

Equivalent list form:

```yaml
initial_state:
  tle:
    lines:
      - "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9005"
      - "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1000"
```

By default, if `simulator.initial_jd_utc` is set, the TLE state is propagated to
that initial epoch. Set `propagate_to_initial_epoch: false` under `tle` to use
the TLE epoch directly. Set `require_checksum: true` to reject TLE lines with
invalid checksum digits.

The built-in TLE initializer is dependency-free and converts TLE mean elements
to an ECI state with a Keplerian/two-body approximation. Subsequent propagation
uses the configured OEL numerical special-perturbations force model. It does
not perform SGP4/general-perturbations propagation or reuse TLE-specific
drag/perturbation terms.

## Ground Stations

Ground stations are passive scene observers. They are useful when you want to
know when a site can see configured objects without adding a sensor, estimator,
controller, or mission behavior to the object itself.

```yaml
ground_stations:
  - id: "colorado_springs"
    lat_deg: 38.803
    lon_deg: -104.526
    alt_km: 1.9
    min_elevation_deg: 10.0
    max_range_km: 2500.0
```

Fields:

- `id`: station identifier used in output payloads.
- `lat_deg`: geodetic latitude in degrees.
- `lon_deg`: geodetic longitude in degrees.
- `alt_km`: altitude above the WGS84 ellipsoid in kilometers.
- `min_elevation_deg`: minimum elevation angle required for access.
- `max_range_km`: optional maximum slant range for access.
- `enabled`: optional boolean, default `true`.

You can also use mapping form when station IDs are more readable as keys:

```yaml
ground_stations:
  colorado_springs:
    lat_deg: 38.803
    lon_deg: -104.526
    alt_km: 1.9
    min_elevation_deg: 10.0
```

Access is true when all configured checks pass:

- geometric line of sight from the station to the object,
- elevation at least `min_elevation_deg`,
- range no more than `max_range_km`, when a maximum range is configured.

Single-run payloads include:

- `ground_station_access`: per-sample station/object histories with `access`,
  `line_of_sight`, `range_km`, `elevation_deg`, and diagnostic `reason`.
- `ground_station_access_summary`: access sample counts, access fraction,
  interval-based access duration, first/last access time, minimum range, and
  maximum elevation.

The same summary is also copied into `summary.ground_station_access_summary`
and appears in `index.md` key results when stations are configured.
Single-run artifacts also include satellite-oriented and ground-station-oriented
Markdown access reports. Report AOS/LOS times use UTC based on
`simulator.initial_jd_utc`; if no epoch is configured, the report epoch defaults
to `2026-01-01T00:00:00Z`.
Add `ground_station_access` to `outputs.plots.figure_ids` for a built-in access,
elevation, and slant-range figure. Set `outputs.plots.draw_earth_map: true`
when static ground-track figures should use a world-map background.

## Algorithm Pointers

Controllers, guidance, mission strategies, and mission execution modules are
referenced by importable Python module paths:

```yaml
orbit_control:
  kind: "python"
  module: "sim.control.orbit.zero_controller"
  class_name: "ZeroController"
  params: {}
```

File-path plugin loading is not supported in scenario YAML. Custom extensions
should live on the Python import path and be referenced with `module`.

## Dynamics And Timing

The simulator section defines run duration, step size, and dynamics models:

```yaml
simulator:
  duration_s: 600.0
  dt_s: 1.0
  resource_profile: config
  acceleration:
    mode: "off"
    warmup: false
  dynamics:
    orbit:
      model: "two_body"
      j2: false
      drag: false
    attitude:
      enabled: true
      attitude_substep_s: 0.1
```

`duration_s` must be a positive integer multiple of `dt_s`. Substeps must divide
the main time step cleanly.

`resource_profile` is optional and defaults to config behavior. Set it to
`laptop-safe`, `standard`, `aggressive`, or `off` to select the runtime resource
profile from the scenario file.

`acceleration.mode` is optional and defaults to `off`. Set it to `auto` to use
available optional numeric acceleration for supported kernels, or `numba` to
request the Numba backend explicitly. Unsupported dynamics combinations fall
back to the standard Python path. Set `acceleration.warmup: true` to compile
supported kernels before the run starts.

### Atmospheric Passes And Re-Entry

Atmospheric passes and re-entry diagnostics live under
`simulator.dynamics.reentry`. This feature does not silently change the force
model: `simulator.dynamics.orbit.drag` still controls whether atmospheric drag
and lift affect the trajectory. Re-entry tracking is active while a selected
object is below `begin_altitude_km`; summaries still record whether the object
ever entered, episode count, latest exit time, and cumulative heat load. That
means a vehicle can dip below the threshold for an aero-assisted pass, burn back
above it, and no longer be considered currently in re-entry.

The metric history records density, relative atmospheric speed, dynamic
pressure, drag deceleration, lift acceleration, lift-to-drag ratio, g-load,
Sutton-Graves heat rate, and integrated heat load.

```yaml
simulator:
  dynamics:
    orbit:
      drag: true
      atmosphere_model: "harris_priester"
    reentry:
      enabled: true
      begin_altitude_km: 300.0
      object_ids: ["chaser"]
      atmosphere_model: "ussa1976"
      termination:
        enabled: true
        max_g_load: 8.0
        max_dynamic_pressure_pa: 50000.0
        max_heat_rate_w_m2: 1000000.0
        max_heat_load_j_m2: 50000000.0
        by_object:
          launch_vehicle:
            enabled: false
          chaser:
            max_g_load: 12.0
```

`simulator.dynamics.orbit.atmosphere_model` and
`simulator.dynamics.reentry.atmosphere_model` can select deterministic local
models such as `exponential`, `ussa1976`, `msis86`, `nrlmsise00`, `jacchia70`,
`jb2006`, `jb2008`, and `harris_priester`. `msis86` and `nrlmsise00` are
source-local backends copied from MATLAB HPOP and can run without external files
by setting `f107`, `f107a`, and `ap` in the atmosphere environment. `msis86`
also accepts model-specific `msis86_f107`, `msis86_f107a`, `msis86_ap`,
`msis86_ap_a`, and optional HPOP-style `msis86_sw_path` inputs. `jacchia70` can
similarly run without external files by setting `jacchia70_f10`,
`jacchia70_f10b`, and `jacchia70_ap`. `jb2006` and
`jb2008` use local Jacchia-Bowman backends with local HPOP-style space-weather
table inputs. `harris_priester` uses the HPOP Harris-Priester coefficient table
copied into OEL source and supports MC variation through `simulator.environment`,
for example `harris_priester_f107`/`solar_flux_f107` and `harris_priester_n`.

Set each vehicle's re-entry nose geometry on its object specs with
`nose_radius_m` or `reentry_nose_radius_m`. Leave `object_ids` empty to track all
active satellites, or use `"*"`/`"all"` for explicit all-satellite tracking. Add
plot preset `reentry` or figure IDs `reentry_summary`, `reentry_aero`, and
`reentry_thermal` to render the re-entry plot suite. Use preset `aero_assist` or
figure ID `atmospheric_pass` when the run is about a recoverable atmospheric
pass rather than terminal disposal. Optional `termination` limits can stop the
run when the tracked object enters re-entry, drops below `min_altitude_km`,
exceeds dynamic pressure, drag deceleration, g-load, heat rate, or integrated
heat load thresholds. Scenario-level re-entry termination values are defaults;
`termination.by_object.<object_id>` can disable termination for a vehicle or
override individual limits while preserving the rest of the defaults.

Earth-impact termination is also object-aware:

```yaml
simulator:
  termination:
    earth_impact_enabled: true
    earth_radius_km: 6378.137
    by_object:
      launch_vehicle:
        earth_impact_enabled: false
      payload:
        earth_impact_enabled: true
```

This lets a launch vehicle, disposed stage, or atmospheric test article coexist
with satellites whose mission should continue.

Vehicle aerodynamics use a shared object-level vocabulary under `specs.aero`.
Keep physical vehicle properties there; use `simulator.dynamics.orbit.drag` to
turn aerodynamic forces on or off, `simulator.dynamics.reentry` for diagnostics
and termination limits, and `simulator.dynamics.rocket.aero` only for detailed
rocket coefficient refinements. The same area, drag, lift, nose-radius,
reference-length, and center-of-pressure terms are used by satellite
re-entry/aero-assist diagnostics and rocket ascent aero. Older flat satellite
aliases such as `specs.cd`, `specs.drag_area_m2`, `specs.cl`, and
`specs.nose_radius_m` still work and take precedence over same-named nested
`specs.aero` values when both are present.

Satellites can use first-pass atmospheric steering when `orbit.drag` is enabled
by setting object aero specs:

```yaml
objects:
  chaser:
    specs:
      aero:
        drag_area_m2: 2.0
        cd: 2.2
        cl: 0.2
        lift_area_m2: 2.0
        lift_axis_body: [0.0, 0.0, 1.0]
        nose_radius_m: 0.5
        reference_length_m: 1.0
        cp_offset_body_m: [0.0, 0.0, 0.0]
```

For attitude-dependent drag/SRP area without running a full mesh force model,
generate a local geometry profile from a body-frame STL mesh and reference the
resulting JSON in satellite specs:

```bash
.venv/bin/python tools/build_geometry_profile.py spacecraft_bus.stl \
  --output assets/geometry/spacecraft_area_profile.json \
  --samples 642 \
  --summary
```

```yaml
objects:
  chaser:
    specs:
      geometry:
        profile_path: assets/geometry/spacecraft_area_profile.json
      cr: 1.2
      aero:
        cd: 2.2
```

The profile stores body-frame incoming directions, projected area, and
projected-area-weighted center of pressure. During propagation, OEL uses the
current attitude to look up `drag_area_m2` and `srp_area_m2` for the existing
drag/SRP models, and disturbance torques use the same profile center of
pressure when drag/SRP torque is enabled. The first implementation is a local,
offline STL facet projection: it does not ray-trace self-shadowing, articulating
solar arrays, material coefficients, or full mesh-level force distribution.

Geometry profiles can also support attitude-aware drag calibration in the
dynamics orbit-determination workflow. `sim.observations fit-orbit` accepts
`--estimate state,cd_scale` with `--drag`, object `--mass-kg`, baseline `--cd`,
`--geometry-profile-path`, and an attitude source. Use
`--attitude-source observed-history` when observation rows include
`attitude_quat_bn`; use `--attitude-source modeled-inline --attitude-mode
sun_track` when the spacecraft attitude should be generated from OEL's mission
attitude stack during each OD candidate propagation. The fitted Cd is
conditional on those attitude, geometry, atmosphere, and mass assumptions.

Rocket objects can use the same object-level block for the common geometry and
baseline drag terms:

```yaml
objects:
  launch_vehicle:
    kind: "rocket"
    specs:
      aero:
        reference_area_m2: 10.0
        reference_length_m: 30.0
        cp_offset_body_m: [-2.5, 0.0, 0.0]
        cd: 0.20
```

`lift_axis_body` is mapped through the current attitude and projected
perpendicular to atmosphere-relative velocity before applying lift. This supports
aero-assisted passes, such as dipping into the atmosphere for plane-change
authority and then burning back up.

The product includes two first-pass controllers for that pattern:

```yaml
orbit_control:
  module: "sim.control.orbit.aero_assist"
  class_name: "AtmosphericPassController"
  params:
    raise_start_s: 240.0
    raise_end_s: 740.0
    prograde_accel_km_s2: 0.0003
attitude_control:
  module: "sim.control.attitude.aero_assist"
  class_name: "AtmosphericLiftAxisController"
  params:
    lift_axis_body: [0.0, 0.0, 1.0]
    desired_lift_ric: [0.0, 0.0, 1.0]
```

See `configs/aero_assisted_plane_change_demo.yaml` for the canonical checked-in
example.

If a satellite is burning during re-entry, the burn affects the propagated
position and velocity through the normal dynamics path before re-entry metrics
are sampled. The reported `drag_decel_m_s2` and `g_load` are aerodynamic
drag-load estimates only; they do not include commanded thrust acceleration,
plume heating, ablation, or higher-fidelity aero than the configured drag/lift
coefficients and attitude-coupled lift axis.

### Spherical Harmonics

Public scenarios can use inline spherical-harmonic terms directly:

```yaml
simulator:
  dynamics:
    orbit:
      spherical_harmonics:
        enabled: true
        degree: 2
        order: 0
        terms:
          - n: 2
            m: 0
            c_nm: -4.841693259705e-04
            s_nm: 0.0
            normalized: true
```

The `hpop_ggm03` source expects an explicit coefficient file path in public
distributions:

```yaml
spherical_harmonics:
  enabled: true
  degree: 8
  order: 8
  source: "hpop_ggm03"
  coeff_path: "/path/to/GGM03C.txt"
```

The private validation tree may contain HPOP reference data, but those files are
not bundled with the public core.

## Mission Recovery Analysis

Mission recovery is an optional single-run analysis that compares an object's
assessment-state orbit against its initial orbit and estimates the delta-v,
propellant, and time needed to reconstitute the original orbit. It is intended
for simulator-backed questions such as "after this burn, how hard is it to get
back?" rather than for replacing the deterministic propagation itself.

Use `goal: orbit_shape` to estimate restoring the original orbit shape, or
`goal: orbit_slot` to search same-apsis phasing opportunities that recover the
original slot within the configured tolerance. These are the supported YAML
terms; do not use `shaper` or `slot` aliases.

```yaml
analysis:
  mission_recovery:
    enabled: true
    object_id: "target"
    goal: "orbit_shape"  # or "orbit_slot"
    assessment_time_s: "final"
    slot_tolerance_deg: 1.0
    max_phasing_orbits: 5000
    planner:
      enabled: true
      modes: [min_delta_v, min_time, constrained]
      max_recovery_time_s: 86400.0
      max_recovery_delta_v_m_s: 25.0
      candidate_count: 12
      simulate_candidates: true
    propulsion:
      spacecraft_mass_kg: 100.0
      isp_s: 220.0
      max_thrust_n: 20.0  # optional; used to estimate burn duration_s
    element_tolerances:
      a_km: 1.0
      ecc: 0.001
```

When `outputs.review.enabled: true`, the run writes
`mission_recovery_summary` and `mission_recovery_elements` tables in
`review/run.sqlite`, plus scalar mission-recovery metrics in `metrics`.
When `planner.enabled: true`, it also writes candidate trade-space rows in
`mission_recovery_candidates`, burn rows in `mission_recovery_burns`, and
expected candidate elements in `mission_recovery_candidate_elements`. Planner
candidates are deterministic two-body estimates for comparing time/fuel trade
options. If `max_thrust_n` and mass are available, candidate burn rows include
an impulsive-equivalent burn duration. Planner rows are not operational flight
plans without separate validation for
the requested fidelity and constraints.

To save a planner trade-space PNG, enable static plots and request the figure
ID:

```yaml
outputs:
  plots:
    enabled: true
    figure_ids: [mission_recovery_trade_space]
```

## Outputs

```yaml
outputs:
  output_dir: "outputs/my_scenario"
  mode: "save"
  stats:
    enabled: true
    save_json: true
    save_full_log: true
    # Optional binary NumPy archive for scalable history analysis.
    save_history_npz: false
  plots:
    enabled: true
    preset: "minimal"
    style: "oel_dark"
    # Other presets: orbit, rendezvous, attitude, estimation, rocket, reentry, debug.
  animations:
    enabled: false
    # Optional override; defaults to outputs.plots.style.
    style: "oel_dark"
  review:
    enabled: false
    detail: "standard"
  resource_limits:
    # Configs may lower this cap; raise it from the caller with
    # --max-history-memory-mb or OEL_MAX_HISTORY_MEMORY_MB.
    max_history_memory_mb: 1024
    checkpoint_enabled: true
    throttle_enabled: true
```

Config-controlled output directories are bounded by the path policy. Relative
paths under the project/output roots work by default; external absolute paths or
directory escapes require explicit trust from the CLI or Python API.

Saved single-run plots and animations use the OEL artifact style by default. Set
`outputs.plots.style` to `oel_dark` for screen/demo artifacts, `oel_light` for
print-friendly report figures, or `matplotlib` to use unbranded Matplotlib
defaults. Saved animations inherit that plot style unless
`outputs.animations.style` is set. Branded artifacts include a public-safe
footer with the OEL version, scenario name, artifact ID, and generation
timestamp.

Set `outputs.review.enabled: true` to write a durable SQLite review store under
`outputs.output_dir/review/`. The initial single-run review store writes
`review/run.sqlite` and `review/schema.json` with normalized metadata, object
state, primary-pair relative state, thrust, ground-access, metric, event, and
artifact tables. See [Review Store Contract](review-store.md) for the current
schema and OEL Evidence Studio direction.

Set `outputs.stats.save_history_npz: true` to write
`master_run_history.npz`, a compressed NumPy archive containing time histories
and an embedded manifest. This is useful for long runs where downstream Python
analysis should avoid parsing `master_run_log.json` history lists.

Before allocating dense in-memory histories, single-run execution estimates the
history arrays required by `duration_s`, `dt_s`, active objects, rocket metrics,
and tracked knowledge pairs. The default caller-controlled cap is 1024 MB. Set
`outputs.resource_limits.max_history_memory_mb` to a smaller scenario-specific
budget, or raise the process budget with `--max-history-memory-mb` /
`OEL_MAX_HISTORY_MEMORY_MB` for intentionally large trusted runs.

For Monte Carlo and validation-style batch runs, `simulator.resource_profile`
controls the resource profile. `laptop-safe` forces one case at a time, disables
plots when the scenario is launched through `run_simulation.py` or the validation
harness, enables checkpoint/resume, and pauses between cases when memory or CPU
load pressure is high. The legacy
`outputs.resource_limits.resource_profile` field is still accepted for older
configs, but new configs should use `simulator.resource_profile`. Use
Private campaign workflows can estimate requirements before long campaign runs.
Monte Carlo checkpoints are stored under `outputs.output_dir/mc_checkpoints`
and keyed by generated iteration config hashes. Remove that checkpoint directory
when you intentionally want a clean private campaign rerun.
Sensitivity-study checkpoints are stored under
`outputs.output_dir/sensitivity_checkpoints` and use the same generated-config
hash rule. Resource preflight estimates sensitivity run count for
one-at-a-time, LHS, and two-parameter grid studies before launch.

Monte Carlo relative-range time-series plots are written through a bounded
streaming artifact writer. This avoids retaining every run's range history in
campaign memory. The plot follows `save_histograms` / `display_histograms` by
default and can be controlled explicitly:

```yaml
outputs:
  monte_carlo:
    save_relative_range_timeseries: true
    display_relative_range_timeseries: false
    relative_range_max_runs: 200
    relative_range_max_points_per_run: 1000
```

Use `configs/automation_smoke.yaml` for the smallest headless example and
`configs/simulation_template.yaml` for the broader reference template.
