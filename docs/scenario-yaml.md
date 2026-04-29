# Scenario YAML

Scenario YAML is the main user-facing interface for repeatable simulations. A
scenario file defines the objects, algorithms, dynamics settings, outputs, and
optional analysis settings for a run.

## Top-Level Shape

```yaml
scenario_name: "my_scenario"

target:
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

The common object sections are `rocket`, `chaser`, and `target`. Disabled
sections can be omitted or set with `enabled: false`.

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
chaser:
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

This means built-in names work directly:

```yaml
target:
  enabled: true
  preset: "basic_satellite"
```

Scenario-local values override preset values. Nested dictionaries, such as
`specs.mass_properties`, merge recursively.

If a scenario overrides with `specs.mass_kg` and does not provide
`dry_mass_kg` or `fuel_mass_kg`, preset dry/fuel masses are ignored for that
agent so the explicit total mass is honored.

## Satellite Initial State

Satellite objects can initialize their orbit from ECI position/velocity,
classical orbital elements, or a TLE.

TLE example:

```yaml
simulator:
  # Optional. When set, TLE mean anomaly is advanced from the TLE epoch to this
  # Julian date using two-body mean motion.
  initial_jd_utc: 2460310.75

target:
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
to an ECI state with a Keplerian/two-body approximation. It does not perform
full SGP4 propagation or model TLE-specific drag/perturbation terms.

## Ground Stations

Ground stations are passive scene observers. They are useful when you want to
know when a site can see a rocket, target, or chaser without adding a sensor,
estimator, controller, or mission behavior to the object itself.

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

## Outputs

```yaml
outputs:
  output_dir: "outputs/my_scenario"
  mode: "save"
  stats:
    enabled: true
    save_json: true
  plots:
    enabled: true
    preset: "minimal"
    # Other presets: orbit, rendezvous, attitude, estimation, rocket, debug.
  animations:
    enabled: false
```

Use `configs/automation_smoke.yaml` for the smallest headless example and
`configs/simulation_template.yaml` for the broader reference template.
