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

## Outputs

```yaml
outputs:
  output_dir: "outputs/my_scenario"
  mode: "save"
  stats:
    enabled: true
    save_json: true
  plots:
    enabled: false
  animations:
    enabled: false
```

Use `configs/automation_smoke.yaml` for the smallest headless example and
`configs/simulation_template.yaml` for the broader reference template.
