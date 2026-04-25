# Scenario YAML Contract

This document defines the 0.1 compatibility contract for scenario YAML files.
It complements the user-facing guide in `docs/scenario-yaml.md` by stating what
scenario authors, examples, GUI workflows, tests, and migration tools may rely
on.

The contract applies to public-core single-run scenarios and to Pro/private
analysis scenarios unless a Pro-only field is explicitly called out.


## Stability Level

This is a pre-1.0 contract. The schema may still evolve, but changes that alter
existing behavior should include:

- a release-note entry,
- an update to this contract,
- a focused regression test,
- and, when practical, a migration note or compatibility shim.

Stable enough to rely on:

- top-level section names,
- strict boolean parsing,
- object preset resolution order,
- plugin pointer shape,
- timing-grid validation,
- output section shape,
- public rejection of Pro-only batch-analysis workflows.

Still maturing:

- formal schema version field,
- complete GUI coverage of advanced YAML fields,
- migration tooling for all historical config shapes,
- exhaustive invalid-combination diagnostics.


## Top-Level Sections

Recognized top-level sections:

- `scenario_name`
- `scenario_description`
- `metadata`
- `rocket`
- `chaser`
- `target`
- `simulator`
- `outputs`
- `monte_carlo`
- `analysis`

Object sections:

- `rocket`, `chaser`, and `target` are the canonical object roles.
- Disabled objects may be omitted or set with `enabled: false`.
- Enabled object sections participate in runtime creation, validation, and
  output histories.

Analysis sections:

- Public core accepts deterministic single-run scenarios.
- Pro/private workspaces support `monte_carlo` and `analysis` workflows.
- Generated public entrypoints should reject enabled Monte Carlo or sensitivity
  analysis with a clear Pro-boundary message.


## Strict Types

Boolean fields must be YAML booleans:

```yaml
enabled: true
drag: false
```

Quoted boolean-like strings are invalid:

```yaml
enabled: "true"   # invalid
drag: "false"    # invalid
```

Strict boolean parsing applies broadly to fields named or shaped like:

- `enabled`
- `strict`
- `j2`, `j3`, `j4`
- `drag`
- `srp`
- `third_body_moon`
- `third_body_sun`
- `parallel_enabled`
- keys beginning with `use_`, `save_`, `display_`, `print_`, or `require_`

Scenario authors should prefer explicit numeric values for times, distances,
masses, and tolerances. Loader behavior may coerce some values, but config files
should not rely on stringly typed numerics.


## Timing Grid

Required timing rules:

- `simulator.duration_s` must be positive.
- `simulator.dt_s` must be positive.
- `duration_s` must be an integer multiple of `dt_s`.
- `orbit_substep_s`, when present, must be positive, no larger than `dt_s`, and
  divide `dt_s` cleanly.
- `attitude_substep_s`, when present, must be positive, no larger than `dt_s`,
  and divide `dt_s` cleanly.

Analysis parameter sweeps should not vary timing fields unless every generated
sample still preserves these grid rules.


## Object Sections

Common object fields:

- `enabled`
- `role`
- `preset`, `preset_yaml`, or `preset_path`
- `specs`
- `initial_state`
- `reference_orbit`
- `guidance`
- `base_guidance`
- `guidance_modifiers`
- `orbit_control`
- `attitude_control`
- `mission_strategy`
- `mission_execution`
- `mission_objectives`
- `bridge`
- `knowledge`

Preset merge contract:

- Presets load before local object overrides.
- Local values override preset values.
- Nested dictionaries merge recursively.
- If a local override provides `specs.mass_kg` without `dry_mass_kg` or
  `fuel_mass_kg`, preset dry/fuel masses are ignored for that object so the
  explicit total mass is honored.

Preset resolution order:

1. absolute path, when provided,
2. path relative to the scenario YAML file,
3. path relative to the current working directory,
4. path relative to the repository root,
5. name inside `sim/presets/objects`,
6. same name with `.yaml` inside `sim/presets/objects`.


## Algorithm Pointers

Python plugin pointer shape:

```yaml
orbit_control:
  kind: "python"
  module: "sim.control.orbit.zero_controller"
  class_name: "ZeroController"
  params: {}
```

Pointer contract:

- `module` must be an importable Python module.
- A class pointer uses `class_name`.
- A function pointer uses `function`, but only plugin types that explicitly
  allow functions may use it.
- `params` must be a mapping.
- File-path plugin loading is not part of the scenario YAML contract.

Plugin validation checks constructor/importability and required callable
methods for supported plugin roles. Validation is intentionally structural; it
does not prove physical correctness.


## Simulator Section

Common fields:

- `scenario_type`
- `duration_s`
- `dt_s`
- `initial_jd_utc`
- `dynamics`
- `environment`
- `plugin_validation`
- `termination`

Dynamics contract:

- Orbit dynamics live under `simulator.dynamics.orbit`.
- Attitude dynamics live under `simulator.dynamics.attitude`.
- Rocket dynamics configuration lives under `simulator.dynamics.rocket`.
- Time-dependent environment behavior may use `initial_jd_utc` and environment
  ephemeris settings.

Termination contract:

- `termination.earth_impact_enabled` controls Earth-impact termination.
- `termination.earth_radius_km` defines the impact radius when enabled.


## Outputs Section

Common fields:

- `output_dir`
- `mode`
- `stats`
- `plots`
- `animations`
- `monte_carlo`
- `ai_report`

Output modes:

- `save` is the preferred headless/CI mode.
- `interactive` may open windows or require GUI-capable environments.
- In automation contexts, interactive mode may be coerced to save mode.

Stats contract:

- `stats.save_json` controls `master_run_summary.json`.
- `stats.save_full_log` controls `master_run_log.json`.
- `stats.print_summary` controls console summary printing.

Plot and animation contract:

- Plots and animations are optional artifacts.
- Disabled plots/animations should not be required for simulation correctness.
- Artifact filenames and figure IDs should be consumed through summary artifact
  mappings rather than hard-coded when possible.


## Public And Pro Fields

Public-core scenarios should use:

- deterministic single-run object sections,
- public controller/estimator/mission modules,
- public dynamics settings,
- public output settings,
- curated examples under `examples/configs`.

Pro/private scenarios may additionally use:

- `monte_carlo.enabled: true`,
- `analysis.enabled: true`,
- controller-benchmark configs,
- AI report settings,
- validation harness configs,
- HPOP/MATLAB validation configs,
- integration/SIL bridge workflows.

The generated public export may contain stub modules or boundary errors for
Pro-only features. Public configs should not require those features to run.


## Migration Rules

When the scenario contract changes:

- update `docs/scenario-yaml.md` when user-facing authoring guidance changes,
- update this contract when compatibility expectations change,
- add regression tests for old and new behavior where practical,
- update curated configs before release,
- note migration-sensitive changes in `CHANGELOG.md`.

Do not silently reinterpret old fields in a way that changes physics,
controller behavior, or output semantics without tests and release notes.


## Known Gaps

- No formal `schema_version` field is enforced yet.
- Not every supported field has a generated machine-readable schema.
- GUI editing does not yet expose every advanced config combination.
- Public/Pro field boundaries are enforced mostly through export stubs, docs,
  config curation, and runtime checks rather than a standalone schema compiler.
