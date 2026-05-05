# Python API

Use the Python API when you want to load scenarios, run simulations, inspect
payloads, or build analysis scripts without shelling out to the CLI.

The API follows the same scenario YAML contract as the CLI. Only run scenario
YAML from sources you trust; plugin pointers can import Python modules.

## Run A Scenario

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/quickstart_5min.yaml")
session = SimulationSession.from_config(cfg)
result = session.run()

print(result.summary["scenario_name"])
print(result.summary["samples"])
```

The `SimulationResult` wraps the run payload and exposes the most common
fields as properties.

```python
print(result.time_s[:5])
print(result.truth.keys())
print(result.applied_thrust.keys())
```

## Inspect A Snapshot

For single-run scenarios, snapshots provide one indexed view of truth, belief,
and applied commands.

```python
snap = result.snapshot(0)

print(snap["time_s"])
print(snap["truth"].keys())
```

Snapshot fields are useful for notebooks and small scripts. For reusable tools,
prefer summary fields and artifact maps documented in the payload contract.

## Step A Session

Use stepping when you need interactive control, custom loop logic, or live
inspection.

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/quickstart_5min.yaml")
session = SimulationSession.from_config(cfg)

snap = session.reset()
while not session.done:
    snap = session.step()
    print(snap["step_index"], snap["time_s"])
```

Calling `step()` after completion returns the final snapshot.

## Ground-Station Access

When a scenario defines `ground_stations`, the result exposes passive access
histories.

```python
cfg = SimulationConfig.from_yaml(
    "examples/configs/public_ground_station_access_from_tle.yaml"
)
result = SimulationSession.from_config(cfg).run()

summary = result.summary["ground_station_access_summary"]
for station_id, station_summary in summary.items():
    print(station_id, station_summary)

for station_id, station_payload in result.ground_station_access.items():
    for object_id, access in station_payload["targets"].items():
        print(station_id, object_id, access["access"][:5])
```

Ground-station access is passive. It does not modify truth, belief, control,
knowledge, or termination.

## Artifacts

Output artifacts are recorded in the result summary. Use artifact maps instead
of constructing filenames from assumptions.

```python
result = SimulationSession.from_config(
    SimulationConfig.from_yaml("configs/plotting_rendezvous_demo.yaml")
).run()

print(result.summary["output_index_md"])
print(result.summary["plot_outputs"])
```

If you need arbitrary time-series analysis, configure:

```yaml
outputs:
  stats:
    save_json: true
    save_full_log: true
```

Then load `master_run_log.json` from the output directory.

## Lower-Level Execution Helper

For scripts that want the raw payload directly:

```python
from sim.execution import run_simulation_config_file

payload = run_simulation_config_file("configs/quickstart_5min.yaml")
print(payload["summary"]["scenario_name"])
```

The `SimulationSession` wrapper is the preferred public API when you need a
stable object-oriented surface.

## Extension Guidance

Supported extension paths:

- scenario YAML
- object presets
- importable plugin pointers
- public controller or mission interfaces
- `SimulationSession`

Avoid importing private helpers or classes whose names start with `_`. Those
are implementation details and may change while the project is pre-1.0.

Related docs:

- [Scenario YAML](scenario-yaml.md)
- [Payload And Artifact Contract](contracts/payload-artifact-contract.md)
- [Engine Contract](contracts/engine-contract.md)
