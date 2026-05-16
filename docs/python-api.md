# Python API

Use the Python API when you want to load scenarios, run simulations, inspect
payloads, or build analysis scripts without shelling out to the CLI.

The API follows the same scenario YAML contract as the CLI. Only run scenario
YAML from sources you trust; plugin pointers can import Python modules.

## Workspace Facade

Use `SimulationWorkspace` when you want CLI-style workflows from Python.

```python
from sim import SimulationWorkspace

workspace = SimulationWorkspace()

validation = workspace.validate("configs/quickstart_5min.yaml")
if not validation["ok"]:
    raise RuntimeError(validation["errors"])

result = workspace.run("configs/quickstart_5min.yaml")
print(result.summary["scenario_name"])
```

The private Pro distribution adds controller-bench, campaign, and saved-output
report workflows on top of this facade. Public-core Python workflows are
limited to deterministic single-run scenarios.

## Custom Metrics

`SimulationResult` exposes helpers for custom analysis that is too specific for
built-in simulator metrics.

```python
import numpy as np

from sim import SimulationWorkspace

workspace = SimulationWorkspace()
result = workspace.run("configs/my_deploy_case.yaml")

ric = result.relative_state("deployed_sat", "carrier_sat", frame="ric_rect")
range_after_10_min = result.range_between(
    "deployed_sat",
    "carrier_sat",
    start_s=600.0,
)

print(ric[0])
print(float(np.min(range_after_10_min)))
```

For one-off analysis, pass metric callbacks:

```python
def collision_metric(result):
    ranges = result.range_between("deployed_sat", "carrier_sat", start_s=600.0)
    min_range_km = float(np.min(ranges))
    return {
        "min_range_after_10_min_km": min_range_km,
        "collision_after_10_min": bool(min_range_km < 0.01),
    }

metrics = result.evaluate_metrics([collision_metric])
print(metrics)
```

Numeric custom metrics are summarized with count, mean, min, max, and
percentiles. Boolean metrics are summarized as true/false counts and
`probability_true` by Pro campaign helpers. In the public core,
`result.evaluate_metrics(...)` returns the per-run callback outputs directly.

Common event helpers are available directly on `SimulationResult`:

```python
pair = result.primary_pair
print(result.object_ids)
print(result.reference_object_id)

min_range_m = result.min_range("deployed_sat", "carrier_sat", units="m")
t_ca_s = result.time_of_min_range("deployed_sat", "carrier_sat")

collision = result.collision_event(
    "deployed_sat",
    "carrier_sat",
    radius_km=0.01,
    start_s=600.0,
)

violations = result.keepout_violations(
    "deployed_sat",
    "carrier_sat",
    radius_km=0.1,
)
```

## DataFrames And Records

Time histories can be exported as record lists. If `pandas` is installed, the
`to_dataframe` helpers return a DataFrame; otherwise they return the same
record-list shape.

```python
truth = result.to_records("truth", object_id="deployed_sat")
truth_df = result.to_dataframe("truth", object_id="deployed_sat")

relative_df = result.relative_dataframe(
    "deployed_sat",
    "carrier_sat",
    frame="ric_rect",
)
```

## Config Mutations

Use `SimulationConfig` mutation helpers for small what-if studies:

```python
cfg = workspace.load("configs/my_deploy_case.yaml")
fast = cfg.with_value(
    "deployed_sat.initial_state.relative_to_target_ric.state[4]",
    0.001,
)
fast = fast.with_output_dir("outputs/deploy_fast")
result = workspace.run(fast)
```

For a complete public script using this API surface, run:

```bash
python examples/python/flagship_analysis.py
```

It runs `configs/hcw_pd_10km_experiment.yaml` and writes a custom metrics JSON
and CSV package under `outputs/flagship_hcw_pd_10km/custom_analysis/`.

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

print(snap.time_s)
print(snap.truth.keys())
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
    print(snap.step_index, snap.time_s)
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

Single-run artifacts also include two Markdown access reports when access data
exists: one organized by satellite and one organized by ground station. Report
AOS/LOS times are UTC based on `simulator.initial_jd_utc`, defaulting to
`2026-01-01T00:00:00Z` when no epoch is configured.

Ground-station access is passive. It does not modify truth, belief, control,
knowledge, or termination.

Add `ground_station_access` to `outputs.plots.figure_ids` for the built-in
access/elevation/range figure, and set `outputs.plots.draw_earth_map: true`
when static ground tracks should use a world-map background.

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
print(payload["run"]["scenario_name"])
```

The `SimulationSession` wrapper is the preferred public API when you need a
stable object-oriented surface.

See [Plotting](plotting.md) for the current single-run figure ID catalog and
the public plotting helper functions.

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
