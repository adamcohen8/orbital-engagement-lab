# Quickstart

This guide takes you from a fresh checkout to a completed headless simulation
run.

## Install

```bash
git clone https://github.com/adamcohen8/orbital-engagement-lab.git
cd orbital-engagement-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install ".[dev]"
```

## Check Your Environment

```bash
python run_simulation.py --doctor
```

Warnings for optional plotting, GUI, or game packages do not block the
headless quickstart path.

Only run scenario YAML files from sources you trust. Scenario configs can point
at importable Python modules/classes for controllers, guidance, mission
strategies, and mission execution modules; loading an untrusted scenario can run
untrusted Python code.

## Validate The Five-Minute Scenario

```bash
python run_simulation.py --quickstart --validate-only
```

Validation loads the YAML, checks timing and plugin pointers, and confirms the
scenario is structurally ready to run.

## Run The Scenario

```bash
python run_simulation.py --quickstart
```

The quickstart scenario is intentionally small and headless. It propagates a
two-satellite rendezvous setup with public controllers, sensing, and EKF
knowledge updates, then writes summary artifacts under
`outputs/quickstart_5min/`. Open `outputs/quickstart_5min/index.md` first for
the run summary, review order, and artifact inventory.

Plots are disabled in this first path to keep the first run fast, headless, and
focused on the generated summary artifacts.

To open the output folder automatically after the run:

```bash
python run_simulation.py --quickstart --open-output
```

For a guided walkthrough, see [First Five Minutes](first-five-minutes.md).

## Run The Flagship Review Scenario

After the quickstart succeeds, run the polished 10 km HCW PD rendezvous
workflow:

```bash
python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml --validate-only
python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml
```

This scenario is longer than the quickstart. It exercises a tuned HCW PD orbit
controller, attitude dynamics, reaction-wheel control, and thrust-alignment
gating, then writes review artifacts under `outputs/flagship_hcw_pd_10km/`.
Open `outputs/flagship_hcw_pd_10km/index.md` first.

For the scenario-specific review path, see
[Flagship HCW PD 10 km Scenario](flagship-hcw-pd-10km.md).

## Get Config Field Help

When editing scenario YAML, use `config_help.py` to list valid values for a
field or topic:

```bash
python config_help.py "ephemeris model"
python config_help.py "plot preset"
python config_help.py --list
```

Add `--config` to inspect the value currently set in a scenario file without
loading scenario plugins or running the simulation:

```bash
python config_help.py "ephemeris model" --config configs/hcw_pd_10km_experiment.yaml
```

The helper accepts fuzzy queries, so near-misses such as `"emphemeris model"`
still resolve to the ephemeris-mode field.

## Use The API

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/quickstart_5min.yaml")
session = SimulationSession.from_config(cfg)
result = session.run()

print(result.summary)
```

If your scenario defines `ground_stations`, the same result exposes passive
access histories:

```python
for station_id, station_payload in result.ground_station_access.items():
    for object_id, access in station_payload["targets"].items():
        print(station_id, object_id, access["access"])
```

Access is computed from line of sight, minimum elevation, and optional maximum
range. See [Scenario YAML](scenario-yaml.md) for the ground-station fields.
For more examples, see [Python API](python-api.md).

The flagship scenario also has a companion analysis script:

```bash
python examples/python/flagship_analysis.py
```

It writes custom review metrics under
`outputs/flagship_hcw_pd_10km/custom_analysis/`.

## Next Scenarios

See [Examples Matrix](examples-matrix.md) for the maintained public examples,
what each one demonstrates, and which output artifacts to inspect first.

Run the compact rendezvous example:

```bash
python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml
```

The broader plotting example writes artifacts such as:

```text
outputs/examples/public_rendezvous_closed_loop/rendezvous_summary.png
outputs/examples/public_rendezvous_closed_loop/ground_track_multi.png
outputs/examples/public_rendezvous_closed_loop/sensor_access.png
```

See [Plotting](plotting.md) for the maintained plot presets, figure IDs, and
map-backed ground-track option.

## Optional Profiles

```bash
python -m pip install ".[gui]"
python -m pip install ".[ml]"
python -m pip install ".[full]"
```

The base package already installs NumPy and Matplotlib for simulation and
plotting support. The GUI profile enables `python run_gui.py`. The ML profile
enables the bundled Gymnasium-style environments.

## Gravity Coefficient Files

The public core supports spherical-harmonic gravity from inline YAML terms or
from coefficient files you provide. HPOP/GGM03 validation data is not bundled in
the public distribution, so scenarios that set `source: "hpop_ggm03"` should also
set `coeff_path` to a local coefficient file.
