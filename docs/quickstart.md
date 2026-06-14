# Quickstart

This guide takes you from a fresh checkout to a completed headless simulation
run.

## Install

```bash
git clone https://github.com/adamcohen8/orbital-engagement-lab.git
cd orbital-engagement-lab
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install ".[dev]"
```

Use Python 3.10 through 3.12. Replace `python3.11` with `python3.10` or
`python3.12` if that is your installed interpreter.

The commands below use `.venv/bin/python` so they work even on systems where
`python` is not on `PATH`. If you activate the virtual environment first,
`python` is equivalent.

## Check Your Environment

```bash
.venv/bin/python run_simulation.py --doctor
```

Warnings for optional plotting, experimental GUI, or game packages do not block
the headless quickstart path.

Only run scenario YAML files from sources you trust. Scenario configs can point
at importable Python modules/classes for controllers, guidance, mission
strategies, and mission execution modules; loading an untrusted scenario can run
untrusted Python code.

If you only want to inspect a scenario from an unknown source, start with:

```bash
.venv/bin/python run_simulation.py --config <path> --safe-validate
```

That mode avoids importing configured plugin modules. It does not make the
scenario safe to execute.

For shared classroom, government, or enterprise review, validate with the
restricted profile before execution:

```bash
.venv/bin/python run_simulation.py --config <path> --sealed-mode --validate-only
```

Sealed mode blocks arbitrary plugin modules, hosted/custom AI endpoints,
non-loopback cFS/SIL networking, and high-detail output retention unless the
caller explicitly opts into the specific exception.

## Validate The Five-Minute Scenario

```bash
.venv/bin/python run_simulation.py --quickstart --validate-only
```

Validation loads the YAML, checks timing and plugin pointers, and confirms the
scenario is structurally ready to run.

## Run The Scenario

```bash
.venv/bin/python run_simulation.py --quickstart
```

The quickstart scenario is intentionally small and headless. It propagates a
two-satellite rendezvous setup with public controllers, sensing, and EKF
knowledge updates, then writes summary artifacts under
`outputs/quickstart_5min/`. Open `outputs/quickstart_5min/index.md` first for
the run summary, review order, and artifact inventory.

Plots are disabled in this first path to keep the first run fast, headless, and
focused on the generated summary artifacts. The quickstart does write
`review/run.sqlite`, so you can inspect the completed run with the review CLI:

```bash
.venv/bin/python -m sim.review outputs/quickstart_5min --saved-query run_metadata
```

To open the output folder automatically after the run:

```bash
.venv/bin/python run_simulation.py --quickstart --open-output
```

For a guided walkthrough, see [First Five Minutes](first-five-minutes.md).

## Run The Flagship Review Scenario

After the quickstart succeeds, run the polished 10 km RIC_PD rendezvous
workflow:

```bash
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
```

This scenario is longer than the quickstart. It exercises a tuned RIC_PD orbit
controller, attitude dynamics, reaction-wheel control, and thrust-alignment
gating, then writes review artifacts under `outputs/flagship_ric_pd_10km/`.
Open `outputs/flagship_ric_pd_10km/index.md` first.

For the scenario-specific review path, see
[Flagship RIC_PD 10 km Scenario](flagship-ric-pd-10km.md).

## Get Config Field Help

When editing scenario YAML, use `config_help.py` to list valid values for a
field or topic:

```bash
.venv/bin/python config_help.py "ephemeris model"
.venv/bin/python config_help.py "plot preset"
.venv/bin/python config_help.py --list
```

Add `--config` to inspect the value currently set in a scenario file without
loading scenario plugins or running the simulation:

```bash
.venv/bin/python config_help.py "ephemeris model" --config configs/ric_pd_10km_experiment.yaml
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
.venv/bin/python examples/python/flagship_analysis.py
```

It writes custom review metrics under
`outputs/flagship_ric_pd_10km/custom_analysis/`.

## Next Scenarios

See [Examples Matrix](examples-matrix.md) for the maintained public examples,
what each one demonstrates, and which output artifacts to inspect first.

Run the compact rendezvous example:

```bash
.venv/bin/python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml
```

The broader plotting example writes artifacts such as:

```text
outputs/examples/public_rendezvous_closed_loop/rendezvous_summary.png
outputs/examples/public_rendezvous_closed_loop/ground_track_multi.png
outputs/examples/public_rendezvous_closed_loop/sensor_access.png
```

See [Plotting](plotting.md) for the maintained plot presets, figure IDs, and
map-backed ground-track option.

Atmospheric re-entry diagnostics have a short public smoke config and a longer
interactive plotting demo:

```bash
.venv/bin/python run_simulation.py --config configs/reentry_smoke.yaml
.venv/bin/python run_simulation.py --config examples/configs/public_reentry_interactive_demo.yaml
```

## Optional Profiles

```bash
.venv/bin/python -m pip install ".[gui]"
.venv/bin/python -m pip install ".[ml]"
.venv/bin/python -m pip install ".[full]"
```

The base package already installs NumPy and Matplotlib for simulation and
plotting support. The GUI profile enables experimental desktop surfaces,
including the Output Review Workbench dynamic plot creator for completed runs.
The recommended public onboarding path remains CLI/YAML/Python API, and the
scripted review path remains `.venv/bin/python -m sim.review`. The ML profile
enables the bundled Gymnasium-style environments.

## Gravity Coefficient Files

The public core supports spherical-harmonic gravity from inline YAML terms or
from coefficient files you provide. HPOP/GGM03 validation data is not bundled in
the public distribution, so scenarios that set `source: "hpop_ggm03"` should also
set `coeff_path` to a local coefficient file.
