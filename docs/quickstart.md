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

Plots are disabled in this first path so local Matplotlib/NumPy installation
issues cannot block the first successful run.

To open the output folder automatically after the run:

```bash
python run_simulation.py --quickstart --open-output
```

For a guided walkthrough, see [First Five Minutes](first-five-minutes.md).

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

## Next Scenarios

- `examples/configs/public_rendezvous_closed_loop.yaml` demonstrates closed-loop rendezvous with orbit control, attitude pointing, sensing, EKF knowledge, and plots.
- `examples/configs/public_orbit_environment_stack.yaml` demonstrates deterministic high-fidelity orbit/environment propagation.
- `examples/configs/public_manual_engagement.yaml` demonstrates the manual/game scenario shape.

Run the rendezvous example:

```bash
python run_simulation.py --config examples/configs/public_rendezvous_closed_loop.yaml
```

Expected plot artifacts include:

```text
outputs/examples/public_rendezvous_closed_loop/rendezvous_summary.png
outputs/examples/public_rendezvous_closed_loop/ground_track_multi.png
outputs/examples/public_rendezvous_closed_loop/sensor_access.png
```

## Optional Profiles

```bash
python -m pip install ".[gui]"
python -m pip install ".[ml]"
python -m pip install ".[full]"
```

The GUI profile enables `python run_gui.py`. The ML profile enables the bundled
Gymnasium-style environments.

## Gravity Coefficient Files

The public core supports spherical-harmonic gravity from inline YAML terms or
from coefficient files you provide. HPOP/GGM03 validation data is not bundled in
the public distribution, so scenarios that set `source: "hpop_ggm03"` should also
set `coeff_path` to a local coefficient file.
