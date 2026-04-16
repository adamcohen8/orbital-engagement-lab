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

## Validate A Scenario

```bash
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
```

Validation loads the YAML, checks timing and plugin pointers, and confirms the
scenario is structurally ready to run.

## Run A Scenario

```bash
python run_simulation.py --config configs/automation_smoke.yaml
```

The smoke scenario is intentionally small and headless. It propagates one target
spacecraft with simple two-body orbit dynamics and attitude dynamics, then writes
summary artifacts under `outputs/automation_smoke/`.

## Use The API

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/automation_smoke.yaml")
session = SimulationSession.from_config(cfg)
result = session.run()

print(result.summary)
```

## Next Scenarios

- `configs/simulation_template.yaml` is the broad feature template.
- `configs/hcw_lqr_two_body_perfect.yaml` is a compact rendezvous-control case.
- `configs/plotting_rendezvous_demo.yaml` generates dashboard, rendezvous, control, estimation, and sensor plots.
- `examples/Orbit_HCW_LQR_Demo.py` demonstrates relative orbit control.
- `examples/Rendezvous_HCW_AttitudeLQR_Demo.py` demonstrates integrated orbit and attitude behavior.

Generate the plotting demo:

```bash
python run_simulation.py --config configs/plotting_rendezvous_demo.yaml
```

Expected plot artifacts include:

```text
outputs/plotting_rendezvous_demo/run_dashboard.png
outputs/plotting_rendezvous_demo/rendezvous_summary.png
outputs/plotting_rendezvous_demo/sensor_access.png
```

## Optional Profiles

```bash
python -m pip install ".[gui]"
python -m pip install ".[ml]"
python -m pip install ".[full]"
```

The GUI profile enables `python run_gui.py`. The ML profile enables the bundled
Gymnasium-style environments and training examples.
