# Orbital Engagement Lab

Open-core spacecraft simulation framework for closed-loop rendezvous, proximity
operations, attitude control, sensing, estimation, and mission prototyping.

This public repository contains the inspectable simulation foundation: core
dynamics, controllers, estimators, config loading, single-run execution, API
workflows, examples, and starter validation tools.

## What This Public Core Includes

- deterministic step-based simulation
- multi-object orbit and attitude dynamics
- two-body, perturbation, atmosphere, SRP, third-body, and spherical harmonics support
- actuator limits, saturation, lag, and mass depletion
- relative sensing and object-knowledge primitives
- orbit and attitude estimators
- orbit and attitude controller interfaces and reference controllers
- YAML-backed scenario configuration
- Python API, CLI, GUI entrypoints, and examples
- basic validation harnesses and HPOP comparison helpers
- starter cFS/SIL integration mock

## What Is Not Included

Some advanced/product workflows are intentionally not part of the public core:

- controller-benchmark suites and leaderboards
- optimization and gain-tuning workflows
- Monte Carlo and sensitivity campaign orchestration
- campaign dashboards, baselines, and review-ready reports
- curated validation and mission-assurance scenario packs

Those capabilities live in the private/product distribution.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install .
```

Optional profiles:

```bash
python -m pip install ".[dev]"
python -m pip install ".[gui]"
python -m pip install ".[ml]"
python -m pip install ".[full]"
```

## Quick Start

Validate a scenario:

```bash
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
```

Run it:

```bash
python run_simulation.py --config configs/automation_smoke.yaml
```

Use the API:

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/automation_smoke.yaml")
session = SimulationSession.from_config(cfg)
result = session.run()

print(result.summary["scenario_name"])
```

Open the GUI:

```bash
python -m pip install ".[gui]"
python run_gui.py
```

## Validation Data

The public repo includes lightweight validation assets. Large or
redistribution-sensitive ephemeris data is not stored in Git. If you run
DE440-backed HPOP parity checks, place `DE440Coeff.mat` locally at:

```text
validation/data/DE440Coeff.mat
```

## Project Layout

- `sim/core/` kernel, models, scheduling
- `sim/config/` config schema, fidelity profiles, plugin validation
- `sim/api.py` public programmatic API
- `sim/dynamics/` orbit and attitude dynamics
- `sim/actuators/` actuator models
- `sim/sensors/` sensor models
- `sim/estimation/` EKF/UKF and joint-state estimation
- `sim/control/` orbit and attitude control
- `sim/knowledge/` object knowledge tracking
- `sim/mission/` mission modules and executive patterns
- `sim/gui/` native desktop GUI
- `sim/rocket/` ascent/rocket components
- `integrations/cfs_sil/` cFS/SIL starter integration
- `validation/` validation harnesses and comparison scripts
- `examples/` runnable demos

## Scope And Safety

This project is intended for research, prototyping, pre-flight engineering
analysis, and software-in-the-loop experimentation. It is not flight-qualified
software and should not be treated as operational decision-grade without
independent validation for the relevant mission envelope.

## License

Apache License 2.0. See `LICENSE.txt`.
