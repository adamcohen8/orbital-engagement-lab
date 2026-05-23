# Orbital Engagement Lab

[![CI](https://github.com/adamcohen8/orbital-engagement-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/adamcohen8/orbital-engagement-lab/actions/workflows/ci.yml)

Open-core Python simulator for closed-loop spacecraft rendezvous and
proximity-operations prototyping. Define a scenario in YAML, run deterministic
single-run simulations through the CLI, API, or GUI, and inspect generated
summaries, plots, and artifacts.

The public core is intended for research, education, prototyping, pre-flight
engineering analysis, and software-in-the-loop experimentation. It is not
flight-qualified software and should not be treated as operational
decision-grade without independent validation for the relevant mission envelope.

## Personal-Capacity And No-Endorsement Notice

Orbital Engagement Lab is an independent personal-capacity software project. It
is not an official product, program, or endorsement of the Department of
Defense, Department of the Air Force, United States Space Force, or any other
U.S. Government organization.

The project is intended for research, education, prototyping, pre-flight
engineering analysis, software-in-the-loop experimentation, and validation
workflow development. It is not flight software and is not an operational
decision system.

Project development is intended to use public technical references, original
work, open-source dependencies, personal resources, and personal time. The
public repository should not contain nonpublic government information,
government-provided code, controlled operational scenarios, classified
information, or customer-controlled technical data.

Users are responsible for their own validation, security review, export-control
review, mission qualification, and compliance obligations before using the
software in any sensitive, commercial, government, or operational context.

A checked-in dashboard from the flagship 10 km HCW PD RPO scenario:

![Flagship run dashboard](docs/assets/plots/run_dashboard.png)

Orbital Engagement Lab exists to make it easier to prototype spacecraft behavior
as a full closed loop: orbit dynamics, attitude dynamics, sensors, estimators,
controllers, actuators, mission logic, and outputs all running from the same
scenario definition.

Orbital Engagement Pro adds workflow acceleration around that foundation:
controller benchmarking, optimization, campaign orchestration, sensitivity
studies, dashboards, AI-assisted reports, curated validation scenario packs, and
integration workflows.

## Who This Is For

- Researchers and engineers prototyping closed-loop RPO, sensing, estimation,
  control, and mission-logic behavior.
- Educators and students who want concrete spacecraft relative-motion examples
  and an approachable RPO trainer.
- Technical evaluators who want to inspect the public simulation core before
  considering Pro workflow acceleration.

## First Run

```bash
git clone https://github.com/adamcohen8/orbital-engagement-lab.git
cd orbital-engagement-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install ".[dev]"
```

Check the environment:

```bash
python run_simulation.py --doctor
```

Validate the five-minute quickstart scenario:

```bash
python run_simulation.py --quickstart --validate-only
```

Run it:

```bash
python run_simulation.py --quickstart
```

Expected result: the run completes headlessly and writes summary artifacts under
`outputs/quickstart_5min/`. Open `outputs/quickstart_5min/index.md` first.
Plots are disabled on this first path to keep the first run fast, headless, and
focused on the generated summary artifacts.

Success looks like:

```text
Scenario : quickstart_5min
Samples  : 301
Output   : outputs/quickstart_5min
Start Here: outputs/quickstart_5min/index.md
```

To open the output folder automatically after the run:

```bash
python run_simulation.py --quickstart --open-output
```

For a guided walkthrough, see [First Five Minutes](docs/first-five-minutes.md).

Review the flagship 10 km HCW PD RPO scenario:

```bash
python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml --validate-only
python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml
```

Expected result: the 6000-second rendezvous run writes dashboard, rendezvous,
control-effort, relative-range, trajectory, attitude, quaternion-error, and
thrust-alignment plots under `outputs/flagship_hcw_pd_10km/`. Open
`outputs/flagship_hcw_pd_10km/index.md` first, then compare against the
checked-in [Plot Gallery](docs/plot-gallery.md).

## Just Here For The Video Game?

Clone the public repo, install the game extras, and launch the RPO trainer:

```bash
git clone https://github.com/adamcohen8/orbital-engagement-lab.git
cd orbital-engagement-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install ".[game]"
python run_game.py
```

Running `run_game.py` opens the level selector. Use Up/Down or W/S to choose a
level, Left/Right to change assists, Enter or Space to launch, and Escape to
return to the selector.

## Use-Case Cookbook

| If you want to... | Run this |
| --- | --- |
| Predict where a satellite will be two hours from a TLE | `python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml` |
| Compute ground-station access windows from a TLE | `python run_simulation.py --config examples/configs/public_ground_station_access_from_tle.yaml` |
| Run a closed-loop chaser/target rendezvous with HCW LQR | `python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml` |
| Review the flagship 10 km HCW PD RPO scenario | `python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml` |
| Inspect orbit perturbations, drag, SRP, and third-body toggles | `python run_simulation.py --config examples/configs/public_orbit_environment_stack.yaml` |
| Evaluate attitude hold under initial error and disturbance torque | `python run_simulation.py --config examples/configs/public_attitude_hold_disturbance.yaml` |
| Open the guided RPO trainer level selector | `python run_game.py` |
| Practice manual RPO/game-style control from a public config | `python run_game.py examples/configs/public_manual_rpo_training.yaml` |

Use the API:

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/quickstart_5min.yaml")
session = SimulationSession.from_config(cfg)
result = session.run()

print(result.summary["scenario_name"])
```

Open the GUI:

```bash
python -m pip install ".[gui]"
python run_gui.py
```

Try the RPO trainer game mode:

```bash
python -m pip install ".[game]"
python run_game.py
```

Running `run_game.py` without a config opens the level selector. Pick a level
with Up/Down or W/S, toggle video recording with V or the Video button, press
Enter or Space to launch, and press Escape in a level to return to the selector.
Recordings are saved under `outputs/game_recordings/` when a level reaches
pass/fail; restarting or quitting early discards the current attempt video. The
saved MP4 includes the level's mapped music track looped over the video when a
track is available.

Structured training levels also save Markdown debrief reports under
`outputs/game_debriefs/<scenario_id>/attempt_.../`. These reports include
pass/fail status, failure reasons, summary stats, mission timeline and burn
interval figures, 2D RIC plots, relative range and velocity histories,
cumulative delta-v, and control-command plots. On the pass/fail screen, press
`D` to close the game and open the attempt folder. Sandbox and Pursuit Arcade
skip debrief reports because they are open-ended/replayable modes.

The normal public repository includes the game music so the default download has
the full trainer experience. To clone a smaller no-music copy, use a partial
clone with sparse checkout:

```bash
git clone --filter=blob:none --no-checkout https://github.com/adamcohen8/orbital-engagement-lab.git
cd orbital-engagement-lab
git sparse-checkout init --no-cone
git sparse-checkout set "/*" "!/sim/game/music/*.wav"
git checkout main
```

The bundled progression now starts with a tutorial, then covers coast-relative
motion, V-bar and R-bar approaches, close rendezvous, passively safe inspection,
eccentric-orbit approach and NMC lessons, defensive-target tracking,
evasive-target survival, and an arcade pursuit variant with tightening goals,
randomized later-round starts, elliptical boss rounds, a conserved-delta-v time
economy, boss eccentricity ramping to 0.20, and a post-round-20 target
defensive delta-v ramp.

You can also launch a level directly:

```bash
python run_game.py sim/game/configs/game_training_rpo_01_coast_relative_motion.yaml
```

Default trainer controls are RIC translation pulses: W/S radial, A/D in-track,
Left/Right cross-track, Space pause/resume, period single-step, R reset,
Up/Down speed, `D` to open a completed debrief folder when available, and Escape
level exit.

The public CLI and GUI are intentionally scoped to deterministic single-run
scenarios. Batch analysis settings are not exposed in public examples, and
configs with enabled Monte Carlo or sensitivity studies are rejected with a clear
Pro-boundary message.

Only run scenario YAML files from sources you trust. Scenario configs can point
at importable Python modules/classes for controllers, guidance, mission
strategies, and mission execution modules; loading an untrusted scenario can run
untrusted Python code.

Spherical-harmonic gravity can use inline YAML terms or coefficient files you
provide. HPOP/GGM03 validation data is not bundled in the public core, so
`source: "hpop_ggm03"` scenarios should also set `coeff_path`.

## What This Public Core Includes

- deterministic step-based simulation
- multi-object orbit and attitude dynamics
- two-body, perturbation, atmosphere, SRP, third-body, and spherical harmonics support
- actuator limits, saturation, lag, and mass depletion
- relative sensing and object-knowledge primitives
- passive ground-station access histories using line of sight, elevation, and range
- orbit and attitude estimators
- orbit and attitude controller interfaces and reference controllers
- YAML-backed scenario configuration with reusable object presets
- Python API, CLI, GUI entrypoints, and curated config examples
- Pygame RPO trainer game mode with bundled training levels
- single-run dashboards, trajectory plots, estimation plots, sensor-access plots, and access summaries
- machine-learning environment helpers
- public/private boundary and known-limitations documentation

## Pro Layer

- controller-benchmark suites and leaderboards
- optimization and gain-tuning workflows
- Monte Carlo and sensitivity campaign orchestration
- campaign dashboards, baselines, and review-ready reports
- AI-assisted report generation with user-supplied LLM API keys
- report cost estimation before hosted LLM calls
- curated validation and mission-assurance scenario packs
- cFS/SIL and program-specific flight-software integration workflows

The public core is intended to be useful on its own. The pro layer is for teams
that need repeatable analysis workflows, tuning loops, campaign management, and
reporting on top of the same simulation foundation. Public examples do not
require hosted AI accounts or API keys.

## Start Here

- [Quickstart](docs/quickstart.md)
- [Flagship HCW PD 10 km Scenario](docs/flagship-hcw-pd-10km.md)
- [Scenario YAML](docs/scenario-yaml.md)
- [Python API](docs/python-api.md)
- [Examples Matrix](docs/examples-matrix.md)
- [Plotting](docs/plotting.md)
- [Plot Gallery](docs/plot-gallery.md)
- [Custom Analysis](docs/custom-analysis.md)
- [Public Core And Pro Boundary](docs/public-vs-pro.md)
- [Known Limitations](docs/known-limitations.md)
- [Engine Contract](docs/contracts/engine-contract.md)
- [Scenario YAML Contract](docs/contracts/scenario-yaml-contract.md)
- [Payload And Artifact Contract](docs/contracts/payload-artifact-contract.md)

## Curated Examples

Curated examples are YAML scenario configs under `examples/configs/`:

- `examples/configs/public_tle_2hr_propagation.yaml` for TLE propagation
- `examples/configs/public_ground_station_access_from_tle.yaml` for ground-station access windows
- `examples/configs/public_closed_loop_rendezvous_lqr.yaml` for closed-loop rendezvous
- `examples/configs/public_orbit_environment_stack.yaml` for perturbation/environment propagation
- `examples/configs/public_attitude_hold_disturbance.yaml` for attitude-control recovery
- `examples/configs/public_manual_rpo_training.yaml` for manual/game scenario wiring

The built-in flagship review scenario lives at
`configs/hcw_pd_10km_experiment.yaml`; it is the recommended next run after the
quickstart when you want the full plot/artifact path.

New examples use the canonical `objects` map in scenario YAML. Conventional
names such as `chaser` and `target` are readable scenario IDs, not fixed engine
slots.

Ground stations are configured in scenario YAML with a top-level
`ground_stations` list or mapping. Single-run outputs include
`ground_station_access` and `ground_station_access_summary` so users can inspect
when each site has access to each active object. Add the `ground_station_access`
figure ID for a built-in access/elevation/range plot, and set
`outputs.plots.draw_earth_map: true` when you want static ground-track figures
to use a world-map background.

The public examples are intentionally config-first. Experimental Python demos
and local-artifact-dependent workflows are not part of the supported public
example surface.

## Install Profiles

```bash
python -m pip install .
python -m pip install ".[dev]"
python -m pip install ".[gui]"
python -m pip install ".[game]"
python -m pip install ".[ml]"
python -m pip install ".[full]"
```

## Project Layout

- `sim/config/` config schema, fidelity profiles, plugin validation
- `sim/api.py` public programmatic API
- `sim/single_run.py` and `sim/single_run_support.py` single-run orchestration
- `sim/core/` shared core models and scheduling utilities
- `sim/dynamics/` orbit and attitude dynamics
- `sim/actuators/` actuator models
- `sim/sensors/` sensor models
- `sim/estimation/` EKF/UKF and joint-state estimation
- `sim/control/` orbit and attitude control
- `sim/knowledge/` object knowledge tracking
- `sim/mission/` mission modules and executive patterns
- `sim/presets/` reusable object and hardware presets
- `sim/gui/` native desktop GUI
- `sim/rocket/` ascent/rocket components
- `machine_learning/` public environment helpers and training entrypoints
- `examples/` curated runnable configs
- `docs/` user-facing documentation

## License

Apache License 2.0. See `LICENSE.txt`.
