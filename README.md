# Orbital Engagement Lab

[![CI](https://github.com/adamcohen8/orbital-engagement-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/adamcohen8/orbital-engagement-lab/actions/workflows/ci.yml)

Open-core Python simulator for closed-loop spacecraft rendezvous and
proximity-operations prototyping. Define a scenario in YAML, run deterministic
single-run simulations through the CLI or API, and inspect generated
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

## Contents

- [Who This Is For](#who-this-is-for)
- [First Run](#first-run)
- [Use OEL With AI Coding Agents](#use-oel-with-ai-coding-agents)
- [Just Here For The Video Game?](#just-here-for-the-video-game)
- [Use-Case Cookbook](#use-case-cookbook)
- [What This Public Core Includes](#what-this-public-core-includes)
- [Security And Procurement](#security-and-procurement)
- [Pro Layer](#pro-layer)
- [Start Here](#start-here)
- [Curated Examples](#curated-examples)
- [Install Profiles](#install-profiles)
- [Project Layout](#project-layout)
- [License](#license)

A checked-in OEL-styled dashboard from the flagship 10 km RIC_PD RPO scenario:

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

Use Python 3.10 through 3.12. The commands below use Python 3.11; replace
`python3.11` with `python3.10` or `python3.12` if that is your installed
interpreter.

```bash
git clone https://github.com/adamcohen8/orbital-engagement-lab.git
cd orbital-engagement-lab
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install ".[dev]"
```

The commands below use `.venv/bin/python` so they work even on systems where
`python` is not on `PATH`. If you activate the virtual environment first,
`python` is equivalent.

Check the environment:

```bash
.venv/bin/python run_simulation.py --doctor
```

Validate the five-minute quickstart scenario:

```bash
.venv/bin/python run_simulation.py --quickstart --validate-only
```

Run it:

```bash
.venv/bin/python run_simulation.py --quickstart
```

Expected result: the run completes headlessly and writes summary artifacts under
`outputs/quickstart_5min/`. Open `outputs/quickstart_5min/index.md` first.
Plots are disabled on this first path to keep the first run fast, headless, and
focused on the generated summary artifacts.

Success looks like:

```text
Scenario : quickstart_5min
Samples  : 37
Output   : outputs/quickstart_5min
Start Here: outputs/quickstart_5min/index.md
```

To open the output folder automatically after the run:

```bash
.venv/bin/python run_simulation.py --quickstart --open-output
```

For a guided walkthrough, see [First Five Minutes](docs/first-five-minutes.md).

## Use OEL With AI Coding Agents

The public repo includes OEL Agents v0, a checked-in and tested workflow for AI
coding assistants such as Codex, Cursor, Claude Code, Gemini CLI, and Grok
Build. The core loop is: natural-language request -> scenario YAML ->
validation -> deterministic run -> review-store query or artifact inspection ->
evidence-backed answer.

Examples and task cards are onboarding and evaluation rails, not the limits of
agent support. Agents should use documented OEL interfaces to create the
minimum viable validated scenario for the user's actual question.
Use [`docs/agent-capability-routing.md`](docs/agent-capability-routing.md) to
map broad requests to public workflows, evidence, clarifying questions, and
limits.

For the fuller public workflow, point your agent at
[`AGENTS.md`](AGENTS.md), [`agents/public/AGENTS.md`](agents/public/AGENTS.md),
[`docs/oel-agents.md`](docs/oel-agents.md), and
[`docs/agent-task-cards.md`](docs/agent-task-cards.md), then try prompts like:

- "Create a short public rendezvous scenario where the chaser starts 3 km
  behind the target, validate it, run it, and summarize the output artifacts."
- "Use the TLE propagation example to explain what force models are enabled and
  whether this is SGP4 or OEL numerical propagation."
- "Review `outputs/quickstart_5min/index.md` and
  `master_run_summary.json`; tell me what happened and what limitations I
  should keep in mind."

Runs that opt into `outputs.review.enabled: true` also produce a local SQLite
review store for structured inspection. The recommended path for agents and
scripts is the SELECT-only review CLI/API, including built-in saved queries for
common agent tasks:

```bash
.venv/bin/python -m sim.review outputs/my_run --query "SELECT scenario_name, duration_s FROM run_metadata"
.venv/bin/python -m sim.review outputs/my_run --saved-query run_metadata
```

If an agent finds public-safe OEL workflow friction, it can follow
[`docs/agent-feedback-loop.md`](docs/agent-feedback-loop.md): prepare a draft,
show the user what would be sent, ask for approval, and then open an Agent
Feedback issue. Agents must not submit feedback silently.

`run_orw.py --output outputs/my_run` exists as an experimental desktop preview,
but ORW is not currently recommended for routine review workflows.

Review the flagship 10 km RIC_PD RPO scenario:

```bash
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
```

Expected result: the 12000-second rendezvous run writes OEL-styled dashboard,
rendezvous, control-effort, relative-range, trajectory, attitude,
quaternion-error, and thrust-alignment plots under
`outputs/flagship_ric_pd_10km/`. Open `outputs/flagship_ric_pd_10km/index.md`
first, then compare against the checked-in [Plot Gallery](docs/plot-gallery.md).

## Just Here For The Video Game?

![OEL RPO Trainer start screen](sim/game/assets/OEL_RPO_Trainer.png)

Clone the public repo, install the game extras, and launch the RPO trainer:

Use the same Python 3.10 through 3.12 interpreter you used for the main install.

```bash
git clone https://github.com/adamcohen8/orbital-engagement-lab.git
cd orbital-engagement-lab
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install ".[game]"
.venv/bin/python run_game.py
```

Running `run_game.py` opens the level selector. Use Up/Down or W/S to choose a
level, Left/Right to change assists, Enter or Space to launch, and Escape to
return to the selector.

## Use-Case Cookbook

| If you want to... | Run this |
| --- | --- |
| Predict where a satellite will be two hours from a TLE | `.venv/bin/python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml` |
| Compute ground-station access windows from a TLE | `.venv/bin/python run_simulation.py --config examples/configs/public_ground_station_access_from_tle.yaml` |
| Run a closed-loop chaser/target rendezvous with HCW LQR | `.venv/bin/python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml` |
| Review the flagship 10 km RIC_PD RPO scenario | `.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml` |
| Smoke-test public actuator presets | `.venv/bin/python run_simulation.py --config configs/actuator_lab_presets_smoke.yaml` |
| Review atmospheric re-entry diagnostics and kill criteria | `.venv/bin/python run_simulation.py --config configs/reentry_smoke.yaml` |
| Open the 10-day interactive re-entry plotting demo | `.venv/bin/python run_simulation.py --config examples/configs/public_reentry_interactive_demo.yaml` |
| Explore lift-axis atmospheric steering and raise-burn recovery | `.venv/bin/python run_simulation.py --config configs/aero_assisted_plane_change_demo.yaml` |
| Inspect orbit perturbations, drag, SRP, and third-body toggles | `.venv/bin/python run_simulation.py --config examples/configs/public_orbit_environment_stack.yaml` |
| Evaluate attitude hold under initial error and disturbance torque | `.venv/bin/python run_simulation.py --config examples/configs/public_attitude_hold_disturbance.yaml` |
| Open the guided RPO trainer level selector | `.venv/bin/python run_game.py` |
| Practice manual RPO/game-style control from a public config | `.venv/bin/python run_game.py examples/configs/public_manual_rpo_training.yaml` |

TLE examples use the TLE only to initialize an ECI state. OEL does not run
SGP4/general-perturbations propagation; after initialization it numerically
integrates the configured OEL force model.

Use the API:

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/quickstart_5min.yaml")
session = SimulationSession.from_config(cfg)
result = session.run()

print(result.summary["scenario_name"])
```

Experimental Output Review Workbench preview:

```bash
.venv/bin/python -m pip install ".[gui]"
.venv/bin/python run_orw.py --output outputs/quickstart_5min
```

ORW is not currently recommended for routine review. Prefer
`.venv/bin/python -m sim.review` for structured output questions.

Try the RPO trainer game mode:

```bash
.venv/bin/python -m pip install ".[game]"
.venv/bin/python run_game.py
```

Running `run_game.py` without a config opens the level selector. Pick a level
with Up/Down or W/S, toggle video recording with V or the Video button, press
Enter or Space to launch, and press Escape in a level to return to the selector.
Recordings are saved under `outputs/game_recordings/` when a level reaches
pass/fail; restarting or quitting early discards the current attempt video. The
saved MP4 includes three seconds of the level brief, three seconds of the
pass/fail screen, and the level's mapped music track looped over the video when
a track is available.
During a level, press G to start a short social clip, press G again to discard
it, or press Enter/Return to save it under `outputs/game_recordings/clips/`.
F9 remains an alternate clip key when the operating system forwards function
keys to the game.
Clips can only be started during gameplay; if a level ends while a clip is
active, save or discard it from the pass/fail screen before leaving.
If a generated clip filename already exists, OEL appends a numeric suffix
instead of overwriting the older clip.

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
.venv/bin/python run_game.py sim/game/configs/game_training_rpo_01_coast_relative_motion.yaml
```

Default trainer controls are RIC translation pulses: W/S radial, A/D in-track,
Left/Right cross-track, Space pause/resume, period single-step, R reset,
Up/Down speed, `D` to open a completed debrief folder when available, and Escape
level exit.

The public CLI is intentionally scoped to deterministic single-run
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
- atmospheric re-entry diagnostics with aero/thermal histories, plots, and
  threshold termination criteria
- shared vehicle aero properties under `objects.<id>.specs.aero`, including
  first-pass drag/lift steering and atmospheric-pass examples
- actuator limits, saturation, lag, mass depletion, full six-axis RCS clusters,
  electric propulsion, magnetorquers, CMGs, gimbaled thrusters, desaturation
  assist, and fault/degradation wrappers
- relative sensing and object-knowledge primitives
- passive ground-station access histories using line of sight, elevation, and range
- orbit and attitude estimators
- orbit and attitude controller interfaces and reference controllers
- YAML-backed scenario configuration with reusable object presets
- Python API, CLI, experimental ORW preview, and curated config examples
- Pygame RPO trainer game mode with bundled training levels
- single-run dashboards, trajectory plots, estimation plots, sensor-access plots, and access summaries
- machine-learning environment helpers
- public/private boundary and known-limitations documentation

## Security And Procurement

For vulnerability reporting, supply-chain evidence, data handling, export/CUI
boundaries, and incident response, start with:

- [Security Policy](SECURITY.md)
- [Supply Chain And Procurement Baseline](docs/security/supply-chain.md)
- [Data Handling And Boundary Statement](docs/security/data-handling.md)
- [Security Incident Process](docs/security/incident-response.md)

## Bug Reports

For non-security bugs, open a GitHub Issue and use the Bug Report template.
Maintainers monitor repository issue notifications.

Include the OEL version or commit, operating system, Python version, install
profile, exact command, minimal scenario/config or reproduction steps, expected
behavior, actual behavior, and relevant traceback or artifact path. For
simulation-model concerns, include the physical claim, scenario, reference
source or tolerance, and generated evidence.

Do not post secrets, API keys, customer data, CUI, export-controlled data,
classified information, or private generated report packets in public issues.
Report vulnerabilities or sensitive-data exposure privately through
[SECURITY.md](SECURITY.md).

## Pro Layer

- controller-benchmark suites and leaderboards
- optimization and gain-tuning workflows
- custom GNC/controller workbench scaffolding
- Monte Carlo and sensitivity campaign orchestration
- campaign dashboards, baselines, and review-ready reports
- AI-assisted report generation with user-supplied LLM API keys
- report cost estimation before hosted LLM calls
- curated validation and mission-assurance scenario packs
- custom and program-specific flight-software integration workflows

The public core is intended to be useful on its own. The pro layer is for teams
that need repeatable analysis workflows, tuning loops, campaign management, and
reporting on top of the same simulation foundation. Public examples do not
require hosted AI accounts or API keys.

## Start Here

- [Quickstart](docs/quickstart.md)
- [Product Inventory](docs/product-inventory.md)
- [Flagship RIC_PD 10 km Scenario](docs/flagship-ric-pd-10km.md)
- [RIC_PD 10 km Validation Package](docs/validation-ric-pd-10km.md)
- [Scenario YAML](docs/scenario-yaml.md)
- [Python API](docs/python-api.md)
- [OEL Agents](docs/oel-agents.md)
- [Examples Matrix](docs/examples-matrix.md)
- [Plotting](docs/plotting.md)
- [Plot Gallery](docs/plot-gallery.md)
- [Custom Analysis](docs/custom-analysis.md)
- [Public Core And Pro Boundary](docs/public-vs-pro.md)
- [Known Limitations](docs/known-limitations.md)
- [ML/RL Policy Contracts](docs/ml-rl-contracts.md)
- [Engine Contract](docs/contracts/engine-contract.md)
- [Scenario YAML Contract](docs/contracts/scenario-yaml-contract.md)
- [Payload And Artifact Contract](docs/contracts/payload-artifact-contract.md)
- [Review Store Contract](docs/review-store.md)

## Curated Examples

Curated examples are YAML scenario configs under `examples/configs/`:

- `examples/configs/public_tle_2hr_propagation.yaml` for TLE-initialized OEL numerical propagation
- `examples/configs/public_ground_station_access_from_tle.yaml` for TLE-initialized ground-station access windows
- `examples/configs/public_closed_loop_rendezvous_lqr.yaml` for closed-loop rendezvous
- `examples/configs/public_orbit_environment_stack.yaml` for perturbation/environment propagation
- `examples/configs/public_attitude_hold_disturbance.yaml` for attitude-control recovery
- `examples/configs/public_reentry_interactive_demo.yaml` for atmospheric re-entry plots and threshold termination
- `examples/configs/public_manual_rpo_training.yaml` for manual/game scenario wiring

The built-in flagship review scenario lives at
`configs/ric_pd_10km_experiment.yaml`; it is the recommended next run after the
quickstart when you want the full plot/artifact path.
Atmospheric re-entry examples live at `configs/reentry_smoke.yaml` for a short
headless check and `examples/configs/public_reentry_interactive_demo.yaml` for
a longer plotting demo with conservative satellite-kill thresholds.

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
.venv/bin/python -m pip install .
.venv/bin/python -m pip install ".[dev]"
.venv/bin/python -m pip install ".[gui]"
.venv/bin/python -m pip install ".[game]"
.venv/bin/python -m pip install ".[ml]"
.venv/bin/python -m pip install ".[full]"
```

## Project Layout

- `sim/config/` config schema, fidelity profiles, plugin validation
- `sim/api.py` public programmatic API
- `sim/single_run.py` and `sim/single_run_support.py` single-run orchestration
- `sim/core/` shared core models and scheduling utilities
- `sim/dynamics/` orbit and attitude dynamics
- `sim/actuators/` public actuator models; see `docs/actuators.md`
- `sim/sensors/` sensor models
- `sim/estimation/` EKF/UKF and joint-state estimation
- `sim/control/` orbit and attitude control
- `sim/knowledge/` object knowledge tracking
- `sim/mission/` mission modules and executive patterns
- `sim/presets/` reusable object and hardware presets
- `sim/gui/` experimental desktop Output Review Workbench preview
- `sim/rocket/` ascent/rocket components
- `machine_learning/` public environment helpers and training entrypoints
- `examples/` curated runnable configs
- `docs/` user-facing documentation

## License

Apache License 2.0. See `LICENSE.txt`.
