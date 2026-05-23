# Examples Matrix

Curated examples are YAML configs under `examples/configs/`. They are intended
to be validated and run through the standard CLI.

Run validation first:

```bash
python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml --validate-only
```

Then run the scenario:

```bash
python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml
```

## Public Examples

| Config | What It Demonstrates | Main Outputs | Notes |
| --- | --- | --- | --- |
| `public_tle_2hr_propagation.yaml` | TLE initialization and two-hour propagation | summary JSON, trajectory histories | Uses the built-in two-body TLE approximation, not full SGP4. |
| `public_ground_station_access_from_tle.yaml` | Passive station access from a TLE object | access summary, access/elevation/range plot, and map-backed ground track | Good starting point for line-of-sight, elevation, and range checks. |
| `public_closed_loop_rendezvous_lqr.yaml` | Compact closed-loop chaser/target rendezvous with HCW LQR | run summary and rendezvous metrics | Fastest controller example for reading the YAML shape. |
| `public_rendezvous_closed_loop.yaml` | Broader rendezvous with attitude pointing, sensing, EKF knowledge, and plots | dashboard, rendezvous, control, estimation, sensor-access, and ground-track plots | Best public example for end-to-end closed-loop artifact review. |
| `public_orbit_environment_stack.yaml` | Perturbation and environment toggles | summary JSON and optional plots | Use to inspect deterministic force-model configuration. |
| `public_attitude_hold_disturbance.yaml` | Attitude hold under initial error and disturbance torque | attitude histories and control artifacts | Requires no GUI; useful for attitude-control sanity checks. |
| `public_manual_rpo_training.yaml` | Manual/game-style RPO scenario wiring | game/manual-control-compatible config | Can be launched with `run_game.py` when the `game` extra is installed. |
| `public_manual_engagement.yaml` | Manual engagement-style scenario with knowledge and defensive behavior | run outputs or game-mode behavior depending on entrypoint | More advanced than the first manual training config. |

## Built-In Non-Example Configs

| Config | Purpose | Normal Command |
| --- | --- | --- |
| `configs/quickstart_5min.yaml` | Fast first-run smoke scenario | `python run_simulation.py --quickstart` |
| `configs/hcw_pd_10km_experiment.yaml` | Flagship 10 km HCW PD RPO review scenario with attitude-gated thrust | `python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml` |
| `configs/automation_smoke.yaml` | Small headless validation config | `python run_simulation.py --config configs/automation_smoke.yaml --validate-only` |
| `configs/plotting_rendezvous_demo.yaml` | Standalone rendezvous plotting demo | `python run_simulation.py --config configs/plotting_rendezvous_demo.yaml` |

## Game Levels

The optional Pygame trainer includes packaged levels under `sim/game/configs/`.
The launcher progression starts with `rpo_00_tutorial`, then runs through
Levels 1-10 and a replayable `rpo_arcade_pursuit` variant. Levels 6-8 use an
elliptical coast-projection model for eccentric-orbit RPO lessons; Pursuit
Arcade adds tightening goals, randomized later-round starts, and every-fifth
round elliptical boss encounters. Completed structured training levels save
Markdown debrief reports with JSON summaries and matplotlib plots under
`outputs/game_debriefs/`; Sandbox and Pursuit Arcade skip reports because they
are open-ended/replayable modes.

```bash
python -m pip install ".[game]"
python run_game.py
```

Running `run_game.py` without a config opens the level selector. Direct launch
also works:

```bash
python run_game.py sim/game/configs/game_training_rpo_04_rendezvous.yaml
```

## Choosing A Starting Point

- New user: start with `configs/quickstart_5min.yaml`.
- Flagship review workflow: run `configs/hcw_pd_10km_experiment.yaml`, then
  `examples/python/flagship_analysis.py`.
- TLE propagation: start with `public_tle_2hr_propagation.yaml`.
- Ground-station access: start with `public_ground_station_access_from_tle.yaml`.
- Closed-loop control: start with `public_closed_loop_rendezvous_lqr.yaml`.
- Plot/artifact review: start with `configs/hcw_pd_10km_experiment.yaml`.
- Guided RPO trainer: start with `run_game.py`.
- Manual RPO config wiring: start with
  `public_manual_rpo_training.yaml`.

Public examples use the canonical `objects` map. Names such as `chaser` and
`target` are readable scenario IDs, not fixed engine slots.
