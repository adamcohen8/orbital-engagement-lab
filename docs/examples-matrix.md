# Examples Matrix

Curated examples are YAML configs under `examples/configs/`. They are intended
to be validated and run through the standard CLI.

Run validation first:

```bash
.venv/bin/python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml --validate-only
```

Then run the scenario:

```bash
.venv/bin/python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml
```

Most TLE examples use the TLE only to initialize an ECI state, then numerically
integrate the configured OEL force model. The explicit
`public_sgp4_passive_propagation.yaml` example uses passive OGP-SGP4 general
perturbations.

## Public Examples

| Config | What It Demonstrates | Main Outputs | Notes |
| --- | --- | --- | --- |
| `public_tle_2hr_propagation.yaml` | OGP-backed TLE-to-ECI initialization followed by two-hour ONP propagation with configured force-model toggles | summary JSON, `object_initialization`, trajectory histories | Not continuous OGP-SGP4/SDP4 catalog propagation: OEL initializes from the TLE, then ONP numerically integrates its configured force model. |
| `public_sgp4_passive_propagation.yaml` | Passive OGP-SGP4 TLE catalog-object propagation with `propagation_method: general` and `general.model: sgp4` | summary JSON, review store, `object_propagation`, `object_state_frame`, trajectory histories | OGP-SGP4 object is passive: no thrust, controllers, or OEL force-model modifiers apply to that object's trajectory. Use `output_frame: teme` for native TEME rows, or `output_frame: eci` plus `frame_transform: teme_as_eci` for the legacy ECI-compatible approximation. Deep-space TLEs route to OGP-SDP4 through the same general-propagation surface. |
| `public_sgp4_passive_eci_transform.yaml` | Passive TLE catalog-object propagation with opt-in Vallado IAU-80 TEME-to-ECI output | summary JSON, review store, `object_propagation`, `object_state_frame`, trajectory histories | Uses `output_frame: eci` and `frame_transform: teme_to_eci_iau80`. This is deterministic frame-reduction plumbing for validation and review workflows, not an EOP-driven operational frame service. |
| `public_ground_station_access_from_tle.yaml` | Passive station access from an OGP-initialized TLE object | access summary, access/elevation/range plot, and map-backed ground track | Not continuous OGP-SGP4/SDP4 catalog propagation: uses the same TLE initializer, then ONP with J2 enabled for a quick access screen. |
| `public_closed_loop_rendezvous_lqr.yaml` | Compact closed-loop chaser/target rendezvous with HCW LQR | run summary and rendezvous metrics | Fastest controller example for reading the YAML shape. |
| `public_rendezvous_closed_loop.yaml` | Broader rendezvous with attitude pointing, sensing, EKF knowledge, and plots | dashboard, rendezvous, control, estimation, sensor-access, and ground-track plots | Best public example for end-to-end closed-loop artifact review. |
| `public_orbit_environment_stack.yaml` | Perturbation and environment toggles | summary JSON and optional plots | Use to inspect deterministic force-model configuration. |
| `public_reentry_interactive_demo.yaml` | 10-day atmospheric re-entry diagnostics with conservative kill thresholds | interactive re-entry summary, aero, and thermal plots plus JSON artifacts | Starts at the 300 km entry threshold with drag and re-entry termination enabled. |
| `public_rocket_launch_to_orbit.yaml` | Educational rocket ascent to a low-Earth insertion orbit with GNC, max-Q, TVC, fuel, and orbital-element diagnostics | rocket ascent, GNC, fuel, mission timeline, scorecard, ground-track, and orbital-element plots | Public rocket/ascent example, not a validated launch-vehicle design or operational launch analysis. |
| `public_attitude_hold_disturbance.yaml` | Attitude hold under initial error and disturbance torque | attitude histories and control artifacts | Headless-safe and useful for attitude-control sanity checks. |
| `public_actuator_presets_smoke.yaml` | Satellite actuator vocabulary for RCS, electric propulsion, and CMG attitude steering | summary JSON and validated object/control wiring | Trimmed from the local actuator lab so the example stays readable. |
| `public_orbital_elements_stationkeeping.yaml` | Single-satellite orbital-elements stationkeeping using mission strategy and controller execution | summary JSON and review store | Useful non-rendezvous control example for the current `objects` architecture. |
| `public_mission_recovery_planner.yaml` | Mission-recovery planner recommendations after a small in-track perturbing burn | recovery trade-space plot, summary JSON, and review store | Demonstrates the public mission recovery/reconstitution workflow without campaign machinery. |
| `public_manual_rpo_training.yaml` | Manual/game-style RPO scenario wiring | game/manual-control-compatible config | Can be launched with `run_game.py` when the `game` extra is installed. |
| `public_manual_engagement.yaml` | Manual engagement-style scenario with knowledge and defensive behavior | run outputs or game-mode behavior depending on entrypoint | More advanced than the first manual training config. |

## Built-In Non-Example Configs

| Config | Purpose | Normal Command |
| --- | --- | --- |
| `configs/quickstart_5min.yaml` | Fast first-run smoke scenario | `.venv/bin/python run_simulation.py --quickstart` |
| `configs/ric_pd_10km_experiment.yaml` | Flagship 10 km RIC_PD RPO review scenario with attitude-gated thrust | `.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml` |
| `configs/automation_smoke.yaml` | Small headless validation config | `.venv/bin/python run_simulation.py --config configs/automation_smoke.yaml --validate-only` |
| `configs/plotting_rendezvous_demo.yaml` | Standalone rendezvous plotting demo | `.venv/bin/python run_simulation.py --config configs/plotting_rendezvous_demo.yaml` |
| `configs/reentry_smoke.yaml` | Short atmospheric re-entry diagnostics and plot smoke case | `.venv/bin/python run_simulation.py --config configs/reentry_smoke.yaml` |

## Game Levels

The optional Pygame trainer includes packaged levels under `sim/game/configs/`.
The launcher progression starts with `rpo_00_tutorial`, then runs through
Levels 1-10, `rpo_bonus_cislunar_rendezvous`, and the replayable
`rpo_arcade_pursuit` variant. Levels 6-8 use an elliptical coast-projection
model for eccentric-orbit RPO lessons; the cislunar bonus level uses CR3BP
propagation and Moon-centered RIC controls around an L2 NRHO target; Pursuit
Arcade adds tightening goals, randomized later-round starts, and every-fifth
round elliptical boss encounters. Completed structured training levels save
Markdown debrief reports with JSON summaries and matplotlib plots under
`outputs/game_debriefs/`; Sandbox and Pursuit Arcade skip reports because they
are open-ended/replayable modes.

```bash
.venv/bin/python -m pip install ".[game]"
.venv/bin/python run_game.py
```

Running `run_game.py` without a config opens the level selector. Direct launch
also works:

```bash
.venv/bin/python run_game.py sim/game/configs/game_training_rpo_04_rendezvous.yaml
```

## Choosing A Starting Point

- New user: start with `configs/quickstart_5min.yaml`.
- Flagship review workflow: run `configs/ric_pd_10km_experiment.yaml`, then
  `examples/python/flagship_analysis.py`.
- TLE-initialized numerical propagation: start with `public_tle_2hr_propagation.yaml`.
- Ground-station access: start with `public_ground_station_access_from_tle.yaml`.
- Atmospheric re-entry: start with `configs/reentry_smoke.yaml`, then use
  `public_reentry_interactive_demo.yaml` for the longer interactive plot view.
- Rocket/ascent dynamics: start with `public_rocket_launch_to_orbit.yaml`.
- Actuator vocabulary: start with `public_actuator_presets_smoke.yaml`.
- Orbital-elements control: start with `public_orbital_elements_stationkeeping.yaml`.
- Mission recovery planning: start with `public_mission_recovery_planner.yaml`.
- Closed-loop control: start with `public_closed_loop_rendezvous_lqr.yaml`.
- Plot/artifact review: start with `configs/ric_pd_10km_experiment.yaml`.
- Guided RPO trainer: start with `run_game.py`.
- Manual RPO config wiring: start with
  `public_manual_rpo_training.yaml`.

Public examples use the canonical `objects` map. Names such as `chaser` and
`target` are readable scenario IDs, not fixed engine slots.
