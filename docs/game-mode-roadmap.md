# Video Game Mode Roadmap

## Audience

Primary users are new Space Force officers and Air Force Academy cadets learning
rendezvous and proximity operations. The product goal is not arcade realism or
full mission rehearsal first. The first goal is to make relative orbital motion
legible through safe, repeatable interaction.

## Product Thesis

Game mode should become an RPO intuition trainer:

- cadets command simple maneuvers,
- the simulator shows the resulting Hill-frame/RIC motion,
- mistakes are visible and safe,
- each run produces a short debrief,
- instructors can choose focused scenarios and compare attempts.
- levels have explicit pass/fail criteria tied to the learning objective.

PvP can come later. The near-term foundation should make single-player RPO
motion understandable before adding adversarial or cooperative play.

## RPO Trainer v0

Status: implemented as the current single-player Pygame trainer.

Target outcome:

A cadet can launch a curated scenario, command radial/in-track/cross-track
thrust, see relative motion in RIC, avoid keepout violations, and receive a
small after-action summary.

Minimum features:

- direct RIC translation control mode,
- RIC trajectory visualization,
- relative range and speed display,
- keepout and goal-region overlays,
- short coaching hints,
- scenario metadata for training goals,
- after-action metrics: closest approach, final range, final relative speed,
  time inside keepout, approximate delta-v, and goal success.

Initial curated scenarios:

1. `rpo_01_coast_relative_motion`
   Learn that an offset chaser drifts naturally and that a passive 3D natural
   motion trajectory has a 2:1 RI ellipse plus an out-of-plane harmonic that
   appears as a tilted RC ellipse.
   Pass by matching the target NMT radial and cross-track amplitudes within
   tolerance, satisfying the passive NMT velocity relationship, staying under
   the time and delta-v budgets, and avoiding keepout.

2. `rpo_02_vbar_approach`
   Practice small in-track corrections and patience.
   Pass by entering the stationkeeping box behind the target with low relative
   speed, under the time and delta-v budgets, without entering keepout.

3. `rpo_03_rbar_approach`
   Compare radial motion intuition against along-track drift.
   Pass by entering the radial hold box with low relative speed, under the
   time and delta-v budgets, without entering keepout.

4. `rpo_04_rendezvous`
   Complete final approach from the hold point to rendezvous.
   Pass by getting within 25 meters of the target with less than 1 meter per
   second of relative velocity.

5. `rpo_05_keepout_recovery`
   Recover from an unsafe closing geometry without entering keepout.
   Pass by preserving keepout margin, arresting closure, and settling into the
   V-bar hold box with low relative speed under the time and delta-v budgets.

6. `rpo_06_defensive_target_demo`
   Later single-player bridge toward PvP: target uses a simple defensive policy.
   Pass by preserving keepout margin, tracking the maneuvering target, and
   settling into a safe trailing corridor with low relative speed.

## Controls

Default cadet trainer controls should be RIC translation:

- W/S: radial +/-R
- A/D: in-track +/-I
- Left/Right arrows: cross-track +/-C
- R: reset attitude/control target where supported
- Esc: quit

The older attitude-plus-thruster mode remains useful for advanced spacecraft
attitude/thruster coupling lessons, but it should not be the default RPO
intuition trainer.

## Visualization Priorities

The dashboard should emphasize:

- target-centered RIC axes,
- chaser trajectory trail,
- current relative velocity,
- current thrust vector,
- keepout region,
- goal/stationkeeping region,
- short current-state hints.

Future additions:

- ghost "coast from here" prediction,
- burn markers on the trajectory,
- replay slider,
- instructor freeze/step controls,
- scenario objective cards.

## Debrief Priorities

Each run should produce a concise debrief:

- learning objective,
- closest approach,
- final range,
- final relative speed,
- keepout time,
- approximate delta-v,
- goal success or miss reason,
- pass/fail result for the level,
- relative-motion element errors for NMT-focused levels,
- one or two coaching observations.

This debrief matters as much as live control. Cadets will learn from seeing why
an approach became unstable.

## Roadmap

### Phase 1 - RPO Trainer Foundation

- Done: direct RIC translation control mode.
- Done: training scenario metadata.
- Done: keepout/goal scoring.
- Done: text debrief at run end.
- Done: curated training configs for the initial six RPO levels.

### Phase 2 - Visual Teaching Overlays

- Done: keepout, point-goal, and 3D NMT-goal overlays.
- Done: relative velocity and thrust vector overlays.
- Done: difficulty-scaled coast prediction.
- Done: burn markers.
- Done: close-rendezvous zoom behavior for final approach.
- Done: keepout-margin metric for recovery and approach levels.

### Phase 3 - Scenario Pack

- In progress: build the six initial cadet scenarios.
- Done: `rpo_01_coast_relative_motion`.
- Done: `rpo_02_vbar_approach`.
- Done: `rpo_03_rbar_approach`.
- Done: `rpo_04_rendezvous`.
- Done: `rpo_05_keepout_recovery`.
- Done: `rpo_06_defensive_target_demo`.
- In progress: add instructor notes for each scenario.
- Done for implemented levels: add success thresholds and scorecards.
- Done for implemented levels: treat each mission as a pass/fail level.
- Done for implemented levels: verify each scenario can run without local
  artifacts.

### Phase 4 - Instructor Workflow

- Done: single-window scenario selection when launching `run_game.py` without a
  config path.
- Done: scenario reset control.
- Done: pause, single-step, and runtime speed controls.
- Later: add replay controls.
- Export debrief artifacts.
- Provide classroom guidance.

### Phase 5 - Advanced/Competitive Modes

- Reintroduce attitude/thruster coupling as an advanced lesson.
- Add target defensive behaviors as structured exercises.
- Add two-player PvP after the single-player trainer is stable.

## Near-Term Implementation Target

Current implementation:

- `pygame` is the default live game backend.
- The legacy Matplotlib game backend has been removed; Pygame is the single
  supported live game runtime.
- Game configs live under `sim/game/configs`.
- `control_mode: ric_translation` is available in game metadata.
- The Pygame view launches fullscreen, grabs input through SDL, and uses Escape
  as a reliable quit path.
- The live trainer has a ghost coast trajectory, burn markers, labeled
  relative-velocity and thrust vectors, pause/resume, single-step, and scenario
  reset.
- Coast-prediction assistance is difficulty-scaled: easy shows one full target
  orbit, medium shows half an orbit, hard shows a quarter orbit, and extreme
  hides the projection.
- Runtime speed is adjustable in-game with Up/Down across 1x, 2x, 5x, 10x,
  25x, and 50x.
- Live mission metrics show time, delta-v, NMT element errors, point-goal
  error, keepout margin, and relative-speed thresholds as appropriate.
- Level pass/fail freezes the simulation and displays a mission banner.
- Close rendezvous levels zoom around the current state and goal so meter-scale
  criteria stay visible.

Implemented levels:

- `rpo_01_coast_relative_motion`: match a 3D NMT with radial/cross-track
  amplitude tolerances, passive velocity consistency, time and delta-v budgets,
  and keepout avoidance.
- `rpo_02_vbar_approach`: enter the V-bar stationkeeping box with low relative
  speed, under time and delta-v budgets, without entering keepout.
- `rpo_03_rbar_approach`: enter the radial hold box with low relative speed,
  under time and delta-v budgets, without entering keepout.
- `rpo_04_rendezvous`: get within 25 meters of the target with less than
  1 meter per second of relative velocity.
- `rpo_05_keepout_recovery`: recover from an unsafe closing state by keeping
  out of the keepout zone and returning to the V-bar hold box with low relative
  speed.
- `rpo_06_defensive_target_demo`: track a target with simple defensive pulses,
  maintain keepout margin, and settle into a safe trailing corridor. This level
  uses the target reference orbit as the RIC display/control frame so the
  target maneuver is visible, and caps target defensive delta-v at 5 m/s.

Next focus:

- Done: scenario-selection preview with objective, brief, pass criteria,
  budgets, and instructor notes.
- Later: add level-locking or course-progress behavior if the training flow
  needs it.
- Add instructor-facing notes and classroom guidance for the implemented
  levels.
- Add replay/debrief export once the six-level pack is stable.
