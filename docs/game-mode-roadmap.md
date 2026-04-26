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

3. `rpo_03_rbar_approach`
   Compare radial motion intuition against along-track drift.

4. `rpo_04_stationkeeping`
   Hold a relative point with low delta-v.

5. `rpo_05_keepout_recovery`
   Recover from an unsafe closing geometry without entering keepout.

6. `rpo_06_defensive_target_demo`
   Later single-player bridge toward PvP: target uses a simple defensive policy.

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

- Add direct RIC translation control mode.
- Add training scenario metadata.
- Add keepout/goal scoring.
- Add text debrief at run end.
- Add one or two curated training configs.

### Phase 2 - Visual Teaching Overlays

- Add keepout and goal overlays to RIC plots.
- Add relative velocity and thrust vector overlays.
- Add coast prediction.
- Add burn markers.

### Phase 3 - Scenario Pack

- Build the six initial cadet scenarios.
- Add instructor notes for each scenario.
- Add success thresholds and scorecards.
- Treat each mission as a pass/fail level.
- Verify each scenario can run without local artifacts.

### Phase 4 - Instructor Workflow

- Add scenario selection and reset controls.
- Add pause, step, speed, and replay controls.
- Export debrief artifacts.
- Provide classroom guidance.

### Phase 5 - Advanced/Competitive Modes

- Reintroduce attitude/thruster coupling as an advanced lesson.
- Add target defensive behaviors as structured exercises.
- Add two-player PvP after the single-player trainer is stable.

## Near-Term Implementation Target

Implemented foundation:

- `pygame` is the default live game backend.
- The legacy Matplotlib game backend has been removed; Pygame is the single
  supported live game runtime.
- `control_mode: ric_translation` is available in game metadata.
- The Pygame view launches fullscreen, grabs input through SDL, and uses Escape
  as a reliable quit path.
- The live trainer has a ghost coast trajectory, burn markers, labeled
  relative-velocity and thrust vectors, pause/resume, single-step, and scenario
  reset.

Continue RPO Trainer v0 as a narrow slice:

- `control_mode: ric_translation` in game metadata,
- a pure trainer scoring module,
- one curated training config,
- dashboard keepout, point-goal, and 3D NMT-goal overlays,
- pass/fail criteria for `rpo_01` based on NMT amplitude errors, passive
  velocity consistency, time budget, delta-v budget, and keepout,
- run-end text debrief,
- focused tests for controls and scoring.
