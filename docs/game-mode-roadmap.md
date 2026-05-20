# Video Game Mode Roadmap

## Audience

Primary users are aerospace students, early-career space operators, and
instructors teaching rendezvous and proximity operations. The product goal is
not arcade realism or full mission rehearsal first. The first goal is to make
relative orbital motion legible through safe, repeatable interaction.

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

Bundled training scenarios:

0. `rpo_00_tutorial`
   Learn the RIC translation controls, pulse-and-coast rhythm, RI/RC plot
   layout, and speed controls before introducing natural-motion matching,
   keepout constraints, or defensive behavior.

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
   Pass by getting within 10 meters of the target with less than 0.1 meters per
   second of relative velocity, without entering the 25 meter proximity zone too fast.

5. `rpo_05_passive_cross_track_approach`
   Build a passively safe cross-track orbit and drift through inspection gates
   without entering the in-track forbidden cylinder.

6. `rpo_06_elliptic_burn_then_approach`
   Learn the new eccentric-orbit behavior by testing radial and in-track burns,
   then complete a slow approach to an in-track hold circle using the elliptical
   coast predictor.

7. `rpo_07_elliptic_nmc`
   Enter a natural-motion circumnavigation about the eccentric-orbit target.
   Pass by matching the target radial/cross-track amplitudes and local passive
   drift relationship under the time and delta-v budgets.

8. `rpo_08_elliptic_rendezvous`
   Repeat the Level 4 terminal rendezvous in the eccentric-orbit environment.
   Pass by getting within 10 meters of the target with less than 0.1 meters per
   second of relative velocity while using the elliptical coast predictor.

9. `rpo_09_defensive_target_demo`
   Later single-player bridge toward PvP: target uses a simple defensive policy.
   Pass by tracking the maneuvering target and closing within 100 meters while
   staying under the chaser delta-v budget.

10. `rpo_10_evasive_target_survival`
   Reverse the roles: the player flies the target while an autonomous HCW PD
   chaser attempts rendezvous. Pass by maintaining at least 100 meters of
   separation until the timer expires while staying under the target delta-v
   budget.

Arcade variant:

- `rpo_arcade_pursuit`
  Repeat the Level 9 pursuit problem across randomized target-evasion rounds.
  Each cleared round resets fuel, changes the target evasion direction, tightens
  the goal radius by 5 meters down to a 5 meter floor, adds bonus time, and
  rolls the weighted round score into the run total. Round 1 preserves the
  Level 9 start; later rounds randomize the chaser's RIC state while matching
  target/chaser orbital energy. Every fifth round becomes a boss round with an
  elliptical target orbit, randomized target true anomaly, TH projection, boss
  scoring, and boss music.

## Controls

Default trainer controls should be RIC translation:

- W/S: radial +/-R
- A/D: in-track +/-I
- Left/Right arrows: cross-track +/-C
- Space: pause/resume
- Period: single-step while paused
- R: reset the current attempt
- Up/Down: adjust runtime speed
- Esc: leave the active level; when launched from the selector, this returns to
  level selection.

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

Implemented teaching overlays include ghost "coast from here" prediction, burn
markers, relative-velocity/thrust vectors, keepout/goal overlays, forbidden
regions, approach gates, inspection gates, objective checklists, and terminal
mission banners.

Future additions:

- replay slider,
- richer instructor freeze/step controls,
- prediction-before-action prompts.

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
- Done: curated training configs for the tutorial and numbered RPO levels.

### Phase 2 - Visual Teaching Overlays

- Done: keepout, point-goal, and 3D NMT-goal overlays.
- Done: relative velocity and thrust vector overlays.
- Done: difficulty-scaled coast prediction.
- Done: burn markers.
- Done: close-rendezvous zoom behavior for final approach.
- Done: keepout-margin metric for recovery and approach levels.

### Phase 3 - Scenario Pack

- Done: build the tutorial plus ten numbered cadet scenarios.
- Done: `rpo_00_tutorial`.
- Done: `rpo_01_coast_relative_motion`.
- Done: `rpo_02_vbar_approach`.
- Done: `rpo_03_rbar_approach`.
- Done: `rpo_04_rendezvous`.
- Done: `rpo_05_passive_cross_track_approach`.
- Done: `rpo_06_elliptic_burn_then_approach`.
- Done: `rpo_07_elliptic_nmc`.
- Done: `rpo_08_elliptic_rendezvous`.
- Done: `rpo_09_defensive_target_demo`.
- Done: `rpo_10_evasive_target_survival`.
- Done: add instructor notes for each scenario.
- Done for implemented levels: add success thresholds and scorecards.
- Done for implemented levels: treat each mission as a pass/fail level.
- Done for implemented levels: verify each scenario can run without local
  artifacts.
- Done: add `rpo_arcade_pursuit` as a replayable arcade pursuit variant.

### Phase 4 - Instructor Workflow

- Done: single-window scenario selection when launching `run_game.py` without a
  config path.
- Done: scenario reset control.
- Done: pause, single-step, and runtime speed controls.
- Done: level-selector video toggle with per-attempt MP4 recording saved on
  pass/fail and discarded on restart or early quit.
- Done: JSON debrief artifacts for terminal attempts.
- Later: add replay controls.
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
- Running `run_game.py` without a config opens the level selector.
- `control_mode: ric_translation` is available in game metadata.
- `coast_prediction_model: tschauner_hempel` enables the elliptical linearized
  coast-projection overlay used by the eccentric-orbit levels; older circular
  levels keep the HCW projection by default.
- The Pygame view launches fullscreen, grabs input through SDL, and uses Escape
  as a reliable level-exit path. Selector-launched runs return to the level
  selector; direct config launches exit the game process.
- The live trainer has a ghost coast trajectory, burn markers, labeled
  relative-velocity and thrust vectors, pause/resume, single-step, and scenario
  reset.
- Coast-prediction assistance is difficulty-scaled: easy shows one full target
  orbit, medium shows half an orbit, hard shows a quarter orbit, and extreme
  hides the projection.
- Runtime speed is adjustable in-game with Up/Down across 1x, 2x, 5x, 10x,
  25x, 50x, and 100x.
- Live mission metrics show time, delta-v, NMT element errors, point-goal
  error, keepout margin, and relative-speed thresholds as appropriate.
- Level pass/fail freezes the simulation and displays a mission banner.
- Terminal attempts can write per-run JSON debriefs, and the launcher can save
  MP4 recordings for completed attempts.
- Close rendezvous levels zoom around the current state and goal so meter-scale
  criteria stay visible.

Implemented levels:

- `rpo_00_tutorial`: introduce RIC pulse controls, RI/RC views, speed controls,
  and a generous 250 meter approach objective.
- `rpo_01_coast_relative_motion`: match a 3D NMT with radial/cross-track
  amplitude tolerances, passive velocity consistency, time and delta-v budgets,
  and keepout avoidance.
- `rpo_02_vbar_approach`: enter the V-bar stationkeeping box with low relative
  speed, under time and delta-v budgets, without entering keepout.
- `rpo_03_rbar_approach`: enter the radial hold box with low relative speed,
  under time and delta-v budgets, without entering keepout.
- `rpo_04_rendezvous`: get within 10 meters of the target with less than
  0.1 meters per second of relative velocity, without entering the 25 meter
  proximity zone faster than 0.1 meters per second.
- `rpo_05_passive_cross_track_approach`: build a passively safe cross-track
  orbit and drift through the RC inspection gates without entering the in-track
  forbidden cylinder.
- `rpo_06_elliptic_burn_then_approach`: test radial and in-track burns around
  an eccentric-orbit target, then enter the hold circle behind the target.
- `rpo_07_elliptic_nmc`: enter an eccentric-orbit natural-motion
  circumnavigation using the elliptical coast predictor.
- `rpo_08_elliptic_rendezvous`: repeat the close Level 4 rendezvous against
  the eccentric-orbit target, using the elliptical coast predictor to control
  closure into the 25 meter proximity zone and final 10 meter goal.
- `rpo_09_defensive_target_demo`: track a target with simple defensive pulses,
  close within 100 meters, and stay under the chaser delta-v budget. This level
  uses the target reference orbit as the RIC display/control frame so the
  target maneuver is visible, and caps target defensive delta-v at 0.1 m/s.
- `rpo_10_evasive_target_survival`: fly the target vehicle, evade an autonomous
  HCW PD chaser, preserve at least 100 meters of separation, and survive until
  the timer expires under the target delta-v budget.
- `rpo_arcade_pursuit`: clear repeated pursuit rounds against randomized target
  evasion directions, tightening goal radius, randomized energy-matched starts,
  elliptical boss rounds, round-weighted scoring, and bonus time that strongly
  rewards conserved chaser delta-v.

Next focus:

- Done: scenario-selection preview with objective, brief, pass criteria,
  budgets, and instructor notes.
- Done: add a tutorial level, eccentric-orbit approach/NMC levels,
  evasive-target survival level, and pursuit arcade variant.
- Later: add level-locking or course-progress behavior if the training flow
  needs it.
- Add classroom guidance and one-page instructor lesson cards for the
  implemented levels.
- Add replay/trace-review controls on top of the exported debrief data.
