# Video Game Mode Roadmap

## Audience

Primary users are aerospace students, early-career space operators, and
instructors teaching rendezvous and proximity operations. The product goal is
not arcade realism or full mission rehearsal first. The first goal is to make
relative orbital motion legible through safe, repeatable interaction.

## Product Thesis

Game mode should become an RPO intuition trainer:

- cadets command simple maneuvers,
- cadets can also script planned impulsive burns before a level,
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
thrust or scripted impulsive burns, see relative motion in RIC, avoid keepout
violations, and receive a small after-action summary.

Minimum features:

- direct RIC translation control mode,
- operator burn scripting and view-only playback,
- RIC trajectory visualization,
- relative range and speed display,
- keepout and goal-region overlays,
- short coaching hints,
- scenario metadata for training goals,
- after-action metrics: closest approach, final range, final relative speed,
  time inside keepout, approximate delta-v, and goal success.

Bundled training scenarios:

0. Pilot Tutorial / Operator Tutorial
   Learn the RIC translation controls, pulse-and-coast rhythm, RI/RC plot
   layout, speed controls, and, in Operator Mode, scripted burn planning before
   introducing natural-motion matching, keepout constraints, or defensive
   behavior.

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

6. `rpo_06_sun_angle_inspection`
   Fly a GEO safe-inspection pattern while keeping the chaser inside an
   allowed target-centered Sun-angle inspection beam. Pass by visiting the
   inspection gates while inside the amber lighting corridor, avoiding the
   forbidden proximity sphere, and staying under the time and delta-v budgets.

7. `rpo_07_elliptic_burn_then_approach`
   Learn the new eccentric-orbit behavior by testing radial and in-track burns,
   then complete a slow approach to an in-track hold circle using the elliptical
   coast predictor.

8. `rpo_08_elliptic_nmc`
   Enter a natural-motion circumnavigation about the eccentric-orbit target.
   Pass by matching the target radial/cross-track amplitudes and local passive
   drift relationship under the time and delta-v budgets.

9. `rpo_09_elliptic_rendezvous`
   Repeat the Level 4 terminal rendezvous in the eccentric-orbit environment.
   Pass by getting within 10 meters of the target with less than 0.1 meters per
   second of relative velocity while using the elliptical coast predictor.

10. Level 10 - Evasion
   Reverse the roles: the player flies the target while an autonomous RIC_PD
   chaser attempts rendezvous. Pass by maintaining at least 100 meters of
   separation until the timer expires while staying under the target delta-v
   budget.

11. Level 11 - Pursuit
   Later single-player bridge toward PvP: target uses a simple defensive policy.
   Pass by tracking the maneuvering target and closing within 100 meters while
   staying under the chaser delta-v budget.

11B. `game_training_rpo_11b_safe_inspection_clone`
   Safe Inspection Clone. This is a mechanical continuity clone of Level 5,
   retained so the selector can offer a non-pursuit inspection lesson beside
   the defensive sequence. It is not a new dynamics model.

Selector availability differs by mode: Level 11 Pursuit (the
`rpo_10_defensive_target_demo` scenario ID retained for compatibility) is
Pilot-only; Level 11B is available in both Pilot and Operator modes. The web
arcade is excluded from the downloadable selector in both modes.

Web-preview arcade variant:

- `rpo_arcade_pursuit`
  Repeat the Level 11 pursuit problem across randomized target-evasion rounds.
  Each cleared round resets fuel, changes the target evasion direction, tightens
  the goal radius by 5 meters down to a 5 meter floor, and rolls the weighted
  round score into the run total. Round 1 preserves the Level 11 pursuit setup.
  The chaser has a 3 m/s round delta-v budget;
  cleared rounds award 75% of the target orbital period plus conserved chaser
  delta-v at 1000 seconds per unused m/s. Later rounds randomize the chaser's
  RIC state while matching target/chaser
  orbital energy. Every fifth round becomes a boss round with an elliptical
  target orbit, randomized target true anomaly, elliptical coast projection,
  boss scoring, boss music, and an additional 5000 second flat time bonus. Boss
  eccentricity ramps from 0.05 to 0.20, and the target defensive delta-v budget
  holds at 0.1 m/s through round 20 before increasing by 0.01 m/s per round.
  Each new round resets playback to 1x speed and full-trajectory camera framing.
  This mode is intended for the browser preview and hosted-score workflow
  rather than the downloadable classroom level list.

Bonus level:

- `rpo_bonus_cislunar_rendezvous`
  Complete a close rendezvous with a target initialized on a corrected
  Earth-Moon L2 NRHO seed. The mission uses the opt-in CR3BP rotating-frame
  propagator, target-centered Moon-RIC controls, custom cislunar sprites, and
  Moon-centered orbit-plane visualization for the target NRHO. This is an
  educational cislunar teaching level, not an operational mission-design model.

## Controls

Default trainer controls should be RIC translation:

- W/S: radial +/-R
- A/D: in-track +/-I
- Left/Right arrows: cross-track +/-C
- Space: pause/resume
- R: reset the current attempt
- D: open the debrief folder from the pass/fail screen, when available
- Up/Down: adjust runtime speed
- O/P: swap the RI or RC plot panel into an orbit-plane view when supported
- Esc: leave the active level; when launched from the selector, this returns to
  level selection.

The older attitude-plus-thruster mode remains useful for advanced spacecraft
attitude/thruster coupling lessons, but it should not be the default RPO
intuition trainer.

Operator Mode uses the same R/I/C frame language but changes the interaction:
players enter time-tagged burn rows, launch view-only playback, and inspect the
trajectory preview, mission brief, equation sheet, and next-burn status instead
of flying continuous keyboard inputs.

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

The local trainer also supports configurable RIC display conventions. OEL
Default keeps positive in-track visually to the right; the Space Force preset
can flip the display mapping while the underlying physical RIC dynamics remain
unchanged.

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

Current implementation writes Markdown debrief reports for structured training
levels under `outputs/game_debriefs/<scenario_id>/attempt_.../`. Reports include
`summary.json`, a mission timeline figure with filled burn intervals, 2D RIC
trajectory plots, relative range and velocity histories, cumulative delta-v, and
control-command plots. Sandbox and Pursuit Arcade intentionally skip report
generation because they are open-ended/replayable modes; the tutorial report is
scoped to the final free-maneuver phase.

## Roadmap

### Phase 1 - RPO Trainer Foundation

- Done: direct RIC translation control mode.
- Done: operator burn scripting and playback mode.
- Done: training scenario metadata.
- Done: keepout/goal scoring.
- Done: terminal pass/fail summary at run end.
- Done: curated training configs for the tutorial and numbered RPO levels.

### Phase 2 - Visual Teaching Overlays

- Done: keepout, point-goal, and 3D NMT-goal overlays.
- Done: relative velocity and thrust vector overlays.
- Done: difficulty-scaled coast prediction.
- Done: burn markers.
- Done: close-rendezvous zoom behavior for final approach.
- Done: keepout-margin metric for recovery and approach levels.
- Done: signed +R/-R, +I/-I, and +C/-C plot labels.
- Done: OEL Default / Space Force display-convention selector.
- Done: in-level pause/equation-sheet reference overlays.

### Phase 3 - Scenario Pack

- Done: build the tutorial plus eleven numbered cadet scenarios.
- Done: mode-specific Pilot and Operator tutorials.
- Done: `rpo_01_coast_relative_motion`.
- Done: `rpo_02_vbar_approach`.
- Done: `rpo_03_rbar_approach`.
- Done: `rpo_04_rendezvous`.
- Done: `rpo_05_passive_cross_track_approach`.
- Done: `rpo_06_sun_angle_inspection` (supported educational selector level;
  its lighting corridor remains a demonstration model, not sensor assurance).
- Done: `rpo_07_elliptic_burn_then_approach`.
- Done: `rpo_08_elliptic_nmc`.
- Done: `rpo_09_elliptic_rendezvous`.
- Done: Level 10 Evasion.
- Done: Level 11 Pursuit.
- Done: Level 11B Safe Inspection Clone, mechanically matched to Level 5.
- Done: `rpo_bonus_cislunar_rendezvous`.
- Done: add instructor notes for each scenario.
- Done for implemented levels: add success thresholds and scorecards.
- Done for implemented levels: treat each mission as a pass/fail level.
- Done for implemented levels: verify each scenario can run without local
  artifacts.
- Done: keep `rpo_arcade_pursuit` as a replayable web-preview arcade variant
  rather than a downloadable classroom level.

### Phase 4 - Instructor Workflow

- Done: single-window scenario selection when launching `run_game.py` without a
  config path.
- Done: scenario reset control.
- Done: pause/resume and runtime speed controls.
- Done: level-selector video toggle with per-attempt MP4 recording saved on
  pass/fail, padded with three seconds of briefing/result screen context, and
  discarded on restart or early quit.
- Done: in-level manual clip recording with G start/discard during gameplay,
  F9 as an alternate key when available, Enter/Return save, HUD status, and clip
  outputs under `outputs/game_recordings/clips/`.
- Done: Markdown debrief reports with JSON summaries and matplotlib plots for
  structured terminal attempts.
- Done: level-selector Pilot/Operator mode toggle, persistent last mode,
  separate progress records, and saved per-level operator scripts.
- Done: operator script screen with mission brief, numeric objectives, equation
  sheet, burn-table scrolling, trajectory preview, velocity vectors, and
  trajectory probe readouts.
- Later: add replay controls.
- Provide classroom guidance.

### Phase 5 - Advanced/Competitive Modes

- Reintroduce attitude/thruster coupling as an advanced lesson.
- Add target defensive behaviors as structured exercises.
- Beta delivered: two-player RPO Duel with an authoritative Cloudflare
  Worker/Durable Object room service, plus a computer-opponent path. Remaining
  work is post-Beta network/device playtesting, usage measurement, persistence,
  and promotion—not initial PvP implementation.

## Near-Term Implementation Target

Current implementation:

- `pygame` is the default live game backend.
- The legacy Matplotlib game backend has been removed; Pygame is the single
  supported live game runtime.
- Game configs live under `sim/game/configs`.
- Running `run_game.py` without a config opens the level selector.
- `control_mode: ric_translation` is available in game metadata.
- `coast_prediction_model: tschauner_hempel` enables the elliptical linearized
  coast-projection overlay used by the eccentric-orbit levels. The live trainer
  now uses a YA closed-form STM projection for this path, with the previous
  numerical TH-style projection retained as fallback; older circular levels
  keep the HCW projection by default.
- The Pygame view launches fullscreen, grabs input through SDL, and uses Escape
  as a reliable level-exit path. Selector-launched runs return to the level
  selector; direct config launches exit the game process.
- The live trainer has a ghost coast trajectory, burn markers, labeled
  relative-velocity and thrust vectors, pause/resume, and scenario reset.
- Coast-prediction assistance is difficulty-scaled in Pilot Mode: easy shows
  one full target orbit, medium shows half an orbit, hard shows a quarter orbit,
  and extreme hides the projection. Operator Mode always shows the full
  projection, and difficulty instead maps to actuator execution error.
- Runtime speed is adjustable in-game with Up/Down. Most LEO trainer levels use
  1x through 200x; the cislunar level starts at 10x and extends to 2000x with a
  shared speed-dependent game tick schedule for smoother low-speed playback.
- Live mission metrics show time, delta-v, NMT element errors, point-goal
  error, keepout margin, and relative-speed thresholds as appropriate.
- Level pass/fail freezes the simulation and displays a mission banner.
- Terminal attempts can write per-run Markdown debrief reports with JSON
  summaries and plots, and the launcher can save MP4 recordings for completed
  attempts with three seconds of level brief and pass/fail screen context.
- Players can also capture short gameplay clips with G and save them with
  Enter/Return for social/demo use.
- Close rendezvous levels zoom around the current state and goal so meter-scale
  criteria stay visible.
- Operator Mode burn animations temporarily cap playback speed and scale the
  projection-transition duration with burn magnitude.
- The level selector remembers the last selected mode and frame-convention
  choice locally.

Implemented levels:

- Level 0 Pilot Tutorial: introduce RIC pulse controls, RI/RC views, speed
  controls, and a generous 250 meter approach objective.
- Level 0 Operator Tutorial: reuse the RIC primer, then demonstrate each burn
  direction as a prefilled script that the student launches and observes.
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
- `rpo_06_sun_angle_inspection`: fly a GEO inspection pattern while satisfying
  the target-centered Sun-angle beam and avoiding the forbidden proximity sphere.
- `rpo_07_elliptic_burn_then_approach`: test radial and in-track burns around
  an eccentric-orbit target, then enter the hold circle behind the target.
- `rpo_08_elliptic_nmc`: enter an eccentric-orbit natural-motion
  circumnavigation using the elliptical coast predictor.
- `rpo_09_elliptic_rendezvous`: repeat the close Level 4 rendezvous against
  the eccentric-orbit target, using the elliptical coast predictor to control
  closure into the 25 meter proximity zone and final 10 meter goal.
- Level 10 Evasion: fly the target vehicle, evade an autonomous
  RIC_PD chaser, preserve at least 100 meters of separation, and survive until
  the timer expires under the target delta-v budget.
- Level 11 Pursuit: track a target with simple defensive pulses, close within
  100 meters, and stay under the chaser delta-v budget. This level uses the
  target reference orbit as the RIC display/control frame so the target maneuver
  is visible, and caps target defensive delta-v at 0.1 m/s.
- `rpo_bonus_cislunar_rendezvous`: rendezvous near an Earth-Moon L2 NRHO seed
  using Moon-centered RIC controls, CR3BP propagation, high-speed cislunar time
  scaling, and a Moon-centered orbit view of the target NRHO.
- Web preview Pursuit Arcade: clear repeated pursuit rounds against randomized
  target evasion directions, tightening goal radius, randomized energy-matched
  starts, elliptical boss rounds, round-weighted scoring, conserved-delta-v time
  rewards, ramping boss eccentricity, and a late-round target delta-v ramp.

Next focus:

- Done: scenario-selection preview with objective, brief, pass criteria,
  budgets, and instructor notes.
- Done: add a tutorial level, eccentric-orbit approach/NMC levels,
  evasive-target survival level, and pursuit arcade variant.
- Done: add Operator Mode, frame-convention settings, script previews, and
  mode-specific tutorials.
- Later: add level-locking or course-progress behavior if the training flow
  needs it.
- Add classroom guidance and one-page instructor lesson cards for the
  implemented levels.
- Add replay/trace-review controls on top of the exported debrief data.
