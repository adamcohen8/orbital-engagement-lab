# Game Music Assets

Original, synthetic arcade-style cues for the Pygame RPO trainer. They were
generated from scratch with procedural oscillators, noise, and plucked-string
synthesis; no external samples or third-party music assets were used.

The WAV files are optional runtime assets. They are included in the default
public source/export distribution so the RPO trainer has music out of the box
from a normal clone, but they are intentionally omitted from Python wheels to
keep the core install small. Lean/no-music exports, wheels, and sparse checkouts
can run without them; missing files simply disable music or sound effects.

Default public/runtime tracks:

- `01_insert_coin_to_orbit.wav`: bright level-select/menu loop.
- `02_rendezvous_vector.wav`: steady in-mission approach loop.
- `04_docking_bay_neon.wav`: upbeat training/action loop.
- `05_final_burn_victory_loop.wav`: energetic success/final-approach loop.
- `06_casting_the_orbit_line.wav`: longer heroic arcade-space build inspired
  by the broad feeling of cinematic sci-fi rescue cues, while using original
  melody, harmony, and arrangement.
- `07_starfield_attract_mode.wav`: slower ambient menu/attract-mode drift.
- `08_silent_running_radar.wav`: sparse stealth/radar tension loop.
- `09_defender_boss_vector.wav`: aggressive boss/defensive-target pressure.
- `10_training_grid_sunrise.wav`: optimistic tutorial/training-grid loop.
- `15_mission_failed_lament_credits.wav`: darker failed-mission cue built around
  dissonant drones, warning pulses, and an unresolved ending.
- `17_orbital_boss_metal.wav`: arcade boss-metal cue with synthetic distorted
  riffs, double-kick pressure, cold space pads, and high arcade lead hooks.
- `18_keepout_zone_accelerando.wav`: extended keepout-zone escalation cue that
  starts sparse and slow, then progressively accelerates into denser alarms,
  drums, and dissonant warning pulses.
- `19_cross_track_ghost_orbit.wav`: cross-track inspection cue with drifting
  harmonic motion and a wider, haunted orbital feel.
- `21_pursuit_arcade_overdrive_no_siren_demo.wav`: Pursuit Arcade cue with the
  same arrangement as track 20, minus the radar-siren layer.
- `22_arcade_round_clear_flyover.wav`: short arcade flyover sound effect for
  cleared Pursuit Arcade rounds.
- `23_elliptic_final_burn_cinematic.wav`: original Level 9 elliptical-rendezvous
  cue with urgent propulsion hits, organ-like synth pressure, warning tones,
  and a rising final-burn escalation.
- `28_high_shred_boss_riff.wav`: high-energy boss-round guitar-riff cue with a
  higher-pitched shred-style solo overlay.
- `30_far_side_navigation_demo.wav`: cislunar rendezvous cue with cold
  navigation pings, wide lunar pads, and a slower far-side cockpit feel.
- `33_amber_terminator_demo.wav`: warm amber-cone cue for the GEO Sun-angle
  inspection lesson.
- `39_perigee_afterburner_demo.wav`: high-energy Perigee Afterburner loop for
  the browser-native RPO Duel Beta.

Additional experimental or alternate WAVs may exist in private/local workspaces,
but they are not included in the default public distribution unless the game
runtime references them.

Asset provenance and licensing posture should stay part of the normal
public/private release review process.
