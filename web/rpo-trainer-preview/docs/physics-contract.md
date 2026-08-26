# RPO Trainer Preview Physics Contract

The browser preview is a teaching and marketing surface for Orbital Engagement
Lab. It is not the canonical OEL simulator and must not be described as a
validated replacement for `run_game.py`.

## Purpose

The preview teaches the first RPO control intuition and provides a lightweight
browser arcade competition surface:

- RIC axes,
- pulse-and-coast translation,
- curved relative motion after simple burns,
- gentle final approach under a relative-speed limit.
- deterministic browser replay validation for Pursuit Arcade attempts.

## Tutorial Model

Tutorial uses a deterministic circular-reference Hill-frame model:

```text
R_ddot = 3 n^2 R + 2 n I_dot + a_R
I_ddot = -2 n R_dot + a_I
C_ddot = -n^2 C + a_C
```

The reference orbit uses `a = 7000 km` and the existing browser contract's
Earth `mu = 398600.4418 km^3/s^2`. The current OEL engine constant is
`398600.4415 km^3/s^2`; the browser value remains pinned to preserve existing
`web-two-body-v2` challenge hashes and replay packets. The contract test allows
at most `1e-3 km^3/s^2` difference and the OEL-generated trajectory comparison
below bounds the resulting path difference. A future value change requires a
new physics version. Manual inputs produce bounded RIC accelerations and the
browser integrates the state with a fixed-step semi-implicit Euler update.

## Sandbox Model

Sandbox mirrors the downloadable preflight field contract: six target
classical orbital elements and six chaser target-centered rectangular RIC
state values. The browser converts those values into target and chaser ECI
states, propagates both under central Earth two-body gravity, and maps bounded
manual RIC acceleration into ECI at each deterministic step. Operator previews
and playback use the same configured target orbit and relative initial state.

This includes elliptical target orbits, but it remains a browser-native
two-body model. It does not add OEL scenario loading, perturbations, estimator
behavior, recordings, or OEL engine validation.

## Pursuit Arcade Model

Pursuit Arcade uses a separate deterministic browser-native competition engine.
It propagates target and chaser ECI states under central Earth two-body gravity,
maps player commands from target RIC into ECI acceleration, records inputs by
simulation tick, and validates submitted scores by replaying the attempt packet.

The canonical arcade replay step remains the challenge `dt_s` value, currently
1 second. The preview UI has a speed-dependent tick helper for parity with the
downloadable trainer policy, but the helper clamps to the browser challenge's
base step so leaderboard attempts continue to validate from canonical ticks.

This supports the browser arcade and hosted leaderboard without running the
full Python game server-side. It is still not the downloadable trainer physics
stack.

## Boundary

The preview does not include:

- scenario YAML loading,
- OEL engine validation,
- high-fidelity perturbations,
- sensor or estimator behavior,
- controller benchmarking,
- full debrief artifact generation,
- full Pygame parity,
- the complete downloadable training catalog.

## Product Language

Use language like:

> The RPO Trainer Preview teaches the core control intuition in your browser.
> Pursuit Arcade attempts are replay-validated by the browser competition
> engine. The full OEL trainer runs simulator-backed scenarios locally.

Avoid language like:

> This is the full OEL simulator in the browser.

## Contract Checks

`tools/generate-oel-contract-fixtures.py` derives the checked-in Level 0,
Sandbox, and Pursuit Arcade browser contracts from their canonical OEL scenario
YAML. Its `--check` mode fails when those fixtures drift from the downloadable
contracts.

The same tool generates three passive Level 0-style reference trajectories
with OEL's two-body engine: +I, +R, and +C initial relative-velocity cases.
`tests/preview-contract.test.mjs` propagates each case with the browser's actual
0.1 second semi-implicit Euler HCW step and checks R/I/C position at 0, 60, 300,
and 600 seconds. The documented acceptance tolerance is `5e-5 km` (5 cm) per
position axis. This is approximation-continuity evidence for the teaching
preview, not a claim that HCW replaces the downloadable OEL engine.

`tests/sandbox-setup.test.mjs` pins the downloadable Sandbox field order,
defaults, numeric bounds, target COE mapping, and the m/s-to-km/s conversion for
chaser RIC relative rates.
