# RPO Trainer Preview Physics Contract

The browser preview is a teaching and marketing surface for Orbital Engagement
Lab. It is not the canonical OEL simulator and must not be described as a
validated replacement for `run_game.py`.

## Purpose

The preview teaches the first RPO control intuition:

- RIC axes,
- pulse-and-coast translation,
- curved relative motion after simple burns,
- gentle final approach under a relative-speed limit.

## Preview Model

Version 1 uses a deterministic circular-reference Hill-frame model:

```text
R_ddot = 3 n^2 R + 2 n I_dot + a_R
I_ddot = -2 n R_dot + a_I
C_ddot = -n^2 C + a_C
```

The reference orbit uses `a = 7000 km` and Earth `mu = 398600.4418 km^3/s^2`.
Manual inputs produce bounded RIC accelerations and the browser integrates the
state with a fixed-step semi-implicit Euler update.

## Boundary

The preview does not include:

- scenario YAML loading,
- OEL engine validation,
- high-fidelity perturbations,
- sensor or estimator behavior,
- controller benchmarking,
- full debrief artifact generation,
- Pygame parity,
- the ten-level training pack.

## Product Language

Use language like:

> The RPO Trainer Preview teaches the core control intuition in your browser.
> The full OEL trainer runs simulator-backed scenarios locally.

Avoid language like:

> This is the full OEL simulator in the browser.

## Future Checks

The next validation step should generate a small set of OEL reference
trajectories for Level 0-style burns and compare the preview paths to those
references within a documented tolerance.
