# Coverage Sensitivity Evidence Contract

Status: **programmatic comparison adapter implemented v0.2**.

Contract identifier: `oel.coverage-sensitivity-evidence.v0.2`.

This adapter compares two internally coherent, already-computed global
coverage products over the same exact horizon. `cadence` comparisons require
the same HEALPix order, more refined samples, and exact retention of every
baseline epoch. `resolution` comparisons require a higher refined HEALPix
order and exactly identical analysis epochs. Domain disposition and every
non-refinement scientific configuration field must match; only analysis ID,
the selected refinement axis, and bounded execution/resource controls are
excluded from that comparison.

The evidence packet binds both source semantic hashes, analysis IDs, orders,
sample counts, the normalized matched-assumption digest, caller-declared
acceptance limits, and absolute changes in:

- time-weighted mean covered fraction;
- ever-covered fraction; and
- mean finite complete-revisit gap when both products evaluate it.

A pass means only that these supplied finite-horizon results are within the
declared limits. The adapter does not choose acceptable limits, rerun a study,
prove a convergence rate, establish steady-state behavior, or replace an
independent matched-assumption validation.
