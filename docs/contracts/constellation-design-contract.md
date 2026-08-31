# Constellation Design Contract

Problem schema: `oel.constellation_design_problem.v1`.

Evidence schema: `oel.constellation_design_evidence.v1`.

The contract owns one bounded exact evaluation over an explicit inventory of
Walker Delta, Walker Star, or circular shell candidates. The authoritative
implementation is `sim.analysis.constellation_design`; the stable public API
and CLI façade is `sim.constellation_design`.

## Deterministic semantics

- Candidate and site identities are normalized into lexical order.
- Walker RAAN and argument-of-latitude generation uses the declared integer
  `T/P/F` geometry; a shell substitutes an explicit RAAN span.
- Every member is propagated serially by the configured public ONP path.
- Global coverage and ground links are evaluated by their existing public
  owners rather than reimplemented in the design layer.
- Feasibility uses the two declared minimum service thresholds.
- Score is the exact sum of the retained coverage and network rewards and
  satellite and site penalties.
- Ranking is feasible first, then descending score, then lexical design ID.
- `recommended_design_id` is the first ranked feasible design, or `null` when
  every supplied candidate is infeasible.

## Public resource envelope

- 1-8 explicit candidates;
- 2-24 satellites per candidate;
- 1-4 selected sites per candidate from at most 8 declared sites;
- at most 721 samples;
- at most 120,000,000 asset/cell/time comparisons; and
- at most 100,000 spacecraft/site/time link samples.

Validation rejects work above any bound before execution.

## Artifact and replay contract

Publication is atomic into an absent, non-symlink directory. The manifest
binds the two payload artifacts by simple filename, byte length, and SHA-256.
Replay requires the exact three-file inventory, bounded no-follow reads, valid
receipts, and byte equality with a fresh authoritative rerun.

The evidence is accepted by the public study lifecycle only with status
`complete`. The lifecycle ceiling is `VC-1`; domain replay remains the
authority for the physics and ranking evidence.

## Non-claims

This contract does not provide global optimization, automatic design-space
search, crosslink routing, station contention/capacity, deployment or
maintenance planning, calibrated environment/hardware availability,
uncertainty robustness, or operational qualification.
