# Coverage Tasking Contract

Status: **frozen and implemented for the bounded Phase 6 programmatic core
v0.2**.

Contract identifier: `oel.coverage-tasking.v0.2`.

## Product Boundary

Coverage Tasking selects a deterministic schedule from a bounded set of
caller-supplied opportunities. Every opportunity carries the single asset ID
to which it belongs and the semantic SHA-256 of the coverage or link product
that established it. The optimizer fails if an opportunity belongs to a
different asset than the configured single-asset schedule. It never
propagates an orbit, creates an access window, repairs source evidence, or
silently widens an opportunity.

The first slice is a single-asset exact optimizer. It supports observation,
downlink, and other typed opportunities and a maximum of 24 candidates. The
default governed envelope is 20 candidates.

## Opportunity and Resource Semantics

Every opportunity declares unique identity, asset identity, kind, start and
end, nonnegative objective value, storage change, energy cost, optional
target, optional unit ECI pointing vector, and source-product hash.
Observation storage changes are nonnegative; downlink changes are
nonpositive.

The resource configuration declares:

- study horizon;
- optional maximum direct slew rate and a nonnegative settling time;
- maximum payload observation duty cycle;
- storage capacity and initial fill;
- total horizon energy budget; and
- maximum candidate count.

Tasks may not overlap. With slew enabled, the gap from one selected task to the
next must cover the great-circle angle between their pointing vectors divided
by the maximum slew rate, plus settling. Storage changes occur at task
completion and must remain within zero and capacity. Energy is accumulated as
a horizon budget. Observation duration divided by horizon duration must not
exceed the duty-cycle limit.

## Optimization and Determinism

The solver performs exact bounded enumeration with an admissible remaining-
objective bound. It maximizes total declared objective. Equal objectives use a
stable lexicographic opportunity-ID tie break, independent of caller order.

The result preserves selected tasks with post-task storage and cumulative
energy, rejected opportunity dispositions, objective, final resources, duty
cycle, evaluated feasible-leaf count, input semantic hash, and schedule hash.
Stable artifacts are a manifest, summary JSON, schedule CSV, and rejection
CSV.

## Non-Claims

The direct angular-rate slew bound is not full attitude dynamics. The resource
model does not include time-varying power generation, battery dynamics,
thermal behavior, packet protocols, routing, uncertainty, or hardware
calibration. This slice does not perform multi-asset scheduling, constellation
design, launch optimization, or orbit-design optimization. Those require
separate contracts and evidence rather than expansion of this bounded solver.

The programmatic kernel validates the declared source digest's canonical
shape and binds it into the schedule, but it does not dereference an artifact
store or independently prove that the opportunity bounds were extracted from
that product. Completed-run and review-store adapters must perform that
source-window validation before calling the kernel.
