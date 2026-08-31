# Study Lifecycle Contract

This contract defines the bounded public lifecycle for retaining completed OEL
analysis evidence as one inspectable, content-bound study. It is transport
neutral: the CLI, Python API, a future viewer, or an agent may author the
records, but none may replace or approximate the authoritative domain
analysis.

The v1 lifecycle is:

`StudyRequest -> StudyPlan -> StudyRun -> StudyEvidence -> StudyClaims -> StudyReceipt`

The machine-readable record schema is
[`oel-study-lifecycle-v1.schema.json`](schemas/oel-study-lifecycle-v1.schema.json).

## Supported capability registry

V1 accepts only completed evidence from these exact public contracts:

| Capability | Required interface | Evidence schema | Accepted status | Maximum claim label |
| --- | --- | --- | --- | --- |
| `constellation_design` | `python -m sim.constellation_design solve` | `oel.constellation_design_evidence.v1` | `complete` | `VC-1` |
| `trajectory_targeting` | `python -m sim.trajectory_design solve` | `oel.trajectory_targeting_evidence.v1` | `converged` | `VC-1` |
| `conjunction_assessment` | `python -m sim.conjunction assess` | `oel.conjunction_assessment_evidence.v1` | `completed` | `VC-1` |
| `mission_scheduling` | `python -m sim.mission_scheduling solve` | `oel.mission_scheduling_evidence.v1` | `complete` | `VC-1` |
| `orbit_lifetime` | `python -m sim.orbit_lifetime analyze` | `oel.orbit_lifetime_evidence.v1` | `completed` | `VC-1` |
| `spacecraft_power` | `python -m sim.spacecraft_power analyze` | `oel.spacecraft_power_evidence.v1` | `completed` | `VC-1` |

The registry is deliberately closed. An unknown capability, interface,
evidence schema, terminal status, plan dependency, or acceptance criterion
fails validation.

Each registered capability also owns a structural evidence adapter. The adapter
checks the domain record's required identity, result, resource, limitation, and
authoritative-check fields before the lifecycle accepts it. This is a
fail-closed source-contract check, not a second physics implementation or a
replacement for the domain's authoritative replay.

## Record semantics

- `oel.study_request.v1` states the question, capabilities, assumptions,
  clarifications, frame/time/unit context, fidelity, and acceptance criteria.
- `oel.study_plan.v1` binds the normalized request digest and maps every
  criterion to one or more dependency-checked capability steps.
- `oel.study_run.v1` records that accepted completed evidence was bound to the
  plan. It does not claim that the lifecycle layer executed domain physics.
- `oel.study_evidence.v1` records the retained path, byte count, byte digest,
  semantic digest, schema, and terminal status for each step.
- `oel.study_claims.v1` binds the normalized plan digest. Every positive claim
  maps each claimed criterion to a cited plan step that covers it, cites an
  existing retained value with a JSON Pointer, and carries an author-declared
  validation-level label. Explicit non-claims are required. The lifecycle
  verifier validates the label vocabulary, enforces each cited capability's
  maximum (currently `VC-1` for every v1 capability), and checks citation
  binding; it does not determine that the evidence merits the selected level
  within that ceiling.
- `oel.study_receipt.v1` binds the byte and semantic identity of the five
  preceding root records and publishes the bundle semantic digest and
  lifecycle limitations.

`request_sha256` and `plan_sha256` may be `auto` only in authored input. The
normalized persisted records always contain the computed lowercase SHA-256
digest.

## Normalization and identity

Scientific identity is computed from finite JSON serialized with sorted keys
and compact separators. Order-insensitive authored lists and records are
normalized into deterministic order. Retained evidence also preserves and
binds its original UTF-8 JSON bytes, so whitespace-only evidence changes alter
the byte receipt while the semantic digest remains available for diagnosis.

Changing a bound request changes the plan binding. Changing a plan changes the
run, evidence, and claims bindings. Changing retained evidence changes its
evidence record, run record, and final receipt. The verifier reconstructs this
graph and rejects stale or partially updated records.

## Bundle layout

A valid directory contains exactly:

```text
study_request.json
study_plan.json
study_run.json
study_evidence.json
study_claims.json
study_receipt.json
evidence/
  <step-id>.json
```

There may be at most 12 steps. Each retained evidence file must be a finite
UTF-8 JSON object between 1 byte and 16 MiB. Evidence bindings must exactly
match plan step IDs. Explicit file or bundle symbolic links, non-regular
files, unexpected artifacts, invalid JSON Pointers, and pre-existing output
directories are rejected. A build occurs in a sibling temporary directory and
is atomically promoted only after all checks pass.

## Inspect, replay, and compare

Inspection verifies the complete identity graph before returning a summary.
Lifecycle replay reconstructs that graph from the retained records and returns
`identity_verified` only when it matches exactly.

Lifecycle replay does **not** recompute trajectory targeting, conjunction
assessment, scheduling, orbit lifetime, or spacecraft power. Scientific recomputation remains
the responsibility of each domain's authoritative replay or repropagation
contract. This prevents a provenance tool from silently becoming a second
physics implementation.

Comparison first verifies both bundles, then reports whether their bundle
semantic digests match, which root records differ, and which evidence step
semantic digests changed.

## Validation and product boundary

The public acceptance suite covers schema validation, normalization and digest
binding, dependency cycles, unknown fields, criterion coverage, evidence
schema/status checks, JSON Pointer resolution, byte and semantic tampering,
symbolic links, unexpected artifacts, deterministic comparison, CLI behavior,
and canonical studies built from real public OEL evidence, including
orbit-lifetime and schedule-coupled spacecraft-power studies.

The schema retains the `VC-0` through `VC-4` vocabulary, but the v1 capability
registry rejects any label above `VC-1`. Acceptance at or below that ceiling is
record-shape and binding validation, not an independent assessment of
validation maturity. Promotion remains a review decision based on cited domain
evidence and a future capability contract with an appropriate maximum.

This v1 slice does not provide execution queues, monitoring, cancellation,
resume, migrations, a visual workbench, campaign orchestration, team review,
authorization, retention policy, customer data governance, or operational
qualification. Those remain future or Pro workflow layers.
