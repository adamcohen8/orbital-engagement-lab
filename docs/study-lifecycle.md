# Integrated Study Lifecycle

OEL v0.29 can retain a bounded completed analysis as one deterministic study
bundle. A bundle records the question, plan, completed domain evidence,
evidence-backed claims and non-claims, and a content receipt that detects stale
or altered material.

V1 supports completed public trajectory-targeting, conjunction-assessment,
mission-scheduling, constellation-design, orbit-lifetime, and spacecraft-power
evidence. It does not execute those analyses for you; produce evidence with
the documented domain workflow first.

## Run the canonical example

This example runs all three real public workflows and creates one lifecycle
bundle per capability:

```bash
.venv/bin/python examples/python/study_lifecycle_three_domains.py \
  --output-root outputs/study_lifecycle_three_domains
```

The command finishes with `status: verified`. Each study also reports
`replay_status: identity_verified` and its stable bundle semantic digest.

The [spacecraft-power canonical example](spacecraft-power.md) demonstrates the
fourth registered capability and deliberately keeps authoritative power replay
separate from lifecycle identity replay.

The [orbit-lifetime canonical workflow](orbit-lifetime.md) demonstrates the
fifth registered capability and similarly separates authoritative ONP
recomputation from lifecycle identity replay.

The [constellation-design workflow](constellation-design.md) is the sixth
registered capability and keeps its generation, propagation, coverage, link,
scoring, and ranking replay separate from lifecycle identity replay.

## Author and validate records

Create request, plan, and claims JSON objects using the
[`oel.study_*` v1 schema](contracts/schemas/oel-study-lifecycle-v1.schema.json)
and the field semantics in the
[Study Lifecycle Contract](contracts/study-lifecycle-contract.md). An authored
plan may set `request_sha256` to `auto`; authored claims may similarly set
`plan_sha256` to `auto`. Validation resolves those placeholders to normalized
content digests.

```bash
.venv/bin/python -m sim.study validate-request request.json
.venv/bin/python -m sim.study validate-plan request.json plan.json
.venv/bin/python -m sim.study validate-claims request.json plan.json claims.json
```

The same commands are available through the unified CLI as `oel study ...`.

## Build a bundle

Bind exactly one completed JSON evidence file to each plan step:

```bash
.venv/bin/python -m sim.study build \
  request.json plan.json claims.json \
  --evidence trajectory-targeting=outputs/targeting/evidence.json \
  --output-dir outputs/studies/transfer-study
```

The destination must not already exist. A successful build prints a verified
summary and leaves the six lifecycle records plus retained evidence under the
new directory.

## Inspect, replay identity, and compare

```bash
.venv/bin/python -m sim.study inspect outputs/studies/transfer-study
.venv/bin/python -m sim.study replay outputs/studies/transfer-study
.venv/bin/python -m sim.study compare \
  outputs/studies/transfer-study \
  outputs/studies/transfer-study-variant
```

`inspect` and `replay` fail closed if a bound record, cited value, retained
evidence byte, schema, status, or artifact set no longer matches. `compare`
reports changed root records and changed evidence steps after verifying both
bundles.

Study replay is provenance replay: it verifies the retained lifecycle graph.
It does not rerun domain physics. Use the trajectory targeter's mandatory
repropagation evidence and the conjunction, scheduler, constellation-design,
lifetime, or power
replay surfaces when you need scientific recomputation.

## Claim discipline

Every claim must:

- map to one or more request acceptance criteria;
- cite a known plan step that covers every claimed acceptance criterion;
- resolve to an existing value in that retained evidence with a JSON Pointer;
- carry an author-declared validation-level label from the `VC-0` through
  `VC-4` vocabulary, subject to the cited capability's maximum; and
- coexist with at least one explicit non-claim.

A valid receipt proves content identity and internal lifecycle consistency. It
does not substantiate the selected validation-level label. Every capability in
the v1 registry is capped at `VC-1`, so the current verifier rejects `VC-2`
through `VC-4` claims even though those values remain in the versioned schema
vocabulary for future capability contracts. Within the allowed ceiling,
reviewers must assess whether the retained evidence actually supports the
authored level. A valid receipt also does not prove global optimality,
operational suitability, flight qualification, or authorization to act.

## Public and Pro boundary

The versioned records, strict local CLI/Python API, identity replay,
comparison, schema, and small canonical studies are public. Managed execution,
large campaigns, optimization, dashboards, team review/signoff, reusable
program templates, customer data governance, and hosted collaboration remain
Pro or future work.
