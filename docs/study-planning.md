# Local BYO Study Planning (Prototype)

OEL's Phase 3A study-planning prototype lets a user-funded frontier agent ask
what public and Pro operations exist, propose a typed study, and receive one of
three deterministic dispositions:

- `CLARIFICATION_REQUIRED` while material user choices remain unresolved;
- `PLAN_VALID` when the exact plan is supported, validated, and inside the
  prototype resource envelope; or
- `UNSUPPORTED` when OEL cannot answer the question as proposed.

This local prototype does **not** execute a study, accept payment, authorize a
quote, approve a plan, submit feedback, connect to a hosted worker, or receive
the user's model-provider credentials. Pro capability descriptors are public
contracts; they do not expose or distribute the Pro implementation.

For a package intended for Hosted execution, use the opt-in local package
validator documented in `docs/hosted-pro-package.md`. It accepts public-only
plans while recommending the free local venue, and it can plan a known
public-safe Pro contract even though no Pro executor is installed locally.
This does not change ordinary `oel study preflight`, which remains fail-closed
when a capability has no authoritative executor binding in the selected
catalog.

## Command-Line Use

The managed launcher exposes the transport-neutral contract:

```bash
oel study capabilities --scope discovery
oel study schema oel.hosted_study_plan.v2
oel study preflight \
  --request request.json \
  --plan proposed-plan.json \
  --config scenario_main=configs/my-study.yaml \
  --workspace-root . \
  --scope discovery
oel study plan-review --planning-result planning-result.json
```

The request and plan arguments are JSON objects. The plan is a proposal: OEL
generates its immutable identifiers and digests after validating the closed
contract. Each `--config` uses `CONFIG_REF=PATH`, must remain inside the
selected workspace, and is parsed and safely validated without importing
plugins or advancing simulation state.

Every `config_constraints[].path` is an RFC 6901 JSON Pointer into the
normalized configuration. It must begin with `/`, separate tokens with `/`,
encode `/` inside a token as `~1`, and encode `~` as `~0`. For example, use
`/scenario_name` or `/simulator/duration_s`; dotted paths such as
`simulator.duration_s` are rejected by the plan contract.

Configuration validation receipts explicitly report:

- source and normalized-config digests;
- safe-validation status and errors;
- resource estimate and policy action;
- `plugins_imported: false`;
- `execution_advanced: false`; and
- `charge_created: false`.

`PLAN_VALID` means the proposal passed this prototype preflight. It is not
approval, payment authorization, scientific qualification, or permission to
execute. The result always contains `payment_authorized: false`,
`execution_authorized: false`, and `agent_may_authorize: false`.

## Public-Free And Hosted-Pro Routing

Public OEL publishes public-safe descriptors for both public and Pro
capabilities. Availability is derived from an explicit executor binding, never
from the presence of a descriptor or importable source. The current public
capabilities identify their concrete executor contracts. Pro descriptors remain
discoverable, but are marked unavailable until a hosted adapter is implemented
and bound. The descriptors reveal typed inputs, outputs, bounds, maturity, and
limitations; they do not expose private implementation source or a local
execution surface.

OEL deterministically classifies the exact operation graph from those declared
capability editions. It returns structured facts for the frontier agent to
explain in its own words; OEL does not map questions to canned product messages:

- `LOCAL_FREE_AVAILABLE` means every required operation is included in the
  public OEL core. Local execution is available for a `$0` OEL execution fee
  and is the recommendation, not a requirement. A valid package may also be
  submitted for authoritative Hosted preflight and a paid Hosted quote.
- `HOSTED_PRO_REQUIRED` identifies a locally valid Hosted package plan whose
  required Pro operations use published public-safe contracts. The package
  result identifies public and Pro capability IDs separately, returns no local
  quote, and requires the hosted service to confirm authoritative executor
  availability and repeat private semantic preflight before it can offer
  execution terms.
- `NOT_ELIGIBLE` means the plan is incomplete, unsupported, invalid, or outside
  policy. No payment or execution route is offered.

The BYO model provider may charge for agent usage in every route. Public-free
routing means no OEL execution fee. A Pro-required route does not establish a
price or hosted entitlement. Local preflight never executes, charges, exposes
a Pro execution tool, or authorizes the agent to approve anything.

## Compatible Frontier-Agent Hosts

Install the optional MCP dependency and start the restricted local stdio
server:

```bash
python -m pip install '.[mcp]'
oel-study-mcp --doctor
OEL_MCP_READ_ROOTS=/absolute/path/to/project oel-study-mcp
```

An agent host may equivalently launch:

```text
python -m integrations.oel_mcp.study_server
```

with `OEL_MCP_ADAPTER=sdk` and `OEL_MCP_READ_ROOTS` set to the explicitly
selected project root. Keep model-provider credentials in the agent host; do
not place them in the OEL MCP environment.

The restricted server exposes only:

- `oel.describe_capabilities.v1`;
- `oel.study.capabilities.v1`;
- `oel.study.preflight.v1`;
- `oel.study.plan_review.v1`;
- `oel.hosted.validate_package.v1`; and
- `oel.study.prepare_capability_request.v1`.

All six tools declare `writes: false`, `executes: false`, and
`external_communication: false`. No existing scenario-run, Pro execution,
filesystem-enumeration, shell, Python, payment, or feedback-submission tool is
included in this profile.

## Agent Planning Loop

1. Read `oel.study.capabilities.v1` and the plan schema in the preflight tool.
2. Discuss the open-ended orbital-analysis question with the user.
3. Record material decisions as user choices; do not silently choose them.
4. Propose a typed `StudyPlan` using only declared capabilities.
5. Call `oel.study.preflight.v1` with explicitly granted configs.
6. If clarification is required, ask the user and revise the request.
7. If unsupported, explain the structured blocker. Optionally prepare a
   sanitized feedback preview; nothing is submitted in Phase 3A.
8. If valid, render the preflight-bound plan review. Stop there: hosted
   approval, payment, and execution belong to later phases.

For a known public or Pro capability, the agent may instead assemble the explicit package
layout from `docs/hosted-pro-package.md` and call
`oel.hosted.validate_package.v1`. That opt-in path can return a locally valid
`LOCAL_FREE_AVAILABLE` or `HOSTED_PRO_REQUIRED` plan while keeping Pro code
absent. It still performs no upload, quote, payment, approval, or execution.

The frontier model may compose capabilities in a novel operation graph, but it
may not invent replacement physics, silently weaken evidence, bypass a
resource refusal, or approve its own proposal.

## Refusal Feedback Preview

`oel.study.prepare_capability_request.v1` accepts only an `UNSUPPORTED` result.
It returns a local, editable `CapabilityRequest` with
`submission_authorized: false`. By default it contains no original question,
plan, transcript, configuration, project file, or attachment. A future hosted
submission action must show the exact payload and obtain separate user
approval.

## Current Boundary

Implemented in Phases 1-3A:

- closed, versioned lifecycle contracts and content digests;
- an explicit public/Pro capability vocabulary with authoritative executor
  bindings and fail-closed availability;
- deterministic public-free versus hosted-Pro execution routing;
- deterministic capability, reference, transfer, config-parity, evidence, and
  resource preflight;
- free safe config-validation receipts;
- a `$0` public-local result and a reserved hosted-preflight route for future
  bound Pro plans;
- a non-authorized plan review that accepts only a `PLAN_VALID` preflight result;
- local CLI and official-SDK MCP adapters; and
- sanitized refusal-feedback preparation.

Deferred:

- paid frontier-model evaluation runs;
- profile linking, remote transport, and hosted execution;
- user approval and payment capture;
- immutable execution receipts and artifact return; and
- external feedback or complaint submission.

## Private Local Loopback Proof

The private development workspace has a synchronous loopback proof in
`sim.hosted_study`. It consumes a `PLAN_VALID` result that deterministically
routes to `HOSTED_PRO_REQUIRED`, calculates a compute-based quote under a
versioned rate card targeting 75% direct-compute contribution margin, requires
a separate content-bound user approval, verifies input grants, executes the
reviewed adapters in dependency order, and returns digest-verified artifacts.
Its substantive proof graph is configuration validation, a canonical
identity-bound baseline lifecycle run, a serial bounded Monte Carlo campaign,
completed-run inspection, and Pro report-packet assembly. The campaign reuses
OEL's checked-in deterministic engine and per-iteration checkpoints.

The quote binds declared limits for operation count, cases, wall and CPU time,
peak memory, storage, maximum artifact size, and parallel workers. OA adapters
may additionally bind input bytes, observations, arc duration, propagation
steps, estimator evaluations, batch evaluations, and stations. Execution
records actual consumption and checks it during the operation graph. It cannot
expand a resource or spending authorization. A cap overrun fails closed. An interruption after a valid
campaign checkpoint produces an immutable incomplete attempt receipt and may
resume against the same plan, quote, approval, and spending cap. Generic
execution failure is terminal. Failed, interrupted, partial, and
evidence-incomplete attempts return no successful study evidence.

This package is explicitly excluded from the public export. The public
capability catalog exposes Pro descriptors but still marks their executors
unavailable. The loopback proof performs no payment capture, remote
communication, model inference, or general hosted-worker execution.
