# OEL MCP Supported Local Surface

OEL MCP is an optional interoperability adapter over documented Orbital
Engagement Lab workflows. It does not replace `AGENTS.md`, scenario YAML, the
CLI, Python APIs, deterministic physics, or saved evidence.

The supported M5.2 surface is intentionally local. It uses the
official Python MCP SDK v2 over stdio while preserving the frozen OEL handlers,
policy, schemas, and result envelopes. In addition to the M3 discovery and
read-only evidence tools, it supports bounded planning, structural-first
validation, one deterministic scenario run, completed-run comparison,
allowlisted evidence plots, and supported public task recipes. It cannot
authorize itself, execute an untrusted config, expose private resources, start
remote transport, or communicate externally.

## Local Stdio Server

Install the bounded optional profile and start its packaged console command:

```bash
.venv/bin/python -m pip install ".[mcp]"
.venv/bin/oel-mcp
```

For an installed distribution, the equivalent profile is:

```bash
.venv/bin/python -m pip install "orbital-engagement-lab[mcp]"
oel-mcp
```

The `integrations.oel_mcp` package, console entry point, and resource data are
included in the OEL wheel. The SDK remains a bounded optional
`mcp>=2.0.0,<3` dependency; OEL core does not import or require it. The console
command selects the SDK adapter by default. `OEL_MCP_ADAPTER=legacy` selects the
dependency-free M3 read-only rollback registry; M4 tools require SDK lifecycle
and cancellation support and are not advertised by that adapter. Any other
value fails closed.

From a source checkout, `.venv/bin/python -m integrations.oel_mcp` is equivalent
to the console command. A local MCP server is a process role; it does not listen
on a network port.

Check the installation and generate a host-specific starting configuration
before connecting an agent:

```bash
.venv/bin/oel-mcp --doctor
.venv/bin/oel-mcp --print-host-config codex
.venv/bin/oel-mcp --print-host-config claude
```

When the console entry point is unavailable, configuration generation falls
back to the current Python executable with `-m integrations.oel_mcp`. A custom
launcher can be expressed with `--command` and repeated `--arg` options.

The doctor reports the SDK/protocol versions, active registry, configured root
counts, and whether write, execution, or plugin-trust effects remain disabled.
It does not start the stdio server or perform a tool effect. Generated host
configuration contains no approval IDs; add short-lived, purpose-specific IDs
only when a connected workflow needs them.

M4 interoperability is verified: the official SDK client, pinned MCP Inspector
2.0.0, Codex CLI, and Claude Code all pass the same public capability fixture.
Reproduce the checks with:

```bash
.venv/bin/python -m integrations.oel_mcp.interop --all \
  --python .venv/bin/python \
  --output /tmp/oel-mcp-interop.json
```

This command invokes the configured model-backed host CLIs and may incur their
normal usage. It isolates host configuration and enables only the read-only
capability tool. See the
[versioned result](operations/evidence/oel_mcp_sdk_v2/interop-m4-2026-07-31.json).

By default, paths are restricted to the OEL workspace. Configure distinct
operating-system path-separated roots when an authorized host needs other
locations:

```text
OEL_MCP_READ_ROOTS=/approved/read/root
OEL_MCP_WRITE_ROOTS=/approved/write/root
```

The legacy `OEL_MCP_ALLOWED_ROOTS` setting remains a temporary compatibility
alias. M4 write and execution tools require separately authorized write roots;
read authority never implies write authority.

## Operator Approval

Write and execution authorization is configured when the server starts, not by
the model. Use distinct opaque references for the approvals granted to that
server process:

```text
OEL_MCP_WRITE_APPROVAL_IDS=review-plot-2026-07-31
OEL_MCP_EXECUTION_APPROVAL_IDS=public-run-2026-07-31
OEL_MCP_TRUST_APPROVAL_IDS=reviewed-config-2026-07-31
```

The caller must echo an allowlisted `approval_id` with the exact `write` or
`execute` scope. Trusted validation that imports plugin modules separately
requires an allowlisted ID with `trust` scope. If the applicable environment
allowlist is absent, empty, or does not match, the operation fails before the
sensitive effect. Approval IDs are audit references, not credentials, and
should not contain secrets.

Approval and other policy denials happen before execution, so MCP hosts may
surface them as protocol-level errors rather than completed tool envelopes.
The envelope field `audit.arguments_sha256` intentionally hashes argument
names and handling labels, not argument values. This prevents the audit record
from becoming a side channel for paths, SQL, or other payload content. Exact
execution provenance comes from content-bound validation IDs and durable MCP
execution manifests.

## Public Tools

| Tool | Risk | Behavior |
| --- | --- | --- |
| `oel.describe_capabilities.v1` | `R0_read` | Reports the active registry, maturity, effects, limits, and non-claims |
| `oel.inspect_run.v1` | `R0_read` | Inspects an existing output without writing a new packet |
| `oel.query_review.v1` | `R0_read` | Executes one bounded `SELECT` or `WITH` query |
| `oel.plan_run.v1` | `R0_read` | Safe-validates and estimates an exact config/output/profile proposal without authorizing it |
| `oel.validate_scenario.v1` | `R0_read` | Runs safe validation first, then optional trusted validation, and returns a normalized-config validation ID |
| `oel.run_scenario.v1` | `R2_execute` | Executes one matching trusted validation into a new approved output directory |
| `oel.compare_runs.v1` | `R0_read` | Compares allowlisted semantic metrics from two completed review stores without rerunning them |
| `oel.plot_evidence.v1` | `R1_write` | Writes one allowlisted evidence plot and operation manifest |
| `oel.run_agent_task.v1` | `R2_execute` | Runs one checked-in supported public scenario recipe and writes its evidence packet |
| `oel.prepare_report_packet.v1` | `R1_write` | Writes a bounded, hashed, provider-neutral evidence packet and authoring brief from one completed run |
| `oel.audit_report.v1` | `R1_write` | Verifies packet/artifact hashes, report structure, and evidence references without calling a model |
| `oel.inspect_handoff.v1` | `R0_read` | Validates and summarizes a typed product or handoff manifest with supported next actions |
| `oel.export_run_product.v1` | `R1_write` | Exports an exact state, atomic snapshot, or maneuver-detection product from completed review evidence |
| `oel.emit_scenario_overlay.v1` | `R1_write` | Emits a source-bound typed scenario overlay; it does not apply or execute it |
| `oel.materialize_onp_handoff.v1` | `R1_write` | Materializes and validates an ONP scenario from an accepted absolute state or atomic snapshot |
| `oel.materialize_scenario_patch.v1` | `R1_write` | Applies an accepted source-bound patch into a new validated scenario without execution |
| `oel.compare_handoff.v1` | `R1_write` | Writes semantic-parity evidence across product, materialization, manifest, and optional consumer evidence |
| `oel.assess_maneuver_readiness.v1` | `R1_write` | Applies explicit thresholds and writes a fail-closed readiness packet |

Data-bearing calls require handling metadata containing an authoritative
marking and one of these release scopes: `public`, `local_only`, or
`frontier_eligible`. Missing or conflicting handling metadata fails closed.
A direct-frontier deployment view rejects `local_only` data.
It also replaces local paths under authorized roots with opaque
`oel-local-ref:<digest>` values, including paths found inside projected query
rows. Unexpected internal errors do not return local diagnostic details.

All tools return a versioned envelope containing tool ID, risk class, status,
effects, evidence completeness/empty/truncation state, structured error, a
payload-free audit record, and an explicitly projected result. Beginning with
v0.24.2, a projected result must satisfy the tool's advertised result schema
before the adapter can return a successful envelope.

## Execution Contract

Planning and validation take the same scenario path, proposed output directory,
and required resource profile (`laptop-safe` or `standard`). Validation always
runs without plugin imports first. Trusted validation is a separate
operator-enabled request and returns a validation ID bound to the fully
normalized config,
including the selected resource profile, forced review output, quiet stdio
posture, and output directory.

`oel.run_scenario.v1` recomputes trusted validation and refuses a stale or
mismatched ID. It also refuses unsafe resource estimates and non-empty output
directories. During execution it writes `mcp_execution_config.yaml` and
`mcp_execution_manifest.json`. The manifest moves from `running` to
`completed`, `failed`, or `cancelled`; cancelled evidence is never reported as
complete. SDK cancellation is propagated through deterministic simulation step
callbacks, and progress is reported only for meaningful validation,
execution, and evidence phases.

`oel.run_agent_task.v1` accepts only recipes that are checked in, tagged
`public`, marked `supported`, and use the ordinary scenario-run workflow. Pro,
prototype, experimental, and specialized private workflows fail closed.

## Provider-Neutral Report Evidence

`oel.prepare_report_packet.v1` inspects one completed local run, inventories at
most 100 artifacts that remain beneath that run directory, hashes each included
file, and writes `report_evidence_packet.json`, `report_brief.md`, and an
operation manifest into a new approved directory. OEL does not call a provider
or author prose.

An agent or human may write Markdown from that packet and cite file artifacts
or saved-query results with `[evidence:<evidence_id>]`. Pass bounded names from
`oel://review/saved-queries/v1` as `query_names` when preparing a packet so
analytical claims receive stable `[evidence:query.<name>]` citations. The
packet and brief also project concise execution provenance without approval
IDs or source payloads. `oel.audit_report.v1` then verifies the packet content
hash, current artifact hashes, unique evidence identities, required `Evidence`
and `Limitations` sections, and known/available evidence references. The audit
explicitly records
`semantic_claim_review_performed: false`: structural and integrity checks do
not prove that every narrative interpretation is correct.

These tools are local-only and absent from `direct_frontier_restricted`.

## Public Resources

| URI | Content | Source |
| --- | --- | --- |
| `oel://capabilities/tools/v1` | Public tool schemas and effect annotations | Public MCP registry |
| `oel://review/saved-queries/v1` | Allowlisted read-only saved-query metadata | Review query registry |
| `oel://agent/tasks/v1` | Public-tagged agent task definitions | Agent task registry |
| `oel://docs/operator-guide/v1` | Packaged local operator guidance | MCP package data |
| `oel://handoff/product-kinds/v1` | Public-safe product-kind producers, next actions, and non-execution rules | `sim.handoff` |

Resource discovery is one bounded page. Each resource is at most 500,000
encoded bytes, has an explicit media type, and resolves only from checked-in
source. Arbitrary file URIs, resource templates, subscriptions, prompts, and
private resource discovery are not supported.

## Boundaries

- Code under `sim/` does not import or require this integration.
- The adapter delegates to documented `sim.agent_task` and `sim.review` APIs.
- Review stores retain SQLite read-only/query-only mode, the OEL SQL authorizer,
  a single-statement rule, and an execution-step budget.
- Read and write paths remain inside separately configured roots.
- Response, row, query-step, and review-store size budgets fail closed.
- Tool discovery is deployment-specific; the resource catalog is public-safe
  in every profile and never includes Pro resources.
- Workflow and glue tools are absent from `direct_frontier_restricted`, `mendicant_sealed`,
  and `mendicant_tandem`; only local public/Pro processes may discover them.
- MCP discovery and transport are not authorization or release policy.
- Deployment-profile selection is trusted operator configuration, not
  authentication, entitlement, or proof of caller identity.
- Remote HTTP, unrestricted scenario generation, external communication, and
  unrestricted filesystem or shell access remain outside the supported surface.

See [MCP Local Stdio Threat Model](security/mcp-local-stdio-threat-model.md) for
the trust assumptions, protected assets, deployment-profile interpretation,
and stop-line for later remote or authenticated operation.

`AGENTS.md` remains the behavioral doctrine. MCP is only a typed control
surface, and deterministic OEL artifacts remain the evidence authority.

## Acceptance And Release Gate

Run the offline compatibility and complete public-workflow gate without a
model-provider call:

```bash
.venv/bin/python -m integrations.oel_mcp.interop --release-gate \
  --python .venv/bin/python \
  --output /tmp/oel-mcp-release-gate.json
```

This verifies the restricted three-tool profile, then exercises all eighteen
public tools—including product export and inspection, ONP and patch
materialization, semantic handoff comparison, and maneuver readiness—through
real SDK stdio subprocesses. Generated exports record
`oel_commit: unavailable_public_export` instead of requiring `.git`. Add
`--with-hosts` only when model-backed Codex/Claude and Inspector checks are
intentionally authorized; those optional host checks may incur their normal
usage.
