# OEL MCP Pre-v2 Prototype

OEL MCP is an optional interoperability adapter over documented Orbital
Engagement Lab workflows. It does not replace `AGENTS.md`, scenario YAML, the
CLI, Python APIs, deterministic physics, or saved evidence.

The pre-v2 public prototype is intentionally local, dependency-free, and
read-only. It supports capability discovery, completed-run inspection, and
bounded read-only review queries. It cannot execute simulation physics,
validate or run private workflows, or communicate externally.

## Local Stdio Server

From a source checkout, start the newline-delimited JSON-RPC stdio process with:

```bash
.venv/bin/python -m integrations.oel_mcp
```

The pre-v2 prototype is intentionally source-checkout-only. It is not included
in the built OEL wheel and does not add an MCP package dependency. A packaging
test freezes that boundary until the stable SDK v2 work deliberately introduces
an optional MCP installation profile.

This is a temporary protocol adapter used to freeze OEL-owned contracts before
adopting the official Python MCP SDK v2 stable release. A local MCP server is a
process role; it does not listen on a network port.

By default, paths are restricted to the OEL workspace. Configure distinct
operating-system path-separated roots when an authorized host needs other
locations:

```text
OEL_MCP_READ_ROOTS=/approved/read/root
OEL_MCP_WRITE_ROOTS=/approved/write/root
```

The legacy `OEL_MCP_ALLOWED_ROOTS` setting remains a temporary compatibility
alias. Public pre-v2 tools do not write, but separate roots prevent a future
write-capable tool from inheriting read authority.

## Public Tools

| Tool | Risk | Behavior |
| --- | --- | --- |
| `oel.describe_capabilities.v1` | `R0_read` | Reports the active registry, maturity, effects, limits, and non-claims |
| `oel.inspect_run.v1` | `R0_read` | Inspects an existing output without writing a new packet |
| `oel.query_review.v1` | `R0_read` | Executes one bounded `SELECT` or `WITH` query |

Data-bearing calls require handling metadata containing an authoritative
marking and one of these release scopes: `public`, `local_only`, or
`frontier_eligible`. Missing or conflicting handling metadata fails closed.
A direct-frontier deployment view rejects `local_only` data.
It also replaces local paths under authorized roots with opaque
`oel-local-ref:<digest>` values, including paths found inside projected query
rows. Unexpected internal errors do not return local diagnostic details.

All tools return a versioned envelope containing tool ID, risk class, status,
effects, evidence completeness/empty/truncation state, structured error, a
payload-free audit record, and an explicitly projected result.

## Boundaries

- Code under `sim/` does not import or require this integration.
- The adapter delegates to documented `sim.agent_task` and `sim.review` APIs.
- Review stores retain SQLite read-only/query-only mode, the OEL SQL authorizer,
  a single-statement rule, and an execution-step budget.
- Read and write paths remain inside separately configured roots.
- Response, row, query-step, and review-store size budgets fail closed.
- Tool discovery is deployment-specific.
- MCP discovery and transport are not authorization or release policy.
- Deployment-profile selection is trusted operator configuration, not
  authentication, entitlement, or proof of caller identity.
- Remote HTTP, simulation execution, external communication, and unrestricted
  filesystem or shell access are outside the pre-v2 prototype.

See [MCP Local Stdio Threat Model](security/mcp-local-stdio-threat-model.md) for
the trust assumptions, protected assets, deployment-profile interpretation,
and stop-line for later remote or authenticated operation.

`AGENTS.md` remains the behavioral doctrine. MCP is only a typed control
surface, and deterministic OEL artifacts remain the evidence authority.
