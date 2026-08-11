# OEL MCP Operator Guide

This packaged guide covers the supported local stdio OEL MCP surface.

- Install with `python -m pip install "orbital-engagement-lab[mcp]"`.
- Start with `oel-mcp`; the official SDK adapter is the default.
- Run `oel-mcp --doctor` before connecting a host. Use
  `oel-mcp --print-host-config codex` or `claude` for a launchable starting
  configuration. For a custom command, repeat `--arg` for every server
  argument.
- Use `OEL_MCP_ADAPTER=legacy` only as the documented M3 read-only rollback
  path; it does not advertise M4 workflow tools.
- Configure the narrowest required `OEL_MCP_READ_ROOTS` and
  `OEL_MCP_WRITE_ROOTS`.
- Configure `OEL_MCP_WRITE_APPROVAL_IDS` and
  `OEL_MCP_EXECUTION_APPROVAL_IDS` outside the model before enabling M4
  write or execution calls. Configure `OEL_MCP_TRUST_APPROVAL_IDS` before a
  validation call may import scenario plugins. Approval IDs are non-secret
  audit references.
- Treat deployment profiles and handling metadata as operator policy inputs,
  not authentication or release approval.
- Treat deterministic OEL artifacts and review stores as the evidence
  authority.
- Read `oel://review/saved-queries/v1` before composing free-form SQL. Pass the
  relevant saved `query_names` to `inspect_run` or `prepare_report_packet` so
  analytical claims can cite stable query evidence IDs instead of only the
  entire review store.
- For GNC v2 evidence, prefer `fsw_invocation_summary`,
  `fsw_sensor_deliveries`, `actuator_command_chain`, `fsw_deadline_misses`,
  `safety_requirement_status`, and `fsw_checkpoint_summary`. These are
  additive saved-query names; MCP tool versions and approval semantics are
  unchanged.
- Read `oel://handoff/product-kinds/v1` before routing a typed product. Inspect
  a product before consuming it; materialization tools validate and write new
  scenarios but never authorize or execute those scenarios.
- Policy and approval denials occur before tool execution and may be returned
  by a host as protocol-level MCP errors. Operation failures after policy
  admission use the standard structured OEL envelope.
- `audit.arguments_sha256` deliberately hashes argument names and handling
  labels only. It is a payload-free shape audit, not a unique invocation or
  evidence identity; use validation IDs and execution manifests for content
  provenance.

The local surface also supports content-bound planning/validation, one approved
deterministic scenario run, completed-run comparison, allowlisted plots, and
supported public scenario-task recipes. Validation never authorizes execution.
Run tools require a matching trusted validation ID where applicable, a safe
resource preflight, a new output directory, and a server-configured approval.

Provider-neutral report packet and audit tools can hash completed local
evidence and verify report structure/references. They do not call a model or
perform semantic claim review.

The glue-aware surface can export exact completed-run states, atomic snapshots,
and maneuver detections; materialize accepted products or patches; compare
handoff semantics; and assess maneuver readiness.
Every write/execute effect requires an operator-configured approval, uses a new
output target, and records a durable operation manifest.

Successful results are checked against the advertised result schema. Pro Scale
refresh runs against an isolated snapshot of the exact validated store, and
Pro workflows that require more than one feature report and enforce every
entitlement.

Remote transport, arbitrary filesystem resources, prompts, sampling,
unrestricted scenario generation, private campaigns, and frontier release
brokering are not part of the supported local surface.
