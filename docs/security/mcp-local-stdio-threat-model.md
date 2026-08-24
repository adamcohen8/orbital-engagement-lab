# MCP Local Stdio Threat Model

Status: supported M5.2 local stdio surface

## Scope

This threat model covers the packaged OEL MCP process using the official Python
SDK v2 over local standard input and output, plus its dependency-free rollback
adapter. It does not cover
remote HTTP, OAuth, multi-tenant service operation, external model-provider
transmission, or an accredited deployment.

The MCP process is a local server role, not a network service. It accepts
newline-delimited protocol requests from the parent host process, invokes
bounded OEL handlers, and writes protocol responses to stdout. It must not
open a network listener or communicate externally.

## Trust Assumptions

- The operator controls the source checkout, Python environment, launch
  command, deployment profile, and configured filesystem roots.
- The parent MCP host and the operating-system account launching OEL are
  trusted to the degree required by the selected deployment profile.
- Scenario plugins, referenced modules, and external paths remain untrusted
  until separately reviewed. MCP validation or discovery is not permission to
  import or execute them.
- `public_local`, `pro_local`, `mendicant_sealed`, `mendicant_tandem`, and
  `direct_frontier_restricted` are trusted operator configuration. A profile
  name is not authentication, entitlement, classification, release approval,
  or proof of the caller's identity.
- Handling metadata is a required policy assertion supplied by an authorized
  caller or broker. It is not independently authenticated by the M4 process.
- Trust, write, and execution approval IDs are configured in the server
  environment by the operator. They are non-secret audit references, not
  authentication.

## Protected Assets

- public and Pro source separation;
- local run evidence and review stores;
- private IHE study inputs and visible manifests;
- hidden evaluation truth;
- local filesystem layout and operator/customer path names;
- deterministic OEL semantics and evidence status;
- stdout protocol integrity and payload-free audit metadata.

## Threats And Controls

| Threat | M4 control | Residual limit |
| --- | --- | --- |
| Path traversal or symlink escape | Canonical resolution plus separate authorized read/write roots | The launching OS account still controls root contents and permissions |
| Unsafe review SQL | One read-only `SELECT`/`WITH` statement, SQLite query-only mode, authorizer, row and VM-step limits | Authorized query results may still contain sensitive evidence and require correct handling metadata |
| Local path disclosure | Direct-frontier results replace authorized-root paths with opaque `oel-local-ref` values; unexpected errors are generic | Public/Pro-local profiles intentionally retain inspectable local paths for authorized local users |
| Hidden IHE truth disclosure | Public registry cannot import Pro; visible-manifest projection fixes `hidden_truth_visible=false` | Authorized scorer workflows remain outside this MCP surface |
| Capability confusion | Closed per-profile registries and explicit effects/risk metadata | Profile selection itself is trusted configuration, not authenticated authorization |
| Data release confusion | Required marking/release scope; direct frontier rejects `local_only` | Metadata authenticity depends on the trusted caller or future release broker |
| Resource exhaustion | File, database, row, query-step, and response-size budgets | No multi-tenant concurrency or service-level admission control exists |
| Protocol/log injection | Official SDK stdio framing; stdout is protocol-only | The parent process still controls stderr collection and display |
| Resource path or metadata disclosure | Seven fixed public URIs, no arbitrary resolver, public allowlists, explicit media types, and a 500,000-byte limit | Resource contents still require ordinary public-release review |
| External exfiltration | No network transport or external-communication tool effect | The parent host may itself be external; direct-frontier use is therefore treated as disclosure |
| Dependency compromise | Bounded optional `mcp>=2.0.0,<3`; exact wheel hashes, licenses, SBOM, offline install, and zero-finding audit recorded | Point-in-time evidence must be regenerated for later releases |
| Model self-authorization | Write/execute calls must match operator-configured approval-ID allowlists; direct-frontier and Mendicant profiles do not discover M4 tools | The operator remains responsible for granting and revoking the server process's approval scope |
| Validation confused with execution permission | Safe validation is always first; trusted validation returns a content-bound ID but explicitly reports `execution_authorized=false` | An authorized operator must separately review plugin/path trust and grant execution approval |
| Config changes after validation | Execution recomputes the normalized config and requires the exact validation ID, output directory, and resource profile | External files referenced by a trusted plugin remain under operator trust and filesystem controls |
| Output overwrite or path escape | Canonical write-root enforcement and a new-or-empty output requirement | A separately authorized process can still modify output contents after the run |
| Resource exhaustion during execution | Only `laptop-safe` or `standard`; config size, history memory, worker, response, and resource-pressure gates; unsafe estimates fail closed | Long but permitted runs still consume local resources until completion or cancellation |
| Ambiguous cancellation evidence | SDK cancellation sets a thread-safe token checked at deterministic step boundaries; durable manifests record cancelled/incomplete state | Cancellation latency is bounded by the next callback boundary, not instantaneous process termination |
| Protocol corruption by simulation output | M4 materialized configs force `outputs.stats.print_summary=false`; stdout remains reserved for MCP framing | Third-party code explicitly trusted by the operator could still violate process-output discipline |
| Report packet or artifact substitution | Content-bound packet identity, artifact-relative paths confined to the source run, SHA-256 verification, new audit output, and explicit evidence-reference checks | Structural integrity does not prove every narrative interpretation is analytically correct |
| Diagnostic secret disclosure | Doctor reports approval counts and entitlement availability, never approval values, license identities, or credentials | Authorized local root paths are intentionally visible to the local operator running doctor |

## Deployment Profile Interpretation

- `public_local` exposes only public inspection contracts to a local trusted
  host, plus M4 workflow tools when operator approvals are configured.
- `pro_local` exposes public plus authorized Pro contracts locally.
- `mendicant_sealed` and `mendicant_tandem` are registry/policy views for a
  future Mendicant runtime. They do not yet implement a release broker,
  provider call, packet approval, or distinct authenticated runtime.
- `direct_frontier_restricted` exposes only the public registry, rejects
  local-only handling scope, and removes authorized local paths from results.
  Selecting this profile does not itself approve any data for release.

## Operator Requirements

1. Launch only from a trusted checkout and environment.
2. Configure the narrowest read and write roots required for the task.
3. Select the deployment profile outside model control.
4. Treat connection to a frontier-hosted agent as disclosure even over local
   stdio.
5. Do not use MCP discovery, validation, or handling metadata as permission to
   execute untrusted scenarios or plugins.
6. Retain deterministic OEL artifacts as the analytical authority.
7. Use `OEL_MCP_ADAPTER=legacy` only as the reviewed rollback path when the SDK
   profile cannot be approved or operated.
8. Grant short-lived, purpose-specific trust, write, and execution approval
   IDs; do not place secrets in approval IDs.
9. Treat `trust_plugins=true` as a statement that referenced code and external
   paths were reviewed, not as a convenience flag.

## Stop Line

Remote transport, authenticated principals, entitlement enforcement,
multi-tenant isolation, release-packet approval, and external provider calls
require separate designs and threat models. They must not be enabled by merely
changing the transport or deployment-profile string.
