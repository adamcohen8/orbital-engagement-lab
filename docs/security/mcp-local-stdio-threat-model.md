# MCP Local Stdio Threat Model

Status: experimental pre-v2 local prototype

## Scope

This threat model covers the dependency-free OEL MCP process started from a
trusted source checkout over local standard input and output. It does not cover
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
  caller or broker. It is not independently authenticated by the pre-v2
  process.

## Protected Assets

- public and Pro source separation;
- local run evidence and review stores;
- private IHE study inputs and visible manifests;
- hidden evaluation truth;
- local filesystem layout and operator/customer path names;
- deterministic OEL semantics and evidence status;
- stdout protocol integrity and payload-free audit metadata.

## Threats And Controls

| Threat | Pre-v2 control | Residual limit |
| --- | --- | --- |
| Path traversal or symlink escape | Canonical resolution plus separate authorized read/write roots | The launching OS account still controls root contents and permissions |
| Unsafe review SQL | One read-only `SELECT`/`WITH` statement, SQLite query-only mode, authorizer, row and VM-step limits | Authorized query results may still contain sensitive evidence and require correct handling metadata |
| Local path disclosure | Direct-frontier results replace authorized-root paths with opaque `oel-local-ref` values; unexpected errors are generic | Public/Pro-local profiles intentionally retain inspectable local paths for authorized local users |
| Hidden IHE truth disclosure | Public registry cannot import Pro; visible-manifest projection fixes `hidden_truth_visible=false` | Authorized scorer workflows remain outside this MCP surface |
| Capability confusion | Closed per-profile registries and explicit effects/risk metadata | Profile selection itself is trusted configuration, not authenticated authorization |
| Data release confusion | Required marking/release scope; direct frontier rejects `local_only` | Metadata authenticity depends on the trusted caller or future release broker |
| Resource exhaustion | File, database, row, query-step, and response-size budgets | No multi-tenant concurrency or service-level admission control exists |
| Protocol/log injection | stdout is protocol-only; notifications receive no response | Structured SDK lifecycle, cancellation, and logging wait for stable SDK v2 |
| External exfiltration | No network transport or external-communication tool effect | The parent host may itself be external; direct-frontier use is therefore treated as disclosure |
| Dependency compromise | No MCP SDK dependency; source-only prototype; ordinary OEL supply-chain controls | The Python environment and OEL dependencies still require normal provenance and audit |

## Deployment Profile Interpretation

- `public_local` exposes only public inspection contracts to a local trusted
  host.
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

## Stop Line

Remote transport, authenticated principals, entitlement enforcement,
multi-tenant isolation, release-packet approval, and external provider calls
require separate designs and threat models. They must not be enabled by merely
changing the transport or deployment-profile string.
