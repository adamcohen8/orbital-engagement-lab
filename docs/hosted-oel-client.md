# Public Hosted OEL Client

**Closed alpha: access is not publicly available.** Public OEL remains free to
run locally without a Hosted account. The bundled client and package validator
do not grant service access. Only invited operators with a configured profile
can request alpha offers; the alpha uses simulated settlement, not real payments.

The public `oel hosted` command is a narrow client for a separately operated
Hosted OEL service. It does not contain OEL Pro executors, worker source,
service credentials, model credentials, payment logic, or a hidden simulation
engine. When an approved plan needs only public capabilities, local OEL is the
recommended zero-fee venue; an invited operator may test a quoted Hosted run with simulated settlement.

This client is currently a local black-box reference implementation. Its
transport launches a user-configured argv command, sends one JSON request on
standard input, and reads one JSON response. It never invokes a shell. A
production release still needs authenticated HTTPS, remote file transfer,
profile linking, and the deployed service boundary described by the Hosted OEL
MVP plan.

## Agent-visible flow

The intended flow is:

```text
discover capabilities -> create proposal -> classify or request hosted offer
-> user reviews exact quote -> explicit approval -> durable status
-> verified result import
```

The service authors the authoritative hosted preflight and quote. A frontier
agent may prepare the request and proposed plan, but it cannot authorize
execution. The service persists each authoritative offer for fifteen minutes
and consumes it once. Approval requires a short-lived user-scoped session whose
signed subject matches `--authorized-by`, plus the exact
`APPROVE <offer_sha256>` confirmation after the user reviews the offer.

Before staging, Public OEL can validate a complete public or Pro analysis
package without importing Pro code or uploading bytes:

```bash
oel hosted package-schema
oel hosted validate-package ./my-study --workspace-root .
```

The package reuses the public StudyRequest and StudyPlan contracts, binds actual
input hashes and byte counts, compiles typed operation edges, validates public
scenario inputs, and returns a `CLIENT_VALID` receipt whose route is
`LOCAL_FREE_AVAILABLE` or `HOSTED_PRO_REQUIRED`. See
`docs/hosted-pro-package.md`. The receipt is only a local planning and integrity
result. The service must author the authoritative Hosted validation receipt and
quote from the exact sealed bytes.

## Link a scoped local profile

The profile stores a signed, scope-limited Hosted OEL session token and one
argv transport command. It does not store frontier-model provider credentials.
Treat the transport command as trusted local configuration.

```bash
oel hosted profile link \
  --profile .oel/hosted/profile.json \
  --service-label hosted-oel-proof \
  --transport-command hosted-transport.json \
  --session-token hosted-session.json
```

`hosted-transport.json` contains an argv array, not a shell command:

```json
{
  "command": ["hosted-oel-transport"]
}
```

## Discover and route

```bash
oel hosted capabilities --profile .oel/hosted/profile.json
oel hosted route --planning-result planning-result.json
```

Routing has three explicit dispositions:

- `execution_options_available`: the plan uses public OEL capabilities, local
  execution is free and recommended; Hosted is unavailable without an alpha profile;
- `hosted_pro_offer`: the plan needs Pro; without an alpha profile, revise the plan or use a public fallback; or
- `not_eligible`: clarify the request or prepare an optional capability
  feedback preview.

## Stage, offer, and approve

Only exact user-selected files are staged. Their content digests must already
be bound by the validated package, request, and plan.

```bash
oel hosted stage \
  --profile .oel/hosted/profile.json \
  --source completed-run.json \
  --input-id completed_run_input \
  --kind completed_run \
  --expected-sha256 SHA256

oel hosted offer \
  --profile .oel/hosted/profile.json \
  --request request.json \
  --plan proposed-plan.json \
  --grant GRANT_ID \
  --worker-image-sha256 IMAGE_SHA256

oel hosted approve \
  --profile .oel/hosted/profile.json \
  --offer hosted-offer.json \
  --idempotency-key USER_CHOSEN_KEY \
  --authorized-by USER_IDENTITY \
  --confirmation "APPROVE OFFER_SHA256"
```

Session tokens are issuer- and audience-bound, expire within eight hours, and
can be revoked by the service operator. Hosted profiles must remain mode 0600.
Upload grants are tenant-scoped, expiring, single-use records subject to both
per-object and cumulative tenant quotas.

The local proof records the pricing-policy digest, estimated usage, estimated
price, and maximum authorization. It meters and settles actual usage into a
`captured_local_proof` receipt but does not perform a real payment capture.

## Inspect, cancel, and import

```bash
oel hosted status JOB_ID --profile .oel/hosted/profile.json
oel hosted cancel JOB_ID --profile .oel/hosted/profile.json
oel hosted pull JOB_ID \
  --profile .oel/hosted/profile.json \
  --destination hosted-results/JOB_ID
```

The client verifies the result-transfer identity, every file size and digest,
and all relative paths before atomically creating the destination. It writes a
content-bound `result_import_receipt.json` beside the imported evidence.

## Transaction ledger

Client actions append to `.oel/hosted/transactions.jsonl` by default. The
ledger is a chained, content-bound local record of routing, grants, offers,
approval, durable status, pricing-policy and spending-cap identities, and
imported artifact counts and bytes. It does not retain input contents, result contents, model
credentials, conversation history, or hosted internal paths.

The ledger is product telemetry evidence for a controlled pilot; it is not a
payment receipt, scientific qualification, or proof of customer satisfaction.
