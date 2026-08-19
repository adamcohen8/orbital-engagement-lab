# Updating OEL

OEL installs engines side by side. Downloading or globally activating a newer
engine never rewrites scenario YAML, flight-software source, dependency locks,
tests, or prior outputs. A registered workspace keeps its exact engine pin
until its owner audits and explicitly adopts another version.

## Check And Install The Latest Release

```text
oel update status --full
oel update check
oel update install latest
```

The official installer persists the stable signed-channel URL and trusted
release keys. `check` therefore needs no URL or key arguments for an ordinary
managed installation. It performs a network read and rejects an invalid
signature, wrong edition/channel, expired or revoked trust key, or release-feed
rollback. `install latest` checks that channel, downloads the exact referenced
manifest and artifact, verifies their identity, and installs the release side
by side. It does not activate the release or edit a workspace.

An installation created before channel persistence can configure its official
endpoint once:

```text
oel update configure-channel https://<official-oel-host>/stable/channel.json
oel update check
```

Operators may still use the lower-level explicit forms for mirrors and release
diagnostics:

```text
oel update check --channel-url <signed-channel-url> --public-keys <trusted-keys.json>
oel update download <signed-release-manifest-url> --public-keys <trusted-keys.json>
oel update install <cached-release-manifest.json> --public-keys <trusted-keys.json>
```

`download` retains a bounded partial file for retry, then verifies exact size
and SHA-256. None of these commands activates a release or edits a workspace.
Key rotation is explicit: a new `oel.trusted-key-registry.v1` must be signed by
a currently trusted, non-revoked key and is installed with
`oel update rotate-trusted-keys <registry.json>`.

## Install And Activate

```text
oel update activate <version>
oel update rollback
```

Install extracts into an incomplete transaction directory, validates host and
Python compatibility, checks the archive source version against the signed
manifest, creates a constrained environment, runs `pip check`, writes a
content-bound installation record, and only then publishes the version.
Activation replaces the stable launcher and current selector as a recoverable
transaction. Global activation does not change registered workspace pins.

Pro releases also require `--license` and `--license-public-keys`. The license
must be valid, unexpired, for `oel-pro`, and permit the target version through
`allowed_versions`. Public release keys and customer-license keys are separate
trust roots.

## Audit And Adopt Per Workspace

```text
oel workspace check path/to/workspace --against <version> --release-manifest <manifest.json>
oel workspace migrate path/to/workspace --to <version> --release-manifest <manifest.json>
oel workspace migrate --apply-plan <migration-plan.json>
oel workspace use path/to/workspace <version>
oel workspace rollback path/to/workspace
```

The audit reads YAML, manifests, lock metadata, and prior evidence without
importing candidate code, executing a simulation, or using the network.
Possible dispositions are `compatible`, `compatible_with_warnings`,
`migration_available`, `manual_review`, `blocked`, `invalid`, `incomplete`,
and `cancelled`. `workspace use` accepts only installed verified engines and a
compatible audit. Migrations require a content-bound plan and explicit apply;
backups and a receipt remain beneath `.oel/migrations/`.

## Cleanup And Uninstall

```text
oel update cleanup
oel update cleanup --apply --keep 2
oel update uninstall <version>
oel update uninstall <version> --apply
```

Both commands default to a dry run. Current, previous, workspace-pinned, and
process-leased engines are protected. Uninstall never deletes registered
workspaces. Removed engines are recoverable by reinstalling the same verified
release artifact; user work is not part of an engine installation.

For a sanitized diagnostic artifact, run:

```text
oel support-receipt oel-support-receipt.json
```

The receipt omits workspace paths, user source, config contents, and customer
data and sends no telemetry.
