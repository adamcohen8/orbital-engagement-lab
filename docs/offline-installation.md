# Offline And Air-Gapped Installation

An offline release bundle is a ZIP containing the signed release manifest,
source artifact, trusted public release keys, checksums, installers, and the
content-bound supply-chain evidence shipped with that release.
Official offline bundles are platform/architecture specific and contain an
exact signed wheel inventory. Runtime creation forces pip `--no-index` against
that wheelhouse; a missing, extra, changed, or incomplete wheelhouse fails
closed instead of contacting a package index.

On a connected verification host, inspect the bundle inventory, verify the
published bundle digest through the approved out-of-band channel, and transfer
it using local policy. On the target host:

```text
oel update install-bundle <oel-public-version.bundle.zip> --public-keys <trusted-release-keys.json>
oel update activate <version>
oel update status --full
oel doctor
```

The bundle path performs no network operation. Archive traversal, links,
devices, duplicate members, excessive expansion, signature failure, hash/size
mismatch, wrong platform, unsupported Python, and source/manifest version
mismatch are rejected before activation.

For Pro bundles, also supply the offline license and its separate public-key
registry. Expired licenses and versions outside `allowed_versions` are
rejected. External runtimes such as cFS, MATLAB, GPU drivers, or ML stacks are
reported as qualified prerequisites; their absence does not broaden core
claims or cause the updater to fabricate replacements.

Rollback and cleanup are local operations. Retain at least the current and
previous engine, every workspace-pinned engine, the signed bundle, and its
published digest according to site policy.
