# OEL Workspaces

An OEL workspace is user-owned and lives outside immutable engine versions.
Users may edit its scenario configs, flight software, tests, dependency lock,
and generated outputs. The updater owns none of those files.

Create and inspect a workspace:

```text
oel workspace init path/to/workspace
oel workspace register path/to/workspace
oel workspace status path/to/workspace
```

The default layout is:

```text
workspace/
  oel-workspace.yaml
  requirements.lock
  configs/
  fsw/
  tests/
  outputs/
  .oel/
    template-manifest.json
    compatibility/
    migrations/
    receipts/
```

`oel-workspace.yaml` uses `oel.workspace.v1`. It records a supported engine
range, an exact `locked_version`, independently versioned scenario/FSW/candidate
contracts, confined relative paths, dependency-lock location, and network/code
trust policy. Workspace paths may not escape through `..`, absolute paths, or
symbolic links.

Commands launched below the workspace discover its manifest automatically, or
use `--workspace` explicitly. A workspace pin takes precedence over the global
current engine. The launcher creates a process lease so cleanup cannot remove
an engine while a run is active.

Every managed simulation summary records engine version, official/developer
disposition, signed release-manifest identity, installation transaction, and
workspace manifest identity. Prior outputs remain evidence for the engine and
workspace that produced them; auditing against a new engine does not silently
requalify old evidence.

Template updates are advisory:

```text
oel workspace template-check path/to/workspace \
  --target-manifest <template-manifest.json> --template-root <template-content>
```

The result classifies unchanged, upstream-changed, user-modified, new,
upstream-removed, and conflicting files. It never overwrites user content.
