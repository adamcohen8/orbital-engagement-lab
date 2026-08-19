# Scenario Config Migrations

New scenario YAML should declare:

```yaml
schema_version: oel.scenario.v1
```

Unversioned historical configs remain readable as
`oel.scenario.legacy.v0` during the bounded compatibility window. Unknown
schema versions fail closed.

Migration is two-step and explicit. `oel workspace migrate PATH --to VERSION`
writes a compatibility report, proposed files, semantic unified diffs, original
and proposed SHA-256 digests, and a migration plan beneath `.oel/migrations/`.
It does not edit the workspace. Review the plan, then run
`oel workspace migrate --apply-plan PLAN`.

Apply refuses a plan when the workspace manifest, any target, or any proposed
file changed after planning. It backs up originals before replacement and
restores already-written files if the transaction fails. Reapplying a
successful plan returns its existing receipt. Ambiguous config changes,
plugin/candidate contract changes, FSW rewrites, and physics-changing defaults
remain manual-review items.

OEL never automatically rewrites flight-software source. A material FSW change
must produce a new content-bound candidate identity and new verification
evidence through the documented FSW boundary.
