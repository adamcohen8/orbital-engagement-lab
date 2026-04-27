# Release Checklist

Orbital Engagement Lab uses the private repository as the source of truth and
generates the public repository from private `main`.

## Private Feature Merge

Use this flow for normal development:

```bash
git switch private-main
git pull private main
git switch -c codex/<topic>

# make changes
.venv/bin/python tools/private_merge_check.py --allow-dirty
git add <changed files>
git commit -m "<summary>"
git push private codex/<topic>
```

After review, merge into private `main`:

```bash
git switch private-main
git pull private main
git merge --no-ff codex/<topic>
.venv/bin/python tools/private_merge_check.py
git push private private-main:main
git push private --delete codex/<topic>
git branch -d codex/<topic>
```

## Public Release

Public releases must come from clean private `main`, not from a feature branch.

Prepare the release metadata on a private branch first:

```bash
git switch private-main
git pull private main
git switch -c codex/release-<version>
```

Update:

- `pyproject.toml`
- `CHANGELOG.md`
- any release-specific docs

Merge that branch into private `main`, then run the public release gate:

```bash
.venv/bin/python tools/release_public.py --version <version>
.venv/bin/python tools/release_public.py --version <version> --push
```

The default command is a dry run. It regenerates the public export, checks the
public/private boundary, clones the public repository into a temporary checkout,
copies the export into that checkout, and runs public tests.

`--push` additionally commits the public export, creates tag `v<version>`, and
pushes public `main` plus the tag.

## Release Gate Requirements

- Private checkout is on `private-main`.
- `private-main` tracks `private/main`.
- Private checkout has no uncommitted tracked changes.
- `pyproject.toml` version matches the requested version.
- `CHANGELOG.md` contains `## <version> - YYYY-MM-DD`.
- `v<version>` does not already exist locally or in the public repository.
- Public export regenerates from private source.
- Public export integrity check passes.
- Banned private paths and sensitive patterns are absent from the export.
- Public checkout tests pass.
- Public diff is reviewed before `--push`.

## Branch Protection

Configure GitHub branch protection for both private and public `main` after
GitHub CLI authentication is working:

```bash
gh auth login -h github.com
.venv/bin/python tools/configure_branch_protection.py --require-pr
.venv/bin/python tools/configure_branch_protection.py --require-pr --apply
```

The private repo should require these status checks before merging:

- `public-core`
- `generated-public-export`

The public repo should require:

- `public-core`

GitHub may require a paid plan before branch protection can be enabled on a
private repository. If private branch protection is unavailable, still apply
public protection:

```bash
.venv/bin/python tools/configure_branch_protection.py --scope public --require-pr --apply
```

For the private repository fallback, keep using:

- private feature branches,
- `tools/private_merge_check.py` before merge,
- `tools/release_public.py` for public releases,
- and manual refusal to push directly to public `origin` outside the release
  gate.

Requiring PR review is intentionally stricter than the direct-push public
release path. If branch protection is enabled with `--require-pr`, publish the
public export through a release branch and PR, then tag the merged commit.

## Do Not

- Do not push private feature branches directly to the public repository.
- Do not publish public releases from any branch other than private `main`.
- Do not add customers to the full private repo as a first evaluation path.
- Do not bypass `tools/check_public_export.py` or `tools/release_public.py`.
