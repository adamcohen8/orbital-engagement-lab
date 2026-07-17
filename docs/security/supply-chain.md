# Supply Chain And Procurement Baseline

This page is the OEL baseline for software supply-chain review. It is intended
to give evaluators a clear starting packet, not to replace a customer's own
security, legal, export-control, or mission-assurance process.

## Supported Versions

- Public releases: security fixes target the current public release line,
  currently `v0.21.1`.
- Private/Pro releases: security fixes target the active customer-supported
  release line or pilot branch named in the agreement.
- Python compatibility target: Python 3.10 through 3.12. Blocking CI currently
  exercises Python 3.11; run local acceptance on the exact interpreter used for
  an evaluation or deployment. Python 3.9 is not a supported procurement
  baseline because several vulnerability fixes require Python 3.10 or newer.
- Operating systems: Linux and macOS are the primary development and test
  targets. Windows is best-effort unless a customer package explicitly includes
  Windows acceptance evidence.

## SBOM

Generate a CycloneDX JSON software bill of materials for the installed Python
environment:

```bash
.venv/bin/python tools/generate_python_sbom.py --output outputs/supply_chain/sbom.cdx.json
```

The SBOM records the installed Python distributions, versions, and PyPI package
URLs visible in the current interpreter environment. Generate it after
installing the exact profile under review, for example `.[dev]`, `.[game]`, or
`.[full]`.

## Dependency Audit

Run the dependency audit after installing the evaluated profile:

```bash
.venv/bin/python -m pip install -U pip-audit
.venv/bin/python -m pip_audit --format json --output outputs/supply_chain/pip-audit.json
```

CI also runs dependency audit evidence. Treat a known vulnerability as a release
finding until it is upgraded, removed, documented as not applicable, or accepted
by the evaluator in writing.

Current release audit exceptions:

- `CVE-2025-3000` / `GHSA-rrmf-rvhw-rf47` in `torch`: accepted for the
  optional ML profile while the audit feed advertises no fixed Torch release.
  The advisory applies to `torch.jit.script`; OEL's checked-in ML workflows use
  eager-mode model definitions and training helpers and do not call
  `torch.jit.script`. Revisit this exception when a fixed Torch release is
  available or if OEL adds TorchScript/JIT model export or execution.
- `PYSEC-2026-3447` in the runtime copy of `setuptools`: accepted for the
  optional ML/full profiles because `torch==2.12.0` currently requires
  `setuptools<82`, while the advisory fix begins at `setuptools==83.0.0`.
  OEL's isolated build environment requires `setuptools>=83`, and release
  builds operate only on the controlled OEL source tree; they do not apply
  `MANIFEST.in` exclusion rules to untrusted source trees. Revisit this
  exception when Torch permits setuptools 83 or newer. This exception does not
  apply to build isolation, which must continue to use the patched requirement.

## Reproducible Dependency Workflow

OEL source dependency ranges live in `pyproject.toml` and `requirements/`.
For a procurement or release review, create an environment-specific frozen
record after installation:

```bash
.venv/bin/python -m pip freeze --all > outputs/supply_chain/python-freeze.txt
```

For higher-assurance installs, use the frozen record as input to a customer
lockfile or hash-checked requirements workflow. The recommended buyer-side gate
is:

- install from a clean checkout and selected profile,
- generate `python-freeze.txt`,
- generate the SBOM,
- run `pip-audit`,
- archive those artifacts with the validation evidence matrix and release
  report.

## GitHub Actions Pinning

OEL currently uses semantic action versions such as `actions/checkout@v4` and
`actions/setup-python@v5`. For organizations that require full SHA pinning,
treat the workflow as follows:

- convert each third-party `uses:` entry to a full commit SHA before a
  controlled release branch is approved,
- record the upstream action repository, tag, and resolved SHA in the release
  evidence packet,
- review and refresh pinned SHAs on a planned cadence or when GitHub issues a
  security advisory,
- keep the public release PR draft until CI has passed on the pinned workflow.

## Release Evidence Packet

A procurement-ready packet should include:

- `outputs/supply_chain/sbom.cdx.json`,
- `outputs/supply_chain/pip-audit.json`,
- `outputs/supply_chain/python-freeze.txt`,
- public/private export integrity check output,
- validation evidence matrix and harness reports when available,
- release checklist result and version/commit provenance.

## Non-Claims

The SBOM and audit evidence do not make OEL flight-qualified, FedRAMP
authorized, export-control classified, or approved for classified/CUI handling.
They are review artifacts that help an evaluator decide what additional
controls are required for their environment.
