# Security

Orbital Engagement Lab is research and prototyping software. It is not
flight-qualified software and should not be used as an operational decision
system without independent mission-specific validation.

## Reporting A Vulnerability

Please report suspected security issues privately through GitHub Security
Advisories when available, or to the repository maintainers through a private
channel. Do not open a public issue for vulnerabilities, leaked secrets,
customer data, CUI, export-controlled data, or classified information.

Include:

- a short description of the issue
- steps to reproduce
- affected versions or commits, if known
- any relevant logs or proof-of-concept details

Best-effort response targets for the public project:

- acknowledge receipt within 3 business days,
- provide initial triage within 10 business days,
- provide a remediation plan for critical/high findings within 30 calendar
  days,
- publish or privately deliver a fix as soon as practical after validation.

Customer or pilot agreements may define stricter response targets.

## Supported Versions

Security fixes target the current public release line, currently `v0.21.1`, and
active private/Pro customer-supported release lines. The project targets Python
3.10 through 3.12. Blocking CI currently exercises Python 3.11, so evaluators
should run acceptance on their exact target interpreter and operating system.
Python 3.9 is no longer a supported procurement baseline because several
vulnerability fixes in the Python packaging and ML/plotting ecosystem require
Python 3.10 or newer.

## Supported Scope

Security reports are most useful when they affect:

- arbitrary code execution through scenario loading or plugin pointers
- unsafe handling of local files or paths
- dependency or packaging risks
- external integration or adapter surfaces

Simulation-model correctness issues are also welcome, but they may be handled as
engineering bugs rather than security vulnerabilities unless they create a clear
security impact.

For non-security bugs, use the public repository's GitHub Issues and the Bug
Report template. Do not include sensitive data in public issues.

## Running Untrusted Scenarios

Treat scenario YAML as trusted input. Normal validation imports configured
Python plugin modules so it can check controller, guidance, bridge, and mission
contracts. For a first pass over a scenario from an untrusted source, use:

```bash
.venv/bin/python run_simulation.py --config <path> --safe-validate
```

Safe validation checks YAML structure, path policy, and plugin pointer shape
without importing configured plugin modules. It does not make the scenario safe
to execute.

For shared, classroom, government, or other restricted environments, use sealed
mode for validation and execution:

```bash
.venv/bin/python run_simulation.py --config <path> --sealed-mode --validate-only
.venv/bin/python run_simulation.py --config <path> --sealed-mode
```

Sealed mode allows built-in OEL plugin modules, but blocks arbitrary plugin
module imports, hosted AI providers, custom AI endpoints, non-loopback cFS/SIL
UDP networking, full run logs, full review stores, raw Monte Carlo payloads,
and non-summary AI report packets by default. Each exception requires an
explicit CLI opt-in such as `--allow-untrusted-plugin-imports`,
`--allow-hosted-ai`, `--allow-custom-ai-endpoints`,
`--allow-non-loopback-sil`, or `--allow-high-detail-outputs`.

External paths, external AI prompt files, custom AI endpoints, forwarding hosted
provider API keys to custom endpoints, insecure custom AI endpoints, and
non-loopback cFS/SIL UDP binds all require explicit opt-in flags or environment
variables. Use those only for trusted local or isolated-network workflows.

## Dependency Audit

Release and scheduled CI run a Python dependency audit with `pip-audit`. Local
release review can run the same check with:

```bash
.venv/bin/python -m pip install -U pip-audit
.venv/bin/python -m pip_audit
```

Generate supply-chain evidence for procurement review with:

```bash
.venv/bin/python tools/generate_python_sbom.py --output outputs/supply_chain/sbom.cdx.json
.venv/bin/python -m pip freeze --all > outputs/supply_chain/python-freeze.txt
.venv/bin/python -m pip_audit --format json --output outputs/supply_chain/pip-audit.json
```

See:

- `docs/security/supply-chain.md`
- `docs/security/data-handling.md`
- `docs/security/incident-response.md`
