# Security Incident Process

This process covers OEL vulnerabilities, leaked secrets, unsafe public-export
content, compromised dependencies, and release artifacts that should not have
been published.

## Report Intake

Report suspected security issues privately through GitHub Security Advisories
when available, or by contacting the repository maintainer through a private
channel. Do not open a public issue for vulnerabilities, secrets, customer data,
CUI, export-controlled data, or classified information.

Please include:

- affected version, commit, branch, or release artifact,
- reproduction steps or a minimal proof of concept,
- whether any secret, credential, controlled data, or customer data may be
  involved,
- suggested severity and impact,
- safe contact information for follow-up.

## Response Targets

These targets are best-effort for the public project and can be superseded by a
customer agreement:

- acknowledge receipt within 3 business days,
- provide initial triage within 10 business days,
- provide a remediation plan for critical/high findings within 30 calendar
  days,
- publish or privately deliver a fix as soon as practical after validation.

## Triage

Classify the issue by:

- affected surface: scenario loading, plugin import, path handling, dependency,
  public export, hosted AI/reporting, cFS/SIL integration, release process, or
  documentation,
- impact: code execution, credential exposure, data disclosure, integrity loss,
  denial of service, or misleading safety/validation claim,
- exposure: public release, private repo, generated export, customer package,
  local-only output, or CI artifact.

## Containment

Depending on impact:

- pause affected releases or PRs,
- remove or revoke leaked artifacts,
- rotate exposed credentials,
- disable or document unsafe workflows,
- regenerate the public export from clean private source,
- mark affected validation or release evidence as superseded,
- notify affected users or customers through the appropriate private channel.

## Remediation And Disclosure

Fixes should include focused tests or export checks when practical. Public
disclosure should avoid publishing exploit details until users have had a
reasonable opportunity to update. For customer/private incidents, follow the
applicable agreement and data-handling requirements before disclosing details.

## Post-Incident Review

After containment:

- record affected versions and artifacts,
- document root cause and corrective actions,
- update `SECURITY.md`, release checklists, public-export checks, or CI gates
  if they would have caught the issue,
- regenerate SBOM/dependency-audit/release evidence as needed,
- keep any sensitive incident notes out of public exports.
