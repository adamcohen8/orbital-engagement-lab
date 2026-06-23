# Orbital Engagement Pro

Orbital Engagement Pro is the workflow layer around the public Orbital
Engagement Lab simulation core. The public core is useful on its own for
deterministic single-run simulation, scenario YAML, Python/API workflows,
review artifacts, examples, and the RPO trainer. Pro is for teams that need to
run those capabilities repeatedly, compare alternatives, preserve evidence, and
turn results into review-ready engineering artifacts.

Pro is currently best framed as a paid pilot or private engineering evaluation,
not a self-serve hosted product.

## What Pro Adds

Pro focuses on repeatable engineering workflows:

- campaign analysis for Monte Carlo, sensitivity, parameter sweeps, and batch
  comparisons;
- controller-benchmark suites, comparison reports, and gain-tuning workflows;
- validation automation, reference-data comparison workflows beyond the public
  trust baseline, and mission-assurance scenario packs;
- covariance propagation, encounter uncertainty screening, and
  orbit-determination workflows;
- AI-assisted reports and AI config assistance with local review gates and cost
  estimation before hosted model calls;
- custom GNC workbench scaffolding for trusted local controller/plugin work;
- spacecraft package and digital-twin workflows for proprietary object setup,
  source evidence, and reusable customer-specific packages;
- private distribution, onboarding, support, scenario migration, custom metrics,
  report templates, and integration assistance.

The public core remains the simulation foundation. Pro is intended to reduce
the labor around study setup, campaign execution, evidence inspection, and
engineering review.

## Example Pro Use Cases

Pro is a fit when a team wants to:

- compare controller variants across a repeatable scenario suite;
- run uncertainty or sensitivity studies and preserve the generated evidence;
- turn a completed study into a review packet with plots, tables, JSON, CSV,
  SQLite review stores, and report drafts;
- migrate existing scenario concepts into durable YAML/API workflows;
- package program-specific spacecraft assumptions or object definitions without
  putting proprietary source material into the public core;
- evaluate OEL in an offline, local-first, or restricted-provider environment.

## Access Model

Early Pro access is handled through private engineering pilots. A pilot package
can be shipped as a local source distribution with a signed offline license,
public verification keys, install instructions, and a package manifest.

Typical pilot scope should name:

- the Pro workflows included;
- the supported OEL version range;
- the license duration;
- whether the license is machine-bound;
- the expected evidence or handoff artifacts;
- any data-handling, export-control, procurement, or security constraints the
  customer needs to evaluate.

## Boundary

The public repository does not require Pro to run its documented public
examples. Public examples do not require hosted AI accounts or API keys.

Pro-only workflow internals, customer packages, proprietary spacecraft package
workflows, provider-specific AI report artifacts, private release/customer
validation evidence, and program-specific integrations are not included in the
public export. Public-safe validation evidence that establishes trust in the
core belongs in the public repository.

For the packaging boundary, see [Public Core And Pro Boundary](public-vs-pro.md).
