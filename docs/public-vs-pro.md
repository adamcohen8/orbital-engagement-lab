# Public Core And Pro Boundary

Orbital Engagement Lab is an open-core project. The public repository is a
complete, inspectable simulation foundation; the private Pro repository adds
workflow acceleration, scale, and customer-specific engineering support.

This page is the authoritative product and repository boundary. When another
document disagrees with it, this page and the checked-in public-surface manifest
govern.

## Public Core

The public core includes:

- deterministic single-run orbit and attitude simulation;
- OGP-SGP4/SDP4 passive catalog-style propagation and configurable ONP
  numerical propagation;
- public controllers, sensors, estimators, actuators, and mission primitives;
- the Public FSW Authoring Kit for ADCS/RPO stack scaffolding, safe inspection,
  trusted lifecycle validation, component tests, and one deterministic serial
  smoke run;
- scenario YAML, CLI, Python API, review-store queries, and plotting;
- deterministic whole-Earth coverage, point/region/rich-footprint queries,
  free-space directed-link budgets, communications coverage, constellation
  aggregation, bounded tasking, cadence sensitivity, event refinement, and
  their ONP/OGP/review-history adapters;
- a bounded two-body Lambert transfer planner and mission-recovery estimates;
- public examples, the RPO Trainer, and reproducible public validation evidence;
- educational rocket/ascent primitives and public rocket GNC contracts.

The public core is intended for research, education, prototyping, and
inspectable engineering experiments. It is not flight-qualified or an
operational decision system.

## Pro Layer

Pro adds workflows whose value comes from repeatability, scale, search, or
review-ready packaging:

- Monte Carlo, sensitivity, covariance, and campaign orchestration;
- controller benchmarks, comparison reports, optimization, and gain tuning;
- OEL Scale catalog screening, refinement, operational stores, and synthetic
  data generation;
- intent-hypothesis evaluation and maneuver-investigation workflows;
- orbit determination against external observations or precise products;
- curated validation automation and private release/customer evidence;
- AI-assisted reports and config assistance with explicit review and cost
  gates;
- frontier-model evaluation harnesses, hidden evaluator truth, model-provider
  execution, scoring, and campaign evidence;
- custom GNC workbenches, spacecraft packages, cFS/SIL, and program-specific
  integrations;
- the private FSWDK workflow superset: Controller Bench, tuning, qualification,
  baseline promotion, packaged review evidence, and external-process candidates;
- private `agents/pro/` instructions and capability routing for Pro workflows;
- private onboarding, support, scenario migration, and customer deliverables.

The public Lambert planner is a bounded two-body trade-space tool. It does not
make general optimization, uncertainty analysis, or operational maneuver
planning public.

Public coverage sensitivity compares explicitly supplied deterministic
coverage products; it does not make Pro campaign orchestration public. Public
bounded tasking and constellation aggregation do not include managed
multi-asset scheduling, customer catalogs, operational-scale optimization,
weather/interference services, or proprietary calibrated equipment data.

Frontier-model evaluation source, configs, tests, provider integrations, and
generated evidence remain private. Public release notes may describe the
boundary and compatibility work, but that does not promote the evaluator into
the public core.

## Promotion Rule

Promote a capability when it improves adoption, inspectability, education, or
trust in the core and can be published safely with honest limits.

Keep a capability private when its primary value is expert labor reduction,
analysis at scale, customer-specific integration, proprietary data handling, or
packaged mission/release evidence.

Public-safe validation evidence belongs public when it is reproducible,
redistributable, and tied to a bounded claim. Customer data, proprietary
reference material, generated private reports, and program-specific evidence do
not.

## Repository And Export Model

The private repository is the source of truth. Public releases are generated,
not hand-curated:

1. Develop and validate changes privately.
2. Deliberately promote public paths in
   `docs/operations/public_surface_manifest.yaml`.
3. Keep private surfaces in the defense-in-depth exclusions in
   `docs/operations/public_export_exclude.txt`.
4. Generate the export with `tools/export_public.py`.
5. Run `tools/check_public_export.py` and the release gate.
6. Review the generated public diff before opening or updating a public PR.

Anything outside the positive public manifest is private by default. Never push
the private working tree directly to the public repository.

## Access Model

Public examples require neither Pro nor hosted AI accounts. Early Pro access is
handled through private engineering pilots whose scope should state supported
workflows and versions, license terms, expected evidence, support boundaries,
and any security, procurement, data-handling, or export constraints.

For detailed private workflows, use the Pro User Guide in the full workspace.
For public limitations, use [Known Limitations](known-limitations.md).
