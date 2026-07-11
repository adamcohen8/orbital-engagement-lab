# Documentation Index

Use this page as the documentation map. The goal is to give each audience a
short path instead of making every document look equally important.

## New Public User

Start here if you want to install OEL, run a scenario, and understand the
public simulation core.

1. [Quickstart](quickstart.md)
2. [Examples Matrix](examples-matrix.md)
3. [Scenario YAML](scenario-yaml.md)
4. [Python API](python-api.md)
5. [Known Limitations](known-limitations.md)

## Evaluator Or Buyer

Start here if you are deciding whether OEL is credible for research,
education, engineering prototyping, or Pro workflow evaluation.

1. [Product Inventory](product-inventory.md)
2. [Flagship RIC_PD 10 km Scenario And Validation](validation-ric-pd-10km.md)
3. [Physics Model Reference](physics-models.md)
4. [Validation Claims](validation-claims.md)
5. [Public Core And Pro Boundary](public-vs-pro.md)
6. [Security And Procurement](security/supply-chain.md)

## Agent User

Start here if you want an AI coding assistant to create, validate, run, or
inspect OEL scenarios.

1. [OEL Agents](oel-agents.md)
2. [Agent Capability Routing And Golden Paths](agent-capability-routing.md)
3. [Agent Task Runner](agent-task-runner.md)
4. [Agent Task Cards](agent-task-cards.md)
5. [Agent Review Query Recipes](agent-review-queries.md)
6. [Agent Feedback Loop](agent-feedback-loop.md)
7. [Agent Evaluation Packet](agent-evaluation-packet.md)

The root `AGENTS.md` and `agents/public/AGENTS.md` are the agent playbooks.
The docs above explain the supporting workflow, evidence, evaluation, and
feedback material.

## Workflow User

Start here when you already know the kind of work you want to run.

| Goal | Start with |
| --- | --- |
| Edit or author scenario YAML | [Scenario YAML](scenario-yaml.md) |
| Inspect completed outputs | [Review Store Contract](review-store.md) |
| Make custom tables or plots from a run | [Custom Analysis](custom-analysis.md) |
| Configure built-in figures | [Plotting](plotting.md) |
| View maintained plot examples | [Plot Gallery](plot-gallery.md) |
| Understand physics model assumptions | [Physics Model Reference](physics-models.md) |
| Explore actuator models | [Actuators](actuators.md) |
| Use the orbital calculator | [Orbital Calculator](orbital-calculator.md) |
| Use game/training mode | [Video Game Mode Roadmap](game-mode-roadmap.md) |
| Use ML/RL wrappers | [ML/RL Policy Contracts](ml-rl-contracts.md) |

## Pro Workspace User

These docs are available in the full private/Pro workspace and are omitted from
the public export when they describe private workflow acceleration or operating
processes.

1. Pro User Guide
2. Pro Python API
3. Campaign Analysis
4. Controller Bench
5. Pro Covariance Analysis
6. Pro AI Reports
7. Pro AI Config Assistant
8. Pro GNC Workbench
9. Validation Operations

In the Pro workspace, use the repository root README for buyer-facing Pro
positioning and local handoff notes for project continuity.

## Maintainer Or Contributor

Start here when changing behavior, contracts, release posture, or
public/private packaging.

1. [Engine Contract](contracts/engine-contract.md)
2. [Scenario YAML Contract](contracts/scenario-yaml-contract.md)
3. [Payload And Artifact Contract](contracts/payload-artifact-contract.md)
4. [Review Store Contract](review-store.md)
5. [Controller Naming Conventions](project/controller_naming_conventions.md)
6. [Reference GNC Library Roadmap](project/reference_gnc_library_roadmap.md)
7. [Data Handling And Boundary Statement](security/data-handling.md)
8. [Security Incident Process](security/incident-response.md)

Private maintainers should also use the Validation Operations and Release
Checklist guides available in the full workspace.

Private release/export operating notes live under `docs/operations/` in the
full workspace and are intentionally excluded from the public export.

## Internal Planning

Long-form roadmap, maturity, founder, and commercial-readiness files are
planning memory, not first-run or buyer-facing documentation. Keep them out of
the normal user path unless the task is explicitly about product strategy,
validation investment, release governance, or business/compliance planning.
