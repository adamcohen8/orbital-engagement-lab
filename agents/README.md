# OEL Agents

This directory contains agent-facing instructions, examples, and templates for
using Orbital Engagement Lab with AI coding assistants.

- `public/AGENTS.md` is the public-safe playbook for open-source OEL agents.
- `public/evaluation-rubric.md` is the checklist for judging generated
  scenarios and completed runs.
- `examples/` contains small scenario YAML examples that agents can copy from
  when they fit, use as scaffolds, or validate in smoke tests. They are not a
  complete catalog of supported agent workflows.
- `../docs/agent-golden-paths.md` gives the shortest reproducible adoption
  workflows for propagation, rendezvous, and mission recovery/reconstitution.
- `../docs/agent-capability-routing.md` maps broad user intents to public
  workflows, starting docs, evidence to inspect, clarifying-question triggers,
  and limits agents should not overclaim.
- `../docs/agent-evaluation-packet.md` is the evaluator-facing prompt packet
  for testing whether an agent follows the OEL evidence loop.
- `../docs/agent-review-queries.md` contains reusable review-store SQL recipes
  for evidence-backed summaries.
- `../docs/agent-feedback-loop.md` explains how agents can ask permission to
  submit public-safe workflow feedback upstream.
- `../docs/agent-task-cards.md` is the maintained public agent task-card index.
- Private deployment guidance belongs outside the public agent docs and public
  export surface.

Design principle: AI agents orchestrate OEL workflows; deterministic OEL
simulation code remains the authority for physics, control behavior, output
artifacts, and reported results. Task cards and examples test that principle;
they do not replace general user-intent handling.
