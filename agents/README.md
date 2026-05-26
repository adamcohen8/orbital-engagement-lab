# OEL Agents

This directory contains agent-facing instructions, examples, and templates for
using Orbital Engagement Lab with AI coding assistants.

- `public/AGENTS.md` is the public-safe playbook for open-source OEL agents.
- `public/evaluation-rubric.md` is the checklist for judging generated
  scenarios and completed runs.
- `examples/` contains small scenario YAML examples that agents can copy from
  or validate in smoke tests.
- Private deployment guidance belongs outside the public agent docs and public
  export surface.

Design principle: AI agents orchestrate OEL workflows; deterministic OEL
simulation code remains the authority for physics, control behavior, output
artifacts, and reported results.
