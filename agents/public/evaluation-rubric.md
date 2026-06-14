# Agent Scenario Evaluation Rubric

Use this rubric after an agent creates or edits an OEL scenario. The goal is to
decide whether the scenario is valid, runnable, interpretable, and honest about
its limits.

## 1. Config Readiness

- The scenario YAML lives in an expected project path.
- It uses `objects:` for active spacecraft.
- Units are explicit in field names.
- Plugin pointers refer to public modules/classes that exist in the checkout.
- Plots and animations are intentionally enabled or disabled.
- The config passes:

```bash
.venv/bin/python run_simulation.py --config <scenario.yaml> --validate-only
```

## 2. Execution Evidence

- The scenario runs through `run_simulation.py`.
- The run writes an output directory.
- `index.md` exists when the output writer creates one.
- `master_run_summary.json` exists for single-run review.
- `review/run.sqlite` exists when `outputs.review.enabled: true`.
- The agent uses `.venv/bin/python -m sim.review` for review-store evidence when the
  review store exists.
- `master_run_log.json` exists only when detailed time-history output was
  requested with `outputs.stats.save_full_log: true`.
- Any requested plots, CSV files, or custom artifacts are present.
- The agent records the exact command used to produce the artifacts.

## 3. Physical Interpretation

- The agent identifies the active objects and their roles.
- The agent states the dynamics model, duration, timestep, and relevant
  controller choices.
- The agent explains what changed physically: orbit propagation, relative
  motion, attitude response, access windows, thrusting, or termination.
- The agent reports metrics from artifacts, not from memory.
- The agent states the SQL query or artifact path used for important claims.
- The agent distinguishes simulated model behavior from real mission behavior.

## 4. Goal Fit

- The scenario matches the user's stated goal.
- The initial conditions and duration are appropriate for that goal.
- The output artifacts are sufficient to inspect the goal.
- The agent names missing evidence when the artifacts are insufficient.
- For rendezvous scenarios, the agent distinguishes partial closure from
  terminal rendezvous success and checks user-defined range, time, delta-v, and
  safety thresholds before claiming success.
- For complex controller scenarios, the agent recommends `save_full_log`, CSV,
  or plots when compact summary JSON cannot show trajectory shape, transients,
  or monotonic closure.

## 5. Safety And Limits

- The scenario came from a trusted source.
- The agent did not execute untrusted plugin code.
- No API keys, customer data, local-only files, or generated report packets were
  committed.
- The agent does not claim flight readiness or operational decision authority.
- The agent does not claim validation beyond checked artifacts and tests.
- If the agent proposes upstream feedback, it follows
  `docs/agent-feedback-loop.md`, shows a public-safe draft, and asks for user
  approval before submitting.

## Suggested Result Summary

```text
Status: validated / ran / needs fixes
Scenario: <scenario_name>
Command: <command>
Outputs inspected: <files>
Review query: <SQL query or not applicable>
What happened: <artifact-supported summary>
Key metrics: <metric names and values>
Goal fit: <supported / partial / unsupported>
Limitations: <missing evidence or model caveats>
Next run: <one focused follow-up>
```
