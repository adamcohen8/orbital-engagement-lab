# Agent Feedback Loop

OEL Agents can help improve Orbital Engagement Lab when they encounter
workflow friction, confusing documentation, missing examples, validation
messages that are hard to act on, or output artifacts that do not support the
user's question.

The feedback loop is opt-in. Agents must never submit feedback silently.

## Intended Loop

```text
user asks for OEL work
-> agent creates or edits scenario YAML
-> agent validates, runs, and inspects artifacts
-> agent identifies a feedback-worthy issue
-> agent prepares a public-safe feedback draft
-> agent asks the user for permission
-> user approves
-> agent opens a GitHub Issue using the Agent Feedback template
```

Agents orchestrate the workflow. The deterministic simulator, validators,
review store, and saved artifacts remain the evidence source.

## What Counts As Agent Feedback

Use agent feedback for product or workflow signals discovered while an agent is
trying to help a user. Good examples include:

- no nearby public example exists for a reasonable public request,
- docs do not explain which default the agent should choose,
- validation catches an error but the message is hard to act on,
- a scenario runs but the output artifacts cannot answer the user's question,
- the review store is missing a table or field needed for agent analysis,
- the agent had to ask a clarifying question that a template could avoid,
- agent guidance conflicts across `AGENTS.md`, docs, examples, or tests,
- the public/private or safety boundary is unclear.

Use the normal Bug Report template when the issue is a concrete defect with a
minimal reproduction. Use the Documentation Issue template for a simple docs
correction. Use the private `SECURITY.md` process for vulnerabilities, leaked
secrets, customer data, CUI, export-controlled data, classified information, or
sensitive generated artifacts.

## What Agents Must Not Send

Do not include:

- secrets, API keys, credentials, tokens, or private endpoints,
- customer data, proprietary configs, or local-only files,
- CUI, export-controlled data, classified information, or operational details,
- private generated report packets,
- full run logs or output folders unless they are explicitly public-safe and
  necessary,
- personal information that is not needed to reproduce the issue.

Prefer a short public-safe summary, minimal YAML snippet, command, validation
message, review query, and artifact path.

## Permission Prompt

Before opening an issue, the agent should show the user exactly what it wants
to report:

```text
I found a possible OEL agent-workflow issue:

Summary: <one sentence>
Why it matters: <short explanation>
Public-safe details I would send:
- User goal, paraphrased: <goal>
- Workflow stage: <stage>
- Command/query/artifact evidence: <evidence>
- Suggested improvement: <suggestion>

I will not include secrets, customer data, CUI, export-controlled data,
classified information, private configs, or private generated report packets.

May I open a public GitHub Issue with this feedback?
```

If the user does not approve, do not submit feedback. If the user approves,
open an issue with the Agent Feedback template and link the issue back to the
user.

Agents can prepare a local draft without submitting anything:

```bash
python tools/prepare_agent_feedback.py \
  --agent-tool Codex \
  --workflow-stage "Review-store query" \
  --user-goal "Summarize a public rendezvous run from saved evidence." \
  --summary "The review store did not expose the metric the agent needed." \
  --expected "The agent should be able to query the required metric directly." \
  --suggestion "Add a saved review query or documented metric row."
```

The helper prints Markdown for the GitHub Agent Feedback issue template. It
does not submit feedback or contact GitHub.

## Feedback-Worthy Stages

Classify the feedback by the stage where the friction appeared:

- scenario generation,
- config validation,
- simulation execution,
- review-store query,
- artifact interpretation,
- docs and examples,
- agent safety or public/private boundary,
- install or environment,
- game or training workflow,
- other.

## Evidence To Include

Useful agent feedback includes:

- agent/tool used, such as Codex, Claude Code, Cursor, Gemini CLI, or Grok
  Build,
- the user goal paraphrased without sensitive details,
- the workflow stage,
- the closest public example or doc used,
- exact command or review query when relevant,
- validation error, traceback excerpt, or artifact path when public-safe,
- what the agent expected,
- what happened instead,
- why the agent thinks the feedback is worth reporting,
- one suggested improvement.

## Triage Labels

The Agent Feedback issue template uses the `agent-feedback` label. Maintainers
can also add:

- `documentation` for docs/example gaps,
- `bug` for confirmed defects,
- `enhancement` for new workflow requests,
- `security` only after moving sensitive discussion to the private process.
