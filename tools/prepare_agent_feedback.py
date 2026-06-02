from __future__ import annotations

import argparse
from pathlib import Path

SENSITIVE_REMINDER = (
    "Do not include secrets, API keys, customer data, CUI, export-controlled data, "
    "classified information, private configs, or private generated report packets."
)


def build_agent_feedback_issue(
    *,
    agent_tool: str,
    workflow_stage: str,
    user_goal: str,
    issue_summary: str,
    expected: str,
    evidence: str = "",
    suggestion: str = "",
) -> str:
    return "\n".join(
        [
            "# Agent Feedback Draft",
            "",
            "> Public safety reminder: " + SENSITIVE_REMINDER,
            "",
            "## Agent/tool used",
            "",
            agent_tool.strip() or "Not specified",
            "",
            "## Workflow stage",
            "",
            workflow_stage.strip() or "Not specified",
            "",
            "## User goal, paraphrased",
            "",
            user_goal.strip() or "Not specified",
            "",
            "## What the agent noticed",
            "",
            issue_summary.strip() or "Not specified",
            "",
            "## Public-safe evidence",
            "",
            evidence.strip() or "No public-safe evidence supplied.",
            "",
            "## Expected agent workflow",
            "",
            expected.strip() or "Not specified",
            "",
            "## Suggested improvement",
            "",
            suggestion.strip() or "No suggestion supplied.",
            "",
            "## Consent and public safety checklist",
            "",
            "- [ ] The user approved submitting this public feedback.",
            "- [ ] No secrets, API keys, customer data, CUI, export-controlled data, classified information, private configs, or private generated report packets are included.",
            "- [ ] This is not a suspected vulnerability or sensitive-data exposure. If it is, use the private SECURITY.md process instead.",
            "",
            "Submit with the GitHub Agent Feedback issue template only after user approval.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a public-safe OEL Agent Feedback issue draft without submitting it."
    )
    parser.add_argument("--agent-tool", required=True, help="Agent/tool used, such as Codex, Claude Code, Cursor, Gemini CLI, or Grok Build.")
    parser.add_argument("--workflow-stage", required=True, help="Workflow stage where feedback appeared.")
    parser.add_argument("--user-goal", required=True, help="Public-safe paraphrase of the user's goal.")
    parser.add_argument("--summary", required=True, help="What the agent noticed.")
    parser.add_argument("--expected", required=True, help="Expected agent workflow or product behavior.")
    parser.add_argument("--suggestion", default="", help="Suggested improvement.")
    parser.add_argument("--evidence", default="", help="Public-safe evidence text.")
    parser.add_argument("--evidence-file", help="Read public-safe evidence text from this file.")
    parser.add_argument("--output", help="Write draft Markdown to this path. Defaults to stdout.")
    args = parser.parse_args(argv)

    evidence = str(args.evidence or "")
    if args.evidence_file:
        evidence_path = Path(args.evidence_file)
        evidence = evidence_path.read_text(encoding="utf-8")

    draft = build_agent_feedback_issue(
        agent_tool=args.agent_tool,
        workflow_stage=args.workflow_stage,
        user_goal=args.user_goal,
        issue_summary=args.summary,
        expected=args.expected,
        evidence=evidence,
        suggestion=args.suggestion,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(draft, encoding="utf-8")
    else:
        print(draft, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
