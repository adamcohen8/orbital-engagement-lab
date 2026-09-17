"""Small routing catalog derived from the active registry, without authorizing effects."""

from __future__ import annotations

from collections.abc import Iterable

BOOTSTRAP_URI = "oel://agent/bootstrap/v1"
WORKFLOWS_URI = "oel://agent/workflows/v1"


def workflow_routes(tool_ids: Iterable[str]) -> dict:
    active = frozenset(tool_ids)
    routes = (
        ("scenario", "Validate and execute an existing scenario", (
            "oel.plan_run.v1", "oel.validate_scenario.v1", "oel.run_scenario.v1", "oel.inspect_run.v1")),
        ("study_plan", "Resolve a typed study proposal before execution", (
            "oel.study.capabilities.v1", "oel.study.preflight.v1", "oel.study.plan_review.v1")),
        ("review", "Answer a question from completed evidence", (
            "oel.inspect_run.v1", "oel.query_review.v1")),
        ("comparison", "Compare completed runs", ("oel.compare_runs.v1",)),
        ("plot", "Plan and render a custom evidence plot", (
            "oel.plan_review_plot.v1", "oel.render_review_plot.v2")),
    )
    return {
        "schema_version": 1,
        "bootstrap_uri": BOOTSTRAP_URI,
        "routes": [
            {"route_id": name, "purpose": purpose, "tools": list(steps),
             "all_tools_registered_here": all(step in active for step in steps),
             "missing_tools": [step for step in steps if step not in active]}
            for name, purpose, steps in routes
        ],
        "study_planner": {
            "maturity": "prototype", "separate_server_command": "oel-study-mcp",
            "registered_here": "oel.study.preflight.v1" in active,
            "connection_action": "A host operator configures the separate local planner when needed.",
            "outcomes": ["PLAN_VALID", "CLARIFICATION_REQUIRED", "UNSUPPORTED"],
            "execution_authorized": False,
        },
        "authoring": {
            "main_scenario_input": "existing config_path",
            "route": "Documented scenario YAML or sim.api.ScenarioBuilder through authorized host file tools.",
            "arbitrary_file_write_tool": False,
        },
        "rules": [
            "Registered tools may still require configured operator approvals and entitlements.",
            "Inspect describe_capabilities readiness; it never authorizes an exact input or run.",
            "A missing route does not authorize changing servers or using a more permissive data boundary.",
            "CLI/Python run lifecycle v1 is separate from MCP execution; retain exact run identity.",
        ],
    }
