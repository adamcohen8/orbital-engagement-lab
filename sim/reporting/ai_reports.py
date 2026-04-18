"""Orbital Engagement Pro AI reports are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "AI-assisted campaign reports are part of Orbital Engagement Pro. "
        "The public core includes deterministic single-run simulation, public plots, "
        "examples, and APIs without hosted LLM provider integrations."
    )


build_ai_report_packet = _unavailable
build_ai_report_request = _unavailable
estimate_ai_report_cost = _unavailable
load_ai_report_payload_from_outputs = _unavailable
write_ai_report_artifacts = _unavailable
write_ai_report_estimate_artifacts = _unavailable
