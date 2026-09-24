"""Orbital Engineering Lab Pro campaign review retention is not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Campaign review retention is part of Orbital Engineering Lab Pro. "
        "The public core supports deterministic single-run review evidence."
    )


validate_campaign_retention = _unavailable
policy_from_config = _unavailable
preflight_campaign_retention = _unavailable
configure_iteration = _unavailable
extract_iteration = _unavailable
compact_iteration_result = _unavailable
retire_iteration_review = _unavailable
write_campaign_dataset = _unavailable
