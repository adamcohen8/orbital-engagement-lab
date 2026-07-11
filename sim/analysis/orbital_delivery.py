"""Orbital Engagement Pro delivery-accuracy analysis is not included in the public core."""


def build_orbital_delivery_summary(*, cfg, t_s, truth_hist):
    section = dict(getattr(getattr(cfg, "analysis", None), "orbital_delivery", {}) or {})
    if bool(section.get("enabled", False)):
        raise ImportError(
            "Orbital-delivery accuracy analysis is part of Orbital Engagement Pro. "
            "The public core includes deterministic rocket and payload simulation without campaign scoring."
        )
    return {}


def aggregate_orbital_delivery_runs(runs):
    return {}
