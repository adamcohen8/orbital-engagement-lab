"""Public-core stubs for Orbital Engineering Lab Pro feature checks."""

FEATURE_OBJECT_PARALLELISM = "object_parallelism"


def require_pro_feature(feature: str, **_kwargs):
    raise ImportError(
        f"Feature {feature!r} is available in Orbital Engineering Lab Pro and is not included in the public core."
    )
