"""Orbital Engagement Pro OD observation batches are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Observation batches for orbit determination and parameter estimation are part of Orbital Engagement Pro."
    )


ObservationBatch = _unavailable
