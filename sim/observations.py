"""Orbital Engagement Pro observation data models are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Observation data ingestion and normalization are part of Orbital Engagement Pro. "
        "The public core supports simulator-generated sensor and truth histories."
    )


ObservationRecord = _unavailable
ObservationSeries = _unavailable
load_observations = _unavailable
ObservationPacket = _unavailable
fit_state_from_position_observations = _unavailable
ingest_observations = _unavailable
inspect_observation_packet = _unavailable
kalman_filter_position_observations = _unavailable
load_observation_packet = _unavailable
observation_packet_from_dict = _unavailable
