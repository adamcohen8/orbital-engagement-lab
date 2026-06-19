"""Orbital Engagement Pro data-ingestion tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Data ingestion and observation normalization are part of Orbital Engagement Pro. "
        "The public core supports deterministic scenarios and local simulation artifacts."
    )


load_observation_file = _unavailable
normalize_observations = _unavailable
write_ingestion_manifest = _unavailable
MissionInputPacket = _unavailable
build_basic_propagation_scenario = _unavailable
build_basic_rpo_scenario = _unavailable
ingest_coes = _unavailable
ingest_ephemeris_object_set = _unavailable
ingest_ephemeris_samples = _unavailable
ingest_relative_ric_state = _unavailable
ingest_satellite_card = _unavailable
ingest_state_vector = _unavailable
ingest_tle = _unavailable
inspect_packet = _unavailable
load_packet = _unavailable
merge_packets = _unavailable
packet_from_dict = _unavailable
render_ingestion_summary = _unavailable
