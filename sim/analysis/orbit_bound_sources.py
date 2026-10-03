"""Generate and replay collection/link evidence from one retained ECI history."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.analysis.collection_opportunity import assess_collection_opportunities
from sim.analysis.directed_link import (
    DirectedLinkConfig,
    LinkTerminal,
    TerminalPattern,
    evaluate_directed_link,
    fixed_wgs84_site_history,
    spacecraft_endpoint_history,
    write_directed_link_artifacts,
)
from sim.analysis.history_adapters import AnalysisHistory
from sim.analysis.orbit_history_product import orbit_history_semantic_sha256
from sim.dynamics.orbit.frames import FrameContext
from sim.utils.io import read_regular_file_nofollow

ORBIT_LINK_BINDING_SCHEMA = "oel.orbit_bound_link.v1"


class OrbitBoundSourceError(ValueError):
    """Raised when a collection or link product differs from its shared orbit."""


_EXACT_NUMERIC_FIELDS = frozenset({
    "time_s", "start_s", "end_s", "collection_start_s", "collection_end_s",
    "collection_duration_s", "duration_s", "storage_delta_bytes", "objective_value",
    "generated_data_bytes", "sample_count",
})


def _same_evidence(left: Any, right: Any, *, field: str = "") -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_evidence(left[key], right[key], field=key) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_evidence(a, b, field=field) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float):
        if field in _EXACT_NUMERIC_FIELDS:
            return left == right
        return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-10)
    return left == right


def verify_orbit_bound_collection(path: str | Path, history: AnalysisHistory) -> dict[str, Any]:
    """Recompute the retained collection product from the supplied orbit history."""

    retained = json.loads(read_regular_file_nofollow(Path(path), min_bytes=1, max_bytes=64 * 1024 * 1024))
    if not isinstance(retained, dict) or not isinstance(retained.get("normalized_problem"), dict):
        raise OrbitBoundSourceError("Orbit-bound collection evidence is invalid.")
    if retained.get("orbit_history_semantic_sha256") != orbit_history_semantic_sha256(history):
        raise OrbitBoundSourceError("Collection source cites a different orbit history.")
    expected = assess_collection_opportunities(retained["normalized_problem"], orbit_history=history)
    normalized = json.loads(json.dumps(expected, sort_keys=True, allow_nan=False))
    if not _same_evidence(retained, normalized):
        raise OrbitBoundSourceError("Collection source differs from authoritative history-backed replay.")
    return normalized


def _indices(history: AnalysisHistory, sample_indices: Sequence[int]) -> np.ndarray:
    raw = list(sample_indices)
    if (len(raw) < 2 or len(raw) > history.times_s.size or
            any(isinstance(value, bool) or not isinstance(value, int) for value in raw) or
            any(value < 0 or value >= history.times_s.size for value in raw) or
            any(right <= left for left, right in zip(raw, raw[1:]))):
        raise OrbitBoundSourceError("Link sample indices must be an increasing subset of the orbit history.")
    return np.asarray(raw, dtype=int)


def directed_link_config_from_mapping(value: Mapping[str, Any]) -> DirectedLinkConfig:
    """Parse the retained public directed-link configuration into typed terminals."""
    if not isinstance(value, Mapping):
        raise OrbitBoundSourceError("Directed-link configuration must be a JSON object.")
    raw = dict(value)
    raw.pop("contract_version", None)
    for name in ("tx_terminal", "rx_terminal"):
        terminal_value = raw.get(name)
        if not isinstance(terminal_value, Mapping):
            raise OrbitBoundSourceError(f"Directed-link {name} must be a JSON object.")
        terminal = dict(terminal_value)
        pattern = terminal.get("pattern")
        if not isinstance(pattern, Mapping):
            raise OrbitBoundSourceError(f"Directed-link {name}.pattern must be a JSON object.")
        try:
            terminal["pattern"] = TerminalPattern(**pattern)
            raw[name] = LinkTerminal(**terminal)
        except TypeError as exc:
            raise OrbitBoundSourceError(f"Invalid directed-link {name}: {exc}") from exc
    try:
        return DirectedLinkConfig(**raw)
    except TypeError as exc:
        raise OrbitBoundSourceError(f"Invalid directed-link configuration: {exc}") from exc


def write_orbit_bound_link(
    config: DirectedLinkConfig,
    history: AnalysisHistory,
    *,
    station_latitude_deg: float,
    station_longitude_deg: float,
    station_height_km: float,
    sample_indices: Sequence[int],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a directed-link product whose spacecraft samples come from one parent history."""

    if config.tx_terminal.asset_id != history.object_id or config.rx_terminal.parent_frame != "enu":
        raise OrbitBoundSourceError("Bound link must transmit from the orbit asset to a fixed-site station.")
    indices = _indices(history, sample_indices)
    times = history.times_s[indices]
    frame = FrameContext(
        model="simple_gmst", jd_utc_start=history.initial_jd_utc,
        source="orbit_bound_directed_link",
    )
    spacecraft = spacecraft_endpoint_history(
        asset_id=history.object_id,
        state_provider_id=history.state_provider_id,
        times_s=times,
        positions_eci_km=history.position_eci_km[indices],
        velocities_eci_km_s=history.velocity_eci_km_s[indices],
        attitudes_quat_bn=(None if history.attitude_quat_bn is None else history.attitude_quat_bn[indices]),
        attitude_source_kind=history.attitude_source_kind,
        attitude_provider_id=history.attitude_provider_id,
    )
    station = fixed_wgs84_site_history(
        asset_id=config.rx_terminal.asset_id,
        state_provider_id=f"{config.rx_terminal.asset_id}.fixed_wgs84",
        times_s=times,
        geodetic_latitude_deg=station_latitude_deg,
        longitude_deg=station_longitude_deg,
        ellipsoidal_height_km=station_height_km,
        frame_context=frame,
    )
    result = evaluate_directed_link(
        config, tx_history=spacecraft, rx_history=station, frame_context=frame
    )
    binding = {
        "schema_version": ORBIT_LINK_BINDING_SCHEMA,
        "parent_history_sha256": orbit_history_semantic_sha256(history),
        "asset_id": history.object_id,
        "derivation": "exact_sample_indices",
        "sample_indices": indices.tolist(),
        "station_latitude_deg": float(station_latitude_deg),
        "station_longitude_deg": float(station_longitude_deg),
        "station_height_km": float(station_height_km),
    }
    write_directed_link_artifacts(result, output_dir, orbit_binding=binding)
    return binding


def verify_orbit_bound_link(directory: str | Path, history: AnalysisHistory) -> dict[str, Any]:
    """Recreate every link artifact from the cited parent orbit and sample-index derivation."""

    root = Path(directory).expanduser().resolve()
    manifest = json.loads(read_regular_file_nofollow(
        root / "link_analysis_manifest.json", min_bytes=1, max_bytes=16 * 1024 * 1024
    ))
    binding = manifest.get("orbit_binding") if isinstance(manifest, dict) else None
    if not isinstance(binding, dict) or set(binding) != {
        "schema_version", "parent_history_sha256", "asset_id", "derivation", "sample_indices",
        "station_latitude_deg", "station_longitude_deg", "station_height_km",
    } or binding.get("schema_version") != ORBIT_LINK_BINDING_SCHEMA:
        raise OrbitBoundSourceError("Link product lacks a supported orbit binding.")
    if (binding["parent_history_sha256"] != orbit_history_semantic_sha256(history)
            or binding["asset_id"] != history.object_id
            or binding["derivation"] != "exact_sample_indices"):
        raise OrbitBoundSourceError("Link source cites a different orbit history.")
    config = directed_link_config_from_mapping(manifest["normalized_config"])
    with tempfile.TemporaryDirectory(prefix="oel-orbit-link-replay-") as temporary:
        expected_root = Path(temporary) / "link"
        expected_binding = write_orbit_bound_link(
            config, history,
            station_latitude_deg=binding["station_latitude_deg"],
            station_longitude_deg=binding["station_longitude_deg"],
            station_height_km=binding["station_height_km"],
            sample_indices=binding["sample_indices"],
            output_dir=expected_root,
        )
        if expected_binding != binding:
            raise OrbitBoundSourceError("Link orbit derivation differs from retained binding.")
        expected_files = {item.name for item in expected_root.iterdir()}
        if {item.name for item in root.iterdir()} != expected_files:
            raise OrbitBoundSourceError("Orbit-bound link artifact inventory differs from replay.")
        for name in expected_files:
            if read_regular_file_nofollow(root / name, min_bytes=1, max_bytes=64 * 1024 * 1024) != (
                expected_root / name
            ).read_bytes():
                raise OrbitBoundSourceError(f"Link artifact {name} differs from history-backed replay.")
    return binding


__all__ = [
    "ORBIT_LINK_BINDING_SCHEMA", "OrbitBoundSourceError", "verify_orbit_bound_collection",
    "verify_orbit_bound_link", "write_orbit_bound_link", "directed_link_config_from_mapping",
]
