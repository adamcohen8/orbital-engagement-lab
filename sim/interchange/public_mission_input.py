# ruff: noqa: E501
"""Public-safe mission-input adapter for bounded CCSDS ephemeris products."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKET_VERSION = 1


@dataclass(frozen=True)
class MissionInputPacket:
    """Normalized public mission-input packet returned by CCSDS importers."""

    data: Mapping[str, Any]

    @property
    def objects(self) -> dict[str, Any]:
        value = self.data.get("objects", {})
        return dict(value if isinstance(value, Mapping) else {})

    @property
    def warnings(self) -> list[str]:
        return [str(item) for item in list(self.data.get("warnings", []) or [])]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.data))

    def write_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def ingest_ephemeris_samples(
    *,
    object_id: str,
    samples: Sequence[Mapping[str, Any]],
    role: str | None = None,
    source_label: str = "agent",
    source_type: str = "structured_ephemeris",
    frame: str = "eci",
    position_units: str = "km",
    velocity_units: str = "km/s",
    time_units: str = "s",
    source_metadata: Mapping[str, Any] | None = None,
) -> MissionInputPacket:
    """Build the bounded packet shape used by public OEM/OPM conversion."""

    oid = _clean_id(object_id, "object_id")
    frame_key = _inertial_frame_key(frame)
    normalized = _normalize_ephemeris_samples(
        samples,
        position_units=position_units,
        velocity_units=velocity_units,
        time_units=time_units,
        context=f"ephemeris samples for {oid}",
    )
    first = normalized[0]
    last = normalized[-1]
    initial_state: dict[str, Any] = {
        "position_eci_km": list(first["position_eci_km"]),
        "velocity_eci_km_s": list(first["velocity_eci_km_s"]),
    }
    if first.get("jd_utc") is not None:
        initial_state["epoch_jd_utc"] = float(first["jd_utc"])
    ephemeris: dict[str, Any] = {
        "sample_count": len(normalized),
        "first_time_s": first.get("time_s"),
        "last_time_s": last.get("time_s"),
        "first_jd_utc": first.get("jd_utc"),
        "last_jd_utc": last.get("jd_utc"),
    }
    if source_metadata:
        ephemeris["source_metadata"] = deepcopy(dict(source_metadata))
    obj = {
        "object_id": oid,
        "kind": "satellite",
        "role": str(role or oid),
        "state_type": "structured_ephemeris",
        "frame": "ECI" if frame_key == "eci" else "GCRF treated as ECI",
        "initial_state": initial_state,
        "normalized_units": {"position": "km", "velocity": "km/s", "time": "s"},
        "ephemeris": ephemeris,
        "provenance": {"source_label": str(source_label or "agent"), "source_type": source_type},
    }
    warnings = [
        "Structured ephemeris ingestion uses the first sample as the OEL initial state; it does not replay, interpolate, or fit the full ephemeris history."
    ]
    if frame_key == "gcrf":
        warnings.append("GCRF input is treated as ECI for OEL scenario initialization.")
    return MissionInputPacket(
        {
            "packet_version": PACKET_VERSION,
            "kind": "oel.mission_input_packet",
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source": {"type": str(source_type)},
            "objects": {oid: obj},
            "warnings": warnings,
            "validation": {
                "status": "ready_with_warnings",
                "notes": [
                    "Packet has normalized units and can be converted into OEL scenario YAML.",
                    "Run run_simulation.py --validate-only on generated scenario YAML before execution.",
                ],
            },
        }
    )


def _normalize_ephemeris_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    position_units: str,
    velocity_units: str,
    time_units: str,
    context: str,
) -> list[dict[str, Any]]:
    if not samples:
        raise ValueError(f"{context} must contain at least one sample.")
    position_scale = _unit_scale(position_units, kilometer=True)
    velocity_scale = _unit_scale(velocity_units, kilometer=False)
    time_scale = _time_scale(time_units)
    rows: list[dict[str, Any]] = []
    previous_time: float | None = None
    previous_jd: float | None = None
    for index, raw in enumerate(samples):
        row = dict(raw or {})
        position = row.get("position_eci_km", row.get("position_eci"))
        velocity = row.get("velocity_eci_km_s", row.get("velocity_eci"))
        if position is None or velocity is None:
            raise ValueError(f"{context} sample {index} must include position_eci_km and velocity_eci_km_s.")
        time_s = None if row.get("time_s") is None else _finite(row["time_s"], f"{context} time_s") * time_scale
        jd_utc = None if row.get("jd_utc") is None else _finite(row["jd_utc"], f"{context} jd_utc")
        if time_s is None and jd_utc is None:
            raise ValueError(f"{context} sample {index} must include time_s or jd_utc.")
        if time_s is not None and previous_time is not None and time_s <= previous_time:
            raise ValueError(f"{context} time_s values must be strictly increasing.")
        if jd_utc is not None and previous_jd is not None and jd_utc <= previous_jd:
            raise ValueError(f"{context} jd_utc values must be strictly increasing.")
        previous_time = time_s if time_s is not None else previous_time
        previous_jd = jd_utc if jd_utc is not None else previous_jd
        rows.append(
            {
                "time_s": time_s,
                "jd_utc": jd_utc,
                "position_eci_km": _scaled_vector(position, position_scale, f"{context} position"),
                "velocity_eci_km_s": _scaled_vector(velocity, velocity_scale, f"{context} velocity"),
            }
        )
    return rows


def _clean_id(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or not text.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"{field_name} must contain only letters, numbers, underscores, or hyphens.")
    return text


def _inertial_frame_key(frame: str) -> str:
    key = str(frame or "eci").strip().lower()
    if key == "eci":
        return "eci"
    if key in {"gcrf", "eme2000", "j2000", "icrf"}:
        return "gcrf"
    raise ValueError("ephemeris frame must be 'eci' or a supported inertial frame alias.")


def _unit_scale(units: str, *, kilometer: bool) -> float:
    key = str(units or ("km" if kilometer else "km/s")).strip().lower().replace(" ", "")
    kilometer_units = {"km", "kilometer", "kilometers"} if kilometer else {
        "km/s", "km_s", "kmps", "kilometer/second", "kilometers/second"
    }
    meter_units = {"m", "meter", "meters"} if kilometer else {
        "m/s", "m_s", "mps", "meter/second", "meters/second"
    }
    if key in kilometer_units:
        return 1.0
    if key in meter_units:
        return 1000.0
    raise ValueError("unsupported public ephemeris units.")


def _time_scale(units: str) -> float:
    key = str(units or "s").strip().lower()
    if key in {"s", "sec", "second", "seconds"}:
        return 1.0
    if key in {"min", "minute", "minutes"}:
        return 60.0
    if key in {"h", "hr", "hour", "hours"}:
        return 3600.0
    raise ValueError("time units must be seconds, minutes, or hours.")


def _finite(value: Any, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _scaled_vector(value: Iterable[Any], scale: float, field_name: str) -> list[float]:
    result = [float(item) / scale for item in value]
    if len(result) != 3 or any(not math.isfinite(item) for item in result):
        raise ValueError(f"{field_name} must contain exactly three finite values.")
    return result


__all__ = ["MissionInputPacket", "ingest_ephemeris_samples"]
