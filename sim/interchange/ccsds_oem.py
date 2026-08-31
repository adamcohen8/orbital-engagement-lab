from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sim.dynamics.orbit.eop import EopSeries, load_iers_eop
from sim.dynamics.orbit.epoch import julian_date_to_datetime
from sim.frame_time import (
    FrameTimeError,
    FrameTransformContext,
    TimeScale,
    epoch_julian_date,
    format_epoch,
    leap_second_table_receipt,
    normalize_canonical_frame,
    parse_epoch,
    transform_cartesian_state,
    transform_covariance,
)
from sim.interchange.public_mission_input import MissionInputPacket, ingest_ephemeris_samples
from sim.interchange.ccsds_odm import (
    CcsdsOdmError,
    compare_odm,
    inspect_odm,
    opm_to_mission_input_packet,
    read_odm_kvn,
    write_odm_kvn,
)
from sim.interchange.provenance import sha256_file
from sim.review import ReviewWorkspace

CCSDS_OEM_VERSION = "3.0"
OEL_OEM_PROFILE = "oel.ccsds-oem-kvn.v0.2"
OEM_MEDIA_TYPE = "application/ccsds-oem"

MAX_OEM_BYTES = 64 * 1024 * 1024
MAX_OEM_LINES = 1_000_000
MAX_OEM_SEGMENTS = 1024
MAX_OEM_STATES = 500_000
MAX_OEM_COVARIANCES = 500_000

_HEADER_KEYS = (
    "CCSDS_OEM_VERS",
    "CLASSIFICATION",
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_ID",
)
_METADATA_KEYS = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "REF_FRAME_EPOCH",
    "TIME_SYSTEM",
    "START_TIME",
    "USEABLE_START_TIME",
    "USEABLE_STOP_TIME",
    "STOP_TIME",
    "INTERPOLATION",
    "INTERPOLATION_DEGREE",
)
_REQUIRED_HEADER_KEYS = frozenset({"CCSDS_OEM_VERS", "CREATION_DATE", "ORIGINATOR"})
_REQUIRED_METADATA_KEYS = frozenset(
    {
        "OBJECT_NAME",
        "OBJECT_ID",
        "CENTER_NAME",
        "REF_FRAME",
        "TIME_SYSTEM",
        "START_TIME",
        "STOP_TIME",
    }
)
_OEL_PROFILE_CENTER = "EARTH"
_OEL_PROFILE_FRAME = "EME2000"
_OEL_PROFILE_TIME_SYSTEM = "UTC"
_STATE_COMPONENT_ORDER = ["x", "y", "z", "vx", "vy", "vz"]
_STATE_UNITS = ["km", "km", "km", "km/s", "km/s", "km/s"]
_FLOAT_TOKEN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_CALENDAR_EPOCH = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?P<fraction>\.\d+)?(?P<z>Z)?$"
)
_ORDINAL_EPOCH = re.compile(
    r"^(?P<year>\d{4})-(?P<doy>\d{3})T"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?P<fraction>\.\d+)?(?P<z>Z)?$"
)


class CcsdsOemError(ValueError):
    """Raised when an OEM cannot be represented by the bounded public profile."""


@dataclass(frozen=True)
class OemHeader:
    version: str
    creation_date: str
    originator: str
    message_id: str | None = None
    classification: str | None = None
    comments: tuple[str, ...] = ()


@dataclass(frozen=True)
class OemMetadata:
    object_name: str
    object_id: str
    center_name: str
    ref_frame: str
    time_system: str
    start_time: str
    stop_time: str
    usable_start_time: str | None = None
    usable_stop_time: str | None = None
    ref_frame_epoch: str | None = None
    interpolation: str | None = None
    interpolation_degree: int | None = None
    comments: tuple[str, ...] = ()


@dataclass(frozen=True)
class OemState:
    epoch: str
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]
    acceleration_km_s2: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class OemCovariance:
    epoch: str
    matrix: tuple[tuple[float, ...], ...]
    ref_frame: str | None = None


@dataclass(frozen=True)
class OemSegment:
    metadata: OemMetadata
    states: tuple[OemState, ...]
    comments: tuple[str, ...] = ()
    covariances: tuple[OemCovariance, ...] = ()


@dataclass(frozen=True)
class OemMessage:
    header: OemHeader
    segments: tuple[OemSegment, ...]
    source_sha256: str | None = None


def parse_oem_kvn(text: str, *, source_sha256: str | None = None) -> OemMessage:
    """Parse CCSDS OEM 3.0 KVN without applying frame or time transformations."""

    if not isinstance(text, str):
        raise CcsdsOemError("OEM input must be Unicode text.")
    if "\x00" in text:
        raise CcsdsOemError("OEM input contains a NUL byte.")
    lines = text.splitlines()
    if len(lines) > MAX_OEM_LINES:
        raise CcsdsOemError(f"OEM input exceeds the {MAX_OEM_LINES} line limit.")

    header_values: dict[str, str] = {}
    header_comments: list[str] = []
    segments: list[OemSegment] = []
    metadata_values: dict[str, str] | None = None
    metadata_comments: list[str] = []
    segment_comments: list[str] = []
    states: list[OemState] = []
    covariances: list[OemCovariance] = []
    covariance_values: dict[str, str] | None = None
    covariance_rows: list[tuple[float, ...]] = []
    phase = "header"

    def finish_covariance() -> None:
        nonlocal covariance_values, covariance_rows
        if covariance_values is None:
            raise CcsdsOemError("Covariance matrix values must begin with EPOCH.")
        if "EPOCH" not in covariance_values:
            raise CcsdsOemError("Each OEM covariance matrix requires EPOCH.")
        if len(covariance_rows) != 6:
            raise CcsdsOemError("Each OEM covariance matrix must contain six lower-triangular rows.")
        matrix = np.zeros((6, 6), dtype=float)
        for row_index, row in enumerate(covariance_rows):
            matrix[row_index, : row_index + 1] = row
            matrix[: row_index + 1, row_index] = row
        covariance = OemCovariance(
            epoch=covariance_values["EPOCH"],
            ref_frame=covariance_values.get("COV_REF_FRAME"),
            matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        )
        _validate_covariance(covariance)
        covariances.append(covariance)
        covariance_values = None
        covariance_rows = []

    def finish_segment() -> None:
        nonlocal metadata_values, metadata_comments, segment_comments, states, covariances
        if metadata_values is None:
            return
        if len(segments) >= MAX_OEM_SEGMENTS:
            raise CcsdsOemError(f"OEM input exceeds the {MAX_OEM_SEGMENTS} segment limit.")
        metadata = _metadata_from_values(metadata_values, metadata_comments)
        segment = OemSegment(
            metadata=metadata,
            states=tuple(states),
            comments=tuple(segment_comments),
            covariances=tuple(covariances),
        )
        _validate_segment(segment)
        segments.append(segment)
        metadata_values = None
        metadata_comments = []
        segment_comments = []
        states = []
        covariances = []

    first_content_seen = False
    total_states = 0
    total_covariances = 0
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if not first_content_seen:
            first_content_seen = True
            key, value = _parse_assignment(line, line_number=line_number)
            if key != "CCSDS_OEM_VERS":
                raise CcsdsOemError("The first non-blank OEM line must be CCSDS_OEM_VERS.")
            header_values[key] = value
            continue
        if line == "META_START":
            if phase in {"metadata", "covariance"}:
                raise CcsdsOemError(f"line {line_number}: nested META_START is invalid.")
            if phase in {"data", "post_covariance"}:
                finish_segment()
            metadata_values = {}
            metadata_comments = []
            segment_comments = []
            states = []
            covariances = []
            phase = "metadata"
            continue
        if line == "META_STOP":
            if phase != "metadata" or metadata_values is None:
                raise CcsdsOemError(f"line {line_number}: META_STOP has no matching META_START.")
            phase = "data"
            continue
        if line == "COVARIANCE_START":
            if phase != "data" or not states:
                raise CcsdsOemError(f"line {line_number}: COVARIANCE_START must follow ephemeris data.")
            covariance_values = None
            covariance_rows = []
            phase = "covariance"
            continue
        if line == "COVARIANCE_STOP":
            if phase != "covariance":
                raise CcsdsOemError(f"line {line_number}: COVARIANCE_STOP has no matching COVARIANCE_START.")
            finish_covariance()
            total_covariances += 1
            if total_covariances > MAX_OEM_COVARIANCES:
                raise CcsdsOemError(f"OEM input exceeds the {MAX_OEM_COVARIANCES} covariance limit.")
            phase = "post_covariance"
            continue
        if line in {"DATA_START", "DATA_STOP"}:
            raise CcsdsOemError(f"line {line_number}: {line} is not an OEM KVN delimiter.")
        if line.startswith("COMMENT"):
            comment = _parse_comment(line, line_number=line_number)
            if phase == "header":
                if any(key != "CCSDS_OEM_VERS" for key in header_values):
                    raise CcsdsOemError(
                        f"line {line_number}: header COMMENT is allowed only immediately after CCSDS_OEM_VERS."
                    )
                header_comments.append(comment)
            elif phase == "metadata":
                if metadata_values:
                    raise CcsdsOemError(
                        f"line {line_number}: metadata COMMENT is allowed only immediately after META_START."
                    )
                metadata_comments.append(comment)
            elif phase == "data":
                if states:
                    raise CcsdsOemError(
                        f"line {line_number}: {OEL_OEM_PROFILE} accepts data comments only before state rows."
                    )
                segment_comments.append(comment)
            elif phase in {"covariance", "post_covariance"}:
                raise CcsdsOemError(
                    f"line {line_number}: covariance comments are not preserved by {OEL_OEM_PROFILE}."
                )
            else:
                raise CcsdsOemError(f"line {line_number}: COMMENT appears outside an OEM section.")
            continue
        if phase == "header":
            key, value = _parse_assignment(line, line_number=line_number)
            _store_unique(header_values, key, value, allowed=_HEADER_KEYS, line_number=line_number)
        elif phase == "metadata":
            key, value = _parse_assignment(line, line_number=line_number)
            _store_unique(metadata_values, key, value, allowed=_METADATA_KEYS, line_number=line_number)
        elif phase == "data":
            if "=" in line:
                raise CcsdsOemError(f"line {line_number}: unexpected keyword inside OEM ephemeris data.")
            states.append(_parse_state_line(line, line_number=line_number))
            total_states += 1
            if total_states > MAX_OEM_STATES:
                raise CcsdsOemError(f"OEM input exceeds the {MAX_OEM_STATES} state limit.")
        elif phase == "covariance":
            if "=" in line:
                key, value = _parse_assignment(line, line_number=line_number)
                if key == "EPOCH":
                    if covariance_values is not None:
                        finish_covariance()
                        total_covariances += 1
                        if total_covariances > MAX_OEM_COVARIANCES:
                            raise CcsdsOemError(
                                f"OEM input exceeds the {MAX_OEM_COVARIANCES} covariance limit."
                            )
                    covariance_values = {"EPOCH": value}
                    _parse_epoch(value, label=f"line {line_number} covariance EPOCH")
                elif key == "COV_REF_FRAME":
                    if covariance_values is None or "EPOCH" not in covariance_values:
                        raise CcsdsOemError(
                            f"line {line_number}: COV_REF_FRAME must follow the covariance EPOCH."
                        )
                    if covariance_rows:
                        raise CcsdsOemError(
                            f"line {line_number}: COV_REF_FRAME must precede covariance matrix rows."
                        )
                    if key in covariance_values:
                        raise CcsdsOemError(f"line {line_number}: duplicate OEM keyword {key!r}.")
                    covariance_values[key] = value
                else:
                    raise CcsdsOemError(f"line {line_number}: unsupported covariance keyword {key!r}.")
            else:
                if covariance_values is None:
                    raise CcsdsOemError(f"line {line_number}: covariance values must follow EPOCH.")
                expected = len(covariance_rows) + 1
                if expected > 6:
                    raise CcsdsOemError(f"line {line_number}: covariance matrix has more than six rows.")
                covariance_rows.append(_parse_covariance_row(line, expected=expected, line_number=line_number))
        elif phase == "post_covariance":
            raise CcsdsOemError(f"line {line_number}: content after COVARIANCE_STOP requires a new META_START.")
        else:
            raise CcsdsOemError(f"line {line_number}: OEM content appears outside a section.")

    if phase == "metadata":
        raise CcsdsOemError("OEM ended before META_STOP.")
    if phase == "covariance":
        raise CcsdsOemError("OEM ended before COVARIANCE_STOP.")
    if phase in {"data", "post_covariance"}:
        finish_segment()
    if not first_content_seen:
        raise CcsdsOemError("OEM input is empty.")
    header = _header_from_values(header_values, header_comments)
    message = OemMessage(header=header, segments=tuple(segments), source_sha256=source_sha256)
    validate_oem(message)
    return message


def read_oem_kvn(path: str | Path, *, max_bytes: int = MAX_OEM_BYTES) -> OemMessage:
    source = Path(path).expanduser().resolve()
    size = source.stat().st_size
    if size > int(max_bytes):
        raise CcsdsOemError(f"OEM input exceeds the {int(max_bytes)} byte limit.")
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CcsdsOemError("OEM KVN input must be valid UTF-8.") from exc
    return parse_oem_kvn(text, source_sha256=hashlib.sha256(raw).hexdigest())


def validate_oem(message: OemMessage) -> None:
    if message.header.version != CCSDS_OEM_VERSION:
        raise CcsdsOemError(
            f"{OEL_OEM_PROFILE} supports CCSDS_OEM_VERS {CCSDS_OEM_VERSION}; received {message.header.version!r}."
        )
    _parse_epoch(message.header.creation_date, label="CREATION_DATE")
    if not message.header.originator.strip():
        raise CcsdsOemError("ORIGINATOR must not be empty.")
    if not message.segments:
        raise CcsdsOemError("OEM must contain at least one metadata and ephemeris segment.")
    time_systems = {segment.metadata.time_system.upper() for segment in message.segments}
    if len(time_systems) != 1:
        raise CcsdsOemError("TIME_SYSTEM must remain fixed within an OEM.")
    for segment in message.segments:
        _validate_segment(segment)


def oel_profile_issues(message: OemMessage) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for index, segment in enumerate(message.segments):
        metadata = segment.metadata
        if metadata.center_name.upper() != _OEL_PROFILE_CENTER:
            issues.append(
                _issue(index, "center_name", f"OEL import requires CENTER_NAME {_OEL_PROFILE_CENTER}.")
            )
        if metadata.ref_frame.upper() != _OEL_PROFILE_FRAME:
            issues.append(
                _issue(index, "ref_frame", f"OEL import requires REF_FRAME {_OEL_PROFILE_FRAME}.")
            )
        if metadata.time_system.upper() != _OEL_PROFILE_TIME_SYSTEM:
            issues.append(
                _issue(index, "time_system", f"OEL import requires TIME_SYSTEM {_OEL_PROFILE_TIME_SYSTEM}.")
            )
        if metadata.ref_frame_epoch is not None:
            issues.append(
                _issue(index, "ref_frame_epoch", "OEL EME2000 import does not accept REF_FRAME_EPOCH.")
            )
        for covariance_index, covariance in enumerate(segment.covariances):
            covariance_frame = str(covariance.ref_frame or metadata.ref_frame).upper()
            if covariance_frame != _OEL_PROFILE_FRAME:
                issues.append(
                    {
                        "path": f"segments[{index}].covariances[{covariance_index}].ref_frame",
                        "code": "unsupported_covariance_ref_frame",
                        "message": "OEL mission-input conversion requires covariance in EME2000; use the frame-time API first.",
                    }
                )
    if len(message.segments) != 1:
        issues.append(
            {
                "path": "segments",
                "code": "multiple_segments_not_importable",
                "message": "The first OEL import profile accepts exactly one continuous OEM segment.",
            }
        )
    return issues


def inspect_oem(message_or_path: OemMessage | str | Path) -> dict[str, Any]:
    message = read_oem_kvn(message_or_path) if not isinstance(message_or_path, OemMessage) else message_or_path
    profile_issues = oel_profile_issues(message)
    state_count = sum(len(segment.states) for segment in message.segments)
    covariance_count = sum(len(segment.covariances) for segment in message.segments)
    return {
        "document_type": "ccsds_oem_kvn",
        "ccsds_oem_version": message.header.version,
        "oel_profile": OEL_OEM_PROFILE,
        "valid_oem": True,
        "oel_import_ready": not profile_issues,
        "source_sha256": message.source_sha256,
        "originator": message.header.originator,
        "message_id": message.header.message_id,
        "segment_count": len(message.segments),
        "state_count": state_count,
        "covariance_count": covariance_count,
        "segments": [
            {
                "segment_index": index,
                "object_name": segment.metadata.object_name,
                "object_id": segment.metadata.object_id,
                "center_name": segment.metadata.center_name,
                "ref_frame": segment.metadata.ref_frame,
                "time_system": segment.metadata.time_system,
                "start_time": segment.metadata.start_time,
                "stop_time": segment.metadata.stop_time,
                "state_count": len(segment.states),
                "covariance_count": len(segment.covariances),
                "covariance_ref_frames": sorted(
                    {str(covariance.ref_frame or segment.metadata.ref_frame) for covariance in segment.covariances}
                ),
                "acceleration_present": any(state.acceleration_km_s2 is not None for state in segment.states),
            }
            for index, segment in enumerate(message.segments)
        ],
        "profile_issues": profile_issues,
        "non_claims": [
            "Parsing validates the bounded OEM 3.0 KVN contract; it does not validate orbit accuracy.",
            "The v0.2 profile supports Cartesian 6x6 covariance blocks but does not establish covariance calibration.",
            "XML and direct OEM state-frame/time-scale conversion remain outside this profile.",
            "Originator and object identifiers are preserved but not resolved against online registries.",
        ],
    }


def serialize_oem_kvn(message: OemMessage) -> str:
    validate_oem(message)
    lines = [f"CCSDS_OEM_VERS = {message.header.version}"]
    lines.extend(f"COMMENT  {_clean_comment(comment)}" for comment in message.header.comments)
    if message.header.classification is not None:
        lines.append(f"CLASSIFICATION = {message.header.classification}")
    lines.append(f"CREATION_DATE = {message.header.creation_date}")
    lines.append(f"ORIGINATOR = {message.header.originator}")
    if message.header.message_id is not None:
        lines.append(f"MESSAGE_ID = {message.header.message_id}")
    for segment in message.segments:
        metadata = segment.metadata
        lines.extend(("", "META_START"))
        lines.extend(f"COMMENT  {_clean_comment(comment)}" for comment in metadata.comments)
        lines.append(f"OBJECT_NAME = {metadata.object_name}")
        lines.append(f"OBJECT_ID = {metadata.object_id}")
        lines.append(f"CENTER_NAME = {metadata.center_name}")
        lines.append(f"REF_FRAME = {metadata.ref_frame}")
        if metadata.ref_frame_epoch is not None:
            lines.append(f"REF_FRAME_EPOCH = {metadata.ref_frame_epoch}")
        lines.append(f"TIME_SYSTEM = {metadata.time_system}")
        lines.append(f"START_TIME = {metadata.start_time}")
        if metadata.usable_start_time is not None:
            lines.append(f"USEABLE_START_TIME = {metadata.usable_start_time}")
        if metadata.usable_stop_time is not None:
            lines.append(f"USEABLE_STOP_TIME = {metadata.usable_stop_time}")
        lines.append(f"STOP_TIME = {metadata.stop_time}")
        if metadata.interpolation is not None:
            lines.append(f"INTERPOLATION = {metadata.interpolation}")
            lines.append(f"INTERPOLATION_DEGREE = {metadata.interpolation_degree}")
        lines.append("META_STOP")
        if segment.comments:
            lines.append("")
            lines.extend(f"COMMENT  {_clean_comment(comment)}" for comment in segment.comments)
        lines.append("")
        lines.extend(_serialize_state(state) for state in segment.states)
        if segment.covariances:
            lines.extend(("", "COVARIANCE_START"))
            for covariance_index, covariance in enumerate(segment.covariances):
                if covariance_index:
                    lines.append("")
                lines.append(f"EPOCH = {covariance.epoch}")
                if covariance.ref_frame is not None:
                    lines.append(f"COV_REF_FRAME = {covariance.ref_frame}")
                for row_index, row in enumerate(covariance.matrix):
                    lines.append(" ".join(_format_float(value) for value in row[: row_index + 1]))
            lines.append("COVARIANCE_STOP")
    return "\n".join(lines) + "\n"


def write_oem_kvn(message: OemMessage, path: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(path).expanduser().resolve()
    text = serialize_oem_kvn(message)
    if target.exists() and target.read_text(encoding="utf-8") != text and not overwrite:
        raise CcsdsOemError("OEM output exists with different content; pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return target


def convert_oem(
    message_or_path: OemMessage | str | Path,
    *,
    target_frame: str,
    target_time_system: str,
    eop_series: EopSeries | None = None,
) -> OemMessage:
    """Explicitly convert Earth-centered OEM states/covariances without executing a scenario."""

    message = read_oem_kvn(message_or_path) if not isinstance(message_or_path, OemMessage) else message_or_path
    target = normalize_canonical_frame(target_frame)
    target_scale = TimeScale(str(target_time_system).upper())
    converted_segments: list[OemSegment] = []
    for segment_index, segment in enumerate(message.segments):
        metadata = segment.metadata
        if metadata.center_name.upper() != "EARTH":
            raise CcsdsOemError(f"segment {segment_index}: explicit OEM conversion supports CENTER_NAME EARTH only.")
        source = normalize_canonical_frame(metadata.ref_frame)
        source_scale = _frame_time_scale(metadata.time_system)
        frame_changes = source is not target
        if frame_changes and "ITRF" in {source.value, target.value} and eop_series is None:
            raise CcsdsOemError(
                f"segment {segment_index}: ITRF conversion requires sampled DUT1 and polar-motion values with provenance."
            )
        if frame_changes and any(state.acceleration_km_s2 is not None for state in segment.states):
            raise CcsdsOemError(
                f"segment {segment_index}: acceleration conversion requires second-order frame kinematics and is outside v0.2."
            )
        states: list[OemState] = []
        for state in segment.states:
            epoch, earth_orientation = _conversion_epoch(state.epoch, source_scale, eop_series)
            context = FrameTransformContext(epoch=epoch, earth_orientation=earth_orientation)
            position, velocity = transform_cartesian_state(
                state.position_km,
                state.velocity_km_s,
                source,
                target,
                context=context,
            )
            states.append(
                OemState(
                    epoch=_converted_epoch_text(epoch, target_scale, eop_series),
                    position_km=tuple(float(value) for value in position),
                    velocity_km_s=tuple(float(value) for value in velocity),
                    acceleration_km_s2=state.acceleration_km_s2,
                )
            )
        covariances: list[OemCovariance] = []
        for covariance in segment.covariances:
            covariance_source = normalize_canonical_frame(covariance.ref_frame or metadata.ref_frame)
            if covariance_source is not target and "ITRF" in {covariance_source.value, target.value} and eop_series is None:
                raise CcsdsOemError(
                    f"segment {segment_index}: ITRF covariance conversion requires epoch-covering EOP provenance."
                )
            epoch, earth_orientation = _conversion_epoch(covariance.epoch, source_scale, eop_series)
            context = FrameTransformContext(epoch=epoch, earth_orientation=earth_orientation)
            matrix = transform_covariance(
                covariance.matrix,
                covariance_source,
                target,
                context=context,
            )
            covariances.append(
                OemCovariance(
                    epoch=_converted_epoch_text(epoch, target_scale, eop_series),
                    matrix=tuple(tuple(float(value) for value in row) for row in matrix),
                    ref_frame=target.value,
                )
            )

        comments = list(segment.comments)
        if frame_changes or source_scale is not target_scale:
            comments.append(
                f"Explicitly converted by OEL from {source.value}/{source_scale.value} "
                f"to {target.value}/{target_scale.value}; source interpolation declaration removed."
            )
        converted_segments.append(
            OemSegment(
                metadata=OemMetadata(
                    object_name=metadata.object_name,
                    object_id=metadata.object_id,
                    center_name=metadata.center_name,
                    ref_frame=target.value,
                    time_system=target_scale.value,
                    start_time=str(
                        _convert_metadata_epoch(metadata.start_time, source_scale, target_scale, eop_series)
                    ),
                    stop_time=str(
                        _convert_metadata_epoch(metadata.stop_time, source_scale, target_scale, eop_series)
                    ),
                    usable_start_time=_convert_metadata_epoch(
                        metadata.usable_start_time,
                        source_scale,
                        target_scale,
                        eop_series,
                    ),
                    usable_stop_time=_convert_metadata_epoch(
                        metadata.usable_stop_time,
                        source_scale,
                        target_scale,
                        eop_series,
                    ),
                    ref_frame_epoch=None,
                    interpolation=None,
                    interpolation_degree=None,
                    comments=metadata.comments,
                ),
                states=tuple(states),
                comments=tuple(comments),
                covariances=tuple(covariances),
            )
        )
    converted = OemMessage(header=message.header, segments=tuple(converted_segments))
    validate_oem(converted)
    return converted


def convert_oem_kvn(
    source_path: str | Path,
    *,
    output_path: str | Path,
    target_frame: str = "EME2000",
    target_time_system: str = "UTC",
    eop_path: str | Path | None = None,
    eop_format: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    message = read_oem_kvn(source)
    series = None if eop_path is None else load_iers_eop(eop_path, source_format=eop_format)
    converted = convert_oem(
        message,
        target_frame=target_frame,
        target_time_system=target_time_system,
        eop_series=series,
    )
    text = serialize_oem_kvn(converted)
    output_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    receipt = {
        "schema": "oel.ccsds-oem-conversion-receipt.v1",
        "status": "converted",
        "source_path": str(source),
        "source_sha256": message.source_sha256,
        "output_path": str(target),
        "output_sha256": output_sha256,
        "target_frame": normalize_canonical_frame(target_frame).value,
        "target_time_system": TimeScale(str(target_time_system).upper()).value,
        "segment_count": len(converted.segments),
        "state_count": sum(len(segment.states) for segment in converted.segments),
        "covariance_count": sum(len(segment.covariances) for segment in converted.segments),
        "eop": None if series is None else series.receipt(),
        "semantic_readback": compare_oem(converted, parse_oem_kvn(text)),
        "execution_occurred": False,
        "non_claims": [
            "Conversion applies the named frame/time contract; it does not establish source orbit accuracy.",
            "Interpolation declarations are removed after conversion and must be re-established for downstream interpolation.",
            "Covariance transformation is mathematical and does not establish covariance calibration.",
        ],
    }
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    receipt_text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text and not overwrite:
        raise CcsdsOemError("OEM output exists with different content; pass overwrite=True to replace it.")
    if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != receipt_text and not overwrite:
        raise CcsdsOemError("OEM conversion receipt exists with different content; pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    receipt_path.write_text(receipt_text, encoding="utf-8")
    return {**receipt, "receipt_path": str(receipt_path)}


def _frame_time_scale(value: str) -> TimeScale:
    try:
        return TimeScale(str(value).upper())
    except ValueError as exc:
        raise CcsdsOemError("Explicit OEM conversion supports UTC, TAI, TT, and UT1 time systems.") from exc


def _conversion_epoch(
    text: str,
    scale: TimeScale,
    eop_series: EopSeries | None,
):
    if scale is TimeScale.UT1:
        if eop_series is None:
            raise CcsdsOemError("UT1 OEM conversion requires an epoch-covering EOP source.")
        provisional = parse_epoch(text, scale, dut1_s=0.0)
        provisional_eop = eop_series.sample(provisional)
        epoch = parse_epoch(text, scale, dut1_s=provisional_eop.dut1_s)
    else:
        epoch = parse_epoch(text, scale)
    earth_orientation = None if eop_series is None else eop_series.sample(epoch)
    return epoch, earth_orientation


def _converted_epoch_text(epoch, target_scale: TimeScale, eop_series: EopSeries | None) -> str:
    if target_scale is not TimeScale.UT1:
        return format_epoch(epoch, target_scale)
    if eop_series is None:
        raise CcsdsOemError("UT1 OEM conversion requires an epoch-covering EOP source.")
    return format_epoch(epoch, target_scale, dut1_s=eop_series.sample(epoch).dut1_s)


def _convert_metadata_epoch(
    value: str | None,
    source_scale: TimeScale,
    target_scale: TimeScale,
    eop_series: EopSeries | None,
) -> str | None:
    if value is None:
        return None
    epoch, _earth_orientation = _conversion_epoch(value, source_scale, eop_series)
    return _converted_epoch_text(epoch, target_scale, eop_series)


def compare_oem(
    left: OemMessage | str | Path,
    right: OemMessage | str | Path,
    *,
    position_tolerance_km: float = 1.0e-9,
    velocity_tolerance_km_s: float = 1.0e-12,
    acceleration_tolerance_km_s2: float = 1.0e-15,
    covariance_absolute_tolerance: float = 1.0e-18,
    covariance_relative_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    lhs = read_oem_kvn(left) if not isinstance(left, OemMessage) else left
    rhs = read_oem_kvn(right) if not isinstance(right, OemMessage) else right
    position_tolerance = _nonnegative_finite(position_tolerance_km, "position_tolerance_km")
    velocity_tolerance = _nonnegative_finite(velocity_tolerance_km_s, "velocity_tolerance_km_s")
    acceleration_tolerance = _nonnegative_finite(
        acceleration_tolerance_km_s2,
        "acceleration_tolerance_km_s2",
    )
    covariance_absolute = _nonnegative_finite(
        covariance_absolute_tolerance,
        "covariance_absolute_tolerance",
    )
    covariance_relative = _nonnegative_finite(
        covariance_relative_tolerance,
        "covariance_relative_tolerance",
    )
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: Any) -> None:
        checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})

    check("header.version", lhs.header.version == rhs.header.version, [lhs.header.version, rhs.header.version])
    check("segment_count", len(lhs.segments) == len(rhs.segments), [len(lhs.segments), len(rhs.segments)])
    max_position = 0.0
    max_velocity = 0.0
    max_acceleration = 0.0
    max_covariance = 0.0
    if len(lhs.segments) == len(rhs.segments):
        for index, (left_segment, right_segment) in enumerate(zip(lhs.segments, rhs.segments)):
            left_meta = left_segment.metadata
            right_meta = right_segment.metadata
            left_semantics = _metadata_semantics(left_meta)
            right_semantics = _metadata_semantics(right_meta)
            check(f"segment[{index}].metadata", left_semantics == right_semantics, [left_semantics, right_semantics])
            check(
                f"segment[{index}].state_count",
                len(left_segment.states) == len(right_segment.states),
                [len(left_segment.states), len(right_segment.states)],
            )
            if len(left_segment.states) == len(right_segment.states):
                epoch_equal = True
                acceleration_presence_equal = True
                for left_state, right_state in zip(left_segment.states, right_segment.states):
                    epoch_equal = epoch_equal and _epoch_key(left_state.epoch) == _epoch_key(right_state.epoch)
                    max_position = max(
                        max_position,
                        max(abs(a - b) for a, b in zip(left_state.position_km, right_state.position_km)),
                    )
                    max_velocity = max(
                        max_velocity,
                        max(abs(a - b) for a, b in zip(left_state.velocity_km_s, right_state.velocity_km_s)),
                    )
                    acceleration_presence_equal = acceleration_presence_equal and (
                        (left_state.acceleration_km_s2 is None) == (right_state.acceleration_km_s2 is None)
                    )
                    if left_state.acceleration_km_s2 is not None and right_state.acceleration_km_s2 is not None:
                        max_acceleration = max(
                            max_acceleration,
                            max(
                                abs(a - b)
                                for a, b in zip(left_state.acceleration_km_s2, right_state.acceleration_km_s2)
                            ),
                        )
                check(f"segment[{index}].epochs", epoch_equal, "semantic UTC epoch equality")
                check(
                    f"segment[{index}].acceleration_presence",
                    acceleration_presence_equal,
                    "all acceleration components are either present on both sides or absent on both sides",
                )
            check(
                f"segment[{index}].covariance_count",
                len(left_segment.covariances) == len(right_segment.covariances),
                [len(left_segment.covariances), len(right_segment.covariances)],
            )
            if len(left_segment.covariances) == len(right_segment.covariances):
                covariance_epochs_equal = True
                covariance_frames_equal = True
                covariance_values_equal = True
                for left_covariance, right_covariance in zip(
                    left_segment.covariances,
                    right_segment.covariances,
                ):
                    covariance_epochs_equal = covariance_epochs_equal and (
                        _epoch_key(left_covariance.epoch) == _epoch_key(right_covariance.epoch)
                    )
                    left_frame = str(left_covariance.ref_frame or left_meta.ref_frame).upper()
                    right_frame = str(right_covariance.ref_frame or right_meta.ref_frame).upper()
                    covariance_frames_equal = covariance_frames_equal and left_frame == right_frame
                    left_matrix = np.asarray(left_covariance.matrix, dtype=float)
                    right_matrix = np.asarray(right_covariance.matrix, dtype=float)
                    residual = np.abs(left_matrix - right_matrix)
                    max_covariance = max(max_covariance, float(np.max(residual)))
                    covariance_values_equal = covariance_values_equal and bool(
                        np.all(
                            residual
                            <= covariance_absolute
                            + covariance_relative * np.maximum(np.abs(left_matrix), np.abs(right_matrix))
                        )
                    )
                check(
                    f"segment[{index}].covariance_epochs",
                    covariance_epochs_equal,
                    "semantic covariance epoch equality",
                )
                check(
                    f"segment[{index}].covariance_frames",
                    covariance_frames_equal,
                    "effective covariance frame equality",
                )
                check(
                    f"segment[{index}].covariance_values",
                    covariance_values_equal,
                    max_covariance,
                )
    check("position_residual", max_position <= position_tolerance, max_position)
    check("velocity_residual", max_velocity <= velocity_tolerance, max_velocity)
    check("acceleration_residual", max_acceleration <= acceleration_tolerance, max_acceleration)
    failed = [item["check_id"] for item in checks if not item["passed"]]
    return {
        "schema": "oel.ccsds-oem-semantic-comparison.v2",
        "status": "equivalent" if not failed else "different",
        "oel_profile": OEL_OEM_PROFILE,
        "tolerances": {
            "position_km": position_tolerance,
            "velocity_km_s": velocity_tolerance,
            "acceleration_km_s2": acceleration_tolerance,
            "covariance_absolute": covariance_absolute,
            "covariance_relative": covariance_relative,
        },
        "max_abs_position_residual_km": max_position,
        "max_abs_velocity_residual_km_s": max_velocity,
        "max_abs_acceleration_residual_km_s2": max_acceleration,
        "max_abs_covariance_residual": max_covariance,
        "checks": checks,
        "failed_check_ids": failed,
        "non_claims": [
            "Semantic parity does not establish orbit accuracy or source-product quality.",
            "Header creation time, comments, classification, message ID, and numeric formatting are not parity criteria.",
        ],
    }


def oem_to_mission_input_packet(message_or_path: OemMessage | str | Path) -> MissionInputPacket:
    message = read_oem_kvn(message_or_path) if not isinstance(message_or_path, OemMessage) else message_or_path
    issues = oel_profile_issues(message)
    if issues:
        detail = "; ".join(f"{item['path']}: {item['message']}" for item in issues)
        raise CcsdsOemError(f"OEM is not ready for the {OEL_OEM_PROFILE} import profile: {detail}")
    segment = message.segments[0]
    try:
        first_epoch = parse_epoch(segment.states[0].epoch, TimeScale.UTC)
    except FrameTimeError as exc:
        raise CcsdsOemError("First state epoch is outside the bounded UTC frame-time contract.") from exc
    samples = []
    for state in segment.states:
        try:
            epoch = parse_epoch(state.epoch, TimeScale.UTC)
        except FrameTimeError as exc:
            raise CcsdsOemError("State epoch is outside the bounded UTC frame-time contract.") from exc
        samples.append(
            {
                "time_s": epoch.tai_seconds - first_epoch.tai_seconds,
                "jd_utc": epoch_julian_date(epoch, TimeScale.UTC),
                "position_eci_km": list(state.position_km),
                "velocity_eci_km_s": list(state.velocity_km_s),
            }
        )
    return ingest_ephemeris_samples(
        object_id=segment.metadata.object_id,
        role=segment.metadata.object_name,
        samples=samples,
        source_label=f"CCSDS OEM {message.header.message_id or segment.metadata.object_id}",
        source_type="ccsds_oem_kvn_v3",
        frame="eci",
        position_units="km",
        velocity_units="km/s",
        time_units="s",
        source_metadata={
            "ccsds_oem_version": message.header.version,
            "oel_profile": OEL_OEM_PROFILE,
            "source_sha256": message.source_sha256,
            "center_name": segment.metadata.center_name,
            "source_ref_frame": segment.metadata.ref_frame,
            "oel_frame_mapping": "EME2000 -> OEL/ECI/J2000 (identity by OEL frames-v1 convention)",
            "time_system": segment.metadata.time_system,
            "start_time": segment.metadata.start_time,
            "stop_time": segment.metadata.stop_time,
            "first_epoch_utc": format_epoch(first_epoch, TimeScale.UTC),
            "first_epoch_tai": format_epoch(first_epoch, TimeScale.TAI),
            "first_epoch_jd_utc": epoch_julian_date(first_epoch, TimeScale.UTC),
            "elapsed_time_basis": "TAI SI seconds from the first state epoch",
            "leap_second_table": leap_second_table_receipt(),
            "covariances": [
                {
                    "epoch": covariance.epoch,
                    "frame": covariance.ref_frame or segment.metadata.ref_frame,
                    "component_order": ["X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"],
                    "units": ["km", "km", "km", "km/s", "km/s", "km/s"],
                    "matrix": [list(row) for row in covariance.matrix],
                    "calibrated": False,
                    "calibration_scope": None,
                }
                for covariance in segment.covariances
            ],
            "full_ephemeris_replayed": False,
        },
    )


def export_completed_run_oem(
    completed_run: str | Path,
    *,
    output_path: str | Path,
    object_id: str | None = None,
    object_name: str | None = None,
    originator: str = "OEL",
    message_id: str | None = None,
    overwrite: bool = False,
    max_states: int = MAX_OEM_STATES,
) -> dict[str, Any]:
    """Export one canonical OEL/ECI/J2000 review history as OEM EME2000/UTC."""

    target = Path(output_path).expanduser().resolve()
    with ReviewWorkspace.open(completed_run) as workspace:
        state_limit = int(max_states)
        if state_limit < 1 or state_limit > MAX_OEM_STATES:
            raise CcsdsOemError(f"max_states must be between 1 and {MAX_OEM_STATES}.")
        starting_review_identity = workspace.evidence_identity()
        required = {
            "run_metadata": {"run_id", "scenario_name", "generated_utc", "config_json", "config_sha256"},
            "object_state": {
                "sample_index",
                "time_s",
                "object_id",
                "pos_x_eci_km",
                "pos_y_eci_km",
                "pos_z_eci_km",
                "vel_x_eci_km_s",
                "vel_y_eci_km_s",
                "vel_z_eci_km_s",
            },
            "object_state_frame": {"object_id", "state_frame"},
        }
        columns = workspace.table_columns()
        for table, expected in required.items():
            actual = {str(item["name"]) for item in columns.get(table, [])}
            missing = sorted(expected - actual)
            if missing:
                raise CcsdsOemError(f"review store {table} is missing required columns: {missing}.")
        metadata_rows = workspace.query(
            "SELECT run_id, scenario_name, generated_utc, config_json, config_sha256 FROM run_metadata",
            max_rows=2,
        )
        if metadata_rows.truncated or metadata_rows.row_count != 1:
            raise CcsdsOemError("review store must contain exactly one run_metadata row.")
        run = dict(metadata_rows.rows[0])
        config_text = str(run["config_json"] or "")
        if hashlib.sha256(config_text.encode("utf-8")).hexdigest() != str(run["config_sha256"] or ""):
            raise CcsdsOemError("run_metadata config_json does not match config_sha256.")
        try:
            config = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise CcsdsOemError("run_metadata config_json is invalid JSON.") from exc
        initial_jd = _positive_finite(
            dict(dict(config).get("simulator", {}) or {}).get("initial_jd_utc"),
            "simulator.initial_jd_utc",
        )
        object_rows = workspace.query(
            "SELECT DISTINCT object_id FROM object_state ORDER BY object_id",
            max_rows=1001,
        )
        if object_rows.truncated:
            raise CcsdsOemError("object selection exceeds the 1000-object inspection limit.")
        available = [str(row["object_id"]) for row in object_rows.rows]
        requested = str(object_id or "").strip()
        if requested:
            if requested not in available:
                raise CcsdsOemError(f"object {requested!r} has no object_state history; available: {available}.")
            resolved_object = requested
        elif len(available) == 1:
            resolved_object = available[0]
        else:
            raise CcsdsOemError(f"object selection is ambiguous; specify one of: {available}.")
        frame_rows = workspace.query(
            "SELECT state_frame FROM object_state_frame WHERE object_id = ?",
            (resolved_object,),
            max_rows=2,
        )
        if frame_rows.truncated or frame_rows.row_count != 1:
            raise CcsdsOemError("selected object must have exactly one state-frame record.")
        if str(frame_rows.rows[0]["state_frame"] or "").strip().upper() != "ECI":
            raise CcsdsOemError("OEM export requires canonical OEL ECI review-state evidence.")
        state_rows = workspace.query(
            "SELECT sample_index, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
            "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s "
            "FROM object_state WHERE object_id = ? ORDER BY sample_index",
            (resolved_object,),
            max_rows=state_limit,
            max_vm_steps=max(250_000, state_limit * 200),
        )
        if state_rows.truncated:
            raise CcsdsOemError(f"object-state history exceeds the {state_limit} state export limit.")
        if not state_rows.rows:
            raise CcsdsOemError("selected object has no state rows.")
        covariance_rows = _review_covariance_rows(
            workspace,
            object_id=resolved_object,
            max_rows=state_limit,
        )
        review_identity = workspace.evidence_identity()
        if review_identity["sha256"] != starting_review_identity["sha256"]:
            raise CcsdsOemError("review store changed while the OEM export was being prepared.")

    states: list[OemState] = []
    initial_datetime = julian_date_to_datetime(initial_jd)
    previous_sample = -1
    previous_time = -math.inf
    epoch_by_sample: dict[int, tuple[float, str]] = {}
    for row in state_rows.rows:
        sample_index = int(row["sample_index"])
        time_s = _nonnegative_finite(row["time_s"], "object_state time_s")
        if sample_index <= previous_sample or time_s <= previous_time:
            raise CcsdsOemError("object-state samples must have strictly increasing indices and times.")
        previous_sample = sample_index
        previous_time = time_s
        epoch = _format_utc_epoch(initial_datetime + timedelta(seconds=time_s))
        epoch_by_sample[sample_index] = (time_s, epoch)
        values = [_finite_float(row[name], name) for name in (
            "pos_x_eci_km", "pos_y_eci_km", "pos_z_eci_km",
            "vel_x_eci_km_s", "vel_y_eci_km_s", "vel_z_eci_km_s",
        )]
        states.append(
            OemState(
                epoch=epoch,
                position_km=tuple(values[:3]),
                velocity_km_s=tuple(values[3:]),
            )
        )
    covariances: list[OemCovariance] = []
    calibrated_covariance_count = 0
    calibration_scopes: set[str] = set()
    for row in covariance_rows:
        sample_index = int(row["sample_index"])
        if sample_index not in epoch_by_sample:
            raise CcsdsOemError("Covariance row has no matching exported object-state sample.")
        expected_time, epoch = epoch_by_sample[sample_index]
        covariance_time = _finite_float(row["time_s"], "covariance time_s")
        if not math.isclose(covariance_time, expected_time, rel_tol=0.0, abs_tol=1.0e-9):
            raise CcsdsOemError("Covariance sample time does not match its object-state sample.")
        if str(row["frame"] or "").strip().upper() != "ECI":
            raise CcsdsOemError("OEM export requires canonical ECI covariance evidence.")
        component_order = _json_sequence(row["component_order_json"], "covariance component order")
        units = _json_sequence(row["units_json"], "covariance units")
        if component_order != _STATE_COMPONENT_ORDER or units != _STATE_UNITS:
            raise CcsdsOemError("Covariance ordering or units are incompatible with CCSDS OEM Cartesian state order.")
        matrix = _json_sequence(row["covariance_json"], "covariance matrix")
        covariance = OemCovariance(
            epoch=epoch,
            ref_frame=_OEL_PROFILE_FRAME,
            matrix=tuple(tuple(float(value) for value in matrix_row) for matrix_row in matrix),
        )
        _validate_covariance(covariance)
        if int(row["mathematically_valid"] or 0) != 1:
            raise CcsdsOemError("Covariance is not marked mathematically valid in the review store.")
        calibrated = bool(row["calibrated"])
        scope = str(row["calibration_scope"] or "").strip()
        if calibrated and not scope:
            raise CcsdsOemError("Calibrated covariance is missing calibration_scope.")
        if calibrated:
            calibrated_covariance_count += 1
            calibration_scopes.add(scope)
        covariances.append(covariance)
    created = _normalize_creation_date(run["generated_utc"])
    resolved_name = str(object_name or resolved_object).strip()
    if not resolved_name:
        raise CcsdsOemError("object_name must not be empty.")
    resolved_message_id = str(message_id or f"OEL-{run['run_id']}-{resolved_object}").strip()
    segment = OemSegment(
        metadata=OemMetadata(
            object_name=resolved_name,
            object_id=resolved_object,
            center_name=_OEL_PROFILE_CENTER,
            ref_frame=_OEL_PROFILE_FRAME,
            time_system=_OEL_PROFILE_TIME_SYSTEM,
            start_time=states[0].epoch,
            stop_time=states[-1].epoch,
        ),
        states=tuple(states),
        comments=(
            "OEL canonical OEL/ECI/J2000 review history serialized as CCSDS EME2000 without a numerical transform.",
        ),
        covariances=tuple(covariances),
    )
    message = OemMessage(
        header=OemHeader(
            version=CCSDS_OEM_VERSION,
            creation_date=created,
            originator=str(originator or "").strip(),
            message_id=resolved_message_id,
            comments=(f"Generated from OEL completed run {run['run_id']}.",),
        ),
        segments=(segment,),
    )
    oem_text = serialize_oem_kvn(message)
    oem_bytes = oem_text.encode("utf-8")
    oem_sha256 = hashlib.sha256(oem_bytes).hexdigest()
    reparsed = parse_oem_kvn(oem_text, source_sha256=oem_sha256)
    parity = compare_oem(message, reparsed)
    if parity["status"] != "equivalent":
        raise CcsdsOemError("Serialized OEM failed semantic read-back comparison.")
    receipt = {
        "schema": "oel.ccsds-oem-export-receipt.v1",
        "status": "exported",
        "oel_profile": OEL_OEM_PROFILE,
        "oem_path": str(target),
        "oem_sha256": oem_sha256,
        "source_run_id": str(run["run_id"]),
        "source_scenario_name": str(run["scenario_name"]),
        "source_review_store": review_identity,
        "object_id": resolved_object,
        "state_count": len(states),
        "covariance_count": len(covariances),
        "calibrated_covariance_count": calibrated_covariance_count,
        "covariance_calibration_scopes": sorted(calibration_scopes),
        "source_frame": "OEL/ECI/J2000",
        "oem_ref_frame": _OEL_PROFILE_FRAME,
        "frame_transform_applied": False,
        "time_system": _OEL_PROFILE_TIME_SYSTEM,
        "interpolation_declared": False,
        "semantic_readback": parity,
        "execution_occurred": False,
        "non_claims": [
            "The export preserves completed-run simulator truth; it is not an orbit determination result.",
            "The OEL/ECI/J2000 to EME2000 mapping is an identity under the OEL frames-v1 convention.",
            "Exported covariance retains its source calibration flags in the receipt; OEM syntax alone does not establish calibration.",
            "No attitude, force-model detail, maneuver event, or interpolation method is exported.",
        ],
    }
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    receipt_text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_bytes() != oem_bytes and not overwrite:
        raise CcsdsOemError("OEM output exists with different content; pass overwrite=True to replace it.")
    if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != receipt_text and not overwrite:
        raise CcsdsOemError("OEM receipt exists with different content; pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != oem_bytes:
        target.write_bytes(oem_bytes)
    if sha256_file(target) != oem_sha256:
        raise CcsdsOemError("Written OEM does not match the prepared content hash.")
    if not receipt_path.exists() or receipt_path.read_text(encoding="utf-8") != receipt_text:
        receipt_path.write_text(receipt_text, encoding="utf-8")
    return {**receipt, "receipt_path": str(receipt_path)}


def _review_covariance_rows(
    workspace: ReviewWorkspace,
    *,
    object_id: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    columns = workspace.table_columns().get("object_state_covariance", [])
    if not columns:
        return []
    required = {
        "sample_index",
        "time_s",
        "object_id",
        "frame",
        "component_order_json",
        "units_json",
        "covariance_json",
        "mathematically_valid",
        "calibrated",
        "calibration_scope",
    }
    actual = {str(item["name"]) for item in columns}
    missing = sorted(required - actual)
    if missing:
        raise CcsdsOemError(f"review store object_state_covariance is missing required columns: {missing}.")
    result = workspace.query(
        "SELECT sample_index, time_s, object_id, frame, component_order_json, units_json, "
        "covariance_json, mathematically_valid, calibrated, calibration_scope "
        "FROM object_state_covariance WHERE object_id = ? ORDER BY sample_index",
        (object_id,),
        max_rows=max_rows,
        max_vm_steps=max(250_000, max_rows * 200),
    )
    if result.truncated:
        raise CcsdsOemError(f"covariance history exceeds the {max_rows} row export limit.")
    rows = [dict(row) for row in result.rows]
    indices = [int(row["sample_index"]) for row in rows]
    if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
        raise CcsdsOemError("Covariance sample indices must be unique and strictly increasing.")
    return rows


def _json_sequence(value: Any, label: str) -> list[Any]:
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise CcsdsOemError(f"{label} is not valid JSON.") from exc
    if not isinstance(payload, list):
        raise CcsdsOemError(f"{label} must be a JSON list.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.ccsds",
        description="Inspect and exchange bounded CCSDS OEM 3.0 KVN products without executing scenarios.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect-oem", help="Parse and inspect an OEM read-only.")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    roundtrip_parser = commands.add_parser("roundtrip-oem", help="Parse and deterministically reserialize an OEM.")
    roundtrip_parser.add_argument("path", type=Path)
    roundtrip_parser.add_argument("--output", required=True, type=Path)
    roundtrip_parser.add_argument("--overwrite", action="store_true")
    roundtrip_parser.add_argument("--json", action="store_true")
    compare_parser = commands.add_parser("compare-oem", help="Compare two OEMs semantically.")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--position-tolerance-km", type=float, default=1.0e-9)
    compare_parser.add_argument("--velocity-tolerance-km-s", type=float, default=1.0e-12)
    compare_parser.add_argument("--acceleration-tolerance-km-s2", type=float, default=1.0e-15)
    compare_parser.add_argument("--covariance-absolute-tolerance", type=float, default=1.0e-18)
    compare_parser.add_argument("--covariance-relative-tolerance", type=float, default=1.0e-12)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.add_argument("--json", action="store_true")
    import_parser = commands.add_parser("import-oem", help="Write an OEL mission-input packet from one OEM segment.")
    import_parser.add_argument("path", type=Path)
    import_parser.add_argument("--output", required=True, type=Path)
    import_parser.add_argument("--json", action="store_true")
    export_parser = commands.add_parser("export-oem", help="Export one completed-run state history as OEM.")
    export_parser.add_argument("completed_run", type=Path)
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument("--object-id")
    export_parser.add_argument("--object-name")
    export_parser.add_argument("--originator", default="OEL")
    export_parser.add_argument("--message-id")
    export_parser.add_argument("--max-states", type=int, default=MAX_OEM_STATES)
    export_parser.add_argument("--overwrite", action="store_true")
    export_parser.add_argument("--json", action="store_true")
    convert_parser = commands.add_parser(
        "convert-oem",
        help="Explicitly convert OEM Cartesian state/covariance frame and time metadata.",
    )
    convert_parser.add_argument("path", type=Path)
    convert_parser.add_argument("--output", required=True, type=Path)
    convert_parser.add_argument("--target-frame", required=True)
    convert_parser.add_argument("--target-time-system", required=True)
    convert_parser.add_argument("--eop", type=Path)
    convert_parser.add_argument("--eop-format", choices=("auto", "finals2000a", "c04_csv"), default="auto")
    convert_parser.add_argument("--overwrite", action="store_true")
    convert_parser.add_argument("--json", action="store_true")
    odm_inspect_parser = commands.add_parser("inspect-odm", help="Parse and inspect an OPM or OMM read-only.")
    odm_inspect_parser.add_argument("path", type=Path)
    odm_inspect_parser.add_argument("--json", action="store_true")
    odm_roundtrip_parser = commands.add_parser(
        "roundtrip-odm",
        help="Parse and deterministically reserialize an OPM or OMM.",
    )
    odm_roundtrip_parser.add_argument("path", type=Path)
    odm_roundtrip_parser.add_argument("--output", required=True, type=Path)
    odm_roundtrip_parser.add_argument("--overwrite", action="store_true")
    odm_roundtrip_parser.add_argument("--json", action="store_true")
    odm_compare_parser = commands.add_parser("compare-odm", help="Compare two OPM/OMM products semantically.")
    odm_compare_parser.add_argument("left", type=Path)
    odm_compare_parser.add_argument("right", type=Path)
    odm_compare_parser.add_argument("--output", type=Path)
    odm_compare_parser.add_argument("--json", action="store_true")
    opm_import_parser = commands.add_parser(
        "import-opm",
        help="Write a mission-input packet from one Earth/EME2000/UTC OPM state.",
    )
    opm_import_parser.add_argument("path", type=Path)
    opm_import_parser.add_argument("--output", required=True, type=Path)
    opm_import_parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-oem":
            payload = inspect_oem(args.path)
        elif args.command == "roundtrip-oem":
            message = read_oem_kvn(args.path)
            write_oem_kvn(message, args.output, overwrite=args.overwrite)
            payload = {
                "status": "written",
                "output_path": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "semantic_comparison": compare_oem(message, args.output),
                "execution_occurred": False,
            }
        elif args.command == "compare-oem":
            payload = compare_oem(
                args.left,
                args.right,
                position_tolerance_km=args.position_tolerance_km,
                velocity_tolerance_km_s=args.velocity_tolerance_km_s,
                acceleration_tolerance_km_s2=args.acceleration_tolerance_km_s2,
                covariance_absolute_tolerance=args.covariance_absolute_tolerance,
                covariance_relative_tolerance=args.covariance_relative_tolerance,
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.command == "import-oem":
            packet = oem_to_mission_input_packet(args.path)
            packet.write_json(args.output)
            payload = {
                "status": "imported",
                "packet_path": str(args.output.resolve()),
                "packet_sha256": sha256_file(args.output),
                "warnings": packet.warnings,
                "execution_occurred": False,
            }
        elif args.command == "export-oem":
            payload = export_completed_run_oem(
                args.completed_run,
                output_path=args.output,
                object_id=args.object_id,
                object_name=args.object_name,
                originator=args.originator,
                message_id=args.message_id,
                overwrite=args.overwrite,
                max_states=args.max_states,
            )
        elif args.command == "convert-oem":
            payload = convert_oem_kvn(
                args.path,
                output_path=args.output,
                target_frame=args.target_frame,
                target_time_system=args.target_time_system,
                eop_path=args.eop,
                eop_format=args.eop_format,
                overwrite=args.overwrite,
            )
        elif args.command == "inspect-odm":
            payload = inspect_odm(args.path)
        elif args.command == "roundtrip-odm":
            message = read_odm_kvn(args.path)
            write_odm_kvn(message, args.output, overwrite=args.overwrite)
            payload = {
                "status": "written",
                "output_path": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "semantic_comparison": compare_odm(message, args.output),
                "execution_occurred": False,
            }
        elif args.command == "compare-odm":
            payload = compare_odm(args.left, args.right)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            packet = opm_to_mission_input_packet(args.path)
            packet.write_json(args.output)
            payload = {
                "status": "imported",
                "packet_path": str(args.output.resolve()),
                "packet_sha256": sha256_file(args.output),
                "warnings": packet.warnings,
                "execution_occurred": False,
            }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for key in ("status", "document_type", "oel_profile", "valid_oem", "oel_import_ready", "mission_input_ready", "segment_count", "state_count", "covariance_count", "maneuver_count", "output_path", "packet_path", "oem_path", "receipt_path"):
                if key in payload:
                    print(f"{key}: {payload[key]}")
        return 0 if payload.get("status") not in {"different"} else 2
    except (OSError, CcsdsOdmError, CcsdsOemError, ValueError) as exc:
        print(f"CCSDS command failed: {exc}", file=sys.stderr)
        return 2


def _header_from_values(values: Mapping[str, str], comments: Sequence[str]) -> OemHeader:
    missing = sorted(_REQUIRED_HEADER_KEYS - set(values))
    if missing:
        raise CcsdsOemError(f"OEM header is missing required keywords: {missing}.")
    _validate_keyword_order(values, _HEADER_KEYS, label="OEM header")
    return OemHeader(
        version=values["CCSDS_OEM_VERS"],
        creation_date=values["CREATION_DATE"],
        originator=values["ORIGINATOR"],
        message_id=values.get("MESSAGE_ID"),
        classification=values.get("CLASSIFICATION"),
        comments=tuple(comments),
    )


def _metadata_from_values(values: Mapping[str, str], comments: Sequence[str]) -> OemMetadata:
    missing = sorted(_REQUIRED_METADATA_KEYS - set(values))
    if missing:
        raise CcsdsOemError(f"OEM metadata is missing required keywords: {missing}.")
    _validate_keyword_order(values, _METADATA_KEYS, label="OEM metadata")
    interpolation = values.get("INTERPOLATION")
    degree_raw = values.get("INTERPOLATION_DEGREE")
    if (interpolation is None) != (degree_raw is None):
        raise CcsdsOemError("INTERPOLATION and INTERPOLATION_DEGREE must be supplied together.")
    degree: int | None = None
    if degree_raw is not None:
        try:
            degree = int(degree_raw)
        except ValueError as exc:
            raise CcsdsOemError("INTERPOLATION_DEGREE must be an integer.") from exc
        if degree < 1:
            raise CcsdsOemError("INTERPOLATION_DEGREE must be positive.")
    return OemMetadata(
        object_name=values["OBJECT_NAME"],
        object_id=values["OBJECT_ID"],
        center_name=values["CENTER_NAME"],
        ref_frame=values["REF_FRAME"],
        ref_frame_epoch=values.get("REF_FRAME_EPOCH"),
        time_system=values["TIME_SYSTEM"],
        start_time=values["START_TIME"],
        usable_start_time=values.get("USEABLE_START_TIME"),
        usable_stop_time=values.get("USEABLE_STOP_TIME"),
        stop_time=values["STOP_TIME"],
        interpolation=interpolation,
        interpolation_degree=degree,
        comments=tuple(comments),
    )


def _validate_segment(segment: OemSegment) -> None:
    metadata = segment.metadata
    for label, value in (
        ("OBJECT_NAME", metadata.object_name),
        ("OBJECT_ID", metadata.object_id),
        ("CENTER_NAME", metadata.center_name),
        ("REF_FRAME", metadata.ref_frame),
        ("TIME_SYSTEM", metadata.time_system),
    ):
        if not value.strip():
            raise CcsdsOemError(f"{label} must not be empty.")
    start = _parse_epoch(metadata.start_time, label="START_TIME")
    stop = _parse_epoch(metadata.stop_time, label="STOP_TIME")
    if stop < start:
        raise CcsdsOemError("STOP_TIME must not precede START_TIME.")
    usable_start = _parse_optional_epoch(metadata.usable_start_time, label="USEABLE_START_TIME")
    usable_stop = _parse_optional_epoch(metadata.usable_stop_time, label="USEABLE_STOP_TIME")
    if usable_start is not None and usable_start < start:
        raise CcsdsOemError("USEABLE_START_TIME must not precede START_TIME.")
    if usable_stop is not None and usable_stop > stop:
        raise CcsdsOemError("USEABLE_STOP_TIME must not follow STOP_TIME.")
    if usable_start is not None and usable_stop is not None and usable_stop < usable_start:
        raise CcsdsOemError("USEABLE_STOP_TIME must not precede USEABLE_START_TIME.")
    if metadata.ref_frame_epoch is not None:
        _parse_epoch(metadata.ref_frame_epoch, label="REF_FRAME_EPOCH")
    if not segment.states:
        raise CcsdsOemError("Each OEM segment must contain at least one state row.")
    if metadata.interpolation is not None:
        if metadata.interpolation_degree is None:
            raise CcsdsOemError("INTERPOLATION requires INTERPOLATION_DEGREE.")
        if len(segment.states) < metadata.interpolation_degree + 1:
            raise CcsdsOemError("OEM segment does not contain enough states for its interpolation degree.")
    state_epochs = [_parse_epoch(state.epoch, label="state epoch") for state in segment.states]
    if any(later <= earlier for earlier, later in zip(state_epochs, state_epochs[1:])):
        raise CcsdsOemError("OEM state epochs must be strictly increasing within a segment.")
    if state_epochs[0] < start or state_epochs[-1] > stop:
        raise CcsdsOemError("OEM state epochs must remain inside START_TIME and STOP_TIME.")
    covariance_epochs = [_parse_epoch(covariance.epoch, label="covariance EPOCH") for covariance in segment.covariances]
    if any(later <= earlier for earlier, later in zip(covariance_epochs, covariance_epochs[1:])):
        raise CcsdsOemError("OEM covariance epochs must be strictly increasing within a segment.")
    if covariance_epochs and (covariance_epochs[0] < start or covariance_epochs[-1] > stop):
        raise CcsdsOemError("OEM covariance epochs must remain inside START_TIME and STOP_TIME.")
    covered_epochs = [*state_epochs, *covariance_epochs]
    if min(covered_epochs) != start or max(covered_epochs) != stop:
        raise CcsdsOemError("START_TIME and STOP_TIME must equal the total state and covariance data span.")
    acceleration_flags = {state.acceleration_km_s2 is not None for state in segment.states}
    if len(acceleration_flags) != 1:
        raise CcsdsOemError("Acceleration components must be present on every state row or none.")
    for state in segment.states:
        values: Iterable[float] = (*state.position_km, *state.velocity_km_s)
        if state.acceleration_km_s2 is not None:
            values = (*values, *state.acceleration_km_s2)
        if not all(math.isfinite(float(value)) for value in values):
            raise CcsdsOemError("OEM state values must be finite.")
    for covariance in segment.covariances:
        _validate_covariance(covariance)


def _validate_covariance(covariance: OemCovariance) -> None:
    if covariance.ref_frame is not None and not str(covariance.ref_frame).strip():
        raise CcsdsOemError("COV_REF_FRAME must not be empty when supplied.")
    _parse_epoch(covariance.epoch, label="covariance EPOCH")
    matrix = np.asarray(covariance.matrix, dtype=float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise CcsdsOemError("OEM covariance must be a finite 6x6 matrix.")
    scale = float(np.max(np.abs(matrix)))
    tolerance = 64.0 * np.finfo(float).eps * scale * matrix.shape[0]
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise CcsdsOemError("OEM covariance must be symmetric.")
    symmetric = 0.5 * (matrix + matrix.T)
    if np.any(np.diag(symmetric) < 0.0):
        raise CcsdsOemError("OEM covariance diagonal variances must be non-negative.")
    if float(np.min(np.linalg.eigvalsh(symmetric))) < -tolerance:
        raise CcsdsOemError("OEM covariance must be positive semidefinite within the v0.2 tolerance.")


def _parse_assignment(line: str, *, line_number: int) -> tuple[str, str]:
    if "=" not in line:
        raise CcsdsOemError(f"line {line_number}: expected KEY = VALUE assignment.")
    key, value = (part.strip() for part in line.split("=", 1))
    if not key or not value:
        raise CcsdsOemError(f"line {line_number}: OEM assignment key and value must not be empty.")
    return key.upper(), value


def _parse_comment(line: str, *, line_number: int) -> str:
    if line == "COMMENT" or not line.startswith("COMMENT "):
        raise CcsdsOemError(f"line {line_number}: COMMENT must be followed by at least one space and text.")
    value = line[len("COMMENT") :].strip()
    if not value:
        raise CcsdsOemError(f"line {line_number}: COMMENT text must not be empty.")
    return value


def _store_unique(
    destination: dict[str, str] | None,
    key: str,
    value: str,
    *,
    allowed: Sequence[str],
    line_number: int,
) -> None:
    if destination is None:
        raise CcsdsOemError(f"line {line_number}: assignment appears outside an active section.")
    if key not in allowed:
        raise CcsdsOemError(f"line {line_number}: unsupported or misplaced OEM keyword {key!r}.")
    if key in destination:
        raise CcsdsOemError(f"line {line_number}: duplicate OEM keyword {key!r}.")
    destination[key] = value


def _parse_state_line(line: str, *, line_number: int) -> OemState:
    tokens = line.split()
    if len(tokens) not in {7, 10}:
        raise CcsdsOemError(f"line {line_number}: OEM state row must contain 7 or 10 fields.")
    _parse_epoch(tokens[0], label=f"line {line_number} epoch")
    values = []
    for token in tokens[1:]:
        if not _FLOAT_TOKEN.fullmatch(token):
            raise CcsdsOemError(f"line {line_number}: invalid numeric token {token!r}.")
        value = float(token)
        if not math.isfinite(value):
            raise CcsdsOemError(f"line {line_number}: state values must be finite.")
        values.append(value)
    return OemState(
        epoch=tokens[0],
        position_km=tuple(values[:3]),
        velocity_km_s=tuple(values[3:6]),
        acceleration_km_s2=tuple(values[6:9]) if len(values) == 9 else None,
    )


def _parse_covariance_row(line: str, *, expected: int, line_number: int) -> tuple[float, ...]:
    tokens = line.split()
    if len(tokens) != expected:
        raise CcsdsOemError(
            f"line {line_number}: covariance row {expected} must contain exactly {expected} values."
        )
    values = []
    for token in tokens:
        if not _FLOAT_TOKEN.fullmatch(token):
            raise CcsdsOemError(f"line {line_number}: invalid covariance numeric token {token!r}.")
        value = float(token)
        if not math.isfinite(value):
            raise CcsdsOemError(f"line {line_number}: covariance values must be finite.")
        values.append(value)
    return tuple(values)


def _serialize_state(state: OemState) -> str:
    values = [*state.position_km, *state.velocity_km_s]
    if state.acceleration_km_s2 is not None:
        values.extend(state.acceleration_km_s2)
    return " ".join([state.epoch, *(_format_float(value) for value in values)])


def _parse_epoch(value: str, *, label: str) -> datetime:
    text = str(value or "").strip()
    match = _CALENDAR_EPOCH.fullmatch(text)
    if match:
        parts = match.groupdict()
        second = int(parts["second"])
        if second == 60:
            raise CcsdsOemError(f"{label} uses a leap-second timestamp, which is outside {OEL_OEM_PROFILE}.")
        microsecond = _fraction_to_microseconds(parts.get("fraction"))
        try:
            return datetime(
                int(parts["year"]),
                int(parts["month"]),
                int(parts["day"]),
                int(parts["hour"]),
                int(parts["minute"]),
                second,
                microsecond,
                tzinfo=timezone.utc,
            )
        except ValueError as exc:
            raise CcsdsOemError(f"{label} is not a valid CCSDS calendar epoch: {text!r}.") from exc
    match = _ORDINAL_EPOCH.fullmatch(text)
    if match:
        parts = match.groupdict()
        second = int(parts["second"])
        if second == 60:
            raise CcsdsOemError(f"{label} uses a leap-second timestamp, which is outside {OEL_OEM_PROFILE}.")
        microsecond = _fraction_to_microseconds(parts.get("fraction"))
        try:
            year = int(parts["year"])
            day_of_year = int(parts["doy"])
            if day_of_year < 1:
                raise ValueError("ordinal day must be positive")
            base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
            if base.year != year:
                raise ValueError("ordinal day exceeds the number of days in the year")
            return base.replace(
                hour=int(parts["hour"]),
                minute=int(parts["minute"]),
                second=second,
                microsecond=microsecond,
            )
        except ValueError as exc:
            raise CcsdsOemError(f"{label} is not a valid CCSDS ordinal epoch: {text!r}.") from exc
    raise CcsdsOemError(f"{label} is not a supported absolute CCSDS epoch: {text!r}.")


def _parse_optional_epoch(value: str | None, *, label: str) -> datetime | None:
    return None if value is None else _parse_epoch(value, label=label)


def _fraction_to_microseconds(value: str | None) -> int:
    if value is None:
        return 0
    digits = value[1:]
    if len(digits) > 6:
        discarded = digits[6:]
        if any(char != "0" for char in discarded):
            raise CcsdsOemError(
                f"{OEL_OEM_PROFILE} supports epoch precision through microseconds; nonzero excess digits are rejected."
            )
    return int((digits[:6] + "000000")[:6])


def _epoch_key(value: str) -> int:
    epoch = _parse_epoch(value, label="epoch")
    return int(round(epoch.timestamp() * 1_000_000.0))


def _format_utc_epoch(value: datetime) -> str:
    dt = value.astimezone(timezone.utc)
    text = dt.strftime("%Y-%m-%dT%H:%M:%S.%f").rstrip("0").rstrip(".")
    return text


def _normalize_creation_date(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CcsdsOemError("run_metadata generated_utc is invalid.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return _format_utc_epoch(parsed)


def _format_float(value: Any) -> str:
    number = _finite_float(value, "OEM numeric value")
    return f"{number:.17g}"


def _clean_comment(value: str) -> str:
    text = str(value).strip()
    if not text or "\n" in text or "\r" in text:
        raise CcsdsOemError("OEM comments must contain one non-empty line.")
    return text


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CcsdsOemError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise CcsdsOemError(f"{label} must be a finite number.")
    return number


def _positive_finite(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise CcsdsOemError(f"{label} must be positive.")
    return number


def _nonnegative_finite(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0.0:
        raise CcsdsOemError(f"{label} must be non-negative.")
    return number


def _issue(segment_index: int, field: str, message: str) -> dict[str, str]:
    return {
        "path": f"segments[{segment_index}].metadata.{field}",
        "code": f"unsupported_{field}",
        "message": message,
    }


def _validate_keyword_order(values: Mapping[str, str], allowed: Sequence[str], *, label: str) -> None:
    indices = [allowed.index(key) for key in values]
    if indices != sorted(indices):
        raise CcsdsOemError(f"{label} keywords are not in the required CCSDS order.")


def _metadata_semantics(metadata: OemMetadata) -> tuple[Any, ...]:
    def epoch(value: str | None) -> int | None:
        return None if value is None else _epoch_key(value)

    return (
        metadata.object_name,
        metadata.object_id,
        metadata.center_name.upper(),
        metadata.ref_frame.upper(),
        epoch(metadata.ref_frame_epoch),
        metadata.time_system.upper(),
        epoch(metadata.start_time),
        epoch(metadata.usable_start_time),
        epoch(metadata.usable_stop_time),
        epoch(metadata.stop_time),
        None if metadata.interpolation is None else metadata.interpolation.upper(),
        metadata.interpolation_degree,
    )


__all__ = [
    "CCSDS_OEM_VERSION",
    "OEL_OEM_PROFILE",
    "CcsdsOemError",
    "OemCovariance",
    "OemHeader",
    "OemMessage",
    "OemMetadata",
    "OemSegment",
    "OemState",
    "compare_oem",
    "convert_oem",
    "convert_oem_kvn",
    "export_completed_run_oem",
    "inspect_oem",
    "main",
    "oel_profile_issues",
    "oem_to_mission_input_packet",
    "parse_oem_kvn",
    "read_oem_kvn",
    "serialize_oem_kvn",
    "validate_oem",
    "write_oem_kvn",
]
