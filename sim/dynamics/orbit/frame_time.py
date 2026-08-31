"""Canonical public frame and time-scale contracts over OEL's orbit-frame owner."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import erfa
import numpy as np

from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    FrameContext,
    eci_to_ecef_rotation_context,
    eci_to_ecef_rotation_derivative_context,
    teme_to_eci_matrix_vallado_iau80,
)

FRAME_TIME_CONTRACT = "oel.frame-time.v1"
FRAME_TRANSFORM_MODEL = "oel.iau76-fk5-iau80-eop.v1"
FRAME_TRANSFORM_MODEL_IAU2006 = "oel.iau2006-iau2000a-cio-eop.v1"
LEAP_SECOND_RESOURCE = Path(__file__).with_name("data") / "leap_seconds_iers_bulletin_c_72.json"
LEAP_SECOND_RESOURCE_NAME = "sim/dynamics/orbit/data/leap_seconds_iers_bulletin_c_72.json"
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_JD_UNIX_EPOCH = 2440587.5
_DAY_S = 86400.0
_TT_MINUS_TAI_S = 32.184
_CALENDAR_EPOCH = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?P<fraction>\.\d+)?(?P<z>Z)?$"
)
_ORDINAL_EPOCH = re.compile(
    r"^(?P<year>\d{4})-(?P<doy>\d{3})T"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?P<fraction>\.\d+)?(?P<z>Z)?$"
)


class FrameTimeError(ValueError):
    """Raised when a frame or time conversion is outside the public contract."""


class TimeScale(str, Enum):
    UTC = "UTC"
    TAI = "TAI"
    TT = "TT"
    UT1 = "UT1"


class CanonicalFrame(str, Enum):
    EME2000 = "EME2000"
    TEME = "TEME"
    ITRF = "ITRF"
    GCRF = "GCRF"


@dataclass(frozen=True)
class Epoch:
    """One physical instant represented as linear TAI seconds from 1970-01-01 TAI."""

    tai_seconds: float
    source_scale: TimeScale
    source_text: str
    contract: str = FRAME_TIME_CONTRACT

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.tai_seconds)):
            raise FrameTimeError("Epoch TAI seconds must be finite.")
        if not isinstance(self.source_scale, TimeScale):
            raise TypeError("source_scale must be a TimeScale.")
        if not str(self.source_text).strip():
            raise FrameTimeError("source_text must not be empty.")


@dataclass(frozen=True)
class EarthOrientation:
    """Earth-orientation values sampled for one exact epoch."""

    dut1_s: float
    xp_arcsec: float
    yp_arcsec: float
    source: str
    source_sha256: str | None = None
    ddpsi_rad: float = 0.0
    ddeps_rad: float = 0.0
    dx_mas: float = 0.0
    dy_mas: float = 0.0

    def __post_init__(self) -> None:
        for name in ("dut1_s", "xp_arcsec", "yp_arcsec", "ddpsi_rad", "ddeps_rad", "dx_mas", "dy_mas"):
            if not math.isfinite(float(getattr(self, name))):
                raise FrameTimeError(f"{name} must be finite.")
        if abs(float(self.dut1_s)) >= 2.0:
            raise FrameTimeError("dut1_s must have magnitude below 2 seconds.")
        if abs(float(self.xp_arcsec)) >= 10.0 or abs(float(self.yp_arcsec)) >= 10.0:
            raise FrameTimeError("Polar-motion coordinates must have magnitude below 10 arcseconds.")
        if not str(self.source).strip():
            raise FrameTimeError("Earth-orientation source must not be empty.")
        if self.source_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise FrameTimeError("source_sha256 must be a lowercase SHA-256 digest.")


@dataclass(frozen=True)
class FrameTransformContext:
    epoch: Epoch
    earth_orientation: EarthOrientation | None = None
    contract: str = FRAME_TIME_CONTRACT

    def orbit_frame_context(self, *, require_eop: bool) -> FrameContext:
        if require_eop and self.earth_orientation is None:
            raise FrameTimeError("ITRF transformations require sampled DUT1 and polar-motion values with provenance.")
        jd_utc = epoch_julian_date(self.epoch, TimeScale.UTC)
        dat_s = tai_minus_utc(self.epoch)
        eop = self.earth_orientation
        return FrameContext(
            model=FRAME_MODEL_IAU76_80_EOP,
            jd_utc_start=jd_utc,
            time_scale_model="iers_utc_tai_ut1_tt",
            tt_minus_utc_s=dat_s + _TT_MINUS_TAI_S,
            dut1_s=None if eop is None else float(eop.dut1_s),
            xp_arcsec=None if eop is None else float(eop.xp_arcsec),
            yp_arcsec=None if eop is None else float(eop.yp_arcsec),
            dat_s=dat_s,
            ddpsi_rad=0.0 if eop is None else float(eop.ddpsi_rad),
            ddeps_rad=0.0 if eop is None else float(eop.ddeps_rad),
            source="canonical_frame_time_contract",
        )


@dataclass(frozen=True)
class _LeapEntry:
    effective_utc: datetime
    effective_utc_seconds: float
    tai_minus_utc_s: int


@dataclass(frozen=True)
class _LeapTable:
    table_id: str
    valid_from: datetime
    valid_through_exclusive: datetime
    entries: tuple[_LeapEntry, ...]
    sha256: str
    source: dict[str, Any]


@lru_cache(maxsize=1)
def _leap_table() -> _LeapTable:
    raw = LEAP_SECOND_RESOURCE.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    entries = []
    for item in payload["entries"]:
        effective = datetime.fromisoformat(str(item["effective_utc"])).replace(tzinfo=timezone.utc)
        entries.append(
            _LeapEntry(
                effective_utc=effective,
                effective_utc_seconds=(effective - _UNIX_EPOCH).total_seconds(),
                tai_minus_utc_s=int(item["tai_minus_utc_s"]),
            )
        )
    if not entries or any(
        right.effective_utc <= left.effective_utc
        or right.tai_minus_utc_s != left.tai_minus_utc_s + 1
        for left, right in zip(entries, entries[1:])
    ):
        raise FrameTimeError("Packaged leap-second entries are incomplete or out of order.")
    valid_from = datetime.fromisoformat(str(payload["valid_from_utc"])).replace(tzinfo=timezone.utc)
    valid_through = datetime.fromisoformat(str(payload["valid_through_utc"])).replace(tzinfo=timezone.utc)
    return _LeapTable(
        table_id=str(payload["table_id"]),
        valid_from=valid_from,
        valid_through_exclusive=valid_through + timedelta(microseconds=1),
        entries=tuple(entries),
        sha256=hashlib.sha256(raw).hexdigest(),
        source=dict(payload["source"]),
    )


def leap_second_table_receipt() -> dict[str, Any]:
    table = _leap_table()
    return {
        "schema": "oel.leap-second-table-receipt.v1",
        "table_id": table.table_id,
        "resource": LEAP_SECOND_RESOURCE_NAME,
        "sha256": table.sha256,
        "valid_from_utc": _format_datetime(table.valid_from),
        "valid_through_utc": _format_datetime(table.valid_through_exclusive - timedelta(microseconds=1)),
        "source": dict(table.source),
    }


def parse_epoch(text: str, scale: TimeScale | str, *, dut1_s: float | None = None) -> Epoch:
    resolved_scale = _time_scale(scale)
    components = _parse_components(text, allow_leap=resolved_scale is TimeScale.UTC)
    if components["z"] and resolved_scale is not TimeScale.UTC:
        raise FrameTimeError("A trailing Z is accepted only for UTC epochs.")
    if resolved_scale is TimeScale.UTC:
        tai_seconds = _utc_components_to_tai(components)
    else:
        if components["second"] == 60:
            raise FrameTimeError(f"{resolved_scale.value} does not use UTC leap-second notation.")
        calendar_seconds = _components_to_calendar_seconds(components)
        if resolved_scale is TimeScale.TAI:
            tai_seconds = calendar_seconds
        elif resolved_scale is TimeScale.TT:
            tai_seconds = calendar_seconds - _TT_MINUS_TAI_S
        else:
            dut1 = _require_dut1(dut1_s)
            utc_seconds = calendar_seconds - dut1
            offset = _utc_offset_for_seconds(utc_seconds)
            tai_seconds = utc_seconds + offset
    return Epoch(float(tai_seconds), resolved_scale, str(text))


def format_epoch(
    epoch: Epoch,
    scale: TimeScale | str,
    *,
    dut1_s: float | None = None,
    include_z: bool = False,
) -> str:
    resolved_scale = _time_scale(scale)
    if resolved_scale is TimeScale.UTC:
        text, _utc_seconds, _offset, _inside_leap = _utc_from_tai(epoch.tai_seconds)
    elif resolved_scale is TimeScale.TAI:
        text = _format_calendar_seconds(epoch.tai_seconds)
    elif resolved_scale is TimeScale.TT:
        text = _format_calendar_seconds(epoch.tai_seconds + _TT_MINUS_TAI_S)
    else:
        dut1 = _require_dut1(dut1_s)
        _utc_text, utc_seconds, _offset, inside_leap = _utc_from_tai(epoch.tai_seconds)
        if inside_leap:
            raise FrameTimeError("UT1 formatting at a UTC leap-second instant is outside the v1 contract.")
        text = _format_calendar_seconds(utc_seconds + dut1)
    return text + ("Z" if include_z and resolved_scale is TimeScale.UTC else "")


def epoch_julian_date(epoch: Epoch, scale: TimeScale | str, *, dut1_s: float | None = None) -> float:
    resolved_scale = _time_scale(scale)
    if resolved_scale is TimeScale.TAI:
        seconds = epoch.tai_seconds
    elif resolved_scale is TimeScale.TT:
        seconds = epoch.tai_seconds + _TT_MINUS_TAI_S
    else:
        _text, utc_seconds, _offset, inside_leap = _utc_from_tai(epoch.tai_seconds)
        if inside_leap:
            raise FrameTimeError("Scalar Julian dates at a UTC leap-second instant are outside the v1 contract.")
        seconds = utc_seconds if resolved_scale is TimeScale.UTC else utc_seconds + _require_dut1(dut1_s)
    return _JD_UNIX_EPOCH + float(seconds) / _DAY_S


def tai_minus_utc(epoch: Epoch) -> int:
    _text, _seconds, offset, _inside_leap = _utc_from_tai(epoch.tai_seconds)
    return int(offset)


def epoch_conversion_receipt(
    epoch: Epoch,
    target_scale: TimeScale | str,
    *,
    dut1_s: float | None = None,
) -> dict[str, Any]:
    target = _time_scale(target_scale)
    return {
        "schema": "oel.epoch-conversion-receipt.v1",
        "contract": FRAME_TIME_CONTRACT,
        "input": {"text": epoch.source_text, "scale": epoch.source_scale.value},
        "output": {"text": format_epoch(epoch, target, dut1_s=dut1_s), "scale": target.value},
        "tai_seconds": epoch.tai_seconds,
        "tai_minus_utc_s": tai_minus_utc(epoch),
        "tt_minus_tai_s": _TT_MINUS_TAI_S,
        "dut1_s": None if dut1_s is None else float(dut1_s),
        "leap_seconds": leap_second_table_receipt(),
        "non_claims": [
            "The v1 contract covers UTC, TAI, TT, and sampled-UT1 conversions from 1972 through the table validity bound.",
            "UT1 conversion requires an explicit epoch-matched DUT1 value and does not predict Earth orientation.",
            "TCG, TDB, TCB, GPS, and pre-1972 drift-era UTC are outside this contract.",
        ],
    }


def normalize_canonical_frame(value: CanonicalFrame | str) -> CanonicalFrame:
    if isinstance(value, CanonicalFrame):
        return value
    key = str(value or "").strip().upper()
    aliases = {
        "EME2000": CanonicalFrame.EME2000,
        "OEL/ECI/J2000": CanonicalFrame.EME2000,
        "TEME": CanonicalFrame.TEME,
        "ITRF": CanonicalFrame.ITRF,
        "OEL/ECEF/IAU76_80_EOP": CanonicalFrame.ITRF,
        "GCRF": CanonicalFrame.GCRF,
    }
    if key in aliases:
        return aliases[key]
    raise FrameTimeError(
        f"Unsupported or ambiguous frame {value!r}; use EME2000, TEME, ITRF, GCRF, "
        "OEL/ECI/J2000, or OEL/ECEF/IAU76_80_EOP."
    )


def state_transform_matrix(
    source_frame: CanonicalFrame | str,
    target_frame: CanonicalFrame | str,
    *,
    context: FrameTransformContext,
) -> np.ndarray:
    source = normalize_canonical_frame(source_frame)
    target = normalize_canonical_frame(target_frame)
    if source is target:
        return np.eye(6, dtype=float)
    if source is CanonicalFrame.GCRF or target is CanonicalFrame.GCRF:
        return _state_matrix_with_gcrf(source, target, context)
    source_to_eme = _direct_state_matrix(source, CanonicalFrame.EME2000, context)
    eme_to_target = _direct_state_matrix(CanonicalFrame.EME2000, target, context)
    return eme_to_target @ source_to_eme


def transform_cartesian_state(
    position_km: Any,
    velocity_km_s: Any,
    source_frame: CanonicalFrame | str,
    target_frame: CanonicalFrame | str,
    *,
    context: FrameTransformContext,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.concatenate((_vector3(position_km, "position_km"), _vector3(velocity_km_s, "velocity_km_s")))
    transformed = state_transform_matrix(source_frame, target_frame, context=context) @ state
    return transformed[:3], transformed[3:]


def transform_covariance(
    covariance: Any,
    source_frame: CanonicalFrame | str,
    target_frame: CanonicalFrame | str,
    *,
    context: FrameTransformContext,
) -> np.ndarray:
    matrix = _covariance_matrix(covariance)
    jacobian = state_transform_matrix(source_frame, target_frame, context=context)
    transformed = jacobian @ matrix @ jacobian.T
    return 0.5 * (transformed + transformed.T)


def frame_transform_receipt(
    source_frame: CanonicalFrame | str,
    target_frame: CanonicalFrame | str,
    *,
    context: FrameTransformContext,
) -> dict[str, Any]:
    source = normalize_canonical_frame(source_frame)
    target = normalize_canonical_frame(target_frame)
    state_transform_matrix(source, target, context=context)
    eop = context.earth_orientation
    return {
        "schema": "oel.frame-transform-receipt.v1",
        "contract": FRAME_TIME_CONTRACT,
        "model": _resolved_transform_model(source, target),
        "source_frame": source.value,
        "target_frame": target.value,
        "epoch_utc": format_epoch(context.epoch, TimeScale.UTC),
        "epoch_tai": format_epoch(context.epoch, TimeScale.TAI),
        "epoch_tt": format_epoch(context.epoch, TimeScale.TT),
        "eop": None
        if eop is None
        else {
            "dut1_s": eop.dut1_s,
            "xp_arcsec": eop.xp_arcsec,
            "yp_arcsec": eop.yp_arcsec,
            "ddpsi_rad": eop.ddpsi_rad,
            "ddeps_rad": eop.ddeps_rad,
            "dx_mas": eop.dx_mas,
            "dy_mas": eop.dy_mas,
            "source": eop.source,
            "source_sha256": eop.source_sha256,
        },
        "leap_seconds": leap_second_table_receipt(),
        "non_claims": [
            "EME2000/ITRF uses OEL's IAU-76/FK5 + IAU-80 EOP reduction, not IAU 2000/2006.",
            "TEME/EME2000 follows the existing Vallado IAU-80 state-vector rotation contract.",
            "GCRF/ITRF uses ERFA's IAU 2006/2000A CIO chain with explicit EOP; GCRF is not treated as EME2000.",
        ],
    }


def _direct_state_matrix(
    source: CanonicalFrame,
    target: CanonicalFrame,
    context: FrameTransformContext,
) -> np.ndarray:
    if source is target:
        return np.eye(6, dtype=float)
    if source is CanonicalFrame.TEME and target is CanonicalFrame.EME2000:
        frame_context = context.orbit_frame_context(require_eop=False)
        rotation = teme_to_eci_matrix_vallado_iau80(
            epoch_julian_date(context.epoch, TimeScale.UTC),
            tt_minus_utc_s=frame_context.tt_minus_utc_s,
            ddpsi_rad=frame_context.ddpsi_rad,
            ddeps_rad=frame_context.ddeps_rad,
        )
        return _block_rotation(rotation)
    if source is CanonicalFrame.EME2000 and target is CanonicalFrame.TEME:
        return np.linalg.inv(_direct_state_matrix(CanonicalFrame.TEME, CanonicalFrame.EME2000, context))
    if source is CanonicalFrame.EME2000 and target is CanonicalFrame.ITRF:
        frame_context = context.orbit_frame_context(require_eop=True)
        rotation = eci_to_ecef_rotation_context(0.0, frame_context)
        rotation_dot = eci_to_ecef_rotation_derivative_context(0.0, frame_context)
        matrix = np.zeros((6, 6), dtype=float)
        matrix[:3, :3] = rotation
        matrix[3:, :3] = rotation_dot
        matrix[3:, 3:] = rotation
        return matrix
    if source is CanonicalFrame.ITRF and target is CanonicalFrame.EME2000:
        return np.linalg.inv(_direct_state_matrix(CanonicalFrame.EME2000, CanonicalFrame.ITRF, context))
    raise FrameTimeError(f"Unsupported direct frame transform {source.value} -> {target.value}.")


def _state_matrix_with_gcrf(
    source: CanonicalFrame,
    target: CanonicalFrame,
    context: FrameTransformContext,
) -> np.ndarray:
    if source is target:
        return np.eye(6, dtype=float)
    if source is CanonicalFrame.GCRF and target is CanonicalFrame.ITRF:
        return _gcrf_to_itrf_state_matrix(context)
    if source is CanonicalFrame.ITRF and target is CanonicalFrame.GCRF:
        return np.linalg.inv(_gcrf_to_itrf_state_matrix(context))
    bias = _block_rotation(_gcrf_to_eme2000_rotation())
    if source is CanonicalFrame.GCRF and target is CanonicalFrame.EME2000:
        return bias
    if source is CanonicalFrame.EME2000 and target is CanonicalFrame.GCRF:
        return bias.T
    if source is CanonicalFrame.GCRF and target is CanonicalFrame.TEME:
        return _direct_state_matrix(CanonicalFrame.EME2000, CanonicalFrame.TEME, context) @ bias
    if source is CanonicalFrame.TEME and target is CanonicalFrame.GCRF:
        return bias.T @ _direct_state_matrix(CanonicalFrame.TEME, CanonicalFrame.EME2000, context)
    raise FrameTimeError(f"Unsupported GCRF frame transform {source.value} -> {target.value}.")


def _gcrf_to_eme2000_rotation() -> np.ndarray:
    frame_bias, _precession, _bias_precession = erfa.bp06(2451545.0, 0.0)
    return np.asarray(frame_bias, dtype=float)


def _gcrf_to_itrf_state_matrix(context: FrameTransformContext) -> np.ndarray:
    if context.earth_orientation is None:
        raise FrameTimeError("GCRF/ITRF transformations require epoch-matched IERS EOP values with provenance.")
    rotation = _gcrf_to_itrf_rotation(context, offset_s=0.0)
    # A 10 s centered stencil suppresses floating-point cancellation in the
    # sub-microradian rotation derivative while remaining negligible against
    # Earth-rotation curvature at the frozen validation envelope.
    step_s = 10.0
    rotation_dot = (
        _gcrf_to_itrf_rotation(context, offset_s=step_s)
        - _gcrf_to_itrf_rotation(context, offset_s=-step_s)
    ) / (2.0 * step_s)
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = rotation
    matrix[3:, :3] = rotation_dot
    matrix[3:, 3:] = rotation
    return matrix


def _gcrf_to_itrf_rotation(context: FrameTransformContext, *, offset_s: float) -> np.ndarray:
    eop = context.earth_orientation
    if eop is None:
        raise FrameTimeError("GCRF/ITRF transformations require epoch-matched IERS EOP values with provenance.")
    shifted = Epoch(
        context.epoch.tai_seconds + float(offset_s),
        context.epoch.source_scale,
        context.epoch.source_text,
    )
    jd_tt = epoch_julian_date(shifted, TimeScale.TT)
    jd_ut1 = epoch_julian_date(shifted, TimeScale.UT1, dut1_s=eop.dut1_s)
    tt1, tt2 = 2400000.5, jd_tt - 2400000.5
    ut11, ut12 = 2400000.5, jd_ut1 - 2400000.5
    arcsec_to_rad = math.pi / (180.0 * 3600.0)
    x, y, _s = erfa.xys06a(tt1, tt2)
    x += float(eop.dx_mas) * arcsec_to_rad / 1000.0
    y += float(eop.dy_mas) * arcsec_to_rad / 1000.0
    s = erfa.s06(tt1, tt2, x, y)
    celestial_to_intermediate = erfa.c2ixys(x, y, s)
    era = erfa.era00(ut11, ut12)
    polar_motion = erfa.pom00(
        float(eop.xp_arcsec) * arcsec_to_rad,
        float(eop.yp_arcsec) * arcsec_to_rad,
        erfa.sp00(tt1, tt2),
    )
    return np.asarray(erfa.c2tcio(celestial_to_intermediate, era, polar_motion), dtype=float)


def _resolved_transform_model(source: CanonicalFrame, target: CanonicalFrame) -> str:
    return FRAME_TRANSFORM_MODEL_IAU2006 if CanonicalFrame.GCRF in {source, target} else FRAME_TRANSFORM_MODEL


def _block_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = rotation
    matrix[3:, 3:] = rotation
    return matrix


def _vector3(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise FrameTimeError(f"{label} must contain three finite values.")
    return array


def _covariance_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise FrameTimeError("Covariance must be a finite 6x6 matrix.")
    scale = float(np.max(np.abs(matrix)))
    tolerance = 64.0 * np.finfo(float).eps * scale * matrix.shape[0]
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise FrameTimeError("Covariance must be symmetric within the v1 numerical tolerance.")
    symmetric = 0.5 * (matrix + matrix.T)
    if np.any(np.diag(symmetric) < 0.0):
        raise FrameTimeError(
            "Covariance must be positive semidefinite; diagonal variances must be non-negative."
        )
    if float(np.min(np.linalg.eigvalsh(symmetric))) < -tolerance:
        raise FrameTimeError("Covariance must be positive semidefinite within the v1 numerical tolerance.")
    return symmetric


def _time_scale(value: TimeScale | str) -> TimeScale:
    if isinstance(value, TimeScale):
        return value
    try:
        return TimeScale(str(value or "").strip().upper())
    except ValueError as exc:
        raise FrameTimeError("Supported time scales are UTC, TAI, TT, and UT1.") from exc


def _parse_components(text: str, *, allow_leap: bool) -> dict[str, Any]:
    raw = str(text or "").strip()
    match = _CALENDAR_EPOCH.fullmatch(raw) or _ORDINAL_EPOCH.fullmatch(raw)
    if match is None:
        raise FrameTimeError(f"Invalid CCSDS calendar or day-of-year epoch {text!r}.")
    values = match.groupdict()
    second = int(values["second"])
    if second > 60 or (second == 60 and not allow_leap):
        raise FrameTimeError("Second must be 00 through 59, except a validated UTC leap second.")
    fraction = values.get("fraction") or ""
    if len(fraction) > 7 and any(char != "0" for char in fraction[7:]):
        raise FrameTimeError("The v1 epoch contract supports precision through microseconds.")
    microsecond = int((fraction[1:7] if fraction else "").ljust(6, "0") or "0")
    try:
        if values.get("doy") is not None:
            doy = int(values["doy"])
            if doy < 1:
                raise ValueError
            date = datetime(int(values["year"]), 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
            if date.year != int(values["year"]):
                raise ValueError
            year, month, day = date.year, date.month, date.day
        else:
            year, month, day = int(values["year"]), int(values["month"]), int(values["day"])
            datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError as exc:
        raise FrameTimeError(f"Invalid calendar date in epoch {text!r}.") from exc
    hour, minute = int(values["hour"]), int(values["minute"])
    if hour > 23 or minute > 59:
        raise FrameTimeError(f"Invalid clock time in epoch {text!r}.")
    return {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "microsecond": microsecond,
        "z": bool(values.get("z")),
    }


def _components_to_calendar_seconds(components: dict[str, Any]) -> float:
    instant = datetime(
        components["year"],
        components["month"],
        components["day"],
        components["hour"],
        components["minute"],
        components["second"],
        components["microsecond"],
        tzinfo=timezone.utc,
    )
    return (instant - _UNIX_EPOCH).total_seconds()


def _utc_components_to_tai(components: dict[str, Any]) -> float:
    table = _leap_table()
    if components["second"] == 60:
        if components["hour"] != 23 or components["minute"] != 59:
            raise FrameTimeError("UTC leap-second notation is valid only at 23:59:60.")
        next_midnight = datetime(
            components["year"], components["month"], components["day"], tzinfo=timezone.utc
        ) + timedelta(days=1)
        matching_index = next(
            (index for index, item in enumerate(table.entries[1:], start=1) if item.effective_utc == next_midnight),
            None,
        )
        if matching_index is None:
            raise FrameTimeError("The supplied date is not a positive leap-second date in the packaged IERS table.")
        previous_offset = table.entries[matching_index - 1].tai_minus_utc_s
        utc_boundary = (next_midnight - _UNIX_EPOCH).total_seconds()
        return utc_boundary + previous_offset + components["microsecond"] / 1.0e6
    utc = datetime(
        components["year"],
        components["month"],
        components["day"],
        components["hour"],
        components["minute"],
        components["second"],
        components["microsecond"],
        tzinfo=timezone.utc,
    )
    if utc < table.valid_from or utc >= table.valid_through_exclusive:
        raise FrameTimeError(
            f"UTC epoch is outside leap-second table coverage "
            f"[{_format_datetime(table.valid_from)}, {_format_datetime(table.valid_through_exclusive)})."
        )
    utc_seconds = (utc - _UNIX_EPOCH).total_seconds()
    return utc_seconds + _utc_offset_for_seconds(utc_seconds)


def _utc_offset_for_seconds(utc_seconds: float) -> int:
    table = _leap_table()
    start = (table.valid_from - _UNIX_EPOCH).total_seconds()
    stop = (table.valid_through_exclusive - _UNIX_EPOCH).total_seconds()
    if utc_seconds < start or utc_seconds >= stop:
        raise FrameTimeError("UTC instant is outside the packaged leap-second table coverage.")
    offset = table.entries[0].tai_minus_utc_s
    for entry in table.entries[1:]:
        if utc_seconds < entry.effective_utc_seconds:
            break
        offset = entry.tai_minus_utc_s
    return offset


def _utc_from_tai(tai_seconds: float) -> tuple[str, float, int, bool]:
    table = _leap_table()
    for previous, current in zip(table.entries, table.entries[1:]):
        leap_start = current.effective_utc_seconds + previous.tai_minus_utc_s
        leap_stop = current.effective_utc_seconds + current.tai_minus_utc_s
        if leap_start <= tai_seconds < leap_stop:
            fraction = tai_seconds - leap_start
            leap_day = current.effective_utc - timedelta(days=1)
            text = f"{leap_day:%Y-%m-%d}T23:59:60" + _fraction_suffix(fraction)
            return text, current.effective_utc_seconds + fraction, previous.tai_minus_utc_s, True
    offset = table.entries[0].tai_minus_utc_s
    for entry in table.entries[1:]:
        if tai_seconds < entry.effective_utc_seconds + entry.tai_minus_utc_s:
            break
        offset = entry.tai_minus_utc_s
    utc_seconds = tai_seconds - offset
    start = (table.valid_from - _UNIX_EPOCH).total_seconds()
    stop = (table.valid_through_exclusive - _UNIX_EPOCH).total_seconds()
    if utc_seconds < start or utc_seconds >= stop:
        raise FrameTimeError("TAI instant cannot be represented inside the packaged UTC leap-second table coverage.")
    return _format_calendar_seconds(utc_seconds), utc_seconds, offset, False


def _format_calendar_seconds(seconds: float) -> str:
    microseconds = int(round(float(seconds) * 1.0e6))
    instant = _UNIX_EPOCH + timedelta(microseconds=microseconds)
    return _format_datetime(instant)


def _format_datetime(value: datetime) -> str:
    base = value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return base + _fraction_suffix(value.microsecond / 1.0e6)


def _fraction_suffix(fraction_s: float) -> str:
    microseconds = int(round(float(fraction_s) * 1.0e6))
    if microseconds == 0:
        return ""
    if microseconds >= 1_000_000:
        raise FrameTimeError("Fractional second rounds outside the represented second.")
    return "." + f"{microseconds:06d}".rstrip("0")


def _require_dut1(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)) or abs(float(value)) >= 2.0:
        raise FrameTimeError("UT1 conversion requires a finite epoch-matched dut1_s with magnitude below 2 seconds.")
    return float(value)


__all__ = [
    "FRAME_TIME_CONTRACT",
    "FRAME_TRANSFORM_MODEL",
    "FRAME_TRANSFORM_MODEL_IAU2006",
    "CanonicalFrame",
    "EarthOrientation",
    "Epoch",
    "FrameTimeError",
    "FrameTransformContext",
    "TimeScale",
    "epoch_conversion_receipt",
    "epoch_julian_date",
    "format_epoch",
    "frame_transform_receipt",
    "leap_second_table_receipt",
    "normalize_canonical_frame",
    "parse_epoch",
    "state_transform_matrix",
    "tai_minus_utc",
    "transform_cartesian_state",
    "transform_covariance",
]
