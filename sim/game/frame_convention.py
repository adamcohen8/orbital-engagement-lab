from __future__ import annotations

from dataclasses import dataclass
from typing import Any

IN_TRACK_DIRECTIONS: tuple[str, ...] = ("right", "left")
CROSS_TRACK_SENSES: tuple[str, ...] = ("clockwise", "counter_clockwise")
FRAME_CONVENTION_PRESET_OEL_DEFAULT = "oel_default"
FRAME_CONVENTION_PRESET_SPACE_FORCE = "space_force"


@dataclass(frozen=True)
class FrameConvention:
    positive_in_track: str = "right"
    positive_cross_track: str = "counter_clockwise"


OEL_DEFAULT_FRAME_CONVENTION = FrameConvention(
    positive_in_track="right",
    positive_cross_track="counter_clockwise",
)
SPACE_FORCE_FRAME_CONVENTION = FrameConvention(
    positive_in_track="left",
    positive_cross_track="clockwise",
)


def normalize_frame_convention(value: Any = None) -> FrameConvention:
    if isinstance(value, FrameConvention):
        return frame_convention_from_preset(_frame_convention_preset_from_axes(value))
    raw = dict(value or {}) if isinstance(value, dict) else {}
    preset = _normalize_frame_convention_preset(raw.get("preset"))
    if preset == FRAME_CONVENTION_PRESET_SPACE_FORCE:
        return SPACE_FORCE_FRAME_CONVENTION
    if preset == FRAME_CONVENTION_PRESET_OEL_DEFAULT:
        return OEL_DEFAULT_FRAME_CONVENTION
    axes = FrameConvention(
        positive_in_track=_normalize_in_track_direction(raw.get("positive_in_track")),
        positive_cross_track=_normalize_cross_track_sense(raw.get("positive_cross_track")),
    )
    return frame_convention_from_preset(_frame_convention_preset_from_axes(axes))


def frame_convention_to_yaml(convention: FrameConvention) -> dict[str, str]:
    normalized = normalize_frame_convention(convention)
    return {
        "preset": frame_convention_preset(normalized),
        "positive_in_track": normalized.positive_in_track,
        "positive_cross_track": normalized.positive_cross_track,
    }


def frame_convention_preset(convention: FrameConvention) -> str:
    return _frame_convention_preset_from_axes(normalize_frame_convention(convention))


def frame_convention_from_preset(preset: str) -> FrameConvention:
    normalized = _normalize_frame_convention_preset(preset)
    if normalized == FRAME_CONVENTION_PRESET_SPACE_FORCE:
        return SPACE_FORCE_FRAME_CONVENTION
    return OEL_DEFAULT_FRAME_CONVENTION


def frame_convention_display_axis_sign(convention: FrameConvention | dict[str, Any] | None, axis: int) -> float:
    normalized = normalize_frame_convention(convention)
    if int(axis) == 1 and frame_convention_preset(normalized) == FRAME_CONVENTION_PRESET_SPACE_FORCE:
        return -1.0
    return 1.0


def _frame_convention_preset_from_axes(convention: FrameConvention) -> str:
    if convention.positive_in_track == "left":
        return FRAME_CONVENTION_PRESET_SPACE_FORCE
    return FRAME_CONVENTION_PRESET_OEL_DEFAULT


def _normalize_in_track_direction(value: Any) -> str:
    key = str(value or "right").strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"left", "to_left", "positive_left"}:
        return "left"
    return "right"


def _normalize_cross_track_sense(value: Any) -> str:
    key = str(value or "counter_clockwise").strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"counter_clockwise", "counterclockwise", "ccw"}:
        return "counter_clockwise"
    return "clockwise"


def _normalize_frame_convention_preset(value: Any) -> str | None:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"space_force", "spaceforce", "sf"}:
        return FRAME_CONVENTION_PRESET_SPACE_FORCE
    if key in {"oel_default", "oel", "default", "classic"}:
        return FRAME_CONVENTION_PRESET_OEL_DEFAULT
    return None
