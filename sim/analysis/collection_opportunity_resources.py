"""Bounded storage and retained-downlink screening for collection opportunities."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object.")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(unknown)}.")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean.")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True)
class DownlinkWindowInput:
    window_id: str
    source_product_sha256: str
    start_s: float
    end_s: float
    delivered_data_bytes: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DownlinkWindowInput:
        raw = _mapping(data, "resources.downlink_windows[]")
        _reject_unknown(
            raw,
            {"window_id", "source_product_sha256", "start_s", "end_s", "delivered_data_bytes"},
            "resources.downlink_windows[]",
        )
        item = cls(
            window_id=_required_text(raw.get("window_id"), "resources.downlink_windows[].window_id"),
            source_product_sha256=_required_text(
                raw.get("source_product_sha256"), "resources.downlink_windows[].source_product_sha256"
            ).lower(),
            start_s=_finite_number(raw.get("start_s"), "resources.downlink_windows[].start_s"),
            end_s=_finite_number(raw.get("end_s"), "resources.downlink_windows[].end_s"),
            delivered_data_bytes=_finite_number(
                raw.get("delivered_data_bytes"), "resources.downlink_windows[].delivered_data_bytes"
            ),
        )
        if len(item.source_product_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in item.source_product_sha256
        ):
            raise ValueError("Every downlink window requires a lowercase source_product_sha256.")
        if item.end_s <= item.start_s or item.delivered_data_bytes < 0.0:
            raise ValueError("Downlink windows require end_s > start_s and nonnegative delivered data.")
        return item


@dataclass(frozen=True)
class CollectionResources:
    enabled: bool
    storage_capacity_bytes: float
    initial_storage_bytes: float
    require_downlink_by_horizon: bool
    downlink_windows: tuple[DownlinkWindowInput, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> CollectionResources:
        raw = {} if data is None else _mapping(data, "resources")
        _reject_unknown(
            raw,
            {
                "enabled",
                "storage_capacity_bytes",
                "initial_storage_bytes",
                "require_downlink_by_horizon",
                "downlink_windows",
            },
            "resources",
        )
        enabled = _boolean(raw.get("enabled", False), "resources.enabled")
        capacity = _finite_number(raw.get("storage_capacity_bytes", 0.0), "resources.storage_capacity_bytes")
        initial = _finite_number(raw.get("initial_storage_bytes", 0.0), "resources.initial_storage_bytes")
        if capacity < 0.0 or initial < 0.0:
            raise ValueError("Storage capacity and initial storage must be nonnegative.")
        if not 0.0 <= initial <= capacity:
            raise ValueError("initial_storage_bytes must lie within storage capacity.")
        windows_raw = raw.get("downlink_windows", [])
        if not isinstance(windows_raw, list):
            raise ValueError("resources.downlink_windows must be a JSON array.")
        windows = tuple(DownlinkWindowInput.from_mapping(item) for item in windows_raw)
        identifiers = [item.window_id for item in windows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Downlink window identifiers must be unique.")
        semantic_windows = [
            (item.source_product_sha256, item.start_s, item.end_s, item.delivered_data_bytes)
            for item in windows
        ]
        if len(set(semantic_windows)) != len(semantic_windows):
            raise ValueError("Downlink windows must not contain semantic duplicates.")
        ordered = sorted(windows, key=lambda item: (item.start_s, item.end_s, item.window_id))
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            if right.start_s < left.end_s - 1.0e-12:
                raise ValueError("Downlink windows must not overlap in the v1 single-stream resource screen.")
        require_downlink = _boolean(
            raw.get("require_downlink_by_horizon", False), "resources.require_downlink_by_horizon"
        )
        if not enabled and (windows or require_downlink):
            raise ValueError("Enable resource screening before declaring downlink requirements or windows.")
        if enabled and capacity <= 0.0:
            raise ValueError("Enabled resource screening requires positive storage_capacity_bytes.")
        return cls(
            enabled=enabled,
            storage_capacity_bytes=capacity,
            initial_storage_bytes=initial,
            require_downlink_by_horizon=require_downlink,
            downlink_windows=windows,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "storage_capacity_bytes": self.storage_capacity_bytes,
            "initial_storage_bytes": self.initial_storage_bytes,
            "require_downlink_by_horizon": self.require_downlink_by_horizon,
            "downlink_windows": [asdict(item) for item in self.downlink_windows],
        }


def screen_collection_resources(
    resources: CollectionResources, *, collection_end_s: float, generated_data_bytes: float
) -> dict[str, Any]:
    if not math.isfinite(generated_data_bytes) or generated_data_bytes < 0.0:
        raise ValueError("generated_data_bytes must be finite and nonnegative.")
    if not resources.enabled:
        return {
            "enabled": False,
            "resource_feasible": True,
            "reason": "not_screened",
            "qualification": "Storage/downlink feasibility was not requested.",
        }
    storage_after = resources.initial_storage_bytes + generated_data_bytes
    storage_feasible = storage_after <= resources.storage_capacity_bytes + 1.0e-9
    later_windows = [item for item in resources.downlink_windows if item.start_s + 1.0e-12 >= collection_end_s]
    delivered = sum(item.delivered_data_bytes for item in later_windows)
    downlink_feasible = (not resources.require_downlink_by_horizon) or delivered + 1.0e-9 >= generated_data_bytes
    reason = "available" if storage_feasible and downlink_feasible else "storage_exceeded" if not storage_feasible else "downlink_insufficient"
    return {
        "enabled": True,
        "resource_feasible": storage_feasible and downlink_feasible,
        "reason": reason,
        "storage_capacity_bytes": resources.storage_capacity_bytes,
        "initial_storage_bytes": resources.initial_storage_bytes,
        "storage_after_collection_bytes": storage_after,
        "storage_headroom_bytes": resources.storage_capacity_bytes - storage_after,
        "require_downlink_by_horizon": resources.require_downlink_by_horizon,
        "eligible_downlink_window_ids": [item.window_id for item in later_windows],
        "eligible_downlink_sources": [
            {
                "window_id": item.window_id,
                "source_product_sha256": item.source_product_sha256,
                "start_s": item.start_s,
                "end_s": item.end_s,
                "delivered_data_bytes": item.delivered_data_bytes,
            }
            for item in later_windows
        ],
        "eligible_delivered_data_bytes": delivered,
        "observation_data_downlinked_by_horizon": downlink_feasible,
    }


__all__ = ["CollectionResources", "DownlinkWindowInput", "screen_collection_resources"]
