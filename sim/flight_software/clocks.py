"""Modeled onboard-clock helpers for flight-software boundary tags."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .contracts import ClockScale, ClockTag, TimeValidity


def clock_tag_elapsed_ns(tag: ClockTag) -> int:
    """Return elapsed nanoseconds within one clock reset domain."""

    return int(tag.ticks) * int(tag.tick_period_ns)


def compare_clock_tags(left: ClockTag, right: ClockTag) -> int:
    """Compare tags only when their domain, resolution, scale, and reset match."""

    identity_left = (left.clock_id, left.tick_period_ns, left.scale, left.reset_counter)
    identity_right = (right.clock_id, right.tick_period_ns, right.scale, right.reset_counter)
    if identity_left != identity_right:
        raise ValueError("clock tags are not directly comparable")
    return (left.ticks > right.ticks) - (left.ticks < right.ticks)


@dataclass(frozen=True, slots=True)
class IdealClockModel:
    clock_id: str
    tick_period_ns: int = 1_000_000
    scale: ClockScale = ClockScale.ONBOARD
    reset_counter: int = 0

    def __post_init__(self) -> None:
        if not str(self.clock_id).strip():
            raise ValueError("clock_id must be non-empty")
        if isinstance(self.tick_period_ns, bool) or not isinstance(self.tick_period_ns, int):
            raise TypeError("tick_period_ns must be an integer")
        if self.tick_period_ns <= 0:
            raise ValueError("tick_period_ns must be positive")
        if self.reset_counter < 0:
            raise ValueError("reset_counter must be nonnegative")
        if not isinstance(self.scale, ClockScale):
            raise TypeError("scale must be ClockScale")

    def tag_from_sim_time_ns(self, sim_time_ns: int, *, validity: TimeValidity = TimeValidity.VALID) -> ClockTag:
        if isinstance(sim_time_ns, bool) or not isinstance(sim_time_ns, int):
            raise TypeError("sim_time_ns must be an integer")
        if sim_time_ns < 0:
            raise ValueError("sim_time_ns must be nonnegative")
        return ClockTag(
            clock_id=self.clock_id,
            ticks=sim_time_ns // self.tick_period_ns,
            tick_period_ns=self.tick_period_ns,
            scale=self.scale,
            validity=validity,
            reset_counter=self.reset_counter,
        )

    def sim_time_ns_from_tag(self, tag: ClockTag) -> int:
        self._validate_tag_identity(tag)
        return clock_tag_elapsed_ns(tag)

    def _validate_tag_identity(self, tag: ClockTag) -> None:
        if (
            tag.clock_id != self.clock_id
            or tag.tick_period_ns != self.tick_period_ns
            or tag.scale != self.scale
            or tag.reset_counter != self.reset_counter
        ):
            raise ValueError("clock tag does not belong to this clock model")


@dataclass(frozen=True, slots=True)
class AffineClockModel:
    """Deterministic clock with bias, fractional drift, quantization, and reset."""

    clock_id: str
    tick_period_ns: int = 1_000_000
    bias_ns: int = 0
    drift_fraction: float = 0.0
    scale: ClockScale = ClockScale.ONBOARD
    reset_counter: int = 0

    def __post_init__(self) -> None:
        if not str(self.clock_id).strip():
            raise ValueError("clock_id must be non-empty")
        if isinstance(self.tick_period_ns, bool) or not isinstance(self.tick_period_ns, int):
            raise TypeError("tick_period_ns must be an integer")
        if self.tick_period_ns <= 0:
            raise ValueError("tick_period_ns must be positive")
        if isinstance(self.bias_ns, bool) or not isinstance(self.bias_ns, int):
            raise TypeError("bias_ns must be an integer")
        if not isfinite(float(self.drift_fraction)) or self.drift_fraction <= -1.0:
            raise ValueError("drift_fraction must be finite and greater than -1")
        if self.reset_counter < 0:
            raise ValueError("reset_counter must be nonnegative")
        if not isinstance(self.scale, ClockScale):
            raise TypeError("scale must be ClockScale")

    def tag_from_sim_time_ns(self, sim_time_ns: int, *, validity: TimeValidity = TimeValidity.VALID) -> ClockTag:
        if isinstance(sim_time_ns, bool) or not isinstance(sim_time_ns, int):
            raise TypeError("sim_time_ns must be an integer")
        if sim_time_ns < 0:
            raise ValueError("sim_time_ns must be nonnegative")
        modeled_ns = self.bias_ns + int(round(sim_time_ns * (1.0 + float(self.drift_fraction))))
        if modeled_ns < 0:
            return ClockTag(
                clock_id=self.clock_id,
                ticks=0,
                tick_period_ns=self.tick_period_ns,
                scale=self.scale,
                validity=TimeValidity.INVALID,
                reset_counter=self.reset_counter,
            )
        return ClockTag(
            clock_id=self.clock_id,
            ticks=modeled_ns // self.tick_period_ns,
            tick_period_ns=self.tick_period_ns,
            scale=self.scale,
            validity=validity,
            reset_counter=self.reset_counter,
        )

    def sim_time_ns_from_tag(self, tag: ClockTag) -> int:
        if (
            tag.clock_id != self.clock_id
            or tag.tick_period_ns != self.tick_period_ns
            or tag.scale != self.scale
            or tag.reset_counter != self.reset_counter
        ):
            raise ValueError("clock tag does not belong to this clock model")
        modeled_ns = clock_tag_elapsed_ns(tag)
        return int(round((modeled_ns - self.bias_ns) / (1.0 + float(self.drift_fraction))))
