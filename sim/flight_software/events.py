"""Deterministic event ordering for the v2 satellite boundary."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable


class SatelliteEventPhase(IntEnum):
    PHYSICS_ADVANCE = 0
    PHYSICAL_COMMIT = 1
    SAMPLE_AND_STATUS = 2
    INPUT_DELIVERY = 3
    TASK_RELEASE = 4
    COMMAND_PUBLICATION = 5
    COMMAND_EFFECTIVE = 6
    EVIDENCE = 7
    OUTPUT_SAMPLE = 8


@dataclass(frozen=True, slots=True)
class SatelliteEvent:
    time_ns: int
    phase: SatelliteEventPhase
    event_id: str
    payload: object = None
    order: int = 0
    source_id: str = ""
    sequence: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.time_ns, bool) or not isinstance(self.time_ns, int) or self.time_ns < 0:
            raise ValueError("time_ns must be a nonnegative integer")
        if not isinstance(self.phase, SatelliteEventPhase):
            raise TypeError("phase must be SatelliteEventPhase")
        if not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if self.order < 0 or self.sequence < 0:
            raise ValueError("order and sequence must be nonnegative")


@dataclass(frozen=True, slots=True)
class EventTraceEntry:
    time_ns: int
    phase: SatelliteEventPhase
    event_id: str


@dataclass(order=True, slots=True)
class _QueuedEvent:
    sort_key: tuple[int, int, int, str, int, int]
    event: SatelliteEvent = field(compare=False)


class SatelliteEventKernel:
    """Run exact event boundaries without allowing retroactive same-time work."""

    def __init__(self, *, start_time_ns: int = 0) -> None:
        if start_time_ns < 0:
            raise ValueError("start_time_ns must be nonnegative")
        self.current_time_ns = int(start_time_ns)
        self._queue: list[_QueuedEvent] = []
        self._insertion = 0
        self._active_phase: SatelliteEventPhase | None = None

    def schedule(self, event: SatelliteEvent) -> None:
        if event.time_ns < self.current_time_ns:
            raise ValueError("cannot schedule an event in the processed past")
        if (
            event.time_ns == self.current_time_ns
            and self._active_phase is not None
            and event.phase < self._active_phase
        ):
            raise ValueError("cannot schedule a same-time event into an already processed phase")
        key = (
            event.time_ns,
            int(event.phase),
            event.order,
            event.source_id,
            event.sequence,
            self._insertion,
        )
        self._insertion += 1
        heapq.heappush(self._queue, _QueuedEvent(key, event))

    @property
    def next_event_time_ns(self) -> int | None:
        return None if not self._queue else self._queue[0].event.time_ns

    def run_until(
        self,
        end_time_ns: int,
        *,
        advance_physics: Callable[[int, int], None],
        handle_event: Callable[[SatelliteEvent], None],
    ) -> tuple[EventTraceEntry, ...]:
        if end_time_ns < self.current_time_ns:
            raise ValueError("end_time_ns must not precede current time")
        trace: list[EventTraceEntry] = []
        while self._queue and self._queue[0].event.time_ns <= end_time_ns:
            event_time = self._queue[0].event.time_ns
            if event_time > self.current_time_ns:
                advance_physics(self.current_time_ns, event_time)
                trace.append(EventTraceEntry(event_time, SatelliteEventPhase.PHYSICS_ADVANCE, "physics.advance"))
                self.current_time_ns = event_time
            trace.append(EventTraceEntry(event_time, SatelliteEventPhase.PHYSICAL_COMMIT, "physics.commit"))
            while self._queue and self._queue[0].event.time_ns == event_time:
                queued = heapq.heappop(self._queue)
                self._active_phase = queued.event.phase
                handle_event(queued.event)
                trace.append(EventTraceEntry(event_time, queued.event.phase, queued.event.event_id))
            self._active_phase = None
        if end_time_ns > self.current_time_ns:
            advance_physics(self.current_time_ns, end_time_ns)
            trace.append(EventTraceEntry(end_time_ns, SatelliteEventPhase.PHYSICS_ADVANCE, "physics.advance"))
            self.current_time_ns = end_time_ns
        return tuple(trace)


@dataclass(frozen=True, slots=True)
class PeriodicSchedule:
    schedule_id: str
    period_ns: int
    first_release_ns: int = 0
    order: int = 0

    def __post_init__(self) -> None:
        if not self.schedule_id.strip():
            raise ValueError("schedule_id must be non-empty")
        if self.period_ns <= 0 or self.first_release_ns < 0 or self.order < 0:
            raise ValueError("period must be positive and release/order values nonnegative")

    def releases_through(self, end_time_ns: int) -> tuple[int, ...]:
        if end_time_ns < self.first_release_ns:
            return ()
        count = (end_time_ns - self.first_release_ns) // self.period_ns
        return tuple(self.first_release_ns + index * self.period_ns for index in range(count + 1))
