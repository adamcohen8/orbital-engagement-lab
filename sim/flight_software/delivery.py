"""Independent deterministic transport and input-delivery queues."""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable

from .contracts import DataValidity, InputEvent, InputKind, Quality
from .schemas import from_primitive, to_primitive


class TransportDisposition(str, Enum):
    DELIVERED = "delivered"
    LOST = "lost"
    DELIVERED_DETECTABLY_CORRUPT = "delivered_detectably_corrupt"
    DELIVERED_UNDETECTABLY_CORRUPT = "delivered_undetectably_corrupt"


@dataclass(frozen=True, slots=True)
class TransportRecord:
    packet_identity: tuple[str, str, int]
    disposition: TransportDisposition


class PacketTransport:
    """Seeded packet-local loss/corruption model with separate review evidence."""

    def __init__(
        self,
        *,
        seed: int = 0,
        loss_probability: float = 0.0,
        detectable_corruption_probability: float = 0.0,
        undetectable_corruption_probability: float = 0.0,
        corrupt_payload: Callable[[object], object] | None = None,
    ) -> None:
        probabilities = (loss_probability, detectable_corruption_probability, undetectable_corruption_probability)
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities) or sum(probabilities) > 1.0:
            raise ValueError("transport probabilities must be in [0, 1] and sum to at most 1")
        if sum(probabilities[1:]) > 0.0 and corrupt_payload is None:
            raise ValueError("a payload corruption function is required when corruption is enabled")
        self._rng = random.Random(seed)
        self._loss = float(loss_probability)
        self._detectable = float(detectable_corruption_probability)
        self._undetectable = float(undetectable_corruption_probability)
        self._corrupt_payload = corrupt_payload

    def transmit(self, event: InputEvent) -> tuple[InputEvent | None, TransportRecord]:
        identity = (event.packet_id.source_id, event.packet_id.boot_id, event.packet_id.sequence)
        draw = self._rng.random()
        if draw < self._loss:
            return None, TransportRecord(identity, TransportDisposition.LOST)
        if draw < self._loss + self._detectable:
            payload = self._corrupt_payload(event.payload)  # type: ignore[misc]
            quality = Quality(DataValidity.SUSPECT, event.quality.status_codes + ("transport_corruption_detected",))
            delivered = replace(event, payload=payload, quality=quality)
            return delivered, TransportRecord(identity, TransportDisposition.DELIVERED_DETECTABLY_CORRUPT)
        if draw < self._loss + self._detectable + self._undetectable:
            payload = self._corrupt_payload(event.payload)  # type: ignore[misc]
            return replace(event, payload=payload), TransportRecord(
                identity, TransportDisposition.DELIVERED_UNDETECTABLY_CORRUPT
            )
        return event, TransportRecord(identity, TransportDisposition.DELIVERED)


@dataclass(order=True, slots=True)
class _QueuedInput:
    key: tuple[int, int, str, str, int, int]
    event: InputEvent = field(compare=False)


class InputDeliveryQueue:
    """Resolve equal-time transport order without coupling independent packets."""

    def __init__(self) -> None:
        self._queue: list[_QueuedInput] = []
        self._insertion = 0

    def enqueue(self, event: InputEvent, *, transport_order: int | None = None) -> None:
        delivery_ns = event.delivery_time.ticks * event.delivery_time.tick_period_ns
        explicit_order = transport_order if transport_order is not None else 2**63 - 1
        key = (
            delivery_ns,
            explicit_order,
            event.packet_id.source_id,
            event.packet_id.boot_id,
            event.packet_id.sequence,
            self._insertion,
        )
        self._insertion += 1
        heapq.heappush(self._queue, _QueuedInput(key, event))

    def deliver_due(self, time_ns: int) -> tuple[InputEvent, ...]:
        delivered: list[InputEvent] = []
        while self._queue and self._queue[0].key[0] <= time_ns:
            delivered.append(heapq.heappop(self._queue).event)
        return tuple(delivered)

    @property
    def next_delivery_time_ns(self) -> int | None:
        return None if not self._queue else self._queue[0].key[0]

    def next_delivery_time_ns_for(self, kinds: frozenset[InputKind]) -> int | None:
        """Return the earliest queued delivery that should release a task."""

        matches = (item.key[0] for item in self._queue if item.event.kind in kinds)
        return min(matches, default=None)

    def snapshot_state(self) -> dict[str, object]:
        return {
            "insertion": self._insertion,
            "queued": [
                {"key": list(item.key), "event": to_primitive(item.event)}
                for item in sorted(self._queue)
            ],
        }

    def restore_state(self, state: object) -> None:
        if self._queue or self._insertion:
            raise RuntimeError("input delivery restore requires a fresh queue")
        if not isinstance(state, dict) or set(state) != {"insertion", "queued"}:
            raise ValueError("input delivery checkpoint is invalid")
        insertion = state.get("insertion")
        queued = state.get("queued")
        if isinstance(insertion, bool) or not isinstance(insertion, int) or insertion < 0:
            raise ValueError("input delivery insertion counter is invalid")
        if not isinstance(queued, (list, tuple)):
            raise ValueError("input delivery queued state must be a sequence")
        restored: list[_QueuedInput] = []
        for item in queued:
            if not isinstance(item, dict):
                raise ValueError("input delivery queued item is invalid")
            key_raw = item.get("key")
            if not isinstance(key_raw, (list, tuple)) or len(key_raw) != 6:
                raise ValueError("input delivery ordering key is invalid")
            key = (
                int(key_raw[0]),
                int(key_raw[1]),
                str(key_raw[2]),
                str(key_raw[3]),
                int(key_raw[4]),
                int(key_raw[5]),
            )
            restored.append(_QueuedInput(key, from_primitive(InputEvent, item.get("event"))))
        self._queue = restored
        heapq.heapify(self._queue)
        self._insertion = insertion
