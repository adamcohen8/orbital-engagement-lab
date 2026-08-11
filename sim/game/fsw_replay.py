"""Deterministic game replay records spanning inputs, FSW, hardware, and observers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from sim.flight_software import (
    ActuatorCommandReceipt,
    FlightSoftwareInputBatch,
    FlightSoftwareOutput,
    FlightSoftwareSnapshot,
    canonical_json_bytes,
)


@dataclass(frozen=True, slots=True)
class GameObserverRecord:
    time_ns: int
    policy_id: str
    values: tuple[tuple[str, float | str | bool | None], ...]


@dataclass(frozen=True, slots=True)
class GameScoringRecord:
    time_ns: int
    event_id: str
    values: tuple[tuple[str, float | str | bool | None], ...] = ()


@dataclass(frozen=True, slots=True)
class GameReplayFrame:
    input_batch: FlightSoftwareInputBatch
    output: FlightSoftwareOutput
    receipts: tuple[ActuatorCommandReceipt, ...] = ()
    realization: tuple[tuple[str, float | str | bool | None], ...] = ()
    observer: GameObserverRecord | None = None
    scoring: tuple[GameScoringRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class GameReplayBundle:
    scenario_id: str
    stack_id: str
    input_profile_id: str
    initial_snapshot: FlightSoftwareSnapshot
    frames: tuple[GameReplayFrame, ...]

    @property
    def content_hash_sha256(self) -> str:
        return sha256(canonical_json_bytes(self)).hexdigest()


def command_stream_bytes(bundle: GameReplayBundle) -> bytes:
    return canonical_json_bytes(tuple(frame.output.commands for frame in bundle.frames))
