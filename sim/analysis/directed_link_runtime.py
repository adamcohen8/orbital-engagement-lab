"""Authorized causal task-boundary adapter for one directed-link monitor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim.analysis.directed_link import LINK_REASON_NAMES

RUNTIME_LINK_MONITOR_CONTRACT_VERSION = "oel.directed-link-runtime-monitor.v0.2"


@dataclass(frozen=True)
class RuntimeLinkMonitorConfig:
    monitor_id: str
    link_id: str
    authorized_consumer_id: str
    link_config_semantic_sha256: str
    task_period_s: float
    start_time_s: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("monitor_id", "link_id", "authorized_consumer_id"):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, value)
        digest = str(self.link_config_semantic_sha256 or "").strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("link_config_semantic_sha256 must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "link_config_semantic_sha256", digest)
        period = float(self.task_period_s)
        start = float(self.start_time_s)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("task_period_s must be positive and finite.")
        if not np.isfinite(start):
            raise ValueError("start_time_s must be finite.")
        object.__setattr__(self, "task_period_s", period)
        object.__setattr__(self, "start_time_s", start)


@dataclass(frozen=True)
class RuntimeLinkEvaluation:
    available: bool
    margin_db: float
    primary_reason: str
    link_config_semantic_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.available, (bool, np.bool_)):
            raise ValueError("available must be boolean.")
        object.__setattr__(self, "available", bool(self.available))
        margin = float(self.margin_db)
        reason = str(self.primary_reason or "").strip()
        config_digest = str(self.link_config_semantic_sha256 or "").strip().lower()
        digest = str(self.evidence_sha256 or "").strip().lower()
        if not np.isfinite(margin):
            raise ValueError("margin_db must be finite.")
        if reason not in LINK_REASON_NAMES:
            raise ValueError("primary_reason must be a Directed Link Analysis reason.")
        if self.available != (reason == "available"):
            raise ValueError("available and primary_reason are inconsistent.")
        if len(config_digest) != 64 or any(
            character not in "0123456789abcdef" for character in config_digest
        ):
            raise ValueError("link_config_semantic_sha256 must be a lowercase SHA-256 digest.")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "margin_db", margin)
        object.__setattr__(self, "primary_reason", reason)
        object.__setattr__(self, "link_config_semantic_sha256", config_digest)
        object.__setattr__(self, "evidence_sha256", digest)


@dataclass(frozen=True)
class RuntimeLinkEvent:
    monitor_id: str
    link_id: str
    authorized_consumer_id: str
    sequence: int
    evaluated_time_s: float
    eligible_delivery_time_s: float
    available: bool
    margin_db: float
    primary_reason: str
    link_config_semantic_sha256: str
    evidence_sha256: str


RuntimeLinkEvaluator = Callable[[float], RuntimeLinkEvaluation]


class AuthorizedLinkRuntimeMonitor:
    """Evaluate after state commit and expose results only at the next boundary."""

    def __init__(self, config: RuntimeLinkMonitorConfig) -> None:
        if not isinstance(config, RuntimeLinkMonitorConfig):
            raise ValueError("config must be a validated RuntimeLinkMonitorConfig.")
        self.config = config
        self._next_evaluation_time_s = config.start_time_s
        self._sequence = 0
        self._pending: list[RuntimeLinkEvent] = []

    @property
    def next_evaluation_time_s(self) -> float:
        return self._next_evaluation_time_s

    def evaluate_after_state_commit(
        self,
        time_s: float,
        evaluator: RuntimeLinkEvaluator,
    ) -> RuntimeLinkEvent | None:
        current = float(time_s)
        if not np.isfinite(current):
            raise ValueError("time_s must be finite.")
        tolerance = 1.0e-12
        if current + tolerance < self._next_evaluation_time_s:
            return None
        if abs(current - self._next_evaluation_time_s) > tolerance:
            raise ValueError(
                "Runtime link monitor must be evaluated at its exact declared task boundary."
            )
        evaluation = evaluator(current)
        if not isinstance(evaluation, RuntimeLinkEvaluation):
            raise ValueError("evaluator must return RuntimeLinkEvaluation.")
        if evaluation.link_config_semantic_sha256 != self.config.link_config_semantic_sha256:
            raise ValueError("Runtime link evidence is not bound to the monitor's link configuration.")
        delivery_time = current + self.config.task_period_s
        event = RuntimeLinkEvent(
            monitor_id=self.config.monitor_id,
            link_id=self.config.link_id,
            authorized_consumer_id=self.config.authorized_consumer_id,
            sequence=self._sequence,
            evaluated_time_s=current,
            eligible_delivery_time_s=delivery_time,
            available=bool(evaluation.available),
            margin_db=evaluation.margin_db,
            primary_reason=evaluation.primary_reason,
            link_config_semantic_sha256=self.config.link_config_semantic_sha256,
            evidence_sha256=evaluation.evidence_sha256,
        )
        self._pending.append(event)
        self._sequence += 1
        self._next_evaluation_time_s = delivery_time
        return event

    def deliver_due(self, time_s: float, *, consumer_id: str) -> tuple[RuntimeLinkEvent, ...]:
        consumer = str(consumer_id or "").strip()
        if consumer != self.config.authorized_consumer_id:
            raise PermissionError("Runtime link events may be delivered only to the authorized consumer.")
        current = float(time_s)
        if not np.isfinite(current):
            raise ValueError("time_s must be finite.")
        due = tuple(
            event for event in self._pending if event.eligible_delivery_time_s <= current + 1.0e-12
        )
        self._pending = [event for event in self._pending if event not in due]
        return due


__all__ = [
    "RUNTIME_LINK_MONITOR_CONTRACT_VERSION",
    "AuthorizedLinkRuntimeMonitor",
    "RuntimeLinkEvaluation",
    "RuntimeLinkEvaluator",
    "RuntimeLinkEvent",
    "RuntimeLinkMonitorConfig",
]
