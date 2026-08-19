from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any

import numpy as np

from sim.api import SimulationConfig, SimulationSession
from sim.game.training import RPOTrainingConfig, RPOTrainingScore

_RPO_TRAINING_SCORE_FIELD_NAMES = tuple(item.name for item in fields(RPOTrainingScore))


class GamePhysicsSession:
    """Game-facing OEL physics session with lightweight history retention."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        retained_history_samples: int = 4096,
    ):
        retained = int(max(2, retained_history_samples))
        self._session = SimulationSession.from_config(
            config,
            history_mode="dynamic",
            initial_history_capacity=retained,
            max_history_samples=retained,
        )
        game = dict(config.scenario.metadata.get("game", {}) or {})
        self._observer_policy = str(game.get("observer_policy", "truth_assisted") or "truth_assisted")
        self._scoring_policy = str(game.get("scoring_policy", "configured_training.v1") or "")
        self._controlled_object_id = str(game.get("controlled_object_id", "chaser") or "chaser")
        self._observer_samples: list[dict[str, Any]] = []
        self._scoring_events: list[dict[str, Any]] = []

    @property
    def config(self) -> SimulationConfig:
        return self._session.config

    @property
    def done(self) -> bool:
        return self._session.done

    @property
    def _engine(self) -> Any | None:
        return self._session._engine

    def reset(self, seed: int | None = None) -> Any:
        self._observer_samples.clear()
        self._scoring_events.clear()
        snapshot = self._session.reset(seed=seed)
        self._record_observer(snapshot)
        return snapshot

    def step(self, dt_s: float | None = None) -> Any:
        snapshot = self._session.step(dt_s=dt_s)
        self._record_observer(snapshot)
        return snapshot

    def publish_fsw_input(self, object_id: str, event: object) -> None:
        self._session.publish_fsw_input(object_id, event)

    def add_fsw_input_publisher(self, object_id: str, publisher: Any) -> None:
        self._session.add_fsw_input_publisher(object_id, publisher)

    def request_fsw_input_publisher_poll(self, object_id: str) -> None:
        self._session.request_fsw_input_publisher_poll(object_id)

    def record_scoring(self, score: object) -> None:
        """Record truth-derived scoring outside the onboard input boundary."""

        if not self._scoring_policy:
            return
        if type(score) is RPOTrainingScore:
            # The frozen score contract contains only primitives and immutable
            # tuples. Avoid dataclasses.asdict's recursive deepcopy on every
            # gameplay tick while preserving the exact evidence mapping.
            values = {
                name: getattr(score, name)
                for name in _RPO_TRAINING_SCORE_FIELD_NAMES
            }
        elif is_dataclass(score):
            values = asdict(score)
        elif hasattr(score, "__dict__"):
            values = dict(vars(score))
        else:
            values = {"value": str(score)}
        event_type = (
            "passed"
            if bool(values.get("level_passed", False))
            else "failed"
            if bool(values.get("level_failed", False))
            else "sample"
        )
        time_s = float(self._observer_samples[-1]["time_ns"]) / 1.0e9 if self._observer_samples else 0.0
        self._scoring_events.append(
            {
                "object_id": self._controlled_object_id,
                "time_ns": int(round(time_s * 1.0e9)),
                "scoring_policy": self._scoring_policy,
                "event_type": event_type,
                "detail": values,
            }
        )

    def game_review_evidence(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "game_observer_samples": list(self._observer_samples),
            "game_scoring_events": list(self._scoring_events),
        }

    def _record_observer(self, snapshot: Any | None) -> None:
        if snapshot is None:
            return
        truth_assisted = self._observer_policy in {"truth_assisted", "hybrid"}
        onboard = self._observer_policy in {"onboard_only", "hybrid"}
        detail: dict[str, Any] = {}
        if truth_assisted:
            detail["truth"] = {
                str(object_id): np.asarray(state, dtype=float).tolist()
                for object_id, state in sorted(snapshot.truth.items())
            }
        if onboard:
            detail["onboard"] = {
                str(object_id): np.asarray(state, dtype=float).tolist()
                for object_id, state in sorted(snapshot.belief.items())
            }
        self._observer_samples.append(
            {
                "object_id": self._controlled_object_id,
                "time_ns": int(round(float(snapshot.time_s) * 1.0e9)),
                "observer_policy": self._observer_policy,
                "truth_assisted": truth_assisted,
                "detail": detail,
            }
        )

def _attempt_config_for_training_clock(config: SimulationConfig, training_cfg: RPOTrainingConfig) -> SimulationConfig:
    if training_cfg.max_time_s is None:
        return config
    dt_s = float(max(config.scenario.simulator.dt_s, 1.0e-9))
    duration_s = np.ceil(max(float(training_cfg.max_time_s), dt_s) / dt_s) * dt_s
    return config.with_value("simulator.duration_s", duration_s)


def _install_chaser_delta_v_limiter(
    session: SimulationSession,
    *,
    training_cfg: RPOTrainingConfig,
    dt_s: float,
) -> None:
    if not bool(getattr(training_cfg, "coast_chaser_after_delta_v_budget", False)) or training_cfg.max_delta_v_m_s is None:
        return
    engine = getattr(session, "_engine", None)
    agent = getattr(engine, "agents", {}).get(str(training_cfg.chaser_object_id)) if engine is not None else None
    if agent is None:
        return
    runtime = getattr(agent, "flight_software_runtime", None)
    if runtime is None:
        return
    runtime.max_delta_v_m_s = float(training_cfg.max_delta_v_m_s)


def _set_chaser_delta_v_limiter_dt(
    session: SimulationSession,
    *,
    training_cfg: RPOTrainingConfig,
    dt_s: float,
) -> None:
    engine = getattr(session, "_engine", None)
    agent = getattr(engine, "agents", {}).get(str(training_cfg.chaser_object_id)) if engine is not None else None
    if agent is None:
        return
    # Physical delta-v accounting integrates the actual interval duration, so
    # changing the game step needs no controller-side retiming.
