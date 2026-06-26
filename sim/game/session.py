from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.api import SimulationConfig, SimulationSession
from sim.core.models import Command, StateBelief
from sim.game.training import RPOTrainingConfig


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

    @property
    def config(self) -> SimulationConfig:
        return self._session.config

    @property
    def done(self) -> bool:
        return self._session.done

    @property
    def _engine(self) -> Any | None:
        return self._session._engine

    @property
    def _external_intent_providers(self) -> dict[str, Any]:
        return self._session._external_intent_providers

    def reset(self, seed: int | None = None) -> Any:
        return self._session.reset(seed=seed)

    def step(self, dt_s: float | None = None) -> Any:
        return self._session.step(dt_s=dt_s)

    def set_external_intent_provider(self, object_id: str, provider: Any | None) -> None:
        self._session.set_external_intent_provider(object_id, provider)


@dataclass
class _DeltaVLimitedOrbitController:
    base: Any
    max_delta_v_m_s: float
    dt_s: float
    used_delta_v_m_s: float = 0.0

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        max_delta = float(max(self.max_delta_v_m_s, 0.0))
        if self.used_delta_v_m_s >= max_delta - 1.0e-9:
            return Command(
                thrust_eci_km_s2=np.zeros(3, dtype=float),
                torque_body_nm=np.zeros(3, dtype=float),
                mode_flags={
                    "mode": "delta_v_limited_coast",
                    "delta_v_limit_used_m_s": float(self.used_delta_v_m_s),
                    "delta_v_limit_max_m_s": max_delta,
                    "delta_v_limit_exhausted": True,
                },
            )

        cmd = self.base.act(belief, t_s, budget_ms)
        thrust = np.array(cmd.thrust_eci_km_s2, dtype=float).reshape(3)
        planned_delta_v_m_s = float(np.linalg.norm(thrust)) * float(max(self.dt_s, 0.0)) * 1000.0
        scale = 1.0
        if planned_delta_v_m_s > 0.0:
            remaining = max(max_delta - self.used_delta_v_m_s, 0.0)
            scale = min(1.0, remaining / planned_delta_v_m_s)
            thrust *= scale
            self.used_delta_v_m_s += planned_delta_v_m_s * scale

        mode_flags = dict(cmd.mode_flags or {})
        mode_flags.update(
            {
                "delta_v_limited": True,
                "delta_v_limit_scale": float(scale),
                "delta_v_limit_used_m_s": float(self.used_delta_v_m_s),
                "delta_v_limit_max_m_s": max_delta,
                "delta_v_limit_exhausted": bool(self.used_delta_v_m_s >= max_delta - 1.0e-9),
            }
        )
        return Command(
            thrust_eci_km_s2=thrust,
            torque_body_nm=np.array(cmd.torque_body_nm, dtype=float),
            mode_flags=mode_flags,
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self.base, item)


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
    current = getattr(agent, "orbit_controller", None)
    if current is None:
        return
    base = getattr(current, "base", current)
    if isinstance(base, _DeltaVLimitedOrbitController):
        return
    limited = _DeltaVLimitedOrbitController(
        base=base,
        max_delta_v_m_s=float(training_cfg.max_delta_v_m_s),
        dt_s=float(max(dt_s, 0.0)),
    )
    if hasattr(current, "base"):
        current.base = limited
        if hasattr(current, "_last_eval_t_s"):
            current._last_eval_t_s = None
        if hasattr(current, "_last_cmd"):
            current._last_cmd = Command.zero()
    else:
        agent.orbit_controller = limited


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
    current = getattr(agent, "orbit_controller", None)
    candidates = [current, getattr(current, "base", None)]
    for candidate in candidates:
        if isinstance(candidate, _DeltaVLimitedOrbitController):
            candidate.dt_s = float(max(dt_s, 0.0))
            return
