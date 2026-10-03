"""Explicit, parallel game backend selection; Rust is the Trainer default."""

from __future__ import annotations

from sim.api import SimulationConfig
from sim.game.session import GamePhysicsSession


def game_backend(config: SimulationConfig, backend: str | None = None) -> str:
    game = dict(config.scenario.metadata.get("game", {}) or {})
    value = game.get("backend", "rust") if backend is None else backend
    if not isinstance(value, str) or value.strip().lower() not in {"python", "rust"}:
        raise ValueError("metadata.game.backend must be python or rust")
    return value.strip().lower()


def configure_game_backend(config: SimulationConfig, backend: str | None = None) -> SimulationConfig:
    selected = game_backend(config, backend)
    if selected == "rust":
        from sim.flight_software.rust_game_backend import extension

        extension()
    root = config.to_dict()
    root.setdefault("metadata", {}).setdefault("game", {})["backend"] = selected
    dynamics = root.setdefault("simulator", {}).setdefault("dynamics", {})
    dynamics.setdefault("orbit", {})["numeric_backend"] = selected
    for section in ("attitude", "rocket", "reentry"):
        dynamics.setdefault(section, {})["numeric_backend"] = selected
    for object_config in root.get("objects", {}).values():
        flight_software = object_config.get("flight_software")
        if isinstance(flight_software, dict):
            flight_software.setdefault("params", {})["numeric_backend"] = selected
        knowledge = object_config.get("knowledge")
        if isinstance(knowledge, dict) and knowledge:
            knowledge["numeric_backend"] = selected
            estimation = knowledge.get("estimation")
            if isinstance(estimation, dict):
                estimation["numeric_backend"] = selected
                ekf = estimation.get("ekf")
                if isinstance(ekf, dict):
                    ekf["numeric_backend"] = selected
    if root["simulator"].get("collisions", {}).get("enabled", False):
        root["simulator"]["collisions"]["numeric_backend"] = selected
    # Validate physics support and wheel symbols through the existing contract.
    return SimulationConfig.from_dict(root, source_path=config.source_path)


class RustGamePhysicsSession(GamePhysicsSession):
    """Game session using native physics and the parallel Rust FSW stacks."""

    def __init__(self, config: SimulationConfig, *, retained_history_samples: int = 4096):
        super().__init__(configure_game_backend(config, "rust"), retained_history_samples=retained_history_samples)


def create_game_physics_session(
    config: SimulationConfig, *, backend: str | None = None, retained_history_samples: int = 4096
) -> GamePhysicsSession:
    selected = game_backend(config, backend)
    if selected == "rust":
        return RustGamePhysicsSession(config, retained_history_samples=retained_history_samples)
    return GamePhysicsSession(
        configure_game_backend(config, backend), retained_history_samples=retained_history_samples
    )
