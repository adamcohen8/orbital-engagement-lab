from __future__ import annotations

import os
from pathlib import Path
import select
import sys
import termios
from time import perf_counter
from typing import Any
import tty

import matplotlib.pyplot as plt
import numpy as np

from sim.api import SimulationConfig, SimulationSession
from sim.game.dashboard import LiveBattlespaceDashboard
from sim.game.manual import KeyboardCommandState, ManualGameCommandProvider
from sim.presets.thrusters import resolve_thruster_max_thrust_n_from_specs, resolve_thruster_mount_from_specs


def _key_name(event: Any) -> str:
    return str(getattr(event, "key", "") or "").strip().lower()


def _max_accel_from_config(config: SimulationConfig, controlled_object_id: str) -> float:
    section = config.scenario.chaser if controlled_object_id == "chaser" else config.scenario.target
    game_cfg = dict(config.scenario.metadata.get("game", {}) or {})
    if "player_max_accel_km_s2" in game_cfg:
        return float(game_cfg["player_max_accel_km_s2"])
    params = dict((section.mission_strategy.params if section.mission_strategy is not None else {}) or {})
    if "max_accel_km_s2" in params:
        return float(params["max_accel_km_s2"])
    orbit_params = dict((section.orbit_control.params if section.orbit_control is not None else {}) or {})
    if "max_accel_km_s2" in orbit_params:
        return float(orbit_params["max_accel_km_s2"])
    specs = dict(section.specs or {})
    max_thrust_n = resolve_thruster_max_thrust_n_from_specs(specs)
    dry_mass_kg = specs.get("dry_mass_kg", specs.get("mass_kg"))
    fuel_mass_kg = specs.get("fuel_mass_kg", 0.0)
    if max_thrust_n is not None and dry_mass_kg is not None:
        wet_mass_kg = float(dry_mass_kg) + float(fuel_mass_kg or 0.0)
        if wet_mass_kg > 0.0:
            return float(max_thrust_n) / wet_mass_kg / 1e3
    return 2.0e-5


def _thruster_mounts_from_config(config: SimulationConfig) -> dict[str, dict[str, np.ndarray] | None]:
    out: dict[str, dict[str, np.ndarray] | None] = {}
    for oid, section in (("target", config.scenario.target), ("chaser", config.scenario.chaser)):
        mount = resolve_thruster_mount_from_specs(dict(section.specs or {}))
        if mount is None:
            out[oid] = None
        else:
            out[oid] = {
                "position_body_m": np.array(mount.position_body_m, dtype=float),
                "direction_body": np.array(mount.thrust_direction_body, dtype=float),
            }
    return out


def _attitude_dims_from_config(config: SimulationConfig) -> dict[str, list[float] | np.ndarray]:
    anim_cfg = dict(config.scenario.outputs.animations or {})
    raw = anim_cfg.get("battlespace_dashboard_attitude_dims_m", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _mass_specs_from_config(config: SimulationConfig, key: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for oid, section in (("target", config.scenario.target), ("chaser", config.scenario.chaser)):
        value = dict(section.specs or {}).get(key)
        out[oid] = None if value is None else float(value)
    return out


def _command_status(state: KeyboardCommandState) -> str:
    burn = "FIRE" if state.firing else "coast"
    return (
        "W/S pitch  A/D yaw  Left/Right roll  Space fire  R reset  Esc quit\n"
        "Keys work in the figure window or this terminal; terminal input is pulse/repeat based.\n"
        f"pitch={state.pitch:+.0f} yaw={state.yaw:+.0f} roll={state.roll:+.0f} thrust={burn}"
    )


class _TerminalKeyInput:
    def __init__(self, *, pulse_s: float = 0.8) -> None:
        self.pulse_s = float(max(pulse_s, 0.05))
        self.enabled = bool(sys.stdin.isatty())
        self.fd = sys.stdin.fileno() if self.enabled else -1
        self._old_attrs: list[Any] | None = None
        self._buffer = ""
        self._pitch_until = 0.0
        self._pitch_value = 0.0
        self._yaw_until = 0.0
        self._yaw_value = 0.0
        self._roll_until = 0.0
        self._roll_value = 0.0
        self._fire_until = 0.0
        self._used = False

    def __enter__(self) -> "_TerminalKeyInput":
        if self.enabled:
            self._old_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_: object) -> None:
        if self.enabled and self._old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old_attrs)

    def poll(self, state: KeyboardCommandState) -> None:
        if not self.enabled:
            return
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            chunk = os.read(self.fd, 32).decode(errors="ignore")
            if not chunk:
                break
            self._buffer += chunk
        for key in self._drain_keys():
            self._apply_key(key, state)
        self._apply_expiry(state)

    def _drain_keys(self) -> list[str]:
        keys: list[str] = []
        while self._buffer:
            if self._buffer.startswith("\x1b[D"):
                keys.append("left")
                self._buffer = self._buffer[3:]
            elif self._buffer.startswith("\x1b[C"):
                keys.append("right")
                self._buffer = self._buffer[3:]
            elif self._buffer.startswith("\x1b"):
                keys.append("escape")
                self._buffer = self._buffer[1:]
            else:
                keys.append(self._buffer[0].lower())
                self._buffer = self._buffer[1:]
        return keys

    def _apply_key(self, key: str, state: KeyboardCommandState) -> None:
        now = perf_counter()
        until = now + self.pulse_s
        self._used = True
        if key == "w":
            self._pitch_value = 1.0
            self._pitch_until = until
        elif key == "s":
            self._pitch_value = -1.0
            self._pitch_until = until
        elif key == "a":
            self._yaw_value = -1.0
            self._yaw_until = until
        elif key == "d":
            self._yaw_value = 1.0
            self._yaw_until = until
        elif key == "left":
            self._roll_value = -1.0
            self._roll_until = until
        elif key == "right":
            self._roll_value = 1.0
            self._roll_until = until
        elif key == " ":
            self._fire_until = until
        elif key == "r":
            state.reset_requested = True
        elif key in {"escape", "\x03"}:
            state.quit_requested = True
        self._apply_expiry(state)

    def _apply_expiry(self, state: KeyboardCommandState) -> None:
        if not self._used:
            return
        now = perf_counter()
        state.pitch = self._pitch_value if now <= self._pitch_until else 0.0
        state.yaw = self._yaw_value if now <= self._yaw_until else 0.0
        state.roll = self._roll_value if now <= self._roll_until else 0.0
        state.firing = bool(now <= self._fire_until)
        if now > max(self._pitch_until, self._yaw_until, self._roll_until, self._fire_until):
            self._used = False


def run_game_mode(
    config_path: str | Path,
    *,
    controlled_object_id: str = "chaser",
    attitude_rate_deg_s: float = 45.0,
    realtime: bool = True,
) -> None:
    config = SimulationConfig.from_yaml(config_path)
    session = SimulationSession.from_config(config)
    command_state = KeyboardCommandState()
    provider = ManualGameCommandProvider(
        command_state=command_state,
        max_accel_km_s2=_max_accel_from_config(config, controlled_object_id),
        attitude_rate_deg_s=attitude_rate_deg_s,
        controlled_object_id=controlled_object_id,
    )
    session.set_external_intent_provider(controlled_object_id, provider)
    snapshot = session.reset()
    if snapshot is None:
        raise RuntimeError("Game mode requires a single-run scenario.")
    provider.reset_target_to_current(snapshot.truth[controlled_object_id])

    anim_cfg = dict(config.scenario.outputs.animations or {})
    dashboard = LiveBattlespaceDashboard(
        target_object_id=str(anim_cfg.get("battlespace_dashboard_target_object_id", "target")),
        chaser_object_id=str(anim_cfg.get("battlespace_dashboard_chaser_object_id", "chaser")),
        prism_dims_m_by_object=_attitude_dims_from_config(config),
        thruster_mounts_by_object=_thruster_mounts_from_config(config),
        dry_mass_kg_by_object=_mass_specs_from_config(config, "dry_mass_kg"),
        fuel_capacity_kg_by_object=_mass_specs_from_config(config, "fuel_mass_kg"),
        thruster_active_threshold_km_s2=float(anim_cfg.get("battlespace_dashboard_thruster_active_threshold_km_s2", 1e-15)),
        show_trajectory=bool(anim_cfg.get("battlespace_dashboard_show_trajectory", True)),
    )

    def on_press(event: Any) -> None:
        key = _key_name(event)
        if key == "w":
            command_state.pitch = 1.0
        elif key == "s":
            command_state.pitch = -1.0
        elif key == "a":
            command_state.yaw = -1.0
        elif key == "d":
            command_state.yaw = 1.0
        elif key == "left":
            command_state.roll = -1.0
        elif key == "right":
            command_state.roll = 1.0
        elif key == " ":
            command_state.firing = True
        elif key == "r":
            command_state.reset_requested = True
        elif key == "escape":
            command_state.quit_requested = True

    def on_release(event: Any) -> None:
        key = _key_name(event)
        if key in {"w", "s"}:
            command_state.pitch = 0.0
        elif key in {"a", "d"}:
            command_state.yaw = 0.0
        elif key in {"left", "right"}:
            command_state.roll = 0.0
        elif key == " ":
            command_state.firing = False

    dashboard.connect_key_handlers(on_press, on_release)
    dashboard.push_snapshot(snapshot)
    dashboard.render(command_status=_command_status(command_state))

    dt_s = float(config.scenario.simulator.dt_s)
    last_step_wall = perf_counter()
    with _TerminalKeyInput() as terminal_input:
        while (not session.done) and (not dashboard.closed) and (not command_state.quit_requested):
            terminal_input.poll(command_state)
            dashboard.render(command_status=_command_status(command_state))
            if realtime:
                now = perf_counter()
                wait_s = dt_s - (now - last_step_wall)
                if wait_s > 0.0:
                    plt.pause(min(wait_s, 0.03))
                    continue
                last_step_wall = perf_counter()
            snapshot = session.step()
            dashboard.push_snapshot(snapshot)
            dashboard.render(command_status=_command_status(command_state))
            plt.pause(0.001)

    plt.ioff()
    if not dashboard.closed:
        dashboard.render(command_status=_command_status(command_state) + "\nSimulation ended.")
        plt.show()
