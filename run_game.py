from __future__ import annotations

import argparse
from pathlib import Path

from sim.game.launcher import choose_game_scenario
from sim.game.runner import run_game_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Orbital Engagement Lab game mode.")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Simulation YAML config to run. Omit to open the level selector.",
    )
    parser.add_argument("--controlled-object", default="chaser", help="Object id controlled by keyboard input.")
    parser.add_argument("--attitude-rate-deg-s", type=float, default=45.0, help="Commanded attitude target slew rate.")
    parser.add_argument("--fast", action="store_true", help="Step as fast as the dashboard can render instead of realtime.")
    parser.add_argument(
        "--speed-multiple",
        type=float,
        default=1.0,
        help="Realtime playback speed. For example, 10 means 10 seconds of sim time per 1 second of real time.",
    )
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else choose_game_scenario()
    if config_path is None:
        return

    run_game_mode(
        config_path,
        controlled_object_id=str(args.controlled_object),
        attitude_rate_deg_s=float(args.attitude_rate_deg_s),
        realtime=not bool(args.fast),
        speed_multiple=float(args.speed_multiple),
    )


if __name__ == "__main__":
    main()
