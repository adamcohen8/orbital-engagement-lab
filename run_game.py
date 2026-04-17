from __future__ import annotations

import argparse
from pathlib import Path

from sim.game.runner import run_game_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Orbital Engagement Lab game mode.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/game_mode_basic.yaml",
        help="Simulation YAML config to run.",
    )
    parser.add_argument("--controlled-object", default="chaser", help="Object id controlled by keyboard input.")
    parser.add_argument("--attitude-rate-deg-s", type=float, default=45.0, help="Commanded attitude target slew rate.")
    parser.add_argument("--fast", action="store_true", help="Step as fast as the dashboard can render instead of realtime.")
    args = parser.parse_args()

    run_game_mode(
        Path(args.config),
        controlled_object_id=str(args.controlled_object),
        attitude_rate_deg_s=float(args.attitude_rate_deg_s),
        realtime=not bool(args.fast),
    )


if __name__ == "__main__":
    main()
