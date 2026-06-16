from __future__ import annotations

import argparse
from pathlib import Path

from sim.game.launcher import choose_game_launch, record_game_progress
from sim.game.runner import run_game_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Orbital Engagement Lab game mode.")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Simulation YAML config to run. Omit to open the level selector.",
    )
    parser.add_argument(
        "--controlled-object",
        default=None,
        help="Object id controlled by keyboard input. Defaults to the game config setting.",
    )
    parser.add_argument("--attitude-rate-deg-s", type=float, default=45.0, help="Commanded attitude target slew rate.")
    parser.add_argument(
        "--fast", action="store_true", help="Step as fast as the dashboard can render instead of realtime."
    )
    parser.add_argument(
        "--speed-multiple",
        type=float,
        default=None,
        help=(
            "Realtime playback speed. For example, 10 means 10 seconds of sim time per 1 second of real time. "
            "Defaults to the level's configured value, or 1x."
        ),
    )
    args = parser.parse_args()
    if args.config:
        result = run_game_mode(
            Path(args.config),
            controlled_object_id=None if args.controlled_object is None else str(args.controlled_object),
            attitude_rate_deg_s=float(args.attitude_rate_deg_s),
            realtime=not bool(args.fast),
            speed_multiple=args.speed_multiple,
        )
        if result.level_passed or result.arcade_score > 0:
            record_game_progress(
                result.config_path,
                result.difficulty,
                score=result.arcade_score,
                completed=result.level_passed,
            )
        return

    show_start_screen = True
    while True:
        selection = choose_game_launch(show_start_screen=show_start_screen)
        if selection is None:
            return
        show_start_screen = False
        result = run_game_mode(
            selection.path,
            controlled_object_id=None if args.controlled_object is None else str(args.controlled_object),
            attitude_rate_deg_s=float(args.attitude_rate_deg_s),
            realtime=not bool(args.fast),
            speed_multiple=args.speed_multiple,
            difficulty_override=selection.difficulty,
            music_enabled=selection.music_enabled,
            record_video=selection.record_video,
        )
        if result.level_passed or result.arcade_score > 0:
            record_game_progress(
                result.config_path,
                result.difficulty,
                score=result.arcade_score,
                completed=result.level_passed,
            )


if __name__ == "__main__":
    main()
