from __future__ import annotations

import argparse
from pathlib import Path

from sim.game.launcher import choose_game_launch, record_game_progress
from sim.game.presentation import PRESENTATION_MODES, PRESENTATION_VSYNC_MODES
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
    parser.add_argument(
        "--presentation-mode",
        choices=PRESENTATION_MODES,
        default=None,
        help=(
            "Display architecture. 'compatibility' preserves the current frame pacing; "
            "'standard', 'high_refresh', and 'auto' enable the new presentation path."
        ),
    )
    parser.add_argument(
        "--presentation-fps-cap",
        type=float,
        default=None,
        help="Optional live-display FPS ceiling for the new presentation path.",
    )
    parser.add_argument(
        "--presentation-refresh-hz",
        type=float,
        default=None,
        help="Optional monitor-refresh override when Pygame cannot detect it.",
    )
    parser.add_argument(
        "--presentation-vsync",
        choices=PRESENTATION_VSYNC_MODES,
        default=None,
        help="VSync preference for the new presentation path.",
    )
    parser.add_argument(
        "--presentation-diagnostics",
        action="store_true",
        default=None,
        help="Show the frame diagnostics overlay.",
    )
    parser.add_argument(
        "--presentation-diagnostics-output",
        type=Path,
        default=None,
        help="Write a machine-readable presentation diagnostics summary when the game closes.",
    )
    args = parser.parse_args()
    if args.config:
        result = run_game_mode(
            Path(args.config),
            controlled_object_id=None if args.controlled_object is None else str(args.controlled_object),
            attitude_rate_deg_s=float(args.attitude_rate_deg_s),
            realtime=not bool(args.fast),
            speed_multiple=args.speed_multiple,
            presentation_mode=args.presentation_mode,
            presentation_fps_cap=args.presentation_fps_cap,
            presentation_refresh_hz=args.presentation_refresh_hz,
            presentation_vsync=args.presentation_vsync,
            presentation_diagnostics=args.presentation_diagnostics,
            presentation_diagnostics_output=args.presentation_diagnostics_output,
        )
        if result.level_passed or result.arcade_score > 0:
            record_game_progress(
                result.config_path,
                result.difficulty,
                score=result.arcade_score,
                completed=result.level_passed,
                mode=result.mode,
            )
        return

    show_start_screen = True
    selector_mode = "pilot"
    while True:
        selection = choose_game_launch(show_start_screen=show_start_screen, initial_mode=selector_mode)
        if selection is None:
            return
        show_start_screen = False
        selector_mode = selection.mode
        result = run_game_mode(
            selection.path,
            controlled_object_id=None if args.controlled_object is None else str(args.controlled_object),
            attitude_rate_deg_s=float(args.attitude_rate_deg_s),
            realtime=not bool(args.fast),
            speed_multiple=args.speed_multiple,
            difficulty_override=selection.difficulty,
            music_enabled=selection.music_enabled,
            record_video=selection.record_video,
            game_mode=selection.mode,
            frame_convention=selection.frame_convention,
            operator_burn_plan=selection.operator_burn_plan,
            skip_initial_briefing=selection.skip_initial_briefing,
            presentation_mode=args.presentation_mode or selection.presentation_mode,
            presentation_fps_cap=args.presentation_fps_cap,
            presentation_refresh_hz=args.presentation_refresh_hz,
            presentation_vsync=args.presentation_vsync,
            presentation_diagnostics=args.presentation_diagnostics,
            presentation_diagnostics_output=args.presentation_diagnostics_output,
        )
        if result.level_passed or result.arcade_score > 0:
            record_game_progress(
                result.config_path,
                result.difficulty,
                score=result.arcade_score,
                completed=result.level_passed,
                mode=result.mode,
            )
        selector_mode = result.mode


if __name__ == "__main__":
    main()
