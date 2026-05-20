from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim.game.training import RPOTrainingConfig

GAME_MUSIC_DIR = Path(__file__).resolve().parent / "music"
LEVEL_MUSIC_PATHS: dict[str, Path] = {
    "rpo_00_tutorial": GAME_MUSIC_DIR / "10_training_grid_sunrise.wav",
    "rpo_02_vbar_approach": GAME_MUSIC_DIR / "02_rendezvous_vector.wav",
    "rpo_03_rbar_approach": GAME_MUSIC_DIR / "18_keepout_zone_accelerando.wav",
    "rpo_04_rendezvous": GAME_MUSIC_DIR / "06_casting_the_orbit_line.wav",
    "rpo_05_passive_cross_track_approach": GAME_MUSIC_DIR / "19_cross_track_ghost_orbit.wav",
    "rpo_06_elliptic_burn_then_approach": GAME_MUSIC_DIR / "08_silent_running_radar.wav",
    "rpo_07_elliptic_nmc": GAME_MUSIC_DIR / "04_docking_bay_neon.wav",
    "rpo_08_elliptic_rendezvous": GAME_MUSIC_DIR / "23_elliptic_final_burn_cinematic.wav",
    "rpo_09_defensive_target_demo": GAME_MUSIC_DIR / "17_orbital_boss_metal.wav",
    "rpo_10_evasive_target_survival": GAME_MUSIC_DIR / "09_defender_boss_vector.wav",
    "rpo_arcade_pursuit": GAME_MUSIC_DIR / "21_pursuit_arcade_overdrive_no_siren_demo.wav",
}
MISSION_SUCCESS_MUSIC_PATH = GAME_MUSIC_DIR / "05_final_burn_victory_loop.wav"
MISSION_FAILURE_MUSIC_PATH = GAME_MUSIC_DIR / "15_mission_failed_lament_credits.wav"
ARCADE_ROUND_CLEAR_SOUND_PATH = GAME_MUSIC_DIR / "22_arcade_round_clear_flyover.wav"


def _result_music_path(score: Any) -> Path | None:
    if bool(getattr(score, "level_passed", False)):
        return MISSION_SUCCESS_MUSIC_PATH
    if bool(getattr(score, "level_failed", False)):
        return MISSION_FAILURE_MUSIC_PATH
    return None


def _level_music_path(training_cfg: RPOTrainingConfig) -> Path | None:
    return LEVEL_MUSIC_PATHS.get(str(training_cfg.scenario_id or ""))


def _sync_game_music(
    pygame: Any,
    score: Any,
    *,
    training_cfg: RPOTrainingConfig,
    music_enabled: bool,
    active_path: Path | None,
    override_level_path: Path | None = None,
) -> Path | None:
    result_path = _result_music_path(score)
    desired_path = (result_path or override_level_path or _level_music_path(training_cfg)) if music_enabled else None
    if desired_path == active_path:
        return active_path
    _stop_game_music(pygame)
    if desired_path is None:
        return None
    loops = 0 if result_path is not None else -1
    return desired_path if _play_game_music(pygame, desired_path, loops=loops) else None


def _play_game_music(pygame: Any, path: Path, *, loops: int = 0) -> bool:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.set_volume(0.65)
        pygame.mixer.music.play(loops)
    except (OSError, pygame.error):
        return False
    return True


def _play_game_sound_effect(pygame: Any, path: Path, *, volume: float = 0.75) -> bool:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        sound = pygame.mixer.Sound(str(path))
        sound.set_volume(float(np.clip(volume, 0.0, 1.0)))
        sound.play()
    except Exception:
        return False
    return True


def _stop_game_music(pygame: Any | None) -> None:
    if pygame is None:
        return
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except pygame.error:
        return
