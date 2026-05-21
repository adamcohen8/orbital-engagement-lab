from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sim.game.audio import ARCADE_ROUND_CLEAR_SOUND_PATH, _play_game_sound_effect, _stop_game_music, _sync_game_music
from sim.game.training import RPOTrainingConfig


@dataclass
class GameAudioController:
    pygame: Any
    music_enabled: bool = True
    active_path: Path | None = None

    def sync(
        self,
        score: Any,
        *,
        training_cfg: RPOTrainingConfig,
        override_level_path: Path | None = None,
    ) -> Path | None:
        self.active_path = _sync_game_music(
            self.pygame,
            score,
            training_cfg=training_cfg,
            music_enabled=self.music_enabled,
            active_path=self.active_path,
            override_level_path=override_level_path,
        )
        return self.active_path

    def toggle(
        self,
        score: Any,
        *,
        training_cfg: RPOTrainingConfig,
        override_level_path: Path | None = None,
    ) -> Path | None:
        self.music_enabled = not bool(self.music_enabled)
        return self.sync(score, training_cfg=training_cfg, override_level_path=override_level_path)

    def stop(self) -> None:
        _stop_game_music(self.pygame)
        self.active_path = None

    def clear_active_path(self) -> None:
        self.active_path = None

    def play_round_clear(self, *, volume: float = 0.74) -> None:
        if self.music_enabled:
            _play_game_sound_effect(self.pygame, ARCADE_ROUND_CLEAR_SOUND_PATH, volume=volume)
