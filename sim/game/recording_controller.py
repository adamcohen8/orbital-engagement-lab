from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from sim.api import SimulationConfig
from sim.game.audio import _level_music_path
from sim.game.recording import (
    GameFrameRecorder,
    add_looped_audio_to_video,
    game_clip_recording_path,
    game_recording_path,
)
from sim.game.training import RPOTrainingConfig


@dataclass
class GameRecordingController:
    enabled: bool
    config: SimulationConfig
    difficulty: str
    output_dir: str | Path | None = None
    fps: float = 30.0
    attempt_index: int = 1
    recorder: GameFrameRecorder | None = None
    recording_path: Path | None = None

    def start(self) -> GameFrameRecorder | None:
        self.recorder = start_game_recorder(
            enabled=self.enabled,
            config=self.config,
            difficulty=self.difficulty,
            attempt_index=self.attempt_index,
            output_dir=self.output_dir,
            fps=self.fps,
        )
        return self.recorder

    def restart(self) -> GameFrameRecorder | None:
        self.discard()
        self.attempt_index += 1
        self.recording_path = None
        return self.start()

    def capture(self, dashboard: Any) -> GameFrameRecorder | None:
        self.recorder = safe_capture_recording_frame(self.recorder, dashboard)
        return self.recorder

    def capture_hold(self, dashboard: Any, *, duration_s: float, fps: float | None = None) -> GameFrameRecorder | None:
        frame_count = recording_hold_frame_count(duration_s=duration_s, fps=self.fps if fps is None else fps)
        for _ in range(frame_count):
            if self.recorder is None:
                break
            self.capture(dashboard)
        return self.recorder

    def finish(
        self,
        training_cfg: RPOTrainingConfig,
        *,
        override_level_path: Path | None = None,
    ) -> Path | None:
        if self.recorder is None:
            return None
        self.recording_path = finish_game_recording(
            self.recorder,
            training_cfg,
            override_level_path=override_level_path,
        )
        if self.recording_path is None:
            self.recorder = None
        return self.recording_path

    def discard(self) -> None:
        if self.recorder is not None and not bool(getattr(self.recorder, "saved", False)):
            discard_recorder_safely(self.recorder)
        self.recorder = None


@dataclass
class GameClipRecordingController:
    config: SimulationConfig
    difficulty: str
    output_dir: str | Path | None = None
    fps: float = 30.0
    enabled: bool = True
    clip_index: int = 0
    recorder: GameFrameRecorder | None = None
    recording_path: Path | None = None

    @property
    def recording(self) -> bool:
        return self.recorder is not None and not bool(getattr(self.recorder, "saved", False))

    def start_next(self) -> GameFrameRecorder | None:
        if self.recording:
            return self.recorder
        self.clip_index += 1
        self.recording_path = None
        self.recorder = start_game_clip_recorder(
            enabled=self.enabled,
            config=self.config,
            difficulty=self.difficulty,
            clip_index=self.clip_index,
            output_dir=self.output_dir,
            fps=self.fps,
        )
        return self.recorder

    def capture(self, dashboard: Any) -> GameFrameRecorder | None:
        self.recorder = safe_capture_recording_frame(self.recorder, dashboard)
        return self.recorder

    def finish(
        self,
        training_cfg: RPOTrainingConfig,
        *,
        override_level_path: Path | None = None,
    ) -> Path | None:
        if self.recorder is None:
            return None
        self.recording_path = finish_game_recording(
            self.recorder,
            training_cfg,
            override_level_path=override_level_path,
        )
        self.recorder = None
        return self.recording_path

    def discard(self) -> None:
        if self.recorder is not None and not bool(getattr(self.recorder, "saved", False)):
            discard_recorder_safely(self.recorder)
        self.recorder = None


def start_game_recorder(
    *,
    enabled: bool,
    config: SimulationConfig,
    difficulty: str,
    attempt_index: int,
    output_dir: str | Path | None,
    fps: float,
) -> GameFrameRecorder | None:
    if not bool(enabled):
        return None
    path = next_available_recording_path(
        game_recording_path(
            scenario_name=str(config.scenario.scenario_name or "game"),
            difficulty=difficulty,
            attempt_index=attempt_index,
            output_dir=output_dir,
        )
    )
    try:
        return GameFrameRecorder.start(path, fps=fps)
    except Exception as exc:
        print(f"Disabled game recording; could not start recorder: {exc}")
        return None


def recording_hold_frame_count(*, duration_s: float, fps: float) -> int:
    duration = max(float(duration_s), 0.0)
    frame_rate = max(float(fps), 1.0)
    return max(int(round(duration * frame_rate)), 0)


def start_game_clip_recorder(
    *,
    enabled: bool,
    config: SimulationConfig,
    difficulty: str,
    clip_index: int,
    output_dir: str | Path | None,
    fps: float,
) -> GameFrameRecorder | None:
    if not bool(enabled):
        return None
    path = game_clip_recording_path(
        scenario_name=str(config.scenario.scenario_name or "game"),
        difficulty=difficulty,
        clip_index=clip_index,
        output_dir=output_dir,
    )
    path = next_available_recording_path(path)
    try:
        return GameFrameRecorder.start(path, fps=fps)
    except Exception as exc:
        print(f"Disabled game clip recording; could not start recorder: {exc}")
        return None


def capture_recording_frame(recorder: GameFrameRecorder | None, dashboard: Any) -> None:
    if recorder is None or recorder.saved:
        return
    recorder.capture_surface(dashboard.screen)


def next_available_recording_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    for idx in range(2, 1000):
        numbered = candidate.with_name(f"{candidate.stem}_{idx:02d}{candidate.suffix}")
        if not numbered.exists():
            return numbered
    return candidate.with_name(f"{candidate.stem}_{uuid4().hex}{candidate.suffix}")


def safe_capture_recording_frame(recorder: GameFrameRecorder | None, dashboard: Any) -> GameFrameRecorder | None:
    if recorder is None or recorder.saved:
        return recorder
    try:
        capture_recording_frame(recorder, dashboard)
    except Exception as exc:
        print(f"Disabled game recording; could not capture frame: {exc}")
        discard_recorder_safely(recorder)
        return None
    return recorder


def finish_game_recording(
    recorder: GameFrameRecorder,
    training_cfg: RPOTrainingConfig,
    *,
    override_level_path: Path | None = None,
) -> Path | None:
    try:
        recording_path = recorder.finish()
    except Exception as exc:
        print(f"Discarded game recording; could not finalize video: {exc}")
        discard_recorder_safely(recorder)
        return None
    if recording_path is None:
        return None
    return add_level_music_to_recording(recording_path, training_cfg, override_level_path=override_level_path)


def discard_recorder_safely(recorder: GameFrameRecorder) -> None:
    try:
        recorder.discard()
    except Exception:
        return


def add_level_music_to_recording(
    recording_path: Path,
    training_cfg: RPOTrainingConfig,
    *,
    override_level_path: Path | None = None,
) -> Path:
    music_path = override_level_path or _level_music_path(training_cfg)
    if music_path is None:
        return recording_path
    try:
        return add_looped_audio_to_video(recording_path, music_path)
    except Exception as exc:
        print(f"Saved silent game recording; could not add level music: {exc}")
        return recording_path
