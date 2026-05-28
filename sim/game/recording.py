from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np

WriterFactory = Callable[[Path, float], Any]
CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass
class GameFrameRecorder:
    path: Path
    fps: float = 30.0
    writer_factory: WriterFactory | None = None
    _writer: Any | None = field(default=None, init=False, repr=False)
    frames_written: int = 0
    saved: bool = False
    closed: bool = False

    @classmethod
    def start(
        cls,
        path: str | Path,
        *,
        fps: float = 30.0,
        writer_factory: WriterFactory | None = None,
    ) -> GameFrameRecorder:
        recorder = cls(path=Path(path), fps=float(fps), writer_factory=writer_factory)
        recorder.path.parent.mkdir(parents=True, exist_ok=True)
        recorder._writer = recorder._open_writer()
        return recorder

    def capture_surface(self, surface: Any) -> None:
        if self.closed or self.saved:
            return
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - pygame is already required by the dashboard.
            raise RuntimeError("Game recording requires `pygame`. Install with `pip install .[game]`.") from exc
        frame = pygame.surfarray.array3d(surface)
        self.capture_frame(np.transpose(frame, (1, 0, 2)))

    def capture_frame(self, frame: np.ndarray) -> None:
        if self.closed or self.saved:
            return
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
            raise ValueError("Recorded frames must have shape (height, width, 3|4).")
        assert self._writer is not None
        self._writer.append_data(arr[:, :, :3])
        self.frames_written += 1

    def finish(self) -> Path | None:
        if self.closed:
            return self.path if self.saved else None
        self._close_writer()
        if self.frames_written <= 0:
            self._unlink_output()
            return None
        self.saved = True
        return self.path

    def discard(self) -> None:
        if not self.closed:
            self._close_writer()
        self.saved = False
        self._unlink_output()

    def _open_writer(self) -> Any:
        if self.writer_factory is not None:
            return self.writer_factory(self.path, self.fps)
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Game recording requires `imageio` and `imageio-ffmpeg`. Install with `pip install .[game]`."
            ) from exc
        if find_spec("imageio_ffmpeg") is None:
            raise RuntimeError(
                "Game recording requires the FFMPEG backend from `imageio-ffmpeg`. Install with `pip install .[game]`."
            )
        return imageio.get_writer(self.path, fps=float(self.fps), codec="libx264", quality=8, macro_block_size=16)

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self.closed = True

    def _unlink_output(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def game_recording_path(
    *,
    scenario_name: str,
    difficulty: str,
    attempt_index: int,
    output_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    root = Path(output_dir) if output_dir is not None else Path("outputs") / "game_recordings"
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    scenario = _slug(scenario_name or "game")
    diff = _slug(difficulty or "easy")
    return root / f"{scenario}_{diff}_{stamp}_attempt{max(int(attempt_index), 1):02d}.mp4"


def game_clip_recording_path(
    *,
    scenario_name: str,
    difficulty: str,
    clip_index: int,
    output_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    root = Path(output_dir) if output_dir is not None else Path("outputs") / "game_recordings"
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    scenario = _slug(scenario_name or "game")
    diff = _slug(difficulty or "easy")
    return root / "clips" / f"{scenario}_{diff}_{stamp}_clip{max(int(clip_index), 1):02d}.mp4"


def add_looped_audio_to_video(
    video_path: str | Path,
    audio_path: str | Path | None,
    *,
    ffmpeg_exe: str | Path | None = None,
    runner: CommandRunner = subprocess.run,
) -> Path:
    video = Path(video_path)
    if audio_path is None:
        return video
    audio = Path(audio_path)
    if not video.exists():
        raise FileNotFoundError(video)
    if not audio.exists():
        raise FileNotFoundError(audio)
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "Adding music to game recordings requires `imageio-ffmpeg`. Install with `pip install .[game]`."
            ) from exc
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    tmp = video.with_name(f".{video.stem}.audio-{uuid4().hex}{video.suffix}")
    cmd = [
        str(ffmpeg_exe),
        "-y",
        "-i",
        str(video),
        "-stream_loop",
        "-1",
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        runner(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tmp.replace(video)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return video


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    out = []
    last_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_sep = False
        elif not last_sep:
            out.append("_")
            last_sep = True
    return "".join(out).strip("_") or "game"
