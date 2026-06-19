#!/usr/bin/env python3
"""Create original procedural lunar mission music demos for the RPO trainer."""

from __future__ import annotations

import argparse
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = ROOT / "sim" / "game" / "music"
SAMPLE_RATE = 44_100
TAU = 2.0 * math.pi


NOTE_OFFSETS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


@dataclass(frozen=True)
class DemoSpec:
    filename: str
    bpm: float
    duration_s: float
    chords: tuple[tuple[str, ...], ...]
    bell_pattern: tuple[tuple[float, str, float, float], ...]
    pulse_notes: tuple[str, ...]
    accent_notes: tuple[str, ...]
    color: str
    seed: int


def note_hz(note: str) -> float:
    octave = int(note[-1])
    name = note[:-1]
    semitone = NOTE_OFFSETS[name] + (octave + 1) * 12
    return 440.0 * (2.0 ** ((semitone - 69) / 12.0))


def timebase(duration_s: float) -> np.ndarray:
    n = int(round(duration_s * SAMPLE_RATE))
    return np.arange(n, dtype=np.float64) / SAMPLE_RATE


def equal_power_pan(pan: float) -> tuple[float, float]:
    p = float(np.clip(pan, -1.0, 1.0))
    angle = (p + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def add_mono(stereo: np.ndarray, mono: np.ndarray, pan: float = 0.0) -> None:
    left, right = equal_power_pan(pan)
    stereo[:, 0] += mono * left
    stereo[:, 1] += mono * right


def smooth_gate(t: np.ndarray, start: float, end: float, edge_s: float) -> np.ndarray:
    attack = np.clip((t - start) / max(edge_s, 1e-6), 0.0, 1.0)
    release = np.clip((end - t) / max(edge_s, 1e-6), 0.0, 1.0)
    gate = np.minimum(attack, release)
    return gate * gate * (3.0 - 2.0 * gate)


def pluck(t: np.ndarray, start: float, freq: float, *, decay: float, tone: float, amp: float) -> np.ndarray:
    x = t - start
    active = x >= 0.0
    out = np.zeros_like(t)
    local = x[active]
    env = np.exp(-local / decay)
    mod = np.sin(TAU * freq * 2.01 * local) * tone * np.exp(-local / (decay * 0.55))
    out[active] = amp * env * np.sin(TAU * freq * local + mod)
    return out


def low_pulse(t: np.ndarray, start: float, freq: float, *, amp: float, decay: float, click: float) -> np.ndarray:
    x = t - start
    active = x >= 0.0
    out = np.zeros_like(t)
    local = x[active]
    sweep = freq * (1.0 + 0.18 * np.exp(-local / 0.22))
    phase = TAU * np.cumsum(sweep) / SAMPLE_RATE
    body = np.sin(phase) + 0.35 * np.sin(phase * 0.5)
    env = np.exp(-local / decay)
    tick = click * np.exp(-local / 0.018) * np.sin(TAU * 900.0 * local)
    out[active] = amp * env * body + amp * tick
    return out


def add_pad(stereo: np.ndarray, t: np.ndarray, spec: DemoSpec) -> None:
    beat = 60.0 / spec.bpm
    chord_len = 8.0 * beat
    for idx, chord in enumerate(spec.chords):
        start = idx * chord_len
        end = min(start + chord_len + 2.5, spec.duration_s)
        gate = smooth_gate(t, start, end, edge_s=2.7)
        if not np.any(gate):
            continue
        for note_idx, note in enumerate(chord):
            base = note_hz(note)
            pan = -0.45 + 0.9 * (note_idx / max(len(chord) - 1, 1))
            for detune_cents, gain in ((-5.5, 0.35), (0.0, 0.45), (6.5, 0.32)):
                freq = base * (2.0 ** (detune_cents / 1200.0))
                lfo = 0.0025 * np.sin(TAU * (0.035 + 0.004 * note_idx) * t + note_idx)
                phase = TAU * freq * t + lfo
                voice = gain * np.sin(phase) + 0.16 * gain * np.sin(2.0 * phase + 0.4)
                add_mono(stereo, 0.075 * gate * voice, pan=pan)


def add_bells(stereo: np.ndarray, t: np.ndarray, spec: DemoSpec) -> None:
    beat = 60.0 / spec.bpm
    for beat_offset, note, amp, pan in spec.bell_pattern:
        start = beat_offset * beat
        while start < spec.duration_s - 1.0:
            add_mono(
                stereo,
                pluck(t, start, note_hz(note), decay=2.8, tone=4.3, amp=amp),
                pan=pan,
            )
            start += beat * 16.0


def add_pulses(stereo: np.ndarray, t: np.ndarray, spec: DemoSpec) -> None:
    beat = 60.0 / spec.bpm
    for idx, note in enumerate(spec.pulse_notes):
        start = idx * 2.0 * beat
        while start < spec.duration_s:
            add_mono(
                stereo,
                low_pulse(t, start, note_hz(note), amp=0.10, decay=1.15, click=0.12),
                pan=0.0,
            )
            start += len(spec.pulse_notes) * 2.0 * beat


def add_accents(stereo: np.ndarray, t: np.ndarray, spec: DemoSpec) -> None:
    beat = 60.0 / spec.bpm
    for idx, note in enumerate(spec.accent_notes):
        start = (7.0 + 8.0 * idx) * beat
        if start > spec.duration_s:
            break
        add_mono(
            stereo,
            low_pulse(t, start, note_hz(note), amp=0.18, decay=1.8, click=0.04),
            pan=(-0.25 if idx % 2 == 0 else 0.25),
        )


def add_lunar_breath(stereo: np.ndarray, t: np.ndarray, spec: DemoSpec) -> None:
    rng = np.random.default_rng(spec.seed)
    noise = rng.normal(0.0, 1.0, t.shape[0])
    kernel = np.hanning(2601)
    kernel /= np.sum(kernel)
    drift = np.convolve(noise, kernel, mode="same")
    drift /= max(float(np.max(np.abs(drift))), 1e-9)
    shimmer = np.sin(TAU * 0.047 * t + 0.8) * np.sin(TAU * 0.031 * t)
    bed = 0.023 * drift + 0.014 * shimmer
    if spec.color == "navigation":
        bed += 0.006 * np.sin(TAU * 1350.0 * t) * (0.5 + 0.5 * np.sin(TAU * 0.20 * t))
    elif spec.color == "approach":
        bed += 0.009 * np.sin(TAU * 410.0 * t + 0.4 * np.sin(TAU * 0.07 * t))
    add_mono(stereo, bed, pan=-0.62)
    add_mono(stereo, np.roll(bed, int(0.037 * SAMPLE_RATE)) * 0.8, pan=0.62)


def add_navigation_pings(stereo: np.ndarray, t: np.ndarray, spec: DemoSpec) -> None:
    if spec.color == "nocturne":
        return
    beat = 60.0 / spec.bpm
    spacing = 3.0 * beat if spec.color == "navigation" else 2.0 * beat
    start = 1.5 * beat
    count = 0
    while start < spec.duration_s:
        note = "A5" if count % 3 else "E6"
        amp = 0.028 if spec.color == "navigation" else 0.022
        pan = -0.7 if count % 2 == 0 else 0.7
        add_mono(stereo, pluck(t, start, note_hz(note), decay=1.0, tone=6.0, amp=amp), pan=pan)
        start += spacing
        count += 1


def add_reverb(stereo: np.ndarray) -> np.ndarray:
    wet = np.zeros_like(stereo)
    taps = (
        (0.117, 0.18, -0.45),
        (0.191, 0.13, 0.35),
        (0.283, 0.10, -0.2),
        (0.421, 0.08, 0.25),
    )
    for delay_s, gain, cross in taps:
        delay = int(delay_s * SAMPLE_RATE)
        if delay <= 0:
            continue
        wet[delay:, 0] += gain * ((1.0 - abs(cross)) * stereo[:-delay, 0] + abs(cross) * stereo[:-delay, 1])
        wet[delay:, 1] += gain * ((1.0 - abs(cross)) * stereo[:-delay, 1] + abs(cross) * stereo[:-delay, 0])
    return stereo + wet


def finish(stereo: np.ndarray, duration_s: float) -> np.ndarray:
    n = stereo.shape[0]
    fade = int(1.2 * SAMPLE_RATE)
    if fade > 0 and n > 2 * fade:
        ramp_in = np.linspace(0.0, 1.0, fade)
        ramp_out = np.linspace(1.0, 0.0, fade)
        stereo[:fade] *= ramp_in[:, None]
        stereo[-fade:] *= ramp_out[:, None]
    stereo = add_reverb(stereo)
    peak = max(float(np.max(np.abs(stereo))), 1e-9)
    stereo = stereo / peak * 0.88
    # A tiny loop-friendly fade keeps repeated playback from popping.
    loop_fade = int(0.08 * SAMPLE_RATE)
    stereo[:loop_fade] *= np.linspace(0.0, 1.0, loop_fade)[:, None]
    stereo[-loop_fade:] *= np.linspace(1.0, 0.0, loop_fade)[:, None]
    assert stereo.shape[0] == int(round(duration_s * SAMPLE_RATE))
    return stereo


def render(spec: DemoSpec) -> np.ndarray:
    t = timebase(spec.duration_s)
    stereo = np.zeros((t.shape[0], 2), dtype=np.float64)
    add_lunar_breath(stereo, t, spec)
    add_pad(stereo, t, spec)
    add_pulses(stereo, t, spec)
    add_bells(stereo, t, spec)
    add_accents(stereo, t, spec)
    add_navigation_pings(stereo, t, spec)
    return finish(stereo, spec.duration_s)


def write_wav(path: Path, stereo: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(stereo, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())


def specs() -> tuple[DemoSpec, ...]:
    return (
        DemoSpec(
            filename="29_lunar_ric_nocturne_demo.wav",
            bpm=82.0,
            duration_s=54.0,
            chords=(
                ("D2", "A2", "F3", "C4", "E4"),
                ("Bb1", "F2", "D3", "A3", "C4"),
                ("G2", "D3", "A3", "C4", "F4"),
                ("A1", "E2", "C3", "G3", "D4"),
                ("D2", "A2", "F3", "C4", "E4"),
            ),
            bell_pattern=((2.0, "A5", 0.052, -0.5), (5.5, "E6", 0.034, 0.42), (11.0, "D6", 0.038, 0.05)),
            pulse_notes=("D1", "D1", "A0", "C1"),
            accent_notes=("D1", "A0", "Bb0", "C1", "D1"),
            color="nocturne",
            seed=1301,
        ),
        DemoSpec(
            filename="30_far_side_navigation_demo.wav",
            bpm=88.0,
            duration_s=54.0,
            chords=(
                ("E2", "B2", "G3", "D4", "F#4"),
                ("C2", "G2", "E3", "B3", "D4"),
                ("A1", "E2", "C3", "G3", "B3"),
                ("B1", "F#2", "D3", "A3", "E4"),
                ("E2", "B2", "G3", "D4", "F#4"),
            ),
            bell_pattern=((1.0, "B5", 0.040, -0.65), (4.0, "F#6", 0.030, 0.58), (9.0, "E6", 0.034, -0.05)),
            pulse_notes=("E1", "B0", "E1", "G0"),
            accent_notes=("E1", "B0", "C1", "D1", "E1", "F#1"),
            color="navigation",
            seed=1302,
        ),
        DemoSpec(
            filename="31_perilune_approach_demo.wav",
            bpm=94.0,
            duration_s=54.0,
            chords=(
                ("F2", "C3", "Ab3", "Eb4", "G4"),
                ("Db2", "Ab2", "F3", "C4", "Eb4"),
                ("Bb1", "F2", "Db3", "Ab3", "C4"),
                ("C2", "G2", "Eb3", "Bb3", "F4"),
                ("F2", "C3", "Ab3", "Eb4", "G4"),
            ),
            bell_pattern=((0.5, "C6", 0.036, -0.35), (3.5, "G6", 0.027, 0.55), (6.5, "F6", 0.032, 0.1)),
            pulse_notes=("F1", "C1", "F1", "Eb1"),
            accent_notes=("F1", "C1", "Db1", "Eb1", "F1", "G1"),
            color="approach",
            seed=1303,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MUSIC_DIR)
    args = parser.parse_args()

    for spec in specs():
        stereo = render(spec)
        path = args.output_dir / spec.filename
        write_wav(path, stereo)
        rms = float(np.sqrt(np.mean(np.square(stereo))))
        peak = float(np.max(np.abs(stereo)))
        print(f"{path} duration={spec.duration_s:.1f}s peak={peak:.3f} rms={rms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
