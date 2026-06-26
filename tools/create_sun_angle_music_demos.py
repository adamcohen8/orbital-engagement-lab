#!/usr/bin/env python3
"""Create original procedural Sun-angle inspection music demos for the RPO trainer."""

from __future__ import annotations

import argparse
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = ROOT / "sim" / "game" / "music" / "unused"
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
class SunCueSpec:
    filename: str
    bpm: float
    duration_s: float
    chords: tuple[tuple[str, ...], ...]
    bass: tuple[str, ...]
    bells: tuple[tuple[float, str, float, float], ...]
    motif: tuple[tuple[float, str, float, float], ...]
    color: str
    seed: int


def note_hz(note: str) -> float:
    octave = int(note[-1])
    name = note[:-1]
    semitone = NOTE_OFFSETS[name] + (octave + 1) * 12
    return 440.0 * (2.0 ** ((semitone - 69) / 12.0))


def timebase(duration_s: float) -> np.ndarray:
    n = int(round(float(duration_s) * SAMPLE_RATE))
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
    attack = np.clip((t - start) / max(edge_s, 1.0e-6), 0.0, 1.0)
    release = np.clip((end - t) / max(edge_s, 1.0e-6), 0.0, 1.0)
    gate = np.minimum(attack, release)
    return gate * gate * (3.0 - 2.0 * gate)


def one_pole_lowpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    rc = 1.0 / (TAU * max(float(cutoff_hz), 1.0))
    dt = 1.0 / SAMPLE_RATE
    alpha = dt / (rc + dt)
    y = np.zeros_like(x)
    for idx in range(1, x.shape[0]):
        y[idx] = y[idx - 1] + alpha * (x[idx] - y[idx - 1])
    return y


def pluck(t: np.ndarray, start: float, freq: float, *, amp: float, decay: float, tone: float) -> np.ndarray:
    x = t - float(start)
    active = x >= 0.0
    out = np.zeros_like(t)
    local = x[active]
    env = np.exp(-local / max(float(decay), 1.0e-6))
    mod = tone * np.sin(TAU * freq * 2.01 * local) * np.exp(-local / (decay * 0.55))
    out[active] = amp * env * np.sin(TAU * freq * local + mod)
    return out


def warm_saw(t: np.ndarray, freq: float, phase: float) -> np.ndarray:
    return (
        0.82 * np.sin(TAU * freq * t + phase)
        + 0.24 * np.sin(TAU * 2.0 * freq * t + phase * 0.7)
        + 0.11 * np.sin(TAU * 3.0 * freq * t + phase * 1.4)
    )


def add_solar_bed(stereo: np.ndarray, t: np.ndarray, spec: SunCueSpec) -> None:
    rng = np.random.default_rng(spec.seed)
    noise = rng.normal(0.0, 1.0, t.shape[0])
    slow = one_pole_lowpass(noise, 8.0)
    slow /= max(float(np.max(np.abs(slow))), 1.0e-9)
    shimmer_noise = one_pole_lowpass(rng.normal(0.0, 1.0, t.shape[0]), 2400.0)
    shimmer_noise -= one_pole_lowpass(shimmer_noise, 500.0)
    shimmer_noise /= max(float(np.max(np.abs(shimmer_noise))), 1.0e-9)
    breathing = 0.5 + 0.5 * np.sin(TAU * 0.038 * t + 0.4)
    bed = 0.026 * slow + 0.010 * breathing * shimmer_noise
    if spec.color == "warning":
        bed += 0.010 * np.sin(TAU * 760.0 * t + 0.25 * np.sin(TAU * 0.08 * t)) * breathing
    elif spec.color == "radiant":
        bed += 0.008 * np.sin(TAU * 1320.0 * t) * (0.5 + 0.5 * np.sin(TAU * 0.11 * t))
    else:
        bed += 0.006 * np.sin(TAU * 410.0 * t + 0.5 * np.sin(TAU * 0.05 * t))
    add_mono(stereo, bed, pan=-0.58)
    add_mono(stereo, np.roll(bed, int(0.043 * SAMPLE_RATE)) * 0.78, pan=0.58)


def add_pads(stereo: np.ndarray, t: np.ndarray, spec: SunCueSpec) -> None:
    beat = 60.0 / spec.bpm
    chord_len = 8.0 * beat
    for idx, chord in enumerate(spec.chords):
        start = idx * chord_len
        end = min(start + chord_len + 3.0, spec.duration_s)
        gate = smooth_gate(t, start, end, edge_s=2.4)
        if not np.any(gate):
            continue
        for note_idx, note in enumerate(chord):
            base = note_hz(note)
            pan = -0.52 + 1.04 * note_idx / max(len(chord) - 1, 1)
            for cents, gain in ((-7.0, 0.28), (0.0, 0.42), (7.5, 0.25)):
                freq = base * (2.0 ** (cents / 1200.0))
                lfo = 0.0035 * np.sin(TAU * (0.030 + 0.004 * note_idx) * t + idx)
                voice = warm_saw(t, freq, lfo + 0.33 * note_idx)
                add_mono(stereo, 0.050 * gain * gate * voice, pan=pan)


def add_bass_pulse(stereo: np.ndarray, t: np.ndarray, spec: SunCueSpec) -> None:
    beat = 60.0 / spec.bpm
    for idx, note in enumerate(spec.bass):
        start = idx * 2.0 * beat
        while start < spec.duration_s:
            x = t - start
            active = x >= 0.0
            local = x[active]
            out = np.zeros_like(t)
            freq = note_hz(note)
            sweep = freq * (1.0 + 0.10 * np.exp(-local / 0.22))
            phase = TAU * np.cumsum(sweep) / SAMPLE_RATE
            env = np.exp(-local / 1.15)
            tick = 0.11 * np.exp(-local / 0.026) * np.sin(TAU * 980.0 * local)
            out[active] = 0.080 * env * (np.sin(phase) + 0.25 * np.sin(0.5 * phase)) + 0.080 * tick
            add_mono(stereo, out, pan=0.0)
            start += len(spec.bass) * 2.0 * beat


def add_bells(stereo: np.ndarray, t: np.ndarray, spec: SunCueSpec) -> None:
    beat = 60.0 / spec.bpm
    for beat_offset, note, amp, pan in spec.bells:
        start = beat_offset * beat
        while start < spec.duration_s - 0.8:
            add_mono(stereo, pluck(t, start, note_hz(note), amp=amp, decay=2.9, tone=4.7), pan=pan)
            start += 16.0 * beat


def add_motif(stereo: np.ndarray, t: np.ndarray, spec: SunCueSpec) -> None:
    beat = 60.0 / spec.bpm
    phrase = 16.0 * beat
    phrase_start = 8.0 * beat
    while phrase_start < spec.duration_s - 1.0:
        for beat_offset, note, amp, pan in spec.motif:
            start = phrase_start + beat_offset * beat
            if start < spec.duration_s - 0.4:
                add_mono(stereo, pluck(t, start, note_hz(note), amp=amp, decay=1.45, tone=3.2), pan=pan)
        phrase_start += phrase


def add_reverb(stereo: np.ndarray) -> np.ndarray:
    wet = np.zeros_like(stereo)
    taps = (
        (0.103, 0.16, -0.45),
        (0.177, 0.13, 0.30),
        (0.293, 0.10, -0.25),
        (0.439, 0.075, 0.40),
        (0.617, 0.055, -0.15),
    )
    for delay_s, gain, cross in taps:
        delay = int(delay_s * SAMPLE_RATE)
        wet[delay:, 0] += gain * ((1.0 - abs(cross)) * stereo[:-delay, 0] + abs(cross) * stereo[:-delay, 1])
        wet[delay:, 1] += gain * ((1.0 - abs(cross)) * stereo[:-delay, 1] + abs(cross) * stereo[:-delay, 0])
    return stereo + wet


def finish(stereo: np.ndarray, duration_s: float) -> np.ndarray:
    fade = int(1.5 * SAMPLE_RATE)
    if stereo.shape[0] > 2 * fade:
        stereo[:fade] *= np.linspace(0.0, 1.0, fade)[:, None]
        stereo[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    stereo = add_reverb(stereo)
    peak = max(float(np.max(np.abs(stereo))), 1.0e-9)
    stereo = stereo / peak * 0.86
    loop_fade = int(0.08 * SAMPLE_RATE)
    stereo[:loop_fade] *= np.linspace(0.0, 1.0, loop_fade)[:, None]
    stereo[-loop_fade:] *= np.linspace(1.0, 0.0, loop_fade)[:, None]
    assert stereo.shape[0] == int(round(duration_s * SAMPLE_RATE))
    return stereo


def render(spec: SunCueSpec) -> np.ndarray:
    t = timebase(spec.duration_s)
    stereo = np.zeros((t.shape[0], 2), dtype=np.float64)
    add_solar_bed(stereo, t, spec)
    add_pads(stereo, t, spec)
    add_bass_pulse(stereo, t, spec)
    add_bells(stereo, t, spec)
    add_motif(stereo, t, spec)
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


def specs() -> tuple[SunCueSpec, ...]:
    return (
        SunCueSpec(
            filename="32_sunline_hold_demo.wav",
            bpm=76.0,
            duration_s=58.0,
            chords=(
                ("E2", "B2", "F#3", "A3", "D4"),
                ("C#2", "G#2", "E3", "B3", "F#4"),
                ("A1", "E2", "C#3", "G#3", "B3"),
                ("B1", "F#2", "D3", "A3", "E4"),
                ("E2", "B2", "F#3", "A3", "D4"),
            ),
            bass=("E1", "B0", "E1", "F#1"),
            bells=((2.0, "B5", 0.033, -0.62), (7.0, "F#6", 0.026, 0.46), (11.0, "E6", 0.024, 0.10)),
            motif=((0.0, "E5", 0.026, -0.25), (1.5, "F#5", 0.020, 0.18), (3.0, "B5", 0.023, 0.36)),
            color="precision",
            seed=3201,
        ),
        SunCueSpec(
            filename="33_amber_terminator_demo.wav",
            bpm=82.0,
            duration_s=58.0,
            chords=(
                ("D2", "A2", "E3", "G3", "C4"),
                ("Bb1", "F2", "D3", "A3", "E4"),
                ("G1", "D2", "Bb2", "F3", "A3"),
                ("A1", "E2", "C3", "G3", "D4"),
                ("D2", "A2", "E3", "G3", "C4"),
            ),
            bass=("D1", "A0", "D1", "C1"),
            bells=((1.0, "A5", 0.036, -0.48), (5.0, "E6", 0.028, 0.52), (9.5, "D6", 0.024, -0.05)),
            motif=((0.0, "D5", 0.027, -0.38), (1.0, "E5", 0.019, 0.28), (2.5, "A5", 0.023, 0.42)),
            color="radiant",
            seed=3301,
        ),
        SunCueSpec(
            filename="34_geo_penumbra_drift_demo.wav",
            bpm=70.0,
            duration_s=58.0,
            chords=(
                ("F#2", "C#3", "E3", "A3", "B3"),
                ("D2", "A2", "F#3", "C#4", "E4"),
                ("B1", "F#2", "D3", "A3", "C#4"),
                ("C#2", "G#2", "E3", "B3", "F#4"),
                ("F#2", "C#3", "E3", "A3", "B3"),
            ),
            bass=("F#1", "C#1", "F#1", "E1"),
            bells=((3.0, "C#6", 0.030, -0.54), (6.0, "A5", 0.024, 0.36), (12.0, "F#6", 0.020, 0.05)),
            motif=((0.0, "F#5", 0.021, -0.22), (2.0, "A5", 0.018, 0.20), (4.0, "C#6", 0.020, 0.43)),
            color="warning",
            seed=3401,
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
