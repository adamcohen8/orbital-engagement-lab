#!/usr/bin/env python3
"""Create an original procedural electro-disco groove layer for the RPO trainer."""

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


@dataclass(frozen=True)
class DrumLayerSpec:
    filename: str
    bpm: float = 124.0
    bars: int = 32
    seed: int = 3801

    @property
    def beat_s(self) -> float:
        return 60.0 / float(self.bpm)

    @property
    def duration_s(self) -> float:
        return float(self.bars) * 4.0 * self.beat_s


def timebase(duration_s: float) -> np.ndarray:
    n = int(round(float(duration_s) * SAMPLE_RATE))
    return np.arange(n, dtype=np.float64) / SAMPLE_RATE


def equal_power_pan(pan: float) -> tuple[float, float]:
    p = float(np.clip(pan, -1.0, 1.0))
    angle = (p + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def add_segment(stereo: np.ndarray, mono: np.ndarray, start_s: float, pan: float = 0.0) -> None:
    start = int(round(float(start_s) * SAMPLE_RATE))
    if start >= stereo.shape[0] or mono.size == 0:
        return
    end = min(start + mono.size, stereo.shape[0])
    if end <= start:
        return
    left, right = equal_power_pan(pan)
    segment = mono[: end - start]
    stereo[start:end, 0] += segment * left
    stereo[start:end, 1] += segment * right


def add_stereo_segment(stereo: np.ndarray, segment: np.ndarray, start_s: float) -> None:
    start = int(round(float(start_s) * SAMPLE_RATE))
    if start >= stereo.shape[0] or segment.size == 0:
        return
    end = min(start + segment.shape[0], stereo.shape[0])
    if end <= start:
        return
    stereo[start:end] += segment[: end - start]


def one_pole_lowpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    rc = 1.0 / (TAU * max(float(cutoff_hz), 1.0))
    dt = 1.0 / SAMPLE_RATE
    alpha = dt / (rc + dt)
    y = np.zeros_like(x)
    for idx in range(1, x.shape[0]):
        y[idx] = y[idx - 1] + alpha * (x[idx] - y[idx - 1])
    return y


def highpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    return x - one_pole_lowpass(x, cutoff_hz)


def bandpass_noise(rng: np.random.Generator, n: int, low_hz: float, high_hz: float) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, n)
    shaped = highpass(one_pole_lowpass(noise, high_hz), low_hz)
    peak = max(float(np.max(np.abs(shaped))), 1.0e-9)
    return shaped / peak


def decay_env(n: int, decay_s: float, attack_s: float = 0.001) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    env = np.exp(-t / max(float(decay_s), 1.0e-6))
    if attack_s > 0.0:
        attack_n = max(int(round(float(attack_s) * SAMPLE_RATE)), 1)
        env[:attack_n] *= np.linspace(0.0, 1.0, attack_n)
    return env


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


def note_hz(note: str) -> float:
    octave = int(note[-1])
    name = note[:-1]
    semitone = NOTE_OFFSETS[name] + (octave + 1) * 12
    return 440.0 * (2.0 ** ((semitone - 69) / 12.0))


def kick(gain: float = 1.0) -> np.ndarray:
    n = int(round(0.72 * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    sweep = 42.0 + 105.0 * np.exp(-t / 0.055) + 18.0 * np.exp(-t / 0.22)
    phase = TAU * np.cumsum(sweep) / SAMPLE_RATE
    body = np.sin(phase)
    sub = 0.38 * np.sin(0.5 * phase + 0.2)
    punch = np.sin(TAU * 880.0 * t) * np.exp(-t / 0.010)
    thump = np.sin(TAU * 58.0 * t) * np.exp(-t / 0.34)
    env = np.exp(-t / 0.31)
    out = gain * (0.94 * env * body + 0.28 * thump + 0.09 * punch + 0.16 * env * sub)
    return np.tanh(1.55 * out)


def clap(rng: np.random.Generator, gain: float = 1.0) -> np.ndarray:
    n = int(round(0.42 * SAMPLE_RATE))
    out = np.zeros(n, dtype=np.float64)
    for offset_s, amp in ((0.000, 0.64), (0.014, 0.78), (0.029, 0.92), (0.052, 0.58)):
        start = int(round(offset_s * SAMPLE_RATE))
        length = min(n - start, int(round(0.20 * SAMPLE_RATE)))
        burst = bandpass_noise(rng, length, 780.0, 7600.0)
        env = decay_env(length, 0.072, attack_s=0.0004)
        out[start : start + length] += amp * burst * env
    body = bandpass_noise(rng, n, 1500.0, 4800.0) * decay_env(n, 0.21, attack_s=0.001)
    out += 0.20 * body
    return gain * np.tanh(1.7 * out)


def snare_ghost(rng: np.random.Generator, gain: float = 1.0) -> np.ndarray:
    n = int(round(0.20 * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    snap = bandpass_noise(rng, n, 1100.0, 8200.0) * decay_env(n, 0.045, attack_s=0.0005)
    tone = 0.13 * np.sin(TAU * 192.0 * t) * np.exp(-t / 0.12)
    return gain * np.tanh(1.25 * (0.74 * snap + tone))


def closed_hat(rng: np.random.Generator, gain: float = 1.0) -> np.ndarray:
    n = int(round(0.105 * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    metallic = (
        0.26 * np.sin(TAU * 6240.0 * t)
        + 0.20 * np.sin(TAU * 8110.0 * t + 0.5)
        + 0.16 * np.sin(TAU * 10_240.0 * t + 1.1)
    )
    noise = bandpass_noise(rng, n, 5600.0, 14_000.0)
    env = decay_env(n, 0.031, attack_s=0.0005)
    return gain * np.tanh(1.45 * (0.60 * noise + metallic) * env)


def open_hat(rng: np.random.Generator, gain: float = 1.0) -> np.ndarray:
    n = int(round(0.43 * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    metallic = (
        0.20 * np.sin(TAU * 5100.0 * t + 0.2)
        + 0.18 * np.sin(TAU * 7350.0 * t + 1.0)
        + 0.14 * np.sin(TAU * 9680.0 * t + 2.1)
    )
    noise = bandpass_noise(rng, n, 4500.0, 13_000.0)
    env = decay_env(n, 0.18, attack_s=0.001)
    return gain * np.tanh(1.25 * (0.62 * noise + metallic) * env)


def shaker(rng: np.random.Generator, gain: float = 1.0) -> np.ndarray:
    n = int(round(0.075 * SAMPLE_RATE))
    noise = bandpass_noise(rng, n, 7000.0, 15_000.0)
    env = decay_env(n, 0.025, attack_s=0.0008)
    return gain * noise * env


def tom(gain: float = 1.0, freq_hz: float = 126.0) -> np.ndarray:
    n = int(round(0.34 * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    sweep = float(freq_hz) * (1.0 + 0.35 * np.exp(-t / 0.055))
    phase = TAU * np.cumsum(sweep) / SAMPLE_RATE
    env = decay_env(n, 0.16, attack_s=0.002)
    return gain * np.tanh(1.4 * env * (np.sin(phase) + 0.22 * np.sin(2.0 * phase)))


def synth_bass_note(freq_hz: float, duration_s: float, gain: float = 1.0) -> np.ndarray:
    n = int(round(float(duration_s) * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    env = np.minimum(t / 0.014, 1.0) * np.exp(-t / max(float(duration_s) * 0.78, 1.0e-6))
    release_n = max(int(round(0.030 * SAMPLE_RATE)), 1)
    if n > release_n:
        env[-release_n:] *= np.linspace(1.0, 0.0, release_n)
    vibrato = 0.003 * np.sin(TAU * 5.1 * t)
    phase = TAU * float(freq_hz) * (1.0 + vibrato) * t
    core = (
        0.74 * np.sin(phase)
        + 0.42 * np.sin(2.0 * phase + 0.24)
        + 0.20 * np.sin(3.0 * phase + 0.61)
        + 0.12 * np.sin(4.0 * phase + 0.18)
    )
    bite = highpass(np.tanh(2.5 * core), 95.0)
    rounded = one_pole_lowpass(0.58 * core + 0.42 * bite, 820.0)
    return gain * np.tanh(1.9 * rounded * env)


def add_bass_pattern(stereo: np.ndarray, spec: DrumLayerSpec) -> None:
    beat = spec.beat_s
    patterns = (
        ("F#1", ((0.25, "F#1", 0.55, 0.38), (0.75, "F#2", 0.26, 0.26), (1.50, "F#1", 0.44, 0.34), (2.25, "C#2", 0.30, 0.24), (2.75, "F#1", 0.42, 0.30), (3.50, "E2", 0.30, 0.22))),
        ("A1", ((0.25, "A1", 0.52, 0.36), (0.75, "E2", 0.28, 0.24), (1.50, "A1", 0.42, 0.32), (2.00, "A1", 0.24, 0.18), (2.75, "G1", 0.38, 0.28), (3.50, "E2", 0.28, 0.20))),
        ("E1", ((0.25, "E1", 0.56, 0.38), (1.00, "B1", 0.30, 0.24), (1.50, "E1", 0.42, 0.32), (2.25, "E2", 0.24, 0.20), (2.75, "D2", 0.30, 0.22), (3.50, "B1", 0.30, 0.22))),
        ("B0", ((0.25, "B0", 0.60, 0.40), (0.75, "B1", 0.28, 0.24), (1.50, "B0", 0.46, 0.34), (2.25, "F#1", 0.34, 0.24), (2.75, "A1", 0.31, 0.22), (3.50, "C#2", 0.30, 0.20))),
    )
    for bar in range(spec.bars):
        bar_start = bar * 4.0 * beat
        _root, pattern = patterns[(bar // 8) % len(patterns)]
        section_gain = 0.58 + 0.045 * min(bar // 8, 3)
        for beat_offset, note, note_gain, note_beats in pattern:
            if bar % 8 == 7 and beat_offset >= 3.0:
                continue
            start = bar_start + beat_offset * beat
            length = note_beats * beat
            add_segment(stereo, synth_bass_note(note_hz(note), length, section_gain * note_gain), start, pan=0.0)
        if bar % 8 == 7:
            add_bass_turnaround(stereo, spec, bar_start, section_gain)


def add_bass_turnaround(stereo: np.ndarray, spec: DrumLayerSpec, bar_start: float, gain: float) -> None:
    beat = spec.beat_s
    sixteenth = beat / 4.0
    notes = ("C#2", "E2", "F#2", "E2", "C#2")
    for idx, note in enumerate(notes):
        start = bar_start + 3.0 * beat + idx * 0.75 * sixteenth
        add_segment(stereo, synth_bass_note(note_hz(note), 0.42 * sixteenth, gain * (0.22 + 0.02 * idx)), start)


def chord_voice(freq_hz: float, duration_s: float, gain: float, phase_offset: float) -> np.ndarray:
    n = int(round(float(duration_s) * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    attack = np.minimum(t / 0.022, 1.0)
    decay = np.exp(-t / 0.29)
    release_n = max(int(round(0.045 * SAMPLE_RATE)), 1)
    env = attack * decay
    if n > release_n:
        env[-release_n:] *= np.linspace(1.0, 0.0, release_n)
    wobble = 0.0025 * np.sin(TAU * 4.7 * t + phase_offset)
    phase = TAU * float(freq_hz) * (1.0 + wobble) * t + phase_offset
    voice = (
        0.62 * np.sin(phase)
        + 0.28 * np.sin(2.0 * phase + 0.4)
        + 0.13 * np.sin(3.0 * phase + 1.1)
        + 0.08 * np.sin(5.0 * phase + 0.2)
    )
    return gain * voice * env


def filtered_chord_stab(notes: tuple[str, ...], duration_s: float, *, gain: float, cutoff_hz: float) -> np.ndarray:
    n = int(round(float(duration_s) * SAMPLE_RATE))
    stereo = np.zeros((n, 2), dtype=np.float64)
    for idx, note in enumerate(notes):
        pan = -0.55 + 1.10 * idx / max(len(notes) - 1, 1)
        left, right = equal_power_pan(pan)
        detune = (-5.0 if idx % 2 == 0 else 4.0) / 1200.0
        freq = note_hz(note) * (2.0**detune)
        voice = chord_voice(freq, duration_s, gain / max(len(notes), 1), phase_offset=0.37 * idx)
        stereo[:, 0] += voice * left
        stereo[:, 1] += voice * right
    for channel in range(2):
        stereo[:, channel] = highpass(one_pole_lowpass(stereo[:, channel], cutoff_hz), 135.0)
    return np.tanh(1.55 * stereo)


def add_chord_stabs(stereo: np.ndarray, spec: DrumLayerSpec) -> None:
    beat = spec.beat_s
    sixteenth = beat / 4.0
    chord_sets = (
        ("F#2", "A2", "C#3", "E3", "G#3"),
        ("A2", "C#3", "E3", "F#3", "B3"),
        ("E2", "G#2", "B2", "D3", "F#3"),
        ("B1", "F#2", "A2", "C#3", "E3"),
    )
    hits = (
        (0.50, 0.78),
        (1.75, 0.58),
        (2.50, 0.70),
        (3.25, 0.48),
    )
    for bar in range(spec.bars):
        if bar < 4:
            continue
        bar_start = bar * 4.0 * beat
        chord = chord_sets[(bar // 8) % len(chord_sets)]
        section = min(bar // 8, 3)
        cutoff = 720.0 + 150.0 * section + (70.0 if bar % 8 >= 4 else 0.0)
        base_gain = 0.118 + 0.012 * section
        if bar % 8 == 7:
            local_hits = ((0.50, 0.64), (1.50, 0.52), (2.50, 0.60), (3.00, 0.42), (3.50, 0.38))
        else:
            local_hits = hits
        for beat_offset, accent in local_hits:
            start = bar_start + beat_offset * beat
            duration = 2.15 * sixteenth if beat_offset < 3.0 else 1.45 * sixteenth
            add_stereo_segment(
                stereo,
                filtered_chord_stab(chord, duration, gain=base_gain * accent, cutoff_hz=cutoff),
                start,
            )


def robotic_pluck(freq_hz: float, duration_s: float, *, gain: float, brightness: float) -> np.ndarray:
    n = int(round(float(duration_s) * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    attack = np.minimum(t / 0.006, 1.0)
    decay = np.exp(-t / 0.18)
    env = attack * decay
    release_n = max(int(round(0.025 * SAMPLE_RATE)), 1)
    if n > release_n:
        env[-release_n:] *= np.linspace(1.0, 0.0, release_n)
    glide = 1.0 + 0.018 * np.exp(-t / 0.050)
    phase = TAU * float(freq_hz) * glide * t
    carrier = (
        0.68 * np.sin(phase)
        + 0.26 * np.sin(2.0 * phase + 0.7)
        + 0.14 * np.sin(4.0 * phase + 1.4)
    )
    formant = 0.5 + 0.5 * np.sin(TAU * (780.0 + 260.0 * brightness) * t + 0.55 * np.sin(TAU * 7.0 * t))
    sparkle = 0.26 * np.sin(TAU * (float(freq_hz) * 7.01) * t + 0.2) * np.exp(-t / 0.085)
    octave_glint = 0.16 * np.sin(TAU * (float(freq_hz) * 2.0) * t + 1.1) * np.exp(-t / 0.16)
    tone = (0.78 * carrier * (0.70 + 0.42 * formant)) + brightness * sparkle + octave_glint
    tone = highpass(one_pole_lowpass(tone, 4200.0 + 2200.0 * brightness), 500.0)
    return gain * np.tanh(1.8 * tone * env)


def add_ping_delay(stereo: np.ndarray, source: np.ndarray, start_s: float, pan: float, repeats: int = 2) -> None:
    for repeat in range(1, repeats + 1):
        delay = 0.115 * repeat
        gain = 0.34 / repeat
        add_segment(stereo, source * gain, start_s + delay, pan=-pan * 0.8)


def add_hook_texture(stereo: np.ndarray, spec: DrumLayerSpec) -> None:
    beat = spec.beat_s
    motifs = (
        (("C#5", "E5", "F#5"), (0.00, 0.75, 1.50)),
        (("E5", "F#5", "C#6"), (0.00, 0.50, 1.75)),
        (("B4", "D5", "F#5"), (0.00, 0.75, 1.25)),
        (("C#5", "A5", "G#5", "F#5"), (0.00, 0.50, 1.00, 1.75)),
    )
    for bar in range(spec.bars):
        if bar < 8:
            continue
        bar_start = bar * 4.0 * beat
        section = min(bar // 8, 3)
        if section == 1 and bar % 2 == 1:
            continue
        motif_notes, offsets = motifs[(bar // 4) % len(motifs)]
        brightness = 0.62 + 0.12 * section
        gain = 0.135 + 0.024 * section
        if bar % 8 == 7:
            motif_notes = ("C#6", "B5", "A5", "F#5")
            offsets = (0.00, 0.375, 0.75, 1.50)
            gain *= 1.05
        for idx, (note, beat_offset) in enumerate(zip(motif_notes, offsets)):
            start = bar_start + (2.0 + beat_offset) * beat
            duration = 0.58 * beat if idx < len(motif_notes) - 1 else 0.82 * beat
            pluck = robotic_pluck(note_hz(note), duration, gain=gain * (1.0 - 0.08 * idx), brightness=brightness)
            pan = -0.42 if idx % 2 == 0 else 0.45
            add_segment(stereo, pluck, start, pan=pan)
            if section >= 1:
                add_ping_delay(stereo, pluck, start, pan=pan, repeats=2 if section >= 3 else 1)


def lead_synth_note(freq_hz: float, duration_s: float, *, gain: float, brightness: float) -> np.ndarray:
    n = int(round(float(duration_s) * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    attack = np.minimum(t / 0.018, 1.0)
    hold = np.exp(-t / max(float(duration_s) * 1.8, 1.0e-6))
    release_n = max(int(round(0.060 * SAMPLE_RATE)), 1)
    env = attack * hold
    if n > release_n:
        env[-release_n:] *= np.linspace(1.0, 0.0, release_n)
    pulse_width = 0.48 + 0.07 * np.sin(TAU * 3.0 * t)
    phase = (float(freq_hz) * t) % 1.0
    pulse = np.where(phase < pulse_width, 1.0, -1.0)
    rounded_pulse = one_pole_lowpass(pulse, 1700.0 + 900.0 * brightness)
    sawish = (
        0.58 * np.sin(TAU * float(freq_hz) * t)
        + 0.22 * np.sin(TAU * 2.0 * float(freq_hz) * t + 0.2)
        + 0.12 * np.sin(TAU * 3.0 * float(freq_hz) * t + 1.0)
    )
    slow_vowel = 0.75 + 0.25 * np.sin(TAU * 5.2 * t + 0.8)
    tone = 0.56 * rounded_pulse + 0.44 * sawish * slow_vowel
    tone = highpass(one_pole_lowpass(tone, 3200.0 + 1800.0 * brightness), 260.0)
    return gain * np.tanh(1.65 * tone * env)


def anthem_lead_note(freq_hz: float, duration_s: float, *, gain: float) -> np.ndarray:
    n = int(round(float(duration_s) * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    attack = np.minimum(t / 0.035, 1.0)
    release_n = max(int(round(0.090 * SAMPLE_RATE)), 1)
    env = attack.copy()
    if n > release_n:
        env[-release_n:] *= np.linspace(1.0, 0.0, release_n)
    vibrato = 0.0045 * np.sin(TAU * 5.8 * t) * np.clip((t - 0.10) / 0.35, 0.0, 1.0)
    detunes = (-7.0, 0.0, 6.0)
    tone = np.zeros_like(t)
    for idx, cents in enumerate(detunes):
        freq = float(freq_hz) * (2.0 ** (cents / 1200.0)) * (1.0 + vibrato)
        phase = TAU * freq * t + 0.4 * idx
        tone += (
            0.42 * np.sin(phase)
            + 0.22 * np.sin(2.0 * phase + 0.3)
            + 0.09 * np.sin(3.0 * phase + 1.1)
        )
    vowel = 0.78 + 0.22 * np.sin(TAU * 3.9 * t + 0.5)
    tone = tone * vowel / len(detunes)
    tone = highpass(one_pole_lowpass(tone, 5600.0), 220.0)
    return gain * np.tanh(1.35 * tone * env)


def foreground_melody_note(freq_hz: float, duration_s: float, *, gain: float) -> np.ndarray:
    n = int(round(float(duration_s) * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    attack = np.minimum(t / 0.018, 1.0)
    release_n = max(int(round(0.080 * SAMPLE_RATE)), 1)
    env = attack * np.exp(-t / max(float(duration_s) * 3.8, 1.0e-6))
    if n > release_n:
        env[-release_n:] *= np.linspace(1.0, 0.0, release_n)

    vibrato = 0.0038 * np.sin(TAU * 5.6 * t) * np.clip((t - 0.08) / 0.22, 0.0, 1.0)
    tone = np.zeros_like(t)
    for cents, weight, phase_offset in ((-9.0, 0.30, 0.0), (0.0, 0.46, 0.5), (7.0, 0.28, 1.1)):
        freq = float(freq_hz) * (2.0 ** (cents / 1200.0)) * (1.0 + vibrato)
        phase = TAU * freq * t + phase_offset
        square = np.where((freq * t + phase_offset / TAU) % 1.0 < 0.48, 1.0, -1.0)
        rounded_square = one_pole_lowpass(square, 2400.0)
        tone += weight * (
            0.48 * rounded_square
            + 0.34 * np.sin(phase)
            + 0.14 * np.sin(2.0 * phase + 0.4)
            + 0.08 * np.sin(3.0 * phase + 0.9)
        )

    vowel = 0.82 + 0.18 * np.sin(TAU * 2.8 * t + 0.4)
    tone = tone * vowel
    tone = highpass(one_pole_lowpass(tone, 7200.0), 180.0)
    return gain * np.tanh(1.55 * tone * env)


def add_foreground_melody(stereo: np.ndarray, spec: DrumLayerSpec) -> None:
    beat = spec.beat_s
    # A deliberately obvious 8-bar theme. This is meant to read as the song's
    # tune, not as a tucked-in production texture.
    theme = (
        (
            (0.00, "F#5", 0.75),
            (0.75, "A5", 0.75),
            (1.50, "C#6", 1.00),
            (2.50, "B5", 0.50),
            (3.00, "A5", 1.00),
        ),
        (
            (0.00, "E5", 0.75),
            (0.75, "F#5", 0.75),
            (1.50, "A5", 1.00),
            (2.50, "G#5", 0.50),
            (3.00, "F#5", 1.00),
        ),
        (
            (0.00, "C#6", 0.75),
            (0.75, "B5", 0.75),
            (1.50, "A5", 1.00),
            (2.50, "F#5", 0.50),
            (3.00, "E5", 1.00),
        ),
        (
            (0.00, "G#5", 0.50),
            (0.50, "A5", 0.50),
            (1.00, "B5", 0.50),
            (1.50, "C#6", 0.75),
            (2.25, "B5", 0.50),
            (2.75, "A5", 1.25),
        ),
        (
            (0.00, "F#5", 0.75),
            (0.75, "A5", 0.75),
            (1.50, "C#6", 1.00),
            (2.50, "E6", 0.50),
            (3.00, "C#6", 1.00),
        ),
        (
            (0.00, "B5", 0.75),
            (0.75, "A5", 0.75),
            (1.50, "F#5", 1.00),
            (2.50, "E5", 0.50),
            (3.00, "F#5", 1.00),
        ),
        (
            (0.00, "A5", 0.75),
            (0.75, "C#6", 0.75),
            (1.50, "E6", 1.00),
            (2.50, "C#6", 0.50),
            (3.00, "B5", 1.00),
        ),
        (
            (0.00, "C#6", 0.50),
            (0.50, "B5", 0.50),
            (1.00, "A5", 0.50),
            (1.50, "G#5", 0.50),
            (2.00, "F#5", 2.00),
        ),
    )
    for bar in range(spec.bars):
        bar_start = bar * 4.0 * beat
        phrase = theme[bar % len(theme)]
        section = min(bar // 8, 3)
        gain = 0.86 + 0.05 * section
        for idx, (beat_offset, note, note_beats) in enumerate(phrase):
            start = bar_start + beat_offset * beat
            duration = note_beats * beat
            lead = foreground_melody_note(note_hz(note), duration, gain=gain * (0.92 if idx % 2 else 1.0))
            pan = 0.0 if idx % 3 == 0 else (0.10 if idx % 3 == 1 else -0.10)
            add_segment(stereo, lead, start, pan=pan)
            add_segment(stereo, lead * 0.13, start + 0.118, pan=-0.46)
            if section >= 2:
                add_segment(stereo, lead * 0.10, start + 0.236, pan=0.48)


def backing_arrangement_gain(spec: DrumLayerSpec) -> np.ndarray:
    n = int(round(spec.duration_s * SAMPLE_RATE))
    gain = np.full(n, 0.30, dtype=np.float64)
    beat = spec.beat_s
    intro_end = int(round(8.0 * 4.0 * beat * SAMPLE_RATE))
    ramp_end = int(round(10.0 * 4.0 * beat * SAMPLE_RATE))
    gain[:intro_end] = 0.035
    if ramp_end > intro_end:
        gain[intro_end:ramp_end] = np.linspace(0.035, 0.30, ramp_end - intro_end)
    return gain


def add_anthem_melody(stereo: np.ndarray, spec: DrumLayerSpec) -> None:
    beat = spec.beat_s
    phrases = (
        (
            (0.00, "F#5", 0.75),
            (0.75, "A5", 0.75),
            (1.50, "C#6", 1.00),
            (2.50, "B5", 0.50),
            (3.00, "A5", 1.00),
        ),
        (
            (0.00, "E5", 0.75),
            (0.75, "F#5", 0.75),
            (1.50, "A5", 1.00),
            (2.50, "G#5", 0.50),
            (3.00, "F#5", 1.00),
        ),
        (
            (0.00, "B5", 0.75),
            (0.75, "A5", 0.75),
            (1.50, "G#5", 1.00),
            (2.50, "E5", 0.50),
            (3.00, "F#5", 1.00),
        ),
        (
            (0.00, "C#6", 0.75),
            (0.75, "B5", 0.75),
            (1.50, "A5", 0.75),
            (2.25, "G#5", 0.50),
            (2.75, "F#5", 1.25),
        ),
    )
    for bar in range(spec.bars):
        bar_start = bar * 4.0 * beat
        phrase = phrases[bar % len(phrases)]
        section = min(bar // 8, 3)
        gain = 0.42 + 0.045 * section
        if bar % 8 == 7:
            gain *= 1.12
        for idx, (beat_offset, note, note_beats) in enumerate(phrase):
            start = bar_start + beat_offset * beat
            duration = note_beats * beat
            lead = anthem_lead_note(note_hz(note), duration, gain=gain * (0.94 if idx in (1, 3) else 1.0))
            pan = 0.03 if idx % 2 == 0 else -0.03
            add_segment(stereo, lead, start, pan=pan)
            if section >= 1:
                add_segment(stereo, lead * 0.16, start + 0.125, pan=-0.42)
            if section >= 3:
                add_segment(stereo, lead * 0.12, start + 0.245, pan=0.48)


def add_lead_melody(stereo: np.ndarray, spec: DrumLayerSpec) -> None:
    beat = spec.beat_s
    motifs = (
        (
            (0.00, "C#5", 0.75),
            (0.75, "E5", 0.50),
            (1.25, "F#5", 0.75),
            (2.25, "E5", 0.50),
            (2.75, "C#5", 0.75),
            (3.50, "A4", 0.50),
        ),
        (
            (0.00, "E5", 0.75),
            (0.75, "F#5", 0.50),
            (1.25, "A5", 0.75),
            (2.25, "F#5", 0.50),
            (2.75, "E5", 0.75),
            (3.50, "C#5", 0.50),
        ),
        (
            (0.00, "B4", 0.75),
            (0.75, "D5", 0.50),
            (1.25, "E5", 0.75),
            (2.25, "D5", 0.50),
            (2.75, "B4", 0.75),
            (3.50, "G#4", 0.50),
        ),
        (
            (0.00, "C#5", 0.50),
            (0.50, "E5", 0.50),
            (1.00, "F#5", 0.75),
            (2.00, "A5", 0.50),
            (2.50, "G#5", 0.50),
            (3.00, "F#5", 0.90),
        ),
    )
    for bar in range(spec.bars):
        if bar < 8:
            continue
        section = min(bar // 8, 3)
        if section == 1 and bar % 2 == 1:
            continue
        bar_start = bar * 4.0 * beat
        motif = motifs[(bar // 4) % len(motifs)]
        brightness = 0.48 + 0.13 * section
        gain = 0.145 + 0.030 * section
        if bar % 8 == 7:
            gain *= 1.10
        for idx, (beat_offset, note, note_beats) in enumerate(motif):
            start = bar_start + beat_offset * beat
            duration = note_beats * beat
            lead = lead_synth_note(note_hz(note), duration, gain=gain * (0.95 if idx % 2 else 1.0), brightness=brightness)
            pan = 0.12 if idx % 2 == 0 else -0.08
            add_segment(stereo, lead, start, pan=pan)
            if section >= 2:
                add_segment(stereo, lead * 0.20, start + 0.145, pan=-0.38)


def add_main_pattern(stereo: np.ndarray, spec: DrumLayerSpec, rng: np.random.Generator) -> None:
    beat = spec.beat_s
    sixteenth = beat / 4.0
    swing = 0.018
    for bar in range(spec.bars):
        bar_start = bar * 4.0 * beat
        section = bar // 8
        for beat_idx in range(4):
            accent = 1.0 if beat_idx == 0 else 0.91
            if bar % 8 == 7 and beat_idx == 3:
                accent = 0.76
            add_segment(stereo, kick(0.85 * accent), bar_start + beat_idx * beat)
        for beat_idx in (1, 3):
            add_segment(stereo, clap(rng, 0.56 + 0.04 * section), bar_start + beat_idx * beat, pan=0.04)
            add_segment(stereo, snare_ghost(rng, 0.13), bar_start + beat_idx * beat - 0.135, pan=-0.18)
        for step in range(16):
            step_time = bar_start + step * sixteenth
            if step % 2 == 1:
                step_time += swing
            vel = 0.14 if step % 4 in (1, 3) else 0.09
            if step in (3, 7, 11, 15):
                vel += 0.035
            add_segment(stereo, closed_hat(rng, vel), step_time, pan=-0.25 if step % 2 == 0 else 0.28)
            if section >= 1 and step % 2 == 0:
                add_segment(stereo, shaker(rng, 0.038), step_time + 0.020, pan=0.42)
        for beat_idx in range(4):
            add_segment(stereo, open_hat(rng, 0.19), bar_start + beat_idx * beat + 0.5 * beat + swing, pan=0.22)
        if bar % 4 in (1, 3):
            add_segment(stereo, snare_ghost(rng, 0.11), bar_start + 2.75 * beat + swing, pan=-0.34)
        if bar % 8 == 7:
            add_fill(stereo, spec, rng, bar_start)


def add_fill(stereo: np.ndarray, spec: DrumLayerSpec, rng: np.random.Generator, bar_start: float) -> None:
    beat = spec.beat_s
    sixteenth = beat / 4.0
    fill_start = bar_start + 3.0 * beat
    hits = (
        (0.00, 158.0, 0.22, -0.30),
        (0.75, 132.0, 0.24, 0.18),
        (1.50, 108.0, 0.26, -0.06),
        (2.25, 88.0, 0.30, 0.30),
    )
    for step, freq, gain, pan in hits:
        add_segment(stereo, tom(gain, freq), fill_start + step * sixteenth, pan=pan)
    for step in (0, 1, 2, 3):
        add_segment(stereo, snare_ghost(rng, 0.10 + 0.03 * step), fill_start + (step + 0.5) * sixteenth, pan=-0.15)


def add_bus_room(stereo: np.ndarray) -> np.ndarray:
    wet = np.zeros_like(stereo)
    taps = (
        (0.031, 0.07, 0.30),
        (0.047, 0.05, -0.25),
        (0.083, 0.035, 0.45),
    )
    for delay_s, gain, cross in taps:
        delay = int(round(delay_s * SAMPLE_RATE))
        wet[delay:, 0] += gain * ((1.0 - abs(cross)) * stereo[:-delay, 0] + abs(cross) * stereo[:-delay, 1])
        wet[delay:, 1] += gain * ((1.0 - abs(cross)) * stereo[:-delay, 1] + abs(cross) * stereo[:-delay, 0])
    return stereo + wet


def finish(stereo: np.ndarray) -> np.ndarray:
    stereo = add_bus_room(stereo)
    stereo = np.tanh(1.22 * stereo)
    peak = max(float(np.max(np.abs(stereo))), 1.0e-9)
    stereo = stereo / peak * 0.90
    # Keep the loop mostly full-strength, but remove DAC clicks at the boundaries.
    fade = int(round(0.012 * SAMPLE_RATE))
    stereo[:fade] *= np.linspace(0.0, 1.0, fade)[:, None]
    stereo[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    return stereo


def render(spec: DrumLayerSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    bed = np.zeros((timebase(spec.duration_s).shape[0], 2), dtype=np.float64)
    add_main_pattern(bed, spec, rng)
    add_bass_pattern(bed, spec)
    add_chord_stabs(bed, spec)
    add_hook_texture(bed, spec)
    bed *= backing_arrangement_gain(spec)[:, None]

    melody = np.zeros_like(bed)
    add_foreground_melody(melody, spec)

    return finish(bed + melody)


def write_wav(path: Path, stereo: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(stereo, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())


def default_spec() -> DrumLayerSpec:
    return DrumLayerSpec(filename="38_electro_orbit_drums_layer.wav")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MUSIC_DIR)
    parser.add_argument("--filename", default=default_spec().filename)
    parser.add_argument("--bpm", type=float, default=default_spec().bpm)
    parser.add_argument("--bars", type=int, default=default_spec().bars)
    parser.add_argument("--seed", type=int, default=default_spec().seed)
    args = parser.parse_args()

    spec = DrumLayerSpec(filename=args.filename, bpm=args.bpm, bars=args.bars, seed=args.seed)
    path = args.output_dir / spec.filename
    write_wav(path, render(spec))
    print(f"Wrote {path} ({spec.bars} bars, {spec.bpm:g} BPM, {spec.duration_s:.2f} s)")


if __name__ == "__main__":
    main()
