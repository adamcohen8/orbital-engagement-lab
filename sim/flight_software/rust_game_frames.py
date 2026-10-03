"""Opt-in fused frame arithmetic with NumPy's existing reduction semantics."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from sim.utils import frames

from .rust_game_backend import extension


@lru_cache(maxsize=1)
def frame_converter():
    return extension().FrameConverter()


def quaternion_to_dcm_bn(quaternion):
    rows = frame_converter().quaternion_matrices([np.asarray(quaternion, dtype=float).reshape(-1).tolist()])
    return np.asarray(rows[0], dtype=float)


def training_frame_sample(target, chaser, reference, thrust):
    values = frame_converter().sample(
        target[:6].tolist(), chaser[:6].tolist(),
        None if reference is None else reference[:6].tolist(), thrust.tolist(),
    )
    return tuple(np.asarray(value, dtype=float) for value in values)


def relative_ric_state(deputy, chief):
    if frames._frame_acceleration_enabled():
        return frames.eci_relative_to_ric_rect(deputy, chief)
    values = frame_converter().relative_state(deputy.tolist(), chief.tolist())
    return np.asarray(values, dtype=float)
