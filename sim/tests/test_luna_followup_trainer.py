from __future__ import annotations

import numpy as np

from sim.game.training_history import RPOTrainingTracker
from sim.game.training_models import (
    InspectionGateConfig,
    RPOTrainingConfig,
    SunAngleConstraintConfig,
)


def test_swept_inspection_gate_requires_beam_at_crossing() -> None:
    config = RPOTrainingConfig(
        enabled=True,
        inspection_gates=(
            InspectionGateConfig(
                name="gate",
                center_ric_km=np.array([2.0, 2.0, 0.0]),
                half_width_ric_km=np.array([0.1, 0.1, 0.1]),
            ),
        ),
        sun_angle_constraints=(
            SunAngleConstraintConfig(
                name="beam",
                sun_direction_ric=np.array([0.0, 1.0, 0.0]),
                allowed_center_ric=np.array([0.0, 1.0, 0.0]),
                allowed_half_angle_deg=20.0,
            ),
        ),
    )
    tracker = RPOTrainingTracker(config)
    tracker.t_s = [0.0, 1.0]
    tracker.rel_ric_hist = [
        np.array([3.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    tracker._record_inspection_gate_sample(tracker.rel_ric_hist[-1], time_s=1.0)

    assert tracker._inspection_gate_names == []


def test_swept_inspection_gate_accepts_narrow_valid_overlap_inside_box() -> None:
    config = RPOTrainingConfig(
        enabled=True,
        inspection_gates=(
            InspectionGateConfig(
                name="gate",
                center_ric_km=np.array([2.0, 0.0, 0.0]),
                half_width_ric_km=np.array([1.5, 0.1, 0.1]),
            ),
        ),
        sun_angle_constraints=(
            SunAngleConstraintConfig(
                name="beam",
                sun_direction_ric=np.array([1.0, 0.0, 0.0]),
                allowed_center_ric=np.array([1.0, 0.0, 0.0]),
                allowed_half_angle_deg=5.0,
                min_range_km=1.0,
                max_range_km=1.1,
            ),
        ),
    )
    tracker = RPOTrainingTracker(config)
    tracker.t_s = [0.0, 1.0]
    tracker.rel_ric_hist = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    tracker._record_inspection_gate_sample(tracker.rel_ric_hist[-1], time_s=1.0)

    assert tracker._inspection_gate_names == ["gate"]
