"""Optional scalar scoring geometry; the maintained tracker owns all decisions."""

from functools import lru_cache

import numpy as np

from sim.flight_software.rust_game_backend import extension
from sim.game.training_geometry import nmt_curve_points_km


@lru_cache(maxsize=64)
def _goal_points(kind: str, values: tuple[float, ...]) -> bytes:
    if kind == "nmt":
        points = nmt_curve_points_km(
            radial_amplitude_km=values[0],
            cross_track_amplitude_km=values[1],
            cross_track_phase_deg=values[2],
            center_ric_km=np.asarray(values[3:6]),
        )
        if points.size == 0:
            points = np.asarray(values[3:6]).reshape(1, 3)
    else:
        points = np.asarray(values).reshape(-1, 3)
    return np.asarray(points, dtype="<f8").tobytes()


def score_sample(config, relative, thrust, target_thrust) -> tuple[float, ...]:
    # Recompute the small key so changes to a training goal invalidate the cache.
    if config.goal_nmt_radial_amplitude_km is not None and config.goal_nmt_tolerance_km is not None:
        kind = "nmt"
        values = (
            float(config.goal_nmt_radial_amplitude_km),
            float(config.goal_nmt_cross_track_amplitude_km),
            float(config.goal_nmt_cross_track_phase_deg),
            *map(float, config.goal_nmt_center_ric_km.reshape(3)),
        )
    elif config.inspection_gates and config.goal_nmt_radial_amplitude_km is None and config.goal_range_km is None:
        kind = "points"
        values = tuple(float(x) for gate in config.inspection_gates for x in gate.center_ric_km.reshape(3))
    else:
        kind = "points"
        values = tuple(map(float, config.goal_relative_ric_km.reshape(3)))
    return tuple(
        extension().score_sample(relative.tolist(), thrust.tolist(), target_thrust.tolist(), _goal_points(kind, values))
    )
