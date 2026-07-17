from __future__ import annotations

from unittest.mock import patch

import numpy as np

from sim.dynamics.orbit.de440_hpop import hpop_de440_positions_km
from sim.runtime_support import _cached_compatibility_plan, _compatible_keyword_args


def test_compatible_keyword_plan_is_reused_for_bound_methods() -> None:
    class Plugin:
        def update(self, *, truth: object, t_s: float, optional: int = 1) -> tuple[object, float, int]:
            return truth, t_s, optional

    plugin = Plugin()
    kwargs = {"truth": object(), "t_s": 3.0, "extra": "ignored"}
    _cached_compatibility_plan.cache_clear()

    first = _compatible_keyword_args(plugin.update, kwargs)
    second = _compatible_keyword_args(plugin.update, kwargs)

    assert first == {"truth": kwargs["truth"], "t_s": 3.0}
    assert second == first
    assert _cached_compatibility_plan.cache_info().hits == 1


def test_de440_position_cache_returns_independent_arrays() -> None:
    env = {
        "de440_coeff_path": "/tmp/reference.npz",
        "de440_tai_utc_s": 37.0,
    }
    raw = {
        "sun": np.array([1.0e8, 2.0, 3.0]),
        "moon": np.array([4.0e5, 5.0, 6.0]),
    }
    with patch("sim.dynamics.orbit.de440_hpop.hpop_de440_positions_m", return_value=raw) as evaluate:
        first = hpop_de440_positions_km(2451545.0, env)
        first["sun"][0] = -1.0
        second = hpop_de440_positions_km(2451545.0, env)

    assert evaluate.call_count == 1
    np.testing.assert_array_equal(second["sun"], np.array([1.0e5, 0.002, 0.003]))
    np.testing.assert_array_equal(second["moon"], np.array([400.0, 0.005, 0.006]))
