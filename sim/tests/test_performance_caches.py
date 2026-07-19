from __future__ import annotations

from unittest.mock import patch

import numpy as np

from sim.dynamics.orbit.atmosphere import _local_solar_time_epoch_terms, _local_solar_time_hr
from sim.dynamics.orbit.de440_hpop import hpop_de440_positions_km
from sim.dynamics.orbit.epoch import resolve_time_dependent_env
from sim.dynamics.orbit.frames import (
    _cached_eci_to_ecef_rotation_hpop_like,
    _eci_to_ecef_rotation_hpop_like_uncached,
    _precession_nutation_matrix_approx,
    eci_to_ecef_rotation_hpop_like,
)
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


def test_hpop_frame_rotation_cache_is_exact_and_returns_independent_arrays() -> None:
    kwargs = {
        "jd_utc_start": 2460310.5,
        "dut1_s": 0.3,
        "xp_arcsec": 0.1,
        "yp_arcsec": 0.2,
        "dat_s": 37.0,
        "ddpsi_rad": 1.0e-8,
        "ddeps_rad": -2.0e-8,
    }
    _cached_eci_to_ecef_rotation_hpop_like.cache_clear()
    expected = _eci_to_ecef_rotation_hpop_like_uncached(123.5, **kwargs)

    first = eci_to_ecef_rotation_hpop_like(123.5, **kwargs)
    first[0, 0] = -999.0
    second = eci_to_ecef_rotation_hpop_like(123.5, **kwargs)

    np.testing.assert_array_equal(second, expected)
    assert _cached_eci_to_ecef_rotation_hpop_like.cache_info().hits == 1
    assert _cached_eci_to_ecef_rotation_hpop_like.cache_info().misses == 1


def test_precession_nutation_terms_are_shared_across_frame_consumers() -> None:
    _precession_nutation_matrix_approx.cache_clear()
    first = _precession_nutation_matrix_approx(
        2460310.5,
        ddpsi_rad=1.0e-8,
        ddeps_rad=-2.0e-8,
    )
    second = _precession_nutation_matrix_approx(
        2460310.5,
        ddpsi_rad=1.0e-8,
        ddeps_rad=-2.0e-8,
    )

    assert first is second
    assert _precession_nutation_matrix_approx.cache_info().hits == 1
    assert _precession_nutation_matrix_approx.cache_info().misses == 1


def test_local_solar_time_reuses_epoch_only_terms() -> None:
    from datetime import datetime, timezone

    env = {
        "density_frame_model": "iau76_80_eop",
        "density_eop_path": "/tmp/eop.txt",
        "dut1_s": 0.2,
        "dat_s": 37.0,
    }
    epoch = datetime(2022, 3, 31, 12, 0, tzinfo=timezone.utc)
    _local_solar_time_epoch_terms.cache_clear()
    with (
        patch("sim.dynamics.orbit.atmosphere.apparent_sidereal_time_hpop_like", return_value=1.25) as sidereal,
        patch(
            "sim.dynamics.orbit.atmosphere.sun_position_eci_km_enhanced",
            return_value=np.array([1.0, 2.0, 3.0]),
        ) as sun,
    ):
        first = _local_solar_time_hr(10.0, epoch, env)
        second = _local_solar_time_hr(20.0, epoch, env)

    assert first != second
    assert sidereal.call_count == 1
    assert sun.call_count == 1
    assert _local_solar_time_epoch_terms.cache_info().hits == 1


def test_de440_stage_environment_cache_is_keyed_by_model_inputs() -> None:
    env = {
        "jd_utc_start": 2451545.0,
        "ephemeris_mode": "de440_hpop",
        "de440_coeff_path": "/tmp/de440-a.npz",
    }
    shared_cache: dict = {}
    positions = (np.array([1.0e8, 2.0, 3.0]), np.array([4.0e5, 5.0, 6.0]))
    with patch("sim.dynamics.orbit.epoch.resolve_sun_moon_positions", return_value=positions) as resolve:
        first = resolve_time_dependent_env(env, 10.0, cache_override=shared_cache)
        second = resolve_time_dependent_env(env, 10.0, cache_override=shared_cache)
        changed = resolve_time_dependent_env(
            {**env, "de440_coeff_path": "/tmp/de440-b.npz"},
            10.0,
            cache_override=shared_cache,
        )

    np.testing.assert_array_equal(first["sun_pos_eci_km"], second["sun_pos_eci_km"])
    np.testing.assert_array_equal(changed["moon_pos_eci_km"], positions[1])
    assert resolve.call_count == 2
