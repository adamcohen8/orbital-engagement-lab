"""Exact-epoch analytic cache parity and caller ownership regression tests."""

import numpy as np
import pytest

from sim.dynamics.orbit import epoch


@pytest.mark.parametrize('mode', ['analytic_simple', 'simple', 'analytic_enhanced', 'enhanced', ''])
def test_cached_analytic_positions_match_direct_calculations(mode):
    epoch._analytic_sun_moon_positions.cache_clear()
    simple = mode in ('analytic_simple', 'simple')
    sun_fn = epoch.sun_position_eci_km_simple if simple else epoch.sun_position_eci_km_enhanced
    moon_fn = epoch.moon_position_eci_km_simple if simple else epoch.moon_position_eci_km_enhanced
    for jd in [2451545.0, 2460310.5, 2461110.5, np.nextafter(2461110.5, np.inf)]:
        expected = sun_fn(jd), moon_fn(jd)
        for _ in range(2):
            actual = epoch.resolve_sun_moon_positions({'jd_utc': jd, 'ephemeris_mode': mode}, 0.0)
            for result, direct in zip(actual, expected):
                np.testing.assert_array_equal(result, direct)
    info = epoch._analytic_sun_moon_positions.cache_info()
    assert info.misses == 4 and info.hits == 4


def test_analytic_cache_preserves_writable_independent_results():
    epoch._analytic_sun_moon_positions.cache_clear()
    env = {'jd_utc': 2461110.5, 'ephemeris_mode': 'analytic_enhanced'}
    expected = epoch.resolve_sun_moon_positions(env, 0.0)
    changed = epoch.resolve_sun_moon_positions(env, 0.0)
    for array in changed:
        array[:] = 0.0
    fresh = epoch.resolve_sun_moon_positions(env, 0.0)
    for result, direct in zip(fresh, expected):
        np.testing.assert_array_equal(result, direct)
        assert result.flags.writeable
        assert not np.shares_memory(result, direct)
    for cached in epoch._analytic_sun_moon_positions(2461110.5, False):
        with pytest.raises(ValueError):
            cached.flags.writeable = True


def test_analytic_cache_is_bounded_and_distinguishes_models():
    epoch._analytic_sun_moon_positions.cache_clear()
    for index in range(140):
        epoch.resolve_sun_moon_positions({'jd_utc': 2461110.5 + index / 86400.0}, 0.0)
    assert epoch._analytic_sun_moon_positions.cache_info().currsize == 128
    epoch._analytic_sun_moon_positions.cache_clear()
    for mode in ['simple', 'enhanced']:
        epoch.resolve_sun_moon_positions({'jd_utc': 2461110.5, 'ephemeris_mode': mode}, 0.0)
    assert epoch._analytic_sun_moon_positions.cache_info().misses == 2


def test_explicit_positions_and_custom_ephemerides_keep_precedence():
    epoch._analytic_sun_moon_positions.cache_clear()
    explicit = np.array([1.0, 2.0, 3.0])
    jd = 2461110.5
    sun, moon = epoch.resolve_sun_moon_positions({'jd_utc': jd, 'sun_pos_eci_km': explicit}, 0.0)
    np.testing.assert_array_equal(sun, explicit)
    np.testing.assert_array_equal(moon, epoch.moon_position_eci_km_enhanced(jd))
    calls = []

    def callback(jd_utc, env):
        calls.append(jd_utc)
        return {'sun_pos_eci_km': explicit * len(calls), 'moon_pos_eci_km': explicit}

    env = {'jd_utc': jd, 'ephemeris_callable': callback}
    first = epoch.resolve_sun_moon_positions(env, 0.0)
    second = epoch.resolve_sun_moon_positions(env, 0.0)
    np.testing.assert_array_equal(first[0], explicit)
    np.testing.assert_array_equal(second[0], 2.0 * explicit)
    assert len(calls) == 2
