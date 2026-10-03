"""Parity and fallback for the exact shared JB2006/JB2008 quadrature port."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sim.dynamics.orbit import jb2008_backend as backend
from sim.dynamics.orbit.epoch import datetime_to_julian_date


def _weather_tables(directory, epoch):
    directory.mkdir(parents=True, exist_ok=True)
    sol = directory / 'SOLFSMY_constant.txt'
    ap = directory / 'SOLRESAP_constant.txt'
    dtc = directory / 'DTCFILE_constant.txt'
    jd_floor = int(np.floor(datetime_to_julian_date(epoch)))
    np.savetxt(sol, [[0., 0., jd_floor + offset, 150., 150., 140., 140.,
                      130., 130., 120., 120.] for offset in range(-8, 4)], fmt='%.9f')
    dates = [epoch + timedelta(days=offset) for offset in range(-2, 3)]
    np.savetxt(ap, [[day.year, day.timetuple().tm_yday, 0., 0., *([4.] * 8)]
                    for day in dates], fmt='%.9f')
    np.savetxt(dtc, [[day.year, day.timetuple().tm_yday, *([0.] * 24)]
                     for day in dates], fmt='%.9f')
    return {'jb2006_sol_path': str(sol), 'jb2006_ap_path': str(ap),
            'jb2008_sol_path': str(sol), 'jb2008_dtc_path': str(dtc)}


@pytest.mark.parametrize('altitude', [90., 90.1, 95., 104.999, 105., 105.001, 200., 500., 1200., 2000.])
@pytest.mark.parametrize('coefficients', [(450., 14., 200., 0.07), (700., 20., 300., 0.04), (350., 8., 100., 0.1)])
def test_lower_quadrature_native_preserves_each_intermediate(altitude, coefficients):
    native = pytest.importorskip('oel_rust_orbit')
    function = getattr(native, 'environment_jb_lower_atmosphere', None)
    if function is None:
        pytest.skip('installed wheel predates JB lower quadrature')
    expected = backend._integrate_lower_atmosphere_python(altitude, coefficients)
    actual = function(altitude, coefficients)
    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=5e-12)


def test_lower_quadrature_older_wheel_keeps_python_fallback(monkeypatch):
    import sim.rust_environment_backend as bridge
    monkeypatch.setattr(bridge, 'try_jb_lower_atmosphere', lambda *_: None)
    tc = (450., 14., 200., 0.07)
    assert backend._integrate_lower_atmosphere(400., tc, {'_rust_numeric_backend': 'rust'}) == backend._integrate_lower_atmosphere_python(400., tc)


@pytest.mark.parametrize('altitude', [105., 105.001, 250., 500., 800., 1200., 2000.])
@pytest.mark.parametrize('coefficients', [(450., 14., 200., 0.07), (700., 20., 300., 0.04)])
def test_upper_quadrature_native_preserves_each_intermediate(altitude, coefficients):
    native = pytest.importorskip('oel_rust_orbit')
    function = getattr(native, 'environment_jb_upper_atmosphere', None)
    if function is None:
        pytest.skip('installed wheel predates JB upper quadrature')
    arguments = (105., 105., altitude, 0.025, 0.04, coefficients)
    expected = backend._integrate_upper_atmosphere_python(*arguments)
    actual = function(*arguments)
    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=5e-12)


def test_upper_quadrature_older_wheel_keeps_python_fallback(monkeypatch):
    import sim.rust_environment_backend as bridge
    monkeypatch.setattr(bridge, 'try_jb_upper_atmosphere', lambda *_: None)
    arguments = (105., 105., 500., 0.025, 0.04, (450., 14., 200., 0.07))
    assert backend._integrate_upper_atmosphere(*arguments, {'_rust_numeric_backend': 'rust'}) == backend._integrate_upper_atmosphere_python(*arguments)


@pytest.mark.parametrize('model', ['jb2006', 'jb2008'])
def test_native_lower_quadrature_preserves_complete_density(model, tmp_path):
    native = pytest.importorskip('oel_rust_orbit')
    if getattr(native, 'environment_jb_lower_atmosphere', None) is None:
        pytest.skip('installed wheel predates JB lower quadrature')
    function = getattr(backend, model + '_density')
    epoch = datetime(2022, 3, 31, 12, tzinfo=timezone.utc)
    weather = _weather_tables(tmp_path / 'weather', epoch)
    for altitude in [90., 105., 400., 500., 1200., 1600.]:
        for lat, lon in [(0., 0.), (42., -105.), (-67., 170.)]:
            expected = function(altitude, lat, lon, epoch, weather)
            actual = function(altitude, lat, lon, epoch, {**weather, '_rust_numeric_backend': 'rust'})
            assert actual == pytest.approx(expected, rel=5e-14, abs=0.)
