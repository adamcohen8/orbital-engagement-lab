"""Native rolling-row gravity against the independent full-table evaluator."""
import numpy as np
import pytest

from sim.dynamics.orbit.spherical_harmonics import (
    SphericalHarmonicTerm,
    _analytic_harmonic_accel_hpop_eci_km_s2,
    compile_spherical_harmonic_terms,
)


@pytest.mark.parametrize('degree,order', [(2, 0), (2, 2), (8, 8), (20, 20), (70, 0), (70, 35), (70, 70)])
def test_native_harmonics_match_full_table_at_truncated_orders(degree, order):
    native = pytest.importorskip('oel_rust_orbit')
    terms = [
        SphericalHarmonicTerm(n, m, (-1.0)**m * 1e-6 / n**2, 1e-7 / (n + m)**2, True)
        for n in range(2, degree + 1) for m in range(min(n, order) + 1)
    ]
    compiled = compile_spherical_harmonic_terms(terms)
    tables = [getattr(compiled, name).ravel().tolist() for name in [
        'c_nm', 's_nm', 'legendre_diag_scale', 'legendre_subdiag_scale',
        'legendre_recur_a', 'legendre_recur_b', 'legendre_recur_c',
    ]]
    mu = 398600.4415
    radius = 6378.1363
    context = native.ONPForceContext([1], [mu, radius, 300.0, 2.2, 1.0, 0.0, 1.0, 1.2,
                                        4.56e-6, 149597870.7, radius, 695700.0, 0.0, 0.0, mu],
                                   0, (degree, order), tables)
    stage = np.eye(3).ravel().tolist() + [0.0] * 10
    positions = [[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0], [0.01, 0.02, 7000.0],
                 [0.01, -0.02, -7000.0], [-7000.0, 100.0, 900.0]]
    for position in positions:
        r = np.asarray(position)
        expected = _analytic_harmonic_accel_hpop_eci_km_s2(
            r_eci_km=r, t_s=0.0, terms=terms, mu_km3_s2=mu, re_km=radius,
            jd_utc_start=None, frame_model='simple', eop_path=None, compiled=compiled,
        ) - mu * r / np.linalg.norm(r)**3
        actual = context.acceleration(position + [0.0, 7.5, 1.0], stage, [0.0] * 3)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-16)
