"""Earth radiation optimized quadrature and batch validation contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from sim import rust_environment_backend as backend
from sim.pro_perturbations.earth_radiation import _quadrature


def native():
    extension = pytest.importorskip("oel_rust_orbit")
    if not callable(getattr(extension, "perturbation_earth_radiation_batch", None)):
        pytest.skip("installed wheel lacks Earth radiation batches")
    return extension


# The established analytic tolerance covers these orders. At order 128 the
# unchanged sequential Rust reduction's baseline error exceeds it; that order
# is checked bit-for-bit against the frozen Rust oracle instead.
@pytest.mark.parametrize("order", [8, 9, 32, 64])
@pytest.mark.parametrize("axis", [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
def test_uniform_native_ir_keeps_exact_spherical_solution(order, axis):
    extension = native()
    radius = 7000.0
    axis = np.asarray(axis)
    sun_distance = 149597870.7  # established native AU constant
    albedo, infrared = extension.perturbation_earth_radiation(
        (radius * axis).tolist(),
        [sun_distance, 0.0, 0.0],
        0.0,
        6378.137,
        *(q.tolist() for q in _quadrature(order)),
        True,
        True,
        0.0,
        0.68,
    )
    expected = 4.5606e-6 * 0.68 / 4 * (6378.137 / radius) ** 2 * axis
    np.testing.assert_allclose(infrared, expected, rtol=2e-13, atol=1e-22)
    np.testing.assert_array_equal(albedo, np.zeros(3))


@pytest.mark.parametrize("flags", [(True, True), (False, False)])
@pytest.mark.parametrize("bad", ["position", "sun", "time", "inside"])
def test_batch_still_validates_every_later_row(flags, bad):
    extension = native()
    positions = np.array([[7000.0, 0.0, 0.0], [0.0, 0.0, 7100.0]])
    suns = np.array([[1.3e8, 0.0, 0.0], [1.3e8, 0.0, 0.0]])
    times = np.array([0.0, 123.25])
    if bad == "position":
        positions[1, 0] = np.nan
    elif bad == "sun":
        suns[1, 1] = np.inf
    elif bad == "time":
        times[1] = np.nan
    else:
        positions[1] = 0.0
    args = (
        positions.tobytes(),
        suns.tobytes(),
        times.tobytes(),
        6378.137,
        *(q.tolist() for q in _quadrature(8)),
        *flags,
        None,
        None,
    )
    if bad == "inside" and not any(flags):
        actual = np.frombuffer(extension.perturbation_earth_radiation_batch(*args), dtype="<f8")
        np.testing.assert_array_equal(actual, np.zeros(12))
    else:
        with pytest.raises(ValueError):
            extension.perturbation_earth_radiation_batch(*args)


def test_batch_retains_first_row_validation_error_precedence():
    extension = native()
    # Invalid shared quadrature and invalid first position: position validation
    # has always preceded shared-parameter validation.
    positions = np.array([[np.nan, 0.0, 0.0], [7000.0, 0.0, 0.0]])
    suns = np.tile([1.3e8, 0.0, 0.0], (2, 1))
    with pytest.raises(ValueError, match="spacecraft position must contain only finite"):
        extension.perturbation_earth_radiation_batch(
            positions.tobytes(), suns.tobytes(), np.zeros(2).tobytes(), 6378.137, [], [], [], [], True, True, None, None
        )


def test_missing_radiation_symbols_keep_legacy_fallback(monkeypatch):
    monkeypatch.setattr(backend, "_extension", lambda: SimpleNamespace())
    assert (
        backend.try_earth_radiation_components(None, None, None, order=32, include_albedo=True, include_infrared=True)
        is None
    )
    assert (
        backend.try_earth_radiation_components_batch(
            None, None, None, order=32, include_albedo=True, include_infrared=True
        )
        is None
    )
