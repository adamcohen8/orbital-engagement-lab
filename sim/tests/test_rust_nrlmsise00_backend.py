from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from sim.dynamics.orbit.nrlmsise00_backend import nrlmsise00_density


def test_native_nrlmsise_weather_parity_across_lower_and_upper_atmosphere() -> None:
    native = pytest.importorskip("oel_rust_orbit")
    if not hasattr(native, "environment_nrlmsise00_low_altitude_density"):
        pytest.skip("Rust orbit wheel predates NRLMSISE-00 density")
    rng = np.random.default_rng(27)
    times = [datetime(2022, 4, 1, 12, tzinfo=timezone.utc),
             datetime(2024, 9, 20, 3, 15, tzinfo=timezone.utc)]
    for index in range(96):
        alt = (0.0, 72.5, 95.0, 120.0, 250.0, 300.0)[index] if index < 6 else float(rng.uniform(0.0, 950.0))
        lat = float(rng.uniform(-80.0, 80.0))
        lon = float(rng.uniform(-180.0, 180.0))
        env = {
            "f107a": float(rng.uniform(70.0, 230.0)),
            "f107": float(rng.uniform(70.0, 230.0)),
            "ap": float(rng.uniform(2.0, 80.0)),
            "nrlmsise00_ap_a": rng.uniform(2.0, 80.0, size=7).tolist(),
        }
        when = times[index % len(times)]
        expected = nrlmsise00_density(alt, lat, lon, when, env, lst_hr=14.0)
        actual = nrlmsise00_density(alt, lat, lon, when,
                                    {**env, "_rust_numeric_backend": "rust"}, lst_hr=14.0)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-14, atol=0.0)

    low_env = {"f107a": 150.0, "f107": 150.0, "ap": 40.0,
               "nrlmsise00_ap_a": [40.0] * 7}
    for alt in (0.0, 72.5, 95.0, 100.0, 250.0, 299.999999):
        expected = nrlmsise00_density(alt, 20.0, 30.0, times[0], low_env, lst_hr=14.0)
        actual = nrlmsise00_density(alt, 20.0, 30.0, times[0],
                                    {**low_env, "_rust_numeric_backend": "rust"}, lst_hr=14.0)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-14, atol=0.0)


def test_low_altitude_uses_python_when_wheel_lacks_capability(monkeypatch) -> None:
    import sim.dynamics.orbit.nrlmsise00_backend as backend

    when = datetime(2024, 9, 20, 3, 15, tzinfo=timezone.utc)
    env = {"f107a": 150.0, "f107": 150.0, "ap": 40.0}
    expected = nrlmsise00_density(95.0, 20.0, 30.0, when, env, lst_hr=14.0)
    monkeypatch.setattr(backend, "_native_low_altitude_density_kernel", lambda: None)
    actual = nrlmsise00_density(95.0, 20.0, 30.0, when,
                                {**env, "_rust_numeric_backend": "rust"}, lst_hr=14.0)
    assert actual == expected
