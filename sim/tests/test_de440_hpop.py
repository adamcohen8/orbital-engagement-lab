import numpy as np

from sim.acceleration.settings import acceleration_context
from sim.dynamics.orbit.de440_hpop import (
    _BODY_SPECS,
    _cheb3d,
    _eval_body,
    _extract_axis,
    _find_coeff_row,
    _find_light_body_row,
    _find_light_row,
    hpop_de440_positions_km,
    hpop_de440_positions_m,
    hpop_de440_sun_moon_positions_km,
    mjd_tt_to_mjd_tdb,
)


def test_cheb3d_constant_vector():
    out = _cheb3d(1.5, 3, 1.0, 2.0, np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
    assert np.allclose(out, np.array([1.0, 2.0, 3.0]))


def test_find_coeff_row_selects_covering_interval():
    pc = np.array([[0.0, 32.0, 1.0], [32.0, 64.0, 2.0]], dtype=float)
    row = _find_coeff_row(pc, 40.0)
    assert float(row[2]) == 2.0


def test_light_row_cache_preserves_boundary_and_nonmonotonic_selection():
    light = {
        "row_start_jd_tdb": np.array([1.0, 2.0]),
        "row_end_jd_tdb": np.array([2.0, 3.0]),
        "moon_start_jd_tdb": np.array([1.0, 2.0]),
        "moon_end_jd_tdb": np.array([2.0, 3.0]),
    }

    assert _find_light_row(light, 1.5) == 0
    assert _find_light_row(light, 2.5) == 1
    assert _find_light_row(light, 2.0) == 0
    assert _find_light_row(light, 1.25) == 0

    assert _find_light_body_row(light, 1.5, "moon") == 0
    assert _find_light_body_row(light, 2.5, "moon") == 1
    assert _find_light_body_row(light, 2.0, "moon") == 0
    assert _find_light_body_row(light, 1.25, "moon") == 0


def test_eval_body_constant_segment_returns_scaled_constant():
    row = np.zeros(1000, dtype=float)
    row[0] = 0.0
    row[1] = 32.0
    row[440] = 1.0
    row[453] = 2.0
    row[466] = 3.0
    out = _eval_body(row, 2.0, "moon")
    assert np.allclose(out, np.array([1000.0, 2000.0, 3000.0]))


def test_eval_body_earthmoon_uses_matlab_segment_columns():
    row = np.zeros(1000, dtype=float)
    row[0] = 0.0
    row[1] = 32.0
    row[230] = 1.0
    row[243] = 2.0
    row[256] = 3.0
    row[269] = 4.0
    row[282] = 5.0
    row[295] = 6.0

    first = _eval_body(row, 1.0, "earthmoon")
    second = _eval_body(row, 17.0, "earthmoon")

    assert np.allclose(first, np.array([1000.0, 2000.0, 3000.0]))
    assert np.allclose(second, np.array([4000.0, 5000.0, 6000.0]))


def test_mjd_tt_to_tdb_is_close():
    out = mjd_tt_to_mjd_tdb(51544.5)
    assert abs(out - 51544.5) < 1e-4


def _constant_body(row: np.ndarray, body: str, xyz_km: tuple[float, float, float]) -> None:
    spec = _BODY_SPECS[body]
    starts = tuple(int(v) for v in spec["starts"])  # type: ignore[index]
    coeff_count = int(spec["coeffs"])  # type: ignore[index]
    for start, value in zip(starts, xyz_km, strict=False):
        row[start - 1] = float(value)
        row[start - 1 + coeff_count] = float(value) + 10.0
        row[start - 1 + 2 * coeff_count] = float(value) + 20.0


def _light_payload_from_row(row: np.ndarray) -> dict[str, np.ndarray]:
    bodies = np.array(["earthmoon", "moon", "sun"])
    payload = {
        "format_version": np.array("oel_de440_light_v1"),
        "source": np.array("unit-test"),
        "row_start_jd_tdb": np.array([row[0]], dtype=float),
        "row_end_jd_tdb": np.array([row[1]], dtype=float),
        "bodies": bodies,
    }
    for body in bodies:
        spec = _BODY_SPECS[str(body)]
        starts = tuple(int(v) for v in spec["starts"])  # type: ignore[index]
        coeff_count = int(spec["coeffs"])  # type: ignore[index]
        payload[f"{body}_coeffs"] = np.array(coeff_count, dtype=np.int64)
        payload[f"{body}_segments"] = np.array(int(spec["segments"]), dtype=np.int64)  # type: ignore[index]
        payload[f"{body}_span_days"] = np.array(float(spec["span_days"]), dtype=float)  # type: ignore[index]
        payload[f"{body}_x"] = np.atleast_2d(_extract_axis(row, starts, coeff_count))
        payload[f"{body}_y"] = np.atleast_2d(_extract_axis(row, tuple(s + coeff_count for s in starts), coeff_count))
        payload[f"{body}_z"] = np.atleast_2d(_extract_axis(row, tuple(s + 2 * coeff_count for s in starts), coeff_count))
    return payload


def test_oel_de440_light_matches_full_sun_moon_coefficients(tmp_path):
    row = np.zeros(900, dtype=float)
    row[0] = 2451545.0
    row[1] = 2451577.0
    _constant_body(row, "earthmoon", (100.0, 200.0, 300.0))
    _constant_body(row, "moon", (1.0, 2.0, 3.0))
    _constant_body(row, "sun", (1000.0, 2000.0, 3000.0))

    path = tmp_path / "oel_de440_light_test.npz"
    np.savez_compressed(path, **_light_payload_from_row(row))
    out = hpop_de440_positions_m(2451546.0, coeff_path=path)

    r_earthmoon = _eval_body(row, 2451546.0, "earthmoon")
    r_moon = _eval_body(row, 2451546.0, "moon")
    r_sun = _eval_body(row, 2451546.0, "sun")
    r_earth = r_earthmoon - (1.0 / (1.0 + 81.3005682214972154)) * r_moon

    assert set(out) == {"earth", "sun_ssb", "sun", "moon"}
    assert np.allclose(out["earth"], r_earth)
    assert np.allclose(out["moon"], r_moon)
    assert np.allclose(out["sun_ssb"], r_sun)
    assert np.allclose(out["sun"], -r_earth + r_sun)


def test_oel_de440_light_body_specific_records(tmp_path):
    path = tmp_path / "oel_de440_light_body_records.npz"
    payload = {
        "format_version": np.array("oel_de440_light_v1"),
        "source": np.array("unit-test"),
        "row_start_jd_tdb": np.array([2451545.0], dtype=float),
        "row_end_jd_tdb": np.array([2451549.0], dtype=float),
        "bodies": np.array(["earthmoon", "moon", "sun"]),
    }
    for body, xyz in {
        "earthmoon": (100.0, 200.0, 300.0),
        "moon": (1.0, 2.0, 3.0),
        "sun": (1000.0, 2000.0, 3000.0),
    }.items():
        payload[f"{body}_coeffs"] = np.array(1, dtype=np.int64)
        payload[f"{body}_segments"] = np.array(1, dtype=np.int64)
        payload[f"{body}_span_days"] = np.array(4.0, dtype=float)
        payload[f"{body}_start_jd_tdb"] = np.array([2451545.0], dtype=float)
        payload[f"{body}_end_jd_tdb"] = np.array([2451549.0], dtype=float)
        payload[f"{body}_x"] = np.array([[xyz[0]]], dtype=float)
        payload[f"{body}_y"] = np.array([[xyz[1]]], dtype=float)
        payload[f"{body}_z"] = np.array([[xyz[2]]], dtype=float)
    np.savez_compressed(path, **payload)

    out = hpop_de440_positions_m(2451546.0, coeff_path=path)
    r_earthmoon = np.array([100.0, 200.0, 300.0]) * 1e3
    r_moon = np.array([1.0, 2.0, 3.0]) * 1e3
    r_sun = np.array([1000.0, 2000.0, 3000.0]) * 1e3
    r_earth = r_earthmoon - (1.0 / (1.0 + 81.3005682214972154)) * r_moon

    assert np.allclose(out["earth"], r_earth)
    assert np.allclose(out["moon"], r_moon)
    assert np.allclose(out["sun"], -r_earth + r_sun)


def test_oel_de440_light_acceleration_is_exact_at_boundaries_and_nonmonotonic_times(tmp_path):
    path = tmp_path / "oel_de440_light_acceleration_parity.npz"
    payload = {
        "format_version": np.array("oel_de440_light_v1"),
        "source": np.array("unit-test"),
        "row_start_jd_tdb": np.array([2451545.0, 2451549.0], dtype=float),
        "row_end_jd_tdb": np.array([2451549.0, 2451553.0], dtype=float),
        "bodies": np.array(["earthmoon", "moon", "sun"]),
    }
    for body_index, body in enumerate(("earthmoon", "moon", "sun"), start=1):
        payload[f"{body}_coeffs"] = np.array(2, dtype=np.int64)
        payload[f"{body}_segments"] = np.array(1, dtype=np.int64)
        payload[f"{body}_span_days"] = np.array(4.0, dtype=float)
        payload[f"{body}_start_jd_tdb"] = np.array([2451545.0, 2451549.0], dtype=float)
        payload[f"{body}_end_jd_tdb"] = np.array([2451549.0, 2451553.0], dtype=float)
        base = float(body_index * 10)
        payload[f"{body}_x"] = np.array([[base, 0.125], [base + 1.0, -0.25]], dtype=float)
        payload[f"{body}_y"] = np.array([[base + 2.0, -0.375], [base + 3.0, 0.5]], dtype=float)
        payload[f"{body}_z"] = np.array([[base + 4.0, 0.625], [base + 5.0, -0.75]], dtype=float)
    np.savez_compressed(path, **payload)

    sample_times = (2451546.25, 2451549.0, 2451551.75, 2451545.5)
    for jd_tdb in sample_times:
        with acceleration_context("off"):
            expected = hpop_de440_positions_m(jd_tdb, coeff_path=path)
        with acceleration_context("auto"):
            actual = hpop_de440_positions_m(jd_tdb, coeff_path=path)
        assert tuple(actual) == tuple(expected)
        for body in expected:
            np.testing.assert_array_equal(actual[body], expected[body])


def test_oel_de440_light_acceleration_off_does_not_load_compiled_kernel(tmp_path):
    row = np.zeros(900, dtype=float)
    row[0] = 2451545.0
    row[1] = 2451577.0
    _constant_body(row, "earthmoon", (100.0, 200.0, 300.0))
    _constant_body(row, "moon", (1.0, 2.0, 3.0))
    _constant_body(row, "sun", (1000.0, 2000.0, 3000.0))
    path = tmp_path / "oel_de440_light_acceleration_off.npz"
    np.savez_compressed(path, **_light_payload_from_row(row))

    from unittest.mock import patch

    with (
        acceleration_context("off"),
        patch(
            "sim.dynamics.orbit.de440_hpop._compiled_de440_light_core",
            side_effect=AssertionError("compiled DE440 kernel must stay disabled"),
        ),
    ):
        out = hpop_de440_positions_m(2451546.0, coeff_path=path)
    assert np.linalg.norm(out["sun"]) > 0.0


def test_oel_de440_accelerated_utc_sun_moon_pair_is_exact(tmp_path):
    row = np.zeros(900, dtype=float)
    row[0] = 2451545.0
    row[1] = 2451577.0
    _constant_body(row, "earthmoon", (100.0, 200.0, 300.0))
    _constant_body(row, "moon", (1.0, 2.0, 3.0))
    _constant_body(row, "sun", (1000.0, 2000.0, 3000.0))
    path = tmp_path / "oel_de440_light_utc_pair.npz"
    np.savez_compressed(path, **_light_payload_from_row(row))
    env = {"de440_coeff_path": str(path), "de440_tai_utc_s": 37.0}

    with acceleration_context("off"):
        positions = hpop_de440_positions_km(2451545.5, dict(env))
    with acceleration_context("auto"):
        sun, moon = hpop_de440_sun_moon_positions_km(2451545.5, dict(env))

    np.testing.assert_array_equal(sun, positions["sun"])
    np.testing.assert_array_equal(moon, positions["moon"])
