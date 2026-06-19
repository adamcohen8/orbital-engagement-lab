import numpy as np

from sim.dynamics.orbit.de440_hpop import (
    _BODY_SPECS,
    _cheb3d,
    _eval_body,
    _extract_axis,
    _find_coeff_row,
    hpop_de440_positions_m,
    mjd_tt_to_mjd_tdb,
)


def test_cheb3d_constant_vector():
    out = _cheb3d(1.5, 3, 1.0, 2.0, np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
    assert np.allclose(out, np.array([1.0, 2.0, 3.0]))


def test_find_coeff_row_selects_covering_interval():
    pc = np.array([[0.0, 32.0, 1.0], [32.0, 64.0, 2.0]], dtype=float)
    row = _find_coeff_row(pc, 40.0)
    assert float(row[2]) == 2.0


def test_eval_body_constant_segment_returns_scaled_constant():
    row = np.zeros(1000, dtype=float)
    row[0] = 0.0
    row[1] = 32.0
    row[440] = 1.0
    row[453] = 2.0
    row[466] = 3.0
    out = _eval_body(row, 2.0, "moon")
    assert np.allclose(out, np.array([1000.0, 2000.0, 3000.0]))


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
