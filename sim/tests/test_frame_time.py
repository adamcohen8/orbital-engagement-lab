from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sim.dynamics.orbit.eop import EopError, EopRecord, EopSeries, audit_eop_series, load_iers_eop
from sim.frame_time import (
    CanonicalFrame,
    EarthOrientation,
    FrameTimeError,
    FrameTransformContext,
    TimeScale,
    epoch_conversion_receipt,
    epoch_julian_date,
    format_epoch,
    frame_transform_receipt,
    leap_second_table_receipt,
    main,
    normalize_canonical_frame,
    parse_epoch,
    state_transform_matrix,
    tai_minus_utc,
    transform_cartesian_state,
    transform_covariance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _context(epoch_text: str = "2024-01-01T00:00:00") -> FrameTransformContext:
    return FrameTransformContext(
        epoch=parse_epoch(epoch_text, TimeScale.UTC),
        earth_orientation=EarthOrientation(
            dut1_s=0.0087572,
            xp_arcsec=0.136928,
            yp_arcsec=0.202199,
            ddpsi_rad=-5.0e-8,
            ddeps_rad=3.0e-8,
            source="synthetic test EOP sample",
            source_sha256="0" * 64,
        ),
    )


def test_packaged_leap_table_is_hash_bound_and_current_through_2026() -> None:
    receipt = leap_second_table_receipt()
    path = REPO_ROOT / receipt["resource"]

    assert receipt["table_id"] == "iers-bulletin-c-72-2026-07-06"
    assert receipt["valid_through_utc"] == "2026-12-31T23:59:59.999999"
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tai_minus_utc(parse_epoch("2026-08-29T12:00:00", "UTC")) == 37


def test_positive_leap_second_is_a_distinct_one_second_instant() -> None:
    before = parse_epoch("2016-12-31T23:59:59", "UTC")
    leap = parse_epoch("2016-12-31T23:59:60", "UTC")
    after = parse_epoch("2017-01-01T00:00:00", "UTC")

    assert leap.tai_seconds - before.tai_seconds == 1.0
    assert after.tai_seconds - leap.tai_seconds == 1.0
    assert format_epoch(leap, "UTC") == "2016-12-31T23:59:60"
    assert format_epoch(leap, "TAI") == "2017-01-01T00:00:36"
    assert format_epoch(leap, "TT") == "2017-01-01T00:01:08.184"


def test_invalid_or_uncovered_utc_epochs_fail_closed() -> None:
    with pytest.raises(FrameTimeError, match="not a positive leap-second date"):
        parse_epoch("2016-12-30T23:59:60", "UTC")
    with pytest.raises(FrameTimeError, match="outside leap-second table coverage"):
        parse_epoch("2027-01-01T00:00:00", "UTC")
    with pytest.raises(FrameTimeError, match="outside the v1 contract"):
        epoch_julian_date(parse_epoch("2016-12-31T23:59:60", "UTC"), "UTC")


def test_utc_tai_tt_and_sampled_ut1_conversion_round_trip() -> None:
    utc = parse_epoch("2024-01-01T00:00:00.125", "UTC")
    tai = parse_epoch(format_epoch(utc, "TAI"), "TAI")
    tt = parse_epoch(format_epoch(utc, "TT"), "TT")
    ut1_text = format_epoch(utc, "UT1", dut1_s=0.0087572)
    ut1 = parse_epoch(ut1_text, "UT1", dut1_s=0.0087572)

    assert tai.tai_seconds == pytest.approx(utc.tai_seconds, abs=1.0e-6)
    assert tt.tai_seconds == pytest.approx(utc.tai_seconds, abs=1.0e-6)
    assert ut1.tai_seconds == pytest.approx(utc.tai_seconds, abs=1.0e-6)
    assert epoch_julian_date(utc, "TT") - epoch_julian_date(utc, "TAI") == pytest.approx(
        32.184 / 86400.0,
        abs=2.0e-10,
    )
    receipt = epoch_conversion_receipt(utc, "UT1", dut1_s=0.0087572)
    assert receipt["output"] == {"text": ut1_text, "scale": "UT1"}
    assert receipt["dut1_s"] == 0.0087572


def test_frame_names_reject_generic_or_unvalidated_aliases() -> None:
    assert normalize_canonical_frame("OEL/ECI/J2000") is CanonicalFrame.EME2000
    assert normalize_canonical_frame("OEL/ECEF/IAU76_80_EOP") is CanonicalFrame.ITRF
    for ambiguous in ("ECI", "ECEF", "J2000", "ITRF2000"):
        with pytest.raises(FrameTimeError, match="ambiguous frame"):
            normalize_canonical_frame(ambiguous)


def test_gcrf_is_named_and_not_silently_equated_to_eme2000() -> None:
    matrix = state_transform_matrix("GCRF", "EME2000", context=_context())

    assert not np.array_equal(matrix, np.eye(6))
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(6), rtol=0.0, atol=4.0e-16)


def test_gcrf_itrf_state_and_covariance_round_trip() -> None:
    context = _context()
    position = np.array([7000.0, 120.0, 30.0])
    velocity = np.array([-0.2, 7.45, 1.1])
    covariance = np.diag([4.0e-4, 9.0e-4, 1.6e-3, 4.0e-10, 9.0e-10, 1.6e-9])

    fixed = transform_cartesian_state(position, velocity, "GCRF", "ITRF", context=context)
    recovered = transform_cartesian_state(*fixed, "ITRF", "GCRF", context=context)
    fixed_covariance = transform_covariance(covariance, "GCRF", "ITRF", context=context)
    recovered_covariance = transform_covariance(fixed_covariance, "ITRF", "GCRF", context=context)

    np.testing.assert_allclose(recovered[0], position, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(recovered[1], velocity, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(recovered_covariance, covariance, rtol=2.0e-12, atol=2.0e-18)


def test_eme2000_itrf_state_transform_round_trips_position_and_velocity() -> None:
    context = _context()
    position = np.array([7000.0, 120.0, 30.0])
    velocity = np.array([-0.2, 7.45, 1.1])

    fixed_position, fixed_velocity = transform_cartesian_state(
        position,
        velocity,
        "EME2000",
        "ITRF",
        context=context,
    )
    inertial_position, inertial_velocity = transform_cartesian_state(
        fixed_position,
        fixed_velocity,
        "ITRF",
        "EME2000",
        context=context,
    )

    assert np.linalg.norm(fixed_velocity - state_transform_matrix("EME2000", "ITRF", context=context)[3:, 3:] @ velocity) > 0.1
    np.testing.assert_allclose(inertial_position, position, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(inertial_velocity, velocity, rtol=0.0, atol=2.0e-12)


def test_teme_eme2000_state_transform_round_trips_under_vallado_contract() -> None:
    context = _context()
    position = np.array([7000.0, -25.0, 10.0])
    velocity = np.array([0.01, 7.5, -0.2])

    inertial = transform_cartesian_state(position, velocity, "TEME", "EME2000", context=context)
    recovered = transform_cartesian_state(*inertial, "EME2000", "TEME", context=context)

    np.testing.assert_allclose(recovered[0], position, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(recovered[1], velocity, rtol=0.0, atol=2.0e-12)


def test_covariance_transform_round_trip_and_seeded_sampling_agree() -> None:
    context = _context()
    diagonal = np.array([4.0e-4, 9.0e-4, 1.6e-3, 4.0e-10, 9.0e-10, 1.6e-9])
    covariance = np.diag(diagonal)
    transformed = transform_covariance(covariance, "EME2000", "ITRF", context=context)
    recovered = transform_covariance(transformed, "ITRF", "EME2000", context=context)

    np.testing.assert_allclose(recovered, covariance, rtol=2.0e-12, atol=2.0e-18)
    assert float(np.min(np.linalg.eigvalsh(transformed))) >= -1.0e-15

    rng = np.random.default_rng(290)
    samples = rng.multivariate_normal(np.zeros(6), covariance, size=120_000)
    jacobian = state_transform_matrix("EME2000", "ITRF", context=context)
    sampled = np.cov((samples @ jacobian.T), rowvar=False, ddof=1)
    scale = np.sqrt(np.outer(np.diag(transformed), np.diag(transformed)))
    assert float(np.max(np.abs(sampled - transformed) / scale)) < 0.01


def test_covariance_validation_rejects_asymmetry_and_negative_variance() -> None:
    asymmetric = np.eye(6)
    asymmetric[0, 1] = 0.5
    with pytest.raises(FrameTimeError, match="symmetric"):
        transform_covariance(asymmetric, "EME2000", "TEME", context=_context())
    invalid = np.eye(6)
    invalid[0, 0] = -1.0
    with pytest.raises(FrameTimeError, match="positive semidefinite"):
        transform_covariance(invalid, "EME2000", "TEME", context=_context())
    tiny_negative = np.diag([-1.0e-13] * 6)
    with pytest.raises(FrameTimeError, match="diagonal variances"):
        transform_covariance(tiny_negative, "EME2000", "TEME", context=_context())


def test_itrf_transform_receipt_binds_time_and_eop_provenance() -> None:
    receipt = frame_transform_receipt("EME2000", "ITRF", context=_context())

    assert receipt["source_frame"] == "EME2000"
    assert receipt["target_frame"] == "ITRF"
    assert receipt["epoch_utc"] == "2024-01-01T00:00:00"
    assert receipt["epoch_tai"] == "2024-01-01T00:00:37"
    assert receipt["eop"]["source_sha256"] == "0" * 64
    assert receipt["leap_seconds"]["table_id"] == "iers-bulletin-c-72-2026-07-06"


def test_itrf_transform_requires_epoch_matched_eop() -> None:
    context = FrameTransformContext(epoch=parse_epoch("2024-01-01T00:00:00", "UTC"))
    with pytest.raises(FrameTimeError, match="require sampled DUT1"):
        state_transform_matrix("EME2000", "ITRF", context=context)
    with pytest.raises(FrameTimeError, match="require sampled DUT1"):
        frame_transform_receipt("EME2000", "ITRF", context=context)


def test_packaged_finals2000a_excerpt_loads_interpolates_and_audits() -> None:
    source = REPO_ROOT / "sim/dynamics/orbit/data/finals2000a_2024_01_01_04_excerpt.txt"
    series = load_iers_eop(source)
    exact = series.sample(parse_epoch("2024-01-01T00:00:00", "UTC"))
    middle = series.sample(parse_epoch("2024-01-01T12:00:00", "UTC"))

    assert exact.xp_arcsec == pytest.approx(0.136894)
    assert exact.yp_arcsec == pytest.approx(0.202185)
    assert exact.dut1_s == pytest.approx(0.0087572)
    assert exact.dx_mas == pytest.approx(0.295)
    assert exact.dy_mas == pytest.approx(-0.095)
    mas_to_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    assert exact.ddpsi_rad == pytest.approx(0.283 * mas_to_rad)
    assert exact.ddeps_rad == pytest.approx(-0.183 * mas_to_rad)
    assert middle.xp_arcsec == pytest.approx((series.records[0].xp_arcsec + series.records[1].xp_arcsec) / 2)

    audit = audit_eop_series(series, as_of=parse_datetime("2024-01-03T00:00:00Z"))
    assert audit["status"] == "pass"
    assert audit["eop"]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(EopError, match="outside source coverage"):
        series.sample(parse_epoch("2024-01-05T00:00:00", "UTC"))


def test_c04_csv_aliases_and_expiry_are_bounded(tmp_path: Path) -> None:
    source = tmp_path / "c04.csv"
    source.write_text(
        "MJD,xp,yp,UT1-UTC,dX,dY,LOD,quality\n"
        "60310,0.136894,0.202185,0.0087572,0.283,-0.183,0.121,observed-final\n"
        "60311,0.136920,0.202210,0.0086900,0.280,-0.180,0.122,predicted\n",
        encoding="utf-8",
    )
    series = load_iers_eop(source)
    audit = audit_eop_series(series, as_of=parse_datetime("2024-01-03T00:00:00Z"))

    assert series.source_format == "c04_csv"
    assert audit["status"] == "fail"
    assert audit["eop"]["freshness"]["status"] == "expired"


def test_dut1_interpolation_preserves_utc_leap_discontinuity() -> None:
    series = EopSeries(
        records=(
            EopRecord(57753.0, 0.0, 0.0, -0.4),
            EopRecord(57754.0, 0.0, 0.0, 0.6),
        ),
        source_format="c04_csv",
        source_label="synthetic-leap.csv",
        source_sha256="0" * 64,
    )

    before = series.sample(parse_epoch("2016-12-31T12:00:00", "UTC"))
    after = series.sample(parse_epoch("2017-01-01T00:00:00", "UTC"))

    assert before.dut1_s == pytest.approx(-0.4)
    assert after.dut1_s == pytest.approx(0.6)


def test_eop_freshness_rejects_invalid_policy_and_does_not_use_future_observations() -> None:
    series = EopSeries(
        records=(
            EopRecord(61000.0, 0.0, 0.0, 0.0),
            EopRecord(61001.0, 0.0, 0.0, 0.0),
        ),
        source_format="c04_csv",
        source_label="future.csv",
        source_sha256="0" * 64,
    )

    audit = audit_eop_series(series, as_of=parse_datetime("2024-01-01T00:00:00Z"))

    assert audit["status"] == "fail"
    assert audit["eop"]["freshness"]["status"] == "not-yet-valid"
    assert audit["eop"]["last_observed_mjd_utc"] is None
    with pytest.raises(EopError, match="finite and non-negative"):
        audit_eop_series(series, as_of=parse_datetime("2024-01-01T00:00:00Z"), max_observed_age_days=float("nan"))
    with pytest.raises(EopError, match="unsupported quality"):
        EopSeries(
            records=(EopRecord(60310.0, 0.0, 0.0, 0.0, quality="P"), EopRecord(60311.0, 0.0, 0.0, 0.0)),
            source_format="c04_csv",
            source_label="bad-quality.csv",
            source_sha256="0" * 64,
        )


def test_c04_prediction_alias_is_canonicalized_and_audited_as_prediction(tmp_path: Path) -> None:
    source = tmp_path / "prediction.csv"
    source.write_text(
        "MJD,xp,yp,UT1-UTC,quality\n"
        "60310,0.1,0.2,0.01,observed-final\n"
        "60311,0.1,0.2,0.01,P\n",
        encoding="utf-8",
    )
    series = load_iers_eop(source)

    audit = audit_eop_series(series, as_of=parse_datetime("2024-01-01T12:00:00Z"))

    assert series.records[1].quality == "predicted"
    assert audit["status"] == "warning"
    assert audit["eop"]["freshness"]["status"] == "prediction-only"


def test_frame_time_cli_emits_machine_readable_receipts(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([
        "convert-epoch",
        "2024-01-01T00:00:00",
        "--from-scale", "UTC",
        "--to-scale", "TAI",
        "--json",
    ]) == 0
    epoch_payload = json.loads(capsys.readouterr().out)
    assert epoch_payload["output"] == {"scale": "TAI", "text": "2024-01-01T00:00:37"}

    eop_path = REPO_ROOT / "sim/dynamics/orbit/data/finals2000a_2024_01_01_04_excerpt.txt"
    assert main([
        "transform-state",
        "--epoch", "2024-01-01T00:00:00",
        "--source-frame", "GCRF",
        "--target-frame", "ITRF",
        "--position-km", "7000", "120", "30",
        "--velocity-km-s", "-0.2", "7.45", "1.1",
        "--eop", str(eop_path),
        "--json",
    ]) == 0
    state_payload = json.loads(capsys.readouterr().out)
    assert state_payload["status"] == "converted"
    assert state_payload["transform"]["model"] == "oel.iau2006-iau2000a-cio-eop.v1"
    assert state_payload["execution_occurred"] is False


def parse_datetime(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_retained_frame_time_validation_manifest_is_content_bound() -> None:
    data_root = REPO_ROOT / "sim/dynamics/orbit/data"
    manifest = json.loads((data_root / "frame_time_validation_manifest.json").read_text(encoding="utf-8"))

    for artifact in manifest["artifacts"]:
        path = (data_root / artifact["path"]).resolve()
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    report = json.loads((data_root / "frame_time_orekit_13_1_7_acceptance.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["runtime"]["orekit_version"] == "13.1.7"
    assert all(report["checks"].values())

    # Recompute the externally compared GCRF path from current production code.
    # This prevents a stale retained "pass" report from masking implementation drift.
    epoch = parse_epoch(report["input"]["epoch_utc"], "UTC")
    eop = load_iers_eop(data_root / "finals2000a_2024_01_01_04_excerpt.txt")
    context = FrameTransformContext(epoch=epoch, earth_orientation=eop.sample(epoch))
    position = np.asarray(report["input"]["position_km"], dtype=float)
    velocity = np.asarray(report["input"]["velocity_km_s"], dtype=float)
    orekit = report["orekit"]

    def vector(name: str, size: int) -> np.ndarray:
        value = np.asarray([float(item) for item in orekit[name].split(",")], dtype=float)
        assert value.shape == (size,)
        return value

    current_position, current_velocity = transform_cartesian_state(
        position, velocity, "GCRF", "ITRF", context=context
    )
    reference_position = vector("gcrf_to_itrf_position_m", 3) / 1000.0
    reference_velocity = vector("gcrf_to_itrf_velocity_m_s", 3) / 1000.0
    current_jacobian = state_transform_matrix("GCRF", "ITRF", context=context)
    reference_jacobian = vector("gcrf_to_itrf_jacobian", 36).reshape(6, 6)

    assert np.max(np.abs(current_position - reference_position)) * 1000.0 <= manifest["tolerances"]["gcrf_frame_position_max_m"]
    assert np.max(np.abs(current_velocity - reference_velocity)) * 1000.0 <= manifest["tolerances"]["gcrf_frame_velocity_max_m_s"]
    assert np.max(np.abs(current_jacobian - reference_jacobian)) <= manifest["tolerances"]["gcrf_state_jacobian_max_abs"]

    legacy_eop = report["oel"]["frame_transform"]["eop"]
    legacy_context = FrameTransformContext(
        epoch=epoch,
        earth_orientation=EarthOrientation(
            dut1_s=legacy_eop["dut1_s"],
            xp_arcsec=legacy_eop["xp_arcsec"],
            yp_arcsec=legacy_eop["yp_arcsec"],
            ddpsi_rad=legacy_eop["ddpsi_rad"],
            ddeps_rad=legacy_eop["ddeps_rad"],
            source=legacy_eop["source"],
            source_sha256=legacy_eop["source_sha256"],
        ),
    )
    legacy_position, legacy_velocity = transform_cartesian_state(
        position, velocity, "EME2000", "ITRF", context=legacy_context
    )
    teme_position, teme_velocity = transform_cartesian_state(
        position, velocity, "TEME", "EME2000", context=legacy_context
    )
    legacy_jacobian = state_transform_matrix("EME2000", "ITRF", context=legacy_context)
    reference_legacy_jacobian = vector("eme2000_to_itrf_jacobian", 36).reshape(6, 6)
    covariance = np.diag(report["input"]["covariance_diagonal_m_m_s"])
    current_covariance = legacy_jacobian @ covariance @ legacy_jacobian.T
    reference_covariance = reference_legacy_jacobian @ covariance @ reference_legacy_jacobian.T
    covariance_scale = np.sqrt(np.outer(np.diag(reference_covariance), np.diag(reference_covariance)))

    assert np.max(np.abs(legacy_position - vector("eme2000_to_itrf_position_m", 3) / 1000.0)) * 1000.0 <= manifest["tolerances"]["frame_position_max_m"]
    assert np.max(np.abs(legacy_velocity - vector("eme2000_to_itrf_velocity_m_s", 3) / 1000.0)) * 1000.0 <= manifest["tolerances"]["frame_velocity_max_m_s"]
    assert np.max(np.abs(teme_position - vector("teme_to_eme2000_position_m", 3) / 1000.0)) * 1000.0 <= manifest["tolerances"]["frame_position_max_m"]
    assert np.max(np.abs(teme_velocity - vector("teme_to_eme2000_velocity_m_s", 3) / 1000.0)) * 1000.0 <= manifest["tolerances"]["frame_velocity_max_m_s"]
    assert np.max(np.abs(legacy_jacobian - reference_legacy_jacobian)) <= manifest["tolerances"]["state_jacobian_max_abs"]
    assert np.max(np.abs(current_covariance - reference_covariance) / covariance_scale) <= manifest["tolerances"]["covariance_normalized_max"]
