from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.ccsds import (
    CcsdsOemError,
    OemCovariance,
    OemHeader,
    OemMessage,
    OemMetadata,
    OemSegment,
    OemState,
    compare_oem,
    convert_oem,
    export_completed_run_oem,
    inspect_oem,
    oem_to_mission_input_packet,
    parse_oem_kvn,
    read_oem_kvn,
    serialize_oem_kvn,
)
from sim.dynamics.orbit.eop import load_iers_eop
from tools.validate_ccsds_oem import _load_data_receipt as load_oem_data_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REPO_ROOT / "sim/interchange/examples/oel_earth_eme2000_utc_v3.oem"
COVARIANCE_REFERENCE = (
    REPO_ROOT / "sim/interchange/examples/ccsds_502_0_b_3_g13_covariance_excerpt.oem"
)
FIXTURE_MANIFEST = REPO_ROOT / "sim/interchange/examples/ccsds_oem_validation_manifest.json"
EOP_FIXTURE = REPO_ROOT / "sim/dynamics/orbit/data/finals2000a_2024_01_01_04_excerpt.txt"


def test_reference_fixture_is_import_ready_and_round_trips_semantically(tmp_path: Path) -> None:
    message = read_oem_kvn(REFERENCE)
    inspection = inspect_oem(message)

    assert inspection["valid_oem"] is True
    assert inspection["oel_import_ready"] is True
    assert inspection["segment_count"] == 1
    assert inspection["state_count"] == 3
    assert inspection["source_sha256"] == hashlib.sha256(REFERENCE.read_bytes()).hexdigest()

    output = tmp_path / "roundtrip.oem"
    output.write_text(serialize_oem_kvn(message), encoding="utf-8")
    comparison = compare_oem(message, read_oem_kvn(output))

    assert comparison["status"] == "equivalent"
    assert comparison["max_abs_position_residual_km"] == 0.0
    assert comparison["max_abs_velocity_residual_km_s"] == 0.0


def test_official_covariance_example_parses_and_round_trips_semantically() -> None:
    message = read_oem_kvn(COVARIANCE_REFERENCE)
    covariance = message.segments[0].covariances[0]

    assert inspect_oem(message)["covariance_count"] == 1
    assert covariance.epoch == "2019-12-28T21:29:07.267"
    assert covariance.ref_frame == "EME2000"
    assert covariance.matrix[0][0] == pytest.approx(3.3313494e-4)
    assert covariance.matrix[5][0] == covariance.matrix[0][5]
    assert compare_oem(message, parse_oem_kvn(serialize_oem_kvn(message)))["status"] == "equivalent"


def test_validation_fixture_manifest_is_hash_bound_and_retained_report_passes() -> None:
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    root = FIXTURE_MANIFEST.parent
    for artifact in manifest["artifacts"]:
        path = (root / artifact["path"]).resolve()
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    report = json.loads((root / "ccsds_oem_orekit_13_1_7_acceptance.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["runtime"]["orekit_version"] == "13.1.7"
    assert all(report["checks"].values())
    assert report["externally_validated_fields"] == {
        "message": [
            "segment_count", "state_count", "object_name", "object_id",
            "center_name", "ref_frame", "time_system",
        ],
        "sampled_state": ["first_x", "first_vy", "last_x", "last_vy"],
        "covariance": [
            "covariance_count", "covariance_ref_frame", "covariance_00",
            "covariance_10", "covariance_55",
        ],
    }
    asserted_fields = {
        field
        for fields in report["externally_validated_fields"].values()
        for field in fields
    }
    assert asserted_fields == set(report["checks"])


def test_oem_reference_refresh_rejects_stale_orekit_data_receipt(tmp_path: Path) -> None:
    data_dir = tmp_path / "orekit-data"
    data_dir.mkdir()
    (data_dir / "tai-utc.dat").write_text("retained data\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"revision": "fixture", "source": "fixture", "tree_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="data tree does not match its receipt"):
        load_oem_data_receipt(receipt, data_dir=data_dir)


def test_import_creates_provenance_rich_packet_without_replaying_ephemeris() -> None:
    packet = oem_to_mission_input_packet(REFERENCE)
    data = packet.to_dict()
    obj = data["objects"]["2026-001A"]

    assert obj["frame"] == "ECI"
    assert obj["initial_state"]["position_eci_km"] == pytest.approx([7000.0, 0.0, 0.0])
    assert obj["initial_state"]["epoch_jd_utc"] == pytest.approx(2461282.0, rel=0.0, abs=1.0e-9)
    assert obj["ephemeris"]["first_jd_utc"] == pytest.approx(2461282.0, rel=0.0, abs=1.0e-9)
    assert obj["ephemeris"]["last_jd_utc"] == pytest.approx(2461282.001388889, rel=0.0, abs=1.0e-9)
    assert obj["ephemeris"]["source_metadata"]["source_ref_frame"] == "EME2000"
    assert obj["ephemeris"]["source_metadata"]["first_epoch_utc"] == "2026-08-29T12:00:00"
    assert obj["ephemeris"]["source_metadata"]["elapsed_time_basis"] == "TAI SI seconds from the first state epoch"
    assert obj["ephemeris"]["source_metadata"]["full_ephemeris_replayed"] is False
    assert any("first sample" in warning for warning in packet.warnings)


def test_import_elapsed_seconds_include_utc_leap_discontinuity() -> None:
    message = read_oem_kvn(REFERENCE)
    segment = message.segments[0]
    states = (
        replace(segment.states[0], epoch="2016-12-31T23:59:59"),
        replace(segment.states[1], epoch="2017-01-01T00:00:00"),
    )
    changed = replace(
        message,
        segments=(
            replace(
                segment,
                metadata=replace(
                    segment.metadata,
                    start_time=states[0].epoch,
                    stop_time=states[-1].epoch,
                    interpolation=None,
                    interpolation_degree=None,
                ),
                states=states,
            ),
        ),
    )

    obj = oem_to_mission_input_packet(changed).to_dict()["objects"]["2026-001A"]

    assert obj["ephemeris"]["last_time_s"] == 2.0
    assert obj["ephemeris"]["first_jd_utc"] == pytest.approx(2457754.499988426, rel=0.0, abs=1.0e-9)
    assert obj["ephemeris"]["last_jd_utc"] == pytest.approx(2457754.5, rel=0.0, abs=1.0e-9)
    assert obj["ephemeris"]["source_metadata"]["first_epoch_tai"] == "2017-01-01T00:00:35"


def test_import_preserves_uncalibrated_covariance_without_claiming_simulator_use() -> None:
    message = read_oem_kvn(REFERENCE)
    state = message.segments[0].states[0]
    covariance = OemCovariance(
        epoch=state.epoch,
        ref_frame="EME2000",
        matrix=tuple(tuple(value for value in row) for row in (np.eye(6) * 1.0e-6)),
    )
    changed = OemMessage(
        header=message.header,
        segments=(
            OemSegment(
                metadata=message.segments[0].metadata,
                states=message.segments[0].states,
                comments=message.segments[0].comments,
                covariances=(covariance,),
            ),
        ),
    )

    source = oem_to_mission_input_packet(changed).to_dict()["objects"]["2026-001A"]["ephemeris"]["source_metadata"]
    assert source["covariances"][0]["frame"] == "EME2000"
    assert source["covariances"][0]["calibrated"] is False
    assert source["full_ephemeris_replayed"] is False


def test_non_profile_frame_remains_inspectable_but_is_not_importable() -> None:
    message = read_oem_kvn(REFERENCE)
    metadata = message.segments[0].metadata
    changed = OemMessage(
        header=message.header,
        segments=(
            OemSegment(
                metadata=OemMetadata(**{**metadata.__dict__, "ref_frame": "TEME"}),
                states=message.segments[0].states,
                comments=message.segments[0].comments,
            ),
        ),
    )

    inspection = inspect_oem(changed)
    assert inspection["valid_oem"] is True
    assert inspection["oel_import_ready"] is False
    assert inspection["profile_issues"][0]["code"] == "unsupported_ref_frame"
    with pytest.raises(CcsdsOemError, match="REF_FRAME EME2000"):
        oem_to_mission_input_packet(changed)


def test_non_profile_time_system_remains_inspectable_but_is_not_importable() -> None:
    message = read_oem_kvn(REFERENCE)
    metadata = message.segments[0].metadata
    changed = OemMessage(
        header=message.header,
        segments=(
            OemSegment(
                metadata=OemMetadata(**{**metadata.__dict__, "time_system": "TAI"}),
                states=message.segments[0].states,
                comments=message.segments[0].comments,
            ),
        ),
    )

    inspection = inspect_oem(changed)
    assert inspection["valid_oem"] is True
    assert inspection["oel_import_ready"] is False
    assert inspection["profile_issues"][0]["code"] == "unsupported_time_system"
    with pytest.raises(CcsdsOemError, match="TIME_SYSTEM UTC"):
        oem_to_mission_input_packet(changed)


def test_explicit_gcrf_itrf_oem_conversion_round_trips_state_and_covariance() -> None:
    covariance = OemCovariance(
        epoch="2024-01-01T00:00:00",
        ref_frame="GCRF",
        matrix=tuple(tuple(value for value in row) for row in (np.eye(6) * 1.0e-6)),
    )
    source = OemMessage(
        header=OemHeader("3.0", "2024-01-01T00:00:00", "OEL"),
        segments=(
            OemSegment(
                metadata=OemMetadata(
                    object_name="VALIDATION SAT",
                    object_id="2024-001A",
                    center_name="EARTH",
                    ref_frame="GCRF",
                    time_system="UTC",
                    start_time="2024-01-01T00:00:00",
                    stop_time="2024-01-01T00:01:00",
                ),
                states=(
                    OemState("2024-01-01T00:00:00", (7000.0, 120.0, 30.0), (-0.2, 7.45, 1.1)),
                    OemState("2024-01-01T00:01:00", (6980.0, 566.0, 96.0), (-0.68, 7.42, 1.09)),
                ),
                covariances=(covariance,),
            ),
        ),
    )
    eop = load_iers_eop(EOP_FIXTURE)

    fixed = convert_oem(source, target_frame="ITRF", target_time_system="TAI", eop_series=eop)
    recovered = convert_oem(fixed, target_frame="GCRF", target_time_system="UTC", eop_series=eop)
    comparison = compare_oem(
        source,
        recovered,
        position_tolerance_km=1.0e-9,
        velocity_tolerance_km_s=1.0e-11,
        covariance_absolute_tolerance=1.0e-16,
    )

    assert fixed.segments[0].metadata.ref_frame == "ITRF"
    assert fixed.segments[0].metadata.time_system == "TAI"
    assert fixed.segments[0].states[0].epoch == "2024-01-01T00:00:37"
    assert comparison["status"] == "equivalent"


def test_itrf_oem_conversion_requires_epoch_covering_eop() -> None:
    source = read_oem_kvn(REFERENCE)
    with pytest.raises(CcsdsOemError, match="requires sampled DUT1"):
        convert_oem(source, target_frame="ITRF", target_time_system="UTC")


def test_calendar_and_day_of_year_epochs_compare_semantically() -> None:
    calendar = read_oem_kvn(REFERENCE)
    ordinal = parse_oem_kvn(REFERENCE.read_text(encoding="utf-8").replace("2026-08-29", "2026-241"))

    assert compare_oem(calendar, ordinal)["status"] == "equivalent"


@pytest.mark.parametrize("invalid_date", ["2023-366", "2024-367", "2024-000"])
def test_invalid_day_of_year_epochs_fail_closed(invalid_date: str) -> None:
    text = REFERENCE.read_text(encoding="utf-8").replace("2026-08-29", invalid_date)

    with pytest.raises(CcsdsOemError, match="not a valid CCSDS ordinal epoch"):
        parse_oem_kvn(text)


def test_reader_enforces_configured_byte_limit() -> None:
    with pytest.raises(CcsdsOemError, match="byte limit"):
        read_oem_kvn(REFERENCE, max_bytes=REFERENCE.stat().st_size - 1)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ("ORIGINATOR = OEL\nORIGINATOR = DUPLICATE", "duplicate OEM keyword"),
        ("7000 0 0 0 7.5 1", "state row must contain 7 or 10 fields"),
        ("2026-08-29T12:00:00 7000 NaN 0 0 7.5 1", "invalid numeric token"),
        ("2026-08-29T12:00:60 7000 0 0 0 7.5 1", "leap-second timestamp"),
    ],
)
def test_malformed_inputs_fail_closed(replacement: str, match: str) -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    if replacement.startswith("ORIGINATOR"):
        text = text.replace("ORIGINATOR = OEL", replacement)
    else:
        text = text.replace("2026-08-29T12:00:00 7000 0 0 0 7.5 1", replacement)
    with pytest.raises(CcsdsOemError, match=match):
        parse_oem_kvn(text)


def test_compare_uses_frozen_physical_tolerances() -> None:
    message = read_oem_kvn(REFERENCE)
    first = message.segments[0].states[0]
    changed_first = OemState(
        epoch=first.epoch,
        position_km=(first.position_km[0] + 2.0e-8, *first.position_km[1:]),
        velocity_km_s=first.velocity_km_s,
    )
    changed = OemMessage(
        header=message.header,
        segments=(
            OemSegment(
                metadata=message.segments[0].metadata,
                states=(changed_first, *message.segments[0].states[1:]),
                comments=message.segments[0].comments,
            ),
        ),
    )

    assert compare_oem(message, changed, position_tolerance_km=1.0e-7)["status"] == "equivalent"
    assert compare_oem(message, changed, position_tolerance_km=1.0e-9)["status"] == "different"


def test_completed_run_export_writes_oem_and_content_bound_receipt(tmp_path: Path) -> None:
    run_dir = _minimal_review_run(tmp_path)
    output = tmp_path / "exported.oem"

    receipt = export_completed_run_oem(
        run_dir,
        output_path=output,
        object_id="sat",
        object_name="TEST SAT",
        originator="OEL",
    )
    message = read_oem_kvn(output)

    assert receipt["status"] == "exported"
    assert receipt["source_frame"] == "OEL/ECI/J2000"
    assert receipt["oem_ref_frame"] == "EME2000"
    assert receipt["frame_transform_applied"] is False
    assert receipt["interpolation_declared"] is False
    assert receipt["state_count"] == 3
    assert receipt["oem_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert Path(receipt["receipt_path"]).is_file()
    assert message.segments[0].states[0].epoch == "2000-01-01T12:00:00"
    assert message.segments[0].states[-1].epoch == "2000-01-01T12:02:00"
    assert message.segments[0].metadata.interpolation is None
    assert message.segments[0].metadata.interpolation_degree is None
    assert compare_oem(message, output)["status"] == "equivalent"


def test_completed_run_export_rejects_noncanonical_frame(tmp_path: Path) -> None:
    run_dir = _minimal_review_run(tmp_path, state_frame="TEME")
    with pytest.raises(CcsdsOemError, match="canonical OEL ECI"):
        export_completed_run_oem(run_dir, output_path=tmp_path / "bad.oem", object_id="sat")


def test_completed_run_export_includes_matching_covariance_and_calibration_receipt(tmp_path: Path) -> None:
    run_dir = _minimal_review_run(tmp_path, with_covariance=True)
    output = tmp_path / "covariance.oem"

    receipt = export_completed_run_oem(run_dir, output_path=output, object_id="sat")
    message = read_oem_kvn(output)

    assert receipt["covariance_count"] == 1
    assert receipt["calibrated_covariance_count"] == 1
    assert receipt["covariance_calibration_scopes"] == ["synthetic-test-only"]
    assert len(message.segments[0].covariances) == 1
    assert message.segments[0].covariances[0].epoch == message.segments[0].states[1].epoch
    assert message.segments[0].covariances[0].matrix[3][3] == pytest.approx(4.0e-10)


def test_completed_run_export_preflights_receipt_conflict_before_writing_oem(tmp_path: Path) -> None:
    run_dir = _minimal_review_run(tmp_path)
    output = tmp_path / "exported.oem"
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    receipt.write_text("conflicting receipt\n", encoding="utf-8")

    with pytest.raises(CcsdsOemError, match="receipt exists with different content"):
        export_completed_run_oem(run_dir, output_path=output, object_id="sat")

    assert not output.exists()
    assert receipt.read_text(encoding="utf-8") == "conflicting receipt\n"


def test_epoch_precision_beyond_microseconds_is_rejected_if_nonzero() -> None:
    text = REFERENCE.read_text(encoding="utf-8").replace(
        "2026-08-29T12:00:00 7000",
        "2026-08-29T12:00:00.0000001 7000",
    )
    with pytest.raises(CcsdsOemError, match="precision through microseconds"):
        parse_oem_kvn(text)


def test_serializer_rejects_incomplete_interpolation_contract() -> None:
    state = OemState("2026-01-01T00:00:00", (7000.0, 0.0, 0.0), (0.0, 7.5, 0.0))
    message = OemMessage(
        header=OemHeader("3.0", "2026-01-01T00:00:00", "OEL"),
        segments=(
            OemSegment(
                metadata=OemMetadata(
                    object_name="SAT",
                    object_id="SAT",
                    center_name="EARTH",
                    ref_frame="EME2000",
                    time_system="UTC",
                    start_time=state.epoch,
                    stop_time=state.epoch,
                    interpolation="LAGRANGE",
                ),
                states=(state,),
            ),
        ),
    )
    with pytest.raises(CcsdsOemError, match="requires INTERPOLATION_DEGREE"):
        serialize_oem_kvn(message)


def test_covariance_rows_and_psd_fail_closed() -> None:
    text = COVARIANCE_REFERENCE.read_text(encoding="utf-8")
    with pytest.raises(CcsdsOemError, match="row 2 must contain exactly 2"):
        parse_oem_kvn(text.replace("4.6189273e-04 6.7824216e-04", "4.6189273e-04"))
    with pytest.raises(CcsdsOemError, match="diagonal variances"):
        parse_oem_kvn(text.replace("3.3313494e-04", "-3.3313494e-04", 1))

    message = read_oem_kvn(REFERENCE)
    tiny_covariance = np.eye(6) * 1.0e-13
    tiny_covariance[0, 0] = -1.0e-13
    changed = replace(
        message,
        segments=(
            replace(
                message.segments[0],
                covariances=(
                    OemCovariance(
                        epoch=message.segments[0].states[0].epoch,
                        ref_frame="EME2000",
                        matrix=tuple(tuple(float(value) for value in row) for row in tiny_covariance),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(CcsdsOemError, match="diagonal variances"):
        serialize_oem_kvn(changed)


def _minimal_review_run(
    tmp_path: Path,
    *,
    state_frame: str = "ECI",
    with_covariance: bool = False,
) -> Path:
    run_dir = tmp_path / "run"
    review_dir = run_dir / "review"
    review_dir.mkdir(parents=True)
    db_path = review_dir / "run.sqlite"
    config_text = json.dumps({"simulator": {"initial_jd_utc": 2451545.0}}, sort_keys=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE run_metadata (
                run_id TEXT, scenario_name TEXT, generated_utc TEXT,
                config_json TEXT, config_sha256 TEXT
            );
            CREATE TABLE object_state_frame (object_id TEXT, state_frame TEXT);
            CREATE TABLE object_state (
                sample_index INTEGER, time_s REAL, object_id TEXT,
                pos_x_eci_km REAL, pos_y_eci_km REAL, pos_z_eci_km REAL,
                vel_x_eci_km_s REAL, vel_y_eci_km_s REAL, vel_z_eci_km_s REAL
            );
            CREATE TABLE object_state_covariance (
                sample_index INTEGER, time_s REAL, object_id TEXT, frame TEXT,
                component_order_json TEXT, units_json TEXT, covariance_json TEXT,
                mathematically_valid INTEGER, calibrated INTEGER, calibration_scope TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO run_metadata VALUES (?, ?, ?, ?, ?)",
            (
                "run-001",
                "oem_export_test",
                "2026-08-29T12:00:00Z",
                config_text,
                hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
            ),
        )
        conn.execute("INSERT INTO object_state_frame VALUES (?, ?)", ("sat", state_frame))
        conn.executemany(
            "INSERT INTO object_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (0, 0.0, "sat", 7000.0, 0.0, 0.0, 0.0, 7.5, 1.0),
                (1, 60.0, "sat", 6985.0, 449.5, 59.9, -0.484, 7.484, 0.998),
                (2, 120.0, "sat", 6941.9, 897.1, 119.7, -0.966, 7.438, 0.992),
            ],
        )
        if with_covariance:
            covariance = [
                [4.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 9.0e-4, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.6e-3, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0e-10, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 9.0e-10, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.6e-9],
            ]
            conn.execute(
                "INSERT INTO object_state_covariance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    1,
                    60.0,
                    "sat",
                    "ECI",
                    json.dumps(["x", "y", "z", "vx", "vy", "vz"]),
                    json.dumps(["km", "km", "km", "km/s", "km/s", "km/s"]),
                    json.dumps(covariance),
                    1,
                    1,
                    "synthetic-test-only",
                ),
            )
    return run_dir
