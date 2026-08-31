from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sim.ccsds import (
    CcsdsOdmError,
    OmmMessage,
    OpmMessage,
    compare_odm,
    inspect_odm,
    opm_to_mission_input_packet,
    parse_odm_kvn,
    read_odm_kvn,
    serialize_odm_kvn,
)
from tools.validate_ccsds_odm import _load_data_receipt as load_odm_data_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
OPM = REPO_ROOT / "sim/interchange/examples/oel_earth_eme2000_utc_v3.opm"
OMM = REPO_ROOT / "sim/interchange/examples/oel_sgp4_mean_elements_v3.omm"
MANIFEST = REPO_ROOT / "sim/interchange/examples/ccsds_odm_validation_manifest.json"


def test_opm_parses_round_trips_and_is_ready_for_bounded_import() -> None:
    message = read_odm_kvn(OPM)
    inspection = inspect_odm(message)

    assert isinstance(message, OpmMessage)
    assert inspection["valid"] is True
    assert inspection["mission_input_ready"] is True
    assert inspection["maneuver_count"] == 1
    assert inspection["covariance_present"] is True
    assert message.source_sha256 == hashlib.sha256(OPM.read_bytes()).hexdigest()
    assert compare_odm(message, parse_odm_kvn(serialize_odm_kvn(message)))["status"] == "equivalent"


def test_omm_preserves_mean_elements_without_silent_state_conversion() -> None:
    message = read_odm_kvn(OMM)
    inspection = inspect_odm(message)

    assert isinstance(message, OmmMessage)
    assert inspection["mean_element_theory"] == "SGP4"
    assert inspection["mission_input_ready"] is False
    assert message.mean_elements["MEAN_MOTION"] == "15.25"
    assert message.tle_parameters["BSTAR"] == "1.2e-5"
    assert compare_odm(message, parse_odm_kvn(serialize_odm_kvn(message)))["status"] == "equivalent"


def test_opm_import_preserves_maneuver_and_covariance_as_nonexecuted_provenance() -> None:
    packet = opm_to_mission_input_packet(OPM).to_dict()
    source = packet["objects"]["2024-001A"]["ephemeris"]["source_metadata"]

    assert source["maneuvers_preserved_not_scheduled"][0]["MAN_DV_2"] == "0.001"
    assert source["covariance_preserved_not_calibrated"][0][0] == pytest.approx(1.0e-6)
    assert packet["objects"]["2024-001A"]["initial_state"]["position_eci_km"] == [7000.0, 120.0, 30.0]


@pytest.mark.parametrize("unit", ["m", "bananas"])
def test_opm_import_rejects_noncanonical_state_units(unit: str) -> None:
    text = OPM.read_text(encoding="utf-8").replace("X = 7000 [km]", f"X = 7000 [{unit}]")

    with pytest.raises(CcsdsOdmError, match="X unit must be km"):
        parse_odm_kvn(text)


def test_covariance_units_are_validated_and_preserved_by_round_trip() -> None:
    text = OPM.read_text(encoding="utf-8").replace("CX_X = 1e-6", "CX_X = 1e-6 [km**2]")
    message = parse_odm_kvn(text)
    serialized = serialize_odm_kvn(message)

    assert "CX_X = 9.9999999999999995e-07 [km**2]" in serialized
    assert compare_odm(message, parse_odm_kvn(serialized))["status"] == "equivalent"
    with pytest.raises(CcsdsOdmError, match=r"CX_X unit must be km\*\*2"):
        parse_odm_kvn(text.replace("[km**2]", "[m**2]", 1))


def test_user_defined_units_are_preserved_and_compared_semantically() -> None:
    text = OPM.read_text(encoding="utf-8").replace(
        "USER_DEFINED_EARTH_MODEL = WGS-84",
        "USER_DEFINED_EARTH_MODEL = 1 [model]",
    )
    message = parse_odm_kvn(text)
    serialized = serialize_odm_kvn(message)

    assert "USER_DEFINED_EARTH_MODEL = 1 [model]" in serialized
    assert compare_odm(message, parse_odm_kvn(serialized))["status"] == "equivalent"
    changed = parse_odm_kvn(serialized.replace("1 [model]", "1 [other-model]"))
    assert compare_odm(message, changed)["status"] == "different"


def test_numeric_formatting_does_not_change_semantic_comparison() -> None:
    changed = parse_odm_kvn(OPM.read_text(encoding="utf-8").replace("X = 7000 [km]", "X = 7.000e3 [km]"))

    assert compare_odm(read_odm_kvn(OPM), changed)["status"] == "equivalent"


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ("X = NaN [km]", "finite decimal"),
        ("UNSUPPORTED_KEY = 1", "Unsupported OPM keywords"),
        ("CY_Y =", "empty ODM key or value"),
        ("TIME_SYSTEM = TDB", "supports UTC, TAI, TT, and UT1"),
        ("OUT_OF_ORDER_COVARIANCE", "out-of-order OPM keyword COV_REF_FRAME"),
    ],
)
def test_malformed_or_out_of_profile_opm_fails_closed(replacement: str, match: str) -> None:
    text = OPM.read_text(encoding="utf-8")
    if replacement.startswith("X ="):
        text = text.replace("X = 7000 [km]", replacement)
    elif replacement.startswith("CY_Y"):
        text = text.replace("CY_Y = 2e-6", replacement)
    elif replacement.startswith("TIME_SYSTEM"):
        text = text.replace("TIME_SYSTEM = UTC", replacement)
    elif replacement == "OUT_OF_ORDER_COVARIANCE":
        covariance = text[text.index("COV_REF_FRAME = EME2000") : text.index("MAN_EPOCH_IGNITION")]
        text = text.replace(covariance, "").replace("USER_DEFINED_EARTH_MODEL", covariance + "USER_DEFINED_EARTH_MODEL")
    else:
        text = text.replace("USER_DEFINED_SOURCE_NOTE = PUBLIC VALIDATION FIXTURE", replacement)
    with pytest.raises(CcsdsOdmError, match=match):
        parse_odm_kvn(text)


def test_non_eme2000_opm_is_inspectable_but_not_import_ready() -> None:
    message = parse_odm_kvn(OPM.read_text(encoding="utf-8").replace("REF_FRAME = EME2000", "REF_FRAME = GCRF"))

    assert inspect_odm(message)["mission_input_ready"] is False
    with pytest.raises(CcsdsOdmError, match="explicit conversion"):
        opm_to_mission_input_packet(message)


def test_tiny_negative_odm_variance_fails_closed() -> None:
    text = OPM.read_text(encoding="utf-8").replace("CX_X = 1e-6", "CX_X = -1e-13")

    with pytest.raises(CcsdsOdmError, match="diagonal variances"):
        parse_odm_kvn(text)


def test_odm_validation_manifest_is_hash_bound_and_retained_report_passes() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = (MANIFEST.parent / artifact["path"]).resolve()
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    report = json.loads((MANIFEST.parent / "ccsds_odm_orekit_13_1_7_acceptance.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["runtime"]["orekit_version"] == "13.1.7"
    assert all(report["checks"].values())
    assert report["externally_validated_fields"] == {
        "opm": [
            "version", "object_id", "frame", "maneuver_count", "position",
            "velocity", "covariance_presence",
        ],
        "omm": [
            "version", "object_id", "frame", "mean_element_theory",
            "mean_motion", "eccentricity", "inclination", "norad_id",
        ],
    }
    mapped_checks = {
        "opm_version", "opm_object_id", "opm_frame", "opm_maneuver_count",
        "opm_position", "opm_velocity", "opm_covariance_presence",
        "omm_version", "omm_object_id", "omm_frame", "omm_theory",
        "omm_mean_motion", "omm_eccentricity", "omm_inclination", "omm_norad_id",
    }
    assert set(report["checks"]) - mapped_checks == {"opm_roundtrip", "omm_roundtrip"}


def test_odm_reference_refresh_rejects_stale_orekit_data_receipt(tmp_path: Path) -> None:
    data_dir = tmp_path / "orekit-data"
    data_dir.mkdir()
    (data_dir / "tai-utc.dat").write_text("retained data\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"revision": "fixture", "source": "fixture", "tree_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="data tree does not match its receipt"):
        load_odm_data_receipt(receipt, data_dir=data_dir)
