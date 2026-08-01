from __future__ import annotations

import json
from pathlib import Path

import pytest

sgp4_api = pytest.importorskip("sgp4.api")


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_POSITION_COMPONENT_DELTA_KM = 1.0e-8
MAX_VELOCITY_COMPONENT_DELTA_KM_S = 2.0e-12


def _reference_paths() -> list[Path]:
    public_fixture_paths = sorted((REPO_ROOT / "sim" / "tests" / "fixtures" / "sgp4_2_23_reference").glob("*.json"))
    ogp_paths = sorted((REPO_ROOT / "validation" / "data" / "ogp_reference").glob("*python_sgp4_2_23.json"))
    cosmos_path = REPO_ROOT / "validation" / "data" / "sgp4_reference" / "cosmos_2428_python_sgp4_2_23.json"
    if not cosmos_path.is_file():
        return public_fixture_paths
    return [
        *ogp_paths,
        cosmos_path,
    ]


@pytest.mark.parametrize("reference_path", _reference_paths(), ids=lambda path: path.stem)
def test_approved_sgp4_matches_saved_2_23_reference_vectors(
    reference_path: Path,
) -> None:
    """Keep dependency upgrades below the documented native-TEME parity limits."""

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    tle = reference["tle"]
    satellite = sgp4_api.Satrec.twoline2rv(tle["line1"], tle["line2"])

    for sample in reference["samples"]:
        error, position, velocity = satellite.sgp4_tsince(sample["tsince_min"])
        assert error == 0

        position_delta = max(
            abs(actual - expected) for actual, expected in zip(position, sample["position_teme_km"], strict=True)
        )
        velocity_delta = max(
            abs(actual - expected) for actual, expected in zip(velocity, sample["velocity_teme_km_s"], strict=True)
        )
        assert position_delta <= MAX_POSITION_COMPONENT_DELTA_KM
        assert velocity_delta <= MAX_VELOCITY_COMPONENT_DELTA_KM_S
