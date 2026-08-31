from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sim.ccsds import OmmMessage, OpmMessage, compare_odm, parse_odm_kvn, read_odm_kvn, serialize_odm_kvn
from tools.public_export.runtime_hashing import sha256_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
JAVA_SOURCE = REPO_ROOT / "tools/external_reference/OrekitOdmAcceptance.java"
DEFAULT_OREKIT_ROOT = REPO_ROOT / "validation/external/phase4/orekit"
DEFAULT_OPM = REPO_ROOT / "sim/interchange/examples/oel_earth_eme2000_utc_v3.opm"
DEFAULT_OMM = REPO_ROOT / "sim/interchange/examples/oel_sgp4_mean_elements_v3.omm"


def validate_ccsds_odm_with_orekit(
    *,
    output_path: str | Path,
    orekit_root: str | Path = DEFAULT_OREKIT_ROOT,
    opm_path: str | Path = DEFAULT_OPM,
    omm_path: str | Path = DEFAULT_OMM,
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    runtime = Path(orekit_root).expanduser().resolve()
    opm_source = Path(opm_path).expanduser().resolve()
    omm_source = Path(omm_path).expanduser().resolve()
    jars = sorted((runtime / "lib").glob("*.jar"))
    data_dir = runtime / "data/orekit-data-main"
    if not jars or not data_dir.is_dir():
        raise RuntimeError(f"Pinned Orekit runtime is unavailable under {runtime}.")
    data_receipt = _load_data_receipt(
        runtime / "data/orekit-data-receipt.json",
        data_dir=data_dir,
    )
    classpath = os.pathsep.join(str(path) for path in jars)
    with tempfile.TemporaryDirectory(prefix="oel-odm-orekit-") as temporary:
        class_dir = Path(temporary)
        subprocess.run(
            [shutil.which("javac") or "javac", "-cp", classpath, "-d", str(class_dir), str(JAVA_SOURCE)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        completed = subprocess.run(
            [
                shutil.which("java") or "java",
                "-cp",
                os.pathsep.join([str(class_dir), classpath]),
                "OrekitOdmAcceptance",
                str(data_dir),
                str(opm_source),
                str(omm_source),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    orekit = _parse_output(completed.stdout)
    opm = read_odm_kvn(opm_source)
    omm = read_odm_kvn(omm_source)
    if not isinstance(opm, OpmMessage) or not isinstance(omm, OmmMessage):
        raise RuntimeError("Validation fixtures resolved to the wrong ODM message type.")
    position_m = np.asarray([float(opm.state[key]) for key in ("X", "Y", "Z")]) * 1000.0
    velocity_m_s = np.asarray([float(opm.state[key]) for key in ("X_DOT", "Y_DOT", "Z_DOT")]) * 1000.0
    orekit_position = _vector(orekit["opm_position_m"])
    orekit_velocity = _vector(orekit["opm_velocity_m_s"])
    mean_motion_rad_s = float(omm.mean_elements["MEAN_MOTION"]) * 2.0 * math.pi / 86400.0
    inclination_rad = math.radians(float(omm.mean_elements["INCLINATION"]))
    checks = {
        "opm_version": float(orekit["opm_version"]) == float(opm.header.version),
        "opm_object_id": orekit["opm_object_id"] == opm.metadata.object_id,
        "opm_frame": orekit["opm_frame"] == opm.metadata.ref_frame,
        "opm_maneuver_count": int(orekit["opm_maneuver_count"]) == len(opm.maneuvers),
        "opm_position": bool(np.allclose(orekit_position, position_m, rtol=0.0, atol=1.0e-9)),
        "opm_velocity": bool(np.allclose(orekit_velocity, velocity_m_s, rtol=0.0, atol=1.0e-12)),
        "opm_covariance_presence": (orekit["opm_covariance_present"] == "true") == (opm.covariance is not None),
        "opm_roundtrip": compare_odm(opm, parse_odm_kvn(serialize_odm_kvn(opm)))["status"] == "equivalent",
        "omm_version": float(orekit["omm_version"]) == float(omm.header.version),
        "omm_object_id": orekit["omm_object_id"] == omm.metadata.object_id,
        "omm_frame": orekit["omm_frame"] == omm.metadata.ref_frame,
        "omm_theory": orekit["omm_theory"] == omm.metadata.mean_element_theory,
        "omm_mean_motion": math.isclose(float(orekit["omm_mean_motion_rad_s"]), mean_motion_rad_s, abs_tol=1.0e-18),
        "omm_eccentricity": math.isclose(
            float(orekit["omm_eccentricity"]),
            float(omm.mean_elements["ECCENTRICITY"]),
            abs_tol=1.0e-15,
        ),
        "omm_inclination": math.isclose(float(orekit["omm_inclination_rad"]), inclination_rad, abs_tol=1.0e-15),
        "omm_norad_id": int(orekit["omm_norad_id"]) == int(omm.tle_parameters["NORAD_CAT_ID"]),
        "omm_roundtrip": compare_odm(omm, parse_odm_kvn(serialize_odm_kvn(omm)))["status"] == "equivalent",
    }
    report = {
        "schema": "oel.ccsds-opm-omm-orekit-validation.v1",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "pass" if all(checks.values()) else "fail",
        "claim": "OEL and pinned Orekit 13.1.7 agree on the exact OPM and OMM fields enumerated by this frozen selected-field cross-read.",
        "externally_validated_fields": {
            "opm": [
                "version", "object_id", "frame", "maneuver_count", "position",
                "velocity", "covariance_presence",
            ],
            "omm": [
                "version", "object_id", "frame", "mean_element_theory",
                "mean_motion", "eccentricity", "inclination", "norad_id",
            ],
        },
        "checks": checks,
        "fixtures": {
            "opm": {"path": str(opm_source.relative_to(REPO_ROOT)), "sha256": _sha256(opm_source)},
            "omm": {"path": str(omm_source.relative_to(REPO_ROOT)), "sha256": _sha256(omm_source)},
        },
        "orekit": orekit,
        "runtime": {
            "provider": "Orekit",
            "orekit_version": next(path.stem.removeprefix("orekit-") for path in jars if path.name.startswith("orekit-")),
            "java_source": str(JAVA_SOURCE.relative_to(REPO_ROOT)),
            "java_source_sha256": _sha256(JAVA_SOURCE),
            "jars": [{"name": path.name, "sha256": _sha256(path)} for path in jars],
            "orekit_data_revision": data_receipt["revision"],
            "orekit_data_tree_sha256": data_receipt["tree_sha256"],
        },
        "non_claims": [
            "Syntax and frozen semantic agreement do not establish orbit, maneuver, or covariance accuracy.",
            "OMM remains a preserved mean-element product; this validation does not authorize silent osculating conversion.",
            "The bounded public profile is not a claim of complete CCSDS 502.0-B-3 coverage.",
            "Supported fields not enumerated in externally_validated_fields have OEL-only contract and round-trip coverage.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parse_output(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw_line in text.splitlines():
        key, value = raw_line.split("=", 1)
        rows[key] = value
    return rows


def _vector(value: str) -> np.ndarray:
    vector = np.asarray([float(item) for item in value.split(",")], dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise RuntimeError("Orekit vector output must contain three finite values.")
    return vector


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_data_receipt(path: Path, *, data_dir: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Orekit data receipt is unavailable or invalid: {path}") from exc
    required = {"revision", "source", "tree_sha256"}
    missing = sorted(required - set(dict(payload)))
    if missing:
        raise RuntimeError(f"Orekit data receipt is missing: {missing}")
    receipt = {key: str(payload[key]) for key in required}
    expected = receipt["tree_sha256"].strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError("Orekit data receipt tree_sha256 must be a lowercase SHA-256 digest.")
    actual = sha256_tree(data_dir)
    if actual != expected:
        raise RuntimeError(
            f"Orekit data tree does not match its receipt: expected {expected}, actual {actual}."
        )
    receipt["tree_sha256"] = actual
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate bounded OPM/OMM KVN support with pinned Orekit.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orekit-root", type=Path, default=DEFAULT_OREKIT_ROOT)
    parser.add_argument("--opm", type=Path, default=DEFAULT_OPM)
    parser.add_argument("--omm", type=Path, default=DEFAULT_OMM)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_ccsds_odm_with_orekit(
        output_path=args.output,
        orekit_root=args.orekit_root,
        opm_path=args.opm,
        omm_path=args.omm,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
