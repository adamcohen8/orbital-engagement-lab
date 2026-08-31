from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sim.ccsds import inspect_oem, read_oem_kvn
from tools.public_export.runtime_hashing import sha256_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
JAVA_SOURCE = REPO_ROOT / "tools/external_reference/OrekitOemAcceptance.java"
DEFAULT_OEM = REPO_ROOT / "sim/interchange/examples/oel_earth_eme2000_utc_v3.oem"
DEFAULT_COVARIANCE_OEM = (
    REPO_ROOT / "sim/interchange/examples/ccsds_502_0_b_3_g13_covariance_excerpt.oem"
)
DEFAULT_OREKIT_ROOT = REPO_ROOT / "validation/external/phase4/orekit"
EARTH_MU_M3_S2 = 3.986004418e14
MARS_MU_M3_S2 = 4.282837e13


def validate_with_orekit(
    oem_path: str | Path,
    *,
    covariance_oem_path: str | Path = DEFAULT_COVARIANCE_OEM,
    output_path: str | Path,
    orekit_root: str | Path = DEFAULT_OREKIT_ROOT,
) -> dict[str, Any]:
    source = Path(oem_path).expanduser().resolve()
    covariance_source = Path(covariance_oem_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    runtime = Path(orekit_root).expanduser().resolve()
    library_dir = runtime / "lib"
    data_dir = runtime / "data/orekit-data-main"
    jars = sorted(library_dir.glob("*.jar"))
    if not jars or not data_dir.is_dir():
        raise RuntimeError(f"Pinned Orekit runtime is unavailable under {runtime}.")
    data_receipt = _load_data_receipt(
        runtime / "data/orekit-data-receipt.json",
        data_dir=data_dir,
    )
    classpath = os.pathsep.join(str(path) for path in jars)
    with tempfile.TemporaryDirectory(prefix="oel-oem-orekit-") as temporary:
        class_dir = Path(temporary)
        subprocess.run(
            [shutil.which("javac") or "javac", "-cp", classpath, "-d", str(class_dir), str(JAVA_SOURCE)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        orekit = _run_orekit(class_dir, classpath, source, data_dir, EARTH_MU_M3_S2)
        covariance_orekit = _run_orekit(
            class_dir,
            classpath,
            covariance_source,
            data_dir,
            MARS_MU_M3_S2,
        )
    orekit_jar = next((path for path in jars if path.name.startswith("orekit-")), None)
    if orekit_jar is not None:
        orekit["orekit_version"] = orekit_jar.stem.removeprefix("orekit-")
    message = read_oem_kvn(source)
    covariance_message = read_oem_kvn(covariance_source)
    inspection = inspect_oem(message)
    first = message.segments[0].states[0]
    last = message.segments[0].states[-1]
    covariance = covariance_message.segments[0].covariances[0]
    checks = {
        "segment_count": int(orekit["segment_count"]) == inspection["segment_count"],
        "state_count": int(orekit["state_count"]) == inspection["state_count"],
        "object_name": orekit["object_name"] == message.segments[0].metadata.object_name,
        "object_id": orekit["object_id"] == message.segments[0].metadata.object_id,
        "center_name": orekit["center_name"].upper() == message.segments[0].metadata.center_name.upper(),
        "ref_frame": orekit["ref_frame"].upper() == message.segments[0].metadata.ref_frame.upper(),
        "time_system": orekit["time_system"].upper() == message.segments[0].metadata.time_system.upper(),
        "first_x": abs(float(orekit["first_x_m"]) / 1000.0 - first.position_km[0]) <= 1.0e-12,
        "first_vy": abs(float(orekit["first_vy_m_s"]) / 1000.0 - first.velocity_km_s[1]) <= 1.0e-15,
        "last_x": abs(float(orekit["last_x_m"]) / 1000.0 - last.position_km[0]) <= 1.0e-12,
        "last_vy": abs(float(orekit["last_vy_m_s"]) / 1000.0 - last.velocity_km_s[1]) <= 1.0e-15,
        "covariance_count": int(covariance_orekit["covariance_count"]) == 1,
        "covariance_ref_frame": covariance_orekit["first_cov_ref_frame"].upper()
        == str(covariance.ref_frame).upper(),
        "covariance_00": abs(float(covariance_orekit["first_cov_00_si"]) / 1.0e6 - covariance.matrix[0][0])
        <= 1.0e-18,
        "covariance_10": abs(float(covariance_orekit["first_cov_10_si"]) / 1.0e6 - covariance.matrix[1][0])
        <= 1.0e-18,
        "covariance_55": abs(float(covariance_orekit["first_cov_55_si"]) / 1.0e6 - covariance.matrix[5][5])
        <= 1.0e-18,
    }
    report = {
        "schema": "oel.ccsds-oem-orekit-validation.v2",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "pass" if all(checks.values()) else "fail",
        "claim": "Orekit and OEL independently parse the retained OEM KVN fixtures with matching values for the explicitly enumerated metadata, sampled state, and covariance fields.",
        "externally_validated_fields": {
            "message": [
                "segment_count", "state_count", "object_name", "object_id",
                "center_name", "ref_frame", "time_system",
            ],
            "sampled_state": ["first_x", "first_vy", "last_x", "last_vy"],
            "covariance": [
                "covariance_count", "covariance_ref_frame", "covariance_00",
                "covariance_10", "covariance_55",
            ],
        },
        "non_claims": [
            "This comparison does not validate orbit accuracy, interpolation accuracy, covariance calibration, XML, or frame conversion.",
            "Supported fields not enumerated in externally_validated_fields have OEL-only contract and round-trip coverage.",
            "Shared CCSDS grammar interpretation is cross-consumer evidence, not physical truth.",
        ],
        "sources": {
            "states": {
                "path": str(source.relative_to(REPO_ROOT)),
                "sha256": _sha256(source),
            },
            "covariance": {
                "path": str(covariance_source.relative_to(REPO_ROOT)),
                "sha256": _sha256(covariance_source),
            },
        },
        "oel": {
            "states": inspection,
            "covariance": inspect_oem(covariance_message),
        },
        "orekit": {
            "states": orekit,
            "covariance": covariance_orekit,
        },
        "checks": checks,
        "runtime": {
            "provider": "Orekit",
            "orekit_version": orekit["orekit_version"],
            "java_source": str(JAVA_SOURCE.relative_to(REPO_ROOT)),
            "java_source_sha256": _sha256(JAVA_SOURCE),
            "jars": [{"name": path.name, "sha256": _sha256(path)} for path in jars],
            "orekit_data_revision": data_receipt["revision"],
            "orekit_data_source": data_receipt["source"],
            "orekit_data_tree_sha256": data_receipt["tree_sha256"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _run_orekit(
    class_dir: Path,
    classpath: str,
    source: Path,
    data_dir: Path,
    center_mu_m3_s2: float,
) -> dict[str, str]:
    completed = subprocess.run(
        [
            shutil.which("java") or "java",
            "-cp",
            os.pathsep.join([str(class_dir), classpath]),
            "OrekitOemAcceptance",
            str(source),
            str(data_dir),
            repr(center_mu_m3_s2),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_output(completed.stdout)


def _parse_output(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if "=" not in raw_line:
            raise RuntimeError(f"Unexpected Orekit output line: {raw_line!r}")
        key, value = raw_line.split("=", 1)
        if key in rows:
            raise RuntimeError(f"Duplicate Orekit output key: {key!r}")
        rows[key] = value
    required = {
        "orekit_version",
        "segment_count",
        "state_count",
        "covariance_count",
        "object_name",
        "object_id",
        "center_name",
        "ref_frame",
        "time_system",
        "first_x_m",
        "first_vy_m_s",
        "last_x_m",
        "last_vy_m_s",
    }
    missing = sorted(required - set(rows))
    if missing:
        raise RuntimeError(f"Orekit OEM acceptance output is missing: {missing}")
    return rows


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
    parser = argparse.ArgumentParser(description="Validate an OEL OEM KVN fixture with pinned Orekit.")
    parser.add_argument("--oem", type=Path, default=DEFAULT_OEM)
    parser.add_argument("--covariance-oem", type=Path, default=DEFAULT_COVARIANCE_OEM)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orekit-root", type=Path, default=DEFAULT_OREKIT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_with_orekit(
        args.oem,
        covariance_oem_path=args.covariance_oem,
        output_path=args.output,
        orekit_root=args.orekit_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
