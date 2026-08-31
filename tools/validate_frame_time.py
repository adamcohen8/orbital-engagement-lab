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

import numpy as np

from sim.dynamics.orbit.eop import load_iers_eop
from sim.frame_time import (
    EarthOrientation,
    FrameTransformContext,
    TimeScale,
    epoch_conversion_receipt,
    format_epoch,
    frame_transform_receipt,
    leap_second_table_receipt,
    parse_epoch,
    state_transform_matrix,
    transform_cartesian_state,
)
from tools.public_export.runtime_hashing import sha256_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
JAVA_SOURCE = REPO_ROOT / "tools/external_reference/OrekitFrameTimeAcceptance.java"
DEFAULT_OREKIT_ROOT = REPO_ROOT / "validation/external/phase4/orekit"
EPOCH_UTC = "2024-01-01T00:00:00"
POSITION_KM = np.array([7000.0, 120.0, 30.0], dtype=float)
VELOCITY_KM_S = np.array([-0.2, 7.45, 1.1], dtype=float)
EOP_FIXTURE = REPO_ROOT / "sim/dynamics/orbit/data/finals2000a_2024_01_01_04_excerpt.txt"


def validate_frame_time_with_orekit(
    *,
    output_path: str | Path,
    orekit_root: str | Path = DEFAULT_OREKIT_ROOT,
) -> dict[str, Any]:
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
    with tempfile.TemporaryDirectory(prefix="oel-frame-time-orekit-") as temporary:
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
                "OrekitFrameTimeAcceptance",
                str(data_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    orekit = _parse_output(completed.stdout)
    epoch = parse_epoch(EPOCH_UTC, TimeScale.UTC)
    earth_orientation = EarthOrientation(
        dut1_s=float(orekit["dut1_s"]),
        xp_arcsec=float(orekit["xp_arcsec"]),
        yp_arcsec=float(orekit["yp_arcsec"]),
        ddpsi_rad=float(orekit["ddpsi_rad"]),
        ddeps_rad=float(orekit["ddeps_rad"]),
        source=f"Orekit {orekit['orekit_version']} IERS 1996 EOP sample",
        source_sha256=data_receipt["tree_sha256"],
    )
    context = FrameTransformContext(epoch=epoch, earth_orientation=earth_orientation)
    iers_series = load_iers_eop(EOP_FIXTURE)
    gcrf_context = FrameTransformContext(epoch=epoch, earth_orientation=iers_series.sample(epoch))

    oel_itrf_position, oel_itrf_velocity = transform_cartesian_state(
        POSITION_KM,
        VELOCITY_KM_S,
        "EME2000",
        "ITRF",
        context=context,
    )
    oel_teme_position, oel_teme_velocity = transform_cartesian_state(
        POSITION_KM,
        VELOCITY_KM_S,
        "TEME",
        "EME2000",
        context=context,
    )
    oel_jacobian = state_transform_matrix("EME2000", "ITRF", context=context)
    orekit_itrf_position = _vector(orekit["eme2000_to_itrf_position_m"], 3) / 1000.0
    orekit_itrf_velocity = _vector(orekit["eme2000_to_itrf_velocity_m_s"], 3) / 1000.0
    orekit_teme_position = _vector(orekit["teme_to_eme2000_position_m"], 3) / 1000.0
    orekit_teme_velocity = _vector(orekit["teme_to_eme2000_velocity_m_s"], 3) / 1000.0
    orekit_jacobian = _vector(orekit["eme2000_to_itrf_jacobian"], 36).reshape(6, 6)

    oel_gcrf_itrf_position, oel_gcrf_itrf_velocity = transform_cartesian_state(
        POSITION_KM,
        VELOCITY_KM_S,
        "GCRF",
        "ITRF",
        context=gcrf_context,
    )
    oel_gcrf_eme_position, oel_gcrf_eme_velocity = transform_cartesian_state(
        POSITION_KM,
        VELOCITY_KM_S,
        "GCRF",
        "EME2000",
        context=gcrf_context,
    )
    oel_gcrf_jacobian = state_transform_matrix("GCRF", "ITRF", context=gcrf_context)
    orekit_gcrf_itrf_position = _vector(orekit["gcrf_to_itrf_position_m"], 3) / 1000.0
    orekit_gcrf_itrf_velocity = _vector(orekit["gcrf_to_itrf_velocity_m_s"], 3) / 1000.0
    orekit_gcrf_eme_position = _vector(orekit["gcrf_to_eme2000_position_m"], 3) / 1000.0
    orekit_gcrf_eme_velocity = _vector(orekit["gcrf_to_eme2000_velocity_m_s"], 3) / 1000.0
    orekit_gcrf_jacobian = _vector(orekit["gcrf_to_itrf_jacobian"], 36).reshape(6, 6)

    frame_position_max_m = float(np.max(np.abs(oel_itrf_position - orekit_itrf_position)) * 1000.0)
    frame_velocity_max_m_s = float(np.max(np.abs(oel_itrf_velocity - orekit_itrf_velocity)) * 1000.0)
    teme_position_max_m = float(np.max(np.abs(oel_teme_position - orekit_teme_position)) * 1000.0)
    teme_velocity_max_m_s = float(np.max(np.abs(oel_teme_velocity - orekit_teme_velocity)) * 1000.0)
    jacobian_max = float(np.max(np.abs(oel_jacobian - orekit_jacobian)))
    gcrf_position_max_m = float(np.max(np.abs(oel_gcrf_itrf_position - orekit_gcrf_itrf_position)) * 1000.0)
    gcrf_velocity_max_m_s = float(
        np.max(np.abs(oel_gcrf_itrf_velocity - orekit_gcrf_itrf_velocity)) * 1000.0
    )
    gcrf_eme_position_max_m = float(np.max(np.abs(oel_gcrf_eme_position - orekit_gcrf_eme_position)) * 1000.0)
    gcrf_eme_velocity_max_m_s = float(
        np.max(np.abs(oel_gcrf_eme_velocity - orekit_gcrf_eme_velocity)) * 1000.0
    )
    gcrf_jacobian_max = float(np.max(np.abs(oel_gcrf_jacobian - orekit_gcrf_jacobian)))

    covariance = np.diag([400.0, 900.0, 1600.0, 4.0e-4, 9.0e-4, 1.6e-3])
    oel_covariance = oel_jacobian @ covariance @ oel_jacobian.T
    orekit_covariance = orekit_jacobian @ covariance @ orekit_jacobian.T
    covariance_scale = np.sqrt(np.outer(np.diag(orekit_covariance), np.diag(orekit_covariance)))
    covariance_normalized_max = float(np.max(np.abs(oel_covariance - orekit_covariance) / covariance_scale))

    before = parse_epoch("2016-12-31T23:59:59", "UTC")
    leap = parse_epoch("2016-12-31T23:59:60", "UTC")
    after = parse_epoch("2017-01-01T00:00:00", "UTC")
    current = parse_epoch("2026-08-29T12:00:00", "UTC")
    checks = {
        "utc_before": _without_zero_fraction(orekit["utc_before"]) == format_epoch(before, "UTC"),
        "utc_leap": _without_zero_fraction(orekit["utc_leap"]) == format_epoch(leap, "UTC"),
        "utc_after": _without_zero_fraction(orekit["utc_after"]) == format_epoch(after, "UTC"),
        "seconds_before_to_leap": abs((leap.tai_seconds - before.tai_seconds) - float(orekit["seconds_before_to_leap"])) <= 1.0e-12,
        "seconds_leap_to_after": abs((after.tai_seconds - leap.tai_seconds) - float(orekit["seconds_leap_to_after"])) <= 1.0e-12,
        "utc_minus_tai_2026": abs(-float(epoch_conversion_receipt(current, "TAI")["tai_minus_utc_s"]) - float(orekit["utc_minus_tai_2026_s"])) <= 1.0e-12,
        "tt_minus_tai_2026": abs(32.184 - float(orekit["tt_minus_tai_2026_s"])) <= 1.0e-12,
        "eme2000_itrf_position": frame_position_max_m <= 2.0,
        "eme2000_itrf_velocity": frame_velocity_max_m_s <= 0.003,
        "teme_eme2000_position": teme_position_max_m <= 2.0,
        "teme_eme2000_velocity": teme_velocity_max_m_s <= 0.003,
        "state_jacobian": jacobian_max <= 2.0e-7,
        "covariance_jacobian": covariance_normalized_max <= 2.0e-6,
        "gcrf_itrf_position": gcrf_position_max_m <= 0.25,
        "gcrf_itrf_velocity": gcrf_velocity_max_m_s <= 0.0005,
        "gcrf_eme2000_position": gcrf_eme_position_max_m <= 1.0e-5,
        "gcrf_eme2000_velocity": gcrf_eme_velocity_max_m_s <= 1.0e-8,
        "gcrf_state_jacobian": gcrf_jacobian_max <= 5.0e-8,
    }
    report = {
        "schema": "oel.frame-time-orekit-validation.v1",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "pass" if all(checks.values()) else "fail",
        "claim": "OEL's bounded IAU-76/FK5 + IAU-80 and IAU 2006/2000A frame/time contracts agree with pinned Orekit 13.1.7 inside frozen residual envelopes.",
        "non_claims": [
            "This does not validate EOP prediction, long-horizon EOP interpolation accuracy, or covariance calibration.",
            "Orekit comparison is independent implementation evidence, not operational qualification.",
        ],
        "checks": checks,
        "metrics": {
            "eme2000_itrf_position_max_m": frame_position_max_m,
            "eme2000_itrf_velocity_max_m_s": frame_velocity_max_m_s,
            "teme_eme2000_position_max_m": teme_position_max_m,
            "teme_eme2000_velocity_max_m_s": teme_velocity_max_m_s,
            "state_jacobian_max_abs": jacobian_max,
            "covariance_normalized_max": covariance_normalized_max,
            "gcrf_itrf_position_max_m": gcrf_position_max_m,
            "gcrf_itrf_velocity_max_m_s": gcrf_velocity_max_m_s,
            "gcrf_eme2000_position_max_m": gcrf_eme_position_max_m,
            "gcrf_eme2000_velocity_max_m_s": gcrf_eme_velocity_max_m_s,
            "gcrf_state_jacobian_max_abs": gcrf_jacobian_max,
        },
        "tolerances": {
            "frame_position_max_m": 2.0,
            "frame_velocity_max_m_s": 0.003,
            "state_jacobian_max_abs": 2.0e-7,
            "covariance_normalized_max": 2.0e-6,
            "gcrf_frame_position_max_m": 0.25,
            "gcrf_frame_velocity_max_m_s": 0.0005,
            "gcrf_state_jacobian_max_abs": 5.0e-8,
        },
        "input": {
            "epoch_utc": EPOCH_UTC,
            "position_km": POSITION_KM.tolist(),
            "velocity_km_s": VELOCITY_KM_S.tolist(),
            "covariance_diagonal_m_m_s": np.diag(covariance).tolist(),
        },
        "oel": {
            "leap_second_table": leap_second_table_receipt(),
            "frame_transform": frame_transform_receipt("EME2000", "ITRF", context=context),
            "gcrf_frame_transform": frame_transform_receipt("GCRF", "ITRF", context=gcrf_context),
            "eop_source": iers_series.receipt(),
        },
        "orekit": orekit,
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
        "utc_before",
        "utc_leap",
        "utc_after",
        "seconds_before_to_leap",
        "seconds_leap_to_after",
        "utc_minus_tai_2026_s",
        "tt_minus_tai_2026_s",
        "dut1_s",
        "xp_arcsec",
        "yp_arcsec",
        "ddpsi_rad",
        "ddeps_rad",
        "eme2000_to_itrf_position_m",
        "eme2000_to_itrf_velocity_m_s",
        "eme2000_to_itrf_jacobian",
        "teme_to_eme2000_position_m",
        "teme_to_eme2000_velocity_m_s",
        "gcrf_to_itrf_position_m",
        "gcrf_to_itrf_velocity_m_s",
        "gcrf_to_itrf_jacobian",
        "gcrf_to_eme2000_position_m",
        "gcrf_to_eme2000_velocity_m_s",
    }
    missing = sorted(required - set(rows))
    if missing:
        raise RuntimeError(f"Orekit frame/time output is missing: {missing}")
    return rows


def _vector(value: str, size: int) -> np.ndarray:
    array = np.asarray([float(item) for item in value.split(",")], dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise RuntimeError(f"Orekit vector must contain {size} finite values.")
    return array


def _without_zero_fraction(value: str) -> str:
    return value.removesuffix(".000000")


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
    parser = argparse.ArgumentParser(description="Validate OEL frame/time transforms with pinned Orekit.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orekit-root", type=Path, default=DEFAULT_OREKIT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_frame_time_with_orekit(output_path=args.output, orekit_root=args.orekit_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
