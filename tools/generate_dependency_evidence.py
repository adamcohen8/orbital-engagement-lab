from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

DEPENDENCY_EVIDENCE_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_hash(download_info: dict) -> str:
    hashes = dict(dict(download_info.get("archive_info", {}) or {}).get("hashes", {}) or {})
    return str(hashes.get("sha256", "") or "")


def _wheel_tags(filename: str) -> dict[str, str]:
    if not filename.lower().endswith(".whl"):
        return {}
    parts = filename[:-4].rsplit("-", 3)
    if len(parts) != 4:
        return {}
    return {
        "python": parts[1],
        "abi": parts[2],
        "platform": parts[3],
    }


def _source_details(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    filename = unquote(Path(parsed.path).name)
    if parsed.scheme in {"http", "https"}:
        return url, filename
    if parsed.scheme == "file":
        return "<local-source>", filename
    return url, filename


def build_dependency_evidence(
    *,
    install_report: str | Path,
    constraints_file: str | Path,
) -> dict[str, object]:
    report_path = Path(install_report).expanduser().resolve()
    constraints_path = Path(constraints_file).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    packages: list[dict[str, object]] = []
    incomplete: list[str] = []
    for item in list(report.get("install", []) or []):
        metadata = dict(item.get("metadata", {}) or {})
        download_info = dict(item.get("download_info", {}) or {})
        source_url, filename = _source_details(str(download_info.get("url", "") or ""))
        artifact_hash = _archive_hash(download_info)
        artifact_type = "wheel" if filename.lower().endswith(".whl") else ("source" if filename else "installed-or-local")
        if source_url != "<local-source>" and (not artifact_hash or artifact_type != "wheel"):
            incomplete.append(f"{metadata.get('name', '')}=={metadata.get('version', '')}")
        packages.append(
            {
                "name": str(metadata.get("name", "") or ""),
                "version": str(metadata.get("version", "") or ""),
                "requested": bool(item.get("requested", False)),
                "artifact": filename,
                "artifact_type": artifact_type,
                "wheel_tags": _wheel_tags(filename),
                "source_url": source_url,
                "sha256": artifact_hash,
            }
        )
    packages.sort(key=lambda item: (str(item["name"]).lower(), str(item["version"])))

    if incomplete:
        raise ValueError(
            "Dependency evidence requires SHA-256-bound wheel artifacts for every non-local package: "
            + ", ".join(sorted(incomplete))
        )
    return {
        "schema_version": DEPENDENCY_EVIDENCE_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "operating_system": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
        },
        "resolver": {
            "report_path": report_path.name,
            "report_version": str(report.get("version", "") or ""),
            "pip_version": str(report.get("pip_version", "") or ""),
        },
        "constraints": {
            "path": constraints_path.as_posix(),
            "sha256": _sha256(constraints_path),
        },
        "packages": packages,
    }


def write_dependency_evidence(
    output: str | Path,
    *,
    install_report: str | Path,
    constraints_file: str | Path,
) -> Path:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dependency_evidence(
        install_report=install_report,
        constraints_file=constraints_file,
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create OEL dependency and wheel-inventory evidence.")
    parser.add_argument("--install-report", required=True, help="JSON report written by pip install --report.")
    parser.add_argument("--constraints", required=True, help="Approved Python-minor constraints file.")
    parser.add_argument(
        "--output",
        default="outputs/supply_chain/wheel-inventory.json",
        help="Output JSON path.",
    )
    args = parser.parse_args(argv)
    output = write_dependency_evidence(
        args.output,
        install_report=args.install_report,
        constraints_file=args.constraints,
    )
    print(f"Dependency evidence written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
