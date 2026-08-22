from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
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


def _package_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name or "").strip().lower())


def _report_packages(report_path: Path) -> tuple[dict[str, object], list[dict[str, object]], list[str]]:
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
    return report, packages, incomplete


def build_dependency_evidence(
    *,
    install_report: str | Path,
    constraints_file: str | Path,
    additional_install_reports: Iterable[str | Path] = (),
) -> dict[str, object]:
    report_path = Path(install_report).expanduser().resolve()
    constraints_path = Path(constraints_file).expanduser().resolve()
    additional_report_paths = [Path(path).expanduser().resolve() for path in additional_install_reports]
    report, report_packages, incomplete = _report_packages(report_path)

    packages_by_name: dict[str, dict[str, object]] = {}
    for package in report_packages:
        packages_by_name[_package_key(str(package["name"]))] = package
    for additional_path in additional_report_paths:
        _additional_report, additional_packages, additional_incomplete = _report_packages(additional_path)
        incomplete.extend(additional_incomplete)
        for package in additional_packages:
            key = _package_key(str(package["name"]))
            existing = packages_by_name.get(key)
            if existing is None:
                packages_by_name[key] = package
                continue
            comparable_fields = ("version", "artifact", "artifact_type", "source_url", "sha256")
            if any(existing[field] != package[field] for field in comparable_fields):
                raise ValueError(f"Dependency evidence contains conflicting artifacts for {package['name']!r}.")
            existing["requested"] = bool(existing["requested"] or package["requested"])

    packages = list(packages_by_name.values())
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
            "additional_report_paths": [path.name for path in additional_report_paths],
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
    additional_install_reports: Iterable[str | Path] = (),
) -> Path:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dependency_evidence(
        install_report=install_report,
        constraints_file=constraints_file,
        additional_install_reports=additional_install_reports,
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create OEL dependency and wheel-inventory evidence.")
    parser.add_argument("--install-report", required=True, help="JSON report written by pip install --report.")
    parser.add_argument("--constraints", required=True, help="Approved Python-minor constraints file.")
    parser.add_argument(
        "--additional-install-report",
        action="append",
        default=[],
        help="Additional pip install report whose wheel artifacts belong to the same offline closure.",
    )
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
        additional_install_reports=args.additional_install_report,
    )
    print(f"Dependency evidence written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
