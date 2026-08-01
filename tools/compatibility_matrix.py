from __future__ import annotations

import argparse
import json
from typing import Any

SUPPORTED_PYTHONS = (
    ("3.10", "py310"),
    ("3.11", "py311"),
    ("3.12", "py312"),
    ("3.13", "py313"),
    ("3.14", "py314"),
)
BOUNDARY_PYTHONS = {"3.10", "3.14"}
PLATFORMS: tuple[dict[str, str], ...] = (
    {
        "platform_name": "Ubuntu 22.04 x64",
        "runner": "ubuntu-22.04",
        "setup_arch": "x64",
        "evidence_arch": "x64",
        "system": "Linux",
        "machine": "x86_64",
        "acceleration": "available",
    },
    {
        "platform_name": "Windows Server 2022 x64",
        "runner": "windows-2022",
        "setup_arch": "x64",
        "evidence_arch": "x64",
        "system": "Windows",
        "machine": "AMD64",
        "acceleration": "available",
    },
    {
        "platform_name": "macOS 15 arm64",
        "runner": "macos-15",
        "setup_arch": "arm64",
        "evidence_arch": "arm64",
        "system": "Darwin",
        "machine": "arm64",
        "acceleration": "available",
    },
    {
        "platform_name": "macOS 15 Intel x64",
        "runner": "macos-15-intel",
        "setup_arch": "x64",
        "evidence_arch": "x64",
        "system": "Darwin",
        "machine": "x86_64",
        "acceleration": "unavailable",
    },
)


def build_matrix(scope: str) -> dict[str, list[dict[str, Any]]]:
    if scope not in {"canary", "full"}:
        raise ValueError(f"Unsupported compatibility scope: {scope}")

    rows: list[dict[str, Any]] = []
    for platform in PLATFORMS:
        for version, python_id in SUPPORTED_PYTHONS:
            boundary = version in BOUNDARY_PYTHONS
            if scope == "canary" and platform["system"] == "Darwin" and not boundary:
                continue
            rows.append(
                {
                    **platform,
                    "python_version": version,
                    "python_id": python_id,
                    "boundary": boundary,
                    "timeout_minutes": 150 if boundary else 45,
                }
            )
    return {"include": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the hosted OEL compatibility matrix.")
    parser.add_argument("--scope", choices=("canary", "full"), required=True)
    args = parser.parse_args()
    print(json.dumps(build_matrix(args.scope), separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
