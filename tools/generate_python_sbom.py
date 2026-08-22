from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4


def _normalise_pypi_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name).strip().lower())


def _purl(name: str, version: str) -> str:
    return f"pkg:pypi/{quote(_normalise_pypi_name(name))}@{quote(str(version))}"


def _installed_components() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    components: list[dict[str, object]] = []
    dependencies: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    distributions: list[tuple[str, str, str, str, list[str]]] = []
    for dist in metadata.distributions():
        name = str(dist.metadata.get("Name", "") or "").strip()
        version = str(dist.version or "").strip()
        if not name or not version:
            continue
        key = (_normalise_pypi_name(name), version)
        if key in seen:
            continue
        seen.add(key)
        metadata_text = dist.read_text("METADATA") or ""
        license_name = str(
            dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
        ).strip()
        requires = [str(item).split(";", 1)[0].strip() for item in (dist.requires or [])]
        distributions.append((name, version, metadata_text, license_name, requires))

    installed_purls = {
        _normalise_pypi_name(name): _purl(name, version)
        for name, version, _metadata_text, _license_name, _requires in distributions
    }
    for name, version, _metadata_text, license_name, requires in distributions:
        component: dict[str, object] = {
            "type": "library",
            "name": name,
            "version": version,
            "purl": _purl(name, version),
        }
        if license_name:
            component["licenses"] = [{"license": {"name": license_name}}]
        components.append(component)
        dependency_refs: set[str] = set()
        for requirement in requires:
            match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
            if match is None:
                continue
            installed_ref = installed_purls.get(_normalise_pypi_name(match.group(0)))
            if installed_ref is not None:
                dependency_refs.add(installed_ref)
        dependencies.append(
            {
                "ref": _purl(name, version),
                "dependsOn": sorted(dependency_refs),
            }
        )
    return (
        sorted(components, key=lambda item: (str(item["name"]).lower(), str(item["version"]))),
        sorted(dependencies, key=lambda item: str(item["ref"])),
    )


def _project_version(project_name: str) -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        project_section = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
        if project_section is not None:
            section = project_section.group(1)
            name_match = re.search(r'^name\s*=\s*"([^"]+)"', section, re.MULTILINE)
            version_match = re.search(r'^version\s*=\s*"([^"]+)"', section, re.MULTILINE)
            if (
                name_match is not None
                and version_match is not None
                and _normalise_pypi_name(name_match.group(1)) == _normalise_pypi_name(project_name)
            ):
                return version_match.group(1)
    try:
        return metadata.version(project_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def build_sbom(*, project_name: str = "orbital-engagement-lab") -> dict[str, object]:
    """Build a minimal CycloneDX JSON SBOM for the current Python environment."""
    components, dependencies = _installed_components()
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "oel-generate-python-sbom",
                        "version": "1",
                    }
                ]
            },
            "component": {
                "type": "application",
                "name": project_name,
                "version": _project_version(project_name),
            },
        },
        "components": components,
        "dependencies": dependencies,
    }


def write_sbom(path: str | Path, *, project_name: str = "orbital-engagement-lab") -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_sbom(project_name=project_name), indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a CycloneDX JSON SBOM for the current Python environment.")
    parser.add_argument(
        "--output",
        default="outputs/supply_chain/sbom.cdx.json",
        help="Output path for the SBOM JSON file.",
    )
    parser.add_argument(
        "--project-name",
        default="orbital-engagement-lab",
        help="Application component name to record in SBOM metadata.",
    )
    args = parser.parse_args(argv)
    output = write_sbom(args.output, project_name=str(args.project_name))
    print(f"SBOM written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
