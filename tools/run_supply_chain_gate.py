from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_python_sbom import write_sbom  # noqa: E402

PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX_URL = "https://pypi.org/simple"
PIP_VERSION = "26.2.1"
PIP_AUDIT_VERSION = "2.10.1"


def _package_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("pyproject.toml does not declare a package version")
    return match.group(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> dict[str, Any]:
    def capture(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    status = capture("status", "--porcelain")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _package_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("PIP_")}
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_INDEX_URL": PYPI_INDEX_URL,
        }
    )
    return environment


def _run(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, check=False)


def _full_install_command(
    *,
    python_executable: str,
    constraints: Path,
    install_report_path: Path,
    torch_cpu_index: bool,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--index-url",
        PYPI_INDEX_URL,
        "-c",
        str(constraints),
    ]
    if torch_cpu_index:
        command.extend(["--extra-index-url", PYTORCH_CPU_INDEX_URL])
    command.extend([".[full]", "--report", str(install_report_path)])
    return command


def _venv_python(environment_root: Path) -> Path:
    if sys.platform == "win32":
        return environment_root / "Scripts" / "python.exe"
    return environment_root / "bin" / "python"


def _recorded_run(
    command_results: list[dict[str, Any]],
    cmd: list[str],
    *,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    proc = _run(cmd, env=env)
    command_results.append({"command": cmd, "return_code": int(proc.returncode)})
    return proc


def _validate_dependency_sources(path: Path, *, torch_cpu_index: bool) -> str | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_hosts = {"files.pythonhosted.org", "pypi.org"}
    if torch_cpu_index:
        allowed_hosts.add("download.pytorch.org")
    unexpected: list[str] = []
    for row in list(dict(payload).get("packages", []) or []):
        if not isinstance(row, dict):
            continue
        source_url = str(row.get("source_url", "") or "")
        if source_url == "<local-source>":
            continue
        if urlparse(source_url).hostname not in allowed_hosts:
            unexpected.append(source_url or "<missing-source-url>")
    if unexpected:
        return "Dependency evidence contains unapproved source URLs: " + ", ".join(sorted(set(unexpected)))
    return None


def _run_supply_chain_gate_in_environment(
    output: Path,
    *,
    install_full: bool,
    bootstrap_python: str,
    constraints: Path,
    torch_cpu_index: bool,
    isolated_environment_root: Path | None,
) -> dict[str, Any]:
    command_results: list[dict[str, Any]] = []
    package_environment = _package_environment()
    audit_python = str(bootstrap_python)
    if isolated_environment_root is not None:
        create_environment = [bootstrap_python, "-m", "venv", str(isolated_environment_root)]
        created = _recorded_run(command_results, create_environment, env=package_environment)
        audit_python = str(_venv_python(isolated_environment_root))
        if created.returncode != 0:
            audit_python = str(bootstrap_python)

    install_report_path = output / "pip-install-report.json"
    pip_check_path = output / "pip-check.txt"
    wheel_inventory_path = output / "wheel-inventory.json"
    sbom_path = output / "sbom.cdx.json"
    freeze_path = output / "python-freeze.txt"
    audit_path = output / "pip-audit.json"

    if install_full and all(row["return_code"] == 0 for row in command_results):
        commands = [
            [
                audit_python,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--index-url",
                PYPI_INDEX_URL,
                f"pip=={PIP_VERSION}",
            ],
            [
                audit_python,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--index-url",
                PYPI_INDEX_URL,
                f"pip-audit=={PIP_AUDIT_VERSION}",
            ],
            _full_install_command(
                python_executable=audit_python,
                constraints=constraints,
                install_report_path=install_report_path,
                torch_cpu_index=torch_cpu_index,
            ),
        ]
        for cmd in commands:
            proc = _recorded_run(command_results, cmd, env=package_environment)
            if proc.returncode != 0:
                break

        if all(row["return_code"] == 0 for row in command_results):
            pip_check = subprocess.run(
                [audit_python, "-m", "pip", "check"],
                cwd=ROOT,
                env=package_environment,
                text=True,
                capture_output=True,
                check=False,
            )
            pip_check_path.write_text(pip_check.stdout + pip_check.stderr, encoding="utf-8")
            command_results.append(
                {
                    "command": [audit_python, "-m", "pip", "check"],
                    "return_code": int(pip_check.returncode),
                }
            )
            if pip_check.returncode == 0:
                dependency_evidence = [
                    audit_python,
                    "tools/generate_dependency_evidence.py",
                    "--install-report",
                    str(install_report_path),
                    "--constraints",
                    str(constraints),
                    "--output",
                    str(wheel_inventory_path),
                ]
                _recorded_run(command_results, dependency_evidence, env=package_environment)
                if command_results[-1]["return_code"] == 0:
                    source_error = _validate_dependency_sources(
                        wheel_inventory_path,
                        torch_cpu_index=torch_cpu_index,
                    )
                    command_results.append(
                        {
                            "command": ["validate-dependency-sources"],
                            "return_code": 0 if source_error is None else 1,
                            **({"error": source_error} if source_error is not None else {}),
                        }
                    )

    if all(row["return_code"] == 0 for row in command_results):
        if isolated_environment_root is None:
            write_sbom(sbom_path)
        else:
            _recorded_run(
                command_results,
                [audit_python, "tools/generate_python_sbom.py", "--output", str(sbom_path)],
                env=package_environment,
            )

    if all(row["return_code"] == 0 for row in command_results):
        freeze = subprocess.run(
            [audit_python, "-m", "pip", "freeze", "--all"],
            cwd=ROOT,
            env=package_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        freeze_path.write_text(freeze.stdout, encoding="utf-8")
        command_results.append(
            {
                "command": [audit_python, "-m", "pip", "freeze", "--all"],
                "return_code": int(freeze.returncode),
            }
        )
        if freeze.returncode == 0:
            audit_cmd = [
                audit_python,
                "-m",
                "pip_audit",
                "--format",
                "json",
                "--output",
                str(audit_path),
            ]
            _recorded_run(command_results, audit_cmd, env=package_environment)

    artifacts = []
    for path in (
        install_report_path,
        pip_check_path,
        wheel_inventory_path,
        sbom_path,
        freeze_path,
        audit_path,
    ):
        if path.is_file():
            artifacts.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    passed = bool(command_results) and all(row["return_code"] == 0 for row in command_results)
    manifest = {
        "schema_version": 1,
        "kind": "oel_supply_chain_gate",
        "product": "oel-pro",
        "package_version": _package_version(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": audit_python,
        "bootstrap_python": bootstrap_python,
        "isolated_environment": isolated_environment_root is not None,
        "dependency_sources": {
            "primary_index": PYPI_INDEX_URL,
            "pytorch_cpu_index": PYTORCH_CPU_INDEX_URL if torch_cpu_index else None,
            "pip_environment_sanitized": True,
        },
        "git": git_provenance(),
        "audit_exceptions": [],
        "commands": command_results,
        "artifacts": artifacts,
        "passed": passed,
    }
    manifest_path = output / "supply-chain-gate.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def run_supply_chain_gate(
    output_dir: str | Path,
    *,
    install_full: bool = False,
    python_executable: str = sys.executable,
    constraints_file: str | Path | None = None,
    torch_cpu_index: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if constraints_file is None:
        constraints = ROOT / "constraints" / f"py{sys.version_info.major}{sys.version_info.minor}.txt"
    else:
        constraints = Path(constraints_file).expanduser().resolve()

    if install_full:
        if not constraints.is_file():
            raise FileNotFoundError(f"No approved constraints file for this interpreter: {constraints}")
        with tempfile.TemporaryDirectory(prefix="oel-supply-chain-") as temporary_root:
            return _run_supply_chain_gate_in_environment(
                output,
                install_full=True,
                bootstrap_python=python_executable,
                constraints=constraints,
                torch_cpu_index=torch_cpu_index,
                isolated_environment_root=Path(temporary_root) / "audit-env",
            )
    return _run_supply_chain_gate_in_environment(
        output,
        install_full=False,
        bootstrap_python=python_executable,
        constraints=constraints,
        torch_cpu_index=torch_cpu_index,
        isolated_environment_root=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and audit OEL Python supply-chain release evidence.")
    parser.add_argument("--output-dir", default="outputs/supply_chain")
    parser.add_argument(
        "--install-full",
        action="store_true",
        help="Install pip-audit and OEL's full dependency profile before generating evidence.",
    )
    parser.add_argument(
        "--constraints",
        help="Approved constraints file (defaults to constraints/py<major><minor>.txt).",
    )
    parser.add_argument(
        "--torch-cpu-index",
        action="store_true",
        help=(
            "Resolve Torch from PyTorch's official CPU wheel index while resolving the full profile. "
            "Intended for disk-bounded Linux audit runners."
        ),
    )
    args = parser.parse_args(argv)
    manifest = run_supply_chain_gate(
        args.output_dir,
        install_full=bool(args.install_full),
        constraints_file=args.constraints,
        torch_cpu_index=bool(args.torch_cpu_index),
    )
    print(f"Evidence manifest: {manifest['manifest']}")
    print(f"Supply-chain gate: {'PASS' if manifest['passed'] else 'FAIL'}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
