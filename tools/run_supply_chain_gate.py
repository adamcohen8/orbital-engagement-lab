from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_python_sbom import write_sbom  # noqa: E402

IGNORED_VULNERABILITIES = (
    "CVE-2025-3000",
    "PYSEC-2026-3447",
)


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


def _run(cmd: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, text=True, check=False)


def run_supply_chain_gate(
    output_dir: str | Path,
    *,
    install_full: bool = False,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    command_results: list[dict[str, Any]] = []
    if install_full:
        commands = [
            [python_executable, "-m", "pip", "install", "-U", "pip", "pip-audit"],
            [python_executable, "-m", "pip", "install", ".[full]"],
        ]
        for cmd in commands:
            proc = _run(cmd)
            command_results.append({"command": cmd, "return_code": int(proc.returncode)})
            if proc.returncode != 0:
                break

    sbom_path = output / "sbom.cdx.json"
    freeze_path = output / "python-freeze.txt"
    audit_path = output / "pip-audit.json"
    if all(row["return_code"] == 0 for row in command_results):
        write_sbom(sbom_path)
        freeze = subprocess.run(
            [python_executable, "-m", "pip", "freeze", "--all"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        freeze_path.write_text(freeze.stdout, encoding="utf-8")
        command_results.append(
            {
                "command": [python_executable, "-m", "pip", "freeze", "--all"],
                "return_code": int(freeze.returncode),
            }
        )
        audit_cmd = [
            python_executable,
            "-m",
            "pip_audit",
            "--format",
            "json",
            "--output",
            str(audit_path),
        ]
        for vulnerability in IGNORED_VULNERABILITIES:
            audit_cmd.extend(["--ignore-vuln", vulnerability])
        audit = _run(audit_cmd)
        command_results.append({"command": audit_cmd, "return_code": int(audit.returncode)})

    artifacts = []
    for path in (sbom_path, freeze_path, audit_path):
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
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": python_executable,
        "git": git_provenance(),
        "ignored_vulnerabilities": list(IGNORED_VULNERABILITIES),
        "commands": command_results,
        "artifacts": artifacts,
        "passed": passed,
    }
    manifest_path = output / "supply-chain-gate.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and audit OEL Python supply-chain release evidence.")
    parser.add_argument("--output-dir", default="outputs/supply_chain")
    parser.add_argument(
        "--install-full",
        action="store_true",
        help="Install pip-audit and OEL's full dependency profile before generating evidence.",
    )
    args = parser.parse_args(argv)
    manifest = run_supply_chain_gate(args.output_dir, install_full=bool(args.install_full))
    print(f"Evidence manifest: {manifest['manifest']}")
    print(f"Supply-chain gate: {'PASS' if manifest['passed'] else 'FAIL'}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
