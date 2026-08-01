from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_dependency_evidence import write_dependency_evidence  # noqa: E402
from tools.generate_python_sbom import write_sbom  # noqa: E402

SCHEMA_VERSION = 1
LOCAL_ACCEPTANCE = "local-diagnostic"
HOSTED_ACCEPTANCE = "github-hosted-automation"
DESKTOP_ACCEPTANCE = "controlled-windows-11-desktop"
DESKTOP_ATTESTATIONS = (
    "native_folder_open_verified",
    "trainer_window_verified",
    "keyboard_input_verified",
    "display_rendering_verified",
)
FIRST_PARTY_DISTRIBUTIONS = {"orbital-engagement-lab"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _validate_audit_result(
    audit_result: Path,
    *,
    installed_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    audit_payload = json.loads(audit_result.read_text(encoding="utf-8"))
    dependencies = audit_payload.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise RuntimeError("pip-audit evidence contains no audited dependencies.")
    audited = {
        re.sub(r"[-_.]+", "-", str(item["name"]).lower()): str(item["version"])
        for item in dependencies
        if item.get("name") and item.get("version")
    }
    if installed_versions is None:
        installed_versions = {
            re.sub(r"[-_.]+", "-", str(dist.metadata["Name"]).lower()): str(dist.version)
            for dist in metadata.distributions()
            if dist.metadata.get("Name")
        }
    else:
        installed_versions = {
            re.sub(r"[-_.]+", "-", str(name).lower()): str(version)
            for name, version in installed_versions.items()
        }
    missing_from_audit = sorted(
        set(installed_versions) - set(audited) - FIRST_PARTY_DISTRIBUTIONS
    )
    unexpected_in_audit = sorted(set(audited) - set(installed_versions))
    if missing_from_audit or unexpected_in_audit:
        raise RuntimeError(
            "pip-audit evidence does not cover the complete installed environment: "
            + json.dumps(
                {
                    "missing_from_audit": missing_from_audit,
                    "not_installed": unexpected_in_audit,
                },
                sort_keys=True,
            )
        )
    mismatches = {
        name: {"audited": version, "installed": installed_versions.get(name, "missing")}
        for name, version in audited.items()
        if installed_versions.get(name) != version
    }
    if mismatches:
        raise RuntimeError(
            "pip-audit evidence does not match the installed environment: "
            + json.dumps(mismatches, sort_keys=True)
        )
    vulnerabilities = [
        {
            "package": str(item.get("name", "")),
            "version": str(item.get("version", "")),
            "id": str(vulnerability.get("id", "")),
            "fix_versions": list(vulnerability.get("fix_versions", []) or []),
        }
        for item in dependencies
        for vulnerability in list(item.get("vulns", []) or [])
        if isinstance(vulnerability, dict)
    ]
    if vulnerabilities:
        raise RuntimeError(
            "pip-audit evidence contains unresolved vulnerabilities: "
            + json.dumps(vulnerabilities, sort_keys=True)
        )
    return audit_payload


def _run(
    name: str,
    command: list[str],
    *,
    evidence_dir: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = (
        f"$ {' '.join(command)}\n"
        f"returncode: {result.returncode}\n\n"
        f"[stdout]\n{result.stdout}\n"
        f"[stderr]\n{result.stderr}\n"
    )
    _write_text(evidence_dir / "logs" / f"{name}.txt", log)
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {result.returncode}; "
            f"see {evidence_dir / 'logs' / f'{name}.txt'}"
        )
    return result


def _capture_environment(
    *,
    constraints: Path,
    install_report: Path,
    audit_result: Path,
    evidence_dir: Path,
) -> None:
    _validate_audit_result(audit_result)

    pip_check = _run(
        "pip-check",
        [sys.executable, "-m", "pip", "check"],
        evidence_dir=evidence_dir,
    )
    _write_text(evidence_dir / "pip-check.txt", pip_check.stdout)
    pip_freeze = _run(
        "python-freeze",
        [sys.executable, "-m", "pip", "freeze", "--all"],
        evidence_dir=evidence_dir,
    )
    _write_text(evidence_dir / "python-freeze.txt", pip_freeze.stdout)
    write_dependency_evidence(
        evidence_dir / "wheel-inventory.json",
        install_report=install_report,
        constraints_file=constraints,
    )
    write_sbom(evidence_dir / "sbom.cdx.json")
    report_copy = evidence_dir / "pip-install-report.json"
    audit_copy = evidence_dir / "pip-audit.json"
    if report_copy.resolve() != install_report.resolve():
        report_copy.write_bytes(install_report.read_bytes())
    if audit_copy.resolve() != audit_result.resolve():
        audit_copy.write_bytes(audit_result.read_bytes())


def _check_imports(*, acceleration: str) -> dict[str, str]:
    imports = {
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "yaml": "PyYAML",
        "pytest": "pytest",
        "pygame": "pygame-ce" if sys.version_info >= (3, 14) else "pygame",
        "PIL": "Pillow",
        "imageio": "imageio",
        "imageio_ffmpeg": "imageio-ffmpeg",
        "sgp4": "sgp4",
        "setuptools": "setuptools",
    }
    versions: dict[str, str] = {}
    for module_name, distribution_name in imports.items():
        import_module(module_name)
        versions[distribution_name] = metadata.version(distribution_name)
    versions["ruff"] = metadata.version("ruff")
    if sys.version_info < (3, 11):
        import_module("tomli")
        versions["tomli"] = metadata.version("tomli")

    try:
        import_module("numba")
    except ImportError as exc:
        if acceleration == "available":
            raise RuntimeError(
                "Numba acceleration was required for this row but could not be imported"
            ) from exc
        versions["numba"] = "unavailable (expected)"
    else:
        if acceleration == "unavailable":
            raise RuntimeError("Numba unexpectedly resolved on a serial-only compatibility row")
        versions["numba"] = metadata.version("numba")
    return versions


def _query_review(
    *,
    name: str,
    output_dir: Path,
    query: str,
    evidence_dir: Path,
) -> None:
    result = _run(
        name,
        [
            sys.executable,
            "-m",
            "sim.review",
            str(output_dir),
            "--query",
            query,
            "--json",
        ],
        evidence_dir=evidence_dir,
    )
    json.loads(result.stdout)
    _write_text(evidence_dir / "queries" / f"{name}.json", result.stdout)


def _exercise_runtime(*, evidence_dir: Path) -> None:
    python = sys.executable
    _run(
        "matplotlib-prewarm",
        [
            python,
            "-c",
            (
                "import matplotlib; matplotlib.use('Agg'); "
                "from matplotlib import pyplot as plt; "
                "fig=plt.figure(); fig.savefig(r'"
                + str(evidence_dir / "matplotlib-prewarm.png")
                + "'); plt.close(fig)"
            ),
        ],
        evidence_dir=evidence_dir,
    )
    _run("doctor", [python, "run_simulation.py", "--doctor"], evidence_dir=evidence_dir)

    _run(
        "quickstart-validate",
        [python, "run_simulation.py", "--quickstart", "--validate-only"],
        evidence_dir=evidence_dir,
    )
    _run("quickstart-run", [python, "run_simulation.py", "--quickstart"], evidence_dir=evidence_dir)
    _query_review(
        name="quickstart-review",
        output_dir=ROOT / "outputs" / "quickstart_5min",
        query=(
            "SELECT scenario_name, duration_s, dt_s, samples, oel_version, "
            "review_schema_version FROM run_metadata"
        ),
        evidence_dir=evidence_dir,
    )

    ogp_config = "examples/configs/public_sgp4_passive_propagation.yaml"
    _run(
        "ogp-validate",
        [python, "run_simulation.py", "--config", ogp_config, "--validate-only"],
        evidence_dir=evidence_dir,
    )
    _run(
        "ogp-run",
        [python, "run_simulation.py", "--config", ogp_config],
        evidence_dir=evidence_dir,
    )
    _query_review(
        name="ogp-review",
        output_dir=ROOT / "outputs" / "examples" / "public_sgp4_passive_propagation",
        query=(
            "SELECT object_id, propagation_method, general_model, native_frame, "
            "output_frame, frame_transform FROM object_propagation ORDER BY object_id"
        ),
        evidence_dir=evidence_dir,
    )

    plot_config = "examples/configs/public_attitude_hold_disturbance.yaml"
    _run(
        "plot-validate",
        [python, "run_simulation.py", "--config", plot_config, "--validate-only"],
        evidence_dir=evidence_dir,
    )
    _run(
        "plot-run",
        [python, "run_simulation.py", "--config", plot_config],
        evidence_dir=evidence_dir,
    )
    plot_output = ROOT / "outputs" / "examples" / "public_attitude_hold_disturbance"
    if not any(plot_output.rglob("*.png")):
        raise RuntimeError(f"Plot smoke produced no PNG artifacts under {plot_output}")

    trainer_env = dict(os.environ)
    trainer_env.update(
        {
            "MPLBACKEND": "Agg",
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
        }
    )
    _run(
        "platform-smoke",
        [
            python,
            "-m",
            "pytest",
            "-q",
            "sim/tests/test_platform_compat.py",
            "-k",
            "object_worker_bootstraps_with_spawn_context or pygame_dummy_display_trainer_smoke",
            f"--junitxml={evidence_dir / 'platform-smoke.xml'}",
        ],
        evidence_dir=evidence_dir,
        env=trainer_env,
    )


def _load_desktop_attestation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in DESKTOP_ATTESTATIONS if payload.get(name) is not True]
    if missing:
        raise ValueError(
            "Controlled desktop evidence requires true attestations for: "
            + ", ".join(missing)
        )
    if platform.system() != "Windows" or platform.release() != "11":
        raise ValueError(
            "Controlled desktop evidence must be collected on Windows 11; "
            f"detected {platform.system()} {platform.release()}"
        )
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        raise ValueError(
            "Controlled Windows desktop evidence requires x64; "
            f"detected {platform.machine()}"
        )
    return payload


def _github_hosted_provenance() -> dict[str, str]:
    fields = {
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "workflow": os.environ.get("GITHUB_WORKFLOW", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "sha": os.environ.get("GITHUB_SHA", ""),
        "runner_os": os.environ.get("RUNNER_OS", ""),
        "runner_arch": os.environ.get("RUNNER_ARCH", ""),
    }
    required = ("repository", "workflow", "run_id", "sha", "runner_os", "runner_arch")
    missing = [name for name in required if not fields[name]]
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() != "true" or missing:
        detail = ", ".join(missing) if missing else "GITHUB_ACTIONS"
        raise ValueError(
            "GitHub-hosted evidence requires an active GitHub Actions runner and complete "
            f"CI provenance; missing: {detail}"
        )
    return fields


def _artifact_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in paths:
        candidates = sorted(base.rglob("*")) if base.is_dir() else [base]
        for path in candidates:
            if not path.is_file():
                continue
            rows.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return rows


def run_acceptance(
    *,
    constraints: Path,
    install_report: Path,
    audit_result: Path,
    evidence_dir: Path,
    acceptance_class: str,
    expected_system: str | None,
    expected_machine: str | None,
    acceleration: str,
    desktop_attestation: Path | None,
) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(evidence_dir / "matplotlib-cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if expected_system and platform.system().lower() != expected_system.lower():
        raise RuntimeError(
            f"Expected operating system {expected_system}, detected {platform.system()}"
        )
    if expected_machine and platform.machine().lower() != expected_machine.lower():
        raise RuntimeError(
            f"Expected architecture {expected_machine}, detected {platform.machine()}"
        )

    attestation: dict[str, Any] | None = None
    ci_provenance = {
        "repository": "",
        "workflow": "",
        "run_id": "",
        "run_attempt": "",
        "sha": "",
        "runner_os": "",
        "runner_arch": "",
    }
    if acceptance_class == DESKTOP_ACCEPTANCE:
        if desktop_attestation is None:
            raise ValueError("--desktop-attestation is required for controlled desktop evidence")
        attestation = _load_desktop_attestation(desktop_attestation)
    elif acceptance_class == HOSTED_ACCEPTANCE:
        ci_provenance = _github_hosted_provenance()
    elif desktop_attestation is not None:
        raise ValueError("--desktop-attestation is valid only for controlled desktop evidence")

    _capture_environment(
        constraints=constraints,
        install_report=install_report,
        audit_result=audit_result,
        evidence_dir=evidence_dir,
    )
    versions = _check_imports(acceleration=acceleration)
    _exercise_runtime(evidence_dir=evidence_dir)

    runtime_outputs = [
        ROOT / "outputs" / "quickstart_5min",
        ROOT / "outputs" / "examples" / "public_sgp4_passive_propagation",
        ROOT / "outputs" / "examples" / "public_attitude_hold_disturbance",
    ]
    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance_class": acceptance_class,
        "support_claim": (
            "controlled Windows 11 desktop evidence"
            if acceptance_class == DESKTOP_ACCEPTANCE
            else (
                "GitHub-hosted automation evidence; not physical-desktop evidence"
                if acceptance_class == HOSTED_ACCEPTANCE
                else "Local functional diagnostic; not hosted or physical-desktop support evidence"
            )
        ),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "executable": sys.executable,
        },
        "ci": ci_provenance,
        "constraints": {
            "path": constraints.relative_to(ROOT).as_posix(),
            "sha256": _sha256(constraints),
        },
        "resolved_imports": versions,
        "desktop_attestation": attestation,
        "review_queries": [
            "queries/quickstart-review.json",
            "queries/ogp-review.json",
        ],
        "artifact_inventory": _artifact_inventory(runtime_outputs),
    }
    packet_path = evidence_dir / "compatibility-evidence.json"
    _write_text(packet_path, json.dumps(packet, indent=2, sort_keys=True) + "\n")
    return packet_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the public OEL compatibility smoke and retain a support-evidence packet."
    )
    parser.add_argument("--constraints", required=True)
    parser.add_argument("--install-report", required=True)
    parser.add_argument("--audit-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--acceptance-class",
        choices=(LOCAL_ACCEPTANCE, HOSTED_ACCEPTANCE, DESKTOP_ACCEPTANCE),
        default=LOCAL_ACCEPTANCE,
    )
    parser.add_argument("--expected-system")
    parser.add_argument("--expected-machine")
    parser.add_argument(
        "--expected-acceleration",
        choices=("available", "unavailable"),
        default="available",
    )
    parser.add_argument("--desktop-attestation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet = run_acceptance(
        constraints=(ROOT / args.constraints).resolve(),
        install_report=Path(args.install_report).expanduser().resolve(),
        audit_result=Path(args.audit_result).expanduser().resolve(),
        evidence_dir=Path(args.output_dir).expanduser().resolve(),
        acceptance_class=str(args.acceptance_class),
        expected_system=args.expected_system,
        expected_machine=args.expected_machine,
        acceleration=str(args.expected_acceleration),
        desktop_attestation=(
            Path(args.desktop_attestation).expanduser().resolve()
            if args.desktop_attestation
            else None
        ),
    )
    print(f"Compatibility evidence written: {packet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
