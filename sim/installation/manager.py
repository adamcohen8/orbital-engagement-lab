"""Transactional side-by-side installation and update management for OEL."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import venv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sim.project_version import source_project_version

from .archive import safe_extract
from .contracts import (
    CHANNEL_CONFIG_SCHEMA,
    CHANNEL_INDEX_SCHEMA,
    INSTALLATION_RECORD_SCHEMA,
    UPDATE_RECEIPT_SCHEMA,
    ContractError,
    canonical_json_bytes,
    load_json_object,
    reject_unknown_keys,
    release_manifest_digest,
    require_keys,
    sha256_bytes,
    sha256_file,
    sha256_tree,
    validate_release_manifest,
    validate_version,
    version_satisfies,
    version_tuple,
)
from .paths import InstallationPaths
from .signing import RSAPublicKey, load_public_keys, verify_payload
from .state import (
    StateLock,
    atomic_write_json,
    atomic_write_text,
    empty_installation_state,
    read_state,
    recover_stale_lock,
)

MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_RELEASE_BYTES = 8 * 1024 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pro_installation_available() -> bool:
    """Return whether this edition contains the private Pro license verifier."""

    try:
        return importlib.util.find_spec("sim.licensing.offline") is not None
    except ModuleNotFoundError:
        return False


def _python_in(runtime: Path) -> Path:
    return runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _find_source_root(extracted: Path) -> Path:
    direct = extracted / "pyproject.toml"
    if direct.is_file():
        return extracted
    candidates = sorted(path.parent for path in extracted.glob("*/pyproject.toml"))
    if len(candidates) != 1:
        raise ContractError("Release artifact must contain exactly one OEL source root with pyproject.toml.")
    return candidates[0]


def _artifact_path(manifest_path: Path, artifact: Mapping[str, Any]) -> Path:
    text = str(artifact.get("path", "") or "").strip()
    candidates: list[Path] = []
    if text:
        path = Path(text)
        candidates.append((manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve())
    name = str(artifact.get("name", "") or "").strip()
    if name:
        if Path(name).name != name:
            raise ContractError("Release artifact name must be a plain file name.")
        candidates.append((manifest_path.parent / name).resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(item) for item in candidates) or "no local candidates"
    raise FileNotFoundError(f"Release artifact was not found ({searched}).")


def _select_source_artifact(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    host_platform = platform.system().lower()
    host_machine = platform.machine().lower().replace("amd64", "x86_64").replace("aarch64", "arm64")
    candidates: list[tuple[int, Mapping[str, Any]]] = []
    for item in manifest["artifacts"]:
        if item.get("kind") not in {"source", "source_bundle"}:
            continue
        target_platform = str(item.get("platform", "") or "").lower()
        target_machine = str(item.get("architecture", "") or "").lower().replace("amd64", "x86_64").replace("aarch64", "arm64")
        if target_platform and target_platform != host_platform:
            continue
        if target_machine and target_machine != host_machine:
            continue
        candidates.append((int(bool(target_platform)) + int(bool(target_machine)), item))
    if not candidates:
        raise ContractError(f"Release manifest has no source artifact for {platform.system()}/{platform.machine()}.")
    best_score = max(score for score, _ in candidates)
    selected = [item for score, item in candidates if score == best_score]
    if len(selected) != 1:
        raise ContractError("Release manifest source artifact selection is ambiguous for this host.")
    return selected[0]


def _validate_host_compatibility(manifest: Mapping[str, Any]) -> None:
    supported = {str(item).lower() for item in manifest.get("platforms", [])}
    if supported and platform.system().lower() not in supported:
        raise ContractError(f"OEL {manifest['version']} does not support host platform {platform.system()}.")
    expected_arch = str(manifest.get("architecture", "") or "").lower().replace("amd64", "x86_64").replace("aarch64", "arm64")
    host_arch = platform.machine().lower().replace("amd64", "x86_64").replace("aarch64", "arm64")
    if expected_arch and expected_arch != host_arch:
        raise ContractError(f"OEL {manifest['version']} targets architecture {expected_arch}, not host {host_arch}.")
    requires = str(dict(manifest.get("python", {}) or {}).get("requires", "") or "")
    host_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if requires and not version_satisfies(host_python, requires):
        raise ContractError(f"OEL {manifest['version']} requires Python {requires}; this interpreter is {host_python}.")


def _validate_offline_runtime_compatibility(manifest: Mapping[str, Any]) -> None:
    qualification = dict(
        dict(manifest.get("supply_chain", {}) or {}).get("offline_runtime_qualification", {}) or {}
    )
    qualified_python = str(qualification.get("python", "") or "")
    if not qualified_python:
        return
    qualified = version_tuple(qualified_python)
    if len(qualified) < 2:
        raise ContractError("Offline bundle runtime qualification has an invalid Python version.")
    required_minor = qualified[:2]
    host_minor = (sys.version_info.major, sys.version_info.minor)
    if required_minor != host_minor:
        required_tag = f"py{required_minor[0]}{required_minor[1]}"
        host_tag = f"py{host_minor[0]}{host_minor[1]}"
        raise ContractError(
            f"Offline bundle was qualified for CPython {required_minor[0]}.{required_minor[1]} "
            f"({required_tag}), not this interpreter's CPython {host_minor[0]}.{host_minor[1]} ({host_tag}). "
            "Use the bundle whose Python tag matches the installing OEL launcher."
        )


def verify_release_manifest(
    value: Mapping[str, Any],
    *,
    public_keys: Mapping[str, RSAPublicKey] | None,
    require_signature: bool = True,
) -> dict[str, Any]:
    manifest = validate_release_manifest(value)
    signature = manifest.get("signature")
    if require_signature and not signature:
        raise ContractError("Official release manifests must be signed.")
    if signature and (not public_keys or not verify_payload(manifest, public_keys)):
        raise ContractError("Release manifest signature verification failed.")
    return manifest


def verify_release_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    actual_size = path.stat().st_size
    expected_size = int(artifact["bytes"])
    if actual_size != expected_size:
        raise ContractError(f"Release artifact size mismatch: expected {expected_size}, got {actual_size}.")
    actual_digest = sha256_file(path)
    if actual_digest != artifact["sha256"]:
        raise ContractError(f"Release artifact SHA-256 mismatch: expected {artifact['sha256']}, got {actual_digest}.")


def _install_runtime(
    source_root: Path,
    runtime_root: Path,
    *,
    profile: str,
    constraints: Path | None,
    offline_wheelhouse: Path | None = None,
) -> dict[str, Any]:
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt").create(runtime_root)
    python = _python_in(runtime_root)
    requirement = str(source_root) if profile == "core" else f"{source_root}[{profile}]"
    command = [str(python), "-m", "pip", "install", "--only-binary=:all:"]
    if offline_wheelhouse is not None:
        if not offline_wheelhouse.is_dir() or not any(offline_wheelhouse.iterdir()):
            raise ContractError("Offline installation requires a non-empty verified wheelhouse.")
        command.extend(["--no-index", "--find-links", str(offline_wheelhouse)])
    if constraints is not None:
        command.extend(["-c", str(constraints)])
    command.append(requirement)
    environment = dict(os.environ)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if offline_wheelhouse is not None:
        environment["PIP_NO_INDEX"] = "1"
    completed = subprocess.run(command, cwd=source_root, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"OEL runtime installation failed:\n{completed.stdout}\n{completed.stderr}".strip())
    check = subprocess.run([str(python), "-m", "pip", "check"], capture_output=True, text=True, check=False)
    if check.returncode != 0:
        raise RuntimeError(f"Installed dependency graph failed pip check:\n{check.stdout}\n{check.stderr}".strip())
    return {
        "python": str(python),
        "install_command": command,
        "pip_check": check.stdout.strip() or "No broken requirements found.",
    }


def _constraints_for(source_root: Path) -> Path | None:
    minor = f"py{sys.version_info.major}{sys.version_info.minor}.txt"
    candidate = source_root / "constraints" / minor
    return candidate if candidate.is_file() else None


def _tree_inventory(source_root: Path) -> dict[str, Any]:
    files = [
        {"path": path.relative_to(source_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
    ]
    return {"schema_version": "oel.installed-tree.v1", "files": files}


def install_release(
    manifest_path: str | Path,
    *,
    paths: InstallationPaths | None = None,
    public_keys: Mapping[str, RSAPublicKey] | None = None,
    require_signature: bool = True,
    profile: str = "core",
    create_runtime: bool = True,
    license_path: str | Path | None = None,
    license_public_keys: Mapping[str, RSAPublicKey] | None = None,
    offline_wheelhouse: str | Path | None = None,
) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    locations.ensure()
    source_manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = verify_release_manifest(
        load_json_object(source_manifest_path), public_keys=public_keys, require_signature=require_signature
    )
    _validate_host_compatibility(manifest)
    if offline_wheelhouse is not None and create_runtime:
        _validate_offline_runtime_compatibility(manifest)
    version = manifest["version"]
    if profile not in manifest["profiles"]:
        raise ContractError(f"Install profile {profile!r} is not declared by OEL release {version}.")
    if manifest["edition"] == "pro":
        if not pro_installation_available():
            raise ContractError("Managed OEL Pro installation is unavailable in this edition.")
        from sim.licensing.offline import verify_license_file

        license_status = verify_license_file(
            Path(license_path).expanduser() if license_path is not None else None,
            product="oel-pro",
            version=version,
            public_keys=dict(license_public_keys) if license_public_keys is not None else None,
            allow_owner_bypass=False,
        )
        if not license_status.valid:
            raise ContractError(f"OEL Pro release {version} is not licensed: {license_status.message}")
    artifact = _select_source_artifact(manifest)
    archive = _artifact_path(source_manifest_path, artifact)
    verify_release_artifact(archive, artifact)
    if offline_wheelhouse is not None:
        wheelhouse_root = Path(offline_wheelhouse).expanduser().resolve()
        wheel_artifacts = [item for item in manifest["artifacts"] if item.get("kind") == "wheel"]
        if not wheel_artifacts:
            raise ContractError("Offline release manifest does not declare a wheelhouse.")
        declared_paths: set[Path] = set()
        for wheel in wheel_artifacts:
            wheel_path = _artifact_path(source_manifest_path, wheel)
            try:
                wheel_path.relative_to(wheelhouse_root)
            except ValueError as exc:
                raise ContractError(f"Offline wheel is outside the verified wheelhouse: {wheel_path}") from exc
            verify_release_artifact(wheel_path, wheel)
            declared_paths.add(wheel_path)
        actual_paths = {path.resolve() for path in wheelhouse_root.iterdir() if path.is_file()}
        if actual_paths != declared_paths:
            raise ContractError("Offline wheelhouse contents do not exactly match the signed release manifest.")
    final_root = locations.version_root(version)
    with StateLock(locations.transaction_lock, operation=f"install:{version}") as lock:
        if final_root.exists():
            existing = verify_installation(version, paths=locations, full=False)
            if existing["status"] == "official":
                return {
                    "schema_version": UPDATE_RECEIPT_SCHEMA,
                    "status": "ready",
                    "operation": "install",
                    "transaction_id": lock.transaction_id,
                    "version": version,
                    "idempotent": True,
                    "installation": existing,
                    "effects": {"workspace_modified": False, "activated": False},
                }
            raise FileExistsError(f"Installation destination already exists but is not verified: {final_root}")
        transaction_root = locations.versions / f".{version}.{lock.transaction_id}.incomplete"
        transaction_root.mkdir(parents=True, exist_ok=False)
        try:
            extracted = transaction_root / "source"
            safe_extract(archive, extracted, max_bytes=MAX_RELEASE_BYTES)
            source_root = _find_source_root(extracted)
            source_version = source_project_version(source_root=source_root)
            if source_version != version:
                raise ContractError(
                    f"Release artifact source version {source_version!r} does not match signed manifest version {version!r}."
                )
            for name, expected_digest in dict(manifest.get("constraints", {}) or {}).items():
                if Path(str(name)).name != str(name):
                    raise ContractError(f"Release constraint name is unsafe: {name!r}")
                constraint = source_root / "constraints" / str(name)
                if not constraint.is_file() or sha256_file(constraint) != expected_digest:
                    raise ContractError(f"Release constraint is missing or changed: {name}")
            runtime = transaction_root / "runtime"
            runtime_result: dict[str, Any] = {"created": False, "python": sys.executable}
            if create_runtime:
                wheelhouse = Path(offline_wheelhouse).expanduser().resolve() if offline_wheelhouse is not None else None
                runtime_result = {
                    "created": True,
                    **_install_runtime(
                        source_root,
                        runtime,
                        profile=profile,
                        constraints=_constraints_for(source_root),
                        offline_wheelhouse=wheelhouse,
                    ),
                }
            source_digest = sha256_tree(source_root)
            tree_manifest_path = transaction_root / "installed-tree.json"
            atomic_write_json(tree_manifest_path, _tree_inventory(source_root))
            copied_manifest = transaction_root / "release-manifest.json"
            shutil.copy2(source_manifest_path, copied_manifest)
            record = {
                "schema_version": INSTALLATION_RECORD_SCHEMA,
                "status": "official" if require_signature else "developer",
                "version": version,
                "edition": manifest["edition"],
                "channel": manifest["channel"],
                "installed_at": utc_now(),
                "transaction_id": lock.transaction_id,
                "release_manifest_sha256": release_manifest_digest(manifest),
                "release_artifact": {
                    "name": artifact["name"],
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                },
                "source": {"path": str(source_root.relative_to(transaction_root)), "sha256": source_digest},
                "installed_tree_manifest": "installed-tree.json",
                "runtime": runtime_result,
                "profile": profile,
                "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
                "effects": {"workspace_modified": False, "activated": False},
            }
            atomic_write_json(transaction_root / "installation-record.json", record)
            transaction_root.replace(final_root)
            state = read_state(locations.installations_state, default=empty_installation_state())
            installations = dict(state.get("installations", {}) or {})
            installations[version] = {
                "path": str(final_root),
                "record": str(final_root / "installation-record.json"),
                "status": record["status"],
                "installed_at": record["installed_at"],
            }
            state["installations"] = installations
            history = list(state.get("history", []) or [])
            history.append({"operation": "install", "version": version, "transaction_id": lock.transaction_id, "at": utc_now()})
            state["history"] = history[-100:]
            atomic_write_json(locations.installations_state, state)
        except Exception:
            if transaction_root.exists():
                failed_record = {
                    "schema_version": INSTALLATION_RECORD_SCHEMA,
                    "status": "incomplete",
                    "version": version,
                    "transaction_id": lock.transaction_id,
                    "failed_at": utc_now(),
                    "effects": {
                        "workspace_modified": False,
                        "activated": False,
                        "network_used": bool(create_runtime and offline_wheelhouse is None),
                    },
                }
                atomic_write_json(transaction_root / "installation-record.json", failed_record)
            raise
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "install",
        "transaction_id": lock.transaction_id,
        "version": version,
        "idempotent": False,
        "installation": verify_installation(version, paths=locations, full=False),
        "effects": {
            "workspace_modified": False,
            "activated": False,
            "network_used": bool(create_runtime and offline_wheelhouse is None),
        },
    }


def verify_installation(version: str, *, paths: InstallationPaths | None = None, full: bool = False) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    root = locations.version_root(validate_version(version))
    record_path = root / "installation-record.json"
    if not record_path.is_file():
        return {"status": "incomplete", "version": version, "path": str(root), "issues": ["installation record missing"]}
    record = load_json_object(record_path)
    issues: list[str] = []
    if record.get("schema_version") != INSTALLATION_RECORD_SCHEMA:
        issues.append("unsupported installation record schema")
    source = root / str(dict(record.get("source", {}) or {}).get("path", "source"))
    if not source.is_dir():
        issues.append("installed source directory missing")
    elif full:
        actual = sha256_tree(source)
        if actual != dict(record.get("source", {}) or {}).get("sha256"):
            issues.append("installed source tree digest mismatch")
        tree_manifest = root / str(record.get("installed_tree_manifest", "installed-tree.json"))
        if not tree_manifest.is_file():
            issues.append("installed source tree manifest missing")
        elif load_json_object(tree_manifest) != _tree_inventory(source):
            issues.append("installed source tree inventory mismatch")
    manifest_path = root / "release-manifest.json"
    if not manifest_path.is_file():
        issues.append("release manifest missing")
    elif sha256_bytes(canonical_json_bytes(load_json_object(manifest_path))) != record.get("release_manifest_sha256"):
        issues.append("release manifest digest mismatch")
    disposition = "modified" if issues and record.get("status") == "official" else str(record.get("status", "incomplete"))
    if not issues and disposition not in {"official", "developer"}:
        disposition = "incomplete"
    return {
        "status": disposition,
        "version": version,
        "path": str(root),
        "record": str(record_path),
        "profile": record.get("profile"),
        "full_integrity": bool(full),
        "issues": issues,
    }


def activate(version: str, *, paths: InstallationPaths | None = None) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    locations.ensure()
    target = validate_version(version)
    verified = verify_installation(target, paths=locations, full=False)
    if verified["status"] not in {"official", "developer"}:
        raise RuntimeError(f"Cannot activate unverified OEL installation {target}: {verified}")
    with StateLock(locations.transaction_lock, operation=f"activate:{target}") as lock:
        current = read_state(locations.current_state, default={"schema_version": "oel.current-installation.v1"})
        previous = current.get("current")
        payload = {
            "schema_version": "oel.current-installation.v1",
            "current": target,
            "previous": previous if previous != target else current.get("previous"),
            "activated_at": utc_now(),
            "transaction_id": lock.transaction_id,
        }
        old_state = locations.current_state.read_bytes() if locations.current_state.is_file() else None
        old_launchers = {
            path: path.read_bytes() if path.is_file() else None
            for path in (locations.launcher / "oel", locations.launcher / "oel.cmd")
        }
        try:
            launchers = write_launchers(target, paths=locations)
            atomic_write_json(locations.current_state, payload)
        except Exception:
            for path, content in old_launchers.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
            if old_state is None:
                locations.current_state.unlink(missing_ok=True)
            else:
                locations.current_state.write_bytes(old_state)
            raise
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "activate",
        "transaction_id": lock.transaction_id,
        "current": target,
        "previous": payload["previous"],
        "launchers": launchers,
        "effects": {"workspace_modified": False, "activated": True},
    }


def write_launchers(version: str, *, paths: InstallationPaths | None = None) -> dict[str, str]:
    locations = paths or InstallationPaths.default()
    locations.ensure()
    source, python = _source_and_python(version, locations)
    del source
    posix = locations.launcher / "oel"
    windows = locations.launcher / "oel.cmd"
    atomic_write_text(
        posix,
        "#!/bin/sh\n"
        + (
            f"exec {shlex.quote(str(python))} -m sim.installation.cli "
            f"--data-root {shlex.quote(str(locations.data_root))} "
            f"--config-root {shlex.quote(str(locations.config_root))} \"$@\"\n"
        ),
    )
    try:
        posix.chmod(0o755)
    except OSError:
        pass
    atomic_write_text(
        windows,
        (
            f'@echo off\r\n"{python}" -m sim.installation.cli '
            f'--data-root "{locations.data_root}" --config-root "{locations.config_root}" %*\r\n'
        ),
    )
    return {"posix": str(posix), "windows": str(windows)}


def _source_and_python(version: str, paths: InstallationPaths) -> tuple[Path, Path]:
    root = paths.version_root(validate_version(version))
    record = load_json_object(root / "installation-record.json")
    source = root / str(dict(record.get("source", {}) or {}).get("path", "source"))
    runtime = dict(record.get("runtime", {}) or {})
    python = Path(str(runtime.get("python") or sys.executable))
    relocated = root / "runtime" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if runtime.get("created") and relocated.is_file():
        python = relocated
    # Keep the managed virtualenv interpreter path intact. Resolving the POSIX
    # ``bin/python`` symlink selects the base interpreter and silently drops
    # the virtualenv's installed packages when the launcher runs.
    return source.resolve(), python.expanduser().absolute()


def rollback(*, paths: InstallationPaths | None = None) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    current = read_state(locations.current_state)
    previous = str(current.get("previous", "") or "")
    if not previous:
        raise RuntimeError("No previous OEL installation is recorded for rollback.")
    return activate(previous, paths=locations)


def installation_status(*, paths: InstallationPaths | None = None, full: bool = False) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    locations.ensure()
    state = read_state(locations.installations_state, default=empty_installation_state())
    current = read_state(locations.current_state, default={})
    channel_config = read_state(
        locations.channel_config,
        default={"schema_version": CHANNEL_CONFIG_SCHEMA, "default": None, "channels": {}},
    )
    versions = sorted(dict(state.get("installations", {}) or {}))
    incomplete = [
        {"path": str(path), "record": str(path / "installation-record.json")}
        for path in sorted(locations.versions.glob(".*.incomplete"))
        if path.is_dir()
    ]
    return {
        "schema_version": "oel.update-status.v1",
        "status": "ready",
        "paths": {"data_root": str(locations.data_root), "config_root": str(locations.config_root)},
        "current": current.get("current"),
        "previous": current.get("previous"),
        "update_channels": {
            "default": channel_config.get("default"),
            "configured": sorted(dict(channel_config.get("channels", {}) or {})),
            "path": str(locations.channel_config),
        },
        "installations": [verify_installation(version, paths=locations, full=full) for version in versions],
        "incomplete_transactions": incomplete,
        "lock": str(locations.transaction_lock) if locations.transaction_lock.exists() else None,
    }


def write_support_receipt(output: str | Path, *, paths: InstallationPaths | None = None) -> dict[str, Any]:
    """Write a sanitized support snapshot without workspace paths or user content."""

    locations = paths or InstallationPaths.default()
    status = installation_status(paths=locations, full=False)
    registry = read_state(locations.workspaces_state, default={})
    workspaces = dict(registry.get("workspaces", {}) or {})
    payload = {
        "schema_version": "oel.support-receipt.v1",
        "generated_at": utc_now(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "current": status.get("current"),
        "previous": status.get("previous"),
        "installations": [
            {"version": item["version"], "status": item["status"], "profile": item.get("profile"), "issues": item["issues"]}
            for item in status["installations"]
        ],
        "workspaces": [
            {"workspace_id": workspace_id, "locked_version": item.get("locked_version")}
            for workspace_id, item in sorted(workspaces.items())
            if isinstance(item, dict)
        ],
        "privacy": {
            "workspace_paths_included": False,
            "user_source_included": False,
            "config_contents_included": False,
            "telemetry_sent": False,
        },
    }
    destination = Path(output).expanduser().resolve()
    atomic_write_json(destination, payload)
    return {"status": "ready", "path": str(destination), "receipt": payload}


def cleanup(*, paths: InstallationPaths | None = None, dry_run: bool = True, keep: int = 2) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    status = installation_status(paths=locations)
    current = {item for item in (status.get("current"), status.get("previous")) if item}
    workspace_registry = read_state(locations.workspaces_state, default={})
    referenced = {
        str(item.get("locked_version"))
        for item in dict(workspace_registry.get("workspaces", {}) or {}).values()
        if isinstance(item, dict) and item.get("locked_version")
    }
    all_versions = sorted((item["version"] for item in status["installations"]), key=version_tuple, reverse=True)
    retained = set(all_versions[: max(0, int(keep))]) | current | referenced
    candidates = [version for version in all_versions if version not in retained]
    for version in candidates:
        lease_root = locations.version_root(version) / "leases"
        if lease_root.is_dir() and any(lease_root.iterdir()):
            retained.add(version)
    candidates = [version for version in candidates if version not in retained]
    incomplete_candidates = [Path(item["path"]) for item in status.get("incomplete_transactions", [])]
    removed: list[str] = []
    if not dry_run:
        with StateLock(locations.transaction_lock, operation="cleanup"):
            for version in candidates:
                shutil.rmtree(locations.version_root(version))
                removed.append(version)
            for path in incomplete_candidates:
                shutil.rmtree(path)
            state = read_state(locations.installations_state, default=empty_installation_state())
            installations = dict(state.get("installations", {}) or {})
            for version in removed:
                installations.pop(version, None)
            state["installations"] = installations
            atomic_write_json(locations.installations_state, state)
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "cleanup",
        "dry_run": bool(dry_run),
        "candidates": candidates,
        "removed": removed,
        "incomplete_candidates": [str(path) for path in incomplete_candidates],
        "incomplete_removed": [str(path) for path in incomplete_candidates] if not dry_run else [],
        "retained": sorted(retained),
        "recoverable_by_redownload": bool(removed),
    }


def uninstall(
    version: str,
    *,
    paths: InstallationPaths | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove one unreferenced managed engine while preserving every workspace."""

    locations = paths or InstallationPaths.default()
    target = validate_version(version)
    current = read_state(locations.current_state, default={})
    protected_by = [name for name in ("current", "previous") if current.get(name) == target]
    registry = read_state(locations.workspaces_state, default={})
    workspace_refs = [
        str(workspace_id)
        for workspace_id, item in dict(registry.get("workspaces", {}) or {}).items()
        if isinstance(item, dict) and item.get("locked_version") == target
    ]
    lease_root = locations.version_root(target) / "leases"
    active_leases = sorted(path.name for path in lease_root.iterdir()) if lease_root.is_dir() else []
    blockers = {
        "selectors": protected_by,
        "workspaces": workspace_refs,
        "active_leases": active_leases,
    }
    blocked = any(blockers.values())
    removed = False
    if not dry_run and blocked:
        raise RuntimeError(f"Cannot uninstall referenced OEL {target}: {blockers}")
    if not dry_run:
        with StateLock(locations.transaction_lock, operation=f"uninstall:{target}"):
            root = locations.version_root(target)
            if root.is_dir():
                shutil.rmtree(root)
                removed = True
            state = read_state(locations.installations_state, default=empty_installation_state())
            installations = dict(state.get("installations", {}) or {})
            installations.pop(target, None)
            state["installations"] = installations
            history = list(state.get("history", []) or [])
            history.append({"operation": "uninstall", "version": target, "at": utc_now()})
            state["history"] = history[-100:]
            atomic_write_json(locations.installations_state, state)
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "blocked" if blocked else "ready",
        "operation": "uninstall",
        "version": target,
        "dry_run": bool(dry_run),
        "removed": removed,
        "blockers": blockers,
        "retained_workspaces": sorted(dict(registry.get("workspaces", {}) or {})),
        "effects": {"workspace_modified": False, "activated": False},
    }


def recover_lock(*, paths: InstallationPaths | None = None, stale_after_s: float = 3600.0) -> dict[str, Any]:
    locations = paths or InstallationPaths.default()
    return recover_stale_lock(locations.transaction_lock, stale_after_s=stale_after_s)


def _read_url(url: str, *, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "OEL-Updater/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise ContractError(f"Remote content exceeds the {max_bytes} byte limit.")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ContractError(f"Remote content exceeds the {max_bytes} byte limit.")
    return payload


def _validate_remote_url(url: str, *, allow_local_file: bool) -> None:
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme == "https":
        return
    if allow_local_file and scheme == "file":
        return
    raise ContractError("Online OEL update URLs must use HTTPS.")


def configure_channel(
    channel_url: str,
    *,
    edition: str = "public",
    channel: str = "stable",
    paths: InstallationPaths | None = None,
    source: str = "explicit",
    set_default: bool = True,
    allow_local_file: bool = False,
) -> dict[str, Any]:
    """Persist one trusted-metadata endpoint without weakening signature checks."""

    _validate_remote_url(channel_url, allow_local_file=allow_local_file)
    if edition not in {"public", "pro"}:
        raise ContractError("Configured update channel edition must be public or pro.")
    if channel not in {"stable", "preview"}:
        raise ContractError("Configured update channel must be stable or preview.")
    locations = paths or InstallationPaths.default()
    locations.ensure()
    with StateLock(locations.transaction_lock, operation=f"configure-channel:{edition}:{channel}"):
        config = read_state(
            locations.channel_config,
            default={"schema_version": CHANNEL_CONFIG_SCHEMA, "default": None, "channels": {}},
        )
        reject_unknown_keys(config, {"schema_version", "default", "channels"}, label="update channel configuration")
        if config.get("schema_version") != CHANNEL_CONFIG_SCHEMA:
            raise ContractError("Update channel configuration has an unsupported schema.")
        channels = dict(config.get("channels", {}) or {})
        channel_id = f"{edition}:{channel}"
        channels[channel_id] = {
            "edition": edition,
            "channel": channel,
            "url": channel_url,
            "source": source,
            "configured_at": utc_now(),
        }
        config["channels"] = channels
        if set_default or not config.get("default"):
            config["default"] = channel_id
        atomic_write_json(locations.channel_config, config)
    return {
        "schema_version": CHANNEL_CONFIG_SCHEMA,
        "status": "ready",
        "channel_id": channel_id,
        "channel_url": channel_url,
        "default": config["default"],
        "path": str(locations.channel_config),
    }


def configured_channel_url(
    *,
    edition: str = "public",
    channel: str = "stable",
    paths: InstallationPaths | None = None,
    explicit_url: str | None = None,
    allow_local_file: bool = False,
) -> str:
    if explicit_url:
        _validate_remote_url(explicit_url, allow_local_file=allow_local_file)
        return explicit_url
    locations = paths or InstallationPaths.default()
    config = read_state(locations.channel_config)
    reject_unknown_keys(config, {"schema_version", "default", "channels"}, label="update channel configuration")
    if config.get("schema_version") != CHANNEL_CONFIG_SCHEMA:
        raise ContractError(
            "No supported update channel is configured. Re-run the official installer or provide --channel-url."
        )
    channel_id = f"{edition}:{channel}"
    entry = dict(dict(config.get("channels", {}) or {}).get(channel_id, {}) or {})
    require_keys(entry, {"edition", "channel", "url", "source", "configured_at"}, label=f"configured channel {channel_id}")
    reject_unknown_keys(
        entry,
        {"edition", "channel", "url", "source", "configured_at"},
        label=f"configured channel {channel_id}",
    )
    if entry["edition"] != edition or entry["channel"] != channel:
        raise ContractError(f"Configured update channel identity does not match {channel_id}.")
    url = str(entry["url"])
    _validate_remote_url(url, allow_local_file=allow_local_file)
    return url


def _download_url(url: str, destination: Path, *, max_bytes: int) -> None:
    """Download with a bounded partial-file resume when the server supports ranges."""

    offset = destination.stat().st_size if destination.is_file() else 0
    if offset > max_bytes:
        raise ContractError(f"Partial download exceeds the {max_bytes} byte limit.")
    headers = {"User-Agent": "OEL-Updater/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        total = offset if resumed else 0
        with destination.open(mode) as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ContractError(f"Remote content exceeds the {max_bytes} byte limit.")
                stream.write(chunk)


def check_channel(
    channel_url: str | None = None,
    *,
    public_keys: Mapping[str, RSAPublicKey],
    edition: str = "public",
    channel: str = "stable",
    paths: InstallationPaths | None = None,
    allow_feed_rollback: bool = False,
    allow_local_file: bool = False,
) -> dict[str, Any]:
    channel_url = configured_channel_url(
        edition=edition,
        channel=channel,
        paths=paths,
        explicit_url=channel_url,
        allow_local_file=allow_local_file,
    )
    _validate_remote_url(channel_url, allow_local_file=allow_local_file)
    payload = json.loads(_read_url(channel_url, max_bytes=MAX_METADATA_BYTES))
    if not isinstance(payload, dict) or payload.get("schema_version") != CHANNEL_INDEX_SCHEMA:
        raise ContractError("Update channel metadata has an unsupported schema.")
    channel_fields = {
        "schema_version",
        "edition",
        "channel",
        "latest",
        "manifest_url",
        "published_at",
        "signature",
    }
    reject_unknown_keys(payload, channel_fields, label="release channel")
    require_keys(payload, channel_fields, label="release channel")
    if not verify_payload(payload, public_keys):
        raise ContractError("Update channel signature verification failed.")
    if payload.get("edition") != edition or payload.get("channel") != channel:
        raise ContractError("Update channel edition or channel does not match the request.")
    _validate_remote_url(str(payload.get("manifest_url", "")), allow_local_file=allow_local_file)
    latest = validate_version(payload.get("latest"), label="channel latest")
    locations = paths or InstallationPaths.default()
    locations.ensure()
    state = read_state(locations.channel_state, default={"schema_version": "oel.channel-state.v1", "channels": {}})
    channel_id = f"{edition}:{channel}"
    previous = dict(state.get("channels", {}) or {}).get(channel_id)
    if previous and version_tuple(latest) < version_tuple(str(previous.get("latest", "0.0"))) and not allow_feed_rollback:
        raise ContractError(
            f"Release feed rollback rejected: {channel_id} previously advertised {previous['latest']}, now {latest}."
        )
    if (
        previous
        and version_tuple(latest) == version_tuple(str(previous.get("latest", "0.0")))
        and str(payload.get("published_at", "")) < str(previous.get("published_at", ""))
        and not allow_feed_rollback
    ):
        raise ContractError(f"Release feed timestamp rollback rejected for {channel_id} at version {latest}.")
    channels = dict(state.get("channels", {}) or {})
    channels[channel_id] = {"latest": latest, "published_at": payload.get("published_at"), "checked_at": utc_now()}
    state["channels"] = channels
    atomic_write_json(locations.channel_state, state)
    return {
        "schema_version": "oel.update-check.v1",
        "status": "ready",
        "edition": edition,
        "channel": channel,
        "channel_url": channel_url,
        "latest": latest,
        "manifest_url": payload.get("manifest_url"),
        "published_at": payload.get("published_at"),
        "effects": {"network_used": True, "downloaded_release": False, "workspace_modified": False},
    }


def download_release(
    manifest_url: str,
    *,
    paths: InstallationPaths | None = None,
    public_keys: Mapping[str, RSAPublicKey],
    allow_local_file: bool = False,
) -> dict[str, Any]:
    _validate_remote_url(manifest_url, allow_local_file=allow_local_file)
    locations = paths or InstallationPaths.default()
    locations.ensure()
    manifest_payload = json.loads(_read_url(manifest_url, max_bytes=MAX_METADATA_BYTES))
    if not isinstance(manifest_payload, dict):
        raise ContractError("Release manifest response must be an object.")
    manifest = verify_release_manifest(manifest_payload, public_keys=public_keys, require_signature=True)
    artifact = _select_source_artifact(manifest)
    url = str(artifact.get("url", "") or "")
    if not url:
        raise ContractError("Online release artifact is missing url.")
    resolved_url = urllib.parse.urljoin(manifest_url, url)
    _validate_remote_url(resolved_url, allow_local_file=allow_local_file)
    cache_root = locations.cache / manifest["version"]
    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / artifact["name"]
    temporary = archive.with_suffix(archive.suffix + ".partial")
    _download_url(resolved_url, temporary, max_bytes=min(MAX_RELEASE_BYTES, int(artifact["bytes"])))
    verify_release_artifact(temporary, artifact)
    temporary.replace(archive)
    manifest_path = cache_root / "release-manifest.json"
    # Keep signed content byte-for-byte equivalent at the contract level. The
    # local installer resolves the downloaded artifact by its signed name.
    atomic_write_json(manifest_path, manifest)
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "download",
        "version": manifest["version"],
        "manifest": str(manifest_path),
        "artifact": str(archive),
        "effects": {"network_used": True, "workspace_modified": False, "activated": False},
    }


def install_latest_release(
    *,
    paths: InstallationPaths | None = None,
    public_keys: Mapping[str, RSAPublicKey],
    channel_url: str | None = None,
    edition: str = "public",
    channel: str = "stable",
    profile: str = "core",
    create_runtime: bool = True,
    license_path: str | Path | None = None,
    license_public_keys: Mapping[str, RSAPublicKey] | None = None,
    allow_local_file: bool = False,
) -> dict[str, Any]:
    """Check, download, and install the newest signed release without activation."""

    locations = paths or InstallationPaths.default()
    checked = check_channel(
        channel_url,
        public_keys=public_keys,
        edition=edition,
        channel=channel,
        paths=locations,
        allow_local_file=allow_local_file,
    )
    downloaded = download_release(
        str(checked["manifest_url"]),
        paths=locations,
        public_keys=public_keys,
        allow_local_file=allow_local_file,
    )
    downloaded_manifest = load_json_object(downloaded["manifest"])
    expected_identity = (str(checked["latest"]), edition, channel)
    actual_identity = (
        str(downloaded_manifest.get("version", "")),
        str(downloaded_manifest.get("edition", "")),
        str(downloaded_manifest.get("channel", "")),
    )
    if actual_identity != expected_identity:
        raise ContractError(
            "Signed channel and release manifest identities do not agree: "
            f"expected {expected_identity}, got {actual_identity}."
        )
    installed = install_release(
        downloaded["manifest"],
        paths=locations,
        public_keys=public_keys,
        require_signature=True,
        profile=profile,
        create_runtime=create_runtime,
        license_path=license_path,
        license_public_keys=license_public_keys,
    )
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "install-latest",
        "version": checked["latest"],
        "activated": False,
        "workspace_modified": False,
        "check": checked,
        "download": downloaded,
        "installation": installed,
        "next": {
            "global": f"oel update activate {checked['latest']}",
            "workspace": f"oel workspace check <path> --against {checked['latest']}",
        },
    }


def install_bundle(
    bundle: str | Path,
    *,
    paths: InstallationPaths | None = None,
    public_keys: Mapping[str, RSAPublicKey] | None = None,
    require_signature: bool = True,
    profile: str = "core",
    create_runtime: bool = True,
    license_path: str | Path | None = None,
    license_public_keys: Mapping[str, RSAPublicKey] | None = None,
) -> dict[str, Any]:
    source = Path(bundle).expanduser().resolve()
    if source.is_dir():
        manifest = source / "release-manifest.json"
        wheelhouse = source / "wheelhouse"
        return install_release(
            manifest,
            paths=paths,
            public_keys=public_keys,
            require_signature=require_signature,
            profile=profile,
            create_runtime=create_runtime,
            license_path=license_path,
            license_public_keys=license_public_keys,
            offline_wheelhouse=wheelhouse,
        )
    locations = paths or InstallationPaths.default()
    locations.ensure()
    with tempfile.TemporaryDirectory(prefix="oel-bundle-", dir=locations.cache) as temporary:
        root = Path(temporary)
        safe_extract(source, root, max_bytes=MAX_RELEASE_BYTES)
        manifests = list(root.rglob("release-manifest.json"))
        if len(manifests) != 1:
            raise ContractError("Offline bundle must contain exactly one release-manifest.json.")
        wheelhouse = manifests[0].parent / "wheelhouse"
        return install_release(
            manifests[0],
            paths=locations,
            public_keys=public_keys,
            require_signature=require_signature,
            profile=profile,
            create_runtime=create_runtime,
            license_path=license_path,
            license_public_keys=license_public_keys,
            offline_wheelhouse=wheelhouse,
        )


def keys_from_path(path: str | Path | None, *, paths: InstallationPaths | None = None) -> dict[str, RSAPublicKey]:
    locations = paths or InstallationPaths.default()
    source = Path(path).expanduser() if path else locations.trusted_release_keys
    if not source.is_file():
        raise FileNotFoundError(f"Trusted OEL release-key registry was not found: {source}")
    return load_public_keys(source)


def rotate_trusted_release_keys(
    registry_path: str | Path,
    *,
    paths: InstallationPaths | None = None,
    current_keys: Mapping[str, RSAPublicKey] | None = None,
) -> dict[str, Any]:
    """Adopt a key registry only when it is authorized by a currently trusted key."""

    locations = paths or InstallationPaths.default()
    locations.ensure()
    payload = load_json_object(registry_path)
    if payload.get("schema_version") != "oel.trusted-key-registry.v1":
        raise ContractError("Trusted release-key registry has an unsupported schema.")
    reject_unknown_keys(
        payload,
        {"schema_version", "published_at", "keys", "signature"},
        label="trusted release-key registry",
    )
    require_keys(payload, {"schema_version", "keys", "signature"}, label="trusted release-key registry")
    trusted = dict(current_keys or keys_from_path(None, paths=locations))
    if not verify_payload(payload, trusted):
        raise ContractError("Trusted release-key registry signature verification failed.")
    replacement = load_public_keys(registry_path)
    if not replacement or not any(not key.revoked for key in replacement.values()):
        raise ContractError("Trusted release-key registry must retain at least one non-revoked key.")
    atomic_write_json(locations.trusted_release_keys, payload)
    return {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ready",
        "operation": "rotate-trusted-release-keys",
        "key_ids": sorted(replacement),
        "effects": {"workspace_modified": False, "activated": False, "network_used": False},
    }
