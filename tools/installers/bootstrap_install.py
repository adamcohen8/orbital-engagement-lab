"""Standalone first-install bootstrap for a verified OEL release.

This file is rendered by ``tools/build_installable_release.py`` with a trusted
release-key registry, then its digest is embedded in the platform bootstrap.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

TRUSTED_KEYS_B64 = "__OEL_TRUSTED_KEYS_B64__"
TRUSTED_KEYS_RENDERED = "__OEL_TRUSTED_KEYS_RENDERED__"
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_RELEASE_BYTES = 8 * 1024 * 1024 * 1024
_SHA256_DIGESTINFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")


def _fetch(url: str, *, maximum: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "OEL-Bootstrap/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > maximum:
            raise RuntimeError(f"Remote content exceeds the {maximum} byte safety limit.")
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise RuntimeError(f"Remote content exceeds the {maximum} byte safety limit.")
    return data


def _fetch_to_file(url: str, destination: Path, *, maximum: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "OEL-Bootstrap/1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as stream:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > maximum:
            raise RuntimeError(f"Remote content exceeds the {maximum} byte safety limit.")
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RuntimeError(f"Remote content exceeds the {maximum} byte safety limit.")
            stream.write(chunk)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    raw = text.encode("ascii")
    return int.from_bytes(base64.urlsafe_b64decode(raw + b"=" * ((4 - len(raw) % 4) % 4)), "big")


def _verify(payload: dict[str, Any], keys_payload: dict[str, Any]) -> bool:
    signature = payload.get("signature")
    if not isinstance(signature, dict) or signature.get("alg") != "RS256":
        return False
    key_id = str(signature.get("key_id", ""))
    items = keys_payload.get("keys", [])
    item = next((entry for entry in items if isinstance(entry, dict) and entry.get("key_id") == key_id), None)
    if item is None:
        return False
    if bool(item.get("revoked", False)):
        return False
    now = datetime.now(timezone.utc)
    for field, is_lower_bound in (("not_before", True), ("expires_at", False)):
        raw_time = item.get(field)
        if raw_time:
            try:
                boundary = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                if boundary.tzinfo is None:
                    boundary = boundary.replace(tzinfo=timezone.utc)
            except ValueError:
                return False
            if (is_lower_bound and now < boundary) or (not is_lower_bound and now >= boundary):
                return False
    modulus = _decode_int(item["n"])
    exponent = _decode_int(item.get("e", "AQAB"))
    if modulus.bit_length() < 2048 or exponent != 65537:
        return False
    signature_text = str(signature.get("value", ""))
    raw = signature_text.encode("ascii")
    signature_bytes = base64.urlsafe_b64decode(raw + b"=" * ((4 - len(raw) % 4) % 4))
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    message = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    width = (modulus.bit_length() + 7) // 8
    if len(signature_bytes) != width:
        return False
    encoded = pow(int.from_bytes(signature_bytes, "big"), exponent, modulus).to_bytes(width, "big")
    expected = _SHA256_DIGESTINFO_PREFIX + hashlib.sha256(message).digest()
    if not encoded.startswith(b"\x00\x01"):
        return False
    try:
        separator = encoded.index(b"\x00", 2)
    except ValueError:
        return False
    padding = encoded[2:separator]
    return len(padding) >= 8 and all(byte == 0xFF for byte in padding) and encoded[separator + 1 :] == expected


def _safe_extract(archive: Path, destination: Path) -> Path:
    total = 0
    names: set[str] = set()
    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        for member in members:
            normalized = member.name.replace("\\", "/")
            relative = PurePosixPath(normalized)
            if not normalized or not relative.parts or normalized in {".", "./"} or relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe archive path: {member.name!r}")
            canonical_name = relative.as_posix()
            if canonical_name in names:
                raise RuntimeError(f"Duplicate archive member: {canonical_name}")
            names.add(canonical_name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise RuntimeError(f"Unsupported archive member: {normalized}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsupported archive member type: {normalized}")
            total += max(0, member.size)
            if total > MAX_RELEASE_BYTES:
                raise RuntimeError("Release archive expands beyond the safety limit.")
        for member in members:
            relative = PurePosixPath(member.name.replace("\\", "/"))
            output = destination.joinpath(*relative.parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Could not read release member: {member.name}")
            with stream, output.open("wb") as sink:
                shutil.copyfileobj(stream, sink)
            output.chmod(0o755 if member.mode & 0o111 else 0o644)
    roots = sorted(path.parent for path in destination.glob("*/pyproject.toml"))
    if len(roots) != 1:
        raise RuntimeError("Release archive must contain exactly one OEL source root.")
    return roots[0]


def _publish_user_launcher(activation: dict[str, Any]) -> dict[str, Any]:
    if sys.platform == "win32":
        user_bin = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Programs" / "OEL" / "bin"
        source = Path(activation["launchers"]["windows"])
        target = user_bin / "oel.cmd"
    else:
        user_bin = Path.home() / ".local" / "bin"
        source = Path(activation["launchers"]["posix"])
        target = user_bin / "oel"
    user_bin.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        managed = (
            target.is_symlink() and target.resolve() == source.resolve()
        ) or (target.is_file() and "sim.installation.cli" in target.read_text(encoding="utf-8", errors="ignore"))
        if not managed:
            raise RuntimeError(f"Refusing to replace a non-OEL launcher: {target}")
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)
        if sys.platform != "win32":
            target.chmod(0o755)
    path_updated = False
    if sys.platform == "win32":
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            try:
                current, _ = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current = ""
            entries = [item for item in str(current).split(";") if item]
            if str(user_bin).lower() not in {item.lower() for item in entries}:
                entries.append(str(user_bin))
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, ";".join(entries))
                path_updated = True
    return {"path": str(target), "directory": str(user_bin), "path_updated": path_updated}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install one signed OEL release.")
    parser.add_argument("--manifest-url", required=True)
    parser.add_argument("--channel-url")
    parser.add_argument("--profile", default="core")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--developer-unsigned", action="store_true")
    args = parser.parse_args(argv)
    if not ((3, 10) <= sys.version_info[:2] < (3, 15)):
        raise SystemExit("OEL requires CPython >=3.10,<3.15.")
    manifest_scheme = urllib.parse.urlparse(args.manifest_url).scheme.lower()
    if manifest_scheme != "https" and not (args.developer_unsigned and manifest_scheme == "file"):
        raise SystemExit("Official OEL bootstrap URLs must use HTTPS.")
    if args.channel_url:
        channel_scheme = urllib.parse.urlparse(args.channel_url).scheme.lower()
        if channel_scheme != "https" and not (args.developer_unsigned and channel_scheme == "file"):
            raise SystemExit("Official OEL update channel URLs must use HTTPS.")
    manifest_data = _fetch(args.manifest_url, maximum=MAX_METADATA_BYTES)
    manifest = json.loads(manifest_data)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "oel.release-manifest.v1":
        raise SystemExit("Release manifest has an unsupported schema.")
    rendered = TRUSTED_KEYS_RENDERED == "true"
    keys: dict[str, Any] = {}
    if rendered:
        keys = json.loads(base64.b64decode(TRUSTED_KEYS_B64).decode("utf-8"))
    if not args.developer_unsigned and (not rendered or not _verify(manifest, keys)):
        raise SystemExit("Release manifest signature verification failed.")
    artifacts = [item for item in manifest.get("artifacts", []) if item.get("kind") in {"source", "source_bundle"}]
    if len(artifacts) != 1:
        raise SystemExit("Release manifest must declare exactly one source artifact.")
    artifact = artifacts[0]
    artifact_location = str(artifact.get("url") or artifact.get("path") or "").strip()
    if not artifact_location:
        raise SystemExit("Release source artifact is missing both url and path.")
    artifact_url = urllib.parse.urljoin(args.manifest_url, artifact_location)
    artifact_scheme = urllib.parse.urlparse(artifact_url).scheme.lower()
    if artifact_scheme != "https" and not (args.developer_unsigned and artifact_scheme == "file"):
        raise SystemExit("Official OEL release artifact URLs must use HTTPS.")
    with tempfile.TemporaryDirectory(prefix="oel-bootstrap-") as temporary:
        root = Path(temporary)
        archive = root / str(artifact["name"])
        _fetch_to_file(artifact_url, archive, maximum=min(MAX_RELEASE_BYTES, int(artifact["bytes"])))
        if archive.stat().st_size != int(artifact["bytes"]) or _sha256_file(archive) != artifact["sha256"]:
            raise SystemExit("Release artifact size or SHA-256 verification failed.")
        inspection = root / "inspection"
        source_root = _safe_extract(archive, inspection)
        sys.path.insert(0, str(source_root))
        from sim.installation.manager import activate, configure_channel, install_release
        from sim.installation.paths import InstallationPaths
        from sim.installation.signing import load_public_keys

        manifest_path = root / "release-manifest.json"
        # Preserve the signed manifest. The managed installer resolves this
        # verified download by the signed artifact name in the same directory.
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        keys_path = root / "trusted-release-keys.json"
        keys_path.write_text(json.dumps(keys, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths = InstallationPaths.default()
        if args.data_root or args.config_root:
            paths = InstallationPaths(
                (args.data_root or paths.data_root).expanduser().resolve(),
                (args.config_root or paths.config_root).expanduser().resolve(),
            )
        result = install_release(
            manifest_path,
            paths=paths,
            public_keys=None if args.developer_unsigned else load_public_keys(keys_path),
            require_signature=not args.developer_unsigned,
            profile=args.profile,
            create_runtime=True,
        )
        activation = activate(str(manifest["version"]), paths=paths)
        paths.trusted_release_keys.parent.mkdir(parents=True, exist_ok=True)
        if rendered:
            paths.trusted_release_keys.write_text(json.dumps(keys, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        channel_configuration = None
        if args.channel_url:
            channel_configuration = configure_channel(
                args.channel_url,
                edition=str(manifest["edition"]),
                channel=str(manifest["channel"]),
                paths=paths,
                source="official-bootstrap" if not args.developer_unsigned else "developer-bootstrap",
                allow_local_file=bool(args.developer_unsigned),
            )
        user_launcher = _publish_user_launcher(activation)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "installation": result,
                    "activation": activation,
                    "channel_configuration": channel_configuration,
                    "user_launcher": user_launcher,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
