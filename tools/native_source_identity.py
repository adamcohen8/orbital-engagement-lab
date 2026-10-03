"""Location-independent source receipts for native release builds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def native_source_identity(crate: Path) -> dict:
    crate = Path(crate)
    if crate.is_symlink():
        raise ValueError("Native build inputs cannot be symbolic links")
    crate = crate.resolve(strict=True)
    paths = [crate / name for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml")]
    for directory in (crate / "src", crate / "vendor", crate / ".cargo"):
        if directory.is_symlink():
            raise ValueError("Native build inputs cannot be symbolic links")
        if directory.exists():
            for path in directory.rglob("*"):
                # Path.rglob does not descend through directory symlinks. Reject
                # each link before filtering so Rust sources behind one cannot
                # disappear from the receipt while remaining buildable by Cargo.
                if path.is_symlink():
                    raise ValueError("Native build inputs cannot be symbolic links")
                if path.is_file() and (
                    path.suffix == ".rs"
                    or path.name in {"Cargo.toml", "Cargo.lock", "config", "config.toml"}
                ):
                    paths.append(path)
    paths.extend(path for path in (crate / "build.rs", crate / "rust-toolchain", crate / "rust-toolchain.toml") if path.exists())
    files = {}
    for path in sorted(set(paths)):
        if path.is_symlink():
            raise ValueError("Native build inputs cannot be symbolic links")
        files[path.relative_to(crate).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "oel.native_source_identity.v1", "files": files, "sha256": digest}
