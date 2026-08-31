"""Public-safe hashing helpers for retained external-runtime receipts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_tree(path: str | Path) -> str:
    """Hash a directory by sorted relative paths and file bytes."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Runtime snapshot directory is missing: {root}")
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = item.relative_to(root).as_posix()
        digest.update(b"F\0" + relative.encode("utf-8") + b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["sha256_tree"]
