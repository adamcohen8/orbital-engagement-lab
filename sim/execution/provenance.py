from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def runtime_implementation_digest() -> str:
    """Digest executable OEL source so checkpoints cannot survive code changes."""

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
