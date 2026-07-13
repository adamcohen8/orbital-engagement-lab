from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from integrations.oel_mcp.contracts import DEPLOYMENT_PROFILES

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MCPPathPolicy:
    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]

    @classmethod
    def configured(
        cls,
        *,
        read_roots: tuple[str | Path, ...] | None = None,
        write_roots: tuple[str | Path, ...] | None = None,
    ) -> MCPPathPolicy:
        reads = read_roots or _roots_from_environment("OEL_MCP_READ_ROOTS") or _legacy_roots() or (ROOT,)
        writes = write_roots or _roots_from_environment("OEL_MCP_WRITE_ROOTS") or reads
        return cls(
            read_roots=tuple(Path(root).expanduser().resolve() for root in reads),
            write_roots=tuple(Path(root).expanduser().resolve() for root in writes),
        )

    def resolve_read(self, value: str | Path, *, kind: str = "any") -> Path:
        path = self._resolve(value)
        self._require_root(path, self.read_roots)
        if kind == "file" and not path.is_file():
            raise FileNotFoundError("Required authorized input file was not found.")
        if kind == "directory" and not path.is_dir():
            raise FileNotFoundError("Required authorized input directory was not found.")
        if kind == "any" and not path.exists():
            raise FileNotFoundError("Required authorized input path was not found.")
        return path

    def resolve_write(self, value: str | Path) -> Path:
        path = self._resolve(value)
        self._require_root(path, self.write_roots)
        existing_parent = path
        while not existing_parent.exists() and existing_parent != existing_parent.parent:
            existing_parent = existing_parent.parent
        if existing_parent.is_symlink():
            raise PermissionError("Path is not authorized for this operation.")
        return path

    @staticmethod
    def _resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()

    @staticmethod
    def _require_root(path: Path, roots: tuple[Path, ...]) -> None:
        if not any(_is_relative_to(path, root) for root in roots):
            raise PermissionError("Path is not authorized for this operation.")


def validate_handling(profile: str, handling: dict[str, Any] | None) -> dict[str, str]:
    if profile not in DEPLOYMENT_PROFILES:
        raise PermissionError("Deployment profile is not authorized.")
    if not isinstance(handling, dict):
        raise PermissionError("Handling metadata is required for this operation.")
    marking = str(handling.get("marking", "")).strip()
    release_scope = str(handling.get("release_scope", "")).strip()
    owner = str(handling.get("owner", "")).strip()
    if not marking or release_scope not in {"public", "local_only", "frontier_eligible"}:
        raise PermissionError("Handling metadata is missing or conflicting; review is required.")
    if profile == "direct_frontier_restricted" and release_scope not in {"public", "frontier_eligible"}:
        raise PermissionError("Data is not eligible for the active frontier deployment profile.")
    return {"marking": marking, "release_scope": release_scope, "owner": owner}


def _roots_from_environment(name: str) -> tuple[Path, ...]:
    value = os.environ.get(name, "").strip()
    return tuple(Path(item) for item in value.split(os.pathsep) if item.strip())


def _legacy_roots() -> tuple[Path, ...]:
    return _roots_from_environment("OEL_MCP_ALLOWED_ROOTS")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
