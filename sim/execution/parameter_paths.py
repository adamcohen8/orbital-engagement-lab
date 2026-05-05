from __future__ import annotations

from typing import Any


def deep_set(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = root
    for i, tok in enumerate(parts):
        last = i == len(parts) - 1
        if "[" in tok and tok.endswith("]"):
            key, idx_txt = tok[:-1].split("[", 1)
            idx = int(idx_txt)
            if key:
                cur = cur[key]
            if not isinstance(cur, list):
                raise TypeError(f"'{tok}' is not a list segment in path '{path}'.")
            if last:
                cur[idx] = value
                return
            cur = cur[idx]
            continue
        if last:
            cur[tok] = value
            return
        cur = cur[tok]


def object_synced_parameter_paths(root: dict[str, Any], path: str) -> list[str]:
    """Return config paths that should be kept consistent for object aliases."""
    paths = [path]
    objects = root.get("objects")
    if not isinstance(objects, dict):
        return paths

    parts = path.split(".")
    if len(parts) >= 3 and parts[0] == "objects" and parts[1] in objects:
        alias = ".".join([parts[1], *parts[2:]])
        if isinstance(root.get(parts[1]), dict):
            paths.append(alias)
        return _dedupe(paths)

    if len(parts) >= 2 and "[" not in parts[0] and parts[0] in objects:
        paths.append(".".join(["objects", *parts]))
    return _dedupe(paths)


def set_parameter_path_value(root: dict[str, Any], path: str, value: Any) -> None:
    for synced_path in object_synced_parameter_paths(root, path):
        deep_set(root, synced_path, value)


def _dedupe(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out
