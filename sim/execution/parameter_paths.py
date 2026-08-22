from __future__ import annotations

from typing import Any

_PATH_VALUE_FIELDS = {
    "output_dir",
    "summary_json",
    "prompt_file",
    "eop_path",
    "coeff_path",
    "source_path",
    "geometry_profile_path",
    "area_profile_path",
    "attitude_area_profile_path",
    "profile_path",
}


def _reject_path_parameter(path: str) -> None:
    terminal = str(path).rsplit(".", 1)[-1].split("[", 1)[0]
    if terminal in _PATH_VALUE_FIELDS or terminal.endswith("_file"):
        raise ValueError(
            f"Parameter path '{path}' targets a filesystem location. "
            "Batch and sweep parameters may not change config input/output paths."
        )


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
                if idx < 0 or idx >= len(cur):
                    raise KeyError(f"List index {idx} is out of range in parameter path '{path}'.")
                cur[idx] = value
                return
            if idx < 0 or idx >= len(cur):
                raise KeyError(f"List index {idx} is out of range in parameter path '{path}'.")
            cur = cur[idx]
            continue
        if last:
            if not isinstance(cur, dict):
                raise TypeError(f"Cannot set '{tok}' on non-mapping segment in parameter path '{path}'.")
            if tok not in cur:
                raise KeyError(f"Parameter path '{path}' does not exist in the base config.")
            cur[tok] = value
            return
        cur = cur[tok]


def path_exists(root: dict[str, Any], path: str) -> bool:
    cur: Any = root
    try:
        for tok in path.split("."):
            if "[" in tok and tok.endswith("]"):
                key, idx_txt = tok[:-1].split("[", 1)
                idx = int(idx_txt)
                if key:
                    if not isinstance(cur, dict) or key not in cur:
                        return False
                    cur = cur[key]
                if not isinstance(cur, list) or idx < 0 or idx >= len(cur):
                    return False
                cur = cur[idx]
                continue
            if not isinstance(cur, dict) or tok not in cur:
                return False
            cur = cur[tok]
    except (TypeError, ValueError):
        return False
    return True


def object_synced_parameter_paths(root: dict[str, Any], path: str) -> list[str]:
    """Return config paths that should be kept consistent for object aliases."""
    paths = [path]
    objects = root.get("objects")
    if not isinstance(objects, dict):
        return paths

    parts = path.split(".")
    if len(parts) >= 3 and parts[0] == "objects" and parts[1] in objects:
        alias = ".".join([parts[1], *parts[2:]])
        paths = [path]
        if path_exists(root, alias):
            paths.append(alias)
        return _dedupe(paths)

    if len(parts) >= 2 and "[" not in parts[0] and parts[0] in objects:
        canonical = ".".join(["objects", *parts])
        paths = [canonical]
        if path_exists(root, path):
            paths.append(path)
    return _dedupe(paths)


def set_parameter_path_value(root: dict[str, Any], path: str, value: Any) -> None:
    _reject_path_parameter(path)
    synced_paths = object_synced_parameter_paths(root, path)
    existing_paths = [synced_path for synced_path in synced_paths if path_exists(root, synced_path)]
    if not existing_paths:
        raise KeyError(f"Parameter path '{path}' does not exist in the base config.")
    for synced_path in existing_paths:
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
