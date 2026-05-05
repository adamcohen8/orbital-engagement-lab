from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IGNORED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "lightning_logs",
}
IGNORED_FILE_NAMES = {
    ".DS_Store",
}
IGNORED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
}
GENERATED_DIRS = {
    "outputs",
    "examples/outputs",
}
PRUNE_DIR_NAMES = {
    ".git",
    ".venv",
}


def _git_tracked_paths(root: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    raw = proc.stdout.decode("utf-8", errors="ignore")
    return {item for item in raw.split("\0") if item}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _is_tracked_or_contains_tracked(path: Path, tracked: set[str]) -> bool:
    rel = _rel(path)
    if rel in tracked:
        return True
    if path.is_dir():
        prefix = f"{rel}/"
        return any(item.startswith(prefix) for item in tracked)
    return False


def _is_under(path: Path, roots: set[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _iter_tree(root: Path, *, prune_roots: set[Path]):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: item.as_posix())
        except OSError:
            continue
        for child in children:
            if child.name in PRUNE_DIR_NAMES:
                continue
            if _is_under(child, prune_roots):
                continue
            yield child
            if child.is_dir() and not child.is_symlink():
                stack.append(child)


def _collect_candidates(*, include_outputs: bool) -> list[Path]:
    candidates: list[Path] = []
    generated_roots = {ROOT / rel for rel in GENERATED_DIRS}
    if include_outputs:
        for path in sorted(generated_roots):
            if path.exists():
                candidates.append(path)
    prune_roots = set() if include_outputs else {path for path in generated_roots if path.exists()}
    for path in _iter_tree(ROOT, prune_roots=prune_roots):
        if path.is_dir() and path.name in IGNORED_DIR_NAMES:
            candidates.append(path)
            continue
        if path.is_file() and path.name in IGNORED_FILE_NAMES:
            candidates.append(path)
            continue
        if path.is_file() and path.suffix in IGNORED_FILE_SUFFIXES:
            candidates.append(path)
    unique = sorted(set(candidates), key=lambda item: len(item.parts))
    minimized: list[Path] = []
    for path in unique:
        if any(_is_under(path, {parent}) and path != parent for parent in minimized):
            continue
        minimized.append(path)
    return sorted(minimized, key=lambda item: item.as_posix())


def _safe_candidates(candidates: list[Path], tracked: set[str]) -> tuple[list[Path], list[Path]]:
    removable: list[Path] = []
    skipped: list[Path] = []
    for path in candidates:
        if not path.exists():
            continue
        if _is_tracked_or_contains_tracked(path, tracked):
            skipped.append(path)
        else:
            removable.append(path)
    return removable, skipped


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove untracked local caches, outputs, and generated artifacts.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove files. Without this flag the command only prints what would be removed.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep outputs/ and examples/outputs/ while still cleaning caches and OS noise.",
    )
    args = parser.parse_args()

    tracked = _git_tracked_paths(ROOT)
    candidates = _collect_candidates(include_outputs=not bool(args.keep_outputs))
    removable, skipped = _safe_candidates(candidates, tracked)

    action = "Removing" if args.apply else "Would remove"
    for path in removable:
        print(f"{action}: {_rel(path)}")
        if args.apply:
            _remove(path)

    for path in skipped:
        print(f"Skipping tracked path: {_rel(path)}", file=sys.stderr)

    if not removable:
        print("No untracked local artifacts found.")
    elif not args.apply:
        print("")
        print("Dry run only. Rerun with --apply to remove these paths.")

    if skipped:
        print("", file=sys.stderr)
        print("Some candidates were tracked by git and were not removed.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
