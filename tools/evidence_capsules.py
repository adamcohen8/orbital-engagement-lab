"""Plan, apply, inspect, and restore content-bound review-store capsules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sim.review.evidence_capsule import (
    EVIDENCE_CAPSULE_MANIFEST,
    EvidenceCapsuleError,
    create_evidence_capsule,
    evidence_file_sha256,
    restore_evidence_capsule,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "oel.evidence_capsule_plan.v1"
RECEIPT_SCHEMA = "oel.evidence_capsule_apply_receipt.v1"
_PUBLIC_PROVENANCE_PATHS = (
    "sim/reporting/review_store.py",
    "sim/review/evidence_capsule.py",
    "sim/review/workspace.py",
)
_OPTIONAL_PRIVATE_PROVENANCE_PATHS = (
    "sim/flight_software/qualification.py",
    "sim/flight_software/maturation.py",
)
_PROVENANCE_PATHS = _PUBLIC_PROVENANCE_PATHS + _OPTIONAL_PRIVATE_PROVENANCE_PATHS


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path must remain inside the workspace: {path}") from exc


def _qualification_queries(root: Path, logical_path: Path) -> list[str]:
    relative = _relative(logical_path, root)
    queries: list[str] = []
    sources = [*sorted((root / "validation/gnc_v2/qualifications").glob("*.json"))]
    maturation = root / "validation/gnc_v2/maturation/mission_demonstrations.json"
    if maturation.is_file():
        sources.append(maturation)
    for source in sources:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if str(value.get("database", "")) == relative and isinstance(value.get("query"), str):
                    queries.append(str(value["query"]))
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return list(dict.fromkeys(queries))


def build_plan(
    paths: Sequence[str | Path],
    *,
    workspace_root: Path = ROOT,
    compression_level: int = 1,
) -> dict[str, Any]:
    root = workspace_root.resolve()
    candidates: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        blockers: list[str] = []
        try:
            relative = _relative(path, root)
        except ValueError as exc:
            relative = path.as_posix()
            blockers.append(str(exc))
        if path.name != "run.sqlite":
            blockers.append("automatic capsule plans currently support review/run.sqlite only")
        if not path.is_file() or path.is_symlink():
            blockers.append("source is missing, non-regular, or a symlink")
        for suffix in ("-wal", "-shm"):
            if path.with_name(path.name + suffix).exists():
                blockers.append(f"SQLite sidecar is present: {path.name + suffix}")
        candidates.append(
            {
                "path": relative,
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path) if path.is_file() and not path.is_symlink() else None,
                "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
                "verification_queries": _qualification_queries(root, path) if not blockers else [],
                "blockers": blockers,
                "eligible": not blockers,
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "generated_utc": _now_utc(),
        "workspace_root": root.as_posix(),
        "compression_level": int(compression_level),
        "candidates": candidates,
        "summary": {
            "candidate_count": len(candidates),
            "eligible_count": sum(bool(item["eligible"]) for item in candidates),
            "eligible_source_bytes": sum(int(item["bytes"] or 0) for item in candidates if item["eligible"]),
        },
    }


def apply_plan(plan_path: str | Path, *, workspace_root: Path = ROOT) -> dict[str, Any]:
    source = Path(plan_path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported evidence capsule plan: {source}")
    root = workspace_root.resolve()
    if Path(str(payload.get("workspace_root", ""))).resolve() != root:
        raise ValueError("evidence capsule plan belongs to a different workspace")
    missing_public_provenance = [
        item for item in _PUBLIC_PROVENANCE_PATHS if not (root / item).is_file()
    ]
    if missing_public_provenance:
        raise EvidenceCapsuleError(
            "Evidence capsule apply requires the public review provenance sources: "
            + ", ".join(missing_public_provenance)
        )
    provenance_paths = tuple(root / item for item in _PUBLIC_PROVENANCE_PATHS) + tuple(
        root / item
        for item in _OPTIONAL_PRIVATE_PROVENANCE_PATHS
        if (root / item).is_file()
    )
    receipts: list[dict[str, Any]] = []
    for candidate in list(payload.get("candidates", []) or []):
        if not isinstance(candidate, dict) or not candidate.get("eligible"):
            continue
        path = (root / str(candidate["path"])).resolve()
        _relative(path, root)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(candidate["bytes"])
            or path.stat().st_mtime_ns != int(candidate["mtime_ns"])
            or _sha256(path) != str(candidate["sha256"])
        ):
            raise ValueError(f"content-bound capsule plan drift: {candidate['path']}")
        entry = create_evidence_capsule(
            path,
            remove_original=True,
            compression_level=int(payload.get("compression_level", 1)),
            verification_queries=tuple(candidate.get("verification_queries", ()) or ()),
            provenance_paths=provenance_paths,
        )
        receipts.append(
            {
                "path": candidate["path"],
                "original_sha256": entry["original_sha256"],
                "compressed_path": str(Path(candidate["path"]).with_name(entry["compressed_path"])),
                "compressed_sha256": entry["compressed_sha256"],
                "source_removed": not path.exists(),
            }
        )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "plan_path": source.as_posix(),
        "plan_sha256": _sha256(source),
        "applied_utc": _now_utc(),
        "artifacts": receipts,
    }
    receipt_path = source.with_name(f"{source.stem}.receipt.json")
    temporary = receipt_path.with_name(f".{receipt_path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    return {**receipt, "receipt_path": receipt_path.as_posix()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Create a content-bound, non-mutating capsule plan.")
    plan.add_argument("paths", nargs="+", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--compression-level", type=int, default=1)
    apply = subparsers.add_parser("apply", help="Apply an unchanged capsule plan and remove exact sources.")
    apply.add_argument("plan", type=Path)
    restore = subparsers.add_parser("restore", help="Restore one logical evidence file from its capsule.")
    restore.add_argument("path", type=Path)
    restore.add_argument("--overwrite", action="store_true")
    inspect = subparsers.add_parser("inspect", help="Print a capsule manifest and verify its logical digest.")
    inspect.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        payload = build_plan(args.paths, compression_level=args.compression_level)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({**payload["summary"], "plan": output.as_posix()}, sort_keys=True))
        return 0
    if args.command == "apply":
        print(json.dumps(apply_plan(args.plan), indent=2, sort_keys=True))
        return 0
    if args.command == "restore":
        print(restore_evidence_capsule(args.path, overwrite=args.overwrite))
        return 0
    if args.command == "inspect":
        path = args.path.expanduser().resolve()
        manifest = path.parent / EVIDENCE_CAPSULE_MANIFEST
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        print(json.dumps({"manifest": payload, "logical_sha256": evidence_file_sha256(path)}, indent=2, sort_keys=True))
        return 0
    raise EvidenceCapsuleError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
