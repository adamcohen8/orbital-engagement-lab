from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sim.review.evidence_capsule import (
    EvidenceCapsuleError,
    create_evidence_capsule,
    evidence_file_exists,
    evidence_file_sha256,
    materialized_evidence_file,
)
from sim.review.workspace import ReviewWorkspace
from tools import evidence_capsules


def _review_fixture(tmp_path: Path) -> Path:
    review_dir = tmp_path / "run" / "review"
    review_dir.mkdir(parents=True)
    database = review_dir / "run.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE metrics (name TEXT PRIMARY KEY, value REAL)")
        connection.execute("INSERT INTO metrics VALUES ('answer', 42.0)")
    return database


def test_capsule_hydrates_review_workspace_and_qualification_gate(tmp_path: Path, monkeypatch) -> None:
    qualification = pytest.importorskip("sim.flight_software.qualification")
    database = _review_fixture(tmp_path)
    original_sha256 = evidence_file_sha256(database)

    entry = create_evidence_capsule(database, remove_original=True)

    assert not database.exists()
    assert database.with_name("run.sqlite.gz").is_file()
    assert evidence_file_exists(database)
    assert evidence_file_sha256(database) == original_sha256 == entry["original_sha256"]
    with ReviewWorkspace.open(database.parent.parent) as workspace:
        result = workspace.query("SELECT value FROM metrics WHERE name='answer'")
        assert result.rows == [{"value": 42.0}]
        assert workspace.db_path != database
        hydrated_path = workspace.db_path
    assert not hydrated_path.exists()

    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    gate = qualification.ProfileQualificationGate(
        gate_id="capsule-query",
        kind="review_query",
        category="nominal_outcome",
        config={"database": "run/review/run.sqlite", "query": "SELECT value FROM metrics", "op": "eq", "value": 42.0},
    )
    result = qualification._run_review_query_gate(gate)
    assert result["passed"] is True
    assert result["actual"] == 42.0


def test_capsule_rejects_compressed_content_drift(tmp_path: Path) -> None:
    database = _review_fixture(tmp_path)
    create_evidence_capsule(database, remove_original=True)
    compressed = database.with_name("run.sqlite.gz")
    compressed.write_bytes(compressed.read_bytes() + b"tampered")

    with pytest.raises(EvidenceCapsuleError, match="size does not match"):
        with materialized_evidence_file(database):
            pass


def test_capsule_manifest_records_sqlite_integrity_and_row_counts(tmp_path: Path) -> None:
    database = _review_fixture(tmp_path)

    create_evidence_capsule(database)

    manifest = json.loads((database.parent / "evidence_capsule.json").read_text(encoding="utf-8"))
    entry = manifest["artifacts"][0]
    assert entry["verification"]["quick_check"] == "ok"
    assert entry["verification"]["row_counts"] == {"metrics": 1}
    assert entry["original_mtime_ns"] == database.stat().st_mtime_ns


def test_capsule_plan_is_content_bound_before_source_removal(tmp_path: Path, monkeypatch) -> None:
    database = _review_fixture(tmp_path)
    monkeypatch.setattr(evidence_capsules, "ROOT", tmp_path)
    for relative in evidence_capsules._PROVENANCE_PATHS:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative + "\n", encoding="utf-8")
    plan = evidence_capsules.build_plan([database], workspace_root=tmp_path)
    plan_path = tmp_path / "capsule-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    receipt = evidence_capsules.apply_plan(plan_path, workspace_root=tmp_path)

    assert receipt["artifacts"][0]["source_removed"] is True


def test_capsule_apply_allows_public_export_provenance_subset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(evidence_capsules, "ROOT", tmp_path)
    for relative in evidence_capsules._PUBLIC_PROVENANCE_PATHS:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("public source\n", encoding="utf-8")
    database = _review_fixture(tmp_path)
    plan = evidence_capsules.build_plan([database], workspace_root=tmp_path)
    plan_path = tmp_path / "capsule-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    receipt = evidence_capsules.apply_plan(plan_path, workspace_root=tmp_path)

    assert receipt["artifacts"][0]["source_removed"] is True
    assert (database.parent / "run.sqlite.gz").is_file()
    assert not database.exists()
    assert evidence_file_exists(database)


def test_capsule_apply_requires_public_provenance_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(evidence_capsules, "ROOT", tmp_path)
    database = _review_fixture(tmp_path)
    plan = evidence_capsules.build_plan([database], workspace_root=tmp_path)
    plan_path = tmp_path / "capsule-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(EvidenceCapsuleError, match="requires the public review provenance sources"):
        evidence_capsules.apply_plan(plan_path, workspace_root=tmp_path)

    assert database.is_file()


def test_capsule_plan_refuses_content_drift(tmp_path: Path) -> None:
    database = _review_fixture(tmp_path)
    for relative in evidence_capsules._PUBLIC_PROVENANCE_PATHS:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("public source\n", encoding="utf-8")
    plan = evidence_capsules.build_plan([database], workspace_root=tmp_path)
    plan_path = tmp_path / "capsule-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO metrics VALUES ('changed', 1.0)")

    with pytest.raises(ValueError, match="content-bound capsule plan drift"):
        evidence_capsules.apply_plan(plan_path, workspace_root=tmp_path)

    assert database.is_file()
