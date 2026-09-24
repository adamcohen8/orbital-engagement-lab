from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from sim.reporting.review_store import _create_schema
from sim.review.__main__ import main as review_main
from sim.review.slice import create_review_slice, query_review_slice
from sim.review.workspace import ReviewStoreNotFoundError, ReviewWorkspace


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "hosted-run"
    review = run / "review"
    review.mkdir(parents=True)
    with sqlite3.connect(review / "run.sqlite") as db:
        _create_schema(db)
        db.execute("INSERT INTO run_metadata (run_id, config_sha256, duration_s, samples, output_dir) VALUES (?, ?, ?, ?, ?)",
                   ("run-1", "a" * 64, 4.0, 5, "/internal/hosted/result"))
        db.executemany("INSERT INTO objects (object_id) VALUES (?)", [("chaser",), ("target",)])
        db.executemany("INSERT INTO time_samples VALUES (?, ?)", [(i, float(i)) for i in range(5)])
        db.executemany("INSERT INTO object_state (sample_index, time_s, object_id) VALUES (?, ?, ?)",
                       [(i, float(i), obj) for i in range(5) for obj in ("chaser", "target")])
        db.executemany(
            "INSERT INTO impulsive_maneuvers VALUES (" + ", ".join("?" for _ in range(20)) + ")",
            [(f"burn-{i}", "chaser", i, float(i), "eci", *([0.0] * 15)) for i in (0, 1, 4)],
        )
        db.executemany("INSERT INTO relative_state (sample_index, time_s, deputy_id, chief_id) VALUES (?, ?, ?, ?)",
                       [(i, float(i), "chaser", "target") for i in range(5)])
        db.executemany("INSERT INTO fsw_invocations (object_id, invocation_id, invocation_time_ns) VALUES (?, ?, ?)",
                       [("chaser", i, i * 1_000_000_000) for i in range(5)])
        db.executemany("INSERT INTO fsw_diagnostic_fields (object_id, invocation_id, generated_time_ns, field_name) VALUES (?, ?, ?, ?)",
                       [("chaser", i, i * 1_000_000_000, "mode") for i in range(5)])
        db.executemany(
            "INSERT INTO actuator_commands (object_id, invocation_id, command_source_id, command_boot_id, command_sequence) "
            "VALUES (?, ?, ?, ?, ?)",
            [("chaser", i, "fsw", "boot", i) for i in range(5)],
        )
        db.executemany(
            "INSERT INTO actuator_command_receipts (object_id, command_source_id, command_boot_id, command_sequence, received_time_ns) "
            "VALUES (?, ?, ?, ?, ?)",
            [("chaser", "fsw", "boot", i, i * 1_000_000_000 + 100) for i in range(5)],
        )
        db.execute(
            "INSERT INTO ground_access_windows (station_id, object_id, start_s, end_s) VALUES ('site', 'chaser', 0.5, 2.5)"
        )
        db.executemany("INSERT INTO events (event_id, time_s, object_id) VALUES (?, ?, ?)",
                       [(f"event-{i}", float(i), "chaser") for i in range(5)])
        db.execute("INSERT INTO metrics (metric_id, metric_name, value) VALUES ('whole-run', 'final_range', 1)")
    return run


def test_slice_is_partial_queryable_and_source_unchanged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run(tmp_path)
    source = run / "review" / "run.sqlite"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "slice"
    manifest = create_review_slice(
        run, target, start_s=1, end_s=2,
        object_ids=["chaser", "target"],
        families=["state", "relative", "fsw", "events", "access"],
    )
    assert manifest["partial"] is True
    assert manifest["continuation_eligible"] is False
    assert manifest["source"]["sha256"] == source_hash
    assert manifest["row_counts_after"]["object_state"] == 4
    assert manifest["row_counts_after"]["relative_state"] == 2
    assert manifest["row_counts_after"]["fsw_diagnostic_fields"] == 2
    assert manifest["row_counts_after"]["actuator_commands"] == 2
    assert manifest["row_counts_after"]["actuator_command_receipts"] == 2
    assert manifest["row_counts_after"]["ground_access_windows"] == 1
    assert manifest["row_counts_after"]["metrics"] == 0
    assert manifest["row_counts_after"]["impulsive_maneuvers"] == 0
    assert query_review_slice(target, "SELECT output_dir FROM run_metadata").rows[0]["output_dir"] is None
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    result = query_review_slice(target, "SELECT sample_index FROM time_samples ORDER BY sample_index")
    assert [row["sample_index"] for row in result.rows] == [1, 2]
    assert review_main(["slice", "query", str(target), "--sql", "SELECT COUNT(*) AS n FROM object_state"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"][0]["n"] == 4
    with pytest.raises(ReviewStoreNotFoundError):
        ReviewWorkspace.open(target)
    with pytest.raises(ValueError):
        create_review_slice(run, tmp_path / "bad", start_s=1, end_s=2, families=["unknown"])
    assert not (tmp_path / "bad").exists()
    (target / "review_slice.sqlite").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        query_review_slice(target, "SELECT * FROM time_samples")


def test_slice_selects_impulsive_maneuvers_by_control_family_and_time(tmp_path: Path) -> None:
    run = _run(tmp_path)
    target = tmp_path / "control-slice"
    manifest = create_review_slice(run, target, start_s=1, end_s=2, families=["control"])
    assert manifest["row_counts_after"]["impulsive_maneuvers"] == 1
    rows = query_review_slice(target, "SELECT maneuver_id FROM impulsive_maneuvers").rows
    assert [row["maneuver_id"] for row in rows] == ["burn-1"]


def test_slice_rejects_manifest_scope_tampering(tmp_path: Path) -> None:
    run = _run(tmp_path)
    target = tmp_path / "slice"
    create_review_slice(run, target, start_s=1, end_s=2)
    manifest_path = target / "slice_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"]["end_s"] = 4.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid review slice manifest"):
        query_review_slice(target, "SELECT COUNT(*) FROM object_state")
