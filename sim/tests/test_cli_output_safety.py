from __future__ import annotations

from types import SimpleNamespace

import pytest

import run_simulation


def _cfg(output_dir, *, duration_s: float = 10.0, dt_s: float = 1.0):
    return SimpleNamespace(
        outputs=SimpleNamespace(output_dir=str(output_dir)),
        simulator=SimpleNamespace(duration_s=duration_s, dt_s=dt_s),
        objects={"sat": SimpleNamespace(enabled=True, kind="satellite")},
    )


def test_cli_output_guard_refuses_nonempty_directory(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    prior = output / "prior.txt"
    prior.write_text("durable evidence", encoding="utf-8")

    with pytest.raises(SystemExit, match="refusing to overwrite or mix prior evidence"):
        run_simulation._prepare_cli_output_directory(_cfg(output), overwrite=False, allow_unsafe=False)

    assert prior.read_text(encoding="utf-8") == "durable evidence"


def test_cli_output_guard_archives_prior_evidence_when_explicit(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "prior.txt").write_text("durable evidence", encoding="utf-8")

    run_simulation._prepare_cli_output_directory(_cfg(output), overwrite=True, allow_unsafe=False)

    archives = list(tmp_path.glob("run.previous-*"))
    assert len(archives) == 1
    assert (archives[0] / "prior.txt").read_text(encoding="utf-8") == "durable evidence"
    assert output.is_dir()
    assert (output / "output_replacement_manifest.json").is_file()


def test_cli_output_guard_refuses_low_disk_headroom(tmp_path, monkeypatch) -> None:
    usage = SimpleNamespace(total=1024, used=1023, free=1)
    monkeypatch.setattr(run_simulation.shutil, "disk_usage", lambda _path: usage)

    with pytest.raises(SystemExit, match="insufficient filesystem headroom"):
        run_simulation._prepare_cli_output_directory(_cfg(tmp_path / "run"), overwrite=False, allow_unsafe=False)


def test_safe_validation_labels_structural_scope(capsys) -> None:
    assert run_simulation._print_config_validation_report(
        "configs/automation_smoke.yaml", import_plugins=False
    )

    output = capsys.readouterr().out
    assert "STRUCTURALLY PARSED" in output
    assert "SKIPPED (safe validation does not import plugins)" in output
    assert "STRUCTURALLY OK" in output
