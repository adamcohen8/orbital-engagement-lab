from __future__ import annotations

from importlib import metadata
from pathlib import Path

import sim.project_version as project_version

ROOT = Path(__file__).resolve().parents[2]


def test_source_project_version_finds_repository_by_default() -> None:
    assert project_version.source_project_version() == project_version.source_project_version(
        source_root=ROOT
    )


def test_project_version_status_accepts_matching_source_and_metadata(monkeypatch) -> None:
    source_version = project_version.source_project_version(source_root=ROOT)
    assert source_version
    monkeypatch.setattr(project_version.metadata, "version", lambda _name: source_version)

    status = project_version.inspect_project_version(source_root=ROOT)

    assert status.ok
    assert status.required
    assert status.source_version == source_version
    assert status.installed_version == source_version


def test_project_version_status_fails_stale_installed_metadata(monkeypatch) -> None:
    source_version = project_version.source_project_version(source_root=ROOT)
    assert source_version
    monkeypatch.setattr(project_version.metadata, "version", lambda _name: "0.0.1")

    status = project_version.inspect_project_version(source_root=ROOT)

    assert not status.ok
    assert status.required
    assert f"source {source_version}" in status.detail
    assert "installed 0.0.1" in status.detail
    assert "pip install -e ." in status.detail


def test_project_version_status_warns_when_source_is_not_installed(monkeypatch) -> None:
    def _not_installed(_name: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(project_version.metadata, "version", _not_installed)

    status = project_version.inspect_project_version(source_root=ROOT)

    assert not status.ok
    assert not status.required
    assert status.source_version
    assert status.installed_version is None
