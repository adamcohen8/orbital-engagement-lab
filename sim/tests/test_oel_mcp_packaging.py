from __future__ import annotations

import zipfile
from pathlib import Path

from setuptools.build_meta import build_wheel

ROOT = Path(__file__).resolve().parents[2]


def test_pre_v2_wheel_intentionally_excludes_source_checkout_mcp(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    wheel_name = build_wheel(str(wheel_dir))

    with zipfile.ZipFile(wheel_dir / wheel_name) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")

    assert any(name.startswith("sim/") for name in names)
    assert not any(name.startswith("integrations/oel_mcp/") for name in names)
    assert "Requires-Dist: mcp" not in metadata
    assert "MCP-Transport" not in metadata
