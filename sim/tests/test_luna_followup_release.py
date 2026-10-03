from __future__ import annotations

import json
import sys
from pathlib import Path

from sim.installation.desktop import publish_trainer
from sim.installation.paths import InstallationPaths


def _installed_paths(tmp_path: Path) -> InstallationPaths:
    paths = InstallationPaths(tmp_path / 'OEL data "quoted" \\slash', tmp_path / "config")
    paths.ensure()
    paths.current_state.write_text(json.dumps({"current": "0.30.0"}), encoding="utf-8")
    root = paths.version_root("0.30.0")
    root.mkdir(parents=True)
    (root / "installation-record.json").write_text(
        json.dumps({"profile": "game", "runtime": {"created": False, "python": sys.executable}}),
        encoding="utf-8",
    )
    return paths


def test_linux_desktop_exec_escapes_path_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "menu"))
    paths = _installed_paths(tmp_path)
    target = Path(publish_trainer(paths, home=tmp_path, system="linux")["path"])
    raw = str(paths.launcher / "oel")
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$").replace("%", "%%")

    assert f'Exec="{escaped}" trainer' in target.read_text(encoding="utf-8")
