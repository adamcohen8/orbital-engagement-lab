from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim.installation import cli
from sim.installation.desktop import publish_trainer
from sim.installation.paths import InstallationPaths


def installed(tmp_path, profile="game"):
    paths = InstallationPaths(tmp_path / "OEL data 'quoted'", tmp_path / "config")
    paths.ensure()
    paths.current_state.write_text(json.dumps({"current": "0.30.0"}))
    root = paths.version_root("0.30.0")
    root.mkdir(parents=True)
    (root / "installation-record.json").write_text(
        json.dumps({"profile": profile, "runtime": {"created": False, "python": sys.executable}})
    )
    return paths


@pytest.mark.skipif(os.name == "nt", reason="POSIX application fixture")
def test_macos_application_launches_stable_cli_after_update(tmp_path):
    paths = installed(tmp_path)
    receipt = publish_trainer(paths, home=tmp_path, system="darwin")
    app = Path(receipt["path"])
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    executable = app / "Contents/MacOS" / info["CFBundleExecutable"]
    output = tmp_path / "invocation"
    for version in ["0.30.0", "0.31.0"]:
        launcher = paths.launcher / "oel"
        launcher.write_text(f'#!/bin/sh\nprintf "%s %s" "{version}" "$1" > "{output}"\n')
        launcher.chmod(0o755)
        subprocess.run([str(executable)], cwd="/", check=True)
        assert output.read_text() == f"{version} trainer"
    assert publish_trainer(paths, home=tmp_path, system="darwin") == receipt


def test_desktop_preserves_unrelated_app(tmp_path):
    paths = installed(tmp_path)
    app = tmp_path / "Applications/RPO Trainer.app"
    app.mkdir(parents=True)
    (app / "user-file").write_text("preserve")
    with pytest.raises(RuntimeError, match="unrelated"):
        publish_trainer(paths, home=tmp_path, system="darwin")
    assert (app / "user-file").read_text() == "preserve"


def test_linux_entry_and_core_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "menu"))
    paths = installed(tmp_path)
    target = Path(publish_trainer(paths, home=tmp_path, system="linux")["path"])
    assert target == tmp_path / "menu/applications/oel-rpo-trainer.desktop"
    assert "Terminal=false" in target.read_text()
    assert f'Exec="{paths.launcher}/oel" trainer' in target.read_text()
    paths.version_root("0.30.0").joinpath("installation-record.json").write_text(json.dumps({"profile": "core"}))
    assert publish_trainer(paths, home=tmp_path, system="linux")["status"] == "skipped"


@pytest.mark.skipif(os.name == "nt", reason="POSIX fake interpreter fixture")
def test_windows_dispatcher_resolves_current_version_each_launch(tmp_path):
    paths = installed(tmp_path)
    # Publishing on this host also emits the standalone Windows dispatcher.
    publish_trainer(paths, home=tmp_path, system="darwin")
    script = paths.launcher / "rpo-trainer.py"
    for version in ["0.30.0", "0.31.0"]:
        root = paths.version_root(version)
        root.mkdir(exist_ok=True)
        fake_python = root / "fake-python"
        fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        fake_python.chmod(0o755)
        (root / "installation-record.json").write_text(
            json.dumps({"runtime": {"created": False, "python": str(fake_python)}})
        )
        paths.current_state.write_text(json.dumps({"current": version}))
        subprocess.run([sys.executable, str(script)], check=True)
        assert "trainer" in (paths.data_root / "logs/trainer.log").read_text()


def test_trainer_dispatch_uses_managed_engine_and_writable_cwd(tmp_path, monkeypatch):
    paths = installed(tmp_path)
    source = tmp_path / "source"
    monkeypatch.setattr(cli, "_selected_engine", lambda *args: ("0.30.0", source, Path(sys.executable), "developer"))
    calls = []
    monkeypatch.setattr(
        cli.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=0)
    )
    assert cli._dispatch("trainer", ["--help"], paths=paths, workspace_path=None) == 0
    assert calls[0][0] == [sys.executable, str(source / "run_game.py"), "--help"]
    assert calls[0][1]["cwd"] == paths.data_root
    assert calls[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixture")
def test_trainer_ignores_incidental_workspace(tmp_path, monkeypatch):
    paths = installed(tmp_path)
    monkeypatch.setattr(cli, "_find_workspace", lambda: tmp_path / "incidental/oel-workspace.yaml")
    seen = []
    monkeypatch.setattr(cli, "_dispatch", lambda command, arguments, **kwargs: seen.append(kwargs) or 0)
    assert cli.main(["--data-root", str(paths.data_root), "--config-root", str(paths.config_root), "trainer"]) == 0
    assert seen[0]["workspace_path"] is None


def test_windows_launcher_is_a_gui_executable_with_stdlib_dispatcher(tmp_path):
    import zipfile

    import pip._vendor.distlib as distlib
    from pip._vendor.distlib.scripts import ScriptMaker

    paths = installed(tmp_path)
    publish_trainer(paths, home=tmp_path, system="darwin")
    destination = tmp_path / "windows"
    destination.mkdir()
    maker = ScriptMaker("", str(destination))
    maker.executable = r"C:\Python\python.exe"
    maker._is_nt = True
    launcher_kind = []

    def launcher(kind):
        launcher_kind.append(kind)
        return Path(distlib.__file__).with_name("w64.exe").read_bytes()

    maker._get_launcher = launcher
    maker.make(str(paths.launcher / "rpo-trainer.py"), options={"gui": True})
    executable = destination / "rpo-trainer.exe"
    assert executable.read_bytes().startswith(b"MZ")
    assert launcher_kind == ["w"]
    with zipfile.ZipFile(executable) as archive:
        script = archive.read("__main__.py")
    compile(script, "__main__.py", "exec")
    assert b"state/current.json" in script


def test_update_defaults_to_existing_game_profile(tmp_path, monkeypatch):
    paths = installed(tmp_path)
    seen = []
    monkeypatch.setattr(cli, "install_release", lambda *args, **kw: seen.append(kw) or {})
    monkeypatch.setattr(cli, "_find_workspace", lambda: None)
    assert (
        cli.main(
            [
                "--data-root",
                str(paths.data_root),
                "--config-root",
                str(paths.config_root),
                "update",
                "install",
                "release.json",
                "--developer-unsigned",
                "--no-runtime",
            ]
        )
        == 0
    )
    assert seen[0]["profile"] == "game"
