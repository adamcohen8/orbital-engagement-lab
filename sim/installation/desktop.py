"""Per-user RPO Trainer desktop integration for verified managed installs."""

from __future__ import annotations

import json
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .paths import InstallationPaths

MARKER = "oel.rpo-trainer.desktop.v1"


def publish_trainer(paths: InstallationPaths, *, home: Path | None = None, system: str | None = None) -> dict[str, str]:
    """Publish a launcher after activation; never replace an unrelated application."""
    home = home or Path.home()
    system = system or sys.platform
    state = json.loads(paths.current_state.read_text())
    record = json.loads((paths.version_root(state["current"]) / "installation-record.json").read_text())
    if record.get("profile") not in {"game", "full", "cross-platform"}:
        return {"status": "skipped", "reason": "Trainer requires the game dependency profile"}
    paths.ensure()
    # A standalone stdlib dispatcher stays valid when managed versions are updated
    # or removed. The managed CLI verifies and leases the active engine.
    script = paths.launcher / "rpo-trainer.py"
    script.write_text(
        "#!/usr/bin/env pythonw\nimport json, os, pathlib, subprocess\n"
        f"data = pathlib.Path({str(paths.data_root)!r})\n"
        f"config = {str(paths.config_root)!r}\n"
        'state = json.loads((data / "state/current.json").read_text())\n'
        'root = data / "versions" / state["current"]\n'
        'record = json.loads((root / "installation-record.json").read_text())\n'
        'runtime = record["runtime"]\n'
        'python = root / "runtime" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")\n'
        'if not runtime.get("created"): python = pathlib.Path(runtime["python"])\n'
        'logs = data / "logs"\nlogs.mkdir(parents=True, exist_ok=True)\n'
        'with (logs / "trainer.log").open("a") as log:\n'
        '    raise SystemExit(subprocess.call([str(python), "-m", "sim.installation.cli", '
        '"--data-root", str(data), "--config-root", config, "trainer"], '
        "cwd=data, stdout=log, stderr=log, "
        'creationflags=0x08000000 if os.name == "nt" else 0))\n',
        encoding="utf-8",
    )
    if system == "darwin":
        target = home / "Applications" / "RPO Trainer.app"
        marker = target / "Contents" / "oel-owner.json"
        _check_owner(target, marker, paths)
        contents = target / "Contents"
        (contents / "MacOS").mkdir(parents=True, exist_ok=True)
        executable = contents / "MacOS" / "RPO Trainer"
        log_dir = paths.data_root / "logs"
        executable.write_text(
            "#!/bin/sh\nmkdir -p "
            + shlex.quote(str(log_dir))
            + "\nexec "
            + shlex.quote(str(paths.launcher / "oel"))
            + " trainer >> "
            + shlex.quote(str(log_dir / "trainer.log"))
            + " 2>&1\n"
        )
        executable.chmod(0o755)
        (contents / "Info.plist").write_bytes(
            plistlib.dumps(
                {
                    "CFBundleIdentifier": "org.orbitalengagementlab.trainer",
                    "CFBundleName": "RPO Trainer",
                    "CFBundleExecutable": "RPO Trainer",
                    "CFBundlePackageType": "APPL",
                    "CFBundleVersion": "1",
                    "NSHighResolutionCapable": True,
                }
            )
        )
    elif system == "win32":
        target = paths.launcher / "RPO Trainer.exe"
        marker = paths.launcher / "rpo-trainer-owner.json"
        _check_owner(target, marker, paths)
        start_menu = Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "Microsoft/Windows/Start Menu/Programs"
        start_menu.mkdir(parents=True, exist_ok=True)
        shortcut = start_menu / "RPO Trainer.lnk"
        if shortcut.exists() and not marker.exists():
            raise RuntimeError(f"Refusing to replace an unrelated shortcut: {shortcut}")
        # pip is provisioned by the managed runtime; its bundled distlib generates
        # the platform/architecture-specific GUI executable without a compiler.
        runtime_python = paths.version_root(state["current"]) / "runtime/Scripts/python.exe"
        base_python = Path(getattr(sys, "_base_executable", sys.executable))
        windowed_python = base_python.with_name("pythonw.exe")
        if windowed_python.is_file():
            base_python = windowed_python
        with tempfile.TemporaryDirectory(dir=paths.cache) as temp:
            generator = (
                "from pip._vendor.distlib.scripts import ScriptMaker; "
                f'm=ScriptMaker("", {temp!r}); m.executable={str(base_python)!r}; '
                f'm.make({str(script)!r}, options={{"gui": True}})'
            )
            subprocess.run([str(runtime_python), "-c", generator], check=True)
            shutil.copy2(Path(temp) / "rpo-trainer.exe", target)
        marker.write_text(json.dumps({"owner": MARKER, "data_root": str(paths.data_root)}))
        env = dict(os.environ, OEL_TRAINER_LINK=str(shortcut), OEL_TRAINER_EXE=str(target))
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:OEL_TRAINER_LINK); "
                'if ((Test-Path $env:OEL_TRAINER_LINK) -and $s.TargetPath -ne $env:OEL_TRAINER_EXE) { throw "Unrelated shortcut" }; '
                '$s.TargetPath=$env:OEL_TRAINER_EXE; $s.Description="OEL RPO Trainer"; $s.Save()',
            ],
            env=env,
            check=True,
        )
    else:
        target = (
            Path(os.environ.get("XDG_DATA_HOME", home / ".local/share")) / "applications" / "oel-rpo-trainer.desktop"
        )
        marker = target.with_suffix(".owner.json")
        _check_owner(target, marker, paths)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Desktop Exec fields use their own quoting rules, not shell quoting.
        command = (
            str(paths.launcher / "oel")
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("`", "\\`")
            .replace("$", "\\$")
            .replace("%", "%%")
        )
        command = command.replace("\\", "\\\\")
        if any(c in str(paths.launcher) for c in ("\n", "\r", "\t", "=")):
            raise ValueError("Desktop launcher path contains unsupported characters")
        target.write_text(
            "[Desktop Entry]\nType=Application\nName=RPO Trainer\n"
            "Comment=Orbital Engineering Lab RPO Trainer\n"
            f'Exec="{command}" trainer\nTerminal=false\nCategories=Education;Science;\n'
        )
        target.chmod(0o755)
    marker.write_text(json.dumps({"owner": MARKER, "data_root": str(paths.data_root)}))
    return {"status": "ready", "path": str(target)}


def _check_owner(target: Path, marker: Path, paths: InstallationPaths) -> None:
    if target.is_symlink() or marker.is_symlink():
        raise RuntimeError(f"Refusing linked desktop launcher: {target}")
    if target.exists():
        expected = {"owner": MARKER, "data_root": str(paths.data_root)}
        if not marker.is_file() or json.loads(marker.read_text()) != expected:
            raise RuntimeError(f"Refusing to replace an unrelated desktop launcher: {target}")
