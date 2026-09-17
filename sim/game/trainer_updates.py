"""Background signed updates for the installed desktop level selector."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path

from sim.installation import manager
from sim.installation.contracts import version_tuple
from sim.installation.paths import InstallationPaths


class TrainerUpdater:
    def __init__(self, paths: InstallationPaths, running: str, *, worker=None):
        self.paths = paths
        self.running = running
        self.state = "idle"
        self.latest = None
        self.messages = queue.SimpleQueue()
        self.worker = worker or (lambda fn: threading.Thread(target=fn, daemon=True).start())

    def _job(self, action, success):
        def run():
            try:
                self.messages.put((success, action()))
            except Exception as exc:
                try:
                    self.paths.cache.mkdir(parents=True, exist_ok=True)
                    (self.paths.cache / "trainer-update-error.log").write_text(f"{type(exc).__name__}: {exc}\n")
                except OSError:
                    pass
                self.messages.put(("error", None))

        self.worker(run)

    def check(self):
        if self.state not in {"idle", "error"}:
            return
        self.state = "checking"
        self._job(
            self._check,
            "checked",
        )

    def _record(self):
        return json.loads((self.paths.version_root(self.running) / "installation-record.json").read_text())

    def _check(self):
        return manager.check_channel(
            public_keys=manager.keys_from_path(None, paths=self.paths),
            paths=self.paths,
            channel=self._record().get("channel", "stable"),
        )

    def poll(self):
        while not self.messages.empty():
            state, payload = self.messages.get()
            if state == "checked":
                self.latest = payload["latest"]
                self.state = "available" if version_tuple(self.latest) > version_tuple(self.running) else "current"
            else:
                self.state = state
                if state == "ready":
                    self.latest = payload

    def install(self):
        if self.state == "error" and self.latest is None:
            self.check()
            return
        if self.state not in {"available", "error"} or self.latest is None:
            return
        self.state = "installing"
        self._job(self._install, "ready")

    def _install(self):
        if json.loads(self.paths.current_state.read_text())["current"] != self.running:
            raise RuntimeError("The active installation changed; restart Trainer before updating.")
        record = json.loads((self.paths.version_root(self.running) / "installation-record.json").read_text())
        result = manager.install_latest_release(
            paths=self.paths,
            public_keys=manager.keys_from_path(None, paths=self.paths),
            profile=record.get("profile", "game"),
            channel=record.get("channel", "stable"),
        )
        version = result["version"]
        current = json.loads(self.paths.current_state.read_text())["current"]
        if current != self.running or version_tuple(version) <= version_tuple(self.running):
            raise RuntimeError("The active installation changed; restart Trainer before updating.")
        manager.activate(version, paths=self.paths)
        return version

    def relaunch(self):
        if self.state != "ready":
            return False
        try:
            _, python = manager._source_and_python(self.latest, self.paths)
            logs = self.paths.data_root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / "trainer.log").open("a") as log:
                subprocess.Popen(
                    [
                        str(python),
                        "-m",
                        "sim.installation.cli",
                        "--data-root",
                        str(self.paths.data_root),
                        "--config-root",
                        str(self.paths.config_root),
                        "trainer",
                    ],
                    cwd=self.paths.data_root,
                    stdout=log,
                    stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
        except Exception:
            self.state = "relaunch_error"
            return False
        self.state = "launched"
        return True

    @property
    def busy(self):
        return self.state in {"installing", "ready"}

    @property
    def label(self):
        shortcut = "Ctrl+U / Cmd+U" if os.sys.platform == "darwin" else "Ctrl+U"
        if self.state == "available":
            return f"Update {self.latest} available — {shortcut} to install"
        return {
            "checking": "Checking for updates…",
            "installing": "Installing update… please wait",
            "ready": "Update installed — relaunching…",
            "error": f"Update unavailable — {shortcut} to retry",
            "relaunch_error": "Update installed — close and reopen Trainer",
        }.get(self.state, "")


_updater = None


def installed_updater():
    global _updater
    if _updater is not None:
        return _updater
    env = os.environ
    if env.get("OEL_INSTALLATION_DISPOSITION") != "official" or env.get("OEL_ENGINE_EDITION") != "public":
        return None
    root, version = env.get("OEL_MANAGED_DATA_ROOT"), env.get("OEL_ENGINE_VERSION")
    if not root or not version:
        return None
    defaults = InstallationPaths.default()
    _updater = TrainerUpdater(
        InstallationPaths(Path(root), Path(env.get("OEL_MANAGED_CONFIG_ROOT", defaults.config_root))), version
    )
    return _updater


def banner_rect(pygame, width, height):
    return pygame.Rect(54, height - 56, max(width - 300, 100), 36)


def update_event(updater, pygame, event, width, height):
    if updater is None:
        return False
    if updater.busy:
        return True
    hotkey = (
        event.type == pygame.KEYDOWN and event.key == pygame.K_u and event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META)
    )
    click = (
        event.type == pygame.MOUSEBUTTONDOWN
        and event.button == 1
        and banner_rect(pygame, width, height).collidepoint(event.pos)
    )
    if (hotkey or click) and updater.label:
        updater.install()
        return True
    return False


def draw_update(updater, pygame, screen, font):
    if updater is None or not updater.label:
        return
    rect = banner_rect(pygame, screen.get_width(), screen.get_height())
    pygame.draw.rect(screen, (25, 50, 60), rect, border_radius=5)
    label = updater.label
    while label and font.size(label)[0] > rect.width - 16:
        label = label[:-1]
    screen.blit(
        font.render(label, True, (150, 230, 190)), (rect.x + 8, rect.y + (rect.height - font.get_height()) // 2)
    )
