from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sim.game import trainer_updates as updates
from sim.installation.paths import InstallationPaths


def updater(tmp_path, monkeypatch, version="0.30.1"):
    paths = InstallationPaths(tmp_path / "data", tmp_path / "custom config")
    paths.ensure()
    root = paths.version_root("0.30.0")
    root.mkdir(parents=True)
    (root / "installation-record.json").write_text(json.dumps({"profile": "game"}))
    paths.current_state.write_text(json.dumps({"current": "0.30.0"}))
    monkeypatch.setattr(updates.manager, "keys_from_path", lambda *a, **kw: {"trusted": True})
    monkeypatch.setattr(updates.manager, "check_channel", lambda **kw: {"latest": version})
    return updates.TrainerUpdater(paths, "0.30.0", worker=lambda fn: fn())


def test_check_does_not_install_and_duplicate_checks_are_ignored(tmp_path, monkeypatch):
    u = updater(tmp_path, monkeypatch)
    monkeypatch.setattr(
        updates.manager, "install_latest_release", lambda **kw: pytest.fail("background check installed")
    )
    u.check()
    assert u.state == "checking"
    u.check()
    u.poll()
    assert u.state == "available"
    assert "Ctrl+U" in u.label


@pytest.mark.parametrize("version", ["0.30.0", "0.29.0"])
def test_never_offers_equal_or_older_versions(tmp_path, monkeypatch, version):
    u = updater(tmp_path, monkeypatch, version)
    u.check()
    u.poll()
    assert u.state == "current" and not u.label


def test_verification_failure_keeps_current_version_and_allows_retry(tmp_path, monkeypatch):
    u = updater(tmp_path, monkeypatch)
    u.check()
    u.poll()
    monkeypatch.setattr(
        updates.manager, "install_latest_release", lambda **kw: (_ for _ in ()).throw(ValueError("bad signature"))
    )
    monkeypatch.setattr(updates.manager, "activate", lambda *a, **kw: pytest.fail("must not activate failed install"))
    u.install()
    u.poll()
    assert u.state == "error"
    assert json.loads(u.paths.current_state.read_text())["current"] == "0.30.0"
    assert "bad signature" in (u.paths.cache / "trainer-update-error.log").read_text()


def test_install_activate_and_relaunch_order_with_custom_roots(tmp_path, monkeypatch):
    u = updater(tmp_path, monkeypatch)
    u.check()
    u.poll()
    calls = []

    def install(**kwargs):
        assert kwargs["profile"] == "game" and kwargs["public_keys"] == {"trusted": True}
        calls.append("install")
        return {"version": "0.30.1"}

    monkeypatch.setattr(updates.manager, "install_latest_release", install)
    monkeypatch.setattr(updates.manager, "activate", lambda version, **kw: calls.append("activate"))
    monkeypatch.setattr(updates.manager, "_source_and_python", lambda *a: (tmp_path, tmp_path / "new python"))
    launched = []
    monkeypatch.setattr(updates.subprocess, "Popen", lambda argv, **kw: launched.append((argv, kw)))
    u.install()
    u.install()
    assert u.busy
    u.poll()
    assert calls == ["install", "activate"]
    assert u.relaunch()
    assert not u.relaunch()
    assert launched[0][0][0] == str(tmp_path / "new python")
    assert str(u.paths.config_root) in launched[0][0]
    assert launched[0][0][-1] == "trainer"


def test_changed_active_version_is_not_overwritten(tmp_path, monkeypatch):
    u = updater(tmp_path, monkeypatch)
    u.check()
    u.poll()
    u.paths.current_state.write_text(json.dumps({"current": "0.31.0"}))
    monkeypatch.setattr(updates.manager, "install_latest_release", lambda **kw: {"version": "0.30.1"})
    monkeypatch.setattr(
        updates.manager, "activate", lambda *a, **kw: pytest.fail("concurrent activation must be preserved")
    )
    u.install()
    u.poll()
    assert u.state == "error"


def test_developer_preview_does_not_check_network(monkeypatch):
    monkeypatch.setattr(updates, "_updater", None)
    monkeypatch.delenv("OEL_INSTALLATION_DISPOSITION", raising=False)
    assert updates.installed_updater() is None


def test_keyboard_click_and_busy_handling(tmp_path, monkeypatch):
    import pygame

    u = updater(tmp_path, monkeypatch)
    u.check()
    u.poll()
    requests = []
    monkeypatch.setattr(u, "install", lambda: requests.append(True))
    for mods in [pygame.KMOD_CTRL, pygame.KMOD_META]:
        assert updates.update_event(
            u, pygame, SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_u, mod=mods), 1040, 680
        )
    assert not updates.update_event(u, pygame, SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_u, mod=0), 1040, 680)
    assert updates.update_event(
        u, pygame, SimpleNamespace(type=pygame.MOUSEBUTTONDOWN, button=1, pos=(360, 640)), 1040, 680
    )
    assert len(requests) == 3
    u.state = "installing"
    assert updates.update_event(u, pygame, SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_RETURN), 1040, 680)


def test_background_check_returns_while_network_is_pending(tmp_path, monkeypatch):
    import threading

    u = updater(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    threads = []

    def work(fn):
        thread = threading.Thread(target=fn, daemon=True)
        threads.append(thread)
        thread.start()

    def check(**kwargs):
        entered.set()
        assert release.wait(2)
        return {"latest": "0.30.1"}

    u.worker = work
    monkeypatch.setattr(updates.manager, "check_channel", check)
    try:
        u.check()
        assert entered.wait(1)
        assert u.state == "checking"
    finally:
        release.set()
        for thread in threads:
            thread.join(2)
    u.poll()
    assert u.state == "available"


def test_relaunch_failure_keeps_old_selector_open(tmp_path, monkeypatch):
    u = updater(tmp_path, monkeypatch)
    u.state, u.latest = "ready", "0.30.1"
    monkeypatch.setattr(
        updates.manager, "_source_and_python", lambda *a: (_ for _ in ()).throw(OSError("missing runtime"))
    )
    assert not u.relaunch()
    assert u.state == "relaunch_error"
