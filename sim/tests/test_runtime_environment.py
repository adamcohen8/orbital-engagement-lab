from __future__ import annotations

import os
from pathlib import Path

from sim.runtime_environment import configure_headless_runtime


def test_automation_configures_writable_matplotlib_and_font_caches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SIM_AUTOMATION", "1")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    status = configure_headless_runtime(cache_root=tmp_path / "plot-cache")

    assert status.ok
    assert status.enabled
    assert os.environ["MPLBACKEND"] == "Agg"
    assert Path(os.environ["MPLCONFIGDIR"]).is_dir()
    assert Path(os.environ["XDG_CACHE_HOME"]).is_dir()


def test_headless_runtime_preserves_explicit_cache_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mpl_config = tmp_path / "custom-mpl"
    xdg_cache = tmp_path / "custom-xdg"
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_config))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))

    status = configure_headless_runtime(cache_root=tmp_path / "unused-default")

    assert status.ok
    assert status.matplotlib_config_dir == str(mpl_config)
    assert status.xdg_cache_dir == str(xdg_cache)
    assert mpl_config.is_dir()
    assert xdg_cache.is_dir()


def test_interactive_runtime_does_not_force_headless_environment(monkeypatch) -> None:
    for name in ("SIM_AUTOMATION", "CI", "MPLBACKEND", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)

    status = configure_headless_runtime()

    assert not status.enabled
    assert "MPLBACKEND" not in os.environ
    assert "MPLCONFIGDIR" not in os.environ
    assert "XDG_CACHE_HOME" not in os.environ
