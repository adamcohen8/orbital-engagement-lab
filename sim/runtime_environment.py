from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class HeadlessRuntimeStatus:
    enabled: bool
    matplotlib_backend: str = ""
    matplotlib_config_dir: str = ""
    xdg_cache_dir: str = ""
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def configure_headless_runtime(
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
) -> HeadlessRuntimeStatus:
    """Configure writable plotting/font caches before importing Matplotlib.

    OEL automation is headless when ``SIM_AUTOMATION`` or ``CI`` is truthy.
    An explicitly selected Agg backend is also treated as headless. Existing
    environment overrides are preserved.
    """
    enabled = bool(force or _truthy_env("SIM_AUTOMATION") or _truthy_env("CI") or _agg_backend_requested())
    if not enabled:
        return HeadlessRuntimeStatus(enabled=False)

    root = (
        Path(cache_root).expanduser()
        if cache_root is not None
        else Path(tempfile.gettempdir()) / "oel-matplotlib"
    )
    defaults = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(root / "config"),
        "XDG_CACHE_HOME": str(root / "cache"),
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)

    errors: list[str] = []
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME"):
        path = Path(os.environ[name]).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"{name}={path}: {exc}")

    return HeadlessRuntimeStatus(
        enabled=True,
        matplotlib_backend=os.environ["MPLBACKEND"],
        matplotlib_config_dir=os.environ["MPLCONFIGDIR"],
        xdg_cache_dir=os.environ["XDG_CACHE_HOME"],
        errors=tuple(errors),
    )


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def _agg_backend_requested() -> bool:
    return os.environ.get("MPLBACKEND", "").strip().lower() == "agg"
