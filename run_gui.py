from __future__ import annotations

from importlib import import_module

import run_orw


def main(argv: list[str] | None = None) -> int:
    try:
        import_module("PySide6.QtWidgets")
    except ModuleNotFoundError as exc:
        if str(getattr(exc, "name", "") or "").split(".", 1)[0] == "PySide6":
            raise SystemExit(
                'PySide6 is not installed. Install ORW dependencies with `python -m pip install ".[gui]"`.'
            ) from exc
        raise
    return run_orw.main(argv, importer=import_module)

if __name__ == "__main__":
    raise SystemExit(main())
