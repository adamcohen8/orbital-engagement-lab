from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path


def main(argv: list[str] | None = None, *, importer=import_module) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Open an OEL output folder in OEL Evidence Studio, formerly the "
            "experimental Output Review Workbench. Use it to explore review-store "
            "data, ask for brief-ready plots, and save styled figures with provenance."
        )
    )
    parser.add_argument("--output", required=True, help="Completed OEL output folder to review.")
    args, qt_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    try:
        qt_widgets = importer("PySide6.QtWidgets")
        main_window_module = importer("sim.gui.main_window")
    except ModuleNotFoundError as exc:
        if str(getattr(exc, "name", "") or "").split(".", 1)[0] == "PySide6":
            raise SystemExit(
                'PySide6 is not installed. Install Evidence Studio dependencies with `python -m pip install ".[gui]"`.'
            ) from exc
        raise

    app = qt_widgets.QApplication([sys.argv[0], *qt_args])
    window = main_window_module.MainWindow(output_dir=Path(args.output).expanduser())
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
