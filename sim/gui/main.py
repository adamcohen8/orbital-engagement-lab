from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from sim.gui.main_window import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Orbital Engagement Lab GUI.")
    parser.add_argument("--output", help="Open an existing output folder in Output Review Workbench mode.")
    args, qt_args = parser.parse_known_args(sys.argv[1:])
    app = QApplication([sys.argv[0], *qt_args])
    window = MainWindow(output_dir=Path(args.output).expanduser() if args.output else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
