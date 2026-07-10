from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class CleanBuildPy(build_py):
    """Prevent ignored stale build/lib files from leaking into wheels."""

    def run(self) -> None:
        build_lib = Path(self.build_lib)
        if build_lib.exists():
            shutil.rmtree(build_lib)
        super().run()


setup(cmdclass={"build_py": CleanBuildPy})
