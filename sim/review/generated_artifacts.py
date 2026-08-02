from __future__ import annotations

import shutil
from pathlib import Path


def clear_generated_review_artifacts(review_dir: str | Path) -> None:
    """Remove plots and provenance tied to a review store being replaced."""

    root = Path(review_dir)
    index_path = root / "generated_artifacts.json"
    figures_dir = root / "figures"
    try:
        index_path.unlink()
    except FileNotFoundError:
        pass
    if figures_dir.is_symlink() or figures_dir.is_file():
        figures_dir.unlink()
    elif figures_dir.is_dir():
        shutil.rmtree(figures_dir)


__all__ = ["clear_generated_review_artifacts"]
