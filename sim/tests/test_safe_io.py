from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from sim.utils.io import SafeReadError, read_regular_file_nofollow


def test_bounded_nofollow_read_accepts_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "evidence.json"
    source.write_bytes(b'{"status":"ok"}\n')
    assert read_regular_file_nofollow(source, min_bytes=1, max_bytes=100) == source.read_bytes()


def test_bounded_nofollow_read_rejects_size_and_non_file(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.write_bytes(b"")
    with pytest.raises(SafeReadError, match="between 1 and 10 bytes"):
        read_regular_file_nofollow(source, min_bytes=1, max_bytes=10)
    with pytest.raises(SafeReadError, match="regular file"):
        read_regular_file_nofollow(tmp_path, max_bytes=10)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform does not support symlinks")
def test_bounded_nofollow_read_rejects_final_and_parent_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    source = real / "evidence.json"
    source.write_text("{}", encoding="utf-8")
    final_link = tmp_path / "linked.json"
    final_link.symlink_to(source)
    parent_link = tmp_path / "linked-parent"
    parent_link.symlink_to(real, target_is_directory=True)
    for candidate in (final_link, parent_link / source.name):
        with pytest.raises(SafeReadError):
            read_regular_file_nofollow(candidate, max_bytes=100)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS compatibility-root behavior")
def test_bounded_nofollow_read_accepts_standard_macos_temp_alias() -> None:
    with tempfile.TemporaryDirectory(prefix="oel-safe-read-", dir="/private/tmp") as directory:
        physical_source = Path(directory) / "evidence.json"
        physical_source.write_bytes(b'{"status":"ok"}\n')
        source = Path("/tmp") / Path(directory).name / physical_source.name
        source.write_bytes(b'{"status":"ok"}\n')
        assert read_regular_file_nofollow(source, min_bytes=1, max_bytes=100) == physical_source.read_bytes()
