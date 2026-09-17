"""Review rows must preserve every selected evidence value without ambiguity."""
import sqlite3
from pathlib import Path

import pytest

from sim.review import ReviewQueryError, ReviewWorkspace


@pytest.fixture
def workspace(tmp_path: Path):
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    with sqlite3.connect(review_dir / "run.sqlite") as conn:
        conn.execute("CREATE TABLE evidence (value REAL)")
        conn.execute("INSERT INTO evidence VALUES (2.0)")
    with ReviewWorkspace.open(tmp_path) as opened:
        yield opened


def test_case_distinct_aliases_preserve_their_own_values(workspace):
    result = workspace.query("SELECT value AS Range, value * 3 AS range FROM evidence")
    assert result.columns == ["Range", "range"]
    assert result.rows == [{"Range": 2.0, "range": 6.0}]


@pytest.mark.parametrize("where", ["", " WHERE 0"])
def test_duplicate_names_require_explicit_unique_aliases(workspace, where):
    with pytest.raises(ReviewQueryError, match="unique.*aliases"):
        workspace.query("SELECT a.value, b.value FROM evidence a JOIN evidence b" + where)
