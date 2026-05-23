import sqlite3
from pathlib import Path

import pytest

from mayring_core.memory.store import init_memory_db


def test_projects_source_index_exists(tmp_path: Path) -> None:
    p = tmp_path / "memory.db"
    init_memory_db(p).close()
    idx = {r[1] for r in sqlite3.connect(p).execute(
        "PRAGMA index_list('projects')").fetchall()}
    assert "idx_projects_source" in idx
