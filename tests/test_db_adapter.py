from pathlib import Path
import pytest
from src.memory.db_adapter import DBAdapter


def test_memory_creates_in_memory_db():
    db = DBAdapter.memory()
    db.execute("CREATE TABLE t (x INTEGER)")
    db.execute("INSERT INTO t VALUES (1)")
    row = db.execute("SELECT x FROM t").fetchone()
    assert row[0] == 1


def test_create_opens_file_db(tmp_path):
    p = tmp_path / "test.db"
    db = DBAdapter.create(p)
    db.execute("CREATE TABLE t (x INTEGER)")
    db.commit()
    db.close()
    assert p.exists()


def test_get_columns_returns_column_names():
    db = DBAdapter.memory()
    db.execute("CREATE TABLE t (id INTEGER, name TEXT, score REAL)")
    cols = db.get_columns("t")
    assert cols == {"id", "name", "score"}


def test_get_columns_empty_for_missing_table():
    db = DBAdapter.memory()
    assert db.get_columns("nonexistent") == set()


def test_changes_returns_affected_rows():
    db = DBAdapter.memory()
    db.execute("CREATE TABLE t (x INTEGER)")
    db.execute("INSERT INTO t VALUES (1)")
    db.execute("INSERT INTO t VALUES (2)")
    db.execute("DELETE FROM t WHERE x > 0")
    assert db.changes() == 2


def test_wal_mode_set_by_create(tmp_path):
    p = tmp_path / "wal.db"
    db = DBAdapter.create(p)
    row = db.execute("PRAGMA journal_mode").fetchone()
    assert row[0] == "wal"
    db.close()


def test_foreign_keys_enabled():
    db = DBAdapter.memory()
    row = db.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1
