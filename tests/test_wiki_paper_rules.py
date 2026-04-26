"""Tests for wiki paper rules and cache helpers."""
from src.memory.db_adapter import DBAdapter
from src.memory.store import _init_schema, get_paper_cache, set_paper_cache


def _db():
    db = DBAdapter.memory()
    _init_schema(db)
    return db


def test_paper_cache_miss_returns_none():
    assert get_paper_cache(_db(), "src1", "shared_concept") is None


def test_paper_cache_roundtrip():
    db = _db()
    set_paper_cache(db, "src1", "shared_concept", ["Mayring", "Inhaltsanalyse"])
    assert get_paper_cache(db, "src1", "shared_concept") == ["Mayring", "Inhaltsanalyse"]


def test_paper_cache_overwrite():
    db = _db()
    set_paper_cache(db, "src1", "shared_concept", ["A"])
    set_paper_cache(db, "src1", "shared_concept", ["B", "C"])
    assert get_paper_cache(db, "src1", "shared_concept") == ["B", "C"]
