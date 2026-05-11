"""Tests for wiki_v2 category-evidence (Mayring textmarker persistence)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.wiki_v2.store import (
    init_wiki_db, persist_category_evidence, get_category_evidence,
    delete_category_evidence,
)


@pytest.fixture
def wdb(tmp_path):
    db = init_wiki_db(tmp_path / "wiki_v2.db")
    yield db
    db.close()


def test_persist_and_read_by_category(wdb):
    n = persist_category_evidence(wdb, "bene", "src:foo.py", [
        {"span": [10, 25], "excerpt": "def validate_jwt", "category": "auth", "reasoning": "validates a JWT"},
    ], task="wie wird auth gemacht?", chunk_id="chk_1")
    assert n == 1
    ev = get_category_evidence(wdb, "bene", category="auth")
    assert len(ev) == 1
    assert ev[0]["category"] == "auth"
    assert ev[0]["span"] == [10, 25]
    assert ev[0]["excerpt"] == "def validate_jwt"
    assert ev[0]["task"] == "wie wird auth gemacht?"
    assert ev[0]["chunk_id"] == "chk_1"


def test_span_none_persisted_as_minus_one(wdb):
    persist_category_evidence(wdb, "bene", "src:x", [
        {"span": None, "excerpt": "paraphrased text not in chunk", "category": "validation", "reasoning": "r"},
    ])
    ev = get_category_evidence(wdb, "bene", category="validation")
    assert ev[0]["span"] is None  # -1 sentinel → None on read


def test_dedup_within_call(wdb):
    n = persist_category_evidence(wdb, "bene", "src:y", [
        {"span": [0, 5], "excerpt": "hello", "category": "greeting", "reasoning": "r"},
        {"span": [10, 15], "excerpt": "hello", "category": "greeting", "reasoning": "r2"},  # same (cat, excerpt)
        {"span": [0, 5], "excerpt": "hello", "category": "other", "reasoning": "r"},  # different cat → kept
    ])
    assert n == 2


def test_skips_markings_without_category_or_excerpt(wdb):
    n = persist_category_evidence(wdb, "bene", "src:z", [
        {"span": [0, 5], "excerpt": "valid", "category": "good", "reasoning": "r"},
        {"span": [0, 5], "excerpt": "", "category": "no-excerpt", "reasoning": "r"},
        {"span": [0, 5], "excerpt": "no-cat", "category": "", "reasoning": "r"},
        {"not": "a dict"},
    ])
    assert n == 1


def test_read_by_source(wdb):
    persist_category_evidence(wdb, "bene", "src:a", [
        {"span": [0, 3], "excerpt": "abc", "category": "c1", "reasoning": "r"},
        {"span": [4, 7], "excerpt": "def", "category": "c2", "reasoning": "r"},
    ])
    persist_category_evidence(wdb, "bene", "src:b", [
        {"span": [0, 3], "excerpt": "xyz", "category": "c1", "reasoning": "r"},
    ])
    ev_a = get_category_evidence(wdb, "bene", source_id="src:a")
    assert len(ev_a) == 2
    ev_c1 = get_category_evidence(wdb, "bene", category="c1")
    assert len(ev_c1) == 2  # one from each source


def test_workspace_isolation(wdb):
    persist_category_evidence(wdb, "alice", "src:a", [{"span": [0, 3], "excerpt": "abc", "category": "c1", "reasoning": "r"}])
    persist_category_evidence(wdb, "bob", "src:a", [{"span": [0, 3], "excerpt": "abc", "category": "c1", "reasoning": "r"}])
    assert len(get_category_evidence(wdb, "alice")) == 1
    assert len(get_category_evidence(wdb, "bob")) == 1


def test_delete_then_re_mark(wdb):
    persist_category_evidence(wdb, "bene", "src:remark", [
        {"span": [0, 3], "excerpt": "old", "category": "stale-cat", "reasoning": "r"},
    ])
    deleted = delete_category_evidence(wdb, "bene", "src:remark")
    assert deleted == 1
    assert get_category_evidence(wdb, "bene", source_id="src:remark") == []
    # re-mark
    persist_category_evidence(wdb, "bene", "src:remark", [
        {"span": [5, 8], "excerpt": "new", "category": "fresh-cat", "reasoning": "r"},
    ])
    ev = get_category_evidence(wdb, "bene", source_id="src:remark")
    assert len(ev) == 1 and ev[0]["category"] == "fresh-cat"


def test_empty_markings_returns_zero(wdb):
    assert persist_category_evidence(wdb, "bene", "src:empty", []) == 0


def test_category_lowercased_on_query(wdb):
    persist_category_evidence(wdb, "bene", "src:c", [
        {"span": [0, 3], "excerpt": "abc", "category": "MixedCase", "reasoning": "r"},
    ])
    # stored lowercased; query with any case works
    assert len(get_category_evidence(wdb, "bene", category="mixedcase")) == 1
    assert len(get_category_evidence(wdb, "bene", category="MIXEDCASE")) == 1
