"""Tests for predictive topic transitions (Markov chain)."""
from __future__ import annotations
import json

from src.memory.predictive import (
    TopicTransition,
    _extract_topics_from_text,
    build_transition_matrix,
    predict_next_topics,
    persist_transitions,
    load_transitions,
    update_transitions_incremental,
    predict_next_topics_for_query,
)
from src.memory.store import init_memory_db


def test_extract_topics_ordered_and_dedup():
    kw_index = {"auth": ["Authentication"], "login": ["Authentication"], "payment": ["Billing"]}
    topics = _extract_topics_from_text("User auth login then payment; auth again", kw_index)
    assert topics == ["Authentication", "Billing"]


def test_extract_topics_empty_on_missing_keywords():
    assert _extract_topics_from_text("random text", {}) == []
    assert _extract_topics_from_text("", {"foo": ["Bar"]}) == []


def test_predict_next_topics_sorted_by_probability():
    matrix = {"A": {"B": 7, "C": 2, "D": 1}}
    preds = predict_next_topics("A", matrix, top_k=2)
    assert len(preds) == 2
    assert preds[0].to_topic == "B"
    assert preds[0].probability == 0.7
    assert preds[1].to_topic == "C"


def test_predict_next_empty_on_unknown_topic():
    assert predict_next_topics("Nope", {"A": {"B": 1}}) == []


def test_persist_and_load_transitions_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    conn = init_memory_db(db)
    matrix = {"A": {"B": 5, "C": 1}, "B": {"A": 3}}
    persist_transitions(matrix, conn)
    loaded = load_transitions(conn)
    assert loaded == matrix


def test_persist_transitions_upsert_updates_count(tmp_path):
    db = tmp_path / "t.db"
    conn = init_memory_db(db)
    persist_transitions({"A": {"B": 1}}, conn)
    persist_transitions({"A": {"B": 9}}, conn)
    loaded = load_transitions(conn)
    assert loaded["A"]["B"] == 9


def test_build_transition_matrix_from_summaries(tmp_path, monkeypatch):
    import src.memory.predictive as pred_mod
    db = tmp_path / "t.db"
    conn = init_memory_db(db)

    conn.execute(
        'INSERT INTO sources(source_id, source_type, repo, path, branch, "commit", content_hash, captured_at) VALUES(?,?,?,?,?,?,?,?)',
        ("conversation:demo:s1", "conversation_summary", "demo", "demo/1", "local", "", "sha256:abc", "2026-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT INTO chunks(chunk_id, source_id, chunk_level, ordinal, start_offset, end_offset, text, text_hash, dedup_key, created_at, is_active) VALUES(?,?,?,?,?,?,?,?,?,?,1)",
        ("c1", "conversation:demo:s1", "file", 0, 0, 10, "auth flow then billing", "h1", "d1", "2026-01-01T00:00:00"),
    )
    conn.commit()

    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "billing": ["Billing"]})

    matrix = build_transition_matrix(conn, repo_slug="demo", limit=10)
    assert matrix == {"Auth": {"Billing": 1}}


def test_build_transition_matrix_empty_when_no_index(tmp_path, monkeypatch):
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index", lambda slug: {})
    assert build_transition_matrix(conn) == {}


def test_incremental_update_bumps_counts(tmp_path, monkeypatch):
    """Issue #55 follow-up: per-conversation incremental updates instead
    of batch rebuild. First call inserts (count=1), second on overlapping
    pairs increments via UPSERT."""
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "billing": ["Billing"], "search": ["Search"]})

    # First summary: Auth → Billing
    n1 = update_transitions_incremental("auth flow then billing", conn, "demo")
    assert n1 == 1
    assert load_transitions(conn) == {"Auth": {"Billing": 1}}

    # Second summary: Auth → Billing again, plus Billing → Search
    n2 = update_transitions_incremental("auth then billing then search", conn, "demo")
    assert n2 == 2
    matrix = load_transitions(conn)
    assert matrix == {"Auth": {"Billing": 2}, "Billing": {"Search": 1}}


def test_incremental_skips_when_no_index(tmp_path, monkeypatch):
    """No wiki_index.json yet (cold-start): hook must not crash + return 0."""
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index", lambda slug: {})
    assert update_transitions_incremental("some text", conn, "demo") == 0


def test_incremental_skips_self_transitions(tmp_path, monkeypatch):
    """A → A is meaningless (text repeats same topic) — must not bump."""
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "login": ["Auth"]})
    # extract_topics dedup'es to ["Auth"] only → no transitions possible
    n = update_transitions_incremental("auth login auth login", conn, "demo")
    assert n == 0
    assert load_transitions(conn) == {}


def test_predict_for_query_uses_persisted_matrix(tmp_path, monkeypatch):
    """The retrieval-side predictor must read from DB, not need the in-memory
    matrix passed by the caller."""
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "billing": ["Billing"]})

    persist_transitions({"Auth": {"Billing": 4, "Search": 1}}, conn)

    preds = predict_next_topics_for_query("does auth work?", conn, "demo", top_k=2)
    assert len(preds) >= 1
    assert preds[0].from_topic == "Auth"
    assert preds[0].to_topic == "Billing"  # 4/5 = 0.8 wins over Search 1/5


def test_predict_for_query_empty_when_query_has_no_known_topics(tmp_path, monkeypatch):
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"]})
    persist_transitions({"Auth": {"Billing": 1}}, conn)
    assert predict_next_topics_for_query("nothing relevant here", conn, "demo") == []
