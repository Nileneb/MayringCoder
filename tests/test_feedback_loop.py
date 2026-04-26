"""Tests for the Memory Feedback Loop — positive/negative feedback changes retrieval ranking."""
from __future__ import annotations

import datetime

import pytest

from src.memory.db_adapter import DBAdapter
from src.memory.schema import Chunk, Source
from src.memory.store import _init_schema, add_feedback, get_feedback_score, insert_chunk, upsert_source


def _db() -> DBAdapter:
    db = DBAdapter.memory()
    _init_schema(db)
    return db


def _make_chunk(chunk_id: str, source_id: str, text: str, db: DBAdapter) -> Chunk:
    src = Source(source_id=source_id, source_type="note", repo="r", path="p")
    upsert_source(db, src, workspace_id="ws")
    now = datetime.datetime.utcnow().isoformat()
    chunk = Chunk(
        chunk_id=chunk_id, source_id=source_id, text=text,
        text_hash=f"h-{chunk_id}", dedup_key=f"d-{chunk_id}",
        created_at=now, workspace_id="ws",
    )
    insert_chunk(db, chunk)
    return chunk


# ---------------------------------------------------------------------------
# get_feedback_score
# ---------------------------------------------------------------------------

def test_feedback_score_defaults_to_neutral():
    db = _db()
    assert get_feedback_score(db, "nonexistent") == 0.5


def test_feedback_score_all_positive():
    db = _db()
    _make_chunk("chk1", "src1", "text", db)
    for _ in range(5):
        add_feedback(db, "chk1", "positive")
    assert get_feedback_score(db, "chk1") == 1.0


def test_feedback_score_all_negative():
    db = _db()
    _make_chunk("chk2", "src2", "text", db)
    for _ in range(3):
        add_feedback(db, "chk2", "negative")
    assert get_feedback_score(db, "chk2") == 0.0


def test_feedback_score_mixed():
    db = _db()
    _make_chunk("chk3", "src3", "text", db)
    add_feedback(db, "chk3", "positive")
    add_feedback(db, "chk3", "negative")
    score = get_feedback_score(db, "chk3")
    assert score == 0.5


# ---------------------------------------------------------------------------
# _rerank: positive feedback → higher score_final
# ---------------------------------------------------------------------------

def test_positive_feedback_boosts_ranking():
    """Chunk with 5x positive feedback must outrank identical chunk with no feedback."""
    from src.memory.retrieval import _rerank

    db = _db()
    chunk_fb = _make_chunk("chk-fb", "src-fb", "authentication flow context", db)
    chunk_nofb = _make_chunk("chk-nofb", "src-nofb", "authentication flow context", db)

    for _ in range(5):
        add_feedback(db, "chk-fb", "positive")

    # Equal vector + symbolic scores — only feedback should differ
    vector_scores = {"chk-fb": 0.6, "chk-nofb": 0.6}
    symbolic_scores = {"chk-fb": 0.4, "chk-nofb": 0.4}

    ranked = _rerank(
        [chunk_fb, chunk_nofb],
        vector_scores, symbolic_scores,
        top_k=2, conn=db,
    )

    assert ranked[0].chunk_id == "chk-fb", "positively-rated chunk must rank first"
    assert ranked[0].score_final > ranked[1].score_final


def test_negative_feedback_lowers_ranking():
    """Chunk with 5x negative feedback must rank below chunk with no feedback."""
    from src.memory.retrieval import _rerank

    db = _db()
    chunk_neg = _make_chunk("chk-neg", "src-neg", "some context", db)
    chunk_clean = _make_chunk("chk-clean", "src-clean", "some context", db)

    for _ in range(5):
        add_feedback(db, "chk-neg", "negative")

    vector_scores = {"chk-neg": 0.6, "chk-clean": 0.6}
    symbolic_scores = {"chk-neg": 0.4, "chk-clean": 0.4}

    ranked = _rerank(
        [chunk_neg, chunk_clean],
        vector_scores, symbolic_scores,
        top_k=2, conn=db,
    )

    assert ranked[0].chunk_id == "chk-clean", "negatively-rated chunk must rank last"
    assert ranked[0].score_final > ranked[1].score_final


def test_neutral_feedback_has_no_effect():
    """Neutral-marked chunk ranks the same as chunk with no feedback."""
    from src.memory.retrieval import _rerank

    db = _db()
    chunk_neutral = _make_chunk("chk-neutral", "src-neutral", "text", db)
    chunk_none = _make_chunk("chk-none", "src-none", "text", db)

    add_feedback(db, "chk-neutral", "neutral")

    vector_scores = {"chk-neutral": 0.5, "chk-none": 0.5}
    symbolic_scores = {"chk-neutral": 0.3, "chk-none": 0.3}

    ranked = _rerank(
        [chunk_neutral, chunk_none],
        vector_scores, symbolic_scores,
        top_k=2, conn=db,
    )

    assert abs(ranked[0].score_final - ranked[1].score_final) < 0.01
