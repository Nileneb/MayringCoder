"""Write-leak fixes (core, "nie wieder out of context"):

  1. insert_chunk dropped igio_axis/confidence/classified_at entirely → a chunk
     constructed WITH an axis lost it on insert (forced every caller to a separate
     update_chunk_igio_axis). On re-ingest the axis must NOT be clobbered by an
     unclassified (empty) re-insert.
  2. _embed_one_with_retry: a single dropped embedder request used to leave the
     chunk in SQLite but never in Chroma = permanently unsearchable. Bounded retry
     rescues the transient case; permanent failure is re-raised (caller queues
     reembed-pending).
"""
from datetime import datetime, timezone

import pytest

from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import (
    init_memory_db, upsert_source, insert_chunk, get_chunk, update_chunk_igio_axis,
)
from mayring_core.memory.ingestion.core import _embed_one_with_retry


def _src(conn):
    s = Source(source_id="repo:o/r:f.py", source_type="repo_file", repo="o/r",
               path="f.py", branch="main", commit="", content_hash="sha256:1")
    upsert_source(conn, s)
    return s


def _chunk(source_id, igio_axis="", igio_confidence=0.0, text="def f(): pass"):
    return Chunk(
        chunk_id=Chunk.make_id(source_id, 0, "function"),
        source_id=source_id, chunk_level="function", ordinal=0,
        text=text, text_hash=Chunk.compute_text_hash(text),
        igio_axis=igio_axis, igio_confidence=igio_confidence,
        igio_classified_at=("2026-05-29" if igio_axis else ""),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def test_insert_persists_igio_axis(tmp_path):
    conn = init_memory_db(tmp_path / "m.db")
    s = _src(conn)
    insert_chunk(conn, _chunk(s.source_id, igio_axis="goal", igio_confidence=0.9))
    got = get_chunk(conn, Chunk.make_id(s.source_id, 0, "function"))
    assert got.igio_axis == "goal"          # was silently dropped before the fix
    assert got.igio_confidence == 0.9


def test_reingest_does_not_clobber_classified_axis(tmp_path):
    conn = init_memory_db(tmp_path / "m.db")
    s = _src(conn)
    cid = Chunk.make_id(s.source_id, 0, "function")
    # classified once (e.g. by the IGIO cron)
    insert_chunk(conn, _chunk(s.source_id, igio_axis="goal", igio_confidence=0.9))
    # re-ingest with a fresh, UNclassified chunk (axis="") — must preserve "goal"
    insert_chunk(conn, _chunk(s.source_id, igio_axis="", text="def f(): return 1"))
    got = get_chunk(conn, cid)
    assert got.igio_axis == "goal"          # classifier work preserved
    assert got.text == "def f(): return 1"  # but content WAS updated


def test_reingest_with_new_axis_updates(tmp_path):
    conn = init_memory_db(tmp_path / "m.db")
    s = _src(conn)
    cid = Chunk.make_id(s.source_id, 0, "function")
    insert_chunk(conn, _chunk(s.source_id, igio_axis="goal"))
    insert_chunk(conn, _chunk(s.source_id, igio_axis="issue", igio_confidence=0.7))
    got = get_chunk(conn, cid)
    assert got.igio_axis == "issue"         # a real new axis DOES update


def test_update_chunk_igio_axis_still_works_after_insert(tmp_path):
    """The existing separate-update path (repo_watching) must keep working."""
    conn = init_memory_db(tmp_path / "m.db")
    s = _src(conn)
    cid = Chunk.make_id(s.source_id, 0, "function")
    insert_chunk(conn, _chunk(s.source_id))         # axis=""
    update_chunk_igio_axis(conn, cid, "intervention", 0.5)
    assert get_chunk(conn, cid).igio_axis == "intervention"


def test_embed_retry_succeeds_on_second_attempt():
    calls = {"n": 0}

    def flaky(texts, url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("three.linn.games hiccup")
        return [[0.1, 0.2, 0.3]]

    emb = _embed_one_with_retry(flaky, "text", "http://three", attempts=2)
    assert emb == [0.1, 0.2, 0.3]
    assert calls["n"] == 2


def test_embed_retry_reraises_after_exhaustion():
    def always_fail(texts, url):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        _embed_one_with_retry(always_fail, "text", "http://three", attempts=2)
