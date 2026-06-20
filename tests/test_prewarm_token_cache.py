"""Pre-warm of the per-worker symbolic-score token cache (_CHUNK_TOKENS).

WHY: a cold worker re-tokenises the whole workspace (~3.95s for 10k chunks)
on its first search; pre-warming on startup makes every search warm (0.075s).
This guards that the warm path populates the cache for active chunks only and
is a no-op-safe pure function (the startup hook just runs it in a daemon thread).
"""
from datetime import datetime, timezone
from pathlib import Path

from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import init_memory_db, upsert_source, insert_chunk
import mayring_core.memory.retrieval as retr
from src.api.memory_service import prewarm_token_cache


def _src(source_id: str) -> Source:
    return Source(source_id=source_id, source_type="repo_file", repo="owner/test",
                  path="f.py", branch="main", commit="abc", content_hash="sha256:x",
                  captured_at="2026-04-08T10:00:00+00:00")


def _chunk(source_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(chunk_id=Chunk.make_id(source_id, ordinal, "function"),
                 source_id=source_id, chunk_level="function", ordinal=ordinal,
                 text=text, text_hash=Chunk.compute_text_hash(text),
                 category_labels=[], created_at=datetime.now(timezone.utc).isoformat())


def test_prewarm_populates_cache_for_active_chunks(tmp_path: Path) -> None:
    conn = init_memory_db(tmp_path / "m.db")
    src = _src("repo:owner/test:f.py")
    upsert_source(conn, src)
    active = [_chunk(src.source_id, i, f"reranker active version sync token {i}")
              for i in range(3)]
    for c in active:
        insert_chunk(conn, c)
    inactive = _chunk(src.source_id, 99, "inactive chunk should be skipped")
    insert_chunk(conn, inactive)
    conn.execute("UPDATE chunks SET is_active = 0 WHERE chunk_id = ?", (inactive.chunk_id,))
    conn.commit()

    retr._CHUNK_TOKENS.clear()
    n = prewarm_token_cache(conn)

    assert n == 3  # only active chunks
    for c in active:
        assert c.chunk_id in retr._CHUNK_TOKENS
        text_hash, tokens = retr._CHUNK_TOKENS[c.chunk_id]
        assert text_hash == c.text_hash
        assert "reranker" in tokens
    assert inactive.chunk_id not in retr._CHUNK_TOKENS  # is_active=0 not warmed


def test_prewarm_empty_db_is_zero(tmp_path: Path) -> None:
    conn = init_memory_db(tmp_path / "m.db")
    retr._CHUNK_TOKENS.clear()
    assert prewarm_token_cache(conn) == 0


def test_prewarmed_token_set_is_cache_hit(tmp_path: Path) -> None:
    """After prewarm, _chunk_token_set must not re-tokenise (identical scores, fast)."""
    conn = init_memory_db(tmp_path / "m.db")
    src = _src("repo:owner/test:f.py")
    upsert_source(conn, src)
    c = _chunk(src.source_id, 0, "device registry write routing cap")
    insert_chunk(conn, c)
    conn.commit()

    retr._CHUNK_TOKENS.clear()
    prewarm_token_cache(conn)

    calls = {"n": 0}
    orig = retr._tokenize

    def _counting(text: str):
        calls["n"] += 1
        return orig(text)

    retr._tokenize = _counting
    try:
        tokens = retr._chunk_token_set(c)  # cached → no _tokenize calls
    finally:
        retr._tokenize = orig
    assert calls["n"] == 0
    assert "device" in tokens
