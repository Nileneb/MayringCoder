"""Recency-Lane: der laufende Session-Thread bleibt sichtbar, auch wenn die
semantische Ähnlichkeit schwach ist ("nie wieder out of context").

Zwei novel Bausteine:
  1. _session_recency_ids  — resolve die rollierende conversation_summary-Source
     dieser Session zu chunk_ids (matcht micro-batch source_id-Schema).
  2. _rerank session-floor  — Session-Chunks bekommen score_final >= floor +
     reason 'session-recency', damit sie das Hook-Gate (0.45) überleben.
"""
from datetime import datetime, timedelta, timezone

from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import init_memory_db, upsert_source, insert_chunk
from mayring_core.memory.retrieval import (
    _rerank, _session_recency_ids, _SESSION_RECENCY_FLOOR,
)


def _conv_source(workspace_id: str, session_id: str) -> Source:
    source_id = f"conversation:{workspace_id}:{session_id[:16]}"
    return Source(
        source_id=source_id, source_type="conversation_summary",
        repo=workspace_id, path=f"{workspace_id}/incremental", branch="local",
        commit="", content_hash="sha256:deadbeef",
    )


def _repo_source(source_id: str) -> Source:
    return Source(
        source_id=source_id, source_type="repo_file", repo="owner/r",
        path="file.py", branch="main", commit="", content_hash="sha256:cafe",
    )


def _chunk(source_id: str, ordinal: int = 0, text: str = "session work",
           workspace_id: str = "bene", created_at: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=Chunk.make_id(source_id, ordinal, "section"),
        source_id=source_id, chunk_level="section", ordinal=ordinal,
        text=text, text_hash=Chunk.compute_text_hash(text),
        workspace_id=workspace_id,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )


def test_session_recency_ids_resolves_conversation_source(tmp_path):
    conn = init_memory_db(tmp_path / "m.db")
    src = _conv_source("bene", "sess-abc-123456789")
    upsert_source(conn, src)
    c = _chunk(src.source_id, 0)
    insert_chunk(conn, c)

    ids = _session_recency_ids(conn, "bene", "sess-abc-123456789")
    assert c.chunk_id in ids


def test_session_recency_ids_matches_slug_not_uuid(tmp_path):
    """REGRESSION (live-found): conversation source_ids use the repo/workspace
    SLUG as the middle segment ('conversation:mayringcoder:<sess>'), NOT the
    workspace UUID. The lane must match by session-suffix within the workspace,
    or it finds nothing in prod (chunk.workspace_id is the UUID)."""
    conn = init_memory_db(tmp_path / "m.db")
    uuid_ws = "019e14d6-0489-7348-bca8-e29c11293cb7"
    # source middle = slug 'mayringcoder', but the chunk lives under the UUID ws
    src = Source(
        source_id="conversation:mayringcoder:33abee6e-ff1c-42",
        source_type="conversation_summary", repo="mayringcoder",
        path="mayringcoder/incremental", branch="local", commit="",
        content_hash="sha256:abc",
    )
    upsert_source(conn, src)
    c = _chunk(src.source_id, 0, workspace_id=uuid_ws)
    insert_chunk(conn, c)

    ids = _session_recency_ids(conn, uuid_ws, "33abee6e-ff1c-42xxxxx")
    assert c.chunk_id in ids                  # found despite slug != uuid


def test_session_recency_ids_empty_without_session(tmp_path):
    conn = init_memory_db(tmp_path / "m.db")
    assert _session_recency_ids(conn, "bene", None) == []
    assert _session_recency_ids(conn, "bene", "") == []
    # unknown session → no source → empty
    assert _session_recency_ids(conn, "bene", "nope") == []


def test_rerank_floors_session_chunk_above_gate(tmp_path):
    conn = init_memory_db(tmp_path / "m.db")
    src = _conv_source("bene", "sess-1")
    upsert_source(conn, src)
    other_src = _repo_source("repo:owner/r:file.py")
    upsert_source(conn, other_src)
    # old → low natural recency, so the floor (not freshness) is what lifts the
    # session chunk; isolates the session-recency effect from the recency stage.
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    sess = _chunk(src.source_id, 0, text="what I just did", created_at=old)
    other = _chunk(other_src.source_id, 0, text="unrelated code", created_at=old)
    insert_chunk(conn, sess)
    insert_chunk(conn, other)

    # Both semantically weak (vector 0.1, symbolic 0.0) — the OLD gate would
    # drop everything. With the session-floor, the session chunk survives.
    records = _rerank(
        [sess, other],
        {sess.chunk_id: 0.1, other.chunk_id: 0.1},
        {sess.chunk_id: 0.0, other.chunk_id: 0.0},
        top_k=8, conn=conn,
        session_chunk_ids={sess.chunk_id},
    )
    by_id = {r.chunk_id: r for r in records}
    assert by_id[sess.chunk_id].score_final >= _SESSION_RECENCY_FLOOR
    assert "session-recency" in by_id[sess.chunk_id].reasons
    # the non-session chunk is NOT floored
    assert by_id[other.chunk_id].score_final < _SESSION_RECENCY_FLOOR
    assert "session-recency" not in by_id[other.chunk_id].reasons


def test_rerank_no_session_ids_is_noop(tmp_path):
    conn = init_memory_db(tmp_path / "m.db")
    src = _conv_source("bene", "sess-2")
    upsert_source(conn, src)
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    c = _chunk(src.source_id, 0, created_at=old)
    insert_chunk(conn, c)
    records = _rerank(
        [c], {c.chunk_id: 0.1}, {c.chunk_id: 0.0},
        top_k=8, conn=conn,  # no session_chunk_ids
    )
    assert records[0].score_final < _SESSION_RECENCY_FLOOR  # not floored
    assert "session-recency" not in records[0].reasons


def test_session_chunk_guaranteed_in_topk_despite_noise(tmp_path):
    """REGRESSION (live-found): nomic gives ~0.55-0.7 to *everything*, so a
    session chunk merely floored to 0.5 gets buried below noise and truncated by
    top_k. The lane must GUARANTEE the session thread is in the returned top_k,
    not just floor its score — otherwise "nie wieder out of context" fails."""
    conn = init_memory_db(tmp_path / "m.db")
    src = _conv_source("bene", "sess-noise")
    upsert_source(conn, src)
    other_src = _repo_source("repo:owner/r:file.py")
    upsert_source(conn, other_src)

    sess = _chunk(src.source_id, 0, text="what I just did")
    insert_chunk(conn, sess)
    # 5 noisy non-session chunks that all out-score the 0.5 floor
    noise = []
    for i in range(5):
        ns = _repo_source(f"repo:owner/r:n{i}.py")
        upsert_source(conn, ns)
        c = _chunk(ns.source_id, 0, text=f"noise {i}")
        insert_chunk(conn, c)
        noise.append(c)

    vec = {n.chunk_id: 0.65 for n in noise}     # all noise ranks ~0.65
    vec[sess.chunk_id] = 0.1                     # session chunk weak semantically
    sym = {c.chunk_id: 0.0 for c in noise + [sess]}

    records = _rerank(
        noise + [sess], vec, sym, top_k=3, conn=conn,
        session_chunk_ids={sess.chunk_id},
    )
    ids = [r.chunk_id for r in records]
    assert sess.chunk_id in ids                  # guaranteed despite top_k=3 + noise
    assert len(records) <= 3                      # still respects top_k


def test_session_chunk_already_strong_not_lowered(tmp_path):
    """The floor lifts, never caps — a strong session chunk keeps its score."""
    conn = init_memory_db(tmp_path / "m.db")
    src = _conv_source("bene", "sess-3")
    upsert_source(conn, src)
    c = _chunk(src.source_id, 0, text="authenticate user login token")
    insert_chunk(conn, c)
    records = _rerank(
        [c], {c.chunk_id: 0.95}, {c.chunk_id: 1.0},
        top_k=8, conn=conn, session_chunk_ids={c.chunk_id},
    )
    assert records[0].score_final >= _SESSION_RECENCY_FLOOR  # at least floor
    assert records[0].score_final > 0.5  # but driven by its real strong signals
