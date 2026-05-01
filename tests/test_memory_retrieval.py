"""Tests for src/memory_retrieval.py."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.memory.schema import Chunk, RetrievalRecord, Source
from src.memory.store import init_memory_db, upsert_source, insert_chunk, add_feedback
from src.memory.retrieval import (
    _scope_filter,
    _symbolic_score,
    _tokenize,
    _recency_score,
    _source_affinity_score,
    _rerank,
    compress_for_prompt,
)


def _make_source(source_id: str, repo: str = "owner/test", source_type: str = "repo_file") -> Source:
    return Source(
        source_id=source_id,
        source_type=source_type,
        repo=repo,
        path=source_id.split(":", 2)[-1] if ":" in source_id else source_id,
        branch="main",
        commit="abc",
        content_hash="sha256:x",
        captured_at="2026-04-08T10:00:00+00:00",
    )


def _make_chunk(
    source_id: str,
    ordinal: int = 0,
    text: str = "def foo(): pass",
    category_labels: list[str] | None = None,
    created_at: str | None = None,
) -> Chunk:
    text_hash = Chunk.compute_text_hash(text)
    return Chunk(
        chunk_id=Chunk.make_id(source_id, ordinal, "function"),
        source_id=source_id,
        chunk_level="function",
        ordinal=ordinal,
        text=text,
        text_hash=text_hash,
        category_labels=category_labels or [],
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )


class TestScopeFilter:
    def test_filter_by_repo(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src_a = _make_source("repo:owner/a:foo.py", repo="owner/a")
        src_b = _make_source("repo:owner/b:bar.py", repo="owner/b")
        upsert_source(conn, src_a)
        upsert_source(conn, src_b)
        insert_chunk(conn, _make_chunk(src_a.source_id, 0))
        insert_chunk(conn, _make_chunk(src_b.source_id, 0))

        ids = _scope_filter(conn, repo="owner/a")
        assert len(ids) == 1

    def test_filter_by_category(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:owner/t:f.py")
        upsert_source(conn, src)
        insert_chunk(conn, _make_chunk(src.source_id, 0, category_labels=["auth", "api"]))
        insert_chunk(conn, _make_chunk(src.source_id, 1, category_labels=["utility"]))

        ids = _scope_filter(conn, categories=["auth"])
        assert len(ids) == 1

    def test_no_filter_returns_all_active(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:owner/t:f.py")
        upsert_source(conn, src)
        for i in range(3):
            insert_chunk(conn, _make_chunk(src.source_id, i))
        ids = _scope_filter(conn)
        assert len(ids) == 3

    def test_inactive_excluded(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:owner/t:f.py")
        upsert_source(conn, src)
        chunk = _make_chunk(src.source_id, 0)
        insert_chunk(conn, chunk)
        conn.execute("UPDATE chunks SET is_active = 0 WHERE chunk_id = ?", (chunk.chunk_id,))
        conn.commit()
        ids = _scope_filter(conn)
        assert len(ids) == 0


class TestSymbolicScore:
    def test_perfect_overlap(self) -> None:
        chunk = _make_chunk("repo:s", text="def authenticate_user(): pass")
        terms = _tokenize("authenticate user")
        score = _symbolic_score(chunk, terms)
        assert score > 0.5

    def test_zero_overlap(self) -> None:
        chunk = _make_chunk("repo:s", text="import os")
        terms = _tokenize("authentication login token")
        score = _symbolic_score(chunk, terms)
        assert score == 0.0

    def test_empty_query_returns_zero(self) -> None:
        chunk = _make_chunk("repo:s")
        assert _symbolic_score(chunk, set()) == 0.0

    def test_path_bonus_applied(self) -> None:
        chunk = _make_chunk("repo:owner/test:auth/login.py", text="placeholder")
        terms = _tokenize("auth")
        score = _symbolic_score(chunk, terms)
        assert score > 0.0  # path bonus kicks in


class TestRecencyScore:
    def test_fresh_chunk_is_near_one(self) -> None:
        chunk = _make_chunk("repo:s", created_at=datetime.now(timezone.utc).isoformat())
        score = _recency_score(chunk)
        assert score > 0.95

    def test_old_chunk_is_zero(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        chunk = _make_chunk("repo:s", created_at=old)
        score = _recency_score(chunk)
        assert score == 0.0

    def test_fifteen_days_old_is_half(self) -> None:
        mid = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
        chunk = _make_chunk("repo:s", created_at=mid)
        score = _recency_score(chunk)
        assert 0.4 < score < 0.6


class TestSourceAffinity:
    def test_matching_source(self) -> None:
        chunk = _make_chunk("repo:owner/test:src/auth.py")
        score = _source_affinity_score(chunk, "repo:owner/test:src/auth.py")
        assert score == 1.0

    def test_non_matching_source(self) -> None:
        chunk = _make_chunk("repo:owner/test:src/auth.py")
        score = _source_affinity_score(chunk, "repo:owner/test:src/other.py")
        assert score == 0.0

    def test_no_affinity(self) -> None:
        chunk = _make_chunk("repo:s")
        assert _source_affinity_score(chunk, None) == 0.0


class TestCompressForPrompt:
    def _make_record(self, source_id: str, text: str = "short text", score: float = 0.8) -> RetrievalRecord:
        return RetrievalRecord(
            chunk_id=f"chk_{source_id[-4:]}",
            score_vector=score,
            score_symbolic=score,
            score_recency=score,
            score_source_affinity=0.0,
            score_final=score,
            reasons=["test"],
            source_id=source_id,
            text=text,
            summary="",
            category_labels=["utility"],
        )

    def test_empty_results_returns_empty_string(self) -> None:
        assert compress_for_prompt([], 5000) == ""

    def test_output_within_budget(self) -> None:
        results = [self._make_record(f"repo:s:{i}", "x" * 200) for i in range(5)]
        output = compress_for_prompt(results, 300)
        assert len(output) <= 300

    def test_deduplication_by_source(self) -> None:
        same_source = "repo:owner/test:foo.py"
        results = [
            self._make_record(same_source, score=0.9),
            self._make_record(same_source, score=0.7),
            self._make_record("repo:owner/test:bar.py", score=0.5),
        ]
        output = compress_for_prompt(results, 5000)
        # Should only appear once per source
        assert output.count(same_source) == 1

    def test_contains_header(self) -> None:
        results = [self._make_record("repo:s:foo.py")]
        output = compress_for_prompt(results, 5000)
        assert "Memory Context" in output

    def test_prefers_summary_for_long_text(self) -> None:
        r = self._make_record("repo:s:foo.py", text="x" * 600)
        r.summary = "short summary"
        output = compress_for_prompt([r], 5000)
        assert "short summary" in output


class TestSessionCompacted:
    """Tests für session_compacted-Flag in search()."""

    def _make_conv_source(self, tmp_path):
        from src.memory.store import init_memory_db, upsert_source, insert_chunk
        conn = init_memory_db(tmp_path / "mc.db")
        src = Source(
            source_id="repo:conversation:summary/sess-1",
            source_type="conversation_summary",
            repo="conversation",
            path="summary/sess-1",
            branch="sess-1",
            commit="",
            content_hash="sha256:abc",
            captured_at="2026-04-08T10:00:00+00:00",
        )
        upsert_source(conn, src)
        text = "## Architektur\n\nWir haben MCP implementiert."
        text_hash = Chunk.compute_text_hash(text)
        chunk = Chunk(
            chunk_id=Chunk.make_id(src.source_id, 0, "section"),
            source_id=src.source_id,
            chunk_level="section",
            ordinal=0,
            text=text,
            text_hash=text_hash,
            category_labels=["architektur"],
            created_at="2026-04-08T10:00:00+00:00",
        )
        insert_chunk(conn, chunk)
        return conn, chunk

    def test_compacted_boosts_section_chunks_from_conversation(self, tmp_path) -> None:
        from src.memory.retrieval import search

        conn, chunk = self._make_conv_source(tmp_path)

        results_normal = search(
            query="MCP Architektur",
            conn=conn,
            chroma_collection=None,
            ollama_url="http://localhost:11434",
            opts={"top_k": 5},
            session_compacted=False,
        )
        results_compacted = search(
            query="MCP Architektur",
            conn=conn,
            chroma_collection=None,
            ollama_url="http://localhost:11434",
            opts={"top_k": 5},
            session_compacted=True,
        )

        assert len(results_normal) == 1
        assert len(results_compacted) == 1
        assert results_compacted[0].score_final > results_normal[0].score_final

    def test_compacted_false_no_score_boost(self, tmp_path) -> None:
        from src.memory.retrieval import search

        conn, chunk = self._make_conv_source(tmp_path)

        r1 = search("MCP", conn, None, "http://localhost:11434", opts={"top_k": 5}, session_compacted=False)
        r2 = search("MCP", conn, None, "http://localhost:11434", opts={"top_k": 5}, session_compacted=False)

        assert len(r1) == 1 and len(r2) == 1
        # Scores may differ by tiny recency-decay drift between calls; no compaction boost of 0.10
        assert abs(r1[0].score_final - r2[0].score_final) < 0.01


# ---------------------------------------------------------------------------
# Query-Cache tests
# ---------------------------------------------------------------------------

class TestQueryCache:
    """Tests for the in-process query cache in memory_retrieval."""

    def _make_source_and_chunk(self, tmp_path):
        from src.memory.store import init_memory_db, upsert_source, insert_chunk
        from src.memory.schema import Source, Chunk
        conn = init_memory_db(tmp_path / "memory.db")
        src = Source(
            source_id="src::query_cache_test",
            source_type="repo_file",
            repo="https://github.com/test/repo",
            path="src/test.py",
            content_hash="sha256:abc123",
        )
        upsert_source(conn, src)
        chunk = Chunk(
            chunk_id=Chunk.make_id(src.source_id, 0, "function"),
            source_id=src.source_id,
            chunk_level="function",
            ordinal=0,
            text="def hello(): pass",
            text_hash="sha256:def456",
            category_labels=["domain"],
            created_at="2026-04-08T10:00:00+00:00",
        )
        insert_chunk(conn, chunk)
        return conn, chunk

    def test_cache_hit_skips_scope_filter(self, tmp_path) -> None:
        """Second identical search() call hits cache — _scope_filter not called again."""
        from unittest.mock import patch
        from src.memory.retrieval import search, invalidate_query_cache

        invalidate_query_cache()
        conn, chunk = self._make_source_and_chunk(tmp_path)

        call_count = 0
        original_scope_filter = __import__("src.memory.retrieval", fromlist=["_scope_filter"])._scope_filter

        def counting_scope_filter(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_scope_filter(*args, **kwargs)

        with patch("src.memory.retrieval._scope_filter", side_effect=counting_scope_filter):
            r1 = search("hello", conn, None, "http://localhost:11434", opts={"top_k": 5})
            r2 = search("hello", conn, None, "http://localhost:11434", opts={"top_k": 5})

        assert len(r1) == 1
        assert len(r2) == 1
        assert r1[0].chunk_id == r2[0].chunk_id
        # _scope_filter called exactly once — second call hit cache
        assert call_count == 1

    def test_invalidate_clears_cache(self, tmp_path) -> None:
        """After invalidate_query_cache(), next search() runs fresh (calls _scope_filter)."""
        from unittest.mock import patch
        from src.memory.retrieval import search, invalidate_query_cache

        invalidate_query_cache()
        conn, chunk = self._make_source_and_chunk(tmp_path)

        call_count = 0
        original = __import__("src.memory.retrieval", fromlist=["_scope_filter"])._scope_filter

        def counting(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        with patch("src.memory.retrieval._scope_filter", side_effect=counting):
            search("hello", conn, None, "http://localhost:11434", opts={"top_k": 5})
            invalidate_query_cache()
            search("hello", conn, None, "http://localhost:11434", opts={"top_k": 5})

        # Two full searches = two _scope_filter calls
        assert call_count == 2

    def test_different_opts_produce_different_cache_keys(self) -> None:
        """Different opts → different cache entries, no cross-contamination."""
        from src.memory.retrieval import _cache_key

        k1 = _cache_key("hello", {"top_k": 5}, False)
        k2 = _cache_key("hello", {"top_k": 10}, False)
        k3 = _cache_key("hello", {"top_k": 5}, True)

        assert k1 != k2
        assert k1 != k3
        assert k2 != k3


class TestFeedbackRanking:
    def test_positive_feedback_ranks_higher(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:t/r:f.py")
        upsert_source(conn, src)
        # Two chunks with identical text (same symbolic/vector scores)
        ca = _make_chunk(src.source_id, 0, text="def process(): return result")
        cb = _make_chunk(src.source_id, 1, text="def process(): return result")
        insert_chunk(conn, ca)
        insert_chunk(conn, cb)
        add_feedback(conn, ca.chunk_id, "positive")

        records = _rerank(
            [ca, cb],
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            top_k=2,
            conn=conn,
        )
        assert records[0].chunk_id == ca.chunk_id

    def test_negative_feedback_ranks_lower(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:t/r:g.py")
        upsert_source(conn, src)
        ca = _make_chunk(src.source_id, 0, text="def compute(): pass")
        cb = _make_chunk(src.source_id, 1, text="def compute(): pass")
        insert_chunk(conn, ca)
        insert_chunk(conn, cb)
        add_feedback(conn, ca.chunk_id, "negative")

        records = _rerank(
            [ca, cb],
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            top_k=2,
            conn=conn,
        )
        assert records[0].chunk_id == cb.chunk_id

    def test_neutral_no_change(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:t/r:h.py")
        upsert_source(conn, src)
        ca = _make_chunk(src.source_id, 0, text="x")
        cb = _make_chunk(src.source_id, 1, text="x")
        insert_chunk(conn, ca)
        insert_chunk(conn, cb)
        # No feedback for either — both score equally; order stable by ordinal
        records = _rerank(
            [ca, cb],
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            top_k=2,
            conn=conn,
        )
        assert len(records) == 2

    def test_llm_advisor_boosts_high_score(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:t/r:llm.py")
        upsert_source(conn, src)
        ca = _make_chunk(src.source_id, 0, text="aaa")
        cb = _make_chunk(src.source_id, 1, text="aaa")
        insert_chunk(conn, ca)
        insert_chunk(conn, cb)
        records = _rerank(
            [ca, cb],
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            top_k=2,
            conn=conn,
            llm_scores={ca.chunk_id: 0.9, cb.chunk_id: 0.1},
        )
        assert records[0].chunk_id == ca.chunk_id
        assert "llm_advisor_high" in records[0].reasons

    def test_llm_advisor_missing_is_neutral(self, tmp_path: Path) -> None:
        """When llm_scores is None or empty, all chunks get default 0.5 — order
        falls back to other signals (here: stable insertion order)."""
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:t/r:nollm.py")
        upsert_source(conn, src)
        ca = _make_chunk(src.source_id, 0, text="x")
        cb = _make_chunk(src.source_id, 1, text="x")
        insert_chunk(conn, ca)
        insert_chunk(conn, cb)
        records = _rerank(
            [ca, cb],
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            {ca.chunk_id: 0.5, cb.chunk_id: 0.5},
            top_k=2,
            conn=conn,
            llm_scores=None,
        )
        # Both should have identical score_final (no llm bias)
        assert abs(records[0].score_final - records[1].score_final) < 1e-6
