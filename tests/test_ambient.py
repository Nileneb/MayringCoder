"""Tests for ambient context layer (src/memory/ambient.py)."""
import sqlite3
import pytest
from pathlib import Path

from src.memory.ambient import (
    _load_recent_conversations,
    _load_recent_issues,
    _load_wiki_top_connections,
    load_ambient_snapshot,
    generate_ambient_snapshot,
    _cosine,
    trigger_scan,
    _is_trigger_active,
)


def _init_test_db() -> sqlite3.Connection:
    """Create an in-memory test DB with full schema."""
    from src.memory.store import _init_schema
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _init_schema(conn)
    return conn


class TestLoadRecentConversations:
    """Test _load_recent_conversations()."""

    def test_empty_database(self):
        """Empty DB returns empty list."""
        conn = _init_test_db()
        result = _load_recent_conversations(conn, "myrepo")
        assert result == []
        conn.close()

    def test_filters_by_repo(self):
        """Only returns conversations for matching repo."""
        conn = _init_test_db()

        # Insert source with repo="myrepo"
        conn.execute(
            "INSERT INTO sources (source_id, source_type, repo, path, branch, \"commit\", content_hash, captured_at) VALUES (?,?,?,?,?,?,?,?)",
            ("conv:myrepo:1", "conversation_summary", "myrepo", "conv/1", "main", "", "sha256:abc", "2026-01-01T00:00:00")
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, chunk_level, is_active, created_at) VALUES (?,?,?,?,?,?)",
            ("chk1", "conv:myrepo:1", "This is conversation 1", "section", 1, "2026-01-01T00:00:00")
        )

        # Insert source with repo="otherrepo"
        conn.execute(
            "INSERT INTO sources (source_id, source_type, repo, path, branch, \"commit\", content_hash, captured_at) VALUES (?,?,?,?,?,?,?,?)",
            ("conv:otherrepo:1", "conversation_summary", "otherrepo", "conv/1", "main", "", "sha256:def", "2026-01-01T00:00:00")
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, chunk_level, is_active, created_at) VALUES (?,?,?,?,?,?)",
            ("chk2", "conv:otherrepo:1", "This is conversation 2", "section", 1, "2026-01-01T00:00:00")
        )
        conn.commit()

        result = _load_recent_conversations(conn, "myrepo")
        assert len(result) == 1
        assert "conversation 1" in result[0]
        conn.close()


class TestLoadRecentIssues:
    """Test _load_recent_issues()."""

    def test_empty_database(self):
        """Empty DB returns empty list."""
        conn = _init_test_db()
        result = _load_recent_issues(conn, "myrepo")
        assert result == []
        conn.close()

    def test_loads_issue_summaries(self):
        """Loads issue summaries from DB."""
        conn = _init_test_db()

        conn.execute(
            "INSERT INTO sources (source_id, source_type, repo, path, branch, \"commit\", content_hash, captured_at) VALUES (?,?,?,?,?,?,?,?)",
            ("issue:myrepo:1", "github_issue", "myrepo", "issues/1", "main", "", "sha256:abc", "2026-01-01T00:00:00")
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, chunk_level, is_active, created_at) VALUES (?,?,?,?,?,?)",
            ("chk1", "issue:myrepo:1", "Issue about login bug", "section", 1, "2026-01-01T00:00:00")
        )
        conn.commit()

        result = _load_recent_issues(conn, "myrepo")
        assert len(result) == 1
        assert "login bug" in result[0]
        conn.close()


class TestLoadWikiTopConnections:
    """Test _load_wiki_top_connections()."""

    def test_no_wiki_file(self, tmp_path, monkeypatch):
        """No wiki file returns default message."""
        monkeypatch.chdir(tmp_path)
        result = _load_wiki_top_connections("myrepo")
        assert result == "(kein Wiki vorhanden)"


class TestLoadAmbientSnapshot:
    """Test load_ambient_snapshot()."""

    def test_no_snapshot_returns_none(self):
        """No snapshot in DB returns None."""
        conn = _init_test_db()
        result = load_ambient_snapshot(conn, "myrepo")
        assert result is None
        conn.close()

    def test_returns_snapshot_content(self):
        """Loads and returns existing snapshot."""
        conn = _init_test_db()
        source_id = "ambient:myrepo:snapshot"

        # Insert source
        conn.execute(
            "INSERT INTO sources (source_id, source_type, repo, path, branch, \"commit\", content_hash, captured_at) VALUES (?,?,?,?,?,?,?,?)",
            (source_id, "ambient_snapshot", "myrepo", "ambient/snapshot", "local", "", "sha256:abc", "2026-01-01T00:00:00")
        )

        # Insert chunk with snapshot content
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, chunk_level, is_active, created_at) VALUES (?,?,?,?,?,?)",
            ("chk1", source_id, "Mein Snapshot-Text", "ambient_snapshot", 1, "2026-01-01T00:00:00")
        )
        conn.commit()

        result = load_ambient_snapshot(conn, "myrepo")
        assert result == "Mein Snapshot-Text"
        conn.close()


def test_generate_ambient_snapshot_returns_none_on_empty_model():
    """generate_ambient_snapshot returns None immediately when model is empty."""
    conn = _init_test_db()
    result = generate_ambient_snapshot(conn, "http://localhost:11434", "", "myrepo")
    assert result is None
    conn.close()


def test_generate_ambient_snapshot_returns_none_on_llm_error(monkeypatch):
    """generate_ambient_snapshot returns None when LLM raises."""
    conn = _init_test_db()

    def _fake_generate(*args, **kwargs):
        raise RuntimeError("Ollama not reachable")

    monkeypatch.setattr("src.analysis.analyzer._ollama_generate", _fake_generate)
    result = generate_ambient_snapshot(conn, "http://localhost:11434", "llama3", "myrepo")
    assert result is None
    conn.close()


def test_cosine_identical_vectors():
    """Cosine similarity of identical vectors is 1.0."""
    assert abs(_cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0) < 1e-9


def test_cosine_zero_vector():
    """Cosine similarity with zero-vector returns 0.0."""
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_trigger_scan_keyword_hit():
    """trigger_scan returns cluster names on keyword match."""
    idx = {"creditservice": ["CreditCluster"], "billing": ["BillingCluster"]}
    result = trigger_scan("What does CreditService do?", idx, {}, "")
    assert "CreditCluster" in result.context


def test_trigger_scan_no_hit_empty_embs():
    """trigger_scan returns empty context when no keyword hit and empty cluster_embs."""
    result = trigger_scan("completely unrelated query", {}, {}, "http://localhost:11434")
    assert result.context == ""
    assert result.trigger_ids == []


def test_trigger_scan_no_hit_below_threshold(monkeypatch):
    """trigger_scan returns empty context when embedding score is below threshold."""
    cluster_embs = {"ClusterA": [1.0, 0.0]}
    monkeypatch.setattr(
        "src.analysis.context._embed_texts",
        lambda texts, url: [[0.5, 0.866]]  # cosine ~0.5 < threshold 0.75
    )
    result = trigger_scan("some query", {}, cluster_embs, "http://localhost:11434", threshold=0.75)
    assert result.context == ""
    assert result.trigger_ids == []


def test_build_context_no_snapshot():
    """build_context returns empty string when no snapshot in DB."""
    from src.memory.ambient import build_context
    conn = _init_test_db()
    result = build_context("some task", conn, "", "myrepo")
    assert result == ""
    conn.close()


def _safe_slug(repo: str) -> str:
    import hashlib
    return hashlib.sha256(repo.encode("utf-8")).hexdigest()


def _insert_snapshot_for_repo(conn, repo: str, text: str = "Snapshot-Text", chunk_id: str = "chunk1") -> str:
    """Insert an ambient snapshot using the correct hashed source_id."""
    safe = _safe_slug(repo)
    source_id = f"ambient:{safe}:snapshot"
    conn.execute(
        "INSERT OR IGNORE INTO sources (source_id, source_type, repo, path, branch, \"commit\", content_hash, captured_at) VALUES (?,?,?,?,?,?,?,?)",
        (source_id, "ambient_snapshot", repo, "ambient/snapshot", "local", "", "sha256:abc", "2026-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO chunks (chunk_id, source_id, text, chunk_level, is_active, created_at) VALUES (?,?,?,?,?,?)",
        (chunk_id, source_id, text, "ambient_snapshot", 1, "2026-01-01T00:00:00"),
    )
    conn.commit()
    return safe


def test_build_context_snapshot_only(tmp_path, monkeypatch):
    """build_context returns snapshot section when no index/embs files exist."""
    from src.memory.ambient import build_context
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()

    conn = _init_test_db()
    _insert_snapshot_for_repo(conn, "myrepo", "Mein Snapshot-Text")

    result = build_context("some task", conn, "", "myrepo")
    assert "Projekt-Snapshot" in result
    assert "Mein Snapshot-Text" in result
    conn.close()


def _wiki_cache_slug(repo: str) -> str:
    """Return the double-hashed slug used by _safe_cache_file for wiki index files."""
    return _safe_slug(_safe_slug(repo))


def test_build_context_with_trigger_hit(tmp_path, monkeypatch):
    """build_context includes trigger context when keyword matches."""
    import json
    from src.memory.ambient import build_context
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()

    safe = _wiki_cache_slug("myrepo")
    (tmp_path / "cache" / f"{safe}_wiki_index.json").write_text(
        json.dumps({"creditservice": ["CreditCluster"]}), encoding="utf-8"
    )

    conn = _init_test_db()
    _insert_snapshot_for_repo(conn, "myrepo")

    result = build_context("What does CreditService do?", conn, "", "myrepo")
    assert "Projekt-Snapshot" in result
    assert "Trigger-Kontext" in result
    assert "CreditCluster" in result
    conn.close()


def test_parse_args_generate_ambient():
    """--generate-ambient flag is parsed correctly."""
    from src.cli import parse_args
    import sys

    # Simulate command line args
    old_argv = sys.argv
    try:
        sys.argv = ["cli.py", "--generate-ambient", "--repo", "https://github.com/test/repo"]
        args = parse_args()
        assert args.generate_ambient is True
        assert args.repo == "https://github.com/test/repo"
    finally:
        sys.argv = old_argv


def test_generate_ambient_flag_in_help(capsys):
    """--generate-ambient appears in CLI help."""
    from src.cli import parse_args
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["cli.py", "--help"]
        try:
            parse_args()
        except SystemExit:
            pass  # argparse exits after --help
        captured = capsys.readouterr()
        assert "generate-ambient" in captured.out
    finally:
        sys.argv = old_argv


# ── Task 1: Schema + Dataclasses ──────────────────────────────────────────────

def test_trigger_stats_table_exists():
    conn = _init_test_db()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "trigger_stats" in tables
    conn.close()


def test_context_feedback_log_table_exists():
    conn = _init_test_db()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "context_feedback_log" in tables
    conn.close()


def test_trigger_result_dataclass():
    from src.memory.ambient import TriggerResult
    r = TriggerResult(context="[Relevante Cluster: Foo]", trigger_ids=["keyword:foo"])
    assert r.context == "[Relevante Cluster: Foo]"
    assert r.trigger_ids == ["keyword:foo"]


def test_context_feedback_dataclass():
    from src.memory.ambient import ContextFeedback
    fb = ContextFeedback(
        trigger_ids=["keyword:foo"],
        context_text="some context",
        was_referenced=True,
        led_to_retrieval=False,
        relevance_score=0.85,
        captured_at="2026-01-01T00:00:00",
    )
    assert fb.was_referenced is True
    assert fb.relevance_score == 0.85


# ── Task 2: TriggerResult + inactive trigger skip ────────────────────────────

def test_trigger_scan_returns_trigger_result():
    from src.memory.ambient import trigger_scan, TriggerResult
    idx = {"creditservice": ["CreditCluster"]}
    result = trigger_scan("What does CreditService do?", idx, {}, "")
    assert isinstance(result, TriggerResult)
    assert "CreditCluster" in result.context
    assert any("creditservice" in tid for tid in result.trigger_ids)


def test_trigger_scan_skips_inactive_trigger():
    from src.memory.ambient import trigger_scan, TriggerResult
    conn = _init_test_db()
    # Mark keyword as inactive
    conn.execute(
        "INSERT INTO trigger_stats (trigger_id, fire_count, ref_count, is_active, last_fired) VALUES (?,?,?,?,?)",
        ("keyword:creditservice", 50, 0, 0, "2026-01-01")
    )
    conn.commit()
    idx = {"creditservice": ["CreditCluster"]}
    result = trigger_scan("What does CreditService do?", idx, {}, "", conn=conn)
    assert result.context == ""
    assert result.trigger_ids == []
    conn.close()


def test_is_trigger_active_unknown_trigger():
    from src.memory.ambient import _is_trigger_active
    conn = _init_test_db()
    assert _is_trigger_active("keyword:unknown", conn) is True
    conn.close()


def test_build_context_collects_trigger_ids(tmp_path, monkeypatch):
    import json
    from src.memory.ambient import build_context
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()
    safe = _wiki_cache_slug("myrepo")
    (tmp_path / "cache" / f"{safe}_wiki_index.json").write_text(
        json.dumps({"creditservice": ["CreditCluster"]}), encoding="utf-8"
    )
    conn = _init_test_db()
    _insert_snapshot_for_repo(conn, "myrepo", chunk_id="c1")
    out_ids: list = []
    build_context("What does CreditService do?", conn, "", "myrepo", _out_trigger_ids=out_ids)
    assert any("creditservice" in tid for tid in out_ids)
    conn.close()


# ── Task 3: compute_feedback + update_trigger_stats ──────────────────────────

def test_compute_feedback_no_ollama():
    """Empty ollama_url → was_referenced=False, relevance_score=0.0, still persisted to DB."""
    from src.memory.ambient import compute_feedback
    conn = _init_test_db()
    fb = compute_feedback("some context", "some response", ["keyword:foo"], False, conn, "")
    assert fb.was_referenced is False
    assert fb.relevance_score == 0.0
    row = conn.execute("SELECT COUNT(*) FROM context_feedback_log").fetchone()
    assert row[0] == 1
    conn.close()


def test_compute_feedback_persists_to_db():
    """After compute_feedback call, context_feedback_log has 1 row."""
    from src.memory.ambient import compute_feedback
    conn = _init_test_db()
    compute_feedback("ctx", "resp", ["k:foo"], True, conn, "")
    count = conn.execute("SELECT COUNT(*) FROM context_feedback_log").fetchone()[0]
    assert count == 1
    conn.close()


def test_update_trigger_stats_increments():
    """Two calls → fire_count=2."""
    from src.memory.ambient import update_trigger_stats
    conn = _init_test_db()
    update_trigger_stats(["keyword:foo"], True, conn)
    update_trigger_stats(["keyword:foo"], False, conn)
    row = conn.execute(
        "SELECT fire_count, ref_count FROM trigger_stats WHERE trigger_id = ?",
        ("keyword:foo",),
    ).fetchone()
    assert row[0] == 2
    assert row[1] == 1
    conn.close()


def test_update_trigger_stats_deactivates_below_threshold():
    """50 fires, 4 refs (8%) → is_active=0."""
    from src.memory.ambient import update_trigger_stats
    conn = _init_test_db()
    for i in range(50):
        update_trigger_stats(["keyword:bar"], i < 4, conn)
    row = conn.execute(
        "SELECT is_active FROM trigger_stats WHERE trigger_id = ?",
        ("keyword:bar",),
    ).fetchone()
    assert row[0] == 0
    conn.close()


def test_update_trigger_stats_keeps_active_above_threshold():
    """50 fires, 10 refs (20%) → is_active=1."""
    from src.memory.ambient import update_trigger_stats
    conn = _init_test_db()
    for i in range(50):
        update_trigger_stats(["keyword:baz"], i < 10, conn)
    row = conn.execute(
        "SELECT is_active FROM trigger_stats WHERE trigger_id = ?",
        ("keyword:baz",),
    ).fetchone()
    assert row[0] == 1
    conn.close()


# ── Task 4: Pi-Agent Integration ─────────────────────────────────────────────

def test_build_context_out_trigger_ids_empty_when_no_index(tmp_path, monkeypatch):
    """No index file → _out_trigger_ids stays empty."""
    from src.memory.ambient import build_context
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()
    conn = _init_test_db()
    out_ids: list = []
    build_context("some task", conn, "", "myrepo", _out_trigger_ids=out_ids)
    assert out_ids == []
    conn.close()


def test_build_context_out_trigger_ids_populated(tmp_path, monkeypatch):
    """Index with matching keyword → _out_trigger_ids is populated."""
    import json
    from src.memory.ambient import build_context
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()
    safe = _wiki_cache_slug("myrepo")
    (tmp_path / "cache" / f"{safe}_wiki_index.json").write_text(
        json.dumps({"creditservice": ["CreditCluster"]}), encoding="utf-8"
    )
    conn = _init_test_db()
    _insert_snapshot_for_repo(conn, "myrepo", "Snapshot", chunk_id="c1")
    out_ids: list = []
    build_context("What does CreditService do?", conn, "", "myrepo", _out_trigger_ids=out_ids)
    assert any("creditservice" in tid for tid in out_ids)
    conn.close()


# ── Retrieval integration tests ───────────────────────────────────────────────

def test_build_context_skips_retrieval_when_no_chroma(tmp_path, monkeypatch):
    """chroma_collection=None (default) → no crash, no Relevante Erinnerungen section."""
    from src.memory.ambient import build_context
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()
    conn = _init_test_db()
    _insert_snapshot_for_repo(conn, "myrepo", "Snapshot", chunk_id="snap1")
    result = build_context("analyse this", conn, "", "myrepo", chroma_collection=None)
    assert "## Projekt-Snapshot" in result
    assert "## Relevante Erinnerungen" not in result
    conn.close()


def test_build_context_skips_empty_retrieval(tmp_path, monkeypatch):
    """search() returns empty list → Relevante Erinnerungen block not added."""
    from unittest.mock import patch, MagicMock
    from src.memory.ambient import build_context
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()
    conn = _init_test_db()
    _insert_snapshot_for_repo(conn, "myrepo", "Snapshot", chunk_id="snap2")
    fake_chroma = MagicMock()
    with patch("src.memory.retrieval.search", return_value=[]):
        result = build_context("analyse this", conn, "", "myrepo", chroma_collection=fake_chroma)
    assert "## Relevante Erinnerungen" not in result
    conn.close()


def test_build_context_includes_retrieval_section(tmp_path, monkeypatch):
    """search() returns results → ## Relevante Erinnerungen section present."""
    from unittest.mock import patch, MagicMock
    from src.memory.ambient import build_context
    from src.memory.schema import RetrievalRecord
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()
    conn = _init_test_db()
    _insert_snapshot_for_repo(conn, "myrepo", "Snapshot", chunk_id="snap3")
    fake_chroma = MagicMock()
    fake_record = RetrievalRecord(
        chunk_id="c1", source_id="src1", text="Memory chunk about payments",
        score_final=0.9, category_labels=["domain"],
    )
    with patch("src.memory.retrieval.search", return_value=[fake_record]):
        result = build_context("analyse payments", conn, "", "myrepo", chroma_collection=fake_chroma)
    assert "## Relevante Erinnerungen" in result
    assert "payments" in result
    conn.close()


def test_build_context_retrieval_respects_char_budget(tmp_path, monkeypatch):
    """compress_for_prompt is called with char_budget=2400."""
    from unittest.mock import patch, MagicMock
    from src.memory.ambient import build_context
    from src.memory.schema import RetrievalRecord
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()
    conn = _init_test_db()
    _insert_snapshot_for_repo(conn, "myrepo", "Snapshot", chunk_id="snap4")
    fake_chroma = MagicMock()
    fake_record = RetrievalRecord(chunk_id="c1", source_id="s1", text="x", score_final=0.8)
    with patch("src.memory.retrieval.search", return_value=[fake_record]), \
         patch("src.memory.retrieval.compress_for_prompt", return_value="compressed") as mock_cfp:
        build_context("task", conn, "", "myrepo", chroma_collection=fake_chroma)
    mock_cfp.assert_called_once_with([fake_record], char_budget=2400)
    conn.close()
