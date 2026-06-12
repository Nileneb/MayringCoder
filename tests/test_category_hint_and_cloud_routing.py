"""Tests for the 2026-05-10 multi-feature changes:

1. category_hint boost in _rerank (was reviewer-finding #1 — feature
   tot ohne diesen wire-up).
2. cloud-primary routing in chat() / generate() — model-mapping correct.

Diese features wurden heute eingefügt aber von der existing test-suite
nicht abgedeckt (reviewer-finding #8).

(Die Stop-Hook _judge_chunks_with_llm-Tests sind mit dem claude-plugin nach
mayring-claude-plugin ausgezogen, #268.)
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# 1) category_hint boost
# ---------------------------------------------------------------------------

def _empty_conn():
    """In-memory SQLite mit minimal-schema für _rerank's get_feedback_score."""
    import sqlite3
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE chunk_feedback (
        chunk_id TEXT, signal TEXT, metadata TEXT, created_at TEXT
    )""")
    return c


def test_rerank_applies_category_hint_boost():
    """Chunk dessen category_labels mit hint überlappen muss höher ranken
    als chunk ohne overlap, sonst war prompt-categorize-arbeit umsonst."""
    from mayring_core.memory.retrieval import _rerank
    from mayring_core.memory.schema import Chunk

    chunk_with_overlap = Chunk(
        chunk_id="chk_with",
        source_id="s1",
        chunk_level="function",
        ordinal=0,
        text="auth flow with JWT validation",
        category_labels=["auth", "security"],
        workspace_id="default",
    )
    chunk_without = Chunk(
        chunk_id="chk_without",
        source_id="s2",
        chunk_level="function",
        ordinal=0,
        text="some other code path",
        category_labels=["utils", "logging"],
        workspace_id="default",
    )
    vector_scores = {"chk_with": 0.5, "chk_without": 0.5}
    symbolic_scores = {"chk_with": 0.5, "chk_without": 0.5}

    out_with_hint = _rerank(
        [chunk_with_overlap, chunk_without],
        vector_scores, symbolic_scores, top_k=2, conn=_empty_conn(),
        category_hint=["auth"],
    )
    out_no_hint = _rerank(
        [chunk_with_overlap, chunk_without],
        vector_scores, symbolic_scores, top_k=2, conn=_empty_conn(),
    )

    score_with_overlap_hinted = next(r for r in out_with_hint if r.chunk_id == "chk_with").score_final
    score_without_hinted = next(r for r in out_with_hint if r.chunk_id == "chk_without").score_final
    assert score_with_overlap_hinted > score_without_hinted

    score_with_baseline = next(r for r in out_no_hint if r.chunk_id == "chk_with").score_final
    score_without_baseline = next(r for r in out_no_hint if r.chunk_id == "chk_without").score_final
    assert abs(score_with_baseline - score_without_baseline) < 0.001

    # Delta ist mindestens _CAT_HINT_BOOST (0.08)
    assert score_with_overlap_hinted - score_with_baseline >= 0.07


def test_category_id_match_helper():
    from mayring_core.memory.retrieval import _category_id_match

    assert _category_id_match({1, 2}, {2, 3}) == 1.0
    assert _category_id_match({1}, {2}) == 0.0
    assert _category_id_match(set(), {1}) == 0.0
    assert _category_id_match({1}, set()) == 0.0


def test_rerank_structured_category_id_boost(tmp_path):
    """Reranker-v3: ein Kandidat mit chunk_categories-FK zur Query-Kategorie wird
    geboostet — auch wenn seine category_labels NICHT matchen (isoliert cat_match
    vom Free-String _CAT_HINT_BOOST)."""
    from mayring_core.memory.retrieval import _rerank
    from mayring_core.memory.schema import Chunk
    from mayring_core.memory.store import init_memory_db

    db = tmp_path / "m.db"
    conn = init_memory_db(db)
    now = "2026-05-24T00:00:00Z"
    conn.execute("INSERT INTO categories(name,description,status,source,"
                 "evidence_count,embedding_id) VALUES (?,?, 'active','imported',1,?)",
                 ("auth", "a", "cb:t:auth"))
    cat_id = conn.execute("SELECT id FROM categories WHERE name='auth'").fetchone()[0]
    conn.execute("INSERT INTO chunk_categories(chunk_id,category_id,codebook_version,confidence,"
                 "source) VALUES (?,?,1,0.9,'deductive')", ("chk_struct", cat_id))
    conn.commit()

    # beide haben category_labels=["other"] → Free-String-Match feuert NICHT;
    # nur chk_struct hat den chunk_categories-FK zu 'auth'.
    chk_struct = Chunk(chunk_id="chk_struct", source_id="s1", chunk_level="function",
                       ordinal=0, text="x", category_labels=["other"], workspace_id="default")
    chk_none = Chunk(chunk_id="chk_none", source_id="s2", chunk_level="function",
                     ordinal=0, text="y", category_labels=["other"], workspace_id="default")
    vs = {"chk_struct": 0.5, "chk_none": 0.5}
    ss = {"chk_struct": 0.5, "chk_none": 0.5}

    out = _rerank([chk_struct, chk_none], vs, ss, top_k=2, conn=conn, category_hint=["auth"])
    s_struct = next(r for r in out if r.chunk_id == "chk_struct").score_final
    s_none = next(r for r in out if r.chunk_id == "chk_none").score_final
    assert s_struct > s_none  # strukturierter category_id-Match hat geboostet


def test_rerank_no_boost_when_hint_empty():
    """Empty hint-liste darf keine änderung gegen None bewirken."""
    from mayring_core.memory.retrieval import _rerank
    from mayring_core.memory.schema import Chunk

    chunk = Chunk(
        chunk_id="chk_a", source_id="s", chunk_level="function", ordinal=0,
        text="x", category_labels=["auth"], workspace_id="default",
    )
    out_empty = _rerank(
        [chunk], {"chk_a": 0.5}, {"chk_a": 0.5},
        top_k=1, conn=_empty_conn(), category_hint=[],
    )
    out_none = _rerank(
        [chunk], {"chk_a": 0.5}, {"chk_a": 0.5},
        top_k=1, conn=_empty_conn(), category_hint=None,
    )
    assert abs(out_empty[0].score_final - out_none[0].score_final) < 0.001


# ---------------------------------------------------------------------------
# 2) cloud-primary routing
# ---------------------------------------------------------------------------

def test_cloud_model_mapping_translates_local_names():
    """Local models bekommen ihr cloud-äquivalent ZUVOR — sonst 404."""
    from mayring_core.ollama_client import _resolve_cloud_model

    assert _resolve_cloud_model("mistral:7b-instruct") == "gemma3:4b"
    assert _resolve_cloud_model("qwen2.5-coder:7b") == "qwen3-coder-next"
    assert _resolve_cloud_model("phi3:3.8b") == "ministral-3:3b"
    # unbekannt → default
    assert _resolve_cloud_model("totally-unknown") == "gemma3:4b"


def test_cloud_routing_off_without_api_key(monkeypatch):
    """Wenn kein OLLAMA_CLOUD_API_KEY → _should_route_cloud_primary
    returnt IMMER False, egal welcher ratio. Sicherheit gegen ratio=1.0
    bei fehlendem key (würde sonst jeder call cloud-fail+local-retry =
    double-latency)."""
    import mayring_core.ollama_client as oc
    monkeypatch.setattr(oc, "_CLOUD_API_KEY", "")
    monkeypatch.setattr(oc, "_CLOUD_PRIMARY_RATIO", 1.0)
    for _ in range(100):
        assert oc._should_route_cloud_primary() is False


def test_cloud_routing_off_at_zero_ratio(monkeypatch):
    import mayring_core.ollama_client as oc
    monkeypatch.setattr(oc, "_CLOUD_API_KEY", "fake-key")
    monkeypatch.setattr(oc, "_CLOUD_PRIMARY_RATIO", 0.0)
    for _ in range(100):
        assert oc._should_route_cloud_primary() is False
