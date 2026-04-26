"""Tests for wiki paper rules and cache helpers."""
from src.memory.db_adapter import DBAdapter
from src.memory.schema import Chunk
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


from src.wiki_v2.paper_rules import detect_citations, detect_keyword_overlap


def _chunk(chunk_id, source_id, text, workspace_id="ws"):
    c = Chunk(chunk_id=chunk_id, source_id=source_id, text=text,
              text_hash="h", dedup_key="d", created_at="2026-01-01", workspace_id=workspace_id)
    return c


def test_citation_two_papers_sharing_author_year():
    c1 = _chunk("c1", "paper_a", "As shown in [Mayring 2000] this method")
    c2 = _chunk("c2", "paper_b", "[Mayring 2000] qualitative content analysis")
    edges = detect_citations([c1, c2], workspace_id="ws", repo_slug="r")
    assert any(e.type == "citation" for e in edges)
    assert any(e.source in ("paper_a", "paper_b") for e in edges)


def test_keyword_overlap_finds_shared_labels():
    c1 = _chunk("c1", "paper_a", "text")
    c1.category_labels = ["qualitative_analysis", "coding"]
    c2 = _chunk("c2", "paper_b", "text2")
    c2.category_labels = ["qualitative_analysis", "grounded_theory"]
    edges = detect_keyword_overlap([c1, c2], workspace_id="ws", repo_slug="r")
    assert any(e.type == "keyword_cooccurrence" for e in edges)
    assert any(e.source == "paper_a" and e.target == "paper_b" for e in edges)


def test_keyword_overlap_weight_is_0_5():
    c1 = _chunk("c1", "src_a", "x")
    c1.category_labels = ["research_method"]
    c2 = _chunk("c2", "src_b", "y")
    c2.category_labels = ["research_method"]
    edges = detect_keyword_overlap([c1, c2], workspace_id="ws", repo_slug="r")
    assert all(abs(e.weight - 0.5) < 0.001 for e in edges)


def test_detect_from_papers_no_ollama_skips_llm_rules():
    """Without ollama_url, only citation + keyword rules run — no crash."""
    from src.wiki_v2.paper_rules import detect_from_papers
    c1 = _chunk("c1", "src_a", "text with [Smith 2020]")
    c1.category_labels = ["research"]
    c2 = _chunk("c2", "src_b", "text with [Smith 2020]")
    c2.category_labels = ["research"]
    edges = detect_from_papers([c1, c2], conn=None, ollama_url="", model="", workspace_id="ws", repo_slug="r")
    assert isinstance(edges, list)
    assert len(edges) > 0
