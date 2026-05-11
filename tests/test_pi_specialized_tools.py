"""Tests for the specialized Pi-Sub-Tools (#211).

These wrap the local Pi-Agent / Ollama backend with focused schemas:
pi_categorize, pi_judge_relevance, pi_summarize_for_memory.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class _FakeMcp:
    """Minimal mock that captures the tools register_agent_tools registers."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture
def tools():
    from src.api.mcp_agent_tools import register_agent_tools
    mcp = _FakeMcp()
    # Patch tenant helpers so they don't need a real JWT context.
    with patch("src.api.mcp_agent_tools._enforce_tenant", return_value="bene"), \
         patch("src.api.mcp_agent_tools._effective_workspace_id", return_value="bene"):
        register_agent_tools(mcp)
        yield mcp.tools


def _mock_ollama_response(payload: dict):
    """Build a httpx.post mock returning {"response": json.dumps(payload)}.

    Used by pi_judge_relevance + pi_summarize_for_memory (both JSON-mode).
    pi_categorize uses _mock_ollama_text() because the canonical mayring
    prompts return a comma-separated list, not JSON.
    """
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": json.dumps(payload)}
    return resp


def _mock_ollama_text(text: str):
    """httpx.post mock returning a raw text {"response": "..."} — for pi_categorize."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": text}
    return resp


# ── pi_categorize (canonical mayring prompts, comma-separated output) ─────

def test_pi_categorize_returns_labels(tools):
    fn = tools["pi_categorize"]
    with patch("httpx.post", return_value=_mock_ollama_text("auth, middleware")), \
         patch("src.api.mcp_agent_tools._model", return_value="mistral:7b-instruct"):
        result = fn(text="def validate_jwt(token): ...", codebook=["auth", "middleware", "data_access"], mode="deductive")
    assert result["labels"] == ["auth", "middleware"]
    assert result["mode"] == "deductive"


def test_pi_categorize_defaults_to_hybrid_mode(tools):
    fn = tools["pi_categorize"]
    with patch("httpx.post", return_value=_mock_ollama_text("api, error_handling")), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some longer text content here")
    assert result["mode"] == "hybrid"


def test_pi_categorize_invalid_mode_falls_back_to_hybrid(tools):
    fn = tools["pi_categorize"]
    with patch("httpx.post", return_value=_mock_ollama_text("x, y")), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some text content", mode="nonsense")
    assert result["mode"] == "hybrid"


def test_pi_categorize_rejects_short_text(tools):
    fn = tools["pi_categorize"]
    result = fn(text="hi")
    assert "error" in result
    assert "too short" in result["error"]


def test_pi_categorize_caps_max_labels(tools):
    fn = tools["pi_categorize"]
    with patch("httpx.post", return_value=_mock_ollama_text("l0, l1, l2, l3, l4, l5, l6")), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some longer text content here", max_labels=3)
    assert len(result["labels"]) == 3
    assert result["labels"] == ["l0", "l1", "l2"]


def test_pi_categorize_lowercases_and_strips_labels(tools):
    fn = tools["pi_categorize"]
    with patch("httpx.post", return_value=_mock_ollama_text("  Auth ,  Data-Access  ")), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some longer text content here")
    assert result["labels"] == ["auth", "data-access"]


def test_pi_categorize_handles_ollama_error(tools):
    fn = tools["pi_categorize"]
    with patch("httpx.post", side_effect=Exception("connection refused")), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some text content")
    assert "error" in result
    assert "connection refused" in result["error"]


# ── pi_judge_relevance ───────────────────────────────────────────

def test_pi_judge_relevance_scores_chunks(tools):
    fn = tools["pi_judge_relevance"]
    with patch("httpx.post", return_value=_mock_ollama_response(
        {"scores": {"chk_a": 0.9, "chk_b": 0.2}}
    )), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(query="how to validate JWT", chunks=[
            {"chunk_id": "chk_a", "text": "JWT validation code here"},
            {"chunk_id": "chk_b", "text": "unrelated UI styling"},
        ])
    assert result["scores"]["chk_a"] == 0.9
    assert result["scores"]["chk_b"] == 0.2


def test_pi_judge_relevance_clamps_out_of_range(tools):
    fn = tools["pi_judge_relevance"]
    with patch("httpx.post", return_value=_mock_ollama_response(
        {"scores": {"chk_a": 1.5, "chk_b": -0.3}}  # out of [0,1]
    )), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(query="q", chunks=[
            {"chunk_id": "chk_a", "text": "text a"},
            {"chunk_id": "chk_b", "text": "text b"},
        ])
    assert result["scores"]["chk_a"] == 1.0
    assert result["scores"]["chk_b"] == 0.0


def test_pi_judge_relevance_empty_chunks(tools):
    fn = tools["pi_judge_relevance"]
    result = fn(query="q", chunks=[])
    assert "error" in result


def test_pi_judge_relevance_caps_at_20_chunks(tools):
    fn = tools["pi_judge_relevance"]
    big = [{"chunk_id": f"chk_{i}", "text": f"text {i}"} for i in range(30)]
    captured = {}
    def capture_post(url, **kw):
        captured["prompt"] = kw["json"]["prompt"]
        return _mock_ollama_response({"scores": {f"chk_{i}": 0.5 for i in range(20)}})
    with patch("httpx.post", side_effect=capture_post), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(query="q", chunks=big)
    # chk_20..chk_29 should not be in the prompt (capped at 20)
    assert "chk_20" not in captured["prompt"]
    assert "chk_19" in captured["prompt"]


# ── pi_summarize_for_memory ──────────────────────────────────────

def test_pi_summarize_returns_three_reductions(tools):
    fn = tools["pi_summarize_for_memory"]
    with patch("httpx.post", return_value=_mock_ollama_response({
        "paraphrase": "Fixed the deploy by adding postgresql-client.",
        "generalize": "debug",
        "reduce": "Deploy crashed on missing psql; fixed via dockerfile.",
        "suggested_source_id": "debug:2026-05-11-deploy-psql-fix",
    })), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(content="Today the deploy crashed because the production image lacked postgresql-client. We added it to the dockerfile production stage. " * 3)
    assert result["generalize"] == "debug"
    assert result["suggested_source_id"].startswith("debug:")
    assert "paraphrase" in result


def test_pi_summarize_rejects_short_content(tools):
    fn = tools["pi_summarize_for_memory"]
    result = fn(content="too short")
    assert "error" in result
    assert "too short" in result["error"]


def test_pi_summarize_handles_ollama_error(tools):
    fn = tools["pi_summarize_for_memory"]
    with patch("httpx.post", side_effect=Exception("connection refused")), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(content="a" * 50)
    assert "error" in result
    assert "connection refused" in result["error"]


# ── pi_mark_categories (textmarker — span-level evidence) ────────

def test_pi_mark_categories_returns_span_markings(tools):
    fn = tools["pi_mark_categories"]
    text = "The study used a randomized controlled design. Participants were nurses in ICU settings."
    with patch("httpx.post", return_value=_mock_ollama_response({
        "markings": [
            {"excerpt": "randomized controlled design", "category": "study-design", "reasoning": "names the trial design"},
            {"excerpt": "nurses in ICU settings", "category": "population", "reasoning": "describes the sample"},
        ],
    })), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text=text, task="welches studiendesign + welche population?")
    assert len(result["markings"]) == 2
    m0 = result["markings"][0]
    assert m0["category"] == "study-design"
    assert m0["span"] == [text.find("randomized controlled design"),
                          text.find("randomized controlled design") + len("randomized controlled design")]
    assert "trial design" in m0["reasoning"]


def test_pi_mark_categories_span_none_when_excerpt_not_verbatim(tools):
    """If the LLM paraphrased the excerpt instead of quoting verbatim, span=None."""
    fn = tools["pi_mark_categories"]
    with patch("httpx.post", return_value=_mock_ollama_response({
        "markings": [{"excerpt": "this text does not appear", "category": "x", "reasoning": "r"}],
    })), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some actual chunk content here", task="t")
    assert result["markings"][0]["span"] is None


def test_pi_mark_categories_drops_markings_without_category(tools):
    fn = tools["pi_mark_categories"]
    with patch("httpx.post", return_value=_mock_ollama_response({
        "markings": [
            {"excerpt": "valid", "category": "good-cat", "reasoning": "r"},
            {"excerpt": "no category", "category": "", "reasoning": "r"},
            {"excerpt": "", "category": "no-excerpt", "reasoning": "r"},
        ],
    })), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="valid no category content", task="t")
    assert len(result["markings"]) == 1
    assert result["markings"][0]["category"] == "good-cat"


def test_pi_mark_categories_empty_markings_when_off_topic(tools):
    fn = tools["pi_mark_categories"]
    with patch("httpx.post", return_value=_mock_ollama_response({"markings": []})), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="unrelated content here entirely", task="something else")
    assert result["markings"] == []


def test_pi_mark_categories_rejects_short_text(tools):
    fn = tools["pi_mark_categories"]
    result = fn(text="hi", task="t")
    assert "error" in result


def test_pi_mark_categories_handles_json_parse_fail(tools):
    fn = tools["pi_mark_categories"]
    bad = MagicMock()
    bad.raise_for_status = MagicMock()
    bad.json.return_value = {"response": "not json {{{"}
    with patch("httpx.post", return_value=bad), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some chunk content", task="t")
    assert "error" in result and "JSON parse fail" in result["error"]


def test_pi_categorize_accepts_task_arg(tools):
    """pi_categorize now takes a `task` — it should not error and the call goes through."""
    fn = tools["pi_categorize"]
    with patch("httpx.post", return_value=_mock_ollama_text("study-design, population")), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some chunk content here about a trial", task="welches studiendesign?", mode="inductive")
    assert result["labels"] == ["study-design", "population"]


def test_pi_mark_categories_persists_to_wiki(tools, tmp_path, monkeypatch):
    """persist=True + source_id → markings land in wiki_category_evidence."""
    # Point MEMORY_DB_PATH to a tmp dir so wiki_v2.db is created there.
    import src.memory.store as _store
    monkeypatch.setattr(_store, "MEMORY_DB_PATH", tmp_path / "memory.db")

    fn = tools["pi_mark_categories"]
    text = "def validate_jwt(token): return decode(token)"
    with patch("httpx.post", return_value=_mock_ollama_response({
        "markings": [{"excerpt": "def validate_jwt", "category": "auth", "reasoning": "validates a jwt"}],
    })), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text=text, task="wie wird auth gemacht?", source_id="src:foo.py",
                    chunk_id="chk_1", persist=True)
    assert result["persisted"] == 1

    # Read it back via get_category_evidence
    from src.wiki_v2.store import init_wiki_db, get_category_evidence
    wdb = init_wiki_db(tmp_path / "wiki_v2.db")
    try:
        ev = get_category_evidence(wdb, result["workspace_id"], category="auth")
        assert len(ev) == 1
        assert ev[0]["source_id"] == "src:foo.py"
        assert ev[0]["task"] == "wie wird auth gemacht?"
    finally:
        wdb.close()


def test_pi_mark_categories_no_persist_when_persist_false(tools, tmp_path, monkeypatch):
    import src.memory.store as _store
    monkeypatch.setattr(_store, "MEMORY_DB_PATH", tmp_path / "memory.db")
    fn = tools["pi_mark_categories"]
    with patch("httpx.post", return_value=_mock_ollama_response({
        "markings": [{"excerpt": "abc", "category": "x", "reasoning": "r"}],
    })), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="abc content here", task="t", source_id="src:y", persist=False)
    assert result["persisted"] == 0


def test_pi_category_evidence_reads_persisted(tools, tmp_path, monkeypatch):
    import src.memory.store as _store
    monkeypatch.setattr(_store, "MEMORY_DB_PATH", tmp_path / "memory.db")
    from src.wiki_v2.store import init_wiki_db, persist_category_evidence
    wdb = init_wiki_db(tmp_path / "wiki_v2.db")
    persist_category_evidence(wdb, "bene", "src:z", [
        {"span": [0, 3], "excerpt": "abc", "category": "demo-cat", "reasoning": "r"},
    ], task="demo task")
    wdb.close()

    fn = tools["pi_category_evidence"]
    with patch("src.api.mcp_agent_tools._enforce_tenant", return_value="bene"), \
         patch("src.api.mcp_agent_tools._effective_workspace_id", return_value="bene"):
        result = fn(category="demo-cat")
    assert result["count"] == 1
    assert result["evidence"][0]["category"] == "demo-cat"
    assert result["evidence"][0]["task"] == "demo task"
