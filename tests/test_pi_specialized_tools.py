"""Tests for the specialized Pi-Sub-Tools (#211).

pi_mark_categories, pi_category_evidence, pi_summarize_for_memory DELETED —
callers removed (tools collapsed into the ONE Mayring method).

Remaining: pi_categorize (now uses reduce_text_server / mayring_reduce),
pi_judge_relevance (unchanged).
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
    with patch("src.api.mcp_agent_tools._enforce_tenant", return_value="bene"), \
         patch("src.api.mcp_agent_tools._effective_workspace_id", return_value="bene"), \
         patch("src.api.mcp_agent_tools._current_raw_jwt", return_value="test-jwt"):
        register_agent_tools(mcp)
        yield mcp.tools


# ── pi_categorize (ONE method: reduce_text_server → ReduceResult) ──────────

def _fake_reduce_result(label="auth", match="deductive"):
    from unittest.mock import MagicMock
    cand = MagicMock()
    cand.label = label
    cand.match = match
    res = MagicMock()
    res.candidates = [cand]
    res.paraphrase = "validates a JWT token"
    res.generalization = "security"
    return res


def test_pi_categorize_returns_label_and_match(tools):
    fn = tools["pi_categorize"]
    with patch("src.api.routes.codebooks.reduce_text_server",
               return_value=_fake_reduce_result("auth", "deductive")), \
         patch("src.api.mcp_agent_tools._model", return_value="mistral:7b-instruct"):
        result = fn(text="def validate_jwt(token): ...", task="authentication")
    assert result["label"] == "auth"
    assert result["match"] == "deductive"
    assert result["paraphrase"] == "validates a JWT token"
    assert result["generalize"] == "security"


def test_pi_categorize_inductive_match(tools):
    fn = tools["pi_categorize"]
    with patch("src.api.routes.codebooks.reduce_text_server",
               return_value=_fake_reduce_result("new_topic", "inductive")), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some longer text content here about a novel concept")
    assert result["match"] == "inductive"
    assert result["label"] == "new_topic"


def test_pi_categorize_rejects_short_text(tools):
    fn = tools["pi_categorize"]
    result = fn(text="hi")
    assert "error" in result
    assert "too short" in result["error"]


def test_pi_categorize_handles_error(tools):
    fn = tools["pi_categorize"]
    with patch("src.api.routes.codebooks.reduce_text_server",
               side_effect=Exception("connection refused")), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some text content here")
    assert "error" in result
    assert "connection refused" in result["error"]


def test_pi_categorize_empty_candidates(tools):
    """reduce_text_server returns ReduceResult with no candidates → label/match empty."""
    fn = tools["pi_categorize"]
    res = MagicMock()
    res.candidates = []
    res.paraphrase = "p"
    res.generalization = "g"
    with patch("src.api.routes.codebooks.reduce_text_server", return_value=res), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some text content here")
    assert result["label"] == ""
    assert result["match"] == ""


def test_pi_categorize_deleted_tools_absent(tools):
    """Confirm the deleted tools are gone from the registry."""
    assert "pi_mark_categories" not in tools
    assert "pi_category_evidence" not in tools
    assert "pi_summarize_for_memory" not in tools


# ── pi_judge_relevance ────────────────────────────────────────────

def _mock_ollama_response(payload: dict):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"content": json.dumps(payload)}
    return resp


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
        {"scores": {"chk_a": 1.5, "chk_b": -0.3}}
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
    assert "chk_20" not in captured["prompt"]
    assert "chk_19" in captured["prompt"]


def test_pi_judge_relevance_routes_through_pi_run_json(tools):
    captured: dict = {}
    def post(url, **kw):
        captured["url"] = url
        captured["json"] = kw.get("json", {})
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"content": '{"scores":{}}'}
        return resp
    with patch("httpx.post", side_effect=post), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        tools["pi_judge_relevance"](query="q", chunks=[{"chunk_id": "c", "text": "t"}])
    assert captured["url"].endswith("/pi/run")
    assert captured["json"]["kind"] == "judge-relevance"
    assert captured["json"]["response_format"] == "json"


# ── lenient JSON parse ────────────────────────────────────────────

def test_loads_json_lenient_handles_raw_fenced_and_prose():
    from src.api.mcp_agent_tools import _loads_json_lenient
    import json as _json
    assert _loads_json_lenient('{"ok": true}') == {"ok": True}
    assert _loads_json_lenient('```json\n{"ok": true}\n```') == {"ok": True}
    assert _loads_json_lenient('here: {"a": 1} done') == {"a": 1}
    with pytest.raises(_json.JSONDecodeError):
        _loads_json_lenient("not json at all")


def test_pi_judge_relevance_parses_fenced_cloud_output(tools):
    fenced = MagicMock()
    fenced.raise_for_status = MagicMock()
    fenced.json.return_value = {"content": '```json\n{"scores": {"chk_a": 0.8}}\n```'}
    with patch("httpx.post", return_value=fenced), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = tools["pi_judge_relevance"](query="q", chunks=[{"chunk_id": "chk_a", "text": "t"}])
    assert result["scores"]["chk_a"] == 0.8
