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
    """Build a httpx.post mock returning {"response": json.dumps(payload)}."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": json.dumps(payload)}
    return resp


# ── pi_categorize ────────────────────────────────────────────────

def test_pi_categorize_returns_labels(tools):
    fn = tools["pi_categorize"]
    with patch("httpx.post", return_value=_mock_ollama_response(
        {"labels": [{"label": "auth", "confidence": 0.9}, {"label": "middleware", "confidence": 0.6}]}
    )), patch("src.api.mcp_agent_tools._model", return_value="mistral:7b-instruct"):
        result = fn(text="def validate_jwt(token): ...", codebook=["auth", "middleware", "data_access"])
    assert "labels" in result
    assert result["labels"][0]["label"] == "auth"
    assert result["mode"] == "inductive"


def test_pi_categorize_rejects_short_text(tools):
    fn = tools["pi_categorize"]
    result = fn(text="hi")
    assert "error" in result
    assert "too short" in result["error"]


def test_pi_categorize_caps_max_labels(tools):
    fn = tools["pi_categorize"]
    with patch("httpx.post", return_value=_mock_ollama_response(
        {"labels": [{"label": f"l{i}", "confidence": 0.5} for i in range(10)]}
    )), patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some longer text content here", max_labels=3)
    assert len(result["labels"]) == 3


def test_pi_categorize_handles_json_parse_fail(tools):
    fn = tools["pi_categorize"]
    bad_resp = MagicMock()
    bad_resp.raise_for_status = MagicMock()
    bad_resp.json.return_value = {"response": "not valid json {{{"}
    with patch("httpx.post", return_value=bad_resp), \
         patch("src.api.mcp_agent_tools._model", return_value="m"):
        result = fn(text="some text content")
    assert "error" in result
    assert "JSON parse fail" in result["error"]


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
