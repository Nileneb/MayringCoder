"""Tests for the 2026-05-10 multi-feature changes:

1. category_hint boost in _rerank (was reviewer-finding #1 — feature
   tot ohne diesen wire-up).
2. cloud-primary routing in chat() / generate() — model-mapping correct.
3. Stop-Hook _judge_chunks_with_llm — batch rating-parse.

Diese 4 features wurden heute eingefügt aber von der existing test-suite
nicht abgedeckt (reviewer-finding #8).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent


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
    from src.memory.retrieval import _rerank
    from src.memory.schema import Chunk

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


def test_rerank_no_boost_when_hint_empty():
    """Empty hint-liste darf keine änderung gegen None bewirken."""
    from src.memory.retrieval import _rerank
    from src.memory.schema import Chunk

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
    from src.ollama_client import _resolve_cloud_model

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
    import src.ollama_client as oc
    monkeypatch.setattr(oc, "_CLOUD_API_KEY", "")
    monkeypatch.setattr(oc, "_CLOUD_PRIMARY_RATIO", 1.0)
    for _ in range(100):
        assert oc._should_route_cloud_primary() is False


def test_cloud_routing_off_at_zero_ratio(monkeypatch):
    import src.ollama_client as oc
    monkeypatch.setattr(oc, "_CLOUD_API_KEY", "fake-key")
    monkeypatch.setattr(oc, "_CLOUD_PRIMARY_RATIO", 0.0)
    for _ in range(100):
        assert oc._should_route_cloud_primary() is False


# ---------------------------------------------------------------------------
# 3) Stop-Hook _judge_chunks_with_llm
# ---------------------------------------------------------------------------

def _load_stop_hook():
    """Plugin hook nicht auf sys.path — import per file-spec."""
    spec = importlib.util.spec_from_file_location(
        "stop_hook_t", ROOT / "claude-plugin" / "hooks" / "stop_hook.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stop_hook_t"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_judge_parses_rating_response_correctly():
    sh = _load_stop_hook()

    fake_response = json.dumps({"response": "5,2,3"}).encode()

    class FakeResp:
        def __init__(self, body): self._b = body
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): pass

    chunks = [
        {"chunk_id": "a", "source_id": "s1", "text": "JWT auth logic"},
        {"chunk_id": "b", "source_id": "s2", "text": "unrelated config"},
        {"chunk_id": "c", "source_id": "s3", "text": "loose context"},
    ]
    with patch("urllib.request.urlopen", return_value=FakeResp(fake_response)):
        out = sh._judge_chunks_with_llm(
            chunks, "how does auth work?",
            "JWT is validated server-side and rejected if expired.",
        )
    assert out == {"a": "5", "b": "2", "c": "3"}


def test_judge_handles_truncated_response():
    """Wenn LLM nur 2 ratings für 3 chunks zurückgibt → 3. chunk wird
    geskipt (kein eintrag im out-dict), aber kein crash."""
    sh = _load_stop_hook()
    fake_response = json.dumps({"response": "5,4"}).encode()

    class FakeResp:
        def __init__(self, body): self._b = body
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): pass

    chunks = [
        {"chunk_id": "a", "source_id": "s1", "text": "x"},
        {"chunk_id": "b", "source_id": "s2", "text": "y"},
        {"chunk_id": "c", "source_id": "s3", "text": "z"},
    ]
    with patch("urllib.request.urlopen", return_value=FakeResp(fake_response)):
        out = sh._judge_chunks_with_llm(chunks, "q", "a")
    assert "a" in out and "b" in out
    assert "c" not in out


def test_judge_returns_none_when_no_text():
    sh = _load_stop_hook()
    chunks = [{"chunk_id": "a", "source_id": "s", "text": ""}]
    assert sh._judge_chunks_with_llm(chunks, "q", "a") is None


# ---------------------------------------------------------------------------
# 4) IGIO-intent detection + outcome-boost (user-feedback "outcome wird
#    nirgendwo genutzt")
# ---------------------------------------------------------------------------

def test_detect_igio_intent_outcome_de():
    from src.memory.retrieval import detect_igio_intent
    assert detect_igio_intent("Was kam dabei raus?") == "outcome"
    assert detect_igio_intent("Welche Konsequenzen hat das?") == "outcome"
    assert detect_igio_intent("Was war das ergebnis der refactoring?") == "outcome"
    assert detect_igio_intent("Wie war die wirkung auf die latenz?") == "outcome"


def test_detect_igio_intent_outcome_en():
    from src.memory.retrieval import detect_igio_intent
    assert detect_igio_intent("what happened after the deploy?") == "outcome"
    assert detect_igio_intent("Show me the results") == "outcome"
    assert detect_igio_intent("what was the impact?") == "outcome"


def test_detect_igio_intent_issue():
    from src.memory.retrieval import detect_igio_intent
    assert detect_igio_intent("Was ist das Problem mit der auth?") == "issue"
    assert detect_igio_intent("warum failed der test?") == "issue"
    assert detect_igio_intent("what's the root cause?") == "issue"


def test_detect_igio_intent_intervention():
    from src.memory.retrieval import detect_igio_intent
    assert detect_igio_intent("wie implementieren wir das?") == "intervention"
    assert detect_igio_intent("how do I fix this?") == "intervention"


def test_detect_igio_intent_none_for_generic_query():
    from src.memory.retrieval import detect_igio_intent
    assert detect_igio_intent("zeig mir den code") is None
    assert detect_igio_intent("xyz") is None
    assert detect_igio_intent("") is None


def test_rerank_outcome_chunk_boosted_with_outcome_intent():
    """Outcome-chunk muss höher ranken als non-outcome bei outcome-intent."""
    from src.memory.retrieval import _rerank
    from src.memory.schema import Chunk

    chunk_outcome = Chunk(
        chunk_id="chk_out", source_id="s1", chunk_level="function",
        ordinal=0, text="test passed in 2s", category_labels=["testing"],
        igio_axis="outcome", workspace_id="default",
    )
    chunk_intervention = Chunk(
        chunk_id="chk_int", source_id="s2", chunk_level="function",
        ordinal=0, text="implementation steps", category_labels=["api"],
        igio_axis="intervention", workspace_id="default",
    )
    vs = {"chk_out": 0.5, "chk_int": 0.5}
    ss = {"chk_out": 0.5, "chk_int": 0.5}

    out_with_intent = _rerank(
        [chunk_outcome, chunk_intervention], vs, ss,
        top_k=2, conn=_empty_conn(), igio_intent="outcome",
    )
    out_no_intent = _rerank(
        [chunk_outcome, chunk_intervention], vs, ss,
        top_k=2, conn=_empty_conn(),
    )

    score_out_intent = next(r for r in out_with_intent if r.chunk_id == "chk_out").score_final
    score_int_intent = next(r for r in out_with_intent if r.chunk_id == "chk_int").score_final
    assert score_out_intent > score_int_intent

    score_out_baseline = next(r for r in out_no_intent if r.chunk_id == "chk_out").score_final
    # Boost ist _IGIO_INTENT_BOOST = 0.10
    assert score_out_intent - score_out_baseline >= 0.09
