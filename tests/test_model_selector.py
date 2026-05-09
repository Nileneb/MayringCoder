"""Regression tests for model_selector — specifically the embedding-filter
that prevents the ambient-snapshot bug (production incident 2026-05-09:
ambient-job picked `all-minilm:l6-v2` as text-gen model → 400 Bad Request
from /api/generate → 0 snapshots in 5 days, silent failure)."""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from src.model_selector import fetch_ollama_models, resolve_model


def _mock_tags_response(model_names: list[str]):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": n} for n in model_names]}

    return _Resp()


def test_filters_minilm_and_bge_and_nomic_as_embedding_only(monkeypatch):
    """Real production tags returned `all-minilm:l6-v2` which has neither
    'embed' nor 'embedding' in its name — slipped through the old filter
    and crashed /api/generate with 400."""
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **kw: _mock_tags_response([
            "all-minilm:l6-v2",            # embedding model masquerading as gen
            "bge-small-en",                # BGE embedding family
            "gte-large",                   # GTE embedding family
            "nomic-embed-text",            # already-explicit-embed
            "mistral:7b-instruct",         # genuine text-gen
            "qwen2.5-coder:7b",            # genuine text-gen
        ]),
    )
    out = fetch_ollama_models("http://fake")
    assert "mistral:7b-instruct" in out
    assert "qwen2.5-coder:7b" in out
    assert "all-minilm:l6-v2" not in out
    assert "bge-small-en" not in out
    assert "gte-large" not in out
    assert "nomic-embed-text" not in out


def test_resolve_model_picks_text_gen_not_embedding(monkeypatch):
    """End-to-end: resolve_model() with no CLI/env override picks the FIRST
    text-gen model from /api/tags, not the first overall (which used to be
    the embedding model alphabetically)."""
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **kw: _mock_tags_response([
            "all-minilm:l6-v2",
            "mistral:7b-instruct",
        ]),
    )
    monkeypatch.delenv("MAYRING_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    chosen = resolve_model("http://fake", None, None)
    assert chosen == "mistral:7b-instruct"
