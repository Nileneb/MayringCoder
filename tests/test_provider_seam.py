"""Guard tests for the mayring_core provider seam (#267).

WHY: ingest()/trigger_scan() etc. live in mayring_core and reach Ollama via
``mayring_core.providers.embed_texts`` / ``generate_text``. ``src/provider_setup.py``
deliberately resolves the host functions *inside the call* so the canonical
patch points stay ``src.analysis.context_rag._embed_texts`` and
``src.analysis.analyzer._ollama_generate``.

If a refactor moves that seam, ~11 ingestion tests silently stop intercepting
and fall through to the real embed retry-backoff (5×, ~41s) — which once hung
CI and crashed an xdist worker. These guards fail *loud and instantly* on drift
instead, pointing straight at the broken seam.
"""
from __future__ import annotations

from unittest.mock import patch

from mayring_core import providers


def test_embed_seam_is_patchable_at_context_rag():
    sentinel = [[42.0, 42.0]]
    with patch("src.analysis.context_rag._embed_texts", return_value=sentinel) as m:
        out = providers.embed_texts(["hello"], "http://localhost:11434")
    assert out is sentinel, (
        "providers.embed_texts no longer routes through "
        "src.analysis.context_rag._embed_texts — fix src/provider_setup.py or the "
        "test patch seams (see #267)."
    )
    m.assert_called_once()


def test_generate_seam_is_patchable_at_analyzer():
    with patch("src.analysis.analyzer._ollama_generate", return_value="SENTINEL") as m:
        out = providers.generate_text("prompt", "http://localhost:11434", "model", "label")
    assert out == "SENTINEL", (
        "providers.generate_text no longer routes through "
        "src.analysis.analyzer._ollama_generate — fix src/provider_setup.py or the "
        "test patch seams (see #267)."
    )
    m.assert_called_once()
