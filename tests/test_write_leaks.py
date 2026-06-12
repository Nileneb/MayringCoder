"""Write-leak fix (core, "nie wieder out of context"):

  _embed_one_with_retry: a single dropped embedder request used to leave the
  chunk in SQLite but never in Chroma = permanently unsearchable. Bounded retry
  rescues the transient case; permanent failure is re-raised (caller queues
  reembed-pending).
"""
import pytest

from mayring_core.memory.ingestion.core import _embed_one_with_retry


def test_embed_retry_succeeds_on_second_attempt():
    calls = {"n": 0}

    def flaky(texts, url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("three.linn.games hiccup")
        return [[0.1, 0.2, 0.3]]

    emb = _embed_one_with_retry(flaky, "text", "http://three", attempts=2)
    assert emb == [0.1, 0.2, 0.3]
    assert calls["n"] == 2


def test_embed_retry_reraises_after_exhaustion():
    def always_fail(texts, url):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        _embed_one_with_retry(always_fail, "text", "http://three", attempts=2)
