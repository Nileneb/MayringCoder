"""Tests for the corpus-worthiness filter (_is_corpus_worthy).

The whole session's lesson: a finetune corpus is only as good as it is clean. The
task_search_log must NOT collect trivial/test prompts (ping, ok, greetings) or
rows where distillation extracted nothing — else we'd finetune on junk, exactly
the failure that started this. The filter gates LOGGING only, never the search.
"""
from __future__ import annotations

import src.api.memory_service as ms


def test_real_task_is_corpus_worthy():
    assert ms._is_corpus_worthy(
        "JAAA und wie war das mit dem reranker gate wann aktiv",
        "Wann ist der Reranker Gate aktiv?") is True


def test_trivial_prompts_rejected():
    for q in ("ping", "ok", "okay", "ja", "nein", "test", "hi", "hallo",
              "danke", "weiter", "los", "go", "ok!", "ja?", "  ok  "):
        assert ms._is_corpus_worthy(q, "irgendein task") is False, q


def test_too_short_rejected():
    assert ms._is_corpus_worthy("hm", "task") is False


def test_empty_task_rejected():
    assert ms._is_corpus_worthy("eine echte lange frage hier", "") is False
    assert ms._is_corpus_worthy("eine echte lange frage hier", "   ") is False


def test_distillation_extracted_nothing_rejected():
    """task == raw means derive_task fell back to the raw prompt (no real task)."""
    q = "irgendein prompt der nicht destilliert wurde"
    assert ms._is_corpus_worthy(q, q) is False


def test_leading_ok_but_real_task_is_worthy():
    """'ok lass uns X bauen' is NOT trivial — only standalone acks are."""
    assert ms._is_corpus_worthy(
        "ok lass uns den corpus filter bauen", "Corpus-Filter bauen") is True


def test_empty_query_rejected():
    assert ms._is_corpus_worthy("", "task") is False
    assert ms._is_corpus_worthy(None, "task") is False
