"""Regression: Ollama generate() must opt out of thinking by default.

First production smoke-test revealed that `mayring-qwen3:2b` (and any other
thinking model) streams 800+ tokens into a separate `thinking` JSON field
while the `response` field stays empty — the default `num_predict=128` is
usually exhausted before `</think>` closes. Result: mayring_categorize()
got empty strings, category_labels for every chunk came back []. Disabling
thinking via `think=False` in the request body returns the answer directly.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.ollama_client import generate


class _FakeStream:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield from self._lines


def test_generate_sends_think_false_by_default():
    captured: dict = {}

    def fake_stream(method, url, json=None, **kw):
        captured["body"] = json
        return _FakeStream([
            b'{"response":"api","done":true}',
        ])

    with patch("src.ollama_client.httpx.stream", side_effect=fake_stream):
        out = generate("http://x", "mayring-qwen3:2b", "prompt")

    assert out == "api"
    assert captured["body"]["think"] is False


def test_generate_forwards_explicit_think_true():
    captured: dict = {}

    def fake_stream(method, url, json=None, **kw):
        captured["body"] = json
        return _FakeStream([
            b'{"response":"done","done":true}',
        ])

    with patch("src.ollama_client.httpx.stream", side_effect=fake_stream):
        generate("http://x", "qwen3", "prompt", think=True)

    assert captured["body"]["think"] is True


def test_generate_non_stream_also_sends_think():
    captured: dict = {}

    def fake_post(url, json=None, **kw):
        captured["body"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"response": "xx"})
        return resp

    with patch("src.ollama_client.httpx.post", side_effect=fake_post):
        out = generate("http://x", "qwen3", "prompt", stream=False)

    assert out == "xx"
    assert captured["body"]["think"] is False
