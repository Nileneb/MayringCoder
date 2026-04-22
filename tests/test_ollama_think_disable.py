"""Ollama generate() body shape — thinking stays model-default, num_predict raised.

First prod smoke-test showed mayring_categorize came back with empty
labels because thinking models (qwen3-2b) stream their CoT into a
separate ``thinking`` JSON field while ``response`` stays empty until
``</think>``. At Ollama's default num_predict=128 the close-tag usually
never arrives.

The fix is NOT to silently disable thinking (that amputates the feature
for pi-task/second-opinion callers that benefit from CoT). Instead we:

  - raise num_predict to 4096 so thinking + answer both fit,
  - leave the ``think`` flag to the model's own default unless an
    individual caller opts in or out.
"""
from __future__ import annotations

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


def test_generate_omits_think_by_default():
    captured: dict = {}

    def fake_stream(method, url, json=None, **kw):
        captured["body"] = json
        return _FakeStream([b'{"response":"api","done":true}'])

    with patch("src.ollama_client.httpx.stream", side_effect=fake_stream):
        generate("http://x", "qwen3", "prompt")

    assert "think" not in captured["body"]


def test_generate_forwards_explicit_think_true():
    captured: dict = {}

    def fake_stream(method, url, json=None, **kw):
        captured["body"] = json
        return _FakeStream([b'{"response":"x","done":true}'])

    with patch("src.ollama_client.httpx.stream", side_effect=fake_stream):
        generate("http://x", "qwen3", "prompt", think=True)

    assert captured["body"]["think"] is True


def test_generate_forwards_explicit_think_false():
    captured: dict = {}

    def fake_stream(method, url, json=None, **kw):
        captured["body"] = json
        return _FakeStream([b'{"response":"x","done":true}'])

    with patch("src.ollama_client.httpx.stream", side_effect=fake_stream):
        generate("http://x", "qwen3", "prompt", think=False)

    assert captured["body"]["think"] is False


def test_generate_sets_high_num_predict_default():
    captured: dict = {}

    def fake_stream(method, url, json=None, **kw):
        captured["body"] = json
        return _FakeStream([b'{"response":"x","done":true}'])

    with patch("src.ollama_client.httpx.stream", side_effect=fake_stream):
        generate("http://x", "qwen3", "prompt")

    assert captured["body"]["options"]["num_predict"] == 4096


def test_generate_num_predict_overridable():
    captured: dict = {}

    def fake_stream(method, url, json=None, **kw):
        captured["body"] = json
        return _FakeStream([b'{"response":"x","done":true}'])

    with patch("src.ollama_client.httpx.stream", side_effect=fake_stream):
        generate("http://x", "qwen3", "prompt", num_predict=64)

    assert captured["body"]["options"]["num_predict"] == 64


def test_generate_caller_options_merge_with_num_predict():
    captured: dict = {}

    def fake_stream(method, url, json=None, **kw):
        captured["body"] = json
        return _FakeStream([b'{"response":"x","done":true}'])

    with patch("src.ollama_client.httpx.stream", side_effect=fake_stream):
        generate("http://x", "qwen3", "prompt", options={"temperature": 0.1})

    opts = captured["body"]["options"]
    assert opts["num_predict"] == 4096
    assert opts["temperature"] == 0.1


def test_generate_non_stream_also_includes_options():
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
    assert captured["body"]["options"]["num_predict"] == 4096
    assert "think" not in captured["body"]
