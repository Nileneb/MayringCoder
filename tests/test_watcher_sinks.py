"""Tests für die Sink-Abstraktion im conversation_watcher.

Kernfrage nach dem User-Feedback: der Watcher muss **remote** laufen können,
damit er auf dem User-Laptop (dort wo ~/.claude/projects liegt) wirkt und
die fertige Summary über HTTP in den zentralen Memory-Server drückt.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.conversation_watcher import (
    LocalDBSink,
    RemoteHttpSink,
    WatcherSink,
    build_sink,
)


class TestBuildSink:
    def test_remote_when_api_url_given(self):
        sink = build_sink("http://mcp.linn.games", "some-jwt")
        assert isinstance(sink, RemoteHttpSink)

    def test_local_fallback_without_api_url(self):
        with patch("src.api.dependencies.get_conn", return_value=MagicMock()), \
             patch("src.api.dependencies.get_chroma", return_value=MagicMock()):
            sink = build_sink("", "")
            assert isinstance(sink, LocalDBSink)

    def test_remote_requires_jwt(self):
        with pytest.raises(ValueError, match="JWT"):
            RemoteHttpSink("http://mcp.linn.games", "")

    def test_remote_requires_url(self):
        with pytest.raises(ValueError, match="api_url"):
            RemoteHttpSink("", "jwt")


class TestRemoteHttpSink:
    @pytest.fixture
    def sink(self):
        return RemoteHttpSink("http://example.test", "test-jwt")

    def test_posts_turns_to_conversation_micro_batch(self, sink):
        fake_resp = MagicMock(status_code=200, text="{}")
        with patch.object(sink._client, "post", return_value=fake_resp) as post:
            ok = sink.ingest(
                turns=[{"role": "user", "content": "hi", "timestamp": "2026-04-22T01:00:00Z"}],
                session_id="sess-abc",
                workspace_slug="user-repo",
                ollama_url="",
                model="",
                workspace_id="system",
            )
        assert ok is True
        post.assert_called_once()
        url, kwargs = post.call_args[0][0], post.call_args[1]
        assert url.endswith("/conversation/micro-batch")
        body = kwargs["json"]
        assert body["session_id"] == "sess-abc"
        assert body["workspace_slug"] == "user-repo"
        assert len(body["turns"]) == 1
        assert body["turns"][0]["role"] == "user"

    def test_empty_turns_returns_false_without_http_call(self, sink):
        with patch.object(sink._client, "post") as post:
            ok = sink.ingest([], "s", "w", "", "", "system")
        assert ok is False
        post.assert_not_called()

    def test_dedup_against_replay(self, sink):
        fake_resp = MagicMock(status_code=200, text="{}")
        with patch.object(sink._client, "post", return_value=fake_resp) as post:
            turns = [{"role": "u", "content": "x", "timestamp": "t"}]
            sink.ingest(turns, "sid", "w", "", "", "s")
            # second call with identical turns must be a no-op
            sink.ingest(turns, "sid", "w", "", "", "s")
        assert post.call_count == 1

    def test_http_error_returns_false(self, sink):
        fake_resp = MagicMock(status_code=500, text="internal")
        with patch.object(sink._client, "post", return_value=fake_resp):
            ok = sink.ingest(
                [{"role": "u", "content": "x", "timestamp": "t"}],
                "sid", "w", "", "", "s",
            )
        assert ok is False
