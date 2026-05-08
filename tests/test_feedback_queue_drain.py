"""Issue #138 fix: SessionStart drains the feedback queue (Solution C)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def session_start():
    repo_root = Path(__file__).resolve().parent.parent
    hook_path = repo_root / "claude-plugin" / "hooks" / "session_start.py"
    spec = importlib.util.spec_from_file_location("session_start_module", hook_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["session_start_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_drain_skips_when_no_queue_file(session_start, tmp_path, monkeypatch):
    """No file → no-op, no error."""
    monkeypatch.setattr(session_start, "FEEDBACK_QUEUE", str(tmp_path / "nope.jsonl"))
    session_start._drain_feedback_queue()  # must not raise


def test_drain_removes_empty_queue_file(session_start, tmp_path, monkeypatch):
    queue = tmp_path / "fb.jsonl"
    queue.write_text("")
    monkeypatch.setattr(session_start, "FEEDBACK_QUEUE", str(queue))
    monkeypatch.setattr(session_start, "JWT_FILE", str(tmp_path / "missing.jwt"))
    session_start._drain_feedback_queue()
    assert not queue.exists()


def test_drain_skips_when_no_jwt(session_start, tmp_path, monkeypatch):
    queue = tmp_path / "fb.jsonl"
    queue.write_text(json.dumps({"chunk_id": "c", "signal": "positive"}) + "\n")
    monkeypatch.setattr(session_start, "FEEDBACK_QUEUE", str(queue))
    monkeypatch.setattr(session_start, "JWT_FILE", str(tmp_path / "missing.jwt"))
    session_start._drain_feedback_queue()
    # Queue must remain — without JWT we can't replay
    assert queue.exists()
    assert queue.read_text().strip() != ""


def test_drain_replays_and_clears_on_success(session_start, tmp_path, monkeypatch):
    queue = tmp_path / "fb.jsonl"
    jwt = tmp_path / "hook.jwt"
    queue.write_text(
        json.dumps({"chunk_id": "chk_a", "signal": "positive"}) + "\n"
        + json.dumps({"chunk_id": "chk_b", "signal": "negative"}) + "\n"
    )
    jwt.write_text("fake.token.value")
    monkeypatch.setattr(session_start, "FEEDBACK_QUEUE", str(queue))
    monkeypatch.setattr(session_start, "JWT_FILE", str(jwt))

    posted: list[dict] = []

    class _FakeResponse:
        def __init__(self, status: int = 200):
            self.status = status
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        posted.append(body)
        return _FakeResponse()

    monkeypatch.setattr(session_start.urllib.request, "urlopen", fake_urlopen)
    session_start._drain_feedback_queue()

    assert len(posted) == 2
    assert {p["chunk_id"] for p in posted} == {"chk_a", "chk_b"}
    assert not queue.exists(), "queue should be removed when fully drained"


def test_drain_keeps_5xx_failures_for_retry(session_start, tmp_path, monkeypatch):
    """A 5xx response means server problem — entry stays for next session."""
    queue = tmp_path / "fb.jsonl"
    jwt = tmp_path / "hook.jwt"
    queue.write_text(json.dumps({"chunk_id": "chk_x", "signal": "positive"}) + "\n")
    jwt.write_text("fake.token")
    monkeypatch.setattr(session_start, "FEEDBACK_QUEUE", str(queue))
    monkeypatch.setattr(session_start, "JWT_FILE", str(jwt))

    import urllib.error
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "down", {}, None)
    monkeypatch.setattr(session_start.urllib.request, "urlopen", fake_urlopen)

    session_start._drain_feedback_queue()
    assert queue.exists()
    assert "chk_x" in queue.read_text()


def test_drain_drops_4xx_failures(session_start, tmp_path, monkeypatch):
    """Client error (e.g. chunk gone) — retrying won't help; entry dropped."""
    queue = tmp_path / "fb.jsonl"
    jwt = tmp_path / "hook.jwt"
    queue.write_text(json.dumps({"chunk_id": "chk_gone", "signal": "positive"}) + "\n")
    jwt.write_text("fake.token")
    monkeypatch.setattr(session_start, "FEEDBACK_QUEUE", str(queue))
    monkeypatch.setattr(session_start, "JWT_FILE", str(jwt))

    import urllib.error
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "gone", {}, None)
    monkeypatch.setattr(session_start.urllib.request, "urlopen", fake_urlopen)

    session_start._drain_feedback_queue()
    assert not queue.exists()


def test_drain_skips_corrupted_lines(session_start, tmp_path, monkeypatch):
    queue = tmp_path / "fb.jsonl"
    jwt = tmp_path / "hook.jwt"
    queue.write_text(
        json.dumps({"chunk_id": "ok", "signal": "positive"}) + "\n"
        + "NOT A JSON LINE\n"
    )
    jwt.write_text("fake.token")
    monkeypatch.setattr(session_start, "FEEDBACK_QUEUE", str(queue))
    monkeypatch.setattr(session_start, "JWT_FILE", str(jwt))

    posted: list[str] = []
    class _R:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(
        session_start.urllib.request,
        "urlopen",
        lambda req, timeout=None: posted.append(json.loads(req.data)["chunk_id"]) or _R(),
    )

    session_start._drain_feedback_queue()
    # Only the well-formed line was POSTed; bad line was silently dropped
    assert posted == ["ok"]
