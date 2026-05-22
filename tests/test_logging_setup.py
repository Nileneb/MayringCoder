"""Tests für JsonFormatter + MemoryIngestHandler."""
from __future__ import annotations

import json
import logging
import threading
import time
from unittest.mock import patch

from mayring_core.logging_setup import (
    JsonFormatter, MemoryIngestHandler, SEVERE_LEVELS,
)


def test_json_formatter_basic_fields():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="src.foo", level=logging.ERROR, pathname="x.py", lineno=1,
        msg="boom %s", args=("y",), exc_info=None,
    )
    out = fmt.format(record)
    payload = json.loads(out)
    assert payload["level"] == "ERROR"
    assert payload["logger"] == "src.foo"
    assert payload["msg"] == "boom y"
    assert "ts" in payload


def test_json_formatter_includes_extra():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="src.foo", level=logging.WARNING, pathname="x.py", lineno=1,
        msg="m", args=(), exc_info=None,
    )
    record.workspace_id = "bene"
    record.job_id = "abc"
    payload = json.loads(fmt.format(record))
    assert payload["workspace_id"] == "bene"
    assert payload["job_id"] == "abc"


def test_json_formatter_skips_non_json_serializable():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="src.foo", level=logging.WARNING, pathname="x.py", lineno=1,
        msg="m", args=(), exc_info=None,
    )
    class _NotJsonable:
        def __repr__(self):
            return "<obj>"
    record.weird = _NotJsonable()
    payload = json.loads(fmt.format(record))
    assert payload["weird"] == "<obj>"


def test_severe_levels_set_contains_expected():
    assert "ERROR" in SEVERE_LEVELS
    assert "CRITICAL" in SEVERE_LEVELS
    assert "INFO" not in SEVERE_LEVELS
    assert "DEBUG" not in SEVERE_LEVELS


def test_memory_ingest_handler_buffers_low_level():
    """INFO/WARNING ohne severity-trigger → buffered, kein sofort-POST."""
    posts = []

    def fake_urlopen(req, timeout=None):
        posts.append(json.loads(req.data.decode()))
        class R:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return R()

    with patch("mayring_core.logging_setup.urllib.request.urlopen", fake_urlopen):
        h = MemoryIngestHandler(
            api_url="http://x", token="t", service="test",
            batch_size=100,  # hoch → kein flush bei < 100 events
        )
        for _ in range(5):
            r = logging.LogRecord(
                name="src.foo", level=logging.WARNING,
                pathname="x.py", lineno=1, msg="m", args=(),
                exc_info=None,
            )
            h.emit(r)
        time.sleep(0.3)  # worker check-interval
        assert len(posts) == 0  # buffer not full → no flush
        h._stop.set()


def test_memory_ingest_handler_flushes_on_severity():
    posts = []

    def fake_urlopen(req, timeout=None):
        posts.append(json.loads(req.data.decode()))
        class R:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return R()

    with patch("mayring_core.logging_setup.urllib.request.urlopen", fake_urlopen):
        h = MemoryIngestHandler(
            api_url="http://x", token="t", service="test", batch_size=100,
        )
        r = logging.LogRecord(
            name="src.foo", level=logging.ERROR, pathname="x.py", lineno=1,
            msg="real boom", args=(), exc_info=None,
        )
        h.emit(r)
        # Wake-event sollte den worker sofort batchen lassen.
        for _ in range(20):
            time.sleep(0.1)
            if posts:
                break
        assert len(posts) >= 1
        ev = posts[0]["events"][0]
        assert ev["level"] == "ERROR"
        assert ev["msg"] == "real boom"
        h._stop.set()


def test_memory_ingest_handler_persists_on_endpoint_failure(tmp_path,
                                                              monkeypatch):
    """Bei /memory/log-event-Fail → events landen in _pending.jsonl
    statt verloren zu gehen."""
    monkeypatch.setenv("MAYRING_LOG_FILE", str(tmp_path / "logs" / "x.json"))
    posts = []

    def failing_urlopen(req, timeout=None):
        posts.append(1)
        raise OSError("simulated network fail")

    with patch("mayring_core.logging_setup.urllib.request.urlopen", failing_urlopen):
        h = MemoryIngestHandler(
            api_url="http://x", token="t", service="test", batch_size=100,
        )
        r = logging.LogRecord(
            name="src.foo", level=logging.ERROR, pathname="x.py", lineno=1,
            msg="boom", args=(), exc_info=None,
        )
        h.emit(r)
        for _ in range(15):
            time.sleep(0.1)
            if posts:
                break
        h._stop.set()
        time.sleep(0.2)
    pending = tmp_path / "logs" / "_pending.jsonl"
    assert pending.exists()
    content = pending.read_text()
    assert "boom" in content
