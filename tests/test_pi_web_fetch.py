"""Tests for the Pi-Agent web_fetch read-tool (#211)."""
import io
from unittest.mock import patch

import pytest

from mayring_pi_agent.pi import (
    _domain_allowed,
    _execute_web_fetch,
    _WEB_FETCH_CACHE,
    _WEB_FETCH_MAX_BYTES,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    _WEB_FETCH_CACHE.clear()
    yield
    _WEB_FETCH_CACHE.clear()


def test_domain_allowed_matches_host_and_subdomains():
    allow = ["github.com", "docs.python.org"]
    assert _domain_allowed("https://github.com/x", allow)
    assert _domain_allowed("https://api.github.com/x", allow)      # subdomain
    assert _domain_allowed("https://docs.python.org/3/", allow)
    assert not _domain_allowed("https://evil.com/github.com", allow)
    assert not _domain_allowed("https://notgithub.com/x", allow)   # no suffix match


def test_rejects_non_http_scheme(monkeypatch):
    monkeypatch.setenv("PI_WEB_FETCH_ALLOWLIST", "github.com")
    assert "nur http(s)" in _execute_web_fetch("ftp://github.com/x")


def test_deny_when_no_allowlist(monkeypatch):
    monkeypatch.delenv("PI_WEB_FETCH_ALLOWLIST", raising=False)
    out = _execute_web_fetch("https://github.com/x")
    assert "keine Allow-List" in out


def test_deny_domain_not_in_allowlist(monkeypatch):
    monkeypatch.setenv("PI_WEB_FETCH_ALLOWLIST", "github.com")
    out = _execute_web_fetch("https://evil.example.com/x")
    assert "nicht in Allow-List" in out


def test_fetch_allowed_domain_and_cache(monkeypatch):
    monkeypatch.setenv("PI_WEB_FETCH_ALLOWLIST", "example.com")

    class FakeResp:
        def __init__(self, body): self._b = body
        def read(self, n=-1): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): pass

    with patch("urllib.request.urlopen", return_value=FakeResp(b"hello world")) as m:
        out1 = _execute_web_fetch("https://example.com/page")
        assert out1 == "hello world"
        # second call served from cache → urlopen not called again
        out2 = _execute_web_fetch("https://example.com/page")
        assert out2 == "hello world"
        assert m.call_count == 1


def test_body_truncated_at_size_cap(monkeypatch):
    monkeypatch.setenv("PI_WEB_FETCH_ALLOWLIST", "example.com")
    big = b"x" * (_WEB_FETCH_MAX_BYTES + 500)

    class FakeResp:
        def __init__(self, body): self._b = body
        def read(self, n=-1): return self._b[:n] if n and n > 0 else self._b
        def __enter__(self): return self
        def __exit__(self, *a): pass

    with patch("urllib.request.urlopen", return_value=FakeResp(big)):
        out = _execute_web_fetch("https://example.com/big")
    assert "[abgeschnitten bei 200kB]" in out
    assert len(out) <= _WEB_FETCH_MAX_BYTES + 50
