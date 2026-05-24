"""get_chroma_collection routes to a Chroma server when MAYRING_CHROMA_HOST is set.

WHY(api-concurrency-capacity): embedded PersistentClient is not
multi-process-safe and blocks `uvicorn --workers`. In prod the API talks to a
Chroma server via HttpClient; these tests pin both branches so a refactor can't
silently fall back to embedded (which would reintroduce the multi-worker
corruption risk) nor accidentally hit the network in standalone/test runs.
"""
from __future__ import annotations

import mayring_core.memory.store as store


class _SentinelCollection:
    pass


def _fresh_caches(monkeypatch):
    monkeypatch.setattr(store, "_chroma_clients", {})
    monkeypatch.setattr(store, "_chroma_collections", {})


def test_httpclient_used_when_host_set(monkeypatch):
    import chromadb

    _fresh_caches(monkeypatch)
    monkeypatch.setenv("MAYRING_CHROMA_HOST", "mayring-chroma")
    monkeypatch.setenv("MAYRING_CHROMA_PORT", "8000")

    captured: dict = {}

    class _Client:
        def __init__(self, host, port):
            captured["host"], captured["port"] = host, port

        def get_or_create_collection(self, name):
            captured["name"] = name
            return _SentinelCollection()

    def _boom(*a, **k):
        raise AssertionError("PersistentClient used despite MAYRING_CHROMA_HOST set")

    monkeypatch.setattr(chromadb, "HttpClient", _Client)
    monkeypatch.setattr(chromadb, "PersistentClient", _boom)

    coll = store.get_chroma_collection("memory_chunks")
    assert isinstance(coll, _SentinelCollection)
    assert captured == {"host": "mayring-chroma", "port": 8000, "name": "memory_chunks"}


def test_persistentclient_used_when_host_unset(monkeypatch, tmp_path):
    import chromadb

    _fresh_caches(monkeypatch)
    monkeypatch.delenv("MAYRING_CHROMA_HOST", raising=False)

    used: dict = {}

    class _PC:
        def __init__(self, path):
            used["path"] = path

        def get_or_create_collection(self, name):
            return _SentinelCollection()

    def _boom_http(*a, **k):
        raise AssertionError("HttpClient used despite MAYRING_CHROMA_HOST unset")

    monkeypatch.setattr(chromadb, "PersistentClient", _PC)
    monkeypatch.setattr(chromadb, "HttpClient", _boom_http)

    coll = store.get_chroma_collection("memory_chunks", path=tmp_path)
    assert isinstance(coll, _SentinelCollection)
    assert used["path"] == str(tmp_path)
