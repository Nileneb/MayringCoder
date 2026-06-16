"""POST /embed — synchroner, user-workspace-scoped Direkt-Embed (kein Confirmer)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api.dependencies as _deps
from src.api.auth import get_workspace
from src.api.server import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema
    db_file = tmp_path / "memory.db"
    monkeypatch.setenv("MAYRING_LOCAL_DB", str(db_file))
    adapter = DBAdapter.create(db_file, check_same_thread=False)
    _init_schema(adapter)
    monkeypatch.setattr(_deps, "_conn", adapter)
    prev = app.dependency_overrides.get(get_workspace)
    app.dependency_overrides[get_workspace] = lambda: "ws-user"
    yield TestClient(app)
    if prev is None:
        app.dependency_overrides.pop(get_workspace, None)
    else:
        app.dependency_overrides[get_workspace] = prev
    monkeypatch.setattr(_deps, "_conn", None)


def test_embed_direct_returns_vector_without_confirmer(client, monkeypatch):
    # Kein embed-Device registriert → embed_facade fällt auf _direct_embed → Ollama.
    import mayring_core.ollama_client as oc
    monkeypatch.setattr(oc, "embed_single", lambda url, model, text: [0.1, 0.2, 0.3, 0.4])

    r = client.post("/embed", json={"text": "openalex query", "projekt_id": "openalex"})
    assert r.status_code == 200
    body = r.json()
    assert body["vector"] == [0.1, 0.2, 0.3, 0.4]
    assert body["dim"] == 4
    assert body["model"] == "bge-m3"


def test_embed_direct_passes_text_and_model_to_ollama(client, monkeypatch):
    seen = {}
    import mayring_core.ollama_client as oc

    def _fake(url, model, text):
        seen["model"], seen["text"] = model, text
        return [1.0]

    monkeypatch.setattr(oc, "embed_single", _fake)
    client.post("/embed", json={"text": "hallo welt", "model": "bge-m3"})
    assert seen == {"model": "bge-m3", "text": "hallo welt"}
