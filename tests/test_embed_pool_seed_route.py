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
    app.dependency_overrides[get_workspace] = lambda: "system"
    yield TestClient(app)
    app.dependency_overrides.pop(get_workspace, None)
    monkeypatch.setattr(_deps, "_conn", None)


def test_enqueue_seeded_then_confirmer_agrees(client):
    c = client
    c.post("/devices/register", headers={"X-Device-Id": "confirmer"}, json={"capabilities": ["embed"]})
    eid = c.post("/embed_pool/enqueue_seeded",
                 json={"projekt_id": "p", "text": "t", "chunk_ref": "c",
                       "device_a": "player-dev", "vector_a": [0.5, 0.5]}).json()["embed_id"]
    j = c.post("/embed_pool/claim", headers={"X-Device-Id": "confirmer"}, json={}).json()
    assert j["job"]["embed_id"] == eid
    r = c.post("/embed_pool/complete", headers={"X-Device-Id": "confirmer"},
               json={"embed_id": eid, "vector": [0.5, 0.5]}).json()
    assert r["verdict"] == "agreement"


def test_enqueue_seeded_divergence_quarantines_player_device(client):
    c = client
    c.post("/devices/register", headers={"X-Device-Id": "confirmer"}, json={"capabilities": ["embed"]})
    eid = c.post("/embed_pool/enqueue_seeded",
                 json={"projekt_id": "p", "text": "t", "chunk_ref": "c",
                       "device_a": "player-dev", "vector_a": [1.0, 0.0]}).json()["embed_id"]
    c.post("/embed_pool/claim", headers={"X-Device-Id": "confirmer"}, json={})
    r = c.post("/embed_pool/complete", headers={"X-Device-Id": "confirmer"},
               json={"embed_id": eid, "vector": [0.0, 1.0]}).json()
    assert r["verdict"] == "divergence"
    status = c.get(f"/embed_pool/{eid}").json()
    assert status["status"] == "diverged"
