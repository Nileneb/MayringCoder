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
    app.dependency_overrides[get_workspace] = lambda: "ws-test"
    yield TestClient(app)
    if prev is None:
        app.dependency_overrides.pop(get_workspace, None)
    else:
        app.dependency_overrides[get_workspace] = prev
    monkeypatch.setattr(_deps, "_conn", None)


def _register_embed(c, device_id):
    c.post("/devices/register", headers={"X-Device-Id": device_id}, json={"capabilities": ["embed"]})


def test_enqueue_then_dual_claim_and_agree(client):
    c = client
    _register_embed(c, "dA")
    _register_embed(c, "dB")
    eid = c.post("/embed_pool/enqueue",
                 json={"projekt_id": "p1", "text": "hi", "chunk_ref": "paper:1#0"}).json()["embed_id"]
    j1 = c.post("/embed_pool/claim", headers={"X-Device-Id": "dA"}, json={}).json()
    assert j1["job"]["embed_id"] == eid
    j2 = c.post("/embed_pool/claim", headers={"X-Device-Id": "dB"}, json={}).json()
    assert j2["job"]["embed_id"] == eid
    c.post("/embed_pool/complete", headers={"X-Device-Id": "dA"},
           json={"embed_id": eid, "vector": [0.5, 0.5]})
    r = c.post("/embed_pool/complete", headers={"X-Device-Id": "dB"},
               json={"embed_id": eid, "vector": [0.5000001, 0.4999999]}).json()
    assert r["verdict"] == "agreement"
    status = c.get(f"/embed_pool/{eid}").json()
    assert status["status"] == "verified"


def test_unregistered_device_cannot_claim_embed(client):
    c = client
    c.post("/embed_pool/enqueue", json={"projekt_id": "p", "text": "t", "chunk_ref": "c"})
    r = c.post("/embed_pool/claim", headers={"X-Device-Id": "ghost"},
               json={"capabilities": ["embed"]}).json()
    assert r["job"] is None


def test_divergence_quarantines_both(client):
    c = client
    _register_embed(c, "dA")
    _register_embed(c, "dB")
    eid = c.post("/embed_pool/enqueue",
                 json={"projekt_id": "p", "text": "t", "chunk_ref": "c"}).json()["embed_id"]
    c.post("/embed_pool/claim", headers={"X-Device-Id": "dA"}, json={})
    c.post("/embed_pool/claim", headers={"X-Device-Id": "dB"}, json={})
    c.post("/embed_pool/complete", headers={"X-Device-Id": "dA"},
           json={"embed_id": eid, "vector": [1.0, 0.0]})
    r = c.post("/embed_pool/complete", headers={"X-Device-Id": "dB"},
               json={"embed_id": eid, "vector": [0.0, 1.0]}).json()
    assert r["verdict"] == "divergence"
    devs = {d["device_id"]: d for d in c.get("/devices").json()["devices"]}
    assert devs["dA"]["quarantined_until"] != ""
    assert devs["dB"]["quarantined_until"] != ""
    assert devs["dA"]["embed_divergences"] == 1
