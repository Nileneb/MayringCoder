from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
import src.api.dependencies as _deps
from mayring_core.memory import devices as ds
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
    yield TestClient(app), adapter
    app.dependency_overrides.pop(get_workspace, None)
    monkeypatch.setattr(_deps, "_conn", None)


def test_enqueue_audit_unknown_device_is_audited(client):
    c, _ = client
    r = c.post("/embed_pool/enqueue_audit",
               json={"projekt_id": "p", "text": "t", "chunk_ref": "c",
                     "device_a": "player", "vector_a": [0.5, 0.5]}).json()
    assert r["audited"] is True and r["embed_id"]


def test_audit_diverged_then_reap(client):
    c, _ = client
    c.post("/devices/register", headers={"X-Device-Id": "house"}, json={"capabilities": ["embed"]})
    eid = c.post("/embed_pool/enqueue_audit",
                 json={"projekt_id": "p", "text": "t", "chunk_ref": "c",
                       "device_a": "player", "vector_a": [1.0, 0.0]}).json()["embed_id"]
    c.post("/embed_pool/claim", headers={"X-Device-Id": "house"}, json={})
    c.post("/embed_pool/complete", headers={"X-Device-Id": "house"},
           json={"embed_id": eid, "vector": [0.0, 1.0]})
    diverged = c.get("/embed_pool/audits/diverged").json()["audits"]
    assert len(diverged) == 1 and diverged[0]["embed_id"] == eid
    assert diverged[0]["device_a"] == "player" and diverged[0]["projekt_id"] == "p"
    c.post(f"/embed_pool/audits/{eid}/reap")
    assert c.get("/embed_pool/audits/diverged").json()["audits"] == []


def test_audit_agreement_self_cleans(client):
    c, _ = client
    c.post("/devices/register", headers={"X-Device-Id": "house"}, json={"capabilities": ["embed"]})
    eid = c.post("/embed_pool/enqueue_audit",
                 json={"projekt_id": "p", "text": "t", "chunk_ref": "c",
                       "device_a": "player", "vector_a": [0.5, 0.5]}).json()["embed_id"]
    c.post("/embed_pool/claim", headers={"X-Device-Id": "house"}, json={})
    c.post("/embed_pool/complete", headers={"X-Device-Id": "house"},
           json={"embed_id": eid, "vector": [0.5, 0.5]})
    assert c.get("/embed_pool/audits/diverged").json()["audits"] == []  # passed → deleted


def test_high_trust_device_sampled_out(client):
    c, adapter = client
    # seed a trusted, off-cycle device: embed_verified=31, warmup 20, rate 10 → 31%10!=0 → not audited
    ds.upsert_device(adapter, device_id="trusted", workspace_id="system", capabilities=["embed"])
    for _ in range(31):
        ds.record_embed_verified(adapter, "trusted", "system")
    r = c.post("/embed_pool/enqueue_audit",
               json={"projekt_id": "p", "text": "t", "chunk_ref": "c",
                     "device_a": "trusted", "vector_a": [0.5, 0.5]}).json()
    assert r["audited"] is False
    assert "embed_id" not in r or r.get("embed_id") in (None, "")


def test_house_worker_not_quarantined_by_audit_divergence(client):
    c, _ = client
    c.post("/devices/register", headers={"X-Device-Id": "house"}, json={"capabilities": ["embed"]})
    eid = c.post("/embed_pool/enqueue_audit",
                 json={"projekt_id": "p", "text": "t", "chunk_ref": "c",
                       "device_a": "player1", "vector_a": [1.0, 0.0]}).json()["embed_id"]
    c.post("/embed_pool/claim", headers={"X-Device-Id": "house"}, json={})
    c.post("/embed_pool/complete", headers={"X-Device-Id": "house"},
           json={"embed_id": eid, "vector": [0.0, 1.0]})  # divergence
    # house worker must still be able to claim the NEXT audit
    c.post("/embed_pool/enqueue_audit",
           json={"projekt_id": "p", "text": "t2", "chunk_ref": "c2",
                 "device_a": "player2", "vector_a": [0.5, 0.5]})
    j = c.post("/embed_pool/claim", headers={"X-Device-Id": "house"}, json={}).json()
    assert j["job"] is not None  # NOT quarantined → still claims
    devs = {d["device_id"]: d for d in c.get("/devices").json()["devices"]}
    assert devs["house"]["quarantined_until"] == ""  # house worker untouched
