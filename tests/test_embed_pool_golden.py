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
    app.dependency_overrides[get_workspace] = lambda: "ws-test"
    yield TestClient(app)
    app.dependency_overrides.pop(get_workspace, None)
    monkeypatch.setattr(_deps, "_conn", None)


def _embed_reg(c, d):
    c.post("/devices/register", headers={"X-Device-Id": d}, json={"capabilities": ["embed"]})


def _force_divergence(c):
    _embed_reg(c, "dA"); _embed_reg(c, "dB")
    eid = c.post("/embed_pool/enqueue",
                 json={"projekt_id": "p", "text": "t", "chunk_ref": "c"}).json()["embed_id"]
    c.post("/embed_pool/claim", headers={"X-Device-Id": "dA"}, json={})
    c.post("/embed_pool/claim", headers={"X-Device-Id": "dB"}, json={})
    c.post("/embed_pool/complete", headers={"X-Device-Id": "dA"}, json={"embed_id": eid, "vector": [1.0, 0.0]})
    c.post("/embed_pool/complete", headers={"X-Device-Id": "dB"}, json={"embed_id": eid, "vector": [0.0, 1.0]})
    return eid


def test_divergence_enqueues_golden_for_each_device(client):
    c = client
    _force_divergence(c)
    g = c.post("/embed_pool/golden/claim", headers={"X-Device-Id": "dA"}, json={}).json()
    assert g["job"] is not None
    assert g["job"]["is_golden"] == 1


def test_golden_pass_rehabilitates(client):
    c = client
    _force_divergence(c)
    g = c.post("/embed_pool/golden/claim", headers={"X-Device-Id": "dA"}, json={}).json()["job"]
    import json as _json
    gref = _json.loads(g["golden_ref"])  # echo stored reference → deterministic pass
    c.post("/embed_pool/golden/complete", headers={"X-Device-Id": "dA"},
           json={"embed_id": g["embed_id"], "vector": gref})
    devs = {d["device_id"]: d for d in c.get("/devices").json()["devices"]}
    assert devs["dA"]["quarantined_until"] == ""  # rehabilitated
