import sqlite3
import pytest
from mayring_core.memory import devices as ds
from mayring_core.memory import embed_pool as ep
from src import embed_facade


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ds.ensure_tables(c)
    ep.ensure_tables(c)
    yield c
    c.close()


def test_fallback_when_too_few_devices(conn, monkeypatch):
    called = {}
    monkeypatch.setattr(embed_facade, "_direct_embed",
                        lambda text, **k: called.setdefault("v", [9.0]) or [9.0])
    vec = embed_facade.verified_embedding(
        conn, text="t", workspace_id="ws", projekt_id="p", chunk_ref="c",
        now="2026-06-13T12:00:00Z")
    assert vec == [9.0]
    assert "v" in called


def test_uses_pool_when_enough_devices(conn, monkeypatch):
    ds.upsert_device(conn, device_id="dA", workspace_id="ws", capabilities=["embed"])
    ds.upsert_device(conn, device_id="dB", workspace_id="ws", capabilities=["embed"])

    def fake_poll(c, eid, **k):
        ep.claim_replica(c, device_id="dA", workspace_id="ws")
        ep.claim_replica(c, device_id="dB", workspace_id="ws")
        ep.submit_result(c, embed_id=eid, device_id="dA", vector=[0.5, 0.5], threshold=0.9999)
        out = ep.submit_result(c, embed_id=eid, device_id="dB", vector=[0.5, 0.5], threshold=0.9999)
        return out["agreed_vector"]

    monkeypatch.setattr(embed_facade, "_poll_verified", fake_poll)
    monkeypatch.setattr(embed_facade, "_direct_embed",
                        lambda *a, **k: pytest.fail("should not fall back"))
    vec = embed_facade.verified_embedding(
        conn, text="t", workspace_id="ws", projekt_id="p", chunk_ref="c",
        now="2026-06-13T12:00:00Z")
    assert vec == [0.5, 0.5]
