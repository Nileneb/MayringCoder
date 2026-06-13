import sqlite3
import pytest
from mayring_core.memory import devices as ds
from mayring_core.memory import embed_pool as ep


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ds.ensure_tables(c)
    ep.ensure_tables(c)
    yield c
    c.close()


def test_high_trust_device_still_needs_a_second_device(conn):
    """v1 behavior gate: even a high-trust device cannot single-handedly verify —
    dual-send is unconditional below trust_min_devices. The job stays claimed_one
    until a SECOND distinct device claims slot B, and submitting only A's result
    does not produce a verdict."""
    ds.upsert_device(conn, device_id="dA", workspace_id="ws", capabilities=["embed"])
    for _ in range(50):
        ds.record_embed_verified(conn, "dA", "ws")  # trust_score = 50
    eid = ep.enqueue(conn, workspace_id="ws", projekt_id="p", text="t", chunk_ref="c")
    ep.claim_replica(conn, device_id="dA", workspace_id="ws")
    job = ep.get(conn, eid)
    assert job["status"] == "claimed_one"   # NOT verified despite high trust
    out = ep.submit_result(conn, embed_id=eid, device_id="dA", vector=[1.0, 0.0],
                           threshold=0.9999)
    assert out["verdict"] == ""             # waits for the second device
