import sqlite3
import pytest
from mayring_core.memory import embed_pool as ep


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ep.ensure_tables(c)
    yield c
    c.close()


def test_seed_prefills_slot_a(conn):
    eid = ep.enqueue_with_seed(conn, workspace_id="system", projekt_id="p1",
                               text="t", chunk_ref="c", device_a="player-dev",
                               vector_a=[0.5, 0.5])
    job = ep.get(conn, eid)
    assert job["status"] == "claimed_one"
    assert job["device_a"] == "player-dev"
    import json
    assert json.loads(job["result_a"]) == [0.5, 0.5]
    assert job["device_b"] == ""


def test_second_device_claims_slot_b_then_verifies(conn):
    eid = ep.enqueue_with_seed(conn, workspace_id="system", projekt_id="p1",
                               text="t", chunk_ref="c", device_a="player-dev",
                               vector_a=[0.5, 0.5])
    claimed = ep.claim_replica(conn, device_id="confirmer", workspace_id="system")
    assert claimed["embed_id"] == eid
    out = ep.submit_result(conn, embed_id=eid, device_id="confirmer",
                           vector=[0.5000001, 0.4999999], threshold=0.9999)
    assert out["verdict"] == "agreement"
    assert out["agreed_vector"] == [0.5, 0.5]


def test_seed_device_cannot_also_claim_slot_b(conn):
    ep.enqueue_with_seed(conn, workspace_id="system", projekt_id="p1", text="t",
                         chunk_ref="c", device_a="player-dev", vector_a=[1.0, 0.0])
    assert ep.claim_replica(conn, device_id="player-dev", workspace_id="system") is None
