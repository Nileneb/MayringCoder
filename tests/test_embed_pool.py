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


def test_enqueue_creates_queued_row(conn):
    eid = ep.enqueue(conn, workspace_id="ws", projekt_id="p1",
                     text="hello world", chunk_ref="paper:1#0")
    job = ep.get(conn, eid)
    assert job["status"] == "queued"
    assert job["text"] == "hello world"
    assert job["device_a"] == "" and job["device_b"] == ""


def test_first_claim_takes_slot_a(conn):
    eid = ep.enqueue(conn, workspace_id="ws", projekt_id="p1", text="t", chunk_ref="c")
    claimed = ep.claim_replica(conn, device_id="dA", workspace_id="ws")
    assert claimed["embed_id"] == eid
    job = ep.get(conn, eid)
    assert job["status"] == "claimed_one"
    assert job["device_a"] == "dA"


def test_second_claim_distinct_device_takes_slot_b(conn):
    eid = ep.enqueue(conn, workspace_id="ws", projekt_id="p1", text="t", chunk_ref="c")
    ep.claim_replica(conn, device_id="dA", workspace_id="ws")
    claimed = ep.claim_replica(conn, device_id="dB", workspace_id="ws")
    assert claimed["embed_id"] == eid
    job = ep.get(conn, eid)
    assert job["status"] == "claimed_two"
    assert job["device_b"] == "dB"


def test_same_device_cannot_take_both_slots(conn):
    ep.enqueue(conn, workspace_id="ws", projekt_id="p1", text="t", chunk_ref="c")
    ep.claim_replica(conn, device_id="dA", workspace_id="ws")
    assert ep.claim_replica(conn, device_id="dA", workspace_id="ws") is None


def test_claim_scoped_by_workspace(conn):
    ep.enqueue(conn, workspace_id="other", projekt_id="p1", text="t", chunk_ref="c")
    assert ep.claim_replica(conn, device_id="dA", workspace_id="ws") is None


def test_claim_skips_fully_claimed(conn):
    ep.enqueue(conn, workspace_id="ws", projekt_id="p1", text="t", chunk_ref="c")
    ep.claim_replica(conn, device_id="dA", workspace_id="ws")
    ep.claim_replica(conn, device_id="dB", workspace_id="ws")
    assert ep.claim_replica(conn, device_id="dC", workspace_id="ws") is None


def test_claim_on_empty_queue_returns_none(conn):
    assert ep.claim_replica(conn, device_id="dA", workspace_id="ws") is None
