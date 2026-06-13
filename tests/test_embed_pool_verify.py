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


def _two_claims(conn):
    ep.claim_replica(conn, device_id="dA", workspace_id="ws")
    ep.claim_replica(conn, device_id="dB", workspace_id="ws")


def test_first_result_does_not_verify_yet(conn):
    eid = ep.enqueue(conn, workspace_id="ws", projekt_id="p", text="t", chunk_ref="c")
    _two_claims(conn)
    out = ep.submit_result(conn, embed_id=eid, device_id="dA", vector=[1.0, 0.0], threshold=0.9999)
    assert out["status"] == "claimed_two"
    assert out["verdict"] == ""


def test_agreement_writes_verdict_and_returns_vector(conn):
    eid = ep.enqueue(conn, workspace_id="ws", projekt_id="p", text="t", chunk_ref="c")
    _two_claims(conn)
    ep.submit_result(conn, embed_id=eid, device_id="dA", vector=[0.5, 0.5], threshold=0.9999)
    out = ep.submit_result(conn, embed_id=eid, device_id="dB", vector=[0.5000001, 0.4999999], threshold=0.9999)
    assert out["status"] == "verified"
    assert out["verdict"] == "agreement"
    assert out["agreed_vector"] == [0.5, 0.5]
    assert out["devices"] == ["dA", "dB"]


def test_divergence_sets_diverged_and_flags_both(conn):
    eid = ep.enqueue(conn, workspace_id="ws", projekt_id="p", text="t", chunk_ref="c")
    _two_claims(conn)
    ep.submit_result(conn, embed_id=eid, device_id="dA", vector=[1.0, 0.0], threshold=0.9999)
    out = ep.submit_result(conn, embed_id=eid, device_id="dB", vector=[0.0, 1.0], threshold=0.9999)
    assert out["status"] == "diverged"
    assert out["verdict"] == "divergence"
    assert "agreed_vector" not in out
    assert set(out["devices"]) == {"dA", "dB"}


def test_golden_job_pass(conn):
    eid = ep.enqueue_golden(conn, workspace_id="ws", text="probe", reference=[0.6, 0.8])
    claimed = ep.claim_golden(conn, device_id="dQ", workspace_id="ws")
    assert claimed["embed_id"] == eid
    out = ep.submit_golden(conn, embed_id=eid, device_id="dQ", vector=[0.6000001, 0.7999999], threshold=0.9999)
    assert out["passed"] is True


def test_golden_job_fail(conn):
    eid = ep.enqueue_golden(conn, workspace_id="ws", text="probe", reference=[0.6, 0.8])
    ep.claim_golden(conn, device_id="dQ", workspace_id="ws")
    out = ep.submit_golden(conn, embed_id=eid, device_id="dQ", vector=[1.0, 0.0], threshold=0.9999)
    assert out["passed"] is False
