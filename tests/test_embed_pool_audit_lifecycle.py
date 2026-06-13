import sqlite3
import pytest
from mayring_core.memory import embed_pool as ep


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    ep.ensure_tables(c); yield c; c.close()


def _audit(conn, va):
    return ep.enqueue_with_seed(conn, workspace_id="system", projekt_id="p", text="t",
                                chunk_ref="c", device_a="player", vector_a=va, is_audit=True)


def test_audit_agreement_deletes_row(conn):
    eid = _audit(conn, [0.5, 0.5])
    ep.claim_replica(conn, device_id="house", workspace_id="system")
    ep.submit_result(conn, embed_id=eid, device_id="house", vector=[0.5, 0.5], threshold=0.9999)
    assert ep.get(conn, eid) is None


def test_audit_divergence_keeps_row_for_reaper(conn):
    eid = _audit(conn, [1.0, 0.0])
    ep.claim_replica(conn, device_id="house", workspace_id="system")
    ep.submit_result(conn, embed_id=eid, device_id="house", vector=[0.0, 1.0], threshold=0.9999)
    job = ep.get(conn, eid)
    assert job is not None and job["status"] == "diverged"
    assert [d["embed_id"] for d in ep.list_diverged_audits(conn, "system")] == [eid]


def test_reap_deletes_diverged_audit(conn):
    eid = _audit(conn, [1.0, 0.0])
    ep.claim_replica(conn, device_id="house", workspace_id="system")
    ep.submit_result(conn, embed_id=eid, device_id="house", vector=[0.0, 1.0], threshold=0.9999)
    ep.reap_audit(conn, eid)
    assert ep.get(conn, eid) is None
    assert ep.list_diverged_audits(conn, "system") == []


def test_non_audit_agreement_still_marks_verified(conn):
    eid = ep.enqueue_with_seed(conn, workspace_id="system", projekt_id="p", text="t",
                               chunk_ref="c", device_a="player", vector_a=[0.5, 0.5])
    ep.claim_replica(conn, device_id="house", workspace_id="system")
    out = ep.submit_result(conn, embed_id=eid, device_id="house", vector=[0.5, 0.5], threshold=0.9999)
    assert out["status"] == "verified"
    assert ep.get(conn, eid)["status"] == "verified"
