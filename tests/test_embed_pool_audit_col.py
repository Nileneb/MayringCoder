import sqlite3
import pytest
from mayring_core.memory import embed_pool as ep


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    ep.ensure_tables(c); yield c; c.close()


def test_enqueue_with_seed_audit_flag(conn):
    eid = ep.enqueue_with_seed(conn, workspace_id="system", projekt_id="p", text="t",
                               chunk_ref="c", device_a="dev", vector_a=[0.1, 0.2], is_audit=True)
    job = ep.get(conn, eid)
    assert job["is_audit"] == 1
    assert job["status"] == "claimed_one"


def test_enqueue_with_seed_defaults_not_audit(conn):
    eid = ep.enqueue_with_seed(conn, workspace_id="system", projekt_id="p", text="t",
                               chunk_ref="c", device_a="dev", vector_a=[0.1, 0.2])
    assert ep.get(conn, eid)["is_audit"] == 0
