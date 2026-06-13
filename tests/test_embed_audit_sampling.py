import sqlite3
import pytest
from mayring_core.memory import devices as ds
from mayring_core.memory import embed_pool as ep


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    ds.ensure_tables(c); ep.ensure_tables(c); yield c; c.close()


def test_unknown_device_always_audited(conn):
    assert ep.should_audit(conn, "ghost", "system", warmup=20, sample_rate=10) is True


def test_low_trust_always_audited(conn):
    ds.upsert_device(conn, device_id="d", workspace_id="system", capabilities=["embed"])
    for _ in range(5):
        ds.record_embed_verified(conn, "d", "system")
    assert ep.should_audit(conn, "d", "system", warmup=20, sample_rate=10) is True


def test_high_trust_sampled(conn):
    ds.upsert_device(conn, device_id="d", workspace_id="system", capabilities=["embed"])
    for _ in range(30):
        ds.record_embed_verified(conn, "d", "system")
    assert ep.should_audit(conn, "d", "system", warmup=20, sample_rate=10) is True   # 30 % 10 == 0
    ds.record_embed_verified(conn, "d", "system")  # 31
    assert ep.should_audit(conn, "d", "system", warmup=20, sample_rate=10) is False


def test_quarantined_device_always_audited(conn):
    ds.upsert_device(conn, device_id="d", workspace_id="system", capabilities=["embed"])
    for _ in range(30):
        ds.record_embed_verified(conn, "d", "system")
    ds.set_quarantine(conn, "d", "system", until="2999-01-01T00:00:00Z")
    assert ep.should_audit(conn, "d", "system", warmup=20, sample_rate=10) is True
