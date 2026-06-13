import sqlite3
import pytest
from mayring_core.memory import devices as ds


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def test_new_columns_exist_and_default(conn):
    ds.upsert_device(conn, device_id="d1", workspace_id="ws", capabilities=["embed"])
    row = ds.list_devices(conn, "ws")[0]
    assert row["trust_score"] == 0.0
    assert row["embed_verified"] == 0
    assert row["embed_divergences"] == 0
    assert row["quarantined_until"] == ""


def test_record_embed_verified_increments(conn):
    ds.upsert_device(conn, device_id="d1", workspace_id="ws", capabilities=["embed"])
    ds.record_embed_verified(conn, "d1", "ws")
    ds.record_embed_verified(conn, "d1", "ws")
    row = ds.list_devices(conn, "ws")[0]
    assert row["embed_verified"] == 2
    assert row["trust_score"] == pytest.approx(2.0)


def test_record_divergence_and_quarantine(conn):
    ds.upsert_device(conn, device_id="d1", workspace_id="ws", capabilities=["embed"])
    ds.record_embed_divergence(conn, "d1", "ws")
    ds.set_quarantine(conn, "d1", "ws", until="2999-01-01T00:00:00Z")
    row = ds.list_devices(conn, "ws")[0]
    assert row["embed_divergences"] == 1
    assert row["quarantined_until"] == "2999-01-01T00:00:00Z"


def test_eligible_embed_devices_filters(conn):
    ds.upsert_device(conn, device_id="ok", workspace_id="ws", capabilities=["embed"])
    ds.upsert_device(conn, device_id="nocap", workspace_id="ws", capabilities=["local-gpu"])
    ds.upsert_device(conn, device_id="quar", workspace_id="ws", capabilities=["embed"])
    ds.set_quarantine(conn, "quar", "ws", until="2999-01-01T00:00:00Z")
    elig = ds.eligible_embed_devices(conn, "ws", now="2026-06-13T12:00:00Z", fresh_seconds=120)
    assert "ok" in elig
    assert "nocap" not in elig
    assert "quar" not in elig


def test_stale_heartbeat_not_eligible(conn):
    ds.upsert_device(conn, device_id="stale", workspace_id="ws", capabilities=["embed"])
    elig = ds.eligible_embed_devices(conn, "ws", now="2999-01-01T00:00:00Z", fresh_seconds=120)
    assert "stale" not in elig


def test_quarantine_expired_is_eligible(conn):
    ds.upsert_device(conn, device_id="reh", workspace_id="ws", capabilities=["embed"])
    ds.set_quarantine(conn, "reh", "ws", until="2000-01-01T00:00:00Z")
    elig = ds.eligible_embed_devices(conn, "ws", now="2026-06-13T12:00:00Z", fresh_seconds=10**9)
    assert "reh" in elig


def test_empty_last_seen_is_not_eligible_fail_closed(conn):
    """A row with 'embed' capability but no heartbeat must be EXCLUDED (fail-closed)."""
    ds.ensure_tables(conn)
    conn.execute("INSERT INTO devices (device_id, workspace_id, capabilities, last_seen) "
                 "VALUES ('noheartbeat', 'ws', 'embed', '')")
    conn.commit()
    elig = ds.eligible_embed_devices(conn, "ws", now="2026-06-13T12:00:00Z", fresh_seconds=10**9)
    assert "noheartbeat" not in elig


def test_quarantine_mixed_timestamp_format_still_gates(conn):
    """set_quarantine normalises +00:00 input to Z form so the lex compare holds."""
    ds.upsert_device(conn, device_id="d1", workspace_id="ws", capabilities=["embed"])
    ds.set_quarantine(conn, "d1", "ws", until="2999-01-01T00:00:00+00:00")  # non-Z input
    row = ds.list_devices(conn, "ws")[0]
    assert row["quarantined_until"] == "2999-01-01T00:00:00Z"  # normalised
    elig = ds.eligible_embed_devices(conn, "ws", now="2026-06-13T12:00:00Z", fresh_seconds=10**9)
    assert "d1" not in elig  # still quarantined despite mixed input format
