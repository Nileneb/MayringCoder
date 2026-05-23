"""Tests for #274: Device-Registry + Hook-Events + Write-Job-Routing.

Two layers:
  * data layer (`mayring_core.memory.devices`) against a raw sqlite conn
  * HTTP routes (`src/api/routes/devices.py`) via TestClient, incl. the
    headline acceptance: a write-job is only ever claimed by a device
    registered with the 'write' capability.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

import src.api.dependencies as _deps
from mayring_core.memory import devices as ds
from src.agents import pi_jobs
from src.api.auth import get_workspace
from src.api.server import app


# ===========================================================================
# Data layer
# ===========================================================================

@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def test_ensure_tables_creates_schema(conn):
    ds.ensure_tables(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"devices", "hook_events"} <= tables


def test_upsert_and_list_device(conn):
    ds.upsert_device(conn, device_id="d1", workspace_id="ws", name="laptop",
                     os="Linux", capabilities=["local-gpu", "write"])
    devs = ds.list_devices(conn, "ws")
    assert len(devs) == 1
    assert devs[0]["device_id"] == "d1"
    assert devs[0]["capabilities"] == ["local-gpu", "write"]
    assert devs[0]["status"] == "active"
    assert devs[0]["last_seen"]


def test_list_devices_scoped_by_workspace(conn):
    ds.upsert_device(conn, device_id="a", workspace_id="alpha")
    ds.upsert_device(conn, device_id="b", workspace_id="beta")
    assert {d["device_id"] for d in ds.list_devices(conn, "alpha")} == {"a"}
    assert {d["device_id"] for d in ds.list_devices(conn, "beta")} == {"b"}


def test_upsert_preserves_created_at_on_reregister(conn):
    ds.upsert_device(conn, device_id="d1", workspace_id="ws", name="old")
    created = ds.list_devices(conn, "ws")[0]["created_at"]
    ds.upsert_device(conn, device_id="d1", workspace_id="ws", name="new",
                     capabilities=["write"])
    row = ds.list_devices(conn, "ws")[0]
    assert row["created_at"] == created       # preserved
    assert row["name"] == "new"               # overwritten
    assert row["capabilities"] == ["write"]


def test_capabilities_string_normalised(conn):
    ds.upsert_device(conn, device_id="d1", workspace_id="ws",
                     capabilities="local-gpu, write ,, local-gpu")
    assert ds.device_capabilities(conn, "d1", "ws") == ["local-gpu", "write"]


def test_device_capabilities_none_when_unregistered(conn):
    ds.ensure_tables(conn)
    assert ds.device_capabilities(conn, "ghost", "ws") is None


def test_device_capabilities_empty_list_when_registered_no_caps(conn):
    ds.upsert_device(conn, device_id="d1", workspace_id="ws")
    assert ds.device_capabilities(conn, "d1", "ws") == []


# --- write-routing core (effective_capabilities) ---------------------------

def test_effective_caps_registry_is_authoritative(conn):
    """A registered device's stored caps win — a worker cannot self-grant."""
    ds.upsert_device(conn, device_id="d1", workspace_id="ws",
                     capabilities=["local-gpu"])
    caps = ds.effective_capabilities(conn, "d1", "ws", self_reported=["write"])
    assert caps == ["local-gpu"]            # self-reported 'write' ignored


def test_effective_caps_unregistered_strips_write(conn):
    """An unregistered device cannot claim write — 'write' is stripped."""
    ds.ensure_tables(conn)
    caps = ds.effective_capabilities(conn, "ghost", "ws",
                                     self_reported=["local-gpu", "write"])
    assert caps == ["local-gpu"]


def test_effective_caps_registered_write_kept(conn):
    ds.upsert_device(conn, device_id="d1", workspace_id="ws",
                     capabilities=["write"])
    assert ds.effective_capabilities(conn, "d1", "ws", []) == ["write"]


# --- heartbeat + hook events ------------------------------------------------

def test_touch_last_seen_creates_minimal_row(conn):
    ds.touch_last_seen(conn, "d1", "ws")
    devs = ds.list_devices(conn, "ws")
    assert len(devs) == 1
    assert devs[0]["capabilities"] == []     # no caps until real register


def test_touch_last_seen_updates_existing(conn):
    ds.upsert_device(conn, device_id="d1", workspace_id="ws",
                     capabilities=["write"])
    ds.touch_last_seen(conn, "d1", "ws")
    # capabilities must NOT be wiped by a heartbeat
    assert ds.device_capabilities(conn, "d1", "ws") == ["write"]


def test_record_and_list_hook_events(conn):
    ds.record_hook_event(conn, workspace_id="ws", device_id="d1",
                         hook_type="Stop", summary="did a thing")
    ds.record_hook_event(conn, workspace_id="ws", device_id="d1",
                         hook_type="UserPromptSubmit", summary="prompt")
    events = ds.list_hook_events(conn, "ws")
    assert len(events) == 2
    assert {e["hook_type"] for e in events} == {"Stop", "UserPromptSubmit"}
    assert events[0]["payload"].get("summary")


def test_hook_events_scoped_by_workspace(conn):
    ds.record_hook_event(conn, workspace_id="alpha", hook_type="Stop")
    ds.record_hook_event(conn, workspace_id="beta", hook_type="Stop")
    assert len(ds.list_hook_events(conn, "alpha")) == 1


def test_hook_event_since_filter(conn):
    ds.record_hook_event(conn, workspace_id="ws", hook_type="Stop",
                         fired_at="2026-01-01T00:00:00Z")
    recent = ds.list_hook_events(conn, "ws", since="2999-01-01T00:00:00Z")
    assert recent == []


def test_hook_event_derives_heartbeat(conn):
    ds.record_hook_event(conn, workspace_id="ws", device_id="d1",
                         hook_type="Stop")
    # the firing registered the device as alive
    devs = ds.list_devices(conn, "ws")
    assert len(devs) == 1 and devs[0]["device_id"] == "d1"


# ===========================================================================
# HTTP routes
# ===========================================================================

@pytest.fixture
def client(tmp_path, monkeypatch):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema

    db_file = tmp_path / "memory.db"
    # _job_db_path() reads MAYRING_LOCAL_DB → both the device conn (adapter)
    # and the pi_jobs per-call conn point at the SAME file.
    monkeypatch.setenv("MAYRING_LOCAL_DB", str(db_file))
    adapter = DBAdapter.create(db_file, check_same_thread=False)
    _init_schema(adapter)
    monkeypatch.setattr(_deps, "_conn", adapter)

    prev = app.dependency_overrides.get(get_workspace)
    app.dependency_overrides[get_workspace] = lambda: "ws-test"
    yield TestClient(app), db_file
    if prev is None:
        app.dependency_overrides.pop(get_workspace, None)
    else:
        app.dependency_overrides[get_workspace] = prev
    monkeypatch.setattr(_deps, "_conn", None)


def test_route_register_then_list(client):
    c, _ = client
    r = c.post("/devices/register", json={"device_id": "d1", "name": "lap",
                                          "os": "Linux", "capabilities": ["write"]})
    assert r.status_code == 200 and r.json()["registered"] is True
    lst = c.get("/devices").json()
    assert lst["count"] == 1
    assert lst["devices"][0]["capabilities"] == ["write"]


def test_route_device_id_from_header(client):
    c, _ = client
    r = c.post("/devices/register", headers={"X-Device-Id": "hdr-dev"},
               json={"name": "x", "capabilities": ["local-gpu"]})
    assert r.status_code == 200
    assert c.get("/devices").json()["devices"][0]["device_id"] == "hdr-dev"


def test_route_register_requires_device_id(client):
    c, _ = client
    r = c.post("/devices/register", json={"name": "no-id"})
    assert r.status_code == 400


def test_route_heartbeat(client):
    c, _ = client
    r = c.post("/devices/heartbeat", json={"device_id": "d1"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert c.get("/devices").json()["count"] == 1


def test_route_hook_event_recorded_and_listed(client):
    c, _ = client
    r = c.post("/hooks/events", json={"device_id": "d1", "hook_type": "Stop",
                                      "summary": "ran"})
    assert r.status_code == 200 and r.json()["recorded"] is True
    ev = c.get("/hooks/events").json()
    assert ev["count"] == 1 and ev["events"][0]["hook_type"] == "Stop"


def test_route_hook_event_never_5xx(client, monkeypatch):
    """A DB failure in /hooks/events must surface as recorded:false 200,
    never a 5xx that breaks the client's per-prompt hook."""
    c, _ = client
    monkeypatch.setattr(ds, "record_hook_event",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r = c.post("/hooks/events", json={"device_id": "d1", "hook_type": "Stop"})
    assert r.status_code == 200
    assert r.json()["recorded"] is False


# --- write-job routing acceptance ------------------------------------------

def test_write_job_only_claimed_by_write_device(client):
    """#274 headline acceptance: a capability_required='write' job is NOT
    handed to a non-write device, but IS handed to a registered write device."""
    c, db_file = client
    pi_jobs.insert_cloud_job("write me", workspace_id="ws-test",
                             capability_required="write", db_path=db_file)

    # device without write → no claim
    c.post("/devices/register", json={"device_id": "ro", "capabilities": ["local-gpu"]})
    r = c.post("/pi_task_claim_cloud", headers={"X-Device-Id": "ro"},
               json={"capabilities": ["write"]})  # self-report write is ignored
    assert r.status_code == 200 and r.json()["job"] is None

    # device WITH registered write → claims it
    c.post("/devices/register", json={"device_id": "rw", "capabilities": ["write"]})
    r = c.post("/pi_task_claim_cloud", headers={"X-Device-Id": "rw"},
               json={"capabilities": []})
    body = r.json()
    assert body["job"] is not None
    assert body["job"]["task_text"] == "write me"
    assert body["job"]["claimed_by"] == "rw"


def test_unregistered_device_cannot_claim_write(client):
    c, db_file = client
    pi_jobs.insert_cloud_job("w", workspace_id="ws-test",
                             capability_required="write", db_path=db_file)
    r = c.post("/pi_task_claim_cloud", headers={"X-Device-Id": "ghost"},
               json={"capabilities": ["write"]})  # never registered
    assert r.json()["job"] is None


def test_claim_then_complete_cloud(client):
    c, db_file = client
    pi_jobs.insert_cloud_job("do it", workspace_id="ws-test", db_path=db_file)
    c.post("/devices/register", json={"device_id": "w1", "capabilities": []})
    claim = c.post("/pi_task_claim_cloud", headers={"X-Device-Id": "w1"},
                   json={"capabilities": []}).json()
    job_id = claim["job"]["job_id"]
    r = c.post("/pi_task_complete_cloud", json={"job_id": job_id,
                                                "result": {"text": "done"}})
    assert r.status_code == 200 and r.json()["status"] == "completed"
    fetched = pi_jobs.get_job(job_id, db_path=db_file)
    assert fetched.status == "completed"


def test_complete_cloud_foreign_workspace_404(client):
    c, db_file = client
    # job lives in a different workspace than the caller (ws-test)
    job = pi_jobs.insert_cloud_job("other", workspace_id="other-ws", db_path=db_file)
    r = c.post("/pi_task_complete_cloud", json={"job_id": job.job_id,
                                                "result": {"text": "x"}})
    assert r.status_code == 404
