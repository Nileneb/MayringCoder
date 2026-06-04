import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE TABLE chunks (chunk_id TEXT, workspace_id TEXT)")
    conn.execute("CREATE TABLE workspace_aliases (alias TEXT PRIMARY KEY, workspace_id TEXT, created_at TEXT NOT NULL)")
    conn.executemany("INSERT INTO chunks VALUES (?,?)", [("a", "unclaimed:dev1"), ("b", "unclaimed:dev1")])
    conn.commit()

    import src.api.routes.workspace_claim as wc
    monkeypatch.setattr(wc, "_get_conn", lambda: conn)
    app = FastAPI()
    app.include_router(wc.router)
    app.dependency_overrides[wc.get_workspace] = lambda: "myws"
    return app, conn


def test_claim_unclaimed_bucket(monkeypatch):
    app, conn = _app(monkeypatch)
    r = TestClient(app).post("/stats/workspaces/claim", json={"workspace_id": "unclaimed:dev1"})
    assert r.status_code == 200, r.text
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE workspace_id='myws'").fetchone()[0] == 2


def test_claim_rejects_non_unclaimed(monkeypatch):
    app, _ = _app(monkeypatch)
    for bogus in ("system", "019e14d6-real", "public"):
        r = TestClient(app).post("/stats/workspaces/claim", json={"workspace_id": bogus})
        assert r.status_code == 403, f"{bogus}: {r.status_code}"
