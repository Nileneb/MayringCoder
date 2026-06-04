"""C1 project-groups API (workspace-scoped)."""
import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.auth import get_workspace
import src.api.dependencies as _deps

WS = "test-c1-ws"


@pytest.fixture
def client(tmp_path, monkeypatch):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema

    db = DBAdapter.create(tmp_path / "c1.db", check_same_thread=False)
    _init_schema(db)
    monkeypatch.setattr(_deps, "_conn", db)

    app.dependency_overrides[get_workspace] = lambda: WS
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()
    monkeypatch.setattr(_deps, "_conn", None)


def test_create_group_auto_color_and_list(client):
    r = client.post("/project-groups", json={"name": "Mayring-Stack"})
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["name"] == "Mayring-Stack"
    assert g["color"].startswith("#")
    lst = client.get("/project-groups").json()["groups"]
    assert any(x["id"] == g["id"] and x["member_count"] == 0 for x in lst)


def test_create_group_duplicate_name_409(client):
    client.post("/project-groups", json={"name": "Dup"})
    r = client.post("/project-groups", json={"name": "dup"})
    assert r.status_code == 409, r.text


def test_create_group_empty_name_422(client):
    r = client.post("/project-groups", json={"name": "  "})
    assert r.status_code == 422


def test_patch_recolor_and_rename(client):
    gid = client.post("/project-groups", json={"name": "Old"}).json()["id"]
    r = client.patch(f"/project-groups/{gid}",
                     json={"name": "New", "color": "#ef4444"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "New" and r.json()["color"] == "#ef4444"


def test_patch_color_not_in_palette_422(client):
    gid = client.post("/project-groups", json={"name": "G"}).json()["id"]
    r = client.patch(f"/project-groups/{gid}", json={"color": "#123456"})
    assert r.status_code == 422


def test_patch_unknown_group_404(client):
    r = client.patch("/project-groups/does-not-exist", json={"name": "x"})
    assert r.status_code == 404


def test_delete_group_nulls_members(client):
    gid = client.post("/project-groups", json={"name": "Temp"}).json()["id"]
    client.post("/project-groups/assign",
                json={"repo_slug": "nileneb/foo", "group_id": gid})
    assert client.delete(f"/project-groups/{gid}").status_code == 200
    projects = client.get("/projects").json()["projects"]
    foo = next(p for p in projects if (p["repo"] or "").endswith("foo"))
    assert foo["group_id"] is None


def test_assign_creates_project_when_missing(client):
    gid = client.post("/project-groups", json={"name": "Grp"}).json()["id"]
    r = client.post("/project-groups/assign",
                    json={"repo_slug": "Nileneb/Bar", "group_id": gid})
    assert r.status_code == 200, r.text
    projects = client.get("/projects").json()["projects"]
    bar = next(p for p in projects if (p["repo"] or "") == "nileneb/bar")
    assert bar["group_id"] == gid and bar["group_color"].startswith("#")


def test_assign_null_detaches(client):
    gid = client.post("/project-groups", json={"name": "Grp2"}).json()["id"]
    client.post("/project-groups/assign",
                json={"repo_slug": "nileneb/baz", "group_id": gid})
    client.post("/project-groups/assign",
                json={"repo_slug": "nileneb/baz", "group_id": None})
    projects = client.get("/projects").json()["projects"]
    baz = next(p for p in projects if (p["repo"] or "") == "nileneb/baz")
    assert baz["group_id"] is None


def test_assign_foreign_group_404(client):
    r = client.post("/project-groups/assign",
                    json={"repo_slug": "nileneb/x", "group_id": "nope"})
    assert r.status_code == 404
