import pytest
from fastapi.testclient import TestClient

import src.api.dependencies as _deps
from src.api.auth import get_workspace
from src.api.server import app


_CHROMA = {"cb:t:api": [1.0, 0.0, 0.0], "cb:t:domain": [0.0, 1.0, 0.0]}


class _FakeChroma:
    def get(self, ids=None, include=None):
        present = [i for i in (ids or list(_CHROMA)) if i in _CHROMA]
        return {"ids": present, "embeddings": [_CHROMA[i] for i in present]}

    def upsert(self, **kw):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema

    db_file = tmp_path / "memory.db"
    monkeypatch.setenv("MAYRING_LOCAL_DB", str(db_file))
    adapter = DBAdapter.create(db_file, check_same_thread=False)
    _init_schema(adapter)
    # WHY(v19-drop-codebook): kein codebooks-INSERT mehr — flache categories-Tabelle.
    for name, emb in [("api", "cb:t:api"), ("domain", "cb:t:domain")]:
        adapter.execute("INSERT INTO categories(name, description, status, "
                        "source, evidence_count, embedding_id) VALUES (?,?, 'active','imported',5,?)",
                        (name, name, emb))
    adapter.commit()
    cb = 1  # synthetischer Wert für URL-Kompatibilität (codebook_id wird ignoriert)
    monkeypatch.setattr(_deps, "_conn", adapter)

    # mock the Ollama providers + chroma collection used by the route
    import mayring_core.providers as providers
    import mayring_core.memory.store as store
    monkeypatch.setattr(providers, "embed_texts",
                        lambda texts, url: [[1.0, 0.0, 0.0] if "api" in texts[0] else [0.0, 0.0, 1.0]])
    # canonical flow reduces FIRST: the LLM-derived label is what gets embedded + matched.
    # Mirror a goal-anchored reduction → a label that reflects the input topic.
    monkeypatch.setattr(providers, "generate_text",
                        lambda **kw: "api_category" if "api" in kw.get("prompt", "") else "novel_concept")
    monkeypatch.setattr(store, "get_chroma_collection", lambda name: _FakeChroma())

    prev = app.dependency_overrides.get(get_workspace)
    app.dependency_overrides[get_workspace] = lambda: "ws-test"
    yield TestClient(app), cb
    if prev is None:
        app.dependency_overrides.pop(get_workspace, None)
    else:
        app.dependency_overrides[get_workspace] = prev
    monkeypatch.setattr(_deps, "_conn", None)


def test_process_empty_text_400(client):
    c, cb = client
    r = c.post(f"/codebooks/{cb}/process", json={"text": "", "task": "x"})
    assert r.status_code == 400


def test_process_any_codebook_id_accepted(client):
    """v19: codebook_id ist vestigial — /process akzeptiert jeden Wert (404 ist weg)."""
    c, _cb = client
    r = c.post("/codebooks/9999/process", json={"text": "api work", "task": "build"})
    # 200 (deductive) oder 400 (leerer text/task) sind ok; 404 ist nicht mehr möglich
    assert r.status_code != 404


def test_process_deductive_200(client):
    c, cb = client
    r = c.post(f"/codebooks/{cb}/process",
               json={"text": "implement the api endpoint", "task": "build api"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "deductive"
    assert body["category_name"] == "api"
    assert body["proposed"] is False


def test_reject_deletes_proposed_category(client):
    c, cb = client
    import src.api.dependencies as _deps
    conn = _deps._conn
    conn.execute("INSERT INTO categories(name, description, status, "
                 "source, evidence_count, embedding_id) VALUES (?,?, 'proposed','induced',1,?)",
                 ("junk_proposed", "x", "cb:proposed:junk"))
    conn.commit()
    cat_id = conn.execute("SELECT id FROM categories WHERE name='junk_proposed'").fetchone()[0]
    conn.execute("INSERT INTO chunk_categories(chunk_id, category_id, codebook_version, confidence, source) "
                 "VALUES ('ck1', ?, 1, 0.5, 'inductive')", (cat_id,))
    conn.commit()

    r = c.post(f"/codebooks/{cb}/proposals/{cat_id}/reject")
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"
    assert conn.execute("SELECT count(*) FROM categories WHERE id=?", (cat_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM chunk_categories WHERE category_id=?", (cat_id,)).fetchone()[0] == 0


def test_reject_refuses_active_category(client):
    """Safety: aktive Kategorien (bedienen cat_match) dürfen NICHT gelöscht werden."""
    c, cb = client
    import src.api.dependencies as _deps
    api_id = _deps._conn.execute("SELECT id FROM categories WHERE name='api'").fetchone()[0]
    r = c.post(f"/codebooks/{cb}/proposals/{api_id}/reject")
    assert r.status_code == 400
