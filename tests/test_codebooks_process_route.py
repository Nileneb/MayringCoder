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
    now = "2026-05-24T00:00:00Z"
    adapter.execute("INSERT INTO codebooks(slug, description, version, auto_promote_threshold, "
                    "created_at, updated_at) VALUES ('t','test',1,3,?,?)", (now, now))
    cb = adapter.execute("SELECT id FROM codebooks WHERE slug='t'").fetchone()[0]
    for name, emb in [("api", "cb:t:api"), ("domain", "cb:t:domain")]:
        adapter.execute("INSERT INTO codebook_categories(codebook_id, name, description, status, "
                        "source, evidence_count, embedding_id) VALUES (?,?,?, 'active','imported',5,?)",
                        (cb, name, name, emb))
    adapter.commit()
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


def test_process_unknown_codebook_404(client):
    c, _cb = client
    r = c.post("/codebooks/9999/process", json={"text": "api work", "task": "build"})
    assert r.status_code == 404


def test_process_deductive_200(client):
    c, cb = client
    r = c.post(f"/codebooks/{cb}/process",
               json={"text": "implement the api endpoint", "task": "build api"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "deductive"
    assert body["category_name"] == "api"
    assert body["proposed"] is False
