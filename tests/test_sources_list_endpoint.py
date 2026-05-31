import asyncio
from src.api.routes import sources_list as sl
from src.api.jwt_auth import TokenInfo


def test_lists_workspace_sources_with_visibility(tmp_path, monkeypatch):
    from mayring_core.memory import store
    from mayring_core.memory.store import upsert_source
    from mayring_core.memory.schema import Source
    conn = store.init_memory_db(tmp_path / "m.db")
    upsert_source(conn, Source(source_id="note:a", source_type="note", repo="r", path="p", visibility="private", user_id="1"), workspace_id="ws-1")
    upsert_source(conn, Source(source_id="note:b", source_type="note", repo="r", path="p", visibility="public"), workspace_id="ws-1")
    upsert_source(conn, Source(source_id="note:other", source_type="note", repo="r", path="p", visibility="private", user_id="1"), workspace_id="ws-2")
    monkeypatch.setattr(sl, "_get_conn", lambda: conn)
    res = asyncio.run(sl.list_sources_endpoint(workspace_id="ws-1", info=TokenInfo(workspace_id="ws-1", scopes=("admin",))))
    ids = {s["source_id"]: s for s in res["sources"]}
    assert set(ids) == {"note:a", "note:b"}
    assert ids["note:b"]["visibility"] == "public"
