from pathlib import Path

from mayring_core.memory.store import init_memory_db
from mayring_core.memory.tasks import create_task, list_tasks


def test_external_id_upserts_not_duplicates(tmp_path: Path):
    conn = init_memory_db(tmp_path / "m.db")
    ws = "ws-1"
    t1 = create_task(conn, workspace_id=ws, title="do X", created_by="agent",
                     external_id="harness-42")
    t2 = create_task(conn, workspace_id=ws, title="do X (renamed)", created_by="agent",
                     external_id="harness-42")
    rows = list_tasks(conn, ws)
    assert len(rows) == 1, "same external_id must update, not duplicate"
    assert t1["task_id"] == t2["task_id"]
    assert t2["title"] == "do X (renamed)"


def test_no_external_id_always_inserts(tmp_path: Path):
    conn = init_memory_db(tmp_path / "m.db")
    create_task(conn, workspace_id="ws", title="a", created_by="agent")
    create_task(conn, workspace_id="ws", title="a", created_by="agent")
    assert len(list_tasks(conn, "ws")) == 2
