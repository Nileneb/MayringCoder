import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.conversation_watcher import WatcherState, load_state, save_state, read_new_turns, TurnBuffer, ingest_micro_batch, unload_model, run_post_hook, scan_workspace_files


def test_state_roundtrip(tmp_path):
    state = WatcherState(
        file_offsets={"/foo/bar.jsonl": 1234},
        last_hook_run=9999.0,
        last_llm_call=8888.0,
    )
    p = tmp_path / "state.json"
    save_state(state, p)
    loaded = load_state(p)
    assert loaded.file_offsets == {"/foo/bar.jsonl": 1234}
    assert loaded.last_hook_run == 9999.0
    assert loaded.last_llm_call == 8888.0


def test_load_state_missing_file(tmp_path):
    state = load_state(tmp_path / "nonexistent.json")
    assert state.file_offsets == {}
    assert state.last_hook_run == 0.0


def test_read_new_turns_empty_offset(tmp_path):
    p = tmp_path / "session.jsonl"
    turns_data = [
        {"type": "user", "sessionId": "abc", "message": {"content": "Hello"},
         "timestamp": "2026-01-01T00:00:00Z"},
        {"type": "assistant", "sessionId": "abc", "message": {"content": "Hi there"},
         "timestamp": "2026-01-01T00:00:01Z"},
    ]
    p.write_text("\n".join(json.dumps(t) for t in turns_data) + "\n")
    turns, new_offset = read_new_turns(p, 0)
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert new_offset == p.stat().st_size


def test_read_new_turns_from_offset(tmp_path):
    p = tmp_path / "session.jsonl"
    line1 = json.dumps({"type": "user", "sessionId": "abc",
                        "message": {"content": "First"}, "timestamp": "t1"}) + "\n"
    p.write_text(line1)
    offset1 = len(line1.encode())

    line2 = json.dumps({"type": "assistant", "sessionId": "abc",
                        "message": {"content": "Second"}, "timestamp": "t2"}) + "\n"
    with p.open("a") as f:
        f.write(line2)

    turns, new_offset = read_new_turns(p, offset1)
    assert len(turns) == 1
    assert turns[0]["role"] == "assistant"
    assert new_offset > offset1


def test_read_new_turns_no_growth(tmp_path):
    p = tmp_path / "s.jsonl"
    line = json.dumps({"type": "user", "sessionId": "abc",
                       "message": {"content": "x"}, "timestamp": "t"}) + "\n"
    p.write_text(line)
    size = p.stat().st_size
    turns, new_offset = read_new_turns(p, size)
    assert turns == []
    assert new_offset == size


def test_flush_by_count():
    buf = TurnBuffer(flush_count=3, flush_interval=9999.0)
    for i in range(3):
        buf.add({"session_id": "s1", "role": "user", "content": f"msg{i}", "timestamp": ""})
    assert buf.should_flush("s1") is True


def test_no_flush_below_count():
    buf = TurnBuffer(flush_count=5, flush_interval=9999.0)
    buf.add({"session_id": "s1", "role": "user", "content": "x", "timestamp": ""})
    assert buf.should_flush("s1") is False


def test_flush_by_time(monkeypatch):
    buf = TurnBuffer(flush_count=100, flush_interval=1.0)
    buf.add({"session_id": "s1", "role": "user", "content": "x", "timestamp": ""})
    # simulate time passing — monkeypatch time.time
    monkeypatch.setattr("tools.conversation_watcher.time.time", lambda: buf._first_seen["s1"] + 2.0)
    assert buf.should_flush("s1") is True


def test_pop_clears_buffer():
    buf = TurnBuffer(flush_count=2, flush_interval=9999.0)
    buf.add({"session_id": "s2", "role": "user", "content": "a", "timestamp": ""})
    buf.add({"session_id": "s2", "role": "assistant", "content": "b", "timestamp": ""})
    turns = buf.pop("s2")
    assert len(turns) == 2
    assert buf._sessions.get("s2") is None
    assert buf.should_flush("s2") is False


def test_ingest_micro_batch_empty():
    """Empty turns returns False immediately."""
    result = ingest_micro_batch([], "sid", "slug", MagicMock(), MagicMock(), "url", "model")
    assert result is False


def test_ingest_micro_batch_already_ingested():
    """Already-ingested batch returns False."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("sha256:abc",)
    turns = [{"session_id": "s1", "role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:00Z"}]
    with patch("tools.conversation_watcher._already_ingested", return_value=True):
        result = ingest_micro_batch(turns, "s1", "slug", conn, MagicMock(), "url", "model")
    assert result is False


def test_unload_model_calls_ollama():
    """unload_model calls subprocess.run with ollama stop."""
    with patch("tools.conversation_watcher.subprocess.run") as mock_run:
        unload_model("llama3")
        mock_run.assert_called_once_with(
            ["ollama", "stop", "llama3"],
            timeout=10,
            capture_output=True,
        )


def test_unload_model_empty_skips():
    """unload_model with empty string does not call subprocess."""
    with patch("tools.conversation_watcher.subprocess.run") as mock_run:
        unload_model("")
        mock_run.assert_not_called()


def test_run_post_hook_executes_when_due():
    """run_post_hook executes shell command when interval has passed."""
    state = WatcherState(last_hook_run=0.0)
    with patch("tools.conversation_watcher.subprocess.run") as mock_run, \
         patch("tools.conversation_watcher.time.time", return_value=1000.0):
        run_post_hook("echo test", state, hook_interval=60.0)
        mock_run.assert_called_once()
        assert state.last_hook_run == 1000.0


def test_run_post_hook_skips_when_not_due():
    """run_post_hook skips execution when interval has not passed."""
    state = WatcherState(last_hook_run=999.0)
    with patch("tools.conversation_watcher.subprocess.run") as mock_run, \
         patch("tools.conversation_watcher.time.time", return_value=1000.0):
        run_post_hook("echo test", state, hook_interval=60.0)
        mock_run.assert_not_called()


def test_run_post_hook_empty_cmd_skips():
    """run_post_hook with empty command does not call subprocess."""
    state = WatcherState(last_hook_run=0.0)
    with patch("tools.conversation_watcher.subprocess.run") as mock_run:
        run_post_hook("", state, hook_interval=0.0)
        mock_run.assert_not_called()


def test_scan_workspace_files_empty(tmp_path):
    result = scan_workspace_files(tmp_path)
    assert result == []


def test_scan_workspace_files_finds_jsonl(tmp_path):
    ws = tmp_path / "workspace1"
    ws.mkdir()
    (ws / "session1.jsonl").write_text("")
    (ws / "session2.jsonl").write_text("")
    (ws / "notajsonl.txt").write_text("")
    result = scan_workspace_files(tmp_path)
    assert len(result) == 2
    assert all(p.suffix == ".jsonl" for p in result)


def test_scan_workspace_files_nonexistent(tmp_path):
    result = scan_workspace_files(tmp_path / "doesnotexist")
    assert result == []
