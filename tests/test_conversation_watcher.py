import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.conversation_watcher import WatcherState, load_state, save_state, read_new_turns, TurnBuffer


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
