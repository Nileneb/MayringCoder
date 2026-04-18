import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.conversation_watcher import WatcherState, load_state, save_state


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
    """Test reading turns when offset is 0 (start from beginning)."""
    from tools.conversation_watcher import read_new_turns

    jsonl_file = tmp_path / "test.jsonl"
    jsonl_file.write_text('{"role": "user", "content": "hello"}\n{"role": "assistant", "content": "hi"}\n')

    turns = read_new_turns(str(jsonl_file), offset=0)
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"


def test_read_new_turns_from_offset(tmp_path):
    """Test reading turns from a specific byte offset."""
    from tools.conversation_watcher import read_new_turns

    jsonl_file = tmp_path / "test.jsonl"
    line1 = '{"role": "user", "content": "hello"}\n'
    line2 = '{"role": "assistant", "content": "hi"}\n'
    jsonl_file.write_text(line1 + line2)

    offset = len(line1.encode("utf-8"))
    turns = read_new_turns(str(jsonl_file), offset=offset)
    assert len(turns) == 1
    assert turns[0]["role"] == "assistant"


def test_read_new_turns_no_growth(tmp_path):
    """Test reading turns when file has not grown past offset."""
    from tools.conversation_watcher import read_new_turns

    jsonl_file = tmp_path / "test.jsonl"
    line1 = '{"role": "user", "content": "hello"}\n'
    jsonl_file.write_text(line1)

    offset = len(line1.encode("utf-8"))
    turns = read_new_turns(str(jsonl_file), offset=offset)
    assert len(turns) == 0
