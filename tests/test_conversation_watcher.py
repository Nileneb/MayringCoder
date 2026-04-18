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
