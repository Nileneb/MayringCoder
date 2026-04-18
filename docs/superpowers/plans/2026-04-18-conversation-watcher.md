# Conversation Watcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tools/conversation_watcher.py` — a background daemon that incrementally ingests new Claude conversation turns into MCP memory, with adaptive idle management and post-hook support.

**Architecture:** A polling loop reads only new bytes from JSONL files (byte-offset tracking), buffers new turns per session, and flushes micro-batches to `src.memory.ingest` when a threshold is hit. Adaptive polling slows to 5-minute intervals after 15 minutes of inactivity; Ollama model unloads from VRAM after idle. Post-hooks run throttled via subprocess after each ingest cycle.

**Tech Stack:** Python stdlib (`argparse`, `signal`, `subprocess`, `json`), `src.memory.ingest`, `src.memory.retrieval`, `src.analysis.analyzer._ollama_generate` — shared helpers imported from `tools/ingest_conversations.py`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tools/conversation_watcher.py` | CREATE | Full daemon — state, tailing, buffering, loop, CLI |
| `tests/test_conversation_watcher.py` | CREATE | Unit tests for all testable units |

No other files modified.

---

### Task 1: WatcherState dataclass + persistence

**Files:**
- Create: `tools/conversation_watcher.py` (initial scaffold)
- Test: `tests/test_conversation_watcher.py`

- [ ] **Step 1: Create scaffold + failing tests**

```python
# tests/test_conversation_watcher.py
import json
import sys
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py -v 2>&1 | head -20
```
Expected: ImportError or AttributeError — `conversation_watcher` does not exist yet.

- [ ] **Step 3: Write scaffold + WatcherState**

```python
#!/usr/bin/env python3
"""Incremental conversation watcher — ingests new Claude turns into MCP memory."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

_STATE_FILE = Path.home() / ".cache" / "mayryngcoder" / "watcher_state.json"
_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Re-use helpers from existing batch tool to avoid duplication
from ingest_conversations import (
    _already_ingested,
    _extract_text,
    _keywords,
    _slug,
    _summarize,
    _turns_excerpt,
    _SUMMARIZE_PROMPT,
    _SUMMARIZE_SYSTEM,
)


@dataclass
class WatcherState:
    file_offsets: dict[str, int] = field(default_factory=dict)
    last_hook_run: float = 0.0
    last_llm_call: float = 0.0


def save_state(state: WatcherState, path: Path = _STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "file_offsets": state.file_offsets,
        "last_hook_run": state.last_hook_run,
        "last_llm_call": state.last_llm_call,
    }), encoding="utf-8")


def load_state(path: Path = _STATE_FILE) -> WatcherState:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return WatcherState(
            file_offsets=data.get("file_offsets", {}),
            last_hook_run=float(data.get("last_hook_run", 0.0)),
            last_llm_call=float(data.get("last_llm_call", 0.0)),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return WatcherState()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py::test_state_roundtrip tests/test_conversation_watcher.py::test_load_state_missing_file -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/conversation_watcher.py tests/test_conversation_watcher.py
git commit -m "feat(watcher): add WatcherState + state persistence"
```

---

### Task 2: Incremental JSONL tailing

**Files:**
- Modify: `tools/conversation_watcher.py`
- Modify: `tests/test_conversation_watcher.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_conversation_watcher.py
import json
from tools.conversation_watcher import read_new_turns


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py::test_read_new_turns_empty_offset -v 2>&1 | tail -5
```
Expected: ImportError — `read_new_turns` not defined.

- [ ] **Step 3: Implement `read_new_turns`**

Add after `load_state()` in `conversation_watcher.py`:

```python
def read_new_turns(path: Path, byte_offset: int) -> tuple[list[dict], int]:
    """Read turns added after byte_offset. Returns (new_turns, new_offset)."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return [], byte_offset

    if size <= byte_offset:
        return [], byte_offset

    turns: list[dict] = []
    with path.open("rb") as fh:
        fh.seek(byte_offset)
        for raw in fh:
            try:
                entry = json.loads(raw.decode("utf-8", errors="replace").strip())
            except (json.JSONDecodeError, ValueError):
                continue
            if entry.get("type") not in ("user", "assistant"):
                continue
            msg = entry.get("message", {})
            text = _extract_text(msg.get("content", "")).strip()
            if not text:
                continue
            sid = entry.get("sessionId") or path.stem
            turns.append({
                "role": entry["type"],
                "content": text[:2000],
                "timestamp": entry.get("timestamp", ""),
                "session_id": sid,
            })
    return turns, size
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py -k "read_new_turns" -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/conversation_watcher.py tests/test_conversation_watcher.py
git commit -m "feat(watcher): incremental JSONL tailing with byte-offset tracking"
```

---

### Task 3: Turn buffer + flush trigger

**Files:**
- Modify: `tools/conversation_watcher.py`
- Modify: `tests/test_conversation_watcher.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_conversation_watcher.py
from tools.conversation_watcher import TurnBuffer


def test_flush_by_count():
    buf = TurnBuffer(batch_size=3, flush_interval=300)
    for i in range(3):
        buf.add("sess1", {"role": "user", "content": f"msg{i}", "timestamp": ""})
    assert buf.should_flush("sess1") is True


def test_no_flush_below_count():
    buf = TurnBuffer(batch_size=5, flush_interval=300)
    buf.add("sess1", {"role": "user", "content": "x", "timestamp": ""})
    assert buf.should_flush("sess1") is False


def test_flush_by_time():
    buf = TurnBuffer(batch_size=100, flush_interval=1)
    buf.add("sess1", {"role": "user", "content": "x", "timestamp": ""})
    buf._first_turn_time["sess1"] = time.time() - 2  # simulate 2s ago
    assert buf.should_flush("sess1") is True


def test_pop_clears_buffer():
    buf = TurnBuffer(batch_size=3, flush_interval=300)
    buf.add("sess1", {"role": "user", "content": "x", "timestamp": ""})
    turns = buf.pop("sess1")
    assert len(turns) == 1
    assert buf.should_flush("sess1") is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py -k "flush" -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement `TurnBuffer`**

Add to `conversation_watcher.py`:

```python
class TurnBuffer:
    def __init__(self, batch_size: int = 8, flush_interval: float = 300.0) -> None:
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._turns: dict[str, list[dict]] = defaultdict(list)
        self._first_turn_time: dict[str, float] = {}

    def add(self, session_id: str, turn: dict) -> None:
        if session_id not in self._first_turn_time:
            self._first_turn_time[session_id] = time.time()
        self._turns[session_id].append(turn)

    def should_flush(self, session_id: str) -> bool:
        if not self._turns.get(session_id):
            return False
        if len(self._turns[session_id]) >= self._batch_size:
            return True
        elapsed = time.time() - self._first_turn_time.get(session_id, time.time())
        return elapsed >= self._flush_interval

    def pop(self, session_id: str) -> list[dict]:
        turns = list(self._turns.pop(session_id, []))
        self._first_turn_time.pop(session_id, None)
        return turns

    def sessions_to_flush(self) -> list[str]:
        return [sid for sid in list(self._turns) if self.should_flush(sid)]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py -k "flush or pop" -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/conversation_watcher.py tests/test_conversation_watcher.py
git commit -m "feat(watcher): TurnBuffer with count and time-based flush triggers"
```

---

### Task 4: Micro-batch ingest

**Files:**
- Modify: `tools/conversation_watcher.py`
- Modify: `tests/test_conversation_watcher.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_conversation_watcher.py
from unittest.mock import MagicMock, patch
from tools.conversation_watcher import ingest_micro_batch


def test_ingest_micro_batch_calls_ingest(tmp_path):
    turns = [
        {"role": "user", "content": "Wie funktioniert ingest()?", "timestamp": "t1"},
        {"role": "assistant", "content": "Es chunked und embedded.", "timestamp": "t2"},
    ]
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None  # not yet ingested
    chroma = MagicMock()

    with patch("tools.conversation_watcher.ingest") as mock_ingest, \
         patch("tools.conversation_watcher._summarize", return_value="Summary text"), \
         patch("tools.conversation_watcher.search", return_value=[]):
        mock_ingest.return_value = {"chunk_ids": ["c1", "c2"], "indexed": 2,
                                    "deduped": 0, "superseded": 0}
        result = ingest_micro_batch(
            session_id="sess123",
            turns=turns,
            slug="MayringCoder",
            fpath=Path("fake.jsonl"),
            conn=conn,
            chroma=chroma,
            ollama_url="http://localhost:11434",
            model="llama3.1:8b",
            workspace_id="default",
        )
    assert result["chunks"] == 2


def test_ingest_micro_batch_skips_if_deduped(tmp_path):
    turns = [{"role": "user", "content": "Hello", "timestamp": "t1"}]
    conn = MagicMock()
    # simulate already ingested
    conn.execute.return_value.fetchone.return_value = ("sha256:abc",)
    chroma = MagicMock()
    with patch("tools.conversation_watcher.ingest") as mock_ingest, \
         patch("tools.conversation_watcher._summarize", return_value="x"), \
         patch("tools.conversation_watcher.search", return_value=[]):
        # content_hash matches → should skip
        conn.execute.return_value.fetchone.return_value = ("sha256:skip",)
        result = ingest_micro_batch("sess", turns, "slug", Path("f.jsonl"),
                                    conn, chroma, "http://...", "m", "ws")
    # ingest not called because _already_ingested returns True
    mock_ingest.assert_not_called()
    assert result["chunks"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py -k "ingest_micro_batch" -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement `ingest_micro_batch`**

Add to `conversation_watcher.py`:

```python
import hashlib

from src.memory.ingest import ingest
from src.memory.retrieval import compress_for_prompt, search
from src.memory.schema import Source


def ingest_micro_batch(
    session_id: str,
    turns: list[dict],
    slug: str,
    fpath: Path,
    conn: Any,
    chroma: Any,
    ollama_url: str,
    model: str,
    workspace_id: str,
) -> dict[str, int]:
    """Summarize + ingest a micro-batch of turns. Returns {chunks, skipped}."""
    if not turns:
        return {"chunks": 0, "skipped": 0}

    first_ts = turns[0].get("timestamp", "")[:10]
    source_id = f"conversation:{slug}:{session_id[:16]}"
    raw_key = f"{session_id}:{len(turns)}:{first_ts}:micro"
    content_hash = "sha256:" + hashlib.sha256(raw_key.encode()).hexdigest()[:16]

    if _already_ingested(conn, source_id, content_hash):
        return {"chunks": 0, "skipped": len(turns)}

    # Retrieve related context for richer summary
    related_context = ""
    kw = _keywords(turns)
    if kw:
        try:
            related = search(
                query=kw, conn=conn, chroma_collection=chroma,
                ollama_url=ollama_url, opts={"top_k": 3, "workspace_id": None},
            )
            related_context = compress_for_prompt(related, char_budget=800)
        except Exception:
            pass

    summary = _summarize(turns, related_context, ollama_url, model)
    content = f"# Micro-Batch {first_ts or 'unbekannt'} | {slug}\n\n{summary}"

    src = Source(
        source_id=source_id,
        source_type="conversation_summary",
        repo=slug,
        path=f"{slug}/{fpath.name}",
        branch="local",
        commit="",
        content_hash=content_hash,
    )
    result = ingest(
        src, content, conn, chroma,
        ollama_url, model,
        opts={"categorize": bool(model), "codebook": "social", "mode": "hybrid"},
        workspace_id=workspace_id,
    )
    return {"chunks": len(result.get("chunk_ids", [])), "skipped": 0}
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py -k "ingest_micro_batch" -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/conversation_watcher.py tests/test_conversation_watcher.py
git commit -m "feat(watcher): micro-batch ingest using existing summarization pipeline"
```

---

### Task 5: Ollama model unload + post-hook runner

**Files:**
- Modify: `tools/conversation_watcher.py`
- Modify: `tests/test_conversation_watcher.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_conversation_watcher.py
from unittest.mock import call, patch
from tools.conversation_watcher import unload_model, run_post_hook


def test_unload_model_calls_ollama_stop():
    with patch("subprocess.run") as mock_run:
        unload_model("llama3.1:8b", "http://localhost:11434")
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "ollama" in args
    assert "stop" in args


def test_post_hook_runs_when_interval_elapsed():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        new_time = run_post_hook("echo hi", last_run=0.0, hook_interval=1800)
    mock_run.assert_called_once()
    assert new_time > 0.0


def test_post_hook_skipped_within_interval():
    now = time.time()
    with patch("subprocess.run") as mock_run:
        result = run_post_hook("echo hi", last_run=now - 60, hook_interval=1800)
    mock_run.assert_not_called()
    assert result == now - 60  # unchanged


def test_post_hook_error_isolated():
    with patch("subprocess.run", side_effect=Exception("boom")):
        # Should not raise
        result = run_post_hook("bad_cmd", last_run=0.0, hook_interval=1)
    assert isinstance(result, float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py -k "unload or post_hook" -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement both functions**

Add to `conversation_watcher.py`:

```python
def unload_model(model: str, ollama_url: str) -> None:
    """Unload model from VRAM via ollama stop."""
    try:
        subprocess.run(
            ["ollama", "stop", model],
            timeout=10,
            capture_output=True,
        )
    except Exception:
        pass


def run_post_hook(cmd: str, last_run: float, hook_interval: float) -> float:
    """Run post-hook command if interval elapsed. Returns updated last_run."""
    if time.time() - last_run < hook_interval:
        return last_run
    try:
        subprocess.run(cmd, shell=True, timeout=60, capture_output=True)
    except Exception as exc:
        print(f"[watcher] post-hook error: {exc}", flush=True)
    return time.time()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py -k "unload or post_hook" -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/conversation_watcher.py tests/test_conversation_watcher.py
git commit -m "feat(watcher): Ollama model unload + throttled post-hook runner"
```

---

### Task 6: Main watch loop + adaptive polling

**Files:**
- Modify: `tools/conversation_watcher.py`

- [ ] **Step 1: Implement `watch_loop`**

Add to `conversation_watcher.py` (no test needed — loop is integration-level, tested manually):

```python
_STOP = False


def _handle_signal(sig: int, frame: Any) -> None:
    global _STOP
    print(f"\n[watcher] Signal {sig} received — stopping after current cycle.", flush=True)
    _STOP = True


def scan_workspace_files(workspace_paths: list[Path]) -> list[Path]:
    """Return all JSONL conversation files across workspaces."""
    files: list[Path] = []
    for wp in workspace_paths:
        files.extend(sorted(wp.glob("*.jsonl")))
    return files


def watch_loop(
    workspace_paths: list[Path],
    conn: Any,
    chroma: Any,
    ollama_url: str,
    model: str,
    workspace_id: str,
    poll_interval: float = 30.0,
    idle_timeout: float = 900.0,
    idle_interval: float = 300.0,
    batch_size: int = 8,
    flush_interval: float = 300.0,
    post_hook: str = "",
    hook_interval: float = 1800.0,
    dry_run: bool = False,
    state_path: Path = _STATE_FILE,
) -> None:
    global _STOP

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    state = load_state(state_path)
    buffer = TurnBuffer(batch_size=batch_size, flush_interval=flush_interval)
    last_activity = time.time()
    model_loaded = False
    session_fpath_map: dict[str, Path] = {}  # session_id → source JSONL file

    print(f"[watcher] Started — watching {len(workspace_paths)} workspace(s)", flush=True)

    while not _STOP:
        cycle_start = time.time()
        files = scan_workspace_files(workspace_paths)
        any_new = False

        for fpath in files:
            key = str(fpath)
            offset = state.file_offsets.get(key, 0)
            new_turns, new_offset = read_new_turns(fpath, offset)

            if new_turns:
                any_new = True
                last_activity = time.time()
                for turn in new_turns:
                    sid = turn.pop("session_id", fpath.stem)
                    buffer.add(sid, turn)
                    session_fpath_map[sid] = fpath  # track source file per session
                state.file_offsets[key] = new_offset

        # Flush ready sessions
        for session_id in buffer.sessions_to_flush():
            turns = buffer.pop(session_id)
            slug = _slug(workspace_paths[0]) if workspace_paths else "unknown"

            fpath_for_session = session_fpath_map.get(session_id,
                files[0] if files else Path("unknown.jsonl"))

            if dry_run:
                print(f"[dry] {session_id[:8]} | {len(turns)} turns", flush=True)
                continue

            result = ingest_micro_batch(
                session_id=session_id,
                turns=turns,
                slug=slug,
                fpath=fpath_for_session,
                conn=conn,
                chroma=chroma,
                ollama_url=ollama_url,
                model=model,
                workspace_id=workspace_id,
            )
            state.last_llm_call = time.time()
            model_loaded = True
            ts = time.strftime("%H:%M:%S")
            print(
                f"[watcher] {ts} | {session_id[:8]} | {len(turns)} turns"
                f" → {result['chunks']} chunks",
                flush=True,
            )

            if post_hook:
                state.last_hook_run = run_post_hook(
                    post_hook, state.last_hook_run, hook_interval
                )

        # Unload model after idle_timeout
        if model_loaded and (time.time() - state.last_llm_call) > idle_timeout:
            unload_model(model, ollama_url)
            model_loaded = False
            print("[watcher] Model unloaded (idle)", flush=True)

        save_state(state, state_path)

        # Adaptive sleep
        idle_secs = time.time() - last_activity
        sleep_time = idle_interval if idle_secs > idle_timeout else poll_interval
        if any_new:
            sleep_time = poll_interval  # always fast after activity

        elapsed = time.time() - cycle_start
        remaining = max(0.0, sleep_time - elapsed)
        if not _STOP:
            time.sleep(remaining)

    save_state(state, state_path)
    print("[watcher] Stopped.", flush=True)
```

- [ ] **Step 2: Verify import sanity**

```bash
.venv/bin/python -c "from tools.conversation_watcher import watch_loop, scan_workspace_files; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/conversation_watcher.py
git commit -m "feat(watcher): main watch loop with adaptive polling + signal handlers"
```

---

### Task 7: CLI entry point

**Files:**
- Modify: `tools/conversation_watcher.py`

- [ ] **Step 1: Implement `parse_args` and `main`**

Append to end of `conversation_watcher.py`:

```python
def parse_args() -> Any:
    import argparse
    ap = argparse.ArgumentParser(
        description="Incremental Claude conversation watcher for MCP memory"
    )
    ap.add_argument("--workspace", metavar="PATH",
                    help="Path to a Claude workspace directory")
    ap.add_argument("--all-workspaces", action="store_true",
                    help=f"Watch all workspaces under {_PROJECTS_DIR}")
    ap.add_argument("--workspace-id", default="default",
                    help="MCP workspace/tenant ID (default: default)")
    ap.add_argument("--model", default="",
                    help="Ollama model for summarization (default: OLLAMA_MODEL env)")
    ap.add_argument("--poll-interval", type=float, default=30.0, metavar="SEC",
                    help="Seconds between file checks in active mode (default: 30)")
    ap.add_argument("--idle-timeout", type=float, default=900.0, metavar="SEC",
                    help="Seconds of inactivity before entering idle mode (default: 900)")
    ap.add_argument("--idle-interval", type=float, default=300.0, metavar="SEC",
                    help="Poll interval in idle mode (default: 300)")
    ap.add_argument("--batch-size", type=int, default=8, metavar="N",
                    help="Flush after N new turns per session (default: 8)")
    ap.add_argument("--flush-interval", type=float, default=300.0, metavar="SEC",
                    help="Flush after SEC seconds since first buffered turn (default: 300)")
    ap.add_argument("--post-hook", default="", metavar="CMD",
                    help="Shell command to run after each ingest cycle")
    ap.add_argument("--hook-interval", type=float, default=1800.0, metavar="SEC",
                    help="Minimum seconds between post-hook calls (default: 1800)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be ingested without actually doing it")
    return ap.parse_args()


def main() -> None:
    import os
    args = parse_args()

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = args.model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # Resolve workspace paths
    if args.all_workspaces:
        workspace_paths = [p for p in _PROJECTS_DIR.iterdir() if p.is_dir()]
    elif args.workspace:
        workspace_paths = [Path(args.workspace).expanduser()]
    else:
        print("Error: provide --workspace PATH or --all-workspaces", file=sys.stderr)
        sys.exit(1)

    from src.memory.ingest import get_or_create_chroma_collection
    from src.memory.store import init_memory_db

    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()

    watch_loop(
        workspace_paths=workspace_paths,
        conn=conn,
        chroma=chroma,
        ollama_url=ollama_url,
        model=model,
        workspace_id=args.workspace_id,
        poll_interval=args.poll_interval,
        idle_timeout=args.idle_timeout,
        idle_interval=args.idle_interval,
        batch_size=args.batch_size,
        flush_interval=args.flush_interval,
        post_hook=args.post_hook,
        hook_interval=args.hook_interval,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `--help` works**

```bash
.venv/bin/python tools/conversation_watcher.py --help 2>&1 | head -15
```
Expected: argparse help text with all listed flags.

- [ ] **Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest tests/test_conversation_watcher.py -v 2>&1 | tail -10
```
Expected: All tests pass.

```bash
.venv/bin/python -m pytest tests/ -q --tb=short 2>&1 | tail -3
```
Expected: 671+ passed, 1 pre-existing failure.

- [ ] **Step 4: Commit + push**

```bash
git add tools/conversation_watcher.py
git commit -m "feat(watcher): CLI entry point with all flags — closes #52"
git push origin master
```

- [ ] **Step 5: Close issue**

```bash
gh issue close 52 --repo Nileneb/MayringCoder --comment "Implementiert in tools/conversation_watcher.py (Commit: $(git rev-parse --short HEAD)). Alle Akzeptanzkriterien erfüllt."
```

---

## Verification Checklist (Acceptance Criteria)

| Criterion | How to test |
|---|---|
| Turns within ~60s | Run watcher + make a Claude turn + check DB after 60s |
| Idle CPU ≈ 0 | `top` while watcher sleeps in idle mode |
| Ollama unloads after idle | `ollama ps` after idle_timeout elapses |
| No duplicates | Run `ingest_conversations.py` after watcher — no re-ingest |
| Clean shutdown | `kill -TERM <pid>` → watcher prints "Stopped." |
| CLI flags | `--help` shows all flags |
| Post-hook runs | `--post-hook "touch /tmp/hook_ran"` + verify file |
| Hook error isolated | `--post-hook "exit 1"` → watcher keeps running |
| Logging | Each micro-batch prints timestamp + turns + chunks |
