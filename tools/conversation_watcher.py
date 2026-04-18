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
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return WatcherState()


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


@dataclass
class TurnBuffer:
    flush_count: int = 10
    flush_interval: float = 120.0
    _sessions: dict[str, list[dict]] = field(default_factory=dict)
    _first_seen: dict[str, float] = field(default_factory=dict)

    def add(self, turn: dict) -> None:
        """Add a turn to the buffer. turn must have 'session_id' key."""
        sid = turn["session_id"]
        if sid not in self._sessions:
            self._sessions[sid] = []
            self._first_seen[sid] = time.time()
        self._sessions[sid].append(turn)

    def should_flush(self, session_id: str) -> bool:
        """True if session has enough turns OR has been waiting too long."""
        turns = self._sessions.get(session_id, [])
        if not turns:
            return False
        if len(turns) >= self.flush_count:
            return True
        age = time.time() - self._first_seen.get(session_id, time.time())
        return age >= self.flush_interval

    def pop(self, session_id: str) -> list[dict]:
        """Remove and return all buffered turns for session_id."""
        turns = self._sessions.pop(session_id, [])
        self._first_seen.pop(session_id, None)
        return turns

    def sessions_to_flush(self) -> list[str]:
        """Return list of session_ids that should be flushed now."""
        return [sid for sid in list(self._sessions) if self.should_flush(sid)]
