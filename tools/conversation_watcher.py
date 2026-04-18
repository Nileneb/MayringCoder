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


def ingest_micro_batch(
    turns: list[dict],
    session_id: str,
    workspace_slug: str,
    conn: Any,
    chroma: Any,
    ollama_url: str,
    model: str,
    workspace_id: str = "system",
) -> bool:
    """Summarize and ingest a micro-batch of new turns into MCP memory.

    Returns True on success, False on error.
    """
    if not turns:
        return False

    import hashlib
    from src.memory.ingest import ingest
    from src.memory.schema import Source

    first_ts = turns[0].get("timestamp", "")[:10]
    batch_key = f"{session_id}:{len(turns)}:{turns[-1].get('timestamp','')}"
    content_hash = "sha256:" + hashlib.sha256(batch_key.encode()).hexdigest()[:16]
    source_id = f"conversation:{workspace_slug}:{session_id[:16]}"

    if _already_ingested(conn, source_id, content_hash):
        return False

    try:
        related_context = ""
        kw = _keywords(turns)
        if kw:
            try:
                from src.memory.retrieval import compress_for_prompt, search
                related = search(
                    query=kw,
                    conn=conn,
                    chroma_collection=chroma,
                    ollama_url=ollama_url,
                    opts={"top_k": 3, "workspace_id": None},
                )
                related_context = compress_for_prompt(related, char_budget=800)
            except Exception:
                pass

        summary = _summarize(turns, related_context, ollama_url, model)

        content = (
            f"# Session {first_ts or 'unbekannt'} | {workspace_slug}\n\n"
            f"{summary}\n"
        )

        src = Source(
            source_id=source_id,
            source_type="conversation_summary",
            repo=workspace_slug,
            path=f"{workspace_slug}/incremental",
            branch="local",
            commit="",
            content_hash=content_hash,
        )
        ingest(
            src, content, conn, chroma,
            ollama_url, model,
            opts={"categorize": bool(model), "codebook": "social", "mode": "hybrid"},
            workspace_id=workspace_id,
        )
        return True
    except Exception as exc:
        print(f"  ✗ ingest_micro_batch {session_id[:8]}: {exc}", file=sys.stderr)
        return False


def unload_model(model: str) -> None:
    """Tell Ollama to unload the model from VRAM (fire-and-forget)."""
    if not model:
        return
    try:
        subprocess.run(
            ["ollama", "stop", model],
            timeout=10,
            capture_output=True,
        )
    except Exception:
        pass


def run_post_hook(
    hook_cmd: str,
    state: WatcherState,
    hook_interval: float,
) -> None:
    """Run shell post-hook if hook_interval has passed since last run."""
    if not hook_cmd:
        return
    now = time.time()
    if now - state.last_hook_run < hook_interval:
        return
    try:
        subprocess.run(hook_cmd, shell=True, timeout=30, capture_output=True)
        state.last_hook_run = now
    except Exception:
        pass
