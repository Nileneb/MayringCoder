#!/usr/bin/env python3
"""Incremental conversation watcher — ingests new Claude turns into MCP memory."""

from __future__ import annotations

import json
import os
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
# Overridable via env so the Docker service (Issue #52) can mount a different
# host path at /host_claude without patching the default.
_PROJECTS_DIR = (
    Path(os.environ["CLAUDE_PROJECTS_DIR"])
    if os.environ.get("CLAUDE_PROJECTS_DIR")
    else Path.home() / ".claude" / "projects"
)

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


def scan_workspace_files(projects_dir: Path = _PROJECTS_DIR) -> list[Path]:
    """Return all *.jsonl files under projects_dir (non-recursive first level)."""
    if not projects_dir.exists():
        return []
    result: list[Path] = []
    for ws_dir in sorted(projects_dir.iterdir()):
        if ws_dir.is_dir():
            result.extend(sorted(ws_dir.glob("*.jsonl")))
    return result


def watch_loop(
    ollama_url: str,
    model: str,
    workspace_id: str = "system",
    poll_interval: float = 30.0,
    idle_interval: float = 300.0,
    idle_after: float = 900.0,
    flush_count: int = 10,
    flush_interval: float = 120.0,
    hook_cmd: str = "",
    hook_interval: float = 300.0,
    idle_timeout: float = 3600.0,
    projects_dir: Path = _PROJECTS_DIR,
) -> None:
    """Main polling loop. Runs until SIGINT/SIGTERM or idle_timeout exceeded."""
    from src.api.dependencies import get_conn, get_chroma

    conn = get_conn()
    chroma = get_chroma()
    state = load_state()
    buf = TurnBuffer(flush_count=flush_count, flush_interval=flush_interval)

    stop_flag = {"stop": False}

    def _handle_signal(sig, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    last_activity = time.time()

    while not stop_flag["stop"]:
        now = time.time()

        if model and (now - state.last_llm_call) > idle_timeout and state.last_llm_call > 0:
            unload_model(model)
            state.last_llm_call = 0.0

        ws_slug = "default"
        for fpath in scan_workspace_files(projects_dir):
            key = str(fpath)
            offset = state.file_offsets.get(key, 0)
            new_turns, new_offset = read_new_turns(fpath, offset)
            if new_turns:
                last_activity = now
                state.file_offsets[key] = new_offset
                ws_slug = _slug(fpath.parent)
                for turn in new_turns:
                    buf.add(turn)

        for sid in buf.sessions_to_flush():
            turns = buf.pop(sid)
            ok = ingest_micro_batch(
                turns, sid, ws_slug,
                conn, chroma, ollama_url, model, workspace_id,
            )
            if ok:
                state.last_llm_call = time.time()
                run_post_hook(hook_cmd, state, hook_interval)

        save_state(state)

        idle = (time.time() - last_activity) > idle_after
        sleep_time = idle_interval if idle else poll_interval
        for _ in range(int(sleep_time)):
            if stop_flag["stop"]:
                break
            time.sleep(1)


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Incremental Claude conversation watcher — ingests new turns into MCP memory."
    )
    ap.add_argument("--ollama-url", default=None,
                    help="Ollama API URL (default: $OLLAMA_URL or http://localhost:11434)")
    ap.add_argument("--model", default=None,
                    help="Ollama model (default: $OLLAMA_MODEL)")
    ap.add_argument("--workspace-id", default="system",
                    help="MCP workspace ID (default: system)")
    ap.add_argument("--poll-interval", type=float, default=30.0,
                    help="Seconds between polls when active (default: 30)")
    ap.add_argument("--idle-interval", type=float, default=300.0,
                    help="Seconds between polls when idle (default: 300)")
    ap.add_argument("--idle-after", type=float, default=900.0,
                    help="Seconds of inactivity before switching to idle polling (default: 900)")
    ap.add_argument("--flush-count", type=int, default=10,
                    help="Flush session buffer after N turns (default: 10)")
    ap.add_argument("--flush-interval", type=float, default=120.0,
                    help="Flush session buffer after N seconds (default: 120)")
    ap.add_argument("--hook-cmd", default="",
                    help="Shell command to run after each ingest (post-hook)")
    ap.add_argument("--hook-interval", type=float, default=300.0,
                    help="Minimum seconds between post-hook runs (default: 300)")
    ap.add_argument("--idle-timeout", type=float, default=3600.0,
                    help="Seconds of LLM inactivity before unloading model (default: 3600)")
    ap.add_argument("--projects-dir", default=None,
                    help=f"Claude projects dir (default: {_PROJECTS_DIR})")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args(argv)

    ollama_url = args.ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = args.model or os.getenv("OLLAMA_MODEL", "")
    projects_dir = Path(args.projects_dir) if args.projects_dir else _PROJECTS_DIR

    print(f"[watcher] Ollama: {ollama_url} | Model: {model or '(none)'}")
    print(f"[watcher] Projects dir: {projects_dir}")
    print(f"[watcher] Poll: {args.poll_interval}s active / {args.idle_interval}s idle")

    watch_loop(
        ollama_url=ollama_url,
        model=model,
        workspace_id=args.workspace_id,
        poll_interval=args.poll_interval,
        idle_interval=args.idle_interval,
        idle_after=args.idle_after,
        flush_count=args.flush_count,
        flush_interval=args.flush_interval,
        hook_cmd=args.hook_cmd,
        hook_interval=args.hook_interval,
        idle_timeout=args.idle_timeout,
        projects_dir=projects_dir,
    )


if __name__ == "__main__":
    main()
