#!/usr/bin/env python3
"""Stop hook — captures the last user/assistant turn into Memory.

Reads `transcript_path` from stdin (Claude Code Stop-hook payload), pulls the
last user turn + last assistant turn out of the JSONL, and POSTs them to
`/conversation/micro-batch`. The server-side summariser condenses the pair,
dedups via deterministic source_id (`conversation:<workspace>:<session>`),
and ingests as `source_type=conversation_summary`. Closes the gap between
PostCompact events: Memory sees every completed turn, not just compaction
boundaries.

Fire-and-forget (always exits 0). Workspace slug derives from CWD basename.

History note: an earlier version of this hook also auto-rated every injected
chunk as `signal=neutral`. That was strictly harmful — neutral entries are
indistinguishable from "no feedback" in scoring (both yield 0.5) but actively
dilute real signals (1 positive + 10 neutral → 0.545 instead of 1.0). Removed.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

_JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")
_API_URL = os.environ.get("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
_TIMEOUT = 5

_MAX_TURN_CHARS = 4000      # truncate per-turn content fed to the server
_TURN_PAIR_LIMIT = 2        # one user + one assistant turn


def _read_token() -> str:
    try:
        with open(_JWT_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _read_payload() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return {}


def _workspace_slug() -> str:
    return os.path.basename(os.getcwd()).lower() or "default"


def extract_last_turn_pair(transcript_path: str) -> list[dict]:
    """Read the JSONL transcript and return [last_user_turn, last_assistant_turn].

    Each entry is a dict with `role`, `content`, `timestamp`. Skips meta-rows
    (`type` not in {"user","assistant"}). Content is flattened from Claude
    Code's structured `message.content` (list of blocks) to plain text.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return []
    last_user: dict | None = None
    last_assistant: dict | None = None
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                t = row.get("type")
                if t not in ("user", "assistant"):
                    continue
                msg = row.get("message") or {}
                role = msg.get("role") or t
                content = _flatten_content(msg.get("content"))
                if not content.strip():
                    continue
                turn = {
                    "role": role,
                    "content": content[:_MAX_TURN_CHARS],
                    "timestamp": row.get("timestamp", ""),
                }
                if role == "user":
                    last_user = turn
                elif role == "assistant":
                    last_assistant = turn
    except OSError:
        return []
    out = []
    if last_user:
        out.append(last_user)
    if last_assistant:
        out.append(last_assistant)
    return out[-_TURN_PAIR_LIMIT:]


def _flatten_content(content) -> str:
    """Coerce Claude Code's structured content to a flat string.

    Accepts: str | list[dict|str] | None. Tool-use/tool-result blocks are
    skipped — their JSON args are noisy and rarely useful for memory.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(str(block.get("text", "")))
                elif btype in ("thinking", "redacted_thinking"):
                    continue
        return "\n".join(p for p in parts if p)
    return str(content)


def _post_micro_batch(turns: list[dict], session_id: str, workspace_slug: str, token: str) -> int:
    payload = json.dumps({
        "turns": turns,
        "session_id": session_id,
        "workspace_slug": workspace_slug,
    }).encode()
    req = urllib.request.Request(
        f"{_API_URL}/conversation/micro-batch",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=_TIMEOUT)
        return 200
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def _capture_turns(payload: dict, token: str) -> None:
    """Best-effort: ingest the last user/assistant turn pair into Memory."""
    transcript_path = payload.get("transcript_path", "")
    session_id = payload.get("session_id", "") or "unknown"
    if not transcript_path:
        return
    turns = extract_last_turn_pair(transcript_path)
    if len(turns) < 2:
        return  # need at least one user + one assistant turn
    _post_micro_batch(turns, session_id, _workspace_slug(), token)


def main() -> None:
    token = _read_token()
    if not token:
        return
    payload = _read_payload()
    try:
        _capture_turns(payload, token)
    except Exception:
        pass


if __name__ == "__main__":
    main()
