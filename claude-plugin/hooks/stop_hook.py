#!/usr/bin/env python3
"""Stop hook — captures every turn pair AND auto-rates injected chunks.

Two responsibilities, both fire-and-forget (exit 0 always):

1. **Turn capture** — POST the last user/assistant pair to
   /conversation/micro-batch, server-side summariser dedups via
   `conversation:<workspace>:<session>`. Closes the gap between /compact
   events; Memory sees every completed turn.

2. **Auto-feedback** — `memory_inject` (UserPromptSubmit hook) writes a
   block of `chk_xxx : source_id` lines into the prompt context. After
   the assistant has answered, Stop parses those lines back out and
   classifies each chunk:

       positive  → the source's path/basename appears in the assistant's
                   answer (≥5 chars, the path was actually used)
       negative  → injected but never referenced

   That's the auto-feedback that should have run on every memory
   injection from the start. Heuristic, not perfect — but a real signal
   instead of nothing, and it costs zero LLM calls.

Workspace slug derives from CWD basename.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

_JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")
_API_URL = os.environ.get("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
_TIMEOUT = 10  # micro-batch summarises a turn pair on the server (LLM call)
               # — was 5s, frequently hit the deadline mid-summary and silently
               # dropped the turn. 10s buys headroom without blocking Stop.

_MAX_TURN_CHARS = 4000      # truncate per-turn content fed to the server
_TURN_PAIR_LIMIT = 2        # one user + one assistant turn

_AUTO_FEEDBACK_LIMIT = 8    # max chunks to rate per turn
_PATH_KEY_MIN_LEN = 5       # avoid spurious matches on tiny basenames

# Pairs emitted by memory_inject as `- \`chk_xxx\` : \`<source_id>\``
_CHUNK_LINE_RE = re.compile(r"`(chk_[a-f0-9]{16})`\s*:\s*`([^`]+)`")


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


def extract_injected_chunks(user_text: str) -> list[tuple[str, str]]:
    """Pull `(chunk_id, source_id)` pairs out of the memory_inject hint block.

    Idempotent on multiple matches — only the first occurrence per chunk_id
    is kept (later inject blocks within the same prompt would be unusual).
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for cid, sid in _CHUNK_LINE_RE.findall(user_text or ""):
        if cid in seen:
            continue
        seen.add(cid)
        pairs.append((cid, sid))
    return pairs


def classify_chunk_relevance(source_id: str, assistant_text: str) -> str:
    """positive iff the chunk's path/basename appears in the assistant's answer.

    The match is intentionally simple: split source_id on the last colon to
    get the path tail, then check substring + basename. Examples that match
    a chunk like `repo:https://github.com/x/y:src/agents/pi.py`:

      • assistant mentions `src/agents/pi.py`     → positive
      • assistant mentions `pi.py`                → positive (len > 5)
      • assistant mentions `agents/`              → positive (path substring)
      • assistant says only "the pi agent runs…"  → negative

    Heuristic, not semantic. Good enough as a signal — far better than the
    neutral-everything autorater it replaces.
    """
    if not source_id or not assistant_text:
        return "negative"
    path_key = source_id.rsplit(":", 1)[-1]
    if path_key and len(path_key) >= _PATH_KEY_MIN_LEN and path_key in assistant_text:
        return "positive"
    basename = path_key.rsplit("/", 1)[-1] if "/" in path_key else path_key
    if basename and len(basename) >= _PATH_KEY_MIN_LEN and basename in assistant_text:
        return "positive"
    return "negative"


def _post_feedback(chunk_id: str, signal: str, token: str) -> None:
    payload = json.dumps({"chunk_id": chunk_id, "signal": signal}).encode()
    req = urllib.request.Request(
        f"{_API_URL}/memory/feedback",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=_TIMEOUT)
    except Exception:
        pass


def _capture_turns(payload: dict, token: str) -> list[dict]:
    """Best-effort: ingest the last user/assistant turn pair into Memory.

    Returns the extracted turn pair so the auto-feedback step can reuse it
    without re-reading the transcript file.
    """
    transcript_path = payload.get("transcript_path", "")
    session_id = payload.get("session_id", "") or "unknown"
    if not transcript_path:
        return []
    turns = extract_last_turn_pair(transcript_path)
    if len(turns) < 2:
        return turns
    _post_micro_batch(turns, session_id, _workspace_slug(), token)
    return turns


def _auto_feedback(turns: list[dict], token: str) -> None:
    """Rate every chunk that memory_inject announced for this prompt."""
    if len(turns) < 2:
        return
    user_text = turns[0].get("content", "")
    assistant_text = turns[1].get("content", "")
    pairs = extract_injected_chunks(user_text)
    for chunk_id, source_id in pairs[:_AUTO_FEEDBACK_LIMIT]:
        signal = classify_chunk_relevance(source_id, assistant_text)
        _post_feedback(chunk_id, signal, token)


def main() -> None:
    token = _read_token()
    if not token:
        return
    payload = _read_payload()
    try:
        turns = _capture_turns(payload, token)
    except Exception:
        turns = []
    try:
        _auto_feedback(turns, token)
    except Exception:
        pass


if __name__ == "__main__":
    main()
