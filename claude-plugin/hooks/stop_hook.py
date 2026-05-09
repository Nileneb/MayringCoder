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

# Local fallback queue — when /memory/feedback fails after retries (deploy
# window, network hiccup, ANY 5xx), the entry gets appended here.
# session_start.py::_drain_feedback_queue() replays everything to REST on
# the next SessionStart. Successful replay → entry removed. 4xx → dropped
# (no retry will fix it). 5xx → stays for next session.
_FEEDBACK_QUEUE = os.path.expanduser("~/.config/mayring/feedback_queue.jsonl")
# Same idea for /conversation/micro-batch — when the server is down or
# slow, capture-events get queued here. Schema: full POST body of the
# micro-batch endpoint. Drain in session_start::_drain_ingest_queue.
_INGEST_QUEUE = os.path.expanduser("~/.config/mayring/ingest_queue.jsonl")
               # — was 5s, frequently hit the deadline mid-summary and silently
               # dropped the turn. 10s buys headroom without blocking Stop.

_MAX_TURN_CHARS = 4000      # truncate per-turn content fed to the server
_TURN_PAIR_LIMIT = 2        # one user + one assistant turn

_AUTO_FEEDBACK_LIMIT = 8    # max chunks to rate per turn
_PATH_KEY_MIN_LEN = 5       # avoid spurious matches on tiny basenames

# memory_inject persists per-session (chunk_id, source_id) pairs in this
# directory because the inject block isn't part of the user-turn content
# in the transcript JSONL — without the file, the Stop hook would never
# see what was injected. See memory_inject._write_inject_state.
_INJECT_STATE_DIR = os.path.expanduser("~/.config/mayring/inject-state")

# Legacy regex kept for transcripts that DO contain the block inline
# (e.g. paste-ins, debugging). New canonical source is the state file.
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
    # Retry on 502/503/504 + queue on persistent failure. Same pattern
    # as _post_feedback. Without this, every deploy window dropped a
    # turn-pair from Memory.
    import time as _time
    body_str = payload.decode("utf-8")
    backoff = 0.6
    for attempt in range(3):
        try:
            urllib.request.urlopen(req, timeout=_TIMEOUT)
            return 200
        except urllib.error.HTTPError as e:
            if e.code in (502, 503, 504) and attempt < 2:
                _time.sleep(backoff)
                backoff *= 2
                continue
            if 400 <= e.code < 500:
                sys.stderr.write(
                    f"[stop_hook] micro-batch HTTP {e.code} (dropped, no retry)\n"
                )
                return e.code
            _enqueue_ingest(body_str, f"http_{e.code}")
            sys.stderr.write(
                f"[stop_hook] micro-batch HTTP {e.code} → queued for replay\n"
            )
            return e.code
        except TimeoutError:
            if attempt < 2:
                _time.sleep(backoff)
                backoff *= 2
                continue
            _enqueue_ingest(body_str, "timeout")
            sys.stderr.write(
                f"[stop_hook] micro-batch TIMEOUT → queued for replay\n"
            )
            return 0
        except Exception as exc:
            if attempt < 2:
                _time.sleep(backoff)
                backoff *= 2
                continue
            _enqueue_ingest(body_str, type(exc).__name__)
            sys.stderr.write(
                f"[stop_hook] micro-batch {type(exc).__name__} → queued for replay\n"
            )
            return 0
    return 0


def _enqueue_ingest(body_json: str, reason: str) -> None:
    """Append a failed micro-batch payload to the local queue."""
    try:
        os.makedirs(os.path.dirname(_INGEST_QUEUE), exist_ok=True)
        # Wrap with metadata so the drain can decide if it's still relevant.
        entry = json.dumps({
            "body": body_json,
            "queued_at": __import__("time").time(),
            "reason": reason,
        })
        with open(_INGEST_QUEUE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except OSError as exc:
        sys.stderr.write(f"[stop_hook] could not enqueue ingest: {exc}\n")


def extract_injected_chunks(user_text: str) -> list[tuple[str, str]]:
    """Legacy fallback: parse `(chunk_id, source_id)` pairs from inline block.

    The user-turn content in the JSONL transcript does NOT contain the
    inject block (Claude Code prefixes hook output to the prompt, but
    only the typed text lands as message.content). The canonical path
    is now ``read_inject_state(session_id)``; this regex is kept as a
    fallback for transcripts that *do* embed the block (e.g. for tests
    or pasted excerpts).
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for cid, sid in _CHUNK_LINE_RE.findall(user_text or ""):
        if cid in seen:
            continue
        seen.add(cid)
        pairs.append((cid, sid))
    return pairs


def read_inject_state(session_id: str) -> list[tuple[str, str]]:
    """Read the (chunk_id, source_id) pairs memory_inject wrote for this session."""
    if not session_id:
        return []
    path = os.path.join(_INJECT_STATE_DIR, f"{session_id}.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    pairs: list[tuple[str, str]] = []
    for entry in data.get("chunks", []):
        cid = (entry or {}).get("chunk_id")
        sid = (entry or {}).get("source_id", "")
        if cid:
            pairs.append((cid, sid))
    return pairs


def clear_inject_state(session_id: str) -> None:
    if not session_id:
        return
    path = os.path.join(_INJECT_STATE_DIR, f"{session_id}.json")
    try:
        os.remove(path)
    except OSError:
        pass


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


def _enqueue_feedback(chunk_id: str, signal: str, reason: str) -> None:
    """Append a failed feedback entry to the local queue for later replay.
    session_start.py::_drain_feedback_queue() ships these on next session.

    Schema is the SAME as the queue created by `bin/mayring-feedback` so
    both paths share a single drain implementation:
      {"chunk_id":"...", "signal":"positive|negative", "metadata":{...}}
    """
    try:
        os.makedirs(os.path.dirname(_FEEDBACK_QUEUE), exist_ok=True)
        entry = json.dumps({
            "chunk_id": chunk_id,
            "signal": signal,
            "metadata": {"queued_by": "stop_hook", "reason": reason},
        })
        with open(_FEEDBACK_QUEUE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except OSError as exc:
        sys.stderr.write(
            f"[stop_hook] could not enqueue feedback {chunk_id}/{signal}: {exc}\n"
        )


def _post_feedback(chunk_id: str, signal: str, token: str) -> None:
    """POST /memory/feedback with retry on 502/503/504 (deploy windows).

    On retry exhaustion or persistent network error: enqueue locally so
    next SessionStart's drain can replay. Only 4xx errors are dropped
    permanently (no retry will fix a malformed payload or unknown
    chunk).
    """
    import time as _time
    payload = json.dumps({"chunk_id": chunk_id, "signal": signal}).encode()
    req = urllib.request.Request(
        f"{_API_URL}/memory/feedback",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    backoff = 0.6
    for attempt in range(3):
        try:
            urllib.request.urlopen(req, timeout=_TIMEOUT)
            return
        except urllib.error.HTTPError as e:
            if e.code in (502, 503, 504) and attempt < 2:
                _time.sleep(backoff)
                backoff *= 2
                continue
            if 400 <= e.code < 500:
                # 4xx = chunk gone or invalid signal — replay won't help
                sys.stderr.write(
                    f"[stop_hook] feedback POST {chunk_id}/{signal}: "
                    f"HTTP {e.code} (dropped, no retry)\n"
                )
                return
            # 5xx after retries → enqueue
            _enqueue_feedback(chunk_id, signal, f"http_{e.code}")
            sys.stderr.write(
                f"[stop_hook] feedback POST {chunk_id}/{signal}: "
                f"HTTP {e.code} → queued for replay\n"
            )
            return
        except (urllib.error.URLError, OSError) as e:
            if attempt < 2:
                _time.sleep(backoff)
                backoff *= 2
                continue
            _enqueue_feedback(chunk_id, signal, type(e).__name__)
            sys.stderr.write(
                f"[stop_hook] feedback POST {chunk_id}/{signal}: "
                f"{type(e).__name__} → queued for replay\n"
            )
            return
        except Exception as exc:
            _enqueue_feedback(chunk_id, signal, type(exc).__name__)
            sys.stderr.write(
                f"[stop_hook] feedback POST {chunk_id}/{signal}: "
                f"{type(exc).__name__}: {exc} → queued for replay\n"
            )
            return


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


def _auto_feedback(turns: list[dict], session_id: str, token: str) -> None:
    """Rate every chunk that memory_inject announced for this prompt.

    Source of pairs: the per-session state file written by memory_inject
    (transcript content does NOT contain the inject block). Falls back
    to inline-block parsing of the user turn for back-compat / tests.
    Always clears the state file after rating so a missed Stop event
    doesn't double-rate on the next session.
    """
    if len(turns) < 2:
        return
    pairs = read_inject_state(session_id)
    if not pairs:
        # Fallback for legacy/test paths that do embed the block inline
        user_text = turns[0].get("content", "")
        pairs = extract_injected_chunks(user_text)
    if not pairs:
        return
    assistant_text = turns[1].get("content", "")
    posted = 0
    for chunk_id, source_id in pairs[:_AUTO_FEEDBACK_LIMIT]:
        signal = classify_chunk_relevance(source_id, assistant_text)
        _post_feedback(chunk_id, signal, token)
        posted += 1
    sys.stderr.write(f"[stop_hook] auto_feedback: posted {posted} ratings\n")
    clear_inject_state(session_id)


def main() -> None:
    token = _read_token()
    if not token:
        sys.stderr.write(f"[stop_hook] no token at {_JWT_FILE}; skipping\n")
        return
    payload = _read_payload()
    session_id = payload.get("session_id", "") or "unknown"
    try:
        turns = _capture_turns(payload, token)
    except Exception as exc:
        sys.stderr.write(f"[stop_hook] capture_turns crashed: {type(exc).__name__}: {exc}\n")
        turns = []
    try:
        _auto_feedback(turns, session_id, token)
    except Exception as exc:
        sys.stderr.write(f"[stop_hook] auto_feedback crashed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
