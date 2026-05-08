#!/usr/bin/env python3
"""UserPromptSubmit hook: inject relevant memory chunks into the prompt.

Runs on every user message (not only SessionStart). Calls
``POST /search`` on mcp.linn.games with the user's prompt and prints
the resulting ``prompt_context`` block. Output goes to stdout — Claude
Code surfaces UserPromptSubmit-hook stdout as a system reminder.

Idempotent + cheap-fail: silently no-ops when JWT missing or API down,
so it never blocks user input.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")
API = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
TIMEOUT = 6.0
TOP_K = 5
CHAR_BUDGET = 2500
MIN_PROMPT_LEN = 12  # skip 1-word commands like "ls"


def _load_token() -> str:
    try:
        with open(JWT_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def _read_payload() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return {}


def _extract_prompt(payload: dict) -> str:
    prompt = (
        payload.get("user_message")
        or payload.get("message")
        or payload.get("prompt")
        or ""
    )
    return str(prompt).strip()


def _search(query: str, token: str) -> dict | None:
    body = json.dumps({
        "query": query[:600],
        "top_k": TOP_K,
        "include_text": True,
        "char_budget": CHAR_BUDGET,
    }).encode()
    req = urllib.request.Request(
        f"{API}/search",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return None


def main() -> None:
    payload = _read_payload()
    prompt = _extract_prompt(payload)
    if len(prompt) < MIN_PROMPT_LEN:
        return

    token = _load_token()
    if not token:
        return

    result = _search(prompt, token)
    if not result:
        return

    context = (result.get("prompt_context") or "").strip()
    if not context:
        return

    chunk_ids = [r.get("chunk_id", "") for r in result.get("results", []) if r.get("chunk_id")]
    chunk_id_hint = ""
    if chunk_ids:
        chunk_id_hint = (
            "\n\n_Chunk-IDs zur Referenz (für `mcp__claude_ai_Memory__feedback`):_\n"
            + "\n".join(f"- `{cid}`" for cid in chunk_ids[:TOP_K])
        )

    print(
        f"## Memory-Kontext für diesen Prompt\n\n{context}{chunk_id_hint}\n\n"
        "_Pflicht: Nach dem Task `mcp__claude_ai_Memory__feedback` für die genutzten Chunks._"
    )


if __name__ == "__main__":
    main()
