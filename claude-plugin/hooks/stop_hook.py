#!/usr/bin/env python3
"""Stop hook — marks injected-but-unrated memory chunks as neutral.

Called by ~/.claude/settings.json hooks.Stop after every Claude turn.
Queries /memory/unrated-chunks and calls /memory/feedback(signal=neutral)
for each chunk Claude saw but didn't explicitly rate.
Always exits 0 — never blocks Claude.
"""
import json
import os
import sys
import urllib.error
import urllib.request

_JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")
_API_URL = os.environ.get("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
_TIMEOUT = 5
_LOOKBACK_MINUTES = 60


def _read_token() -> str:
    try:
        with open(_JWT_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _get(path: str, token: str) -> dict:
    req = urllib.request.Request(
        f"{_API_URL}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=_TIMEOUT)
        return json.loads(resp.read())
    except Exception:
        return {}


def _post_feedback(chunk_id: str, token: str) -> None:
    payload = json.dumps({"chunk_id": chunk_id, "signal": "neutral"}).encode()
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


token = _read_token()
if not token:
    sys.exit(0)

data = _get(f"/memory/unrated-chunks?minutes={_LOOKBACK_MINUTES}", token)
chunk_ids: list[str] = data.get("chunk_ids", [])

for chunk_id in chunk_ids:
    _post_feedback(chunk_id, token)
