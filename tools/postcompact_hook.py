#!/usr/bin/env python3
"""PostCompact hook — ingests Claude Code compact summary into MayringCoder memory.

Called by ~/.claude/settings.json hooks.PostCompact.
Receives {"summary": "..."} as JSON via stdin.
Always exits 0 — never blocks the compact flow.
"""
import datetime
import hashlib
import json
import os
import sys
import urllib.request

data = json.loads(sys.stdin.read())
summary = data.get("summary", "")
if not summary.strip():
    sys.exit(0)

api = os.environ.get("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
jwt_file = os.path.expanduser("~/.config/mayring/hook.jwt")

try:
    with open(jwt_file) as f:
        token = f.read().strip()
except FileNotFoundError:
    sys.exit(0)

if not token:
    sys.exit(0)

ts = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
sid = f"conversation_summary:compact-{ts}-{hashlib.sha256(summary[:64].encode()).hexdigest()[:8]}"

payload = json.dumps({
    "source_id": sid,
    "source_type": "conversation_summary",
    "content": summary,
    "categorize": True,
}).encode()

req = urllib.request.Request(
    f"{api}/memory/put",
    data=payload,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    method="POST",
)
try:
    urllib.request.urlopen(req, timeout=15)
except Exception:
    pass
