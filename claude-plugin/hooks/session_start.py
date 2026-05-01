#!/usr/bin/env python3
"""SessionStart hook: Memory-Kontext aus mcp.linn.games in den System-Prompt injizieren.

Sendet zusätzlich zur Initial-Query den Inhalt des aktuellsten Plans aus
~/.claude/plans/ als task_context — das gibt dem PI-Advisor-Reranker eine
reichere Beschreibung der anstehenden Arbeit als nur die generische
'session start'-Query.
"""
import glob
import json
import os
import sys
import urllib.request
import urllib.error


PLANS_DIR = os.path.expanduser("~/.claude/plans")
TASK_CONTEXT_BUDGET = 800  # chars from the plan file's Context section


def _load_token() -> str:
    token = os.getenv("MCP_SERVICE_TOKEN", "")
    if token:
        return token
    for env_file in [
        os.path.expanduser("~/app.linn.games/.env.mayring"),
        os.path.expanduser("~/.env.mayring"),
    ]:
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MCP_SERVICE_TOKEN="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def _latest_plan_context() -> str:
    """Return the Context section (or first chunk) of the most-recent plan file."""
    try:
        files = glob.glob(os.path.join(PLANS_DIR, "*.md"))
        if not files:
            return ""
        latest = max(files, key=os.path.getmtime)
        with open(latest) as f:
            content = f.read()
        # Prefer the dedicated "## Context" block when present
        if "## Context" in content:
            after = content.split("## Context", 1)[1]
            # Stop at next H2 to keep it focused
            if "\n## " in after:
                after = after.split("\n## ", 1)[0]
            return after.strip()[:TASK_CONTEXT_BUDGET]
        return content[:TASK_CONTEXT_BUDGET]
    except Exception:
        return ""


payload = json.loads(sys.stdin.read() or "{}")
token = _load_token()
if not token:
    sys.exit(0)

api = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")

initial_message = (
    payload.get("message")
    or payload.get("prompt")
    or payload.get("transcript", [{}])[0].get("content", "")
    or "session start"
)
query = str(initial_message)[:300]

body_dict = {
    "query": query,
    "top_k": 5,
    "include_text": True,
    "char_budget": 3000,
}
task_context = _latest_plan_context()
if task_context:
    body_dict["task_context"] = task_context

body = json.dumps(body_dict).encode()

req = urllib.request.Request(
    f"{api}/search",
    data=body,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
)
try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        result = json.loads(resp.read())
    context = result.get("prompt_context", "").strip()
    if context:
        print(f"## Automatisch injizierter Memory-Kontext\n\n{context}\n\n"
              "⚠️  PFLICHT: Gib am Ende jedes Tasks Feedback via "
              "`mcp__claude_ai_Memory__feedback` für jeden genutzten Chunk.")
except Exception:
    pass
