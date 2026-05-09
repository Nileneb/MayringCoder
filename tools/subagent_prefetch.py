"""Subagent Pre-Fetch — proaktive Memory-Injection für Subagent-Dispatch.

Subagents bekommen ihre eigene Sub-Session und sehen den UserPromptSubmit-
Hook NICHT — d.h. das gesamte Memory-System (reranker-v2, IGIO, rationale)
ist für sie unsichtbar wenn sie nicht selbst /memory/search aufrufen.

Dieses Skript schliesst die Lücke: vor jedem Agent-dispatch ruft die main-
session den helper, formatiert die Top-K-Treffer als Markdown-Block und
injiziert sie in den Subagent-Prompt unter "## Pre-fetched Memory Context".

Usage (aus main session):
    python tools/subagent_prefetch.py "fix the auth pipeline in app.linn.games" \\
        --workspace bene --top-k 5

Output: Markdown-Block ready zum Inject in Agent-Prompt.

Auth: nutzt MAYRING_JWT aus ~/.config/mayring/hook.jwt (gleicher Pfad wie
das Plugin-Hook), so dass workspace-isolation automatisch greift.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")
DEFAULT_API = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
DEFAULT_TOP_K = 5
DEFAULT_CHAR_BUDGET = 2500


def _load_token() -> str:
    try:
        return Path(JWT_FILE).read_text().strip()
    except OSError:
        return ""


def _search(
    query: str, *, api: str, token: str, top_k: int, char_budget: int,
    workspace_hint: str | None,
) -> dict:
    """Run /memory/search. Returns either the parsed response or {_error}."""
    body = {
        "query": query[:600],
        "top_k": top_k,
        "include_text": True,
        "char_budget": char_budget,
        "llm_prefilter": False,
    }
    if workspace_hint:
        body["workspace_hint"] = workspace_hint
    req = urllib.request.Request(
        f"{api}/memory/search",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}"}
    except (urllib.error.URLError, OSError) as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    except json.JSONDecodeError as e:
        return {"_error": f"parse: {e}"}


def _render_block(response: dict, *, query: str) -> str:
    """Format the API response as a Markdown-block ready for the Subagent-prompt."""
    if "_error" in response:
        return ""  # silent skip
    ctx = (response.get("prompt_context") or "").strip()
    if not ctx:
        return ""
    diag = (response.get("diagnostics") or {}).get("vector_stage", "?")
    results = response.get("results") or []
    chunk_ids = [r.get("chunk_id", "") for r in results]

    # Praxistest #2 finding: Subagents übersehen chunk_ids wenn sie nur am
    # Block-Ende stehen. Plus: MCP-self-search ist nach Container-restart
    # fragil → CLI-feedback-Pfad ist robuster. Daher: chunk_ids PROMINENT
    # vorne + bash-helper unten.
    block = [
        "## Pre-fetched Memory Context",
        "",
        f"_Searched for: {query[:80]!r}_  ",
        f"_Diagnostics: {diag}_",
        "",
    ]
    if chunk_ids:
        block.append("**Chunk-IDs für Feedback** (am Task-Ende rate jeden):")
        for r in results[:8]:
            cid = r.get("chunk_id", "")
            sid = (r.get("source_id") or "")[:60]
            block.append(f"- `{cid}` — `{sid}`")
        block.append("")
    block.extend([ctx, ""])
    block.append(
        "_Pre-fetched via `subagent_prefetch.py` (stabil — JWT-direkt). "
        "Wenn du tieferen Kontext brauchst: `mcp__claude_ai_Memory__search_memory` "
        "(kann nach Container-restarts fragile sein) ODER stabil per CLI:_\n"
        "```bash\n"
        "python /home/nileneb/Desktop/MayringCoder/tools/subagent_prefetch.py \\\n"
        "  '<deine Frage>' --top-k 5\n"
        "```\n\n"
        "_Feedback (PFLICHT am Ende, stabil per CLI):_\n"
        "```bash\n"
        "python /home/nileneb/Desktop/MayringCoder/tools/subagent_prefetch.py \\\n"
        "  --feedback chk_xxx positive  # oder negative\n"
        "```"
    )
    return "\n".join(block)


def _send_feedback(chunk_id: str, signal: str, *, api: str, token: str) -> int:
    """POST /memory/feedback for a single chunk. Returns exit-code (0=ok)."""
    body = {"chunk_id": chunk_id, "signal": signal}
    req = urllib.request.Request(
        f"{api}/memory/feedback",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            data = json.loads(resp.read())
            print(f"feedback ok: chunk={chunk_id} signal={signal} → {data.get('status', '?')}")
            return 0
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
        except Exception:
            err = {"raw": "?"}
        print(f"feedback FAIL ({e.code}): {err}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"feedback FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="",
                    help="Subagent-Task-Description (Stichworte) — leer wenn --feedback")
    ap.add_argument("--workspace-hint", default=None)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--char-budget", type=int, default=DEFAULT_CHAR_BUDGET)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--feedback", nargs=2, metavar=("CHUNK_ID", "SIGNAL"),
                    help="Statt search: positive/negative-feedback für chunk_id "
                         "(stabiler als mcp__claude_ai_Memory__feedback nach "
                         "Container-restart, weil JWT-direkt statt MCP-session).")
    args = ap.parse_args()

    token = _load_token()
    if not token:
        return 0  # silent skip

    if args.feedback:
        chunk_id, signal = args.feedback
        if signal not in ("positive", "negative"):
            print(f"signal must be 'positive' or 'negative', got {signal!r}",
                  file=sys.stderr)
            return 2
        return _send_feedback(chunk_id, signal, api=args.api, token=token)

    if not args.query:
        ap.error("query required (or use --feedback CHUNK_ID SIGNAL)")
    response = _search(
        args.query, api=args.api, token=token,
        top_k=args.top_k, char_budget=args.char_budget,
        workspace_hint=args.workspace_hint,
    )
    block = _render_block(response, query=args.query)
    if block:
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
