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
        return ""  # silent skip — Subagent-Prompt ohne Memory ist OK, ein
                  # error-Block würde den Subagent verwirren
    ctx = (response.get("prompt_context") or "").strip()
    if not ctx:
        return ""
    diag = (response.get("diagnostics") or {}).get("vector_stage", "?")
    chunk_ids = [
        r.get("chunk_id", "") for r in (response.get("results") or [])
    ]
    block = [
        "## Pre-fetched Memory Context",
        "",
        f"_Searched for: {query[:80]!r}_  ",
        f"_Diagnostics: {diag}_",
        "",
        ctx,
        "",
        "_Diese Chunks wurden VOR deinem Dispatch von der main-session gesucht. "
        "Wenn du tieferen Kontext brauchst, ruf `mcp__claude_ai_Memory__search_memory` "
        "selbst auf. Gib am Ende per `mcp__claude_ai_Memory__feedback` eine "
        "positive/negative Bewertung pro genutztem chunk_id ab (siehe oben)._",
    ]
    if chunk_ids:
        block.append("")
        block.append("**Chunk-IDs (für feedback):** " + ", ".join(
            f"`{cid}`" for cid in chunk_ids[:8]
        ))
    return "\n".join(block)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Subagent-Task-Description (Stichworte)")
    ap.add_argument("--workspace-hint", default=None,
                    help="Optional: workspace_id-hint, default vom JWT")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--char-budget", type=int, default=DEFAULT_CHAR_BUDGET)
    ap.add_argument("--api", default=DEFAULT_API)
    args = ap.parse_args()

    token = _load_token()
    if not token:
        # silent: Subagent-Prompt ohne Memory ist OK
        return 0

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
