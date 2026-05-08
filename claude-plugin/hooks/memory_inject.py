#!/usr/bin/env python3
"""UserPromptSubmit hook: inject relevant memory chunks into the prompt.

Three parallel /search calls per prompt give a multi-lens view:
  - generic semantic search (current task)
  - ambient_snapshot lens (project-level context)
  - conversation_summary lens (what was decided/done before)

Three queries run concurrently with strict timeouts so the hook never blocks
input for more than ~3 s total. Cheap-fail when JWT missing or API down.
"""
from __future__ import annotations

import concurrent.futures as _cf
import json
import os
import sys
import urllib.request
import urllib.error

JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")
API = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
TIMEOUT = 4.0           # per-request
GLOBAL_TIMEOUT = 6.0    # whole hook
TOP_K_PRIMARY = 4
TOP_K_LENS = 2          # per ambient/conv lens
CHAR_BUDGET = 1800      # per call → ~5400 total max
MIN_PROMPT_LEN = 12     # skip 1-word commands like "ls"


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


def _search(
    query: str, token: str, *, top_k: int = TOP_K_PRIMARY,
    source_type: str | None = None, char_budget: int = CHAR_BUDGET,
) -> dict | None:
    body_dict: dict = {
        "query": query[:600],
        "top_k": top_k,
        "include_text": True,
        "char_budget": char_budget,
    }
    if source_type:
        body_dict["source_type"] = source_type
    body = json.dumps(body_dict).encode()
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


def _multi_lens_search(query: str, token: str) -> dict[str, dict | None]:
    """Run three lens-searches concurrently; return {lens_name: result|None}."""
    lenses = {
        "primary":      {},
        "ambient":      {"source_type": "ambient_snapshot", "top_k": TOP_K_LENS, "char_budget": 1000},
        "conversation": {"source_type": "conversation_summary", "top_k": TOP_K_LENS, "char_budget": 1000},
    }
    results: dict[str, dict | None] = {}
    with _cf.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_search, query, token, **kwargs): name
            for name, kwargs in lenses.items()
        }
        for fut in _cf.as_completed(futures, timeout=GLOBAL_TIMEOUT):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception:
                results[name] = None
    return results


def main() -> None:
    payload = _read_payload()
    prompt = _extract_prompt(payload)
    if len(prompt) < MIN_PROMPT_LEN:
        return

    token = _load_token()
    if not token:
        return

    results = _multi_lens_search(prompt, token)
    primary = results.get("primary") or {}
    if not primary:
        print(f"## Memory: Suche fehlgeschlagen (API={API}, prompt[:50]={prompt[:50]!r})")
        return

    primary_ctx = (primary.get("prompt_context") or "").strip()
    primary_diag = (primary.get("diagnostics") or {}).get("vector_stage", "?")

    if not primary_ctx:
        print(
            f"## Memory: keine Treffer für diesen Prompt "
            f"(vector_stage={primary_diag}, candidates={(primary.get('diagnostics') or {}).get('candidates', 0)})"
        )
        return

    sections: list[str] = [
        f"### Code/Findings (semantic search)",
        f"_diag: {primary_diag}_",
        primary_ctx,
    ]

    ambient = results.get("ambient") or {}
    ambient_ctx = (ambient.get("prompt_context") or "").strip() if ambient else ""
    if ambient_ctx:
        sections.append("\n### Ambient Snapshot (Projekt-Kontext)")
        sections.append(ambient_ctx)

    conv = results.get("conversation") or {}
    conv_ctx = (conv.get("prompt_context") or "").strip() if conv else ""
    if conv_ctx:
        sections.append("\n### Vorherige Sessions / Decisions")
        sections.append(conv_ctx)

    # Collect all chunk_ids across all lenses for feedback reference
    all_chunk_ids: list[str] = []
    for r in (primary, ambient, conv):
        for chunk in (r.get("results") or []):
            cid = chunk.get("chunk_id", "")
            if cid and cid not in all_chunk_ids:
                all_chunk_ids.append(cid)

    chunk_id_hint = ""
    if all_chunk_ids:
        chunk_id_hint = (
            "\n\n_Chunk-IDs für `mcp__claude_ai_Memory__feedback`:_\n"
            + "\n".join(f"- `{cid}`" for cid in all_chunk_ids[:8])
        )

    print(
        f"## Memory-Kontext für diesen Prompt\n\n"
        + "\n\n".join(sections)
        + chunk_id_hint
        + "\n\n_Pflicht: Nach Task `mcp__claude_ai_Memory__feedback` für genutzte Chunks._"
    )


if __name__ == "__main__":
    main()
