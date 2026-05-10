"""Subagent Pre-Fetch — token-sparsame Memory-Injection für Subagent-Dispatch.

Subagents bekommen ihre eigene Sub-Session und sehen den UserPromptSubmit-
Hook NICHT — d.h. das gesamte Memory-System (reranker-v2, rating-feedback,
IGIO, rationale) ist für sie unsichtbar wenn sie nicht selbst /memory/search
aufrufen.

Dieses Skript schliesst die Lücke OHNE den Subagent-Kontext mit
volltexten zu überladen. Standard-modus seit 2026-05-10 (rating-migration):
nur **chunk-IDs + 1-zeilige zusammenfassungen + bisherige ratings**
werden injiziert. Subagent ruft search_memory oder /memory/chunk/<id>
selbst wenn er den vollen text braucht.

Vorteil: prompt-länge ~80 chars × top_k statt 600 × top_k → 87% token-
ersparnis bei gleichem signal-wert.

Usage (aus main session):
    python tools/subagent_prefetch.py "fix the auth pipeline in app.linn.games" \\
        --top-k 5

    # Volltext-modus (alt-verhalten, bei bedarf):
    python tools/subagent_prefetch.py "..." --include-text

    # Feedback (rating 1..5 statt binary):
    python tools/subagent_prefetch.py --feedback chk_abc 5

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
DEFAULT_CHAR_BUDGET = 800   # nur für --include-text relevant


def _load_token() -> str:
    try:
        return Path(JWT_FILE).read_text().strip()
    except OSError:
        return ""


def _search(
    query: str, *, api: str, token: str, top_k: int, char_budget: int,
    workspace_hint: str | None, include_text: bool,
) -> dict:
    """POST /memory/search. Wenn include_text=False, holen wir nur metadata.

    Returns either the parsed response or {_error}.
    """
    body: dict = {
        "query": query[:600],
        "top_k": top_k,
        # WHY(2026-05-10): default include_text=False. Volltext frisst
        # subagent-context (~600 chars/chunk × 5 = 3000 chars unnötig).
        # Subagent ruft /memory/chunk/<id> wenn er den text wirklich braucht.
        "include_text": include_text,
        "char_budget": char_budget if include_text else 200,
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


def _summarize(text: str, limit: int = 80) -> str:
    """1-zeilen-extract aus chunk-text. Erste sinnvolle codezeile/satz."""
    if not text:
        return ""
    # Strip leading whitespace / blank lines, kürze auf erste non-trivial-zeile
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # leerere docstring-zeilen + import-statements überspringen
        if s.startswith(('"""', "'''", "//", "#")) and len(s) < 4:
            continue
        return s[:limit] + ("…" if len(s) > limit else "")
    return text[:limit].replace("\n", " ")


def _render_block(response: dict, *, query: str, include_text: bool) -> str:
    """Token-sparsame Markdown-tabelle als Subagent-prompt-prefix.

    Default-modus (include_text=False) zeigt pro chunk eine zeile:
        - `chk_id` (★X.X) [cats] sid — 1-zeilen-summary

    Mit --include-text rendert das volle prompt_context wie früher.
    """
    if "_error" in response:
        return ""  # silent skip
    results = response.get("results") or []
    if not results:
        return ""
    diag = (response.get("diagnostics") or {}).get("vector_stage", "?")

    block = [
        "## Pre-fetched Memory Context",
        "",
        f"_Searched for: {query[:80]!r}_  ",
        f"_Diagnostics: {diag}_",
        "",
    ]

    if include_text:
        # alt-verhalten: voller prompt_context
        ctx = (response.get("prompt_context") or "").strip()
        if ctx:
            block.append(ctx)
            block.append("")
    else:
        # NEU: nur 1-zeilen-zusammenfassung pro chunk + rating
        block.append("**Chunks** (1-line summary + bisheriges ø-rating, "
                     "details bei bedarf via `/memory/chunk/<id>`):")
        block.append("")
        for r in results[:8]:
            cid = r.get("chunk_id", "")
            sid = (r.get("source_id") or "")[-60:]
            # score_feedback aus reranker-output ∈ [0..1] → re-mappe auf 1..5
            sf = float(r.get("score_feedback") or 0.5)
            rating = round(sf * 4 + 1, 1)
            cats = r.get("category_labels") or []
            cat_str = (",".join(cats[:3])) if cats else ""
            summary = _summarize(r.get("text", "") or "")
            cat_tag = f" [{cat_str}]" if cat_str else ""
            block.append(
                f"- `{cid}` (★{rating}){cat_tag} `{sid}`"
                + (f" — {summary}" if summary else "")
            )
        block.append("")

    # Mini-cheatsheet — wie subagent feedback gibt + details holt
    block.append(
        "_Du brauchst tieferen kontext? Hol einen einzelnen chunk:_\n"
        "```bash\n"
        "python /home/nileneb/Desktop/MayringCoder/tools/subagent_prefetch.py \\\n"
        "  '<spezifische frage>' --include-text --top-k 3\n"
        "```\n\n"
        "_Feedback (PFLICHT am ende, rating 1..5 — 5=primärquelle):_\n"
        "```bash\n"
        "python /home/nileneb/Desktop/MayringCoder/tools/subagent_prefetch.py \\\n"
        "  --feedback chk_xxx 5\n"
        "```"
    )
    return "\n".join(block)


def _send_feedback(chunk_id: str, signal: str, *, api: str, token: str) -> int:
    """POST /memory/feedback for a single chunk. Returns exit-code (0=ok).

    rating-migration: signal MUSS '1'..'5' sein, binary positive/negative
    werden vom server abgewiesen.
    """
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
            print(f"feedback ok: chunk={chunk_id} rating={signal} → {data.get('status', '?')}")
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
    ap.add_argument(
        "--include-text", action="store_true",
        help="(alt) volltext-modus statt 1-zeilen-summaries. Frisst ~6× mehr "
             "subagent-context — nutze nur wenn der subagent absehbar mit "
             "den chunk-inhalten arbeitet statt sie nur referenziert.",
    )
    ap.add_argument(
        "--feedback", nargs=2, metavar=("CHUNK_ID", "RATING"),
        help="rating 1..5 für chunk_id (stabiler als mcp__claude_ai_Memory__"
             "feedback nach container-restart, weil JWT-direkt statt MCP-session).",
    )
    args = ap.parse_args()

    token = _load_token()
    if not token:
        return 0  # silent skip

    if args.feedback:
        chunk_id, signal = args.feedback
        if signal not in ("1", "2", "3", "4", "5"):
            print(
                f"rating must be '1'..'5' (1=irrelevant, 5=primärquelle), "
                f"got {signal!r}",
                file=sys.stderr,
            )
            return 2
        return _send_feedback(chunk_id, signal, api=args.api, token=token)

    if not args.query:
        ap.error("query required (or use --feedback CHUNK_ID RATING)")
    response = _search(
        args.query, api=args.api, token=token,
        top_k=args.top_k, char_budget=args.char_budget,
        workspace_hint=args.workspace_hint,
        include_text=args.include_text,
    )
    block = _render_block(response, query=args.query, include_text=args.include_text)
    if block:
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
