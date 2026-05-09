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
import time as _time
import urllib.request
import urllib.error

JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")
API = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")

# Persistent state for the Stop hook. The injected chunk_id list does NOT
# survive in the user-turn content of the JSONL transcript — Claude Code
# treats hook output as prompt prefix, not user-typed text. So the Stop
# hook has no way to read what was injected from the transcript alone;
# we drop a small state file here that the Stop hook picks up.
INJECT_STATE_DIR = os.path.expanduser("~/.config/mayring/inject-state")

# Per-request timeout budget. Was 4.0s but the hybrid search auto-activates
# the PI-advisor LLM stage when the scope-filter returns >10 candidates,
# which is normal for any populated workspace — that stage adds 2-4s on top
# of vector + symbolic. Sub-4s timeouts caused every prompt to fall through
# to "Suche fehlgeschlagen" even though the API itself was healthy.
TIMEOUT = 9.0           # per-request
GLOBAL_TIMEOUT = 12.0   # whole hook (3 lenses run concurrently)
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
) -> dict:
    """Run one /memory/search lens.

    Returns either the parsed JSON response, or a synthetic dict with a
    `_hook_error` key that surfaces the failure mode in the prompt block.
    Silent ``return None`` on every exception is exactly how this hook
    masked a 4s timeout for weeks — never again.
    """
    body_dict: dict = {
        "query": query[:600],
        "top_k": top_k,
        "include_text": True,
        "char_budget": char_budget,
        # Hook runs in the prompt critical path — skip the LLM advisor
        # stage that adds 2-4s on a populated workspace. Symbolic + vector
        # are still ranked; only the post-hoc relevance scoring is off.
        "llm_prefilter": False,
    }
    if source_type:
        body_dict["source_type"] = source_type
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(
        f"{API}/memory/search",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    # Retry on 502/503/504 — common during MayringCoder deploy windows
    # (~30s nginx returns 502 while uvicorn restarts). Without retry,
    # every UserPromptSubmit during a deploy fails the hook → no Memory
    # Context injection → user sees "kein Kontext mehr injiziert".
    # 3 attempts with backoff; total ≤ TIMEOUT - 1s budget so we don't
    # exceed the per-request timeout.
    last_err: str = ""
    last_status: int = 0
    # 4 attempts, exponentielles Backoff: 1.0 + 2.0 + 4.0 = 7s wait
    # zwischen request 1 und 4 → bridge typische 30s-Deploy-Windows
    # bei mayring-stack-restart (Container-stop + start + healthcheck).
    backoff = 1.0
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code} from /memory/search"
            last_status = e.code
            if e.code in (502, 503, 504) and attempt < 3:
                _time.sleep(backoff)
                backoff *= 2
                continue
            return {"_hook_error": last_err, "_status": last_status}
        except TimeoutError:
            return {"_hook_error": f"TIMEOUT after {TIMEOUT}s — server is slow or down"}
        except urllib.error.URLError as e:
            last_err = f"URLError: {e.reason}"
            if attempt < 3:
                _time.sleep(backoff)
                backoff *= 2
                continue
            return {"_hook_error": last_err}
        except OSError as e:
            return {"_hook_error": f"OSError {e.errno}: {e.strerror}"}
        except ValueError as e:
            return {"_hook_error": f"JSON parse error: {e}"}
    return {"_hook_error": last_err or "unknown"}


def _multi_lens_search(query: str, token: str) -> dict[str, dict]:
    """Run three lens-searches concurrently; one entry per lens.

    Each value is either a real search response or a `{_hook_error: ...}`
    sentinel. Cancellation/timeout in the futures executor itself also
    surfaces as `_hook_error` so the user actually sees what's wrong.
    """
    lenses = {
        "primary":      {},
        "ambient":      {"source_type": "ambient_snapshot", "top_k": TOP_K_LENS, "char_budget": 1000},
        "conversation": {"source_type": "conversation_summary", "top_k": TOP_K_LENS, "char_budget": 1000},
    }
    results: dict[str, dict] = {n: {"_hook_error": "lens did not complete in time"}
                                for n in lenses}
    with _cf.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_search, query, token, **kwargs): name
            for name, kwargs in lenses.items()
        }
        try:
            for fut in _cf.as_completed(futures, timeout=GLOBAL_TIMEOUT):
                name = futures[fut]
                try:
                    results[name] = fut.result()
                except Exception as exc:
                    results[name] = {"_hook_error": f"{type(exc).__name__}: {exc}"}
        except _cf.TimeoutError:
            # Leave the pre-seeded "did not complete" sentinels in place.
            pass
    return results


def _write_inject_state(session_id: str, chunk_pairs: list[tuple[str, str]]) -> None:
    """Persist (chunk_id, source_id) pairs so the Stop hook can rate them.

    Path: ~/.config/mayring/inject-state/<session_id>.json. Stop hook
    reads + deletes after rating. Best-effort, never raises.
    """
    if not session_id or not chunk_pairs:
        return
    try:
        os.makedirs(INJECT_STATE_DIR, exist_ok=True)
        path = os.path.join(INJECT_STATE_DIR, f"{session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": [{"chunk_id": c, "source_id": s} for c, s in chunk_pairs]},
                f,
            )
    except OSError:
        pass


# WHY(observability): silent skip bei deploy-windows. Mayring-API restartet
# bei jedem deploy für ~30s. Ohne 5xx-skip schreibt der Hook bei jedem
# UserPromptSubmit einen lauten Memory-fehler-Block ins Prompt — aber
# alle 3 lenses 502 ist kein User-actionable Fehler. Bei Mix (1× 502 + 2×
# OK oder ein 4xx) bleibt der laute Block, weil das ist eine echte
# Konfig-Anomalie. CHANGE WITH CARE — bei zu lauter silence verlieren wir
# echte Probleme; bei zu sturem laut nervt der deploy-window jeden user.
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
    if "_hook_error" in primary:
        # Sonderfall: deploy-typische 5xx (502/503/504) sind transient
        # — der Stack restartet gerade, in 10s ist alles wieder gut.
        # Kein lauter warning-block, weil das den User pro prompt
        # nervt und keine Aktion erfordert.
        all_5xx = all(
            (r or {}).get("_status") in (502, 503, 504)
            for r in results.values() if "_hook_error" in (r or {})
        )
        if all_5xx:
            return  # silent skip — Memory injizieren wir beim nächsten prompt
        # Sonst: laut, weil der Fehler eine Aktion braucht (4xx, parse,
        # OSError, timeout). Lists ALL three lens errors at once.
        errs = [
            f"  - {lens}: {(r or {}).get('_hook_error', 'no response')}"
            for lens, r in results.items()
            if (r or {}).get("_hook_error")
        ]
        print(
            "## Memory: Hook konnte Memory nicht laden\n"
            f"_API={API}_  _prompt[:50]={prompt[:50]!r}_\n"
            + "\n".join(errs)
            + "\n\n_Wenn dieser Block wiederholt erscheint: API-Healthcheck "
              "(`curl https://mcp.linn.games/health`) prüfen oder Plugin neu "
              "laden (`/reload-plugins`)._"
        )
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
    if ambient and "_hook_error" not in ambient:
        ambient_ctx = (ambient.get("prompt_context") or "").strip()
        if ambient_ctx:
            sections.append("\n### Ambient Snapshot (Projekt-Kontext)")
            sections.append(ambient_ctx)

    conv = results.get("conversation") or {}
    if conv and "_hook_error" not in conv:
        conv_ctx = (conv.get("prompt_context") or "").strip()
        if conv_ctx:
            sections.append("\n### Vorherige Sessions / Decisions")
            sections.append(conv_ctx)

    # Pair each chunk with its source_id so the Stop hook can classify
    # positive/negative automatically (path match against the assistant
    # answer). Format is parsed by stop_hook._CHUNK_LINE_RE — keep stable.
    seen_ids: set[str] = set()
    chunk_pairs: list[tuple[str, str]] = []
    for r in (primary, ambient, conv):
        if not r or "_hook_error" in r:
            continue
        for chunk in (r.get("results") or []):
            cid = chunk.get("chunk_id", "")
            sid = chunk.get("source_id", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                chunk_pairs.append((cid, sid))

    # Persist for the Stop hook (transcript doesn't capture this block).
    _write_inject_state(payload.get("session_id", ""), chunk_pairs[:8])

    chunk_id_hint = ""
    if chunk_pairs:
        chunk_id_hint = (
            "\n\n_Injected chunks (auto-feedback by Stop hook):_\n"
            + "\n".join(f"- `{cid}` : `{sid}`" for cid, sid in chunk_pairs[:8])
        )

    print(
        f"## Memory-Kontext für diesen Prompt\n\n"
        + "\n\n".join(sections)
        + chunk_id_hint
    )


if __name__ == "__main__":
    main()
