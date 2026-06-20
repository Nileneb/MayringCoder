"""Mythos-style sufficiency gate — iterative retrieval halting on the orchestration level.

OpenMythos' core is not its from-scratch RDT model but ACT (Adaptive Computation
Time): a halt scalar that decides WHEN to stop looping, with the explicit warning
that *more loops drift past the solution into noise*. We translate that from the
model level to the RETRIEVAL level: a small thinking model (gemma4:e4b) judges
whether the reranked chunks suffice to answer the query/goal; if not, its `requery`
drives one more retrieval loop. The LOOP is the "think until solved" mechanism; the
model's `sufficient` flag is the ACT halt scalar.

This is NOT re-ranking — the reranker is capped at the feature ceiling (clean-eval
~0.368). The gate works on a different axis: it detects MISSING information and
fetches it. It sits on the ACT path (after _rerank, before the answering LLM), NOT
the per-prompt inject hot path (9s budget / VRAM-thrash scars).

Halting is multi-criteria (OR), so the loop can never hang (the span_judge-hang
lesson): `sufficient OR no_requery OR no_progress OR cap OR budget`.

Probe (2026-06-20): gemma4:e4b gives the structured verdict correctly in BOTH
think modes; think=False is ~6× faster (0.8s vs 7.4s) at an identical verdict, so
depth comes from the loop, not the model's internal <think> channel.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

import httpx

try:
    from tools.span_judge import _loads_lenient
except ImportError:
    from span_judge import _loads_lenient

_log = logging.getLogger(__name__)

SUFFICIENCY_RUBRIC = """Du bist ein Relevanz-Sufficiency-Richter für ein RAG-System.
Gegeben eine QUERY/ein Ziel und eine Liste abgerufener CHUNKS, entscheide:
Reichen diese Chunks aus, um die Query vollständig und korrekt zu beantworten?

Antworte mit STRIKTEM JSON, keine Prosa:
{"sufficient": true|false, "missing": ["konkrete fehlende Information", ...], "requery": "praeziser Suchstring fuer das groesste Loch, oder \\"\\""}

Sei streng: sufficient=true NUR wenn die Chunks die Frage wirklich beantworten.
Bei sufficient=false MUSS requery ein konkreter, neuer Suchstring sein."""

_DEFAULT_MODEL = "gemma4:e4b"
_DEFAULT_TIMEOUT = 30.0
_KEEP_ALIVE = "30m"


def _ollama_url(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")


def _chunk_text(c: dict) -> str:
    return (c.get("text") or "").strip()


def judge_sufficiency(
    query: str,
    chunks: list[dict],
    ollama_url: str | None = None,
    model: str | None = None,
    think: bool = False,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    """One gemma call: do `chunks` suffice to answer `query`?

    Returns {"sufficient": bool, "missing": list[str], "requery": str}.

    Fail-safe: on ANY error / unparseable response → sufficient=True with empty
    missing. That degrades to 'use what we have' instead of looping forever — the
    judge is an optimization, never a blocker (same contract as the LLM advisor).
    This is the explicit 'judge unavailable' path, logged, not a silent swallow.
    """
    url = _ollama_url(ollama_url)
    mdl = model or _DEFAULT_MODEL
    body = {
        "model": mdl,
        "messages": [
            {"role": "system", "content": SUFFICIENCY_RUBRIC},
            {"role": "user", "content": "QUERY: " + (query or "") + "\n\nCHUNKS:\n"
             + "\n".join(f"[{i}] {_chunk_text(c)}" for i, c in enumerate(chunks))},
        ],
        "stream": False,
        "format": "json",
        "think": bool(think),
        "options": {"num_predict": 512, "temperature": 0},
        "keep_alive": _KEEP_ALIVE,
    }
    try:
        resp = httpx.post(f"{url}/api/chat", json=body, timeout=timeout)
        resp.raise_for_status()
        content = (resp.json().get("message", {}) or {}).get("content", "")
        data = _loads_lenient(content)
    except Exception as exc:  # noqa: BLE001 — fail-safe degrade, logged
        _log.warning("sufficiency judge unavailable (%s) — degrading to sufficient", exc)
        return {"sufficient": True, "missing": [], "requery": ""}
    missing = data.get("missing") or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    return {
        "sufficient": bool(data.get("sufficient", True)),
        "missing": [str(m) for m in missing],
        "requery": str(data.get("requery") or "").strip(),
    }


def run_sufficiency_loop(
    query: str,
    initial_chunks: list[dict],
    retrieve_fn: Callable[[str], list[dict]],
    ollama_url: str | None = None,
    *,
    model: str | None = None,
    think: bool = False,
    max_loops: int = 2,
    budget_s: float = 20.0,
    judge_fn: Callable[..., dict] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict:
    """Iteratively grow the chunk set until the gate says it suffices.

    retrieve_fn(requery) → new chunks for the model-identified gap (injected so the
    loop is testable and decoupled from the concrete search backend).

    Halts on the FIRST of (OR): the judge says sufficient; no requery given; the
    re-retrieval brings nothing new (no progress); the loop cap is hit (ACT's
    'more loops → noise' bound); the time budget is exceeded. Returns the grown
    chunk set, the per-iteration verdict trace, and which criterion halted it.
    """
    judge = judge_fn or (lambda q, ch, **kw: judge_sufficiency(
        q, ch, ollama_url, model=model, think=think))
    now = clock or time.monotonic
    start = now()

    chunks = list(initial_chunks)
    seen = {c["chunk_id"] for c in chunks}
    trace: list[dict] = []
    loops = 0

    while True:
        verdict = judge(query, chunks, think=think)
        trace.append(verdict)

        if verdict.get("sufficient"):
            return _result(chunks, trace, "sufficient", loops)
        if loops >= max_loops:
            return _result(chunks, trace, "cap", loops)
        if (now() - start) > budget_s:
            return _result(chunks, trace, "budget", loops)
        requery = (verdict.get("requery") or "").strip()
        if not requery:
            return _result(chunks, trace, "no_requery", loops)

        fresh = [c for c in retrieve_fn(requery) if c["chunk_id"] not in seen]
        if not fresh:
            return _result(chunks, trace, "no_progress", loops)
        for c in fresh:
            seen.add(c["chunk_id"])
            chunks.append(c)
        loops += 1


def _result(chunks: list[dict], trace: list[dict], halted_by: str, loops: int) -> dict:
    return {"final_chunks": chunks, "trace": trace,
            "halted_by": halted_by, "loops": loops}
