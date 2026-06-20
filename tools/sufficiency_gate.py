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

TASK_DISTILL_RUBRIC = """Du destillierst einen rohen Chat-Prompt zu EINER praezisen,
abgeschlossenen Aufgabe (Task), die als Suchanker fuers Memory taugt. Ignoriere
Emotion/Fuellwoerter ("JAAAA", "mach mal", Flueche) — extrahiere die SACHLICHE
Arbeitseinheit. Wenn der Prompt keine sachliche Aufgabe enthaelt, gib "" zurueck.

Antworte mit STRIKTEM JSON: {"task": "praezise Aufgabe als Suchstring, oder \\"\\""}"""

DECOMPOSE_RUBRIC = """Du zerlegst eine Aufgabe in die Teilfragen, die man ans Memory
stellen muss, um sie vollstaendig zu loesen. Je Frage ein konkreter Suchstring.
2 bis 4 Fragen, keine Redundanz.

Antworte mit STRIKTEM JSON: {"questions": ["frage 1", "frage 2", ...]}"""

ANSWERED_RUBRIC = """Du pruefst, ob eine FRAGE durch die gegebenen CHUNKS ausreichend
beantwortet ist. Streng: answered=true NUR wenn die Chunks die Frage konkret beantworten.

Antworte mit STRIKTEM JSON: {"answered": true|false}"""


def _ollama_url(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")


def _chat_json(url: str, model: str, system: str, user: str,
               think: bool = False, timeout: float = _DEFAULT_TIMEOUT) -> dict:
    """One format:json chat turn → parsed dict. Raises on transport/parse error
    (callers decide the fail-safe). Shared by the task/decompose/answered judges."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False, "format": "json", "think": bool(think),
        "options": {"num_predict": 512, "temperature": 0}, "keep_alive": _KEEP_ALIVE,
    }
    resp = httpx.post(f"{url}/api/chat", json=body, timeout=timeout)
    resp.raise_for_status()
    content = (resp.json().get("message", {}) or {}).get("content", "")
    return _loads_lenient(content)


def derive_task(prompt: str, ollama_url: str | None = None,
                model: str | None = None, timeout: float = _DEFAULT_TIMEOUT) -> str:
    """Distill a raw prompt into a precise task string for use as the retrieval
    anchor. Fail-safe: on error / empty → return the raw prompt (never worse than
    the status quo, which IS the raw prompt)."""
    url = _ollama_url(ollama_url)
    try:
        data = _chat_json(url, model or _DEFAULT_MODEL, TASK_DISTILL_RUBRIC,
                          (prompt or "").strip()[:1500], timeout=timeout)
        task = str(data.get("task") or "").strip()
        return task or (prompt or "").strip()
    except Exception as exc:  # noqa: BLE001 — degrade to raw prompt, logged
        _log.warning("task distillation failed (%s) — using raw prompt", exc)
        return (prompt or "").strip()


def decompose_questions(task: str, ollama_url: str | None = None,
                        model: str | None = None, max_q: int = 4,
                        timeout: float = _DEFAULT_TIMEOUT) -> list[str]:
    """Break a task into the sub-questions to ask the memory. Fail-safe: on error
    → [task] (a single question = the task itself)."""
    url = _ollama_url(ollama_url)
    try:
        data = _chat_json(url, model or _DEFAULT_MODEL, DECOMPOSE_RUBRIC,
                          (task or "").strip()[:1500], timeout=timeout)
        qs = data.get("questions") or []
        qs = [str(q).strip() for q in qs if str(q).strip()][:max_q]
        return qs or [task.strip()]
    except Exception as exc:  # noqa: BLE001 — degrade to the task itself, logged
        _log.warning("question decomposition failed (%s) — using task as sole question", exc)
        return [(task or "").strip()]


def is_answered(question: str, chunks: list[dict], ollama_url: str | None = None,
                model: str | None = None, think: bool = False,
                timeout: float = _DEFAULT_TIMEOUT) -> bool:
    """Does the chunk set answer `question`? Fail-safe: error → True (don't loop
    forever on a broken judge)."""
    url = _ollama_url(ollama_url)
    try:
        data = _chat_json(url, model or _DEFAULT_MODEL, ANSWERED_RUBRIC,
                          "FRAGE: " + (question or "") + "\n\nCHUNKS:\n"
                          + "\n".join(f"[{i}] {_chunk_text(c)}" for i, c in enumerate(chunks)),
                          think=think, timeout=timeout)
        return bool(data.get("answered", True))
    except Exception as exc:  # noqa: BLE001 — fail-safe, logged
        _log.warning("answered-judge failed (%s) — treating as answered", exc)
        return True


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


def _parallel_map(fn: Callable, items: list, workers: int) -> list:
    """map(fn, items), concurrently when workers>1. Order-preserving (ex.map), so
    results stay deterministic. Used to fan the sub-question retrievals + answered-
    checks of ONE loop round out in parallel — the sub-questions are independent,
    so a round drops from N×latency to ~max(latency) without skipping any question."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


def run_task_loop(
    task: str,
    retrieve_fn: Callable[[str], list[dict]],
    ollama_url: str | None = None,
    *,
    model: str | None = None,
    think: bool = False,
    max_loops: int = 3,
    budget_s: float = 30.0,
    max_q: int = 4,
    seed_with_task: bool = True,
    parallelism: int = 4,
    decompose_fn: Callable[[str], list[str]] | None = None,
    answered_fn: Callable[..., bool] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict:
    """Task-anchored, question-decomposition retrieval loop (the user's redesign):
    break the TASK into sub-questions, ask each to the memory, keep going until
    every question is answered OR no new info arrives. The SEMANTIC halt
    (all_answered / no_progress) leads; the deterministic bounds (cap, budget) are
    the backstop against non-convergence (OpenMythos' 'drift into noise').

    Why this beats single-shot: a task fans out into several targeted queries →
    broader, more precise recall than one raw query. Halt is defined against the
    TASK (closeable), not a goal (never finished)."""
    decompose = decompose_fn or (lambda t: decompose_questions(
        t, ollama_url, model=model, max_q=max_q))
    answered = answered_fn or (lambda q, ch: is_answered(
        q, ch, ollama_url, model=model, think=think))
    now = clock or time.monotonic
    start = now()

    # WHY(2026-06-20 eval): seed with the TASK query itself, then ADD sub-questions.
    # Without the seed the loop discards the (good) task-anchor retrieval and relies
    # only on gemma's sub-questions, which drift → recall drops below single-shot.
    # The task query is the primary anchor; decomposition only broadens it.
    sub = decompose(task)
    questions: list[str] = []
    for q in ([task] if seed_with_task else []) + sub:
        if q and q not in questions:
            questions.append(q)
    open_qs = list(questions)
    chunks: list[dict] = []
    seen: set = set()
    trace: list[dict] = []
    loops = 0

    while True:
        # Fan the round's sub-questions out in parallel — they're independent, so a
        # round costs ~max(latency) not N×latency. Chroma is a server (thread-safe
        # under concurrent queries); the retrieve_fn must give each thread its own
        # SQLite connection (sqlite objects are not thread-safe). Order-preserving.
        fetched = _parallel_map(retrieve_fn, open_qs, parallelism)
        round_fresh = 0
        for res in fetched:
            for c in res:
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    chunks.append(c)
                    round_fresh += 1
        answers = _parallel_map(lambda q: answered(q, chunks), open_qs, parallelism)
        still_open = [q for q, a in zip(open_qs, answers) if not a]
        trace.append({"loop": loops, "open_before": len(open_qs),
                      "open_after": len(still_open), "fresh_chunks": round_fresh})
        open_qs = still_open

        if not open_qs:
            return _task_result(chunks, questions, trace, "all_answered", loops)
        if loops >= max_loops:
            return _task_result(chunks, questions, trace, "cap", loops)
        if (now() - start) > budget_s:
            return _task_result(chunks, questions, trace, "budget", loops)
        if round_fresh == 0:
            return _task_result(chunks, questions, trace, "no_progress", loops)
        loops += 1


def _task_result(chunks, questions, trace, halted_by, loops) -> dict:
    return {"final_chunks": chunks, "questions": questions, "trace": trace,
            "halted_by": halted_by, "loops": loops,
            "open_questions": [t for t in (trace[-1:] or [{}])][0].get("open_after", 0)}
