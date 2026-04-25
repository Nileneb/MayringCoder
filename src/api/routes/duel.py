from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_workspace
from src.api.job_queue import make_job as _make_job, _JOBS
from src.api.routes.models import BenchmarkTasksRequest, DuelRequest

router = APIRouter(tags=["duel"])

_ROOT = Path(__file__).parent.parent.parent.parent
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def _judge_default() -> str:
    from src.model_router import ModelRouter
    return ModelRouter(_OLLAMA_URL).resolve("analysis")


async def _run_duel(job_id: str, request: DuelRequest, workspace_id: str) -> None:
    """Sequential execution of Pi-task on both models, optional judge + baseline."""
    import json as _json
    from src.agents.pi import run_task_with_memory
    _repo_slug = request.repo_slug or os.getenv("PI_REPO_SLUG", "")
    loop = asyncio.get_event_loop()

    def _run_one(model: str, disable_memory: bool = False) -> tuple[str, float]:
        t0 = time.monotonic()
        try:
            out = run_task_with_memory(
                task=request.task,
                ollama_url=_OLLAMA_URL,
                model=model,
                repo_slug=_repo_slug,
                system_prompt=request.system_prompt,
                timeout=request.timeout,
                disable_memory=disable_memory,
            )
        except Exception as exc:
            out = f"[Fehler] {exc}"
        return out, round((time.monotonic() - t0) * 1000, 1)

    def _judge(task: str, answer_a: str, answer_b: str, judge_model: str) -> dict:
        from src import ollama_client as _oc
        _prompt = (
            f"Aufgabe: {task}\n\n"
            f"Antwort A:\n{answer_a[:1500]}\n\n"
            f"Antwort B:\n{answer_b[:1500]}\n\n"
            "Bewerte beide Antworten sachlich. Antworte NUR mit validem JSON (kein Markdown, keine Erklärung):\n"
            '{"winner":"A","score_a":8,"score_b":6,"reasoning":"..."}'
        )
        try:
            raw = _oc.generate(
                _OLLAMA_URL, judge_model, _prompt,
                num_predict=512, keep_alive="0",
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            return _json.loads(raw)
        except Exception as exc:
            return {"winner": "tie", "score_a": 0, "score_b": 0, "reasoning": f"Judge-Fehler: {exc}"}

    job = _JOBS.get(job_id)
    if not job:
        return
    job["model_a"] = request.model_a
    job["model_b"] = request.model_b

    job["progress"] = "running_a"
    result_a, time_a = await loop.run_in_executor(None, _run_one, request.model_a, False)
    job["result_a"] = result_a
    job["time_a_ms"] = time_a

    job["progress"] = "running_b"
    result_b, time_b = await loop.run_in_executor(None, _run_one, request.model_b, False)
    job["result_b"] = result_b
    job["time_b_ms"] = time_b

    if request.no_memory_baseline:
        job["progress"] = "running_baseline_a"
        baseline_a, btime_a = await loop.run_in_executor(None, _run_one, request.model_a, True)
        job["baseline_a"] = baseline_a
        job["baseline_time_a_ms"] = btime_a

        job["progress"] = "running_baseline_b"
        baseline_b, btime_b = await loop.run_in_executor(None, _run_one, request.model_b, True)
        job["baseline_b"] = baseline_b
        job["baseline_time_b_ms"] = btime_b

    if request.judge:
        job["progress"] = "judging"
        _judge_model = request.judge_model or _judge_default()
        verdict = await loop.run_in_executor(
            None, _judge, request.task, result_a, result_b, _judge_model
        )
        job["verdict"] = verdict

    job["progress"] = "done"
    job["status"] = "finished"
    job["output"] = f"Duell fertig — A: {time_a}ms, B: {time_b}ms"


@router.post("/duel")
async def trigger_duel(
    request: DuelRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Run the same task on two models sequentially. Poll /jobs/{id} for results."""
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="task required")
    if not request.model_a or not request.model_b:
        raise HTTPException(status_code=400, detail="model_a and model_b required")
    job_id = _make_job(workspace_id)
    asyncio.create_task(_run_duel(job_id, request, workspace_id))
    return {"job_id": job_id, "status": "started", "model_a": request.model_a, "model_b": request.model_b}


@router.post("/benchmark/tasks")
async def benchmark_tasks(
    request: BenchmarkTasksRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Run task suite on two models, score each answer and return comparison report."""
    import json as _json
    import yaml
    from pathlib import Path as _Path
    from src.agents.pi import run_task_with_memory
    from src import ollama_client as _oc

    suite_path = _Path(__file__).parent.parent.parent.parent / "benchmarks" / "task_suite.yaml"
    if not suite_path.exists():
        raise HTTPException(status_code=404, detail="task_suite.yaml not found")

    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    tasks = suite.get("tasks", [])
    if request.category:
        tasks = [t for t in tasks if t.get("category") == request.category]
    if not tasks:
        raise HTTPException(status_code=404, detail=f"No tasks for category={request.category!r}")

    _repo_slug = request.repo_slug or os.getenv("PI_REPO_SLUG", "")
    _judge_model = request.judge_model or _judge_default()
    results = []

    def _score_answer(task_text: str, answer: str, keywords: list[str]) -> dict:
        hit = sum(1 for kw in keywords if kw.lower() in answer.lower())
        keyword_score = round(hit / len(keywords), 2) if keywords else 1.0
        return {"keyword_hits": hit, "keyword_total": len(keywords), "keyword_score": keyword_score}

    def _judge(task_text: str, answer_a: str, answer_b: str) -> dict:
        _prompt = (
            f"Aufgabe: {task_text}\n\n"
            f"Antwort A:\n{answer_a[:1200]}\n\n"
            f"Antwort B:\n{answer_b[:1200]}\n\n"
            "Bewerte beide Antworten sachlich. Antworte NUR mit validem JSON:\n"
            '{"winner":"A","score_a":8,"score_b":6,"reasoning":"..."}'
        )
        try:
            raw = _oc.generate(_OLLAMA_URL, _judge_model, _prompt, num_predict=512, keep_alive="0")
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            return _json.loads(raw)
        except Exception as exc:
            return {"winner": "tie", "score_a": 0, "score_b": 0, "reasoning": f"Judge-Fehler: {exc}"}

    loop = asyncio.get_event_loop()

    for task_def in tasks:
        tid = task_def.get("id", "?")
        task_text = task_def.get("task", "")
        keywords = task_def.get("expected_keywords", [])
        requires_memory = task_def.get("requires_memory", False)

        t0 = time.monotonic()
        try:
            ans_a = await loop.run_in_executor(
                None,
                lambda: run_task_with_memory(
                    task=task_text, ollama_url=_OLLAMA_URL, model=request.model_a,
                    repo_slug=_repo_slug, timeout=request.timeout,
                ),
            )
        except Exception:
            ans_a = "[Fehler] Modell nicht verfügbar"
        time_a = round((time.monotonic() - t0) * 1000, 1)

        t0 = time.monotonic()
        try:
            ans_b = await loop.run_in_executor(
                None,
                lambda: run_task_with_memory(
                    task=task_text, ollama_url=_OLLAMA_URL, model=request.model_b,
                    repo_slug=_repo_slug, timeout=request.timeout,
                ),
            )
        except Exception:
            ans_b = "[Fehler] Modell nicht verfügbar"
        time_b = round((time.monotonic() - t0) * 1000, 1)

        score_a = _score_answer(task_text, ans_a, keywords)
        score_b = _score_answer(task_text, ans_b, keywords)
        verdict = await loop.run_in_executor(None, _judge, task_text, ans_a, ans_b)

        results.append({
            "task_id": tid,
            "category": task_def.get("category"),
            "requires_memory": requires_memory,
            "model_a": request.model_a,
            "model_b": request.model_b,
            "answer_a": ans_a[:500],
            "answer_b": ans_b[:500],
            "score_a": score_a,
            "score_b": score_b,
            "time_a_ms": time_a,
            "time_b_ms": time_b,
            "verdict": verdict,
        })

    wins_a = sum(1 for r in results if r["verdict"].get("winner") == "A")
    wins_b = sum(1 for r in results if r["verdict"].get("winner") == "B")
    avg_score_a = round(sum(r["verdict"].get("score_a", 0) for r in results) / len(results), 2)
    avg_score_b = round(sum(r["verdict"].get("score_b", 0) for r in results) / len(results), 2)

    return {
        "workspace_id": workspace_id,
        "tasks_run": len(results),
        "model_a": request.model_a,
        "model_b": request.model_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "avg_score_a": avg_score_a,
        "avg_score_b": avg_score_b,
        "results": results,
    }
