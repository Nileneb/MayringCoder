#!/usr/bin/env python3
"""MayringCoder CLI — **HTTP-Client** an den laufenden mayring-api-Server.

Rollen-Abgrenzung (siehe Issue #66 Phase 5):
  - `checker.py`          — REMOTE-CLIENT. Feuert POST /populate, /analyze,
                            /pi-task etc. an den API-Server und poll-t dann
                            /jobs/{id}. Kein direkter Zugriff auf memory.db
                            oder Ollama. Auth: RS256-JWT aus env
                            MAYRING_JWT (fällt auf Legacy SANCTUM_TOKEN
                            zurück, falls ältere Shell-Skripte davon leben).
  - `src/pipeline.py`     — Shim (re-export) für src/workflows/.
  - `python -m src.cli`   — LOCAL-EXECUTOR, läuft direkt als subprocess im
                            Server-Container (run_checker_job), greift
                            direkt auf memory.db, chromadb, ollama zu.

Wenn du lokal ohne Server ingesten willst → `python -m src.cli --repo ...`.
Wenn du remote gegen mcp.linn.games arbeiten willst → `checker.py --repo ...`.
"""
from __future__ import annotations

import os
import sys
import time

import httpx
from dotenv import load_dotenv


def _api_url() -> str:
    return os.getenv("API_URL", "http://localhost:8080").rstrip("/")


def _call(method: str, path: str, *, json: dict | None = None, token: str = "") -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = httpx.request(method, f"{_api_url()}{path}", json=json, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        print(f"Fehler: API-Server nicht erreichbar unter {_api_url()}")
        print("Starte den Server mit: docker compose up -d  oder  python -m src.main")
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(f"API-Fehler {exc.response.status_code}: {exc.response.text[:200]}")
        sys.exit(1)


def _poll(job_id: str, token: str = "") -> None:
    while True:
        r = _call("GET", f"/jobs/{job_id}", token=token)
        if r["status"] in ("done", "error"):
            print(r.get("output", "").rstrip())
            if r["status"] == "error":
                sys.exit(1)
            return
        time.sleep(2)


def main() -> None:
    load_dotenv()

    import argparse
    p = argparse.ArgumentParser(description="MayringCoder CLI")
    p.add_argument("--repo")
    p.add_argument("--model")
    p.add_argument("--mode", choices=["analyze", "overview", "turbulence"], default="analyze")
    p.add_argument("--full", action="store_true")
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--no-pi", action="store_true")
    p.add_argument("--budget", type=int)
    p.add_argument("--llm", action="store_true")
    p.add_argument("--populate-memory", action="store_true")
    p.add_argument("--force-reingest", action="store_true")
    p.add_argument("--ingest-issues", metavar="REPO")
    p.add_argument("--issues-state", choices=["open", "closed", "all"], default="open")
    p.add_argument("--pi-task", metavar="TASK")
    p.add_argument("--ingest-papers", metavar="DIR")
    p.add_argument("--workspace-id", default="default")
    p.add_argument("--history", action="store_true")
    p.add_argument("--compare", nargs=2, metavar="RUN_ID")
    p.add_argument("--cleanup", type=int, metavar="N")
    # Legacy flags (accepted but forwarded to server-side handling)
    for flag in ["--dry-run", "--show-selection", "--debug", "--reset", "--no-limit",
                 "--log-training-data", "--cache-by-model", "--adversarial-cost-report",
                 "--embedding-prefilter", "--rag-enrichment", "--use-overview-cache",
                 "--use-turbulence-cache", "--multiview", "--gpu-metrics", "--resolve-model-only",
                 "--memory-categorize"]:
        p.add_argument(flag, action="store_true")
    for flag_val in ["--max-chars", "--batch-size", "--embedding-top-k", "--issues-limit", "--max-images"]:
        p.add_argument(flag_val, type=int)
    for flag_val in ["--batch-delay", "--time-budget", "--embedding-threshold"]:
        p.add_argument(flag_val, type=float)
    for flag_val in ["--run-id", "--prompt", "--codebook", "--codebook-profile", "--export",
                     "--embedding-model", "--embedding-query", "--second-opinion",
                     "--vision-model", "--ingest-images", "--min-confidence"]:
        p.add_argument(flag_val)
    args = p.parse_args()

    repo = args.repo or os.getenv("GITHUB_REPO", "")
    # Auth wurde 2026-04 von Sanctum auf RS256-JWT umgestellt. MAYRING_JWT
    # ist der neue Name; SANCTUM_TOKEN bleibt als Fallback für alte Skripte.
    token = os.getenv("MAYRING_JWT") or os.getenv("SANCTUM_TOKEN", "")
    ws = args.workspace_id

    # --resolve-model-only: delegate to src.pipeline directly (no API needed)
    if args.resolve_model_only:
        from src.model_selector import resolve_model
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        env_model = (os.getenv("OLLAMA_MODEL") or "").strip() or None
        print(resolve_model(ollama_url, args.model, env_model))
        return

    # Local-only operations (read SQLite directly, no API needed)
    if args.history:
        from src.analysis.history import list_runs
        runs = list_runs(repo, workspace_id=ws)
        if not runs:
            print("Keine Run-History vorhanden.")
        else:
            print(f"{'Run-ID':<20} {'Modus':<10} {'Modell':<18} {'Dateien':>7} {'Zeit (s)':>8}  Zeitstempel")
            print("-" * 90)
            for r in runs:
                print(f"{r['run_id']:<20} {r['mode']:<10} {r['model']:<18} "
                      f"{r['files_checked']:>7} {r['timing_seconds']:>8.1f}  {r['timestamp']}")
        return

    if args.compare:
        from src.analysis.history import compare_runs
        try:
            cmp = compare_runs(args.compare[0], args.compare[1], repo, workspace_id=ws)
        except FileNotFoundError as exc:
            print(f"Fehler: {exc}"); sys.exit(1)
        s = cmp["summary"]
        print(f"Vergleich: {cmp['run_a']} → {cmp['run_b']}")
        print(f"  Dateien: {s['files_a']} → {s['files_b']}")
        print(f"  Neu: {s['new_count']}  Behoben: {s['resolved_count']}  Severity geändert: {s['severity_changed_count']}")
        return

    if args.cleanup is not None:
        from src.analysis.history import cleanup_runs
        deleted = cleanup_runs(repo, keep=args.cleanup)
        print(f"{deleted} alte Runs gelöscht, {args.cleanup} behalten.")
        return

    if args.ingest_papers:
        from src.pipeline import run_ingest_paper
        from src.model_selector import resolve_model
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        env_model = (os.getenv("OLLAMA_MODEL") or "").strip() or None
        model = resolve_model(ollama_url, getattr(args, "model", None), env_model)
        result = run_ingest_paper(
            papers_dir=args.ingest_papers,
            ollama_url=ollama_url,
            model=model,
            repo_slug=repo,
            force_reingest=bool(args.force_reingest),
            workspace_id=ws,
        )
        print(f"\n[ingest-papers] Fertig: {result['ingested']} ingested, {result['skipped']} skipped (unverändert), {result['failed']} failed, {result['total']} total.")
        return

    # API calls
    if args.populate_memory:
        r = _call("POST", "/populate", json={"repo": repo, "force_reingest": bool(args.force_reingest)}, token=token)
    elif args.ingest_issues:
        r = _call("POST", "/issues/ingest", json={"repo": args.ingest_issues, "state": args.issues_state, "force_reingest": bool(args.force_reingest)}, token=token)
    elif args.pi_task:
        r = _call("POST", "/pi-task", json={"task": args.pi_task, "repo": repo}, token=token)
        print(r.get("result", ""))
        return
    elif args.mode == "overview":
        r = _call("POST", "/overview", json={"repo": repo}, token=token)
    elif args.mode == "turbulence":
        r = _call("POST", "/turbulence", json={"repo": repo, "llm": bool(args.llm)}, token=token)
    else:
        r = _call("POST", "/analyze", json={
            "repo": repo,
            "full": bool(args.full),
            "adversarial": bool(args.adversarial),
            "no_pi": bool(getattr(args, "no_pi", False)),
            "budget": args.budget,
        }, token=token)

    _poll(r["job_id"], token=token)


if __name__ == "__main__":
    main()
