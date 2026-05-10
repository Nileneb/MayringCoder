#!/usr/bin/env python3
"""Replay alte prompts gegen die memory-pipeline um feedback-daten zu generieren.

User-Auftrag (rating-bootstrap): nach dem wipe von 2125 binary-feedback-rows
brauchen wir frische rating 1..5-daten damit reranker-v2 trainierbar wird.
Statt zu warten bis user-traffic akkumuliert, replay'n wir bekannte prompts
(closed issues, benchmark-suite) durch die existing pipeline:

    prompt → categorize → /memory/search → ollama answer → LLM-judge → POST feedback

Source-options:
    --source issues       — geschlossene github-issues aus dem repo
    --source benchmarks   — benchmarks/task_suite.yaml (24 vordefinierte tasks)
    --source conversations — conversation-summary-chunks aus der DB (TBD)

Beispiele:
    # Trockenlauf: zeigt was generiert würde, ohne POST
    python tools/replay_feedback.py --source issues --limit 5 --dry-run

    # Live: 30 issues replayed
    python tools/replay_feedback.py --source issues --limit 30

    # Benchmarks (kleinste batch, sicher zum testen)
    python tools/replay_feedback.py --source benchmarks --limit 24
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


API = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
ANSWER_MODEL = os.environ.get("MAYRING_REPLAY_MODEL", "mistral:7b-instruct")
JUDGE_MODEL = os.environ.get("MAYRING_JUDGE_MODEL", "mistral:7b-instruct")
JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")


def _load_token() -> str:
    try:
        with open(JWT_FILE) as f:
            return f.read().strip()
    except OSError:
        sys.exit(f"missing JWT at {JWT_FILE} — run tools/oauth_install.py first")


def _http(method: str, url: str, token: str, body: dict | None = None,
          timeout: float = 30.0) -> tuple[int, dict | None]:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, None
    except (OSError, json.JSONDecodeError):
        return 0, None


def _ollama_generate(prompt: str, *, model: str, timeout: float = 60.0,
                     system: str | None = None) -> str:
    body: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 600, "temperature": 0.2},
        "think": False,
    }
    if system:
        body["system"] = system
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (json.loads(resp.read()).get("response") or "").strip()
    except (OSError, json.JSONDecodeError, urllib.error.URLError) as exc:
        sys.stderr.write(f"  ollama-fail: {type(exc).__name__}: {exc}\n")
        return ""


# ---------- prompt sources ----------

def load_issues(limit: int, repo: str = "Nileneb/MayringCoder") -> list[dict]:
    """Closed issues via gh CLI — title + body."""
    out = subprocess.run(
        ["gh", "issue", "list", "--repo", repo, "--state", "closed",
         "--limit", str(limit), "--json", "number,title,body"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        sys.exit(f"gh issue list failed: {out.stderr}")
    return json.loads(out.stdout)


def load_benchmarks(limit: int) -> list[dict]:
    """Read tasks from benchmarks/task_suite.yaml."""
    path = ROOT / "benchmarks" / "task_suite.yaml"
    text = path.read_text(encoding="utf-8")
    # crude yaml-task-parse — vermeidet pyyaml-dep
    tasks: list[dict] = []
    cur: dict = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- id:"):
            if cur:
                tasks.append(cur)
            cur = {"id": s.split(":", 1)[1].strip()}
        elif s.startswith("task:"):
            cur["task"] = s.split(":", 1)[1].strip().strip('"')
        elif s.startswith("category:"):
            cur["category"] = s.split(":", 1)[1].strip()
    if cur:
        tasks.append(cur)
    return [t for t in tasks if t.get("task")][:limit]


# ---------- pipeline steps ----------

def search_memory(query: str, token: str) -> list[dict]:
    """POST /memory/search → top chunks with text.

    llm_prefilter=False — qwen3.5:2b ist nicht auf dem server, würde sonst
    pre-filter mit allen scores=0.5 zurückgeben → keine candidates.
    """
    code, body = _http("POST", f"{API}/memory/search", token, body={
        "query": query[:600],
        "top_k": 6,
        "include_text": True,
        "char_budget": 2400,
        "llm_prefilter": False,
    }, timeout=20.0)
    if code != 200 or not body:
        return []
    return body.get("results") or []


def pi_task_answer(prompt: str, token: str, *, timeout: float = 180.0,
                   repo_slug: str | None = None) -> tuple[str, list[dict]]:
    """Call /pi-task — der Pi-Agent macht selbst search_memory + ollama-call.

    Vorteil ggü. lokalem ollama-call:
    - Pi-Agent ist der real-traffic-server: identisches retrieval-pfad-
      verhalten wie bei normaler user-anfrage.
    - läuft remote auf u-server (GPU verfügbar via three.linn.games), kein
      lokales modell-bootup nötig.
    - kontext-injektion ist eingebaut (pi.py::run_task_with_memory).

    Returnt (answer, chunks_used). chunks_used wird aus der separaten
    search-spur extrahiert weil pi-task die nicht exposed.
    """
    # Pi-task selbst (answer-gen)
    code, body = _http("POST", f"{API}/pi-task", token, body={
        "task": prompt[:2000],
        "repo_slug": repo_slug,
        "timeout": timeout,
    }, timeout=timeout + 30)
    if code != 200 or not body:
        return "", []
    answer = (body.get("content") or body.get("result") or "").strip()
    # Hole gleiche chunks via /memory/search separat — pi-task injektiert
    # die intern aber exposed sie nicht im response. Identische query =
    # identische top-k chunks (deterministisches retrieval).
    chunks = search_memory(prompt, token)
    return answer, chunks


_JUDGE_SYSTEM = (
    "You score memory-chunk relevance. Output ratings only."
)


def judge_chunks(chunks: list[dict], user_prompt: str, answer: str) -> dict[str, str]:
    """For each chunk: rating 1..5 based on actual content-usage in answer."""
    chunks_with_text = [c for c in chunks if c.get("text")]
    if not chunks_with_text or not answer:
        return {}
    numbered = "\n".join(
        f"[{i+1}] source={(c.get('source_id') or '?')[:80]}\n"
        f"    text: {(c.get('text') or '')[:500].replace(chr(10), ' ')}"
        for i, c in enumerate(chunks_with_text)
    )
    prompt = (
        f"User asked:\n{user_prompt[:500]}\n\n"
        f"Assistant answered:\n{answer[:1500]}\n\n"
        f"Available memory chunks (numbered):\n{numbered}\n\n"
        "For each chunk, rate how much the answer USED its content:\n"
        "1=irrelevant or misleading  2=barely related  3=loosely relevant\n"
        "4=clearly used in parts  5=primary source\n\n"
        f"Respond with EXACTLY {len(chunks_with_text)} comma-separated "
        "ratings (1-5), in order. Example: 5,2,3,4\n\nAnswer:"
    )
    raw = _ollama_generate(prompt, model=JUDGE_MODEL, system=_JUDGE_SYSTEM,
                           timeout=30.0)
    out: dict[str, str] = {}
    tokens = [t for t in re.split(r"[,\s]+", raw) if t]
    for i, c in enumerate(chunks_with_text):
        if i >= len(tokens):
            break
        m = re.search(r"[1-5]", tokens[i])
        if m:
            out[c["chunk_id"]] = m.group(0)
    return out


def post_feedback(chunk_id: str, rating: str, metadata: dict, token: str,
                  dry_run: bool) -> bool:
    if dry_run:
        return True
    code, _ = _http("POST", f"{API}/memory/feedback", token, body={
        "chunk_id": chunk_id,
        "signal": rating,
        "metadata": metadata,
    }, timeout=10.0)
    return code == 200


# ---------- orchestration ----------

def replay(prompts: list[tuple[str, str]], *, source: str, dry_run: bool,
           sleep_s: float = 1.0, use_pi: bool = True,
           repo_slug: str | None = None) -> dict:
    token = _load_token()
    bucket: Counter = Counter()
    posted = 0
    skipped = 0
    failed = 0

    for i, (key, prompt) in enumerate(prompts, start=1):
        sys.stdout.write(f"\n[{i}/{len(prompts)}] {source}={key}\n")
        sys.stdout.flush()

        if use_pi:
            # Pi-Agent macht search + answer in einem schritt → realistisch
            sys.stdout.write("  via pi_task (remote, kein Opus-token-burn)\n")
            sys.stdout.flush()
            answer, chunks = pi_task_answer(prompt, token, repo_slug=repo_slug)
        else:
            chunks = search_memory(prompt, token)
            if not chunks:
                sys.stdout.write("  no chunks — skip\n")
                skipped += 1
                continue
            sys.stdout.write(f"  {len(chunks)} chunks → ollama-answer ...\n")
            sys.stdout.flush()
            ctx = "\n\n".join(
                f"### {(c.get('source_id') or '?')[:60]}\n{(c.get('text') or '')[:400]}"
                for c in chunks[:5]
            )
            answer = _ollama_generate(
                f"Context:\n{ctx}\n\nQuestion: {prompt}\n\nAnswer briefly using "
                "the context where relevant. If context is irrelevant, say so.",
                model=ANSWER_MODEL, timeout=60.0,
            )
        if not answer:
            sys.stdout.write("  empty answer — skip\n")
            skipped += 1
            continue
        if not chunks:
            sys.stdout.write("  no chunks — skip\n")
            skipped += 1
            continue
        sys.stdout.write(f"  answer={len(answer)} chars → judging...\n")
        sys.stdout.flush()
        ratings = judge_chunks(chunks, prompt, answer)
        if not ratings:
            sys.stdout.write("  no judge-ratings — skip\n")
            skipped += 1
            continue
        for cid, r in ratings.items():
            metadata = {
                "task": prompt[:200].replace("\n", " "),
                "method": "replay",
                "source": f"{source}:{key}",
                "judge_model": JUDGE_MODEL,
            }
            ok = post_feedback(cid, r, metadata, token, dry_run)
            if ok:
                posted += 1
                bucket[r] += 1
            else:
                failed += 1
        sys.stdout.write(f"  posted ratings: {dict(bucket)}\n")
        sys.stdout.flush()
        if sleep_s and not dry_run:
            time.sleep(sleep_s)
    return {
        "prompts_total": len(prompts),
        "posted": posted,
        "skipped": skipped,
        "failed": failed,
        "rating_buckets": dict(bucket),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["issues", "benchmarks", "conversations"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--repo", default="Nileneb/MayringCoder",
                    help="GH-repo when --source=issues")
    ap.add_argument("--dry-run", action="store_true",
                    help="run pipeline aber kein POST /memory/feedback")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds between prompts (rate-limit / ollama-load)")
    ap.add_argument("--use-pi", action="store_true", default=True,
                    help="route answer-gen über Pi-Agent (/pi-task) — "
                         "billiger + identischer pfad wie real traffic")
    ap.add_argument("--no-pi", dest="use_pi", action="store_false",
                    help="lokales ollama statt pi-task (offline-fallback)")
    ap.add_argument("--repo-slug", default=None,
                    help="repo_slug für pi_task memory-scope")
    args = ap.parse_args()

    if args.source == "issues":
        issues = load_issues(args.limit, args.repo)
        prompts = [
            (f"#{i['number']}", f"{i['title']}\n\n{i.get('body','') or ''}")
            for i in issues
        ]
    elif args.source == "benchmarks":
        tasks = load_benchmarks(args.limit)
        prompts = [(t["id"], t["task"]) for t in tasks]
    else:
        sys.exit("--source=conversations TBD")

    if not prompts:
        sys.exit("no prompts loaded")

    backend = "pi_task" if args.use_pi else f"local-ollama({ANSWER_MODEL})"
    print(f"replay: {len(prompts)} prompts from {args.source} "
          f"(dry_run={args.dry_run}, backend={backend})")
    summary = replay(prompts, source=args.source, dry_run=args.dry_run,
                     sleep_s=args.sleep, use_pi=args.use_pi,
                     repo_slug=args.repo_slug)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
