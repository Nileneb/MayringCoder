#!/usr/bin/env python3
"""Poll the status of one or more MayringCoder ingest jobs.

Reads MAYRING_API_URL + MAYRING_JWT (or `gh auth token` fallback) from env and
polls GET /jobs/{job_id} for each job_id passed on the command line.

Usage:
    python tools/ingest_status.py <job_id> [<job_id> ...]
    python tools/ingest_status.py --watch <job_id>          # poll every 5s until done
    python tools/ingest_status.py --recent                  # list recent jobs in workspace

Env:
    MAYRING_API_URL    Base URL (default: https://mcp.linn.games)
    MAYRING_JWT        Bearer token (required for tenant-scoped endpoints)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    import httpx
except ImportError:
    sys.exit("httpx required: pip install httpx")


def _api_base() -> str:
    return os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")


def _headers() -> dict[str, str]:
    jwt = os.getenv("MAYRING_JWT", "").strip()
    if not jwt:
        sys.exit("MAYRING_JWT not set — export it before running")
    return {"Authorization": f"Bearer {jwt}"}


def _fmt_progress(progress: dict | None) -> str:
    if not progress:
        return ""
    cur = progress.get("current", 0)
    tot = progress.get("total", 0)
    if tot:
        pct = int(100 * cur / tot)
        return f"{cur}/{tot} ({pct}%)"
    return f"{cur}"


def _fmt_stages(stages: dict | None) -> str:
    if not stages:
        return ""
    return " · ".join(f"{name}@{(s.get('ts') or '')[11:19]}" for name, s in stages.items())


def show_job(job_id: str) -> dict:
    url = f"{_api_base()}/jobs/{job_id}"
    resp = httpx.get(url, headers=_headers(), timeout=10.0)
    if resp.status_code == 404:
        print(f"[{job_id}] not found (wrong tenant or job expired)")
        return {"status": "not_found"}
    resp.raise_for_status()
    job = resp.json()
    status = job.get("status", "?")
    started = (job.get("started_at") or "")[:19]
    progress = _fmt_progress(job.get("progress"))
    stages = _fmt_stages(job.get("stages"))
    v2 = job.get("v2_jobs") or {}
    line = f"[{job_id}] {status:<8} started={started}"
    if progress:
        line += f"  progress={progress}"
    if stages:
        line += f"  stages={stages}"
    if v2:
        line += f"  v2_jobs={list(v2.keys())}"
    print(line)
    if status == "error":
        out = (job.get("output") or "")[-2000:]
        if out:
            print("--- error output ---")
            print(out)
    return job


def list_recent() -> None:
    url = f"{_api_base()}/stats/summary"
    resp = httpx.get(url, timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    jobs = data.get("recent_jobs") or []
    if not jobs:
        print("(no recent jobs in stats)")
        return
    print(f"{'job_id':<10} {'status':<8} {'started':<20}")
    for j in jobs:
        jid = j.get("job_id", "?")
        st = j.get("status", "?")
        started = (j.get("started_at") or "")[:19]
        print(f"{jid:<10} {st:<8} {started:<20}")


def watch(job_id: str, interval: float = 5.0) -> None:
    while True:
        job = show_job(job_id)
        if job.get("status") in {"done", "error", "not_found"}:
            return
        time.sleep(interval)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job_ids", nargs="*", help="Job IDs to poll")
    p.add_argument("--watch", action="store_true", help="Poll every 5s until done (single job_id only)")
    p.add_argument("--recent", action="store_true", help="List recent jobs from /stats/summary (no auth)")
    p.add_argument("--json", action="store_true", help="Emit raw JSON instead of formatted lines")
    args = p.parse_args()

    if args.recent:
        list_recent()
        return

    if not args.job_ids:
        p.error("provide at least one job_id, or use --recent")

    if args.watch:
        if len(args.job_ids) != 1:
            p.error("--watch requires exactly one job_id")
        watch(args.job_ids[0])
        return

    for jid in args.job_ids:
        if args.json:
            url = f"{_api_base()}/jobs/{jid}"
            resp = httpx.get(url, headers=_headers(), timeout=10.0)
            print(json.dumps({"job_id": jid, **resp.json()}, indent=2))
        else:
            show_job(jid)


if __name__ == "__main__":
    main()
