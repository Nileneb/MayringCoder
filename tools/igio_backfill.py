"""IGIO backfill driver.

Triggers ``POST /stats/igio-backfill`` on production and polls the job until
done. Local dev / on-demand counterpart to the hourly workflow at
``.github/workflows/igio-backfill.yml``.

Auth: reads ``MCP_SERVICE_TOKEN`` from env. On the production server, sourcing
``$HOME/app.linn.games/.env`` is the canonical way:

    export $(grep '^MCP_SERVICE_TOKEN=' ~/app.linn.games/.env | xargs)
    python tools/igio_backfill.py --limit 300

Without ``--api`` the default is ``https://mcp.linn.games``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request


def _req(method: str, url: str, token: str, *, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, method=method, headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default=os.getenv("MAYRING_API",
                                              "https://mcp.linn.games"))
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--workspace-id", default="")
    p.add_argument("--poll-seconds", type=int, default=5)
    p.add_argument("--max-wait", type=int, default=1500)
    args = p.parse_args()

    token = os.getenv("MCP_SERVICE_TOKEN", "")
    if not token:
        print("Fehler: MCP_SERVICE_TOKEN nicht gesetzt.", file=sys.stderr)
        return 2

    qs = f"limit={args.limit}&min_confidence={args.min_confidence}"
    if args.workspace_id:
        qs += f"&workspace_id={args.workspace_id}"
    started = _req("POST", f"{args.api}/stats/igio-backfill?{qs}", token)
    job_id = started.get("job_id")
    if not job_id:
        print(f"Kein job_id zurückgekommen: {started}", file=sys.stderr)
        return 1
    print(f"queued: job_id={job_id}")

    deadline = time.time() + args.max_wait
    while time.time() < deadline:
        time.sleep(args.poll_seconds)
        state = _req("GET", f"{args.api}/stats/igio-backfill/{job_id}", token)
        status = state.get("status")
        print(f"  status={status} picked={state.get('picked')} "
              f"persisted={state.get('persisted')}")
        if status in ("done", "error"):
            print(json.dumps(state, indent=2))
            cov = _req("GET", f"{args.api}/stats/igio-coverage", token)
            print(f"coverage: ratio={cov.get('ratio')} "
                  f"({cov.get('with_axis')}/{cov.get('total_active')})")
            return 0 if status == "done" else 1
    print("::warning:: job did not finish in window", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
