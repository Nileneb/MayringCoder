"""Concurrency load test for /memory/search (the session-start hook fires 3).
Usage: python tools/loadtest_search.py --n 3 --api https://mcp.linn.games"""
import argparse, concurrent.futures as cf, json, os, statistics, time, urllib.request
from pathlib import Path

def _jwt() -> str:
    return Path(os.getenv("MAYRING_HOOK_JWT", str(Path.home()/".config/mayring/hook.jwt"))).read_text().strip()

def _search(api, jwt, q):
    body = json.dumps({"query": q, "top_k": 8, "include_text": False}).encode()
    req = urllib.request.Request(f"{api}/memory/search", data=body,
        headers={"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=40) as r: r.read(); return time.monotonic()-t0, 200
    except Exception as e: return time.monotonic()-t0, getattr(e, "code", 0)

def _health(api):
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(f"{api}/health", timeout=10) as r: r.read(); return time.monotonic()-t0
    except Exception: return time.monotonic()-t0

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=3)
    p.add_argument("--api", default=os.getenv("MAYRING_API_URL", "https://mcp.linn.games"))
    a = p.parse_args(); jwt = _jwt(); api = a.api.rstrip("/")
    qs = [f"loadtest lens {i}" for i in range(a.n)]
    with cf.ThreadPoolExecutor(max_workers=a.n+1) as ex:
        h = ex.submit(_health, api)
        futs = [ex.submit(_search, api, jwt, q) for q in qs]
        results = [f.result() for f in futs]; health = h.result()
    lat = [t for t, _ in results]; codes = [c for _, c in results]
    print(f"n={a.n}  codes={codes}")
    print(f"  per-req: {[round(t,2) for t in lat]}")
    print(f"  p50={statistics.median(lat):.2f}s  max={max(lat):.2f}s  health-during={health:.2f}s")
    ok = all(c == 200 for c in codes) and max(lat) < 5.0 and health < 0.2
    print("  PASS" if ok else "  FAIL (target: all 200, max<5s, health<0.2s)")

if __name__ == "__main__": main()
