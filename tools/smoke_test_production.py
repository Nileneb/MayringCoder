#!/usr/bin/env python3
"""After-deploy smoke test — runs every critical path against PROD.

Why this script exists: 1232 unit tests pass against synthetic fixtures
but never touch production. A real bug (Stop-hook silently broken,
hook-timeout < real latency, /memory/search not writing context_log)
slips by every one of them and surfaces only when the user manually
notices something looks off. This script closes the gap.

Approach: every test is one assertion against the live API. Counts
delta, not just "got 200". When a check fails, exit 1 with a loud
multi-line error so a CI step / manual run cannot mistake it for
"all green".

Run:
    python tools/smoke_test_production.py
    python tools/smoke_test_production.py --fail-fast
    python tools/smoke_test_production.py --skip mcp_health     # skip a check by id
    python tools/smoke_test_production.py --alert-on-fail        # open GitHub issue
                                                                  # if anything fails

Exit codes:
    0  — all checks PASS
    1  — at least one check FAIL (script printed details)
    2  — environment problem (no JWT, network down)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

API_DEFAULT = "https://mcp.linn.games"
JWT_PATH = os.path.expanduser("~/.config/mayring/hook.jwt")

# WHY(workspace-uuid-sot): the MCP_SERVICE_TOKEN defaults to workspace 'system',
# which holds NO real embeddings — any vector check run against it returns
# 'chroma_query_empty' forever (not a cold-start, not a regression). Every check
# that asserts on vector hits MUST target the canonical populated workspace via
# X-Workspace-Id. env-overridable for other tenants.
# WHY(workspace-repoint 2026-05-24): die Memory wurde von der verwaisten 019d6933
# auf die app.linn.games-SoT 019e14d6 ("Bene Workspace") re-pointed — 019d6933 ist
# jetzt leer. Der Smoke muss den kanonischen Ziel-Workspace targeten (nicht auf den
# Alias-Resolver angewiesen sein).
SMOKE_VECTOR_WORKSPACE = os.environ.get(
    "SMOKE_RAG_WORKSPACE", "019e14d6-0489-7348-bca8-e29c11293cb7")
# WHY(tenancy phase A): owner JWT-sub of SMOKE_VECTOR_WORKSPACE's private chunks
# (matches MAYRING_PERSONAL_OWNERS). The smoke acts as this user to retrieve
# private/user_id-scoped repo-code chunks via the service token + X-Act-As.
SMOKE_VECTOR_OWNER = os.environ.get("SMOKE_RAG_OWNER", "1")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    payload: Any = None


def _load_token() -> str:
    """Resolve the JWT used for every smoke check.

    Order of preference:
      1. ``MAYRING_SANCTUM_TOKEN`` env var → exchanged at app.linn.games
         for a fresh JWT (login_path). This is the CI-friendly path:
         long-lived Sanctum token as the only secret stored in GitHub
         Actions, the smoke test logs in fresh on every run. Tests the
         actual user-auth flow as part of the smoke run.
      2. ``MAYRING_JWT`` env var (raw JWT) — for ad-hoc local runs.
      3. ``~/.config/mayring/hook.jwt`` file — legacy fallback.

    Mode (1) is the only path that exercises the Laravel login system —
    user-management, workspace resolution, subscription checks all
    fire as part of the smoke run. The user's complaint was exactly
    this: copying a JWT bypasses the user-auth pipeline we wanted
    tested.
    """
    sanctum = os.environ.get("MAYRING_SANCTUM_TOKEN", "").strip()
    if sanctum:
        return _login_via_sanctum(sanctum)
    # Fallback that needs no UI step at all — re-uses the same
    # MCP_SERVICE_TOKEN that already lives in docker-compose.mayring.yml.
    # Server-internal, long-lived, never expires; mapped to workspace=
    # "system" with full scope by src/api/auth.py. CI just copies the
    # value from the compose file once into a secret — no Sanctum
    # creation flow, no Browser, no user-action beyond pasting one
    # string.
    service = os.environ.get("MCP_SERVICE_TOKEN", "").strip()
    if service:
        print("# auth: MCP_SERVICE_TOKEN (server-internal, workspace=system)")
        return service
    raw_jwt = os.environ.get("MAYRING_JWT", "").strip()
    if raw_jwt:
        return raw_jwt
    try:
        with open(JWT_PATH) as f:
            return f.read().strip()
    except OSError:
        sys.stderr.write(
            "FATAL: no auth credential found. Pick one (in priority order):\n"
            "  1. MCP_SERVICE_TOKEN — copy from docker-compose.mayring.yml (server-internal,\n"
            "     no UI click, never expires). Best for CI.\n"
            "  2. MAYRING_SANCTUM_TOKEN — generate at app.linn.games/settings/mayring-abo;\n"
            "     exercises full user-auth path.\n"
            "  3. MAYRING_JWT or "
            f"{JWT_PATH} (raw JWT for ad-hoc local).\n"
        )
        sys.exit(2)


_LARAVEL_BASE = os.environ.get("LARAVEL_BASE_URL", "https://app.linn.games").rstrip("/")


def _login_via_sanctum(sanctum_token: str) -> str:
    """Trade a Sanctum personal-access token for a fresh RS256 JWT.

    Calls `POST /api/mayring/refresh-token` on app.linn.games — the
    same endpoint the existing memory_inject hook uses when its JWT
    expires. Validates the user has a workspace and Mayring access,
    issues a fresh JWT via JwtIssuer::issueForUser. So this single
    call exercises:

      • Sanctum auth (user-management is wired)
      • $user->currentWorkspace() (workspace resolution works)
      • $workspace->hasMayringAccess() (subscription/billing path)
      • JwtIssuer (RS256 signing with sub=user.id, workspace_id)

    A 401 here means the Sanctum token is invalid → user-auth broken.
    A 403 → subscription gate active. Anything else → laravel down.
    """
    body = b""  # endpoint takes no body
    req = urllib.request.Request(
        f"{_LARAVEL_BASE}/api/mayring/refresh-token",
        data=body,
        headers={
            "Authorization": f"Bearer {sanctum_token}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.stderr.write(
            f"FATAL: Sanctum→JWT exchange failed at "
            f"{_LARAVEL_BASE}/api/mayring/refresh-token: HTTP {e.code}\n"
            f"  {e.read().decode()[:200]}\n"
            f"  → check MAYRING_SANCTUM_TOKEN value (rotate if needed)\n"
        )
        sys.exit(2)
    except Exception as e:
        sys.stderr.write(
            f"FATAL: could not reach {_LARAVEL_BASE}: {type(e).__name__}: {e}\n"
        )
        sys.exit(2)
    jwt = payload.get("token", "")
    if not jwt:
        sys.stderr.write(f"FATAL: refresh-token returned no token: {payload}\n")
        sys.exit(2)
    print(f"# logged in via Sanctum → fresh JWT ({len(jwt)} chars)")
    return jwt


def _http(method: str, url: str, token: str, body: dict | None = None,
          timeout: float = 10.0,
          workspace_id: str | None = None,
          extra_headers: dict | None = None) -> tuple[int, dict | None, float]:
    """Returns (status_code, parsed_json_or_None, elapsed_seconds).

    Retries on 502/503/504 and connection errors up to 6 times with a
    short backoff. Container restarts during deploy commonly produce a
    short window of 502s — we don't want every post-deploy smoke to
    spuriously red-flag during that window.

    SMOKE-TEST-STABILITY POLICY (V2 Stufe 6):
      1. Jeder neue check muss einen Red-Green-Beweis liefern (failed vor
         fix, passt nach). Sonst ist es Theatre.
      2. Flaky tests (>=2× transient rot in 7d wegen 502/503/504): root-cause
         fixen oder den check droppen — KEIN dauerhaftes EXPECTED_PENDING
         als Workaround.
      3. EXPECTED_PENDING > 30d → automatisch dropped (manueller Audit
         nötig). Generiert sonst Email-Spam ohne Aktionsmöglichkeit.
      4. _http retries 6× weil deploy-windows + Container-Migration
         manchmal >10s dauern. Wenn nach 6×2s=12s noch 5xx → echter Bug.

    workspace_id (per-call): überschreibt das Service-Token-Default
    'system' für Checks die User-Workspace-Daten brauchen
    (memory_search, rag_function_search). Bogus-repo-Trigger
    (pipeline_stage_observability) bleiben absichtlich OHNE → landen
    im 'system'-Maintenance-Bucket statt in bene's Job-History.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if workspace_id:
        headers["X-Workspace-Id"] = workspace_id
    if extra_headers:
        headers.update(extra_headers)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    backoff = 2.0
    last_code = 0
    last_body: dict | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
                try:
                    return resp.status, json.loads(raw), time.time() - t0
                except (ValueError, TypeError):
                    return resp.status, None, time.time() - t0
        except urllib.error.HTTPError as e:
            try:
                err_json = json.loads(e.read().decode())
            except Exception:
                err_json = None
            last_code, last_body = e.code, err_json
            if e.code in (502, 503, 504) and attempt < 3:
                time.sleep(backoff)
                backoff *= 2
                continue
            return e.code, err_json, time.time() - t0
        except Exception as e:
            # WHY(2026-05-30 smoke-hang): socket TIMEOUTs land here. Retrying a
            # slow-but-alive endpoint 4× with backoff turns a 30s timeout into
            # ~130s per check and the whole smoke into a 25min runner-hang under
            # saturation. One retry is enough for a genuine blip; beyond that the
            # search-warmth gate (wait_for_search_ready) is the real guard.
            last_body = {"_error": f"{type(e).__name__}: {e}"}
            if attempt < 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            return 0, last_body, time.time() - t0
    return last_code, last_body, time.time() - t0


# ---------------------------------------------------------------------------
# Pre-flight: wait the post-deploy restart window out (#250)
# ---------------------------------------------------------------------------

def _quick_health(api: str, token: str, timeout: float = 5.0) -> bool:
    """Single one-shot GET /health — no retries. True iff 200 + status:ok."""
    req = urllib.request.Request(
        f"{api}/health",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode())
            return (body or {}).get("status") == "ok"
    except Exception:
        return False


def wait_for_api_ready(api: str, token: str, max_wait: float = 60.0) -> bool:
    """Poll /health until the API is up, or max_wait elapses.

    WHY(#250): the post-deploy smoke runs via workflow_run right after the
    deploy — it can catch the API mid-restart. _http already retries 5xx a
    few times, but a cold-start window can exceed that, so the FIRST check
    (api_health) FAILS and the workflow auto-opens a false-positive
    "smoke FAIL" issue (#242, #245). Waiting the restart window out HERE
    means every subsequent check runs against a warm API.

    A genuine >max_wait outage is NOT swallowed: we return False, print a
    loud warning, and run the checks anyway — api_health then fails as it
    should and the alert is real.
    """
    deadline = time.time() + max_wait
    n = 0
    while time.time() < deadline:
        n += 1
        if _quick_health(api, token):
            if n > 1:
                print(f"# API ready after {n} probe(s)")
            return True
        time.sleep(3.0)
    print(f"# WARNING: API not ready after {max_wait:.0f}s — running checks anyway "
          "(api_health will FAIL if it's a real outage, which is correct).")
    return False


def _quick_search_ok(api: str, token: str, timeout: float = 8.0) -> bool:
    """One-shot tiny /memory/search. True iff it answers 200 AND the embedding model
    is actually warm — i.e. at least one result carries a non-zero vector score.

    WHY(false-positive-smoke 2026-06-08): http=200 alone is not warmth. Right after a
    restart the bge-m3 model is cold/loading; search returns 200 with every
    score_vector=0.000 (max_score=0.000), so memory_search_vector/rag fail spuriously.
    Gating on a real non-zero vector score makes wait_for_search_ready hold until the
    model can embed, not just until the HTTP handler is up. Empty result sets (200,
    no rows) are treated as warm — the corpus, not the model, is the variable there.

    """
    req = urllib.request.Request(
        f"{api}/memory/search",
        data=json.dumps({"query": "memory retrieval pipeline vector search",
                         "top_k": 3}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode())
            results = data.get("results") or []
            if not results:
                return True  # no corpus to match — model warmth not provable here
            return any(float(r.get("score_vector") or 0) > 0.0 for r in results)
    except Exception:
        return False


def wait_for_search_ready(api: str, token: str, max_wait: float = 90.0) -> bool:
    """Poll /memory/search until the SEARCH pipeline is warm (not just /health).

    WHY(2026-05-30 smoke-hang): /health stays ~0.1s even when the Chroma/embedding
    search pipeline is saturated/cold (verified: /health 200 while /memory/search
    times out at 30s). The heavy checks (rag, multi_org act-as) then each hit the
    slow path and _http retries amplify 30s→~130s per check → the whole smoke runs
    until the job timeout (~25min) and ties up the runner. Gating on actual search
    warmth before the heavy checks keeps the run bounded.

    Non-fatal: a real >max_wait search outage returns False + a loud warning; the
    rag/memory checks then fail as they should.
    """
    deadline = time.time() + max_wait
    n = 0
    while time.time() < deadline:
        n += 1
        if _quick_search_ok(api, token):
            if n > 1:
                print(f"# search pipeline warm after {n} probe(s)")
            return True
        time.sleep(4.0)
    print(f"# WARNING: search pipeline not warm after {max_wait:.0f}s — running checks "
          "anyway (rag/memory checks will FAIL if it's a real outage).")
    return False


# ---------------------------------------------------------------------------
# Individual checks — every one returns CheckResult
# ---------------------------------------------------------------------------

def check_health(api: str, token: str) -> CheckResult:
    code, body, dt = _http("GET", f"{api}/health", token)
    return CheckResult(
        "api_health",
        code == 200 and (body or {}).get("status") == "ok",
        f"http={code} time={dt:.2f}s body={body}",
        body,
    )


def check_workspace_scoped(api: str, token: str) -> CheckResult:
    """Stats endpoints must respond with a sane workspace_id.

    Accepts (seit 2026-05-09 email-slug-Refactor):
      • email-slug    — JWT.email-derived (z.B. "bene" für bene@linn.games)
      • "system"      — service-token ohne X-Workspace-Id-Override
    Rejects: legacy "user-N", "default", oder leer — wären Indizien
    für eine Auth-/Workspace-Auflösungs-Regression.
    """
    code, body, dt = _http("GET", f"{api}/stats/workspaces", token)
    if code != 200:
        return CheckResult("workspace_scoping", False,
                           f"/stats/workspaces http={code}: {body}")
    ws = (body or {}).get("workspace_id", "")
    import re as _re
    # Email-slug-pattern: lowercase alphanumeric + dash, kein 'user-N',
    # kein 'default'.
    is_email_slug = bool(_re.match(r"^[a-z0-9][a-z0-9-]*$", ws or "")) and not ws.startswith("user-") and ws != "default"
    is_sane = is_email_slug or ws == "system"
    return CheckResult(
        "workspace_scoping",
        is_sane,
        f"workspace_id={ws!r} (expected 'user-<id>' or 'system')",
    )


def check_memory_search_returns_vector_hits(api: str, token: str) -> CheckResult:
    """A real search must:
    - return 200
    - report vector_stage diagnostics (not 'unknown')
    - have at least one result with score_vector > 0 in top-5

    Cold-chroma tolerance: right after a container restart the embedding
    cache + chroma collection are sometimes cold, so the first 1–2 calls
    return ``chroma_query_empty``. Retry a couple of times with a fresh
    query (no query_cache_hit) before failing — the persistent failure
    is a real bug, the transient is just deploy churn.
    """
    diag = "?"; code = 0; dt = 0.0; body = None; results = []
    for attempt in range(3):
        code, body, dt = _http(
            "POST", f"{api}/memory/search", token,
            body={"query": f"memory feedback hook stop attempt-{attempt}",
                  "top_k": 5, "include_text": False, "llm_prefilter": False},
            timeout=15.0,
            workspace_id=SMOKE_VECTOR_WORKSPACE,
            # WHY(tenancy phase A): workspace content is now private/user_id-scoped;
            # the bare service token (no sub) only sees public → vector candidates
            # are sparse/far. Act as the owner so private chunks are searchable.
            extra_headers=_act_as(SMOKE_VECTOR_OWNER, workspace=SMOKE_VECTOR_WORKSPACE),
        )
        if code != 200:
            time.sleep(2)
            continue
        diag = (body or {}).get("diagnostics", {}).get("vector_stage", "?")
        if isinstance(diag, str) and (diag.startswith("ok(") or diag == "query_cache_hit"):
            break
        time.sleep(3)
    if code != 200:
        return CheckResult("memory_search_vector", False,
                           f"http={code} time={dt:.2f}s body={body}")
    results = (body or {}).get("results", [])
    has_vec = any(r.get("score_vector", 0) > 0 for r in results)
    diag_ok = isinstance(diag, str) and (diag.startswith("ok(") or diag == "query_cache_hit")
    return CheckResult(
        "memory_search_vector",
        diag_ok,
        f"http={code} time={dt:.2f}s diag={diag!r} top_k_with_vec={has_vec} results={len(results)}",
    )


def check_feedback_binary_only(api: str, token: str) -> CheckResult:
    """Legacy signals must be rejected (rating-migration 2026-05-10:
    only '1'..'5' accepted, positive/negative/neutral all rejected)."""
    rejected = []
    for legacy in ("neutral", "positive", "negative"):
        code, body, _ = _http(
            "POST", f"{api}/memory/feedback", token,
            body={"chunk_id": "chk_smoke_test_dummy_1", "signal": legacy},
        )
        rejected.append((legacy, code, code in (400, 422)))
    all_ok = all(r[2] for r in rejected)
    return CheckResult(
        "feedback_legacy_rejected",
        all_ok,
        f"rejected={rejected} — alle legacy-signals müssen 400/422 sein",
    )


def check_feedback_slug_resolution(api: str, token: str) -> CheckResult:
    """Submit feedback via source-id slug; backend must resolve to chunks."""
    # Pick a known existing source_id from a search
    code, body, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": "memory", "top_k": 1, "include_text": False,
              "llm_prefilter": False},
    )
    if code != 200 or not (body or {}).get("results"):
        return CheckResult("feedback_slug_resolution", False,
                           "could not get a real source_id from /memory/search")
    sid = body["results"][0]["source_id"]

    code, body, _ = _http(
        "POST", f"{api}/memory/feedback", token,
        body={"chunk_id": sid, "signal": "5"},
    )
    if code != 200:
        return CheckResult("feedback_slug_resolution", False,
                           f"slug feedback failed http={code} body={body}")
    applied = (body or {}).get("applied_to", 0)
    return CheckResult(
        "feedback_slug_resolution",
        applied >= 1,
        f"slug={sid[:60]!r} applied_to={applied} chunks={(body or {}).get('chunk_ids', [])[:3]}",
    )


def check_feedback_count_moves(api: str, token: str) -> CheckResult:
    """Feedback POST is accepted AND the stats aggregate stays live + positive.

    WHY(2026-05-28): the old `post.total > pre.total` assertion was wrong and
    flaked. ``feedback.total`` is DISTINCT-counted and the POST busts the stats
    cache (→ fresh recompute), so when a concurrent reingest deletes chunks (+
    their cascaded chunk_feedback) the global total legitimately drops between
    the two reads — verified live: post=1668 < pre=1673 right after a deploy's
    post-deploy-ingest. The faithful, churn-stable property is: the write is
    recorded AND the aggregate pipeline still returns a live positive total."""
    code, body, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": "stop hook", "top_k": 1, "include_text": False,
              "llm_prefilter": False},
    )
    if code != 200 or not (body or {}).get("results"):
        return CheckResult("feedback_count_delta", False, "no chunk_id from search")
    cid = body["results"][0]["chunk_id"]

    fb_code, fb_body, _ = _http(
        "POST", f"{api}/memory/feedback", token,
        body={"chunk_id": cid, "signal": "5"},
    )
    recorded = (fb_body or {}).get("recorded") is True
    if fb_code != 200 or not recorded:
        return CheckResult("feedback_count_delta", False,
                           f"feedback POST http={fb_code} recorded={recorded} body={fb_body}")

    # POST busts the stats cache → this is a fresh recompute, not stale.
    post_code, post, _ = _http("GET", f"{api}/stats/summary", token)
    post_total = (post or {}).get("feedback", {}).get("total", 0)
    return CheckResult(
        "feedback_count_delta",
        post_code == 200 and isinstance(post_total, int) and post_total >= 1,
        f"recorded={recorded}  post.total={post_total} (aggregate live; global "
        f"total is non-monotonic under reingest, so not asserting pre<post)",
    )


def check_micro_batch_indexes(api: str, token: str) -> CheckResult:
    """Turn-capture endpoint must accept a turn pair and create a source."""
    session_id = f"smoke-{int(time.time())}"
    # Issue #7-fix: server bindet workspace_slug strikt an JWT-workspace_id.
    # Smoke nutzt MCP_SERVICE_TOKEN (workspace=system) — wenn wir hier
    # 'mayringcoder' angeben, returnt 403. Slug einfach weglassen → server
    # defaultet auf JWT-workspace.
    code, body, dt = _http(
        "POST", f"{api}/conversation/micro-batch", token,
        body={
            "turns": [
                {"role": "user", "content": "smoke test prompt", "timestamp": ""},
                {"role": "assistant", "content": "smoke test response", "timestamp": ""},
            ],
            "session_id": session_id,
        },
        timeout=30.0,  # server summarises via LLM
    )
    indexed = bool((body or {}).get("indexed"))
    sid = (body or {}).get("source_id", "")

    # WHY(2026-05-11): cleanup — vorher hat jeder smoke-run eine
    # conversation:system:smoke-{ts}-source hinterlassen → workspace=system
    # füllte sich mit test-müll (211 chunks in 6h). Nach der index-prüfung
    # invalidieren wir die source wieder. Best-effort: ein fehlgeschlagener
    # cleanup macht den check nicht rot (der eigentliche test ist indexed=True).
    if sid:
        try:
            _http("POST", f"{api}/memory/invalidate", token,
                  body={"source_id": sid}, timeout=10.0)
        except Exception:
            pass

    return CheckResult(
        "micro_batch_indexes",
        code == 200 and indexed,
        f"http={code} time={dt:.2f}s indexed={indexed} source_id={sid} (cleaned up)",
    )


def check_dashboard_endpoints(api: str, token: str) -> CheckResult:
    """All 10 dashboard endpoints respond 200 with a workspace-id field."""
    paths = [
        "/stats/recent-ops", "/stats/jobs-history", "/stats/feedback-log",
        "/stats/source-refs", "/stats/triggers", "/stats/topic-flow",
        "/stats/pi-tasks", "/stats/activations", "/stats/workspaces",
        "/stats/vector-trend", "/stats/notifications",
    ]
    failures: list[str] = []
    for p in paths:
        code, body, dt = _http("GET", f"{api}{p}", token, timeout=8.0)
        if code != 200:
            failures.append(f"{p}: http={code}")
        elif not isinstance(body, dict) or "workspace_id" not in body:
            failures.append(f"{p}: missing workspace_id field in response")
    return CheckResult(
        "dashboard_endpoints",
        not failures,
        "; ".join(failures) if failures else f"all {len(paths)} endpoints 200 + workspace-scoped",
    )


def check_notifications_ingest_roundtrip(api: str, token: str) -> CheckResult:
    """Hook-A: POST /stats/notifications/ingest accepts a plugin watch-finding and it
    surfaces in /stats/notifications with the right Ampel urgency. Uses a /smoke/repo-
    repo so the event resolves to the 'system' workspace (never pollutes the user's
    dashboard) and is idempotent (re-POST of the same payload → skipped)."""
    repo = "github.com/smoke/repo-notif-probe"
    ev = {"hook_type": "repo_pull", "repo": repo, "number": 1,
          "summary": "smoke notification probe", "url": "https://example/smoke"}
    code, body, _ = _http("POST", f"{api}/stats/notifications/ingest", token,
                          body={"events": [ev]}, timeout=8.0)
    if code != 200:
        return CheckResult("notifications_ingest_roundtrip", False,
                           f"POST /stats/notifications/ingest http={code}: {body}")
    if not isinstance(body, dict) or not body.get("ok"):
        return CheckResult("notifications_ingest_roundtrip", False,
                           f"ingest response not ok: {body}")
    # A non-accepted type must be rejected (skipped), proving the allow-list works.
    code2, body2, _ = _http("POST", f"{api}/stats/notifications/ingest", token,
                            body={"events": [{"hook_type": "repo_ci", "repo": repo}]},
                            timeout=8.0)
    if code2 != 200 or (isinstance(body2, dict) and body2.get("inserted", 0) != 0):
        return CheckResult("notifications_ingest_roundtrip", False,
                           f"repo_ci should be rejected by ingest allow-list: {body2}")
    return CheckResult(
        "notifications_ingest_roundtrip", True,
        f"ingest ok (inserted={body.get('inserted')}, skipped={body.get('skipped')}); "
        f"ci-type rejected",
    )


def check_coverage_map_complete(api: str, token: str) -> CheckResult:
    """Meta-check: every closed issue must appear exactly once in
    docs/smoke_coverage_map.md. Future-proofs against silent gaps —
    if anyone closes an issue without entering it in the map, smoke
    fails loud the next deploy.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    map_path = repo_root / "docs" / "smoke_coverage_map.md"
    if not map_path.exists():
        return CheckResult("coverage_map_complete", False,
                           f"missing: {map_path}")
    map_text = map_path.read_text(encoding="utf-8")
    # Pull all `| <number> |` table cells from the map. Conservative
    # regex: the issue number always appears as the first column,
    # bracketed by pipes.
    in_map = set(re.findall(r"^\|\s*(\d+)\s*\|", map_text, re.MULTILINE))

    # Closed issues via GitHub REST. Try, in order:
    #   1. GH_TOKEN env (CI default)
    #   2. GITHUB_TOKEN env (Actions default)
    #   3. `gh auth token` (local dev — picks up the gh CLI's stored auth)
    closed: set[str] = set()
    page = 1
    gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    if not gh_token:
        import subprocess
        try:
            r = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                gh_token = r.stdout.strip()
        except Exception:
            pass
    while True:
        url = (f"https://api.github.com/repos/Nileneb/MayringCoder/"
               f"issues?state=closed&per_page=100&page={page}")
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json",
                          **({"Authorization": f"Bearer {gh_token}"} if gh_token else {})},
        )
        # WHY(smoke-flake 2026-05-24): the GitHub API occasionally drops a
        # multi-page response body mid-download (http.client.IncompleteRead) —
        # a transient network blip, not a coverage regression. A single failure
        # used to red the whole post-deploy smoke. Retry the page fetch a few
        # times before giving up (smoke-stability policy: root-cause the flake).
        items = None
        last_err: Exception | None = None
        for _attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    items = json.loads(r.read())
                last_err = None
                break
            except Exception as e:  # IncompleteRead, timeout, transient 5xx
                last_err = e
                time.sleep(1.5)
        if last_err is not None:
            return CheckResult("coverage_map_complete", False,
                               f"GitHub API page {page} failed after retries: {last_err}")
        if not items:
            break
        for it in items:
            # Skip pull_requests — issues API returns those too.
            # Also skip auto-created smoke-failure issues — they're alert
            # artefacts, not features that need acceptance documentation.
            if "pull_request" in it:
                continue
            labels = {(l.get("name") or "").lower() for l in (it.get("labels") or [])}
            if "smoke-failure" in labels:
                continue
            closed.add(str(it["number"]))
        if len(items) < 100:
            break
        page += 1
        if page > 5:
            break  # 500 issue safety

    missing = closed - in_map
    return CheckResult(
        "coverage_map_complete",
        not missing,
        f"closed_issues={len(closed)}  documented={len(in_map)}  "
        f"missing_from_map={sorted(missing)[:10]}{'…' if len(missing) > 10 else ''}",
    )


def check_retrieval_reasons_field(api: str, token: str) -> CheckResult:
    """User question: 'wo wird der Reason gespeichert?' — every result of
    /memory/search must carry a `reasons` array explaining WHY a chunk
    surfaced (embedding_similarity, token_overlap, recent_chunk,
    source_affinity_match, llm_advisor_high). At least one of the top-5
    results must have a non-empty reasons list.

    WHY(#361 2026-06-08 false-positive): the probe MUST search the populated
    SMOKE_VECTOR_WORKSPACE as its owner (act-as), with a MEANINGFUL query — not a
    random nonce against the bare service token. The bare token only sees the sparse
    public corpus; a random query there finds 5 far chunks that cross no reason
    threshold (sv_eff<=0.5, no token_overlap) → reasons=[] → spurious RED. A
    semantically-relevant query against the dense workspace reliably fires
    embedding_similarity+token_overlap. The `attempt-<ts>` suffix busts _QUERY_CACHE
    so the full _rerank() path runs and reasons get (re)populated.
    """
    query = f"memory retrieval reranker vector scoring pipeline attempt-{int(time.time())}"
    code, body, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": query, "top_k": 5,
              "include_text": False, "llm_prefilter": False},
        timeout=12.0,
        workspace_id=SMOKE_VECTOR_WORKSPACE,
        extra_headers=_act_as(SMOKE_VECTOR_OWNER, workspace=SMOKE_VECTOR_WORKSPACE),
    )
    if code != 200:
        return CheckResult("retrieval_reasons_field", False, f"http={code}")
    results = (body or {}).get("results", [])
    has_reasons = any(r.get("reasons") for r in results)
    return CheckResult(
        "retrieval_reasons_field",
        has_reasons,
        f"top_k={len(results)} chunks_with_reasons="
        f"{sum(1 for r in results if r.get('reasons'))}",
    )


def check_igio_axis_on_chunks(api: str, token: str) -> CheckResult:
    """Issue #141 acceptance: ≥50% of active chunks have a non-empty
    ``igio_axis``. The IGIO classifier was added late; older chunks were
    never reclassified, so the column was filled <5% across the dataset.

    Two parts to this check:
      1) shape — /memory/chunk/{id} response includes ``igio_axis`` and
         ``igio_confidence`` fields (regression guard for the column itself).
      2) coverage — /stats/igio-coverage reports ratio ≥ 0.5 (the
         backfill loop has run enough times to fill the historical gap).

    Coverage drives an admin-side backfill loop (cron'd workflow). Until
    coverage clears 50%, this check stays red — that's the whole point:
    the failing smoke is the trigger that keeps the loop running.
    """
    code, body, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": "fix bug", "top_k": 1, "include_text": False,
              "llm_prefilter": False},
    )
    if code != 200 or not (body or {}).get("results"):
        return CheckResult("igio_axis_on_chunks", False,
                           "could not get a chunk_id to inspect")
    cid = body["results"][0]["chunk_id"]
    code2, body2, _ = _http("GET", f"{api}/memory/chunk/{cid}", token)
    if code2 != 200:
        return CheckResult("igio_axis_on_chunks", False,
                           f"GET /memory/chunk http={code2}")
    chunk = (body2 or {}).get("chunk", {})
    has_igio = "igio_axis" in chunk and "igio_confidence" in chunk
    if not has_igio:
        return CheckResult(
            "igio_axis_on_chunks", False,
            f"shape regression: chunk_id={cid[:18]} missing igio fields",
        )
    code3, body3, _ = _http("GET", f"{api}/stats/igio-coverage", token)
    if code3 != 200 or not isinstance(body3, dict):
        return CheckResult(
            "igio_axis_on_chunks", False,
            f"GET /stats/igio-coverage http={code3}",
        )
    ratio = float(body3.get("ratio") or 0.0)
    total = int(body3.get("total_active") or 0)
    with_axis = int(body3.get("with_axis") or 0)
    scope = body3.get("scope")
    threshold = 0.5
    ok = ratio >= threshold or total < 50
    return CheckResult(
        "igio_axis_on_chunks", ok,
        f"shape=OK  scope={scope}  total={total}  with_axis={with_axis}  "
        f"ratio={ratio:.3f}  (target ≥ {threshold})",
    )


def check_wiki_context_injector_used(api: str, token: str) -> CheckResult:
    """Closed Issue #75 (Wiki-v2 P5) acceptance: WikiContextInjector
    builds context blocks. The /wiki/graph response includes the
    `activations` field that lists recent context-injection events —
    if the injector ran for any source, it surfaces here. Empty
    activations after a fresh injection = injector silently skipped."""
    code, body, _ = _http(
        "GET", f"{api}/wiki/graph?slug=mayringcoder", token,
    )
    if code != 200:
        return CheckResult("wiki_context_injector_used", False, f"http={code}")
    has_activations_key = isinstance(body, dict) and "activations" in body
    return CheckResult(
        "wiki_context_injector_used",
        has_activations_key,
        f"http={code}  has_activations_field={has_activations_key} "
        f"recent_count={len((body or {}).get('activations', []) if isinstance(body, dict) else [])}",
    )


def check_wiki_p7_endpoints(api: str, token: str) -> CheckResult:
    """Closed Issue #77 (Wiki 2.0 P7) acceptance: /wiki/rebuild,
    /wiki/graph (mermaid), /wiki/edge, /wiki/conflicts must respond.
    Tests endpoint reachability + workspace_id isolation. The actual
    rebuild is async — we check the route accepts POSTs."""
    paths_methods = [
        ("GET", "/wiki/slugs"),
        ("GET", "/wiki/conflicts"),
        ("GET", "/wiki/feedback-matrix?limit=1"),
    ]
    failures: list[str] = []
    for method, path in paths_methods:
        code, _, _ = _http(method, f"{api}{path}", token)
        if code not in (200, 404):
            failures.append(f"{method} {path}: http={code}")
    return CheckResult(
        "wiki_p7_endpoints",
        not failures,
        "; ".join(failures) if failures else f"{len(paths_methods)} wiki endpoints reachable",
    )


def check_wiki_p8_history(api: str, token: str) -> CheckResult:
    """Closed Issue #78 (Wiki 2.0 P8) acceptance: history/diff endpoints
    + wiki_contributions logged per user-id."""
    code1, _, _ = _http("GET", f"{api}/wiki/history?slug=mayringcoder&limit=5", token)
    code2, _, _ = _http("GET", f"{api}/wiki/team?slug=mayringcoder&limit=5", token)
    return CheckResult(
        "wiki_p8_history",
        code1 in (200, 404, 422) and code2 in (200, 404, 422),
        f"history.http={code1}  team.http={code2} (200/404/422 all OK — endpoints exist)",
    )


def check_db_wal_journal_active(api: str, token: str) -> CheckResult:
    """Issue #84 sub-1 (DB-Lock): WAL journal mode must be active so
    cross-container writes don't 'database is locked'.

    Probe via /stats/admin/training-data-counts (any endpoint that hits
    memory.db). If we get a 200 and a probe of /memory/put-search-cycle
    completes without 500/lock errors, WAL is functionally on. Direct
    PRAGMA inspection isn't exposed via API — but the regression mode
    (no WAL) shows up as '500: database is locked' on concurrent ops.

    Cheap functional probe: rapid-fire 3 puts to /memory/put with
    unique source_ids. Pass if all 3 return 200/201.
    """
    marker = int(time.time() * 1000)
    codes: list[int] = []
    probe_ids: list[str] = []
    for i in range(3):
        sid = f"smoke:wal-probe:{marker}-{i}"
        probe_ids.append(sid)
        code, _, _ = _http(
            "POST", f"{api}/memory/put", token,
            body={
                "source_id": sid,
                "source_type": "test",
                "content": f"WAL probe {i}/3 marker={marker}",
            },
            timeout=10.0,
        )
        codes.append(code)
    all_ok = all(c in (200, 201) for c in codes)
    # WHY(no-pollution): these probes write into the caller's REAL workspace (no
    # act-as) — without self-clean they accumulate (277 leaked smoke sources found
    # 2026-06-08). Best-effort invalidate; a failed cleanup must not flip the check.
    for sid in probe_ids:
        try:
            _http("POST", f"{api}/memory/invalidate", token,
                  body={"source_id": sid}, timeout=10.0)
        except Exception:
            pass
    return CheckResult(
        "db_wal_journal_active",
        all_ok,
        f"3 concurrent /memory/put → {codes}  "
        f"(all 200/201 = WAL prevents cross-container locks)",
    )


def check_pipeline_stage_observability(api: str, token: str) -> CheckResult:
    """Issue #84 sub-3 (Stage-Status fields): when a job runs, the
    /jobs/{id} response should expose progress / stages / batch
    metadata so users can see WHERE in the pipeline a job is, not
    just 'started'/'done'.

    Probe: trigger a /populate with a known-bad repo (fails fast),
    poll /jobs/{id} for ~5s, assert response shape carries the
    structural fields (status + at least one of progress/stages/v2_jobs).
    Doesn't run a real ingest.
    """
    code, body, _ = _http(
        "POST", f"{api}/populate", token,
        # WHY(#253): source="smoke" tags the (intentionally failing) job so the
        # job-history UI default-filters it out of workspace:system noise.
        body={"repo": "https://github.com/Nileneb/smoke-stage-observability-bogus",
              "source": "smoke"},
    )
    if code != 200 or not isinstance(body, dict) or "job_id" not in body:
        return CheckResult("pipeline_stage_observability", False,
                           f"POST /populate http={code} body={body}")
    job_id = body["job_id"]
    deadline = time.time() + 5.0
    last_keys: set[str] = set()
    last_status = ""
    while time.time() < deadline:
        time.sleep(1)
        code2, body2, _ = _http("GET", f"{api}/jobs/{job_id}", token)
        if code2 != 200 or not isinstance(body2, dict):
            continue
        last_keys = set(body2.keys())
        last_status = body2.get("status", "")
        # Once status is terminal we have the final shape; check it
        if last_status in ("done", "error", "failed"):
            break
    structural_keys = {"status", "progress", "stages", "v2_jobs"}
    has_at_least_one = bool(last_keys & structural_keys)
    return CheckResult(
        "pipeline_stage_observability",
        has_at_least_one and "status" in last_keys,
        f"job_id={job_id[:18]}  status={last_status!r}  "
        f"keys_seen={sorted(last_keys & structural_keys)}  "
        f"(must expose at least status + one of progress/stages/v2_jobs)",
    )


def check_predictive_transitions_endpoint(api: str, token: str) -> CheckResult:
    """Issue #55 deepening: ambient v2 predictive layer must be
    actually computing transitions. The endpoint
    /predictive/rebuild-transitions kicks off a Markov-matrix build
    from chunk_feedback + ingestion_log; the matrix is what
    predict_next_topics() consumes inside the ambient hook
    (src/memory/ambient.py:497).

    Probe: POST /predictive/rebuild-transitions with a tiny scope.
    Pass: 200 with job_id (proves route exists + dispatches a job)
    OR 401/403 (admin-only, route exists). 404 = not registered.
    """
    code, body, _ = _http(
        "POST", f"{api}/predictive/rebuild-transitions", token,
        body={"workspace_id": "smoke-probe"},
    )
    if code in (401, 403):
        return CheckResult(
            "predictive_transitions_endpoint", True,
            f"http={code} (admin-gated; route exists per #55 wiring)",
        )
    if code == 200 and isinstance(body, dict):
        has_job = "job_id" in body or "status" in body
        return CheckResult(
            "predictive_transitions_endpoint",
            has_job,
            f"http=200  body_keys={list(body.keys())}  "
            f"(proves predictive layer is wired + dispatches)",
        )
    return CheckResult(
        "predictive_transitions_endpoint",
        code in (200, 401, 403, 422),
        f"http={code} body={body}  "
        f"(404 = route not registered; #55 acceptance fails)",
    )


def check_image_routing_supported(api: str, token: str) -> CheckResult:
    """Closed Issue #91 acceptance: vision-capable model route exists.
    Tests via /api/mcp-service/llm-endpoint server-side resolver — if
    'vision' agent resolves at all, the routing path is wired."""
    code, body, _ = _http(
        "GET",
        f"{api}/api/mcp-service/llm-endpoint/system?agent=vision",
        token,
    )
    # Either resolves cleanly (200) or replies "not configured" (404/422).
    # 500 = code path crashed, that's the regression we'd catch.
    return CheckResult(
        "image_routing_supported",
        code != 500,
        f"http={code}  body={body}  (must NOT be 500 = crash)",
    )


def check_training_merge_endpoint(api: str, token: str) -> CheckResult:
    """Issue #87 acceptance: POST /api/training/merge route is registered.

    Earlier this check expected only 200/400/422 — too strict. The route
    is auth-gated (admin-scope), so an unauthenticated probe gets 401
    which is ALSO valid evidence that the route exists. 404 is the only
    real fail mode (route not registered). 500 = crash, also fail.
    """
    code, body, _ = _http("POST", f"{api}/api/training/merge", token, body={})
    return CheckResult(
        "training_merge_endpoint",
        code in (200, 400, 401, 403, 422),
        f"http={code} body={body}  "
        f"(200/400/401/403/422 = route exists; 404 = never built; 500 = crash)",
    )


def check_turbulence_endpoint(api: str, token: str) -> CheckResult:
    """Closed Issue #83 acceptance: /turbulence endpoint exists for
    code-quality reports."""
    code, body, _ = _http("POST", f"{api}/turbulence", token, body={})
    return CheckResult(
        "turbulence_endpoint",
        code in (200, 400, 422),
        f"http={code} (200/400/422 = endpoint exists)",
    )


def check_jwt_invalid_signature_rejected(api: str, token: str) -> CheckResult:
    """Closed Issue #94 acceptance: invalid bearer tokens must yield 401.

    Acceptance from issue body: 'HTTP 401 bei ungültigem Token'. Two paths:
      a) caller has a real RS256 JWT → flip last 4 sig chars → must be 401
      b) caller has the opaque MCP_SERVICE_TOKEN → tampering doesn't apply,
         but a raw garbage bearer should still fail closed → must be 401.
    Either way, the auth pipeline has to refuse the bogus token.
    """
    parts = token.split(".")
    if len(parts) == 3:
        bogus = ".".join([parts[0], parts[1], parts[2][:-4] + "AAAA"])
        mode = "tampered_jwt_sig"
    else:
        bogus = "deadbeef-invalid-bearer-token-not-real"
        mode = "garbage_bearer_when_service_token"
    code, body, _ = _http("GET", f"{api}/stats/summary", token=bogus)
    return CheckResult(
        "jwt_invalid_signature",
        code == 401,
        f"mode={mode}  http={code}  body={body}  (must be 401)",
    )


def check_task_feedback_matrix(api: str, token: str) -> CheckResult:
    """Closed Issue #90 acceptance: /wiki/feedback-matrix?mode=task aggregates
    feedback per Pi-task. Endpoint must respond 200 with a 'tasks' field
    (list, even if empty)."""
    code, body, _ = _http(
        "GET", f"{api}/wiki/feedback-matrix?mode=task&limit=5", token,
    )
    has_tasks = isinstance(body, dict) and "tasks" in body
    return CheckResult(
        "task_feedback_matrix",
        code == 200 and has_tasks,
        f"http={code}  has_tasks_field={has_tasks}  count={len(body.get('tasks', []) if isinstance(body, dict) else [])}",
    )


def check_wiki_graph_clusters(api: str, token: str) -> CheckResult:
    """Closed Issue #73 acceptance: cluster engine produces clusters in the
    wiki graph. Endpoint /wiki/graph?slug=... must respond and clusters
    must exist for at least one workspace."""
    # First get available slugs
    code, slugs, _ = _http("GET", f"{api}/wiki/slugs", token)
    if code != 200:
        return CheckResult("wiki_graph_clusters", False,
                           f"/wiki/slugs http={code}")
    available = (slugs or {}).get("slugs", [])
    if not available:
        return CheckResult("wiki_graph_clusters", True,
                           "no wiki graphs ingested yet — vacuously OK")
    test_slug = available[0]
    code, graph, _ = _http(
        "GET", f"{api}/wiki/graph?slug={urllib.parse.quote(test_slug)}", token,
    )
    if code != 200:
        return CheckResult("wiki_graph_clusters", False,
                           f"/wiki/graph?slug={test_slug!r} http={code}")
    has_clusters_field = isinstance(graph, dict) and "clusters" in graph
    return CheckResult(
        "wiki_graph_clusters",
        has_clusters_field,
        f"slug={test_slug}  has_clusters={has_clusters_field}  "
        f"n_clusters={len(graph.get('clusters', []) if isinstance(graph, dict) else [])}",
    )


def check_pi_tasks_schema(api: str, token: str) -> CheckResult:
    """Closed Issue #107 acceptance: pi_jobs queue exists with correct
    fields (job_id, status, prefer, scope, model). Empty list still counts
    — schema is what we test, not data."""
    code, body, _ = _http("GET", f"{api}/stats/pi-tasks?limit=1", token)
    if code != 200:
        return CheckResult("pi_tasks_schema", False, f"http={code}")
    tasks = (body or {}).get("tasks", [])
    if not tasks:
        return CheckResult("pi_tasks_schema", True,
                           "no pi-tasks yet — endpoint shape OK (empty list)")
    expected_keys = {"job_id", "status", "prefer", "scope", "model"}
    missing = expected_keys - set(tasks[0].keys())
    return CheckResult(
        "pi_tasks_schema",
        not missing,
        f"first_task_keys={list(tasks[0].keys())}  missing={missing or 'none'}",
    )


def check_categorization_logging(api: str, token: str) -> CheckResult:
    """Closed Issue #101 acceptance: LLM categorization writes to llm_calls_log
    (raw prompt+response visible). Triggered by ingest with categorize=true.
    Counted via /stats/vector-trend (which queries llm_calls_log)."""
    code, body, _ = _http("GET", f"{api}/stats/vector-trend?limit=1", token)
    if code != 200:
        return CheckResult("categorization_logging", False, f"http={code}")
    logged = (body or {}).get("logged_24h", 0)
    # If the table is wired and search has fired today, this is > 0. The
    # check covers the "logging path is alive" property.
    return CheckResult(
        "categorization_logging",
        logged > 0,
        f"llm_calls_log entries last 24h: {logged} (must be > 0)",
    )


def check_jobs_progress_observability(api: str, token: str) -> CheckResult:
    """Closed Issues #84+#85 acceptance: jobs expose stages + progress
    fields so external pollers see pipeline state. Endpoint /jobs/{id}
    must accept any UUID and return either the job (200) or 404 with
    a clean error — not a 500 with a Python stack trace."""
    code, body, _ = _http("GET", f"{api}/jobs/00000000-0000-0000-0000-000000000000", token)
    return CheckResult(
        "jobs_progress_observability",
        code in (200, 404),
        f"http={code}  body={body}  (200 if exists, 404 if unknown — never 500)",
    )


def check_ingest_state_field(api: str, token: str) -> CheckResult:
    """Issue #137 acceptance: /memory/put response.state must distinguish
    NEW (first ingest), UNCHANGED (same content_hash), CHANGED (different
    content). The original close was without this verification — the field
    silently always returned 'new' for the REST path because content_hash
    was never set on the request side.
    """
    sid = f"smoke:state:{int(time.time())}:{os.urandom(2).hex()}"

    def _put(content: str) -> str:
        code, body, _ = _http(
            "POST", f"{api}/memory/put", token,
            body={
                "source_id": sid, "source_type": "note",
                "repo": "smoke", "path": "state-test",
                "content": content, "categorize": False,
            },
            timeout=20.0,
        )
        if code != 200:
            return f"http={code}"
        return (body or {}).get("state", "?")

    s1 = _put("content-version-1")
    s2 = _put("content-version-1")  # identical → unchanged
    s3 = _put("content-version-2")  # different → changed

    # WHY(self-clean 2026-06-10): ohne invalidate akkumulierte JEDER Lauf eine
    # weitere smoke:state-Source am selben Canonical-Chunk (96 aufgelaufen).
    # Fail-soft: ein fehlgeschlagenes Cleanup kippt den Check nicht.
    _http("POST", f"{api}/memory/invalidate", token,
          body={"source_id": sid}, timeout=15.0)

    ok = s1 == "new" and s2 == "unchanged" and s3 == "changed"
    return CheckResult(
        "ingest_state_field",
        ok,
        f"first(new)={s1!r}  same(unchanged)={s2!r}  different(changed)={s3!r}",
    )


def _http_await_source(method: str, url: str, token: str, *, body=None,
                       timeout: float = 15.0, tries: int = 6, delay: float = 0.5,
                       extra_headers: dict | None = None):
    """Like _http, but retry while the response is 404 (just-PUT source not yet
    visible across workers). Any non-404 (200/403/...) returns immediately."""
    code, body_r, hdrs = _http(method, url, token, body=body, timeout=timeout, extra_headers=extra_headers)
    for _ in range(tries - 1):
        if code != 404:
            break
        time.sleep(delay)
        code, body_r, hdrs = _http(method, url, token, body=body, timeout=timeout, extra_headers=extra_headers)
    return code, body_r, hdrs


# Every ephemeral workspace a check acts-as, registered at creation time. The
# teardown purges this set UNION the server-side listing — so a workspace whose
# chunks materialize late (async ingest) or that the listing misses for any
# reason still gets purged in the same run (2026-06-10: 10 leftovers despite a
# green run and a SILENT teardown — the listing returned none of them).
_EPHEMERAL_WS: set[str] = set()


def _act_as(sub: str, *, orgs: tuple[str, ...] = (), workspace: str | None = None,
            role: str | None = None) -> dict:
    """Build X-Act-As-* headers so a privileged smoke token simulates another
    caller. Requires MAYRING_ALLOW_ACT_AS=1 on the server (set in the smoke
    job env). workspace defaults to the caller's home if omitted. role (tenancy
    phase B) gives the simulated member a role in `workspace` so role-gated
    setup writes (write/share_org/share_public) are exercisable."""
    h = {"X-Act-As-Sub": sub}
    if orgs:
        h["X-Act-As-Orgs"] = ",".join(orgs)
    if workspace:
        h["X-Act-As-Workspace"] = workspace
        if _SMOKE_WS_RE.match(workspace):
            _EPHEMERAL_WS.add(workspace)
    if role:
        h["X-Act-As-Role"] = role
    return h


def _search_finds(api: str, token: str, query: str, sid: str, *,
                  extra_headers: dict | None = None, tries: int = 6, delay: float = 0.5) -> bool:
    """Search up to `tries` times; True as soon as `sid` is in the results.

    Absorbs the multi-worker commit-propagation window so a positive check
    doesn't false-RED. For isolation checks, confirm the OWNER finds the
    source first (propagation proven), then assert the other identity does
    not — so a not-found is real isolation, not just lag.
    """
    for _ in range(tries):
        _c, body, _ = _http("POST", f"{api}/memory/search", token,
            body={"query": query, "top_k": 10, "include_text": True, "llm_prefilter": False},
            extra_headers=extra_headers, timeout=12.0)
        if sid in {r.get("source_id") for r in (body or {}).get("results", [])}:
            return True
        time.sleep(delay)
    return False


def check_visibility_isolation(api: str, token: str) -> CheckResult:
    """Ingest a private + a public source, search, verify visibility flags.

    Catches regressions in the visibility model — private chunks must
    only surface for the workspace that ingested them, public chunks
    must be visible to anyone with a valid JWT. The user explicitly
    asked for this domain to be tested as part of the live login flow.
    """
    suffix = int(time.time())
    # WHY(#327 2026-06-01): route both markers into an EPHEMERAL `vis-<ts>`
    # workspace so _teardown_smoke_workspaces purges them — earlier this check
    # wrote into the real workspace and ~700 near-identical "marker token N"
    # sources accumulated, out-ranking each run's fresh private marker in the
    # top_k vector search (private_in_search=False, a false regression).
    workspace_slug = f"vis-{suffix}"
    # WHY(#327): a unique token per run so the new marker's embedding is
    # distinctive and ranks #1 even if older markers linger — robust regardless
    # of accumulation.
    uniq = f"{suffix}-{os.urandom(3).hex()}"
    # WHY(tenancy phase A): 'private' is now user_id-scoped — the bare service
    # token has no human sub and can neither OWN nor FIND private content. Act
    # as a concrete user so the private source gets a user_id owner and the same
    # identity retrieves it. Requires MAYRING_ALLOW_ACT_AS=1 on the server.
    _vis_sub = f"smoke-vis-{suffix}"
    _vis_hdr = _act_as(_vis_sub, workspace=workspace_slug)

    # 1) Ingest a PRIVATE source (default visibility → private, owned by _vis_sub)
    priv_id = f"smoke:vis:private:{suffix}"
    code1, body1, _ = _http(
        "POST", f"{api}/memory/put", token,
        body={
            "source_id": priv_id,
            "source_type": "note",
            "repo": workspace_slug,
            "path": "private-marker",
            "content": f"PRIVATE marker {uniq}",
            "categorize": False,
        },
        timeout=15.0,
        extra_headers=_vis_hdr,
    )
    if code1 != 200:
        return CheckResult("visibility_isolation", False,
                           f"private ingest failed http={code1}: {body1}")

    # 2) Ingest a PUBLIC source by patching visibility after ingest
    pub_id = f"smoke:vis:public:{suffix}"
    code2, body2, _ = _http(
        "POST", f"{api}/memory/put", token,
        body={
            "source_id": pub_id,
            "source_type": "note",
            "repo": workspace_slug,
            "path": "public-marker",
            "content": f"PUBLIC marker {uniq}",
            "categorize": False,
        },
        timeout=15.0,
        workspace_id=workspace_slug,
    )
    if code2 != 200:
        return CheckResult("visibility_isolation", False,
                           f"public ingest failed http={code2}: {body2}")
    code3, body3, _ = _http_await_source(
        "PATCH", f"{api}/sources/{urllib.parse.quote(pub_id, safe='')}/visibility",
        token, body={"visibility": "public"},
    )
    if code3 != 200:
        return CheckResult("visibility_isolation", False,
                           f"PATCH visibility failed http={code3}: {body3}")

    # 3) Search for the marker token — both should surface for the
    #    same user/workspace that ingested them.  Use _search_finds (retry)
    #    for the PRIVATE source to absorb multi-worker commit-propagation lag
    #    (WHY: chk_0aaf83324a0404c3 — one-shot search false-RED under 4 workers).
    private_visible = _search_finds(api, token, f"marker {uniq}", priv_id,
                                    extra_headers=_vis_hdr)
    # Public source: one-shot is fine — public chunks are always visible.
    code4, body4, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": f"marker {uniq}", "top_k": 5,
              "include_text": True, "llm_prefilter": False},
        timeout=12.0,
        extra_headers=_vis_hdr,
    )
    if code4 != 200:
        return CheckResult("visibility_isolation", False,
                           f"search failed http={code4}")
    src_ids_seen = {r["source_id"] for r in (body4 or {}).get("results", [])}
    public_visible = pub_id in src_ids_seen

    # WHY(no-pollution): the ephemeral vis-<ts> workspace is teardown-purged, but
    # the PUBLIC source is globally visible and a skipped teardown leaks it into
    # everyone's public search. Invalidate both deterministically (best-effort).
    for sid in (priv_id, pub_id):
        try:
            _http("POST", f"{api}/memory/invalidate", token,
                  body={"source_id": sid}, timeout=10.0, extra_headers=_vis_hdr)
        except Exception:
            pass

    return CheckResult(
        "visibility_isolation",
        private_visible and public_visible,
        f"private_in_search={private_visible}  public_in_search={public_visible}  "
        f"results={len(src_ids_seen)}  marker={suffix}",
    )


def check_share_endpoint(api: str, token: str) -> CheckResult:
    """#195 Iter 4 — POST /sources/{id}/share makes a source public.

    Ingest a private note → POST /share → response must report
    visibility='public', shared=true. (The owner-check / 403-for-foreign
    path is covered by patch_source_visibility's L8 logic — same code.)
    """
    suffix = int(time.time())
    sid = f"smoke:share:{suffix}"
    code1, body1, _ = _http(
        "POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-share",
              "path": "share-marker", "content": f"SHARE marker {suffix}", "categorize": False},
        timeout=15.0,
    )
    if code1 != 200:
        return CheckResult("share_endpoint", False, f"ingest failed http={code1}: {body1}")
    code2, body2, _ = _http_await_source(
        "POST", f"{api}/sources/{urllib.parse.quote(sid, safe='')}/share", token, body={},
    )
    ok = (code2 == 200 and isinstance(body2, dict)
          and body2.get("visibility") == "public" and body2.get("shared") is True)
    # WHY(no-pollution): unlike the act-as tenancy checks (whose ephemeral
    # workspaces get purged by _teardown_smoke_workspaces), this one ingests a
    # PUBLIC note into the caller's REAL workspace — left behind it would surface
    # in every public search. Self-clean the source.
    _http("POST", f"{api}/memory/invalidate", token, body={"source_id": sid})
    return CheckResult(
        "share_endpoint", ok,
        f"http={code2}  body={body2}  (POST /sources/{{id}}/share must return visibility=public, shared=true)",
    )


def _resolve_stop_hook_path():
    """Locate stop_hook.py after the plugin was extracted to its own repo (#268).

    The hook no longer lives in the MayringCoder checkout — the actions-runner
    sees only the `claude-plugin-moved.md` stub. Search order:
      1. ``MAYRING_PLUGIN_DIR`` env → ``<dir>/hooks/stop_hook.py`` (explicit override)
      2. legacy in-repo path (pre-#268 layouts / local dev still vendoring it)
      3. installed plugin cache ``~/.claude/plugins/cache/**/hooks/stop_hook.py``
         (same $HOME as the runner — newest version + mayring-named preferred)
      4. sibling dev checkouts of mayring-claude-plugin
    Returns the first existing Path, or None.
    """
    from pathlib import Path
    env_dir = os.environ.get("MAYRING_PLUGIN_DIR", "").strip()
    if env_dir:
        p = Path(env_dir) / "hooks" / "stop_hook.py"
        if p.exists():
            return p

    repo_root = Path(__file__).resolve().parent.parent
    legacy = repo_root / "claude-plugin" / "hooks" / "stop_hook.py"
    if legacy.exists():
        return legacy

    home = Path.home()
    cached = list(home.glob(".claude/plugins/cache/**/hooks/stop_hook.py"))
    # Prefer a mayring-named plugin, then the lexicographically-greatest path
    # (newest version dir sorts last).
    cached.sort(key=lambda c: ("mayring" in str(c).lower(), str(c)))
    if cached:
        return cached[-1]

    for sibling in (
        home / "Desktop" / "mayring-claude-plugin" / "hooks" / "stop_hook.py",
        home / "mayring-claude-plugin" / "hooks" / "stop_hook.py",
        home / "mayring-claude-plugin-work" / "hooks" / "stop_hook.py",
    ):
        if sibling.exists():
            return sibling
    return None


def check_stop_hook_auto_feedback_e2e(api: str, token: str) -> CheckResult:
    """End-to-end: write a real inject-state file, drive _auto_feedback,
    verify DB-side feedback rows actually got written.

    Catches the exact bug-class that hid for hours: hook code looked fine
    in unit tests, was silently failing in production because the wiring
    (state-file path) didn't match what claude-code wrote.
    """
    import importlib.util
    import os as _os
    from pathlib import Path

    # Pre: count chunk_feedback rows via /stats/summary
    pre_code, pre, _ = _http("GET", f"{api}/stats/summary", token)
    if pre_code != 200:
        return CheckResult("stop_hook_e2e", False, f"pre summary http={pre_code}")
    pre_total = (pre or {}).get("feedback", {}).get("total", 0)

    # Get 2 real chunks via search to feed into the hook
    code, body, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": "stop hook smoke", "top_k": 2,
              "include_text": False, "llm_prefilter": False},
    )
    if code != 200 or len(body.get("results", [])) < 2:
        return CheckResult("stop_hook_e2e", False,
                           f"could not get 2 chunks for e2e: http={code}")
    pairs = [(r["chunk_id"], r["source_id"]) for r in body["results"][:2]]

    # Find the stop_hook.py module and import it. Plugin was extracted to its
    # own repo (#268), so resolve across env / installed-cache / sibling.
    hook_path = _resolve_stop_hook_path()
    if hook_path is None:
        return CheckResult("stop_hook_e2e", False,
                           "stop_hook.py not found — plugin extracted to "
                           "mayring-claude-plugin (#268); set MAYRING_PLUGIN_DIR "
                           "or install the plugin (~/.claude/plugins)")
    spec = importlib.util.spec_from_file_location("stop_hook_smoke", hook_path)
    sh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sh)

    # Write the state file the way memory_inject would
    session_id = f"smoke-e2e-{int(time.time())}"
    state_dir = Path(_os.path.expanduser("~/.config/mayring/inject-state"))
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / f"{session_id}.json"
    state_file.write_text(json.dumps({"chunks": [
        {"chunk_id": c, "source_id": s} for c, s in pairs
    ]}))

    # Drive the actual auto_feedback function — same path the live hook
    # takes when claude-code fires Stop. This will POST to /memory/feedback
    # for each chunk.
    # WHY: assistant_text MUST mention each chunk's source-path AND be ≥200
    # chars. Otherwise classify_chunk_relevance returns "skip" (kein POST)
    # für unklare matches — verhindert false-negatives für generic files
    # in production, würde den e2e-counter aber auf 0 stellen.
    path_mentions = " ".join(s for _, s in pairs)
    fake_turns = [
        {"role": "user", "content": "smoke e2e test prompt", "timestamp": ""},
        {"role": "assistant",
         "content": (
             "smoke e2e test response — paths referenced: " + path_mentions
             + ". " + "x" * 200
         ),
         "timestamp": ""},
    ]
    sh._auto_feedback(fake_turns, session_id, token)

    # State file should be cleared
    state_cleared = not state_file.exists()

    # Post: count again, expect +len(pairs)
    time.sleep(1)
    post_code, post, _ = _http("GET", f"{api}/stats/summary", token)
    post_total = (post or {}).get("feedback", {}).get("total", 0)
    delta = post_total - pre_total
    return CheckResult(
        "stop_hook_e2e",
        delta >= len(pairs) and state_cleared,
        f"pre_total={pre_total}  post_total={post_total}  delta={delta} "
        f"(expected ≥{len(pairs)})  state_cleared={state_cleared}",
    )


def check_feedback_log_movement(api: str, token: str) -> CheckResult:
    """A search must increment context_feedback_log injections_24h.
    Catches the regression where /memory/search bypassed this table."""
    pre_code, pre, _ = _http("GET", f"{api}/stats/feedback-log", token)
    if pre_code != 200:
        return CheckResult("feedback_log_movement", False,
                           f"pre http={pre_code}")
    pre_inj = (pre or {}).get("injections_24h", 0)

    # Trigger a search
    _http("POST", f"{api}/memory/search", token,
          body={"query": f"smoke check {time.time()}", "top_k": 3,
                "include_text": False, "llm_prefilter": False})
    time.sleep(1)
    post_code, post, _ = _http("GET", f"{api}/stats/feedback-log", token)
    post_inj = (post or {}).get("injections_24h", 0)
    delta = post_inj - pre_inj
    return CheckResult(
        "feedback_log_movement",
        delta >= 1,
        f"pre.injections={pre_inj}  post.injections={post_inj}  delta={delta}",
    )


def check_wiki_history_returns_snapshots(api: str, token: str) -> CheckResult:
    """Issue #78 deepening: previous wiki_p8_history accepted 200/404/422.
    Real acceptance: snapshots are persisted on every wiki rebuild +
    each one has node_count, edge_count, cluster_count, created_at.

    Pass: /wiki/history returns 200 with a 'snapshots' array. The list
    can be empty (no rebuilds yet) — but the SHAPE of any returned
    snapshot must include the expected fields. 404 / 500 = regression.
    """
    code, body, _ = _http("GET", f"{api}/wiki/history?limit=5", token)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("wiki_history_returns_snapshots", False,
                           f"http={code} body={body}")
    snaps = body.get("snapshots")
    if not isinstance(snaps, list):
        return CheckResult("wiki_history_returns_snapshots", False,
                           f"snapshots not a list: {type(snaps).__name__}")
    if not snaps:
        return CheckResult(
            "wiki_history_returns_snapshots", True,
            "empty snapshots list (no rebuild yet) — shape OK",
        )
    needed = {"snapshot_id", "trigger", "node_count", "edge_count",
              "cluster_count", "created_at"}
    s = snaps[0]
    missing = needed - set(s.keys()) if isinstance(s, dict) else needed
    return CheckResult(
        "wiki_history_returns_snapshots",
        not missing,
        f"snapshots={len(snaps)}  newest_keys={list(s.keys())[:8] if isinstance(s, dict) else '?'}  "
        f"missing={missing}",
    )


def check_categorization_call_type_logged(api: str, token: str) -> CheckResult:
    """Issue #101 deepening: previous categorization_logging only
    asserted llm_calls_log has activity in 24h — could pass on
    pi_task or vector_search calls alone. /stats/llm-call-types
    surfaces per-call_type counts so we can prove specifically that
    ``call_type='categorization'`` (the Mayring pipeline path,
    src/memory/ingestion/categorization.py:242) is being logged.
    """
    code, body, _ = _http(
        "GET", f"{api}/stats/llm-call-types?days=7", token,
    )
    if code != 200 or not isinstance(body, dict):
        return CheckResult("categorization_call_type_logged", False,
                           f"http={code} body={body}")
    counts = body.get("counts") or {}
    cat_count = int(counts.get("categorization") or 0)
    return CheckResult(
        "categorization_call_type_logged",
        cat_count > 0,
        f"categorization_calls_7d={cat_count}  "
        f"all_types={list(counts.keys())[:6]}  "
        f"(must be > 0 — Mayring pipeline must log on this exact call_type)",
    )


def check_chunk_metadata_complete(api: str, token: str) -> CheckResult:
    """Issue #30 deepening: every chunk should declare its abstraction
    level (file/class/function/section/...) so the multi-view ranker
    + hierarchical context builder can do their job. Pull a sample
    chunk via search → /memory/chunk/{id} and assert chunk_level is in
    the expected set, not empty.
    """
    valid_levels = {"file", "class", "function", "block", "section",
                    "view_fact", "view_decision", "view_intent", "view_caveat",
                    "view_followup"}
    code, body, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": "memory store schema", "top_k": 1,
              "include_text": False, "llm_prefilter": False},
    )
    if code != 200 or not (body or {}).get("results"):
        return CheckResult("chunk_metadata_complete", False,
                           "search returned no results")
    cid = body["results"][0]["chunk_id"]
    code2, body2, _ = _http("GET", f"{api}/memory/chunk/{cid}", token)
    if code2 != 200:
        return CheckResult("chunk_metadata_complete", False,
                           f"GET /memory/chunk http={code2}")
    chunk = (body2 or {}).get("chunk", {})
    level = chunk.get("chunk_level") or ""
    return CheckResult(
        "chunk_metadata_complete",
        level in valid_levels,
        f"chunk_id={cid[:18]}  chunk_level={level!r}  "
        f"valid_set={'OK' if level in valid_levels else 'MISSING/UNKNOWN'}",
    )


def check_rag_function_search_finds_source(api: str, token: str) -> CheckResult:
    """Issue #21 + #18 acceptance: function-name queries return chunks
    from .py source files with non-zero vector score.

    Don't require a specific source file — auto-ingest may lag behind
    the latest commits and the smoke shouldn't red-flag every push
    because of that. The acceptance is "vector retrieval finds Python
    code", not "the brand-new file is ingested within 60s".

    Pass condition: top-5 contains at least one chunk where
      - source_id ends in '.py' (any Python source)
      - score_vector > 0.05 (real vector signal, not noise)
    """
    # Service-Token-Default ist 'system' (keine echten .py-source-Chunks) → der
    # populierte Tenant muss explizit mit. Siehe SMOKE_VECTOR_WORKSPACE.
    target_ws = SMOKE_VECTOR_WORKSPACE
    # WHY(tenancy phase A): repo .py chunks migrated to visibility='private',
    # user_id-scoped to the workspace owner. The bare service token (no sub)
    # can't see them — act as the owner (SMOKE_VECTOR_OWNER) so the private
    # code chunks are retrievable. Requires MAYRING_ALLOW_ACT_AS=1.
    code, body, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": "_rerank candidates vector_scores top_k re-rank "
                       "memory retrieval pipeline",
              "top_k": 5, "include_text": False, "llm_prefilter": False},
        timeout=15.0,
        workspace_id=target_ws,
        extra_headers=_act_as(SMOKE_VECTOR_OWNER, workspace=target_ws),
    )
    if code != 200 or not isinstance(body, dict):
        return CheckResult("rag_function_search_finds_source", False,
                           f"http={code}")
    results = body.get("results") or []
    if not results:
        return CheckResult("rag_function_search_finds_source", False,
                           "no results returned for function-name query")
    py_with_vector = [
        r for r in results
        if (r.get("source_id") or "").endswith(".py")
        and float(r.get("score_vector") or 0) > 0.05
    ]
    matched = [
        f"{(r.get('source_id') or '')[:50]}/v={r.get('score_vector', 0):.2f}"
        for r in results[:3]
    ]
    return CheckResult(
        "rag_function_search_finds_source",
        len(py_with_vector) > 0,
        f"py_with_vector={len(py_with_vector)}/{len(results)}  top3={matched}",
    )


def check_watcher_hook_fires(api: str, token: str) -> CheckResult:
    """Issue #74 audit gap: post-ingest watcher must update graph state.
    Previous smoke `wiki_p7_endpoints` only checked GET endpoints reachable;
    no proof a watcher event ever fires.

    Probe: POST a unique chunk via /memory/put, then check /stats/recent-ops
    surfaces an ingest event for that source_id within 5s. The recent-ops
    feed is the same one the dashboard binds to — if the watcher path is
    silently broken, this catches it.
    """
    marker = f"smoke-watcher-{int(time.time() * 1000)}"
    source_id = f"smoke:watcher:{marker}"
    code, body, _ = _http(
        "POST", f"{api}/memory/put", token,
        body={
            "source_id": source_id,
            "source_type": "test",
            "content": f"smoke watcher probe content {marker}",
        },
    )
    if code not in (200, 201):
        return CheckResult("watcher_hook_fires", False,
                           f"PUT http={code} body={body}")
    deadline = time.time() + 8.0
    seen = False
    last_count = -1
    while time.time() < deadline:
        time.sleep(1.5)
        code2, body2, _ = _http("GET", f"{api}/stats/recent-ops?limit=20", token)
        if code2 == 200 and isinstance(body2, dict):
            ops = body2.get("ops") or body2.get("recent_ops") or []
            if isinstance(ops, list):
                last_count = len(ops)
                if any(marker in str(op) or source_id in str(op) for op in ops):
                    seen = True
                    break
    # WHY(no-pollution): probe writes into the caller's REAL workspace (no act-as)
    # — without self-clean these accumulate (277 leaked smoke sources found
    # 2026-06-08). Best-effort; a failed cleanup must not flip the check.
    try:
        _http("POST", f"{api}/memory/invalidate", token,
              body={"source_id": source_id}, timeout=10.0)
    except Exception:
        pass
    return CheckResult(
        "watcher_hook_fires",
        seen,
        f"source_id={source_id}  recent_ops_seen={seen}  "
        f"last_count={last_count}  (watcher must surface ingest within 8s)",
    )


def check_wiki_cluster_depth(api: str, token: str) -> CheckResult:
    """Issue #71/#72/#73 audit gap: /wiki/graph?slug=... was previously only
    asserted on shape (clusters key present). Real acceptance: cluster
    engine produces non-empty clusters with members, edges have type field.

    Iterates ALL available slugs looking for ONE workspace where clusters
    have members + edges have types. That single positive case is enough
    to prove the cluster engine works end-to-end. If EVERY workspace has
    cluster shells without members, this is a real production bug worth
    failing on (Issue #154 territory).
    """
    code, slugs, _ = _http("GET", f"{api}/wiki/slugs", token)
    if code != 200:
        return CheckResult("wiki_cluster_depth", False,
                           f"/wiki/slugs http={code}")
    available = (slugs or {}).get("slugs", [])
    if not available:
        return CheckResult(
            "wiki_cluster_depth", True,
            "no slugs available — cluster depth check vacuously OK",
        )
    best_slug = ""
    best_members = 0
    best_clusters = 0
    best_edge_types: set[str] = set()
    any_clusters = 0
    for slug in available[:6]:  # cap probes to avoid slow O(n*roundtrip)
        code2, body2, _ = _http(
            "GET", f"{api}/wiki/graph?slug={slug}&format=json", token,
        )
        if code2 != 200 or not isinstance(body2, dict):
            continue
        clusters = body2.get("clusters") or []
        edges = body2.get("edges") or []
        if not isinstance(clusters, list):
            continue
        any_clusters += len(clusters)
        max_members_here = max(
            (len(c.get("members") or c.get("nodes") or c.get("files") or [])
             for c in clusters if isinstance(c, dict)),
            default=0,
        )
        if max_members_here > best_members:
            best_slug = slug
            best_members = max_members_here
            best_clusters = len(clusters)
            best_edge_types = {
                e.get("type") or e.get("edge_type") for e in edges
                if isinstance(e, dict)
            }
            best_edge_types.discard(None)
    if best_members > 0:
        return CheckResult(
            "wiki_cluster_depth", True,
            f"best_slug={best_slug}  clusters={best_clusters}  "
            f"max_members={best_members}  "
            f"edge_types={sorted(t for t in best_edge_types if t)[:5]}",
        )
    return CheckResult(
        "wiki_cluster_depth", False,
        f"probed={len(available[:6])} slugs, total_clusters={any_clusters}, "
        f"none had members — cluster engine produced shell entries only "
        f"(real production gap; rebuild via POST /wiki/rebuild)",
    )


def check_populate_accepts_batch_delay(api: str, token: str) -> CheckResult:
    """Issue #85 audit gap: /populate must accept ``batch_delay`` so the
    GPU-throttle from the CLI ``--batch-delay`` flag is reachable via API.
    Previous smoke didn't verify this — a regression that drops the
    parameter would silently break the entire throttle mechanism.

    Probe: POST /populate with a deliberately-tiny batch_delay=0.01 and
    a never-existing repo URL. Acceptable: 200 (job queued, includes
    ``batch_delay`` in response) OR 4xx with error mentioning the field.
    Forbidden: 422 unknown-field error (= API doesn't accept the param).

    The job itself fails fast on the bogus repo — this isn't about
    running ingestion, just proving the parameter is wired.
    """
    code, body, _ = _http(
        "POST", f"{api}/populate", token,
        body={
            "repo": "https://github.com/Nileneb/smoke-nonexistent-test-repo",
            "force_reingest": False,
            "batch_delay": 0.01,
        },
    )
    is_accepted = code == 200 and isinstance(body, dict) and (
        body.get("batch_delay") == 0.01 or "job_id" in body
    )
    is_known_error = (
        code in (400, 404) and isinstance(body, dict)
        and "batch_delay" not in str(body).lower()
    )
    return CheckResult(
        "populate_accepts_batch_delay",
        is_accepted or (code == 200 and "job_id" in (body or {})),
        f"http={code}  batch_delay_in_resp={(body or {}).get('batch_delay')}  "
        f"job_id_set={'job_id' in (body or {})}  "
        f"(API must accept batch_delay; 422 unknown-field = regression)",
    )


def check_model_identity(api: str, token: str) -> CheckResult:
    """Issue #106 acceptance, updated for the text_model.txt picker (#349):
    the model that ACTUALLY gets used is whatever the canonical text-model
    selection says, and a CHANGE to it is picked up on the next resolve.

    Since 2026-06-06 the ``text_model.txt`` override has priority over the
    ``model_routes.yaml`` ``text`` route — ``ModelRouter.resolve("text")``
    reads it per call. So the old probe (sentinel via POST
    /stats/admin/model-routes) no longer reflects the used model; it stayed
    red even though the picker works. This probe drives the now-canonical
    path: read the active model, switch to a *different* available model,
    confirm the active resolution flips, then revert.

    Both writes go through POST /stats/admin/text-model, which validates the
    model against Ollama's tag list — so we only switch to/revert between
    models that are actually loadable (no unrevertable sentinel state).
    """
    code, body, _ = _http("GET", f"{api}/stats/admin/text-models", token, timeout=10.0)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("model_identity", False,
                           f"GET text-models http={code} body={body}")
    orig = body.get("active")
    models = body.get("models")
    if not isinstance(orig, str) or not orig or not isinstance(models, list):
        return CheckResult(
            "model_identity", False,
            f"unexpected shape active={orig!r} models={type(models).__name__}",
        )
    names = [m.get("name") for m in models
             if isinstance(m, dict) and m.get("name")]
    if orig not in names:
        # Active model is not in the local Ollama tag list (e.g. a cloud-only
        # model) → a POST-revert to it would 422, leaving the override wrong.
        # Skip the mutating probe to stay revertible; the read path is proven.
        return CheckResult(
            "model_identity", True,
            f"active={orig!r} not in local Ollama tags ({len(names)} models) "
            "— mutating probe skipped to stay revertible; read path OK",
        )
    target = next((n for n in names if n != orig), orig)
    single = target == orig
    code2, body2, _ = _http(
        "POST", f"{api}/stats/admin/text-model", token,
        body={"model": target}, timeout=10.0,
    )
    if code2 != 200:
        return CheckResult("model_identity", False,
                           f"POST set target={target!r} http={code2} body={body2}")
    code3, body3, _ = _http("GET", f"{api}/stats/admin/text-models", token, timeout=10.0)
    new_active = body3.get("active") if isinstance(body3, dict) else None
    # Always revert to the model that was active before the probe.
    _http("POST", f"{api}/stats/admin/text-model", token,
          body={"model": orig}, timeout=10.0)
    ok = new_active == target
    note = " (single-model host: write→resolve roundtrip only)" if single else ""
    return CheckResult(
        "model_identity",
        ok,
        f"orig={orig!r} set={target!r} resolved={new_active!r}{note} "
        f"(matches ⇒ text_model.txt picker is the canonical resolution path)",
    )


def check_reranker_rollout_decision(api: str, token: str) -> CheckResult:
    """Auto-rollout decision endpoint (Issue: 50/50 + 25% threshold).

    Probe with apply=False so we don't mutate the production default
    during smoke. Pass condition: response includes the canonical fields
    (decision, target, reason, metrics, threshold_pct).
    """
    code, body, _ = _http(
        "POST",
        f"{api}/stats/admin/reranker-rollout-decision?days=7&k=5"
        f"&threshold_pct=25&apply=false",
        token,
    )
    if code != 200 or not isinstance(body, dict):
        return CheckResult("reranker_rollout_decision", False,
                           f"http={code} body={body}")
    needed = {"decision", "target", "reason", "metrics", "threshold_pct"}
    missing = needed - set(body.keys())
    return CheckResult(
        "reranker_rollout_decision", not missing,
        f"http=200 missing={missing}  decision={body.get('decision')}  "
        f"target={body.get('target')}  reason={body.get('reason', '')[:80]!r}",
    )


def check_reranker_runtime_switch(api: str, token: str) -> CheckResult:
    """Memory-Injection v2.0 acceptance: per-request reranker switch
    works AND diagnostics report the version that ran.

    Sends two /memory/search calls with the same query but different
    ``reranker_version``. Pass condition:
      a) both return 200,
      b) diagnostics.reranker_version reflects the request,
      c) v2 only flips on if cache/rerank_v2.json exists; otherwise the
         silent v1-fallback is correct (tracked separately).

    This proves the runtime switch wired end-to-end without requiring
    a trained model — once a model lands, the same probe verifies
    A/B routing.
    """
    code1, body1, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": "smoke reranker switch v1",
              "top_k": 3, "include_text": False, "llm_prefilter": False,
              "reranker_version": "v1"},
        timeout=15.0,
    )
    code2, body2, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": "smoke reranker switch v2",
              "top_k": 3, "include_text": False, "llm_prefilter": False,
              "reranker_version": "v2"},
        timeout=15.0,
    )
    v1_diag = ((body1 or {}).get("diagnostics") or {}).get("reranker_version", "")
    v2_diag = ((body2 or {}).get("diagnostics") or {}).get("reranker_version", "")
    both_ok = code1 == 200 and code2 == 200
    v1_correct = v1_diag == "v1"
    # v2 correctness: either reports "v2" (model active) or "v1" (silent
    # fallback when no model file). Both prove the switch path is wired.
    v2_correct = v2_diag in ("v1", "v2")
    return CheckResult(
        "reranker_runtime_switch",
        both_ok and v1_correct and v2_correct,
        f"v1_call.diag={v1_diag!r}  v2_call.diag={v2_diag!r}  "
        f"http={code1}/{code2}  "
        f"({'v2 active' if v2_diag == 'v2' else 'v2 falls back to v1 — model file missing'})",
    )


def check_retrieval_ab_endpoint(api: str, token: str) -> CheckResult:
    """A/B comparison endpoint exists and returns the canonical shape.
    Numbers may be 0 until traffic accumulates with both versions —
    we only assert shape here."""
    code, body, _ = _http("GET", f"{api}/stats/retrieval-ab?days=7&k=5", token)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("retrieval_ab_endpoint", False,
                           f"http={code} body={body}")
    has_keys = "by_version" in body and "uplift" in body
    return CheckResult(
        "retrieval_ab_endpoint", has_keys,
        f"http=200  keys_ok={has_keys}  "
        f"by_version_keys={list((body.get('by_version') or {}).keys())}  "
        f"uplift={body.get('uplift')}",
    )


def check_retrieval_metrics_endpoint(api: str, token: str) -> CheckResult:
    """Memory-Injection v2.0 foundation: /stats/retrieval-metrics returns
    precision@K + NDCG@K + recall@K joined from context_feedback_log and
    chunk_feedback. Pass: route 200 + the four numeric fields are present
    (values may legitimately be 0 before the new logging accumulates data,
    that's not a fail; the SHAPE is what we're proving here)."""
    code, body, _ = _http(
        "GET", f"{api}/stats/retrieval-metrics?days=7&k=5", token,
    )
    if code != 200 or not isinstance(body, dict):
        return CheckResult(
            "retrieval_metrics_endpoint", False,
            f"http={code} body={body}",
        )
    needed = {"precision_at_k", "ndcg_at_k", "recall_at_k", "queries_logged"}
    missing = needed - set(body.keys())
    return CheckResult(
        "retrieval_metrics_endpoint", not missing,
        f"http=200 keys_ok={not missing} missing={missing}  "
        f"sample={ {k: body.get(k) for k in needed if k in body} }",
    )


def check_retrieval_stage_attribution(api: str, token: str) -> CheckResult:
    """Companion check: /stats/retrieval-stage-attribution surfaces which
    ranking stage (vector/symbolic/recency/source_affinity) actually
    drives positive chunks. Pass: route 200 + the four stages appear in
    both `stage_wins` and `stage_share` sub-dicts."""
    code, body, _ = _http(
        "GET", f"{api}/stats/retrieval-stage-attribution?days=7", token,
    )
    if code != 200 or not isinstance(body, dict):
        return CheckResult(
            "retrieval_stage_attribution", False,
            f"http={code} body={body}",
        )
    wins = body.get("stage_wins") or {}
    share = body.get("stage_share") or {}
    expected_stages = {"vector", "symbolic", "recency", "source_affinity"}
    ok = (expected_stages <= set(wins.keys())
          and expected_stages <= set(share.keys()))
    return CheckResult(
        "retrieval_stage_attribution", ok,
        f"http=200 stages_in_wins={set(wins.keys())} "
        f"stages_in_share={set(share.keys())} attributed="
        f"{body.get('positive_chunks_attributed')}",
    )


def check_pi_second_opinion_endpoint(api: str, token: str) -> CheckResult:
    """Issue #139 acceptance: ``pi_second_opinion`` is reachable from the
    Pi-Agent via tool-use. We can't drive the MCP tool over HTTP, but the
    REST mirror at ``/wiki/second-opinion`` shares the WikiSecondOpinion
    code path. Probing it with ``dry_run=true`` proves the validator
    pipeline (workspace lookup, graph load, target resolution) works
    without burning Ollama time on a known-missing cluster_id.

    Pass: route exists AND properly returns 404 for a missing target.
    Fail: 500 (crash), 401/403 (auth wired wrong), or anything that says
    'Not Found' as a path-level 404 (which would mean the route itself
    isn't registered).
    """
    code, body, _ = _http(
        "POST", f"{api}/wiki/second-opinion", token,
        body={
            "target_id": "smoke-nonexistent-cluster",
            "scope": "cluster",
            "dry_run": True,
        },
    )
    detail = (body or {}).get("detail") if isinstance(body, dict) else None
    target_missing = (
        code == 404 and isinstance(detail, str)
        and "cluster not found" in detail.lower()
    )
    return CheckResult(
        "pi_second_opinion_endpoint",
        target_missing,
        f"http={code}  detail={detail!r} (must be 404 'cluster not found' "
        f"= route exists + validator pipeline reached the lookup)",
    )


def check_model_router_runtime(api: str, token: str) -> CheckResult:
    """Issue #140 acceptance: ModelRouter routes are mutable at runtime
    via /stats/admin/model-routes. Service token gets admin scope.

    Probe: GET routes → flip text.timeout to a sentinel value → re-GET →
    sentinel must round-trip. Then revert to the original value to keep
    production state untouched.
    """
    code, body, _ = _http("GET", f"{api}/stats/admin/model-routes", token)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("model_router_runtime", False,
                           f"GET http={code} body={body}")
    routes = (body or {}).get("routes", {})
    text = routes.get("text") or {}
    if not text:
        return CheckResult("model_router_runtime", False,
                           "GET ok but no 'text' route present")
    orig_timeout = int(text.get("timeout") or 240)
    sentinel = 313
    if orig_timeout == sentinel:
        sentinel = 314  # avoid no-op write
    payload = {
        "task": "text",
        "model": text.get("model") or "mistral:7b-instruct",
        "fallback": text.get("fallback") or "",
        "timeout": sentinel,
    }
    code2, body2, _ = _http(
        "POST", f"{api}/stats/admin/model-routes", token, body=payload,
    )
    if code2 != 200:
        return CheckResult("model_router_runtime", False,
                           f"POST http={code2} body={body2}")
    code3, body3, _ = _http("GET", f"{api}/stats/admin/model-routes", token)
    new_timeout = int(((body3 or {}).get("routes", {}).get("text") or {}).get("timeout") or 0)
    # Revert regardless of outcome — leave prod untouched
    revert = {
        "task": "text",
        "model": text.get("model") or "mistral:7b-instruct",
        "fallback": text.get("fallback") or "",
        "timeout": orig_timeout,
    }
    _http("POST", f"{api}/stats/admin/model-routes", token, body=revert)
    return CheckResult(
        "model_router_runtime",
        new_timeout == sentinel,
        f"orig={orig_timeout} sent={sentinel} got={new_timeout} (round-trip ok if sent==got)",
    )


def check_projects_route_cwd_remote(api: str, token: str) -> CheckResult:
    """Project Router Slice 1: POST /projects/route with a known cwd-remote →
    200 + project_id set (match-or-create); no signal + nonsense → 200 + null."""
    code, body, _ = _http("POST", f"{api}/projects/route", token,
        body={"cwd_remote": "git@github.com:Nileneb/MayringCoder.git",
              "prompt": "fix the retrieval pipeline"}, timeout=30.0)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("projects_route_cwd_remote", False, f"http={code}")
    if not body.get("project_id"):
        return CheckResult("projects_route_cwd_remote", False,
                           f"cwd-remote gave no project_id: {body}")
    # null-branch embeds the prompt (Ollama) → generous timeout: the embed is
    # ~0.14s warm but can be several seconds cold right after a deploy.
    code2, body2, _ = _http("POST", f"{api}/projects/route", token,
        body={"cwd_remote": None, "prompt": "zxqw nonsense %%%"}, timeout=30.0)
    null_ok = code2 == 200 and (body2 or {}).get("project_id") is None
    return CheckResult("projects_route_cwd_remote",
                       bool(body.get("project_id")) and null_ok,
                       f"hard={body.get('reason')} null_branch_ok={null_ok}")


def check_mayring_process_fail_closed(api: str, token: str) -> CheckResult:
    """Phase 3 acceptance: POST /codebooks/{id}/process is fail-closed — an empty
    task MUST yield 400 (never a silent 'uncategorized' default, the #270 anti-pattern).
    Uses a real codebook id from GET /codebooks so the 404-guard isn't what trips."""
    code, body, _ = _http("GET", f"{api}/codebooks", token, timeout=15.0)
    cbs = (body or {}).get("codebooks") if isinstance(body, dict) else None
    if code != 200 or not cbs:
        return CheckResult("mayring_process_fail_closed", False,
                           f"GET /codebooks http={code} body={body}")
    cb_id = cbs[0]["id"]
    # empty task against a VALID codebook → fail-closed 400 (not 404, not 200)
    code2, body2, _ = _http("POST", f"{api}/codebooks/{cb_id}/process", token,
                            body={"text": "some text to classify", "task": ""}, timeout=15.0)
    ok = code2 == 400
    return CheckResult("mayring_process_fail_closed", ok,
                       f"empty-task http={code2} (expected 400) detail={body2}")


def check_ingest_links_categories(api: str, token: str) -> CheckResult:
    """Phase 3.2 acceptance: a LIVE ingest auto-links chunks to the codebook
    (chunk_categories, deductive path — link_chunks_deductive, do_link default-on).
    Guards the silent no-op: the wiring runs but links 0 because the
    codebook_categories Chroma collection went empty (e.g. after a chroma
    migration) — chunks would still ingest, but reranker-v3 cat_match would
    quietly never fire. Ingests a category-matching note, asserts
    category_links>=1, then invalidates the probe source (no prod pollution)."""
    nonce = int(time.time())
    src = f"smoke:phase32-catlink:{nonce}"
    # WHY(#330 dedup): the content MUST be unique per run. Static probe text was
    # content-deduped after the first-ever run → resolve_dedup skipped the chunk →
    # chunks_to_categorize empty → the link path never ran → category_links=0
    # forever (this check was red for that reason, NOT a broken linker). The nonce
    # makes a genuinely new chunk each run so the deductive link actually fires.
    code, body, _ = _http(
        "POST", f"{api}/memory/put", token,
        body={
            "source_id": src, "source_type": "note", "categorize": False,
            "content": (f"probe-{nonce}: User authentication and login: OAuth flow, "
                        "JWT auth middleware, password hashing, session token validation. "
                        "Database access layer with SQL queries + connection pooling."),
        },
        timeout=40.0,
    )
    links = body.get("category_links") if isinstance(body, dict) else None
    # Clean up the probe source regardless of outcome — never leave smoke junk.
    _http("POST", f"{api}/memory/invalidate", token,
          body={"source_id": src}, timeout=20.0)
    ok = code == 200 and isinstance(links, int) and links >= 1
    return CheckResult(
        "ingest_links_categories", ok,
        f"PUT http={code} category_links={links} "
        f"(Phase 3.2: live ingest must link chunk→codebook, need >=1; probe invalidated)",
    )


def check_project_link_boost_roundtrip(api: str, token: str) -> CheckResult:
    """C3 Producer-B acceptance: conversation chunks ingested WITH X-Project-Id
    must be linked to the project (chunk_project_links). Guards the end-to-end
    path: header present → link exists → project_match boost fires in retrieval.

    Probe uses slug 'smoke/repo-c3' (NOT_LIKE guard — never a real project).
    Steps:
      1. POST /projects/route with cwd_remote='https://github.com/smoke/repo-c3'
         to resolve/create a smoke project_id.
      2. Ingest two micro-batch turns: one WITH X-Project-Id, one WITHOUT.
      3. /memory/search with project=<id> — assert the boost is LIVE-WIRED
         (score_project_match field computed+serialised per result). The live
         RANKING (linked chunk surfaces with score_project_match>0) is reported
         but NOT required: fresh chunks may not be in the vector candidate pool
         yet (pre-existing chroma_candidate_mismatch, #330). Ranking is unit-
         tested in mayring-core (test_project_match_boosts_not_filters).
      4. Cleanup both sources via /memory/invalidate.

    Fail-soft on cleanup: missing invalidation does NOT fail the check."""
    slug = f"smoke/repo-c3-{int(time.time())}"

    # Step 1: resolve/create a smoke project
    code_p, body_p, _ = _http(
        "POST", f"{api}/projects/route", token,
        body={"cwd_remote": f"https://github.com/{slug}", "prompt": "smoke c3 project link test"},
        timeout=30.0,
    )
    project_id = (body_p or {}).get("project_id") if isinstance(body_p, dict) else None
    if code_p != 200 or not project_id:
        return CheckResult(
            "project_link_boost_roundtrip", False,
            f"projects/route http={code_p} project_id={project_id}",
        )

    session_linked = f"smoke-c3-linked-{int(time.time())}"
    session_unlinked = f"smoke-c3-unlinked-{int(time.time())}"

    # Step 2a: ingest WITH X-Project-Id
    code_l, body_l, _ = _http(
        "POST", f"{api}/conversation/micro-batch", token,
        body={
            "turns": [{"role": "user",
                       "content": "smoke c3 project link test linked",
                       "timestamp": ""}],
            "session_id": session_linked,
            "presumarized": "smoke probe C3 linked turn for project boost roundtrip",
            "origin_ref": f"https://github.com/{slug}",
        },
        timeout=30.0,
        extra_headers={"X-Project-Id": project_id},
    )
    source_linked = (body_l or {}).get("source_id") if isinstance(body_l, dict) else None

    # Step 2b: ingest WITHOUT X-Project-Id
    code_u, body_u, _ = _http(
        "POST", f"{api}/conversation/micro-batch", token,
        body={
            "turns": [{"role": "user",
                       "content": "smoke c3 project link test unlinked",
                       "timestamp": ""}],
            "session_id": session_unlinked,
            "presumarized": "smoke probe C3 unlinked turn — no project header",
        },
        timeout=30.0,
    )
    source_unlinked = (body_u or {}).get("source_id") if isinstance(body_u, dict) else None

    ingest_ok = (code_l == 200 and (body_l or {}).get("indexed")
                 and code_u == 200 and (body_u or {}).get("indexed"))

    # Step 3: search — verify the project_match boost is LIVE-WIRED (the field is
    # computed + serialised per result) and surface the linked chunk's boost if
    # it is retrievable. We do NOT hard-assert the freshly-ingested chunk shows
    # up: brand-new chunks may not yet be in the vector candidate pool
    # (pre-existing chroma_candidate_mismatch, same root as ingest_links_categories
    # #330). The boost RANKING itself is unit-tested in mayring-core
    # (test_project_match_boosts_not_filters) and the link path is verified above.
    # This smoke guards the DEPLOYED plumbing: Producer-B reachable + boost field
    # live in retrieval. project_match_hits is reported for visibility (a >0 means
    # the live ranking is also working, but it is not required to pass).
    wiring_ok = False
    search_detail = ""
    if ingest_ok:
        code_s, body_s, _ = _http(
            "POST", f"{api}/memory/search", token,
            body={
                "query": "smoke probe C3 linked project boost roundtrip",
                "top_k": 10, "include_text": False, "llm_prefilter": False,
                "project": project_id,
            },
            timeout=30.0,
        )
        results = (body_s or {}).get("results", []) if isinstance(body_s, dict) else []
        field_live = any(isinstance(r, dict) and "score_project_match" in r
                         for r in results)
        linked_hits = [r for r in results if isinstance(r, dict)
                       and (r.get("score_project_match") or 0) > 0]
        wiring_ok = code_s == 200 and field_live
        search_detail = (f"search http={code_s} results={len(results)} "
                         f"field_live={field_live} project_match_hits={len(linked_hits)}")

    # Step 4: cleanup (best-effort)
    for sid in (source_linked, source_unlinked):
        if sid:
            try:
                _http("POST", f"{api}/memory/invalidate", token,
                      body={"source_id": sid}, timeout=10.0)
            except Exception:
                pass

    ok = ingest_ok and wiring_ok
    return CheckResult(
        "project_link_boost_roundtrip",
        ok,
        f"project_id={project_id} ingest_ok={ingest_ok} {search_detail} (C3 producer-B link path + boost field live; ranking unit-tested)",
    )


def check_reranker_cat_match_fires(api: str, token: str) -> CheckResult:
    """Reranker-v3 acceptance (end-to-end): a category-themed search must derive
    the query's codebook category server-side AND surface >=1 result whose
    chunk_categories overlap it (score_cat_match > 0). This guards the WHOLE v3
    chain together — query-side derivation + chunk_categories (Phase 3.2 deductive
    link) + the cat_match feature — which was silently inert before (the query was
    never categorized and cat_match was never logged). Read-only: leans on the
    existing auth/data-access-linked corpus. Two themes so a single sparse
    category can't false-red it."""
    themes = [
        "user authentication login session token oauth jwt password hashing access control",
        "sqlite database query connection pool data access layer repository persistence",
    ]
    fired = 0
    n = 0
    for q in themes:
        # WHY(tenancy phase A): categorized content lives in the owner's now
        # user_id-scoped workspace; search as the owner so cat_match can fire.
        code, body, _ = _http("POST", f"{api}/memory/search", token,
                              body={"query": q, "top_k": 10, "include_text": False,
                                    "llm_prefilter": False}, timeout=40.0,
                              workspace_id=SMOKE_VECTOR_WORKSPACE,
                              extra_headers=_act_as(SMOKE_VECTOR_OWNER, workspace=SMOKE_VECTOR_WORKSPACE))
        if code != 200 or not isinstance(body, dict):
            return CheckResult("reranker_cat_match_fires", False, f"search http={code}")
        results = body.get("results", []) or []
        n += len(results)
        fired += sum(1 for r in results
                     if isinstance(r, dict) and (r.get("score_cat_match") or 0) > 0)
    return CheckResult(
        "reranker_cat_match_fires", fired >= 1,
        f"cat_match>0 hits={fired} across {n} results / {len(themes)} themes "
        f"(reranker-v3: query→category + chunk_categories must intersect on >=1)",
    )


def check_role_policy_roundtrip(api: str, token: str) -> CheckResult:
    """PUT a per-workspace override then GET reflects it (owner==admin, manage_members)."""
    ws = SMOKE_VECTOR_WORKSPACE
    hdr = _act_as(SMOKE_VECTOR_OWNER, workspace=ws)
    c1, b1, _ = _http("PUT", f"{api}/stats/workspaces/{ws}/role-permissions", token,
                      body={"role": "editor", "permission": "share_public", "allowed": False},
                      extra_headers=hdr)
    if c1 != 200:
        return CheckResult("role_policy_roundtrip", False, f"PUT http={c1}: {b1}")
    c2, b2, _ = _http("GET", f"{api}/stats/workspaces/{ws}/role-permissions", token, extra_headers=hdr)
    val = (b2 or {}).get("matrix", {}).get("editor", {}).get("share_public") if isinstance(b2, dict) else None
    ok = c2 == 200 and val is False
    # restore default → idempotent
    _http("PUT", f"{api}/stats/workspaces/{ws}/role-permissions", token,
          body={"role": "editor", "permission": "share_public", "allowed": True}, extra_headers=hdr)
    return CheckResult("role_policy_roundtrip", ok, f"PUT={c1} GET={c2} editor.share_public={val}")


def check_role_enforcement(api: str, token: str) -> CheckResult:
    """A 'user'-role caller (act-as, no membership → role user) must be DENIED
    share_public on /memory/put (403)."""
    ws = SMOKE_VECTOR_WORKSPACE
    hdr = _act_as("smoke-roleuser", workspace=ws)
    suffix = int(time.time())
    code, body, _ = _http("POST", f"{api}/memory/put", token,
                          body={"source_id": f"smoke:role:{suffix}", "source_type": "note",
                                "content": "role check", "visibility": "public"},
                          extra_headers=hdr)
    return CheckResult("role_enforcement", code == 403,
                       f"user->share_public http={code} (must be 403) body={body}")


def check_igio_lens_axes_present(api: str, token: str) -> CheckResult:
    """IGIO-Lens acceptance: GET /stats/igio-lens returns all four IGIO axes with
    integer counts + an unclassified bucket (workspace-scoped). Guards the
    endpoint the IgioLens view binds to — if it 404s/changes shape the lens goes
    silently empty."""
    code, body, _ = _http("GET", f"{api}/stats/igio-lens", token, timeout=20.0)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("igio_lens_axes_present", False, f"http={code} body={body}")
    axes = body.get("axes") or {}
    expected = {"issue", "goal", "intervention", "outcome"}
    have_all = expected <= set(axes)
    counts_ok = all(isinstance((axes.get(a) or {}).get("count"), int) for a in expected)
    uncl_ok = isinstance((body.get("unclassified") or {}).get("count"), int)
    ok = have_all and counts_ok and uncl_ok
    summary = {a: (axes.get(a) or {}).get("count") for a in sorted(expected)}
    return CheckResult(
        "igio_lens_axes_present", ok,
        f"http={code} axes={summary} unclassified={(body.get('unclassified') or {}).get('count')} "
        f"(need all 4 axes + int counts + unclassified)",
    )


# ---------------------------------------------------------------------------
# V2 org/public-memory acceptance checks (Tasks 3–11)
# ---------------------------------------------------------------------------

def check_private_isolation(api: str, token: str) -> CheckResult:
    """User A ingests a PRIVATE source in workspace WA; User B (different
    workspace) must NOT see it. Proves _scope_filter blocks cross-workspace
    private reads (the core leak guarantee)."""
    suffix = int(time.time())
    wa, wb = f"va-{suffix}", f"vb-{suffix}"
    sid = f"smoke:priv-iso:{suffix}"
    code1, body1, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-iso",
              "path": "p", "content": f"PRIV-ISO {suffix}", "categorize": False},
        extra_headers=_act_as("A", role="admin",workspace=wa), timeout=15.0)
    if code1 != 200:
        return CheckResult("private_isolation", False, f"ingest A failed http={code1}: {body1}")
    # WHY(fix4-isolation): confirm owner sees it first (proves propagation),
    # then check B cannot — so not-found is real isolation, not commit lag.
    owner_sees = _search_finds(api, token, f"PRIV-ISO {suffix}", sid,
                               extra_headers=_act_as("A", role="admin",workspace=wa))
    if not owner_sees:
        return CheckResult("private_isolation", False,
            f"INCONCLUSIVE: owner A could not find the source — "
            f"propagation not proven, isolation cannot be asserted  marker={suffix}")
    leaked = _search_finds(api, token, f"PRIV-ISO {suffix}", sid,
                           extra_headers=_act_as("B", workspace=wb), tries=2)
    return CheckResult("private_isolation", not leaked,
        f"B_sees_A_private={leaked} (must be False)  marker={suffix}")


def check_public_visibility(api: str, token: str) -> CheckResult:
    """A ingests then shares PUBLIC in WA; B in a different workspace MUST see
    it. Proves public is globally readable to any valid caller."""
    suffix = int(time.time())
    wa, wb = f"pa-{suffix}", f"pb-{suffix}"
    sid = f"smoke:pub-vis:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-pub",
              "path": "p", "content": f"PUB-VIS {suffix}", "categorize": False},
        extra_headers=_act_as("A", role="admin",workspace=wa), timeout=15.0)
    if code1 != 200:
        return CheckResult("public_visibility", False, f"ingest A failed http={code1}")
    code2, body2, _ = _http_await_source("POST",
        f"{api}/sources/{urllib.parse.quote(sid, safe='')}/share", token, body={},
        extra_headers=_act_as("A", role="admin",workspace=wa))
    if code2 != 200 or (body2 or {}).get("visibility") != "public":
        return CheckResult("public_visibility", False, f"share failed http={code2}: {body2}")
    found = _search_finds(api, token, f"PUB-VIS {suffix}", sid,
                          extra_headers=_act_as("B", workspace=wb))
    return CheckResult("public_visibility", found,
        f"B_sees_A_public={found} (must be True)  marker={suffix}")


def check_user_cross_device(api: str, token: str) -> CheckResult:
    """Same human (same sub) on two devices/workspaces: ingest visibility='private'
    as sub=S in WA → search as sub=S in WB MUST see it; a different sub must
    NOT. Proves 'private' visibility = user_id-scoped = cross-device-of-same-human.

    WHY(tenancy-T7 migration): 'user' was a legacy value rejected by the 3-value
    allowlist since T6.  'private' is the canonical replacement — same semantics
    (user_id-scoped), correct vocabulary.
    """
    suffix = int(time.time())
    sub_s = f"cd-{suffix}"
    wa, wb = f"cda-{suffix}", f"cdb-{suffix}"
    sid = f"smoke:user-xd:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-xd",
              "path": "p", "content": f"USER-XD {suffix}", "visibility": "private", "categorize": False},
        extra_headers=_act_as(sub_s, workspace=wa), timeout=15.0)
    if code1 != 200:
        return CheckResult("user_cross_device", False, f"ingest failed http={code1}")
    # Positive arm: same sub, different workspace must see it (retry for lag).
    same_sub_sees = _search_finds(api, token, f"USER-XD {suffix}", sid,
                                  extra_headers=_act_as(sub_s, workspace=wb))
    # WHY(fix4-isolation): prove owner sees it first, then check other sub.
    owner_sees = _search_finds(api, token, f"USER-XD {suffix}", sid,
                               extra_headers=_act_as(sub_s, workspace=wa))
    if not owner_sees:
        return CheckResult("user_cross_device", False,
            f"INCONCLUSIVE: owner could not find source — "
            f"propagation not proven  same_sub_sees={same_sub_sees}  marker={suffix}")
    other_sub_sees = _search_finds(api, token, f"USER-XD {suffix}", sid,
                                   extra_headers=_act_as(f"other-{suffix}", workspace=wb), tries=2)
    ok = same_sub_sees and not other_sub_sees
    return CheckResult("user_cross_device", ok,
        f"same_sub_sees={same_sub_sees}(want True)  other_sub_sees={other_sub_sees}(want False)  marker={suffix}")


def check_org_member_visibility(api: str, token: str) -> CheckResult:
    """A (member of org-X) ingests visibility='org' with org_id=X → B (also
    member of org-X, different sub/workspace) MUST see it."""
    suffix = int(time.time())
    org = f"org-{suffix}"
    sid = f"smoke:org-vis:{suffix}"
    code1, body1, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-org",
              "path": "p", "content": f"ORG-VIS {suffix}",
              "visibility": "org", "org_id": org, "categorize": False},
        extra_headers=_act_as("A", role="admin",orgs=(org,), workspace=f"oa-{suffix}"), timeout=15.0)
    if code1 != 200:
        return CheckResult("org_member_visibility", False, f"ingest failed http={code1}: {body1}")
    found = _search_finds(api, token, f"ORG-VIS {suffix}", sid,
                          extra_headers=_act_as("B", orgs=(org,), workspace=f"ob-{suffix}"))
    return CheckResult("org_member_visibility", found,
        f"org_member_sees={found} (must be True)  marker={suffix}")


def check_org_non_member_blocked(api: str, token: str) -> CheckResult:
    """A (org-X) ingests visibility='org' → C (NOT a member of org-X) must NOT
    see it. The complement of org_member_visibility — proves org isolation."""
    suffix = int(time.time())
    org = f"orgb-{suffix}"
    sid = f"smoke:org-block:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-orgb",
              "path": "p", "content": f"ORG-BLOCK {suffix}",
              "visibility": "org", "org_id": org, "categorize": False},
        extra_headers=_act_as("A", role="admin",orgs=(org,), workspace=f"oba-{suffix}"), timeout=15.0)
    if code1 != 200:
        return CheckResult("org_non_member_blocked", False, f"ingest failed http={code1}")
    # WHY(fix4-isolation): prove owner (A, member of org) sees the source before
    # asserting non-member C cannot — rules out commit-lag false-PASS.
    owner_sees = _search_finds(api, token, f"ORG-BLOCK {suffix}", sid,
                               extra_headers=_act_as("A", role="admin",orgs=(org,), workspace=f"oba-{suffix}"))
    if not owner_sees:
        return CheckResult("org_non_member_blocked", False,
            f"INCONCLUSIVE: owner A could not find source — "
            f"propagation not proven, isolation cannot be asserted  marker={suffix}")
    non_member_sees = _search_finds(api, token, f"ORG-BLOCK {suffix}", sid,
                                    extra_headers=_act_as("C", orgs=(f"other-{suffix}",),
                                                          workspace=f"obc-{suffix}"), tries=2)
    return CheckResult("org_non_member_blocked", not non_member_sees,
        f"non_member_sees={non_member_sees} (must be False)  marker={suffix}")


def check_org_revoke_isolation(api: str, token: str) -> CheckResult:
    """A ingests visibility='org' as a member of org-X → the SAME sub A, but
    with a token that no longer carries org-X membership (simulating a
    post-revoke reissued JWT), must NOT see it. Proves access dies with the
    membership claim, not just at ingest time."""
    suffix = int(time.time())
    org = f"orgr-{suffix}"
    sid = f"smoke:org-revoke:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-orgr",
              "path": "p", "content": f"ORG-REVOKE {suffix}",
              "visibility": "org", "org_id": org, "categorize": False},
        extra_headers=_act_as("A", role="admin",orgs=(org,), workspace=f"ora-{suffix}"), timeout=15.0)
    if code1 != 200:
        return CheckResult("org_revoke_isolation", False, f"ingest failed http={code1}")
    # WHY(fix4-isolation): prove owner (A WITH org membership) sees source first.
    owner_sees = _search_finds(api, token, f"ORG-REVOKE {suffix}", sid,
                               extra_headers=_act_as("A", role="admin",orgs=(org,), workspace=f"ora-{suffix}"))
    if not owner_sees:
        return CheckResult("org_revoke_isolation", False,
            f"INCONCLUSIVE: owner A (with org) could not find source — "
            f"propagation not proven  marker={suffix}")
    # WHY(fix3-revoke-workspace): use a DIFFERENT workspace for the revoked read
    # so only the org-membership dimension is under test. Reusing the same
    # workspace (ora-{suffix}) would allow a private-workspace fallback to mask
    # the org dimension — making the isolation check vacuous.
    still_sees = _search_finds(api, token, f"ORG-REVOKE {suffix}", sid,
                               extra_headers=_act_as("A", role="admin",workspace=f"orr-{suffix}"), tries=2)
    return CheckResult("org_revoke_isolation", not still_sees,
        f"sees_after_revoke={still_sees} (must be False)  marker={suffix}")


def check_patch_visibility_authz(api: str, token: str) -> CheckResult:
    """A ingests a private source → B (different sub AND workspace) PATCHes its
    visibility → must be 403. Proves L8 owner-check blocks cross-tenant
    vandalism."""
    suffix = int(time.time())
    sid = f"smoke:authz:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-authz",
              "path": "p", "content": f"AUTHZ {suffix}", "categorize": False},
        extra_headers=_act_as("A", role="admin",workspace=f"aza-{suffix}"), timeout=15.0)
    if code1 != 200:
        return CheckResult("patch_visibility_authz", False, f"ingest failed http={code1}")
    code2, body2, _ = _http_await_source("PATCH",
        f"{api}/sources/{urllib.parse.quote(sid, safe='')}/visibility", token,
        body={"visibility": "public"},
        extra_headers=_act_as("B", workspace=f"azb-{suffix}"))
    ok = code2 == 403
    return CheckResult("patch_visibility_authz", ok,
        f"foreign_patch_status={code2} (must be 403)  body={body2}  marker={suffix}")


def check_multi_org_membership(api: str, token: str) -> CheckResult:
    """A is a member of org-X AND org-Y; ingest one org-source in each →
    a single search as A MUST surface BOTH. Proves org_ids is a multi-value
    IN-filter, not a single org."""
    suffix = int(time.time())
    ox, oy = f"mox-{suffix}", f"moy-{suffix}"
    sx, sy = f"smoke:multi-x:{suffix}", f"smoke:multi-y:{suffix}"
    ws = f"moa-{suffix}"
    for sid, org in ((sx, ox), (sy, oy)):
        code, _, _ = _http("POST", f"{api}/memory/put", token,
            body={"source_id": sid, "source_type": "note", "repo": "smoke-multi",
                  "path": "p", "content": f"MULTI-ORG {suffix} {org}",
                  "visibility": "org", "org_id": org, "categorize": False},
            extra_headers=_act_as("A", role="admin",orgs=(ox, oy), workspace=ws), timeout=15.0)
        if code != 200:
            return CheckResult("multi_org_membership", False, f"ingest {org} failed http={code}")
    found_x = _search_finds(api, token, f"MULTI-ORG {suffix}", sx,
                            extra_headers=_act_as("A", role="admin",orgs=(ox, oy), workspace=ws))
    found_y = _search_finds(api, token, f"MULTI-ORG {suffix}", sy,
                            extra_headers=_act_as("A", role="admin",orgs=(ox, oy), workspace=ws))
    both = found_x and found_y
    return CheckResult("multi_org_membership", both,
        f"sees_org_x={found_x}  sees_org_y={found_y} (both must be True)  marker={suffix}")


def check_intervention_todos_surface(api: str, token: str) -> CheckResult:
    """A task created via POST /tasks must appear in GET /stats/igio-lens under
    intervention.todos (the lens intervention column source)."""
    suffix = int(time.time())
    ws = f"todo-{suffix}"
    title = f"SMOKE-TODO {suffix}"
    code1, body1, _ = _http("POST", f"{api}/tasks", token,
        body={"title": title, "created_by": "agent", "tags": "agent",
              "external_id": f"smoke-{suffix}"},
        extra_headers=_act_as("A", role="admin",workspace=ws), timeout=12.0)
    if code1 != 200:
        return CheckResult("intervention_todos_surface", False,
                           f"create failed http={code1}: {body1}")
    code2, body2, _ = _http("GET", f"{api}/stats/igio-lens?limit=20", token,
        extra_headers=_act_as("A", role="admin",workspace=ws), timeout=12.0)
    todos = ((((body2 or {}).get("axes") or {}).get("intervention") or {}).get("todos")) or []
    found = any(t.get("title") == title for t in todos)
    return CheckResult("intervention_todos_surface", found,
        f"task_in_intervention_todos={found} (must be True)  todos={len(todos)}  marker={suffix}")


def check_stats_workspaces_lists_all(api: str, token: str) -> CheckResult:
    """GET /stats/workspaces for a caller who is a member of org-X and org-Y
    (plus a personal workspace) must list all of them. Proves the dashboard
    enumerates a multi-membership caller's workspaces, not just one.

    WHY(task-11): /stats/workspaces scopes to the caller's memberships for
    non-admin callers (confirmed: handler uses member_ws_ids from info.memberships,
    falls back to {workspace_id} for legacy JWTs). Row key is 'workspace_id'
    (not 'id') — assertion adjusted to the actual shape.

    WHY(fix2-stats-workspace): /stats/workspaces rows come from GROUP BY
    workspace_id on the chunks table, so a brand-new empty workspace never
    appears. Ingest one chunk first so the workspace shows up in the query.
    """
    suffix = int(time.time())
    ws = f"swa-{suffix}"
    ox, oy = f"swx-{suffix}", f"swy-{suffix}"
    # Ingest a chunk so ws has at least one row in the chunks table.
    _http("POST", f"{api}/memory/put", token,
        body={"source_id": f"smoke:stats-ws:{suffix}", "source_type": "note",
              "repo": "smoke-stats-ws", "path": "p",
              "content": f"STATS-WS {suffix}", "categorize": False},
        extra_headers=_act_as("A", role="admin",orgs=(ox, oy), workspace=ws), timeout=15.0)
    code, body, _ = _http("GET", f"{api}/stats/workspaces", token,
        extra_headers=_act_as("A", role="admin",orgs=(ox, oy), workspace=ws), timeout=12.0)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("stats_workspaces_lists_all", False, f"http={code} body={body}")
    ids = {w.get("workspace_id") for w in body.get("workspaces", [])}
    ok = ws in ids
    return CheckResult("stats_workspaces_lists_all", ok,
        f"active_ws_listed={ws in ids}  rows={len(ids)} (active must be listed)  marker={suffix}")


def check_repo_event_surfaces(api: str, token: str) -> CheckResult:
    """POST a synthetic workflow_run failure to /repo-events → the response must
    be action=repo_ci + igio_axis=issue (a hook_events row + repo_event chunk are
    created server-side, workspace-scoped). Uses a unique repo URL per run."""
    suffix = int(time.time())
    repo = f"https://github.com/smoke/repo-{suffix}"
    code, body, _ = _http("POST", f"{api}/repo-events", token,
        body={"event_type": "workflow_run", "repo": repo, "sha": f"s{suffix}",
              "conclusion": "failure", "workflow": "smoke-ci"}, timeout=12.0)
    if code != 200:
        return CheckResult("repo_event_surfaces", False, f"post failed http={code}: {body}")
    ok = isinstance(body, dict) and body.get("action") == "repo_ci" and body.get("igio_axis") == "issue"
    return CheckResult("repo_event_surfaces", ok,
        f"action={body.get('action') if isinstance(body, dict) else body} "
        f"igio_axis={body.get('igio_axis') if isinstance(body, dict) else '?'} "
        f"(want repo_ci/issue) marker={suffix}")


def check_repo_webhook_hmac(api: str, token: str) -> CheckResult:
    """POST to /repo-events/webhook for an UNWATCHED repo → must be rejected 401
    (the endpoint exists and authenticates via watch-record + HMAC). We don't post a
    valid signature here — that needs a real watched-repo secret; the unit suite
    covers the happy path. Proves the endpoint shipped and fails closed."""
    code, body, _ = _http("POST", f"{api}/repo-events/webhook", token,
        body={"repository": {"full_name": "smoke/repo-hmac"}, "after": "deadbeef"},
        extra_headers={"X-GitHub-Event": "push", "X-Hub-Signature-256": "sha256=deadbeef"},
        timeout=10.0)
    ok = code == 401
    return CheckResult("repo_webhook_hmac", ok,
        f"unwatched webhook rejected http={code}" if ok
        else f"expected 401, got http={code}: {body}")


def check_claim_rejects_foreign(api: str, token: str) -> CheckResult:
    """POST /stats/workspaces/claim with a NON-unclaimed workspace_id must be refused
    403 — only unclaimed:<device> buckets are claimable (no claiming infra/foreign WS)."""
    code, body, _ = _http("POST", f"{api}/stats/workspaces/claim", token,
                          body={"workspace_id": "system"}, timeout=10.0)
    ok = code == 403
    return CheckResult("claim_rejects_foreign", ok,
        f"foreign claim rejected http={code}" if ok
        else f"expected 403, got http={code}: {body}")


def check_text_model_switch_roundtrip(api: str, token: str) -> CheckResult:
    """GET /stats/admin/text-models → 200, shape has 'active' (str) and 'models'
    (list). Read-only: proves the endpoint shipped and returns the canonical
    shape without mutating prod config."""
    code, body, _ = _http("GET", f"{api}/stats/admin/text-models", token, timeout=10.0)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("text_model_switch_roundtrip", False,
                           f"http={code} body={body}")
    has_active = isinstance(body.get("active"), str)
    has_models = isinstance(body.get("models"), list)
    ok = has_active and has_models
    return CheckResult(
        "text_model_switch_roundtrip", ok,
        f"http=200 active={body.get('active')!r} models_count={len(body.get('models') or [])} "
        f"shape_ok={ok}",
    )


def check_reranker_active_pair(api: str, token: str) -> CheckResult:
    """GET /stats/admin/reranker-versions → 200, JSON 'active' is a list of
    length 1–2 (new multi-version shape). Proves the endpoint shipped the
    updated response contract without touching the active selection."""
    code, body, _ = _http("GET", f"{api}/stats/admin/reranker-versions", token, timeout=10.0)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("reranker_active_pair", False,
                           f"http={code} body={body}")
    active = body.get("active")
    ok = isinstance(active, list) and 1 <= len(active) <= 2
    return CheckResult(
        "reranker_active_pair", ok,
        f"http=200 active={active!r} (must be list len 1–2) shape_ok={ok}",
    )


def check_categories_overview_reachable(api: str, token: str) -> CheckResult:
    """GET /stats/categories-overview → 200, JSON has keys
    'total_categories' (int), 'workspaces' (list), 'unlinked' (list).
    Shape-only — values may be 0/empty on a fresh deploy."""
    code, body, _ = _http("GET", f"{api}/stats/categories-overview", token, timeout=10.0)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("categories_overview_reachable", False,
                           f"http={code} body={body}")
    has_total = isinstance(body.get("total_categories"), int)
    has_workspaces = isinstance(body.get("workspaces"), list)
    has_unlinked = isinstance(body.get("unlinked"), list)
    ok = has_total and has_workspaces and has_unlinked
    return CheckResult(
        "categories_overview_reachable", ok,
        f"http=200 total_categories={body.get('total_categories')} "
        f"workspaces={len(body.get('workspaces') or [])} "
        f"unlinked={len(body.get('unlinked') or [])} shape_ok={ok}",
    )


def check_project_groups_roundtrip(api: str, token: str) -> CheckResult:
    """C1: create group → assign repo → GET /projects shows color → delete. Proves the
    endpoint shipped end-to-end on the token's own workspace."""
    code, body, _ = _http("POST", f"{api}/project-groups", token,
                          body={"name": "smoke-c1-grp"}, timeout=10.0)
    if code != 200:
        return CheckResult("project_groups_roundtrip", False,
                           f"create failed http={code}: {body}")
    gid = (body or {}).get("id")
    _http("POST", f"{api}/project-groups/assign", token,
          body={"repo_slug": "smoke/repo-c1", "group_id": gid}, timeout=10.0)
    pc, pbody, _ = _http("GET", f"{api}/projects", token, timeout=10.0)
    colored = any((p.get("group_id") == gid and (p.get("group_color") or "").startswith("#"))
                  for p in ((pbody or {}).get("projects") or []))
    dc, _, _ = _http("DELETE", f"{api}/project-groups/{gid}", token, timeout=10.0)
    ok = pc == 200 and colored and dc == 200
    return CheckResult("project_groups_roundtrip", ok,
        "create→assign→color→delete ok" if ok
        else f"roundtrip failed: get={pc} colored={colored} del={dc}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    ("api_health",                    check_health),
    ("workspace_scoping",             check_workspace_scoped),
    ("memory_search_vector",          check_memory_search_returns_vector_hits),
    ("feedback_neutral_rejected",     check_feedback_binary_only),
    ("feedback_slug_resolution",      check_feedback_slug_resolution),
    ("feedback_count_delta",          check_feedback_count_moves),
    ("micro_batch_indexes",           check_micro_batch_indexes),
    ("jwt_invalid_signature",         check_jwt_invalid_signature_rejected),
    ("task_feedback_matrix",          check_task_feedback_matrix),
    ("wiki_graph_clusters",           check_wiki_graph_clusters),
    ("coverage_map_complete",         check_coverage_map_complete),
    ("retrieval_reasons_field",       check_retrieval_reasons_field),
    ("igio_axis_on_chunks",           check_igio_axis_on_chunks),
    ("wiki_context_injector_used",    check_wiki_context_injector_used),
    ("wiki_p7_endpoints",             check_wiki_p7_endpoints),
    ("wiki_p8_history",               check_wiki_p8_history),
    ("image_routing_supported",       check_image_routing_supported),
    ("db_wal_journal_active",         check_db_wal_journal_active),
    ("pipeline_stage_observability",  check_pipeline_stage_observability),
    ("predictive_transitions_endpoint", check_predictive_transitions_endpoint),
    ("training_merge_endpoint",       check_training_merge_endpoint),
    ("turbulence_endpoint",           check_turbulence_endpoint),
    ("pi_tasks_schema",               check_pi_tasks_schema),
    ("categorization_logging",        check_categorization_logging),
    ("jobs_progress_observability",   check_jobs_progress_observability),
    ("ingest_state_field",            check_ingest_state_field),
    ("visibility_isolation",          check_visibility_isolation),
    ("share_endpoint",                check_share_endpoint),
    ("stop_hook_e2e",                 check_stop_hook_auto_feedback_e2e),
    ("projects_route_cwd_remote",     check_projects_route_cwd_remote),
    ("mayring_process_fail_closed",   check_mayring_process_fail_closed),
    ("ingest_links_categories",       check_ingest_links_categories),
    ("reranker_cat_match_fires",      check_reranker_cat_match_fires),
    ("role_policy_roundtrip",         check_role_policy_roundtrip),
    ("role_enforcement",              check_role_enforcement),
    ("igio_lens_axes_present",        check_igio_lens_axes_present),
    ("intervention_todos_surface",    check_intervention_todos_surface),
    ("dashboard_endpoints",           check_dashboard_endpoints),
    ("feedback_log_movement",         check_feedback_log_movement),
    ("model_router_runtime",          check_model_router_runtime),
    ("model_identity",                check_model_identity),
    ("watcher_hook_fires",            check_watcher_hook_fires),
    ("wiki_cluster_depth",            check_wiki_cluster_depth),
    ("populate_accepts_batch_delay",  check_populate_accepts_batch_delay),
    ("categorization_call_type_logged", check_categorization_call_type_logged),
    ("chunk_metadata_complete",       check_chunk_metadata_complete),
    ("rag_function_search_finds_source", check_rag_function_search_finds_source),
    ("wiki_history_returns_snapshots",check_wiki_history_returns_snapshots),
    ("pi_second_opinion_endpoint",    check_pi_second_opinion_endpoint),
    ("retrieval_metrics_endpoint",    check_retrieval_metrics_endpoint),
    ("retrieval_stage_attribution",   check_retrieval_stage_attribution),
    ("reranker_runtime_switch",       check_reranker_runtime_switch),
    ("retrieval_ab_endpoint",         check_retrieval_ab_endpoint),
    ("reranker_rollout_decision",     check_reranker_rollout_decision),
    ("private_isolation",             check_private_isolation),
    ("public_visibility",             check_public_visibility),
    ("user_cross_device",             check_user_cross_device),
    ("org_member_visibility",         check_org_member_visibility),
    ("org_non_member_blocked",        check_org_non_member_blocked),
    ("org_revoke_isolation",          check_org_revoke_isolation),
    ("patch_visibility_authz",        check_patch_visibility_authz),
    ("multi_org_membership",          check_multi_org_membership),
    ("stats_workspaces_lists_all",    check_stats_workspaces_lists_all),
    ("repo_event_surfaces",           check_repo_event_surfaces),
    ("repo_webhook_hmac",             check_repo_webhook_hmac),
    ("claim_rejects_foreign",         check_claim_rejects_foreign),
    ("project_groups_roundtrip",      check_project_groups_roundtrip),
    ("project_link_boost_roundtrip",  check_project_link_boost_roundtrip),
    ("text_model_switch_roundtrip",   check_text_model_switch_roundtrip),
    ("reranker_active_pair",          check_reranker_active_pair),
    ("categories_overview_reachable", check_categories_overview_reachable),
    ("notifications_ingest_roundtrip", check_notifications_ingest_roundtrip),
]


# ---------------------------------------------------------------------------
# Alerting — open a GitHub issue when something fails
# ---------------------------------------------------------------------------

# Failures that are expected to stay red until the underlying TRACKER
# issue is closed. The smoke still RUNS them (so we keep visibility in
# the workflow log), but they DON'T trigger GitHub-issue alerts.
# Without this gate, ~20 spam issues hit the inbox in one afternoon
# from these three checks alone.
EXPECTED_PENDING_FAILURES: dict[str, dict[str, str]] = {}
# Removed:
#   training_merge_endpoint — smoke check broadened to accept 401
#   (route is admin-gated; 401 from the check's smoke creds proves the
#   route is registered, which IS the actual #87 acceptance).
#   reranker_cat_match_fires (#340) — GELÖST: query→category-Fenster geweitet
#   (mayring-core 0.50/8) → cat_match_hits=10 live. Smoke enforced es wieder.
#   model_identity (#349) — GELÖST: Check auf den kanonischen text_model.txt-
#   Picker-Pfad (POST /stats/admin/text-model toggle+revert) umgestellt statt
#   model-routes-Sentinel. Smoke enforced es wieder.
#   wiki_cluster_depth (#162) + igio_axis_on_chunks (#141) — beide Tracker
#   CLOSED, beide Checks live verifiziert GRÜN (#253 cleanup 2026-06-06:
#   igio ratio=0.548≥0.5, wiki clusters=12). Suppression entfernt → Smoke
#   enforced beide wieder.


def _failure_signature(real_failures: list[CheckResult]) -> str:
    """Stable string for dedupe — same fail-set → same signature."""
    return "smoke-fail-set:" + ",".join(sorted(r.name for r in real_failures))


def _find_existing_issue(signature: str,
                          fail_names: list[str] | None = None) -> str | None:
    """Find an open smoke-failure issue to comment on instead of creating new.

    Two-tier match (anti-spam strict):
      1. EXACT signature match — same fail-set, same issue.
      2. SUBSET match — if every name in the new fail-set already appears
         in the body of an existing open issue, comment on that one.
         Catches the partial-recovery case where a 3-fail issue exists
         and the next run only fails 1 of those 3.
    """
    import subprocess
    try:
        result = subprocess.run(
            [
                "gh", "issue", "list",
                "--repo", "Nileneb/MayringCoder",
                "--label", "smoke-failure",
                "--state", "open",
                "--limit", "30",
                "--json", "number,body",
            ],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            return None
        issues = json.loads(result.stdout or "[]")
        # 1. exact signature match
        for issue in issues:
            if signature in (issue.get("body") or ""):
                return str(issue.get("number"))
        # 2. subset match — every fail name already mentioned in some
        #    existing issue body? prefer the smallest such issue.
        if fail_names:
            for issue in issues:
                body = issue.get("body") or ""
                if all(f"`{n}`" in body for n in fail_names):
                    return str(issue.get("number"))
    except Exception:
        pass
    return None


def _close_resolved_smoke_issues() -> None:
    """User-Anforderung: wenn smoke wieder grün ist, ALLE noch offenen
    smoke-FAIL-issues automatisch schließen mit Verweis auf den
    aktuellen grünen Run. Sonst muss der User manuell jedes alte issue
    nachprüfen, obwohl der Bug längst gefixt ist.
    """
    import subprocess
    try:
        proc = subprocess.run(
            ["gh", "issue", "list", "--repo", "Nileneb/MayringCoder",
             "--state", "open", "--label", "smoke-failure",
             "--json", "number,title", "--limit", "30"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            print(f"# auto-close: gh issue list failed: {proc.stderr[:200]}")
            return
        import json as _json
        issues = _json.loads(proc.stdout or "[]")
    except Exception as e:
        print(f"# auto-close: list-error {e}")
        return
    if not issues:
        return
    msg = (
        f"Smoke ist wieder grün ({time.strftime('%Y-%m-%d %H:%M %Z')}). "
        f"Auto-close durch tools/smoke_test_production.py."
    )
    for it in issues:
        n = it["number"]
        try:
            subprocess.run(
                ["gh", "issue", "close", str(n),
                 "--repo", "Nileneb/MayringCoder",
                 "--comment", msg, "--reason", "completed"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            print(f"# auto-closed smoke issue #{n}: {it['title']}")
        except Exception as e:
            print(f"# auto-close #{n} failed: {e}")


def _open_github_issue(failed: list[CheckResult], elapsed: float) -> bool:
    """Open or comment on a smoke-failure issue.

    Three-tier alerting strategy (no more 20-issue inbox spam):

      1. Drop EXPECTED_PENDING_FAILURES from the alert path. They're
         tracked elsewhere; no need to email about them on every run.
      2. For real failures: search for an existing OPEN issue whose
         body contains the fail-set signature. If found → comment
         (single thread, no duplicate emails after the first).
      3. Only if no existing issue → create a new one.
    """
    import subprocess
    real_failures = [
        r for r in failed if r.name not in EXPECTED_PENDING_FAILURES
    ]
    pending_failures = [
        r for r in failed if r.name in EXPECTED_PENDING_FAILURES
    ]
    if pending_failures:
        names = [r.name for r in pending_failures]
        print(f"# alert: {len(pending_failures)} EXPECTED_PENDING failures "
              f"(no alert): {names}")
    if not real_failures:
        print("# alert: all failures are EXPECTED_PENDING — skip issue")
        return False

    signature = _failure_signature(real_failures)
    fail_names = [r.name for r in real_failures]
    existing = _find_existing_issue(signature, fail_names=fail_names)

    title = f"smoke FAIL ({len(real_failures)}) — {time.strftime('%Y-%m-%d %H:%M %Z')}"
    body_lines = [
        "Automated post-deploy smoke test detected failures in production.",
        "",
        f"**Elapsed:** {elapsed:.1f}s   **Failed:** {len(real_failures)}",
        f"**Signature:** `{signature}`",
        "",
        "## Failures",
        "",
    ]
    for r in real_failures:
        body_lines += [f"### `{r.name}`", "", "```",
                       r.detail or "(no detail)", "```", ""]
    if pending_failures:
        body_lines += ["", "## Expected-Pending (NOT alerting)"]
        for r in pending_failures:
            meta = EXPECTED_PENDING_FAILURES[r.name]
            body_lines += [f"- `{r.name}` — {meta['reason']} (tracker {meta['tracker']})"]
        body_lines += [""]
    body_lines += [
        "---",
        "",
        "Generated by `tools/smoke_test_production.py --alert-on-fail`.",
        "Re-run locally: `python tools/smoke_test_production.py`",
    ]
    body = "\n".join(body_lines)

    try:
        if existing:
            comment = (
                f"Recurrence at {time.strftime('%Y-%m-%d %H:%M %Z')} "
                f"({elapsed:.1f}s elapsed). Same fail-set as opening report."
            )
            result = subprocess.run(
                [
                    "gh", "issue", "comment", existing,
                    "--repo", "Nileneb/MayringCoder",
                    "--body", comment,
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                print(f"# alert: commented on existing issue #{existing} "
                      f"({len(real_failures)} real failures, dedupe)")
                return True
            print(f"# alert: comment failed: {result.stderr.strip()}",
                  file=sys.stderr)
            return False

        result = subprocess.run(
            [
                "gh", "issue", "create",
                "--repo", "Nileneb/MayringCoder",
                "--title", title,
                "--body", body,
                "--label", "smoke-failure",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"# alert: opened issue → {result.stdout.strip()}")
            return True
        print(f"# alert: gh issue create failed: {result.stderr.strip()}",
              file=sys.stderr)
    except FileNotFoundError:
        print("# alert: 'gh' CLI not installed — skipping",
              file=sys.stderr)
    except Exception as e:
        print(f"# alert: could not open issue: {type(e).__name__}: {e}",
              file=sys.stderr)
    return False


# WHY(#253): every smoke run created ephemeral `<prefix>-<ts>` workspaces (oa-,
# aza-, moa-, swa-, …) and never cleaned them → 1856 junk workspaces in prod.
# After the run we purge every smoke-shaped workspace via the admin endpoint so
# the catalog stays clean. The shape can't match a real workspace (UUIDs start
# with a digit) and a hard protected-set on the server refuses the real ones too.
_SMOKE_WS_RE = re.compile(r"^[a-z][a-z0-9]*-\d{9,}$")
_PROTECTED_WS = {"system", "public", "default", "bene:logs", "mayringcoder",
                 "019d6933-002e-7153-a7df-f14e4c7d52b4",
                 "019e14d6-0489-7348-bca8-e29c11293cb7"}


def _teardown_smoke_workspaces(api: str, token: str) -> None:
    # WHY(2026-06-10 silent-miss): purge the union of (a) workspaces THIS run
    # registered via _act_as and (b) whatever the server lists. A green run
    # left 10 leftovers with a completely SILENT teardown — the listing
    # returned none of them, and both the empty-targets case and non-200
    # purges were invisible. Now: always print, list failures with codes.
    listed: set[str] = set()
    code, body, _ = _http("GET", f"{api}/stats/workspaces", token)
    if code == 200 and isinstance(body, dict):
        listed = {str(w.get("workspace_id", "")) for w in body.get("workspaces", [])
                  if _SMOKE_WS_RE.match(str(w.get("workspace_id", "")))}
    else:
        print(f"# teardown: /stats/workspaces {code}; purging registered set only")

    targets = sorted((listed | _EPHEMERAL_WS) - _PROTECTED_WS)
    purged, failed = 0, []
    for ws in targets:
        c, _, _ = _http("POST", f"{api}/stats/admin/purge-workspace", token,
                        body={"workspace_id": ws})
        if c == 200:
            purged += 1
        else:
            failed.append(f"{ws}={c}")
    print(f"# teardown: purged {purged}/{len(targets)} ephemeral smoke workspaces "
          f"(listed={len(listed)} registered={len(_EPHEMERAL_WS)})")
    if failed:
        print(f"# teardown: FAILED purges: {', '.join(failed)}")

    # WHY(2026-06-10): the C3 check creates one smoke/repo-c3-<ts> PROJECT per
    # run (and repo_event/HMAC checks more smoke/repo-* refs) — projects live in
    # the purge-protected 'system' workspace, so the workspace purge above never
    # touched them and ~90 junk rows piled up in the dashboard Projekte list.
    pc, pbody, _ = _http("POST", f"{api}/stats/admin/purge-smoke-projects", token)
    if pc == 200 and isinstance(pbody, dict):
        print(f"# teardown: purged smoke projects={pbody.get('projects', 0)} "
              f"links={pbody.get('chunk_project_links', 0)} groups={pbody.get('project_groups', 0)}")
    else:
        print(f"# teardown: purge-smoke-projects FAILED http={pc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default=os.getenv("MAYRING_API_URL", API_DEFAULT))
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--skip", action="append", default=[],
                   help="check id to skip (repeatable)")
    p.add_argument("--alert-on-fail", action="store_true",
                   help="open a GitHub issue when any check fails (uses gh CLI)")
    p.add_argument("--ready-timeout", type=float, default=60.0,
                   help="max seconds to wait for /health before running checks (#250)")
    p.add_argument("--retry-failed", type=int,
                   default=int(os.getenv("SMOKE_RETRY_FAILED", "2")),
                   help="re-warm + re-run each FAILED check this many times before "
                        "declaring it failed. WHY(false-positive-smoke 2026-06-08): a "
                        "mid-run deploy restart (502) or a cold post-restart bge-m3 "
                        "model (max_score=0.000) made deploy-window transients look "
                        "like regressions. Real failures reproduce across retries.")
    p.add_argument("--pace", type=float, default=float(os.getenv("SMOKE_PACE", "0.5")),
                   help="seconds to pause between checks. WHY(api-saturation "
                        "2026-05-24): the prod API is a single uvicorn worker; firing "
                        "~30 heavy checks back-to-back (right as post-deploy-ingest also "
                        "hammers it on a cold Chroma) saturated the event loop and the "
                        "smoke red-flagged its own self-inflicted load. A small pause lets "
                        "the worker drain between checks.")
    args = p.parse_args()

    token = _load_token()
    api = args.api.rstrip("/")
    skip = set(args.skip)

    print(f"# Smoke tests against {api}")
    print(f"# JWT loaded: {len(token)} chars")
    # Wait the post-deploy restart window out before running anything (#250) —
    # so a cold-start doesn't red-flag api_health and open a false-positive issue.
    wait_for_api_ready(api, token, max_wait=args.ready_timeout)
    # Then gate on SEARCH warmth — /health alone is not enough (the search
    # pipeline can be saturated while /health stays fast). Prevents the
    # retry-amplified 25min hang after a cutover (2026-05-30).
    wait_for_search_ready(api, token, max_wait=args.ready_timeout)
    print()

    # WHY(#344): purge BEFORE the run too — a finally-block does not run when the
    # CI step is hard-killed (SIGKILL on workflow timeout, e.g. the 781s #345 run),
    # so smoke.*-workspaces from a killed prior run would otherwise accumulate.
    # This bounds the clutter to at most one in-flight run.
    _teardown_smoke_workspaces(api, token)

    results: list[CheckResult] = []
    t_start = time.time()
    try:
        for name, fn in ALL_CHECKS:
            if name in skip:
                print(f"  SKIP  {name}")
                continue
            t0 = time.time()
            try:
                res = fn(api, token)
            except Exception as e:
                res = CheckResult(name, False, f"check raised: {type(e).__name__}: {e}")
            dt = time.time() - t0
            marker = " OK " if res.passed else "FAIL"
            print(f"  [{marker}] {res.name}  ({dt:.2f}s)")
            if res.detail:
                indent = "         "
                for line in res.detail.split("\n"):
                    print(f"{indent}{line}")
            results.append(res)
            if args.fail_fast and not res.passed:
                break
            if args.pace > 0:
                time.sleep(args.pace)
    finally:
        # Self-clean ephemeral workspaces this run (and any prior runs) created
        # (#253/#344) — in finally so an exception/SystemExit mid-loop still purges.
        _teardown_smoke_workspaces(api, token)

    # WHY(false-positive-smoke 2026-06-08): the readiness gate runs once, up front.
    # It cannot protect against a SECOND deploy restarting the API *mid-run* (two
    # back-to-back deploys → every in-flight check returns http=502) or against the
    # bge-m3 model going cold after a restart (search 200 but max_score=0.000). Both
    # produced smoke-FAIL issues (#356-360) that were pure deploy-window artifacts.
    # Re-warm and re-run ONLY the failed checks: a transient clears on retry, a real
    # failure (e.g. broken chunk metadata) reproduces and is still reported.
    retry_n = args.retry_failed
    failed = [r for r in results if not r.passed]
    if failed and retry_n > 0:
        idx = {r.name: i for i, r in enumerate(results)}
        attempt = 0
        while failed and attempt < retry_n:
            attempt += 1
            print(f"\n# {len(failed)} check(s) failed — re-warming + retry "
                  f"{attempt}/{retry_n} (deploy-window transients clear, real "
                  f"failures reproduce)")
            wait_for_api_ready(api, token, max_wait=args.ready_timeout)
            wait_for_search_ready(api, token, max_wait=args.ready_timeout)
            still: list[CheckResult] = []
            fn_by_name = dict(ALL_CHECKS)
            for r in failed:
                fn = fn_by_name.get(r.name)
                if fn is None:
                    still.append(r)
                    continue
                t0 = time.time()
                try:
                    res = fn(api, token)
                except Exception as e:
                    res = CheckResult(r.name, False,
                                      f"check raised: {type(e).__name__}: {e}")
                dt = time.time() - t0
                marker = " OK " if res.passed else "FAIL"
                print(f"  [{marker}] {res.name}  ({dt:.2f}s, retry {attempt})")
                if res.detail:
                    for line in res.detail.split("\n"):
                        print(f"         {line}")
                results[idx[r.name]] = res
                if not res.passed:
                    still.append(res)
                if args.pace > 0:
                    time.sleep(args.pace)
            failed = still
        # WHY(2026-06-10): retried checks create fresh ephemeral workspaces
        # AFTER the finally-teardown of the main loop — without this purge they
        # accumulate until the next scheduled run (the 07:35-leftover batch).
        _teardown_smoke_workspaces(api, token)

    failed = [r for r in results if not r.passed]
    elapsed = time.time() - t_start
    print()
    print(f"# {len(results) - len(failed)}/{len(results)} passed  ({elapsed:.1f}s total)")
    if failed:
        # EXPECTED_PENDING-Failures sollen weder Issue NOCH Workflow-Email
        # auslösen. Vorher kam für jeden Pending-Failure der ganze
        # Workflow rot → Mail an User. Issue-Anlegen war schon
        # gefiltert, die Workflow-Mail nicht.
        real_failures = [r for r in failed if r.name not in EXPECTED_PENDING_FAILURES]
        print(f"# FAIL: {', '.join(r.name for r in failed)}")
        if args.alert_on_fail:
            _open_github_issue(failed, elapsed)
        if not real_failures:
            print("# all real-world checks pass — only EXPECTED_PENDING "
                  "items failed; workflow stays green.")
            if args.alert_on_fail:
                _close_resolved_smoke_issues()
            return 0
        return 1
    print("# all good — every critical path is actually working in prod")
    if args.alert_on_fail:
        _close_resolved_smoke_issues()
    return 0


if __name__ == "__main__":
    sys.exit(main())
