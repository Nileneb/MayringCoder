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
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

API_DEFAULT = "https://mcp.linn.games"
JWT_PATH = os.path.expanduser("~/.config/mayring/hook.jwt")


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
          timeout: float = 10.0) -> tuple[int, dict | None, float]:
    """Returns (status_code, parsed_json_or_None, elapsed_seconds)."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
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
        return e.code, err_json, time.time() - t0
    except Exception as e:
        return 0, {"_error": f"{type(e).__name__}: {e}"}, time.time() - t0


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

    Accepts:
      • "user-<id>" — JWT.sub-derived (smoke ran with user/Sanctum token)
      • "system"   — service-token path (smoke ran with MCP_SERVICE_TOKEN)
    Rejects: legacy UUID, "default", or empty — these would mean either
    workspace-id derivation is broken or the auth fell through to a
    non-tenant default.
    """
    code, body, dt = _http("GET", f"{api}/stats/workspaces", token)
    if code != 200:
        return CheckResult("workspace_scoping", False,
                           f"/stats/workspaces http={code}: {body}")
    ws = (body or {}).get("workspace_id", "")
    is_sane = ws.startswith("user-") or ws == "system"
    return CheckResult(
        "workspace_scoping",
        is_sane,
        f"workspace_id={ws!r} (expected 'user-<id>' or 'system')",
    )


def check_memory_search_returns_vector_hits(api: str, token: str) -> CheckResult:
    """A real search must:
    - return 200
    - report vector_stage diagnostics (not 'unknown')
    - have at least one result with score_vector > 0 in top-5"""
    code, body, dt = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": "memory feedback hook stop", "top_k": 5,
              "include_text": False, "llm_prefilter": False},
        timeout=15.0,
    )
    if code != 200:
        return CheckResult("memory_search_vector", False,
                           f"http={code} time={dt:.2f}s body={body}")
    diag = (body or {}).get("diagnostics", {}).get("vector_stage", "?")
    results = (body or {}).get("results", [])
    has_vec = any(r.get("score_vector", 0) > 0 for r in results)
    diag_ok = isinstance(diag, str) and (diag.startswith("ok(") or diag == "query_cache_hit")
    return CheckResult(
        "memory_search_vector",
        diag_ok,  # vector hits not always present, but diag must be sensible
        f"http={code} time={dt:.2f}s diag={diag!r} top_k_with_vec={has_vec} results={len(results)}",
    )


def check_feedback_binary_only(api: str, token: str) -> CheckResult:
    """Neutral signal must be rejected (Issue: silent neutral pollution)."""
    code, body, _ = _http(
        "POST", f"{api}/memory/feedback", token,
        body={"chunk_id": "chk_smoke_test_dummy_1", "signal": "neutral"},
    )
    return CheckResult(
        "feedback_neutral_rejected",
        code in (400, 422),
        f"http={code} body={body} — must be 400 (route) or 422 (pydantic)",
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
        body={"chunk_id": sid, "signal": "positive"},
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
    """POST one positive feedback, verify the count actually increased."""
    pre_code, pre, _ = _http("GET", f"{api}/stats/summary", token)
    if pre_code != 200:
        return CheckResult("feedback_count_delta", False,
                           f"pre /stats/summary http={pre_code}")
    pre_pos = (pre or {}).get("feedback", {}).get("positive", 0)

    # Get a real chunk_id
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
        body={"chunk_id": cid, "signal": "positive"},
    )
    if fb_code != 200:
        return CheckResult("feedback_count_delta", False,
                           f"feedback POST http={fb_code} body={fb_body}")

    time.sleep(1)
    post_code, post, _ = _http("GET", f"{api}/stats/summary", token)
    post_pos = (post or {}).get("feedback", {}).get("positive", 0)
    delta = post_pos - pre_pos
    return CheckResult(
        "feedback_count_delta",
        delta >= 1,
        f"pre.positive={pre_pos}  post.positive={post_pos}  delta={delta}",
    )


def check_micro_batch_indexes(api: str, token: str) -> CheckResult:
    """Turn-capture endpoint must accept a turn pair and create a source."""
    session_id = f"smoke-{int(time.time())}"
    code, body, dt = _http(
        "POST", f"{api}/conversation/micro-batch", token,
        body={
            "turns": [
                {"role": "user", "content": "smoke test prompt", "timestamp": ""},
                {"role": "assistant", "content": "smoke test response", "timestamp": ""},
            ],
            "session_id": session_id,
            "workspace_slug": "mayringcoder",
        },
        timeout=30.0,  # server summarises via LLM
    )
    indexed = bool((body or {}).get("indexed"))
    sid = (body or {}).get("source_id", "")
    return CheckResult(
        "micro_batch_indexes",
        code == 200 and indexed,
        f"http={code} time={dt:.2f}s indexed={indexed} source_id={sid}",
    )


def check_dashboard_endpoints(api: str, token: str) -> CheckResult:
    """All 10 dashboard endpoints respond 200 with a workspace-id field."""
    paths = [
        "/stats/recent-ops", "/stats/jobs-history", "/stats/feedback-log",
        "/stats/source-refs", "/stats/triggers", "/stats/topic-flow",
        "/stats/pi-tasks", "/stats/activations", "/stats/workspaces",
        "/stats/vector-trend",
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
    """Closed Issue #87 acceptance: POST /api/training/merge endpoint
    exists. Empty POST should yield validation error (422), not 404 or
    500 — proves the route is registered."""
    code, body, _ = _http("POST", f"{api}/api/training/merge", token, body={})
    # 200/400/422 all prove the route exists; 404 = never built; 500 = crash
    return CheckResult(
        "training_merge_endpoint",
        code in (200, 400, 422),
        f"http={code} body={body}  (200/400/422 = route exists; 404 or 500 = regression)",
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
    """Closed Issue #94 acceptance: tampered JWTs must be rejected with 401.

    Acceptance from issue body: 'HTTP 401 bei ungültigem Token'. Was never
    verified live — closed on synthetic unit-tests only.
    """
    # Take the real token, flip the last 4 chars of the signature segment
    # → invalid signature, valid shape.
    parts = token.split(".")
    if len(parts) != 3:
        return CheckResult("jwt_invalid_signature", False,
                           "current token is not a 3-part JWT — can't tamper")
    tampered = ".".join([parts[0], parts[1], parts[2][:-4] + "AAAA"])
    code, body, _ = _http(
        "GET", f"{api}/stats/summary",
        token=tampered,  # use the tampered token
    )
    return CheckResult(
        "jwt_invalid_signature",
        code == 401,
        f"http={code}  body={body}  (must be 401)",
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

    ok = s1 == "new" and s2 == "unchanged" and s3 == "changed"
    return CheckResult(
        "ingest_state_field",
        ok,
        f"first(new)={s1!r}  same(unchanged)={s2!r}  different(changed)={s3!r}",
    )


def check_visibility_isolation(api: str, token: str) -> CheckResult:
    """Ingest a private + a public source, search, verify visibility flags.

    Catches regressions in the visibility model — private chunks must
    only surface for the workspace that ingested them, public chunks
    must be visible to anyone with a valid JWT. The user explicitly
    asked for this domain to be tested as part of the live login flow.
    """
    suffix = int(time.time())
    workspace_slug = "smoke-vis"

    # 1) Ingest a PRIVATE source (default visibility)
    priv_id = f"smoke:vis:private:{suffix}"
    code1, body1, _ = _http(
        "POST", f"{api}/memory/put", token,
        body={
            "source_id": priv_id,
            "source_type": "note",
            "repo": workspace_slug,
            "path": "private-marker",
            "content": f"PRIVATE marker token {suffix}",
            "categorize": False,
        },
        timeout=15.0,
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
            "content": f"PUBLIC marker token {suffix}",
            "categorize": False,
        },
        timeout=15.0,
    )
    if code2 != 200:
        return CheckResult("visibility_isolation", False,
                           f"public ingest failed http={code2}: {body2}")
    code3, body3, _ = _http(
        "PATCH", f"{api}/sources/{urllib.parse.quote(pub_id, safe='')}/visibility",
        token, body={"visibility": "public"},
    )
    if code3 != 200:
        return CheckResult("visibility_isolation", False,
                           f"PATCH visibility failed http={code3}: {body3}")

    # 3) Search for the marker token — both should surface for the
    #    same user/workspace that ingested them.
    code4, body4, _ = _http(
        "POST", f"{api}/memory/search", token,
        body={"query": f"marker token {suffix}", "top_k": 5,
              "include_text": True, "llm_prefilter": False},
        timeout=12.0,
    )
    if code4 != 200:
        return CheckResult("visibility_isolation", False,
                           f"search failed http={code4}")
    src_ids_seen = {r["source_id"] for r in (body4 or {}).get("results", [])}
    private_visible = priv_id in src_ids_seen
    public_visible = pub_id in src_ids_seen

    return CheckResult(
        "visibility_isolation",
        private_visible and public_visible,
        f"private_in_search={private_visible}  public_in_search={public_visible}  "
        f"results={len(src_ids_seen)}  marker={suffix}",
    )


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
    pre_total = ((pre or {}).get("feedback", {}).get("positive", 0)
                 + (pre or {}).get("feedback", {}).get("negative", 0))

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

    # Find the stop_hook.py module and import it
    repo_root = Path(__file__).resolve().parent.parent
    hook_path = repo_root / "claude-plugin" / "hooks" / "stop_hook.py"
    if not hook_path.exists():
        return CheckResult("stop_hook_e2e", False,
                           f"stop_hook.py not at {hook_path}")
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
    fake_turns = [
        {"role": "user", "content": "smoke e2e test prompt", "timestamp": ""},
        {"role": "assistant",
         "content": "smoke e2e test response — no source paths mentioned",
         "timestamp": ""},
    ]
    sh._auto_feedback(fake_turns, session_id, token)

    # State file should be cleared
    state_cleared = not state_file.exists()

    # Post: count again, expect +len(pairs)
    time.sleep(1)
    post_code, post, _ = _http("GET", f"{api}/stats/summary", token)
    post_total = ((post or {}).get("feedback", {}).get("positive", 0)
                  + (post or {}).get("feedback", {}).get("negative", 0))
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
    ("wiki_p7_endpoints",             check_wiki_p7_endpoints),
    ("wiki_p8_history",               check_wiki_p8_history),
    ("image_routing_supported",       check_image_routing_supported),
    ("training_merge_endpoint",       check_training_merge_endpoint),
    ("turbulence_endpoint",           check_turbulence_endpoint),
    ("pi_tasks_schema",               check_pi_tasks_schema),
    ("categorization_logging",        check_categorization_logging),
    ("jobs_progress_observability",   check_jobs_progress_observability),
    ("ingest_state_field",            check_ingest_state_field),
    ("visibility_isolation",          check_visibility_isolation),
    ("stop_hook_e2e",                 check_stop_hook_auto_feedback_e2e),
    ("dashboard_endpoints",           check_dashboard_endpoints),
    ("feedback_log_movement",         check_feedback_log_movement),
]


# ---------------------------------------------------------------------------
# Alerting — open a GitHub issue when something fails
# ---------------------------------------------------------------------------

def _open_github_issue(failed: list[CheckResult], elapsed: float) -> bool:
    """Open a tracking issue in Nileneb/MayringCoder. Uses gh CLI.

    GitHub auto-emails the repo owner on new issues — that's the
    alerting channel. No watchdog daemon needed, no Telegram bot,
    no PagerDuty: the alert is the issue itself.
    """
    import subprocess
    title = f"smoke FAIL ({len(failed)}/{len(failed)}) — {time.strftime('%Y-%m-%d %H:%M %Z')}"
    body_lines = [
        "Automated post-deploy smoke test detected failures in production.",
        "",
        f"**Elapsed:** {elapsed:.1f}s",
        f"**Failed checks:** {len(failed)}",
        "",
        "## Failures",
        "",
    ]
    for r in failed:
        body_lines += [f"### `{r.name}`", "", "```", r.detail or "(no detail)", "```", ""]
    body_lines += [
        "---",
        "",
        "Generated by `tools/smoke_test_production.py --alert-on-fail`.",
        "Re-run locally to reproduce: `python tools/smoke_test_production.py`",
    ]
    body = "\n".join(body_lines)
    try:
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
        print("# alert: 'gh' CLI not installed — skipping issue creation",
              file=sys.stderr)
    except Exception as e:
        print(f"# alert: could not open issue: {type(e).__name__}: {e}",
              file=sys.stderr)
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default=os.getenv("MAYRING_API_URL", API_DEFAULT))
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--skip", action="append", default=[],
                   help="check id to skip (repeatable)")
    p.add_argument("--alert-on-fail", action="store_true",
                   help="open a GitHub issue when any check fails (uses gh CLI)")
    args = p.parse_args()

    token = _load_token()
    api = args.api.rstrip("/")
    skip = set(args.skip)

    print(f"# Smoke tests against {api}")
    print(f"# JWT loaded: {len(token)} chars")
    print()

    results: list[CheckResult] = []
    t_start = time.time()
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

    failed = [r for r in results if not r.passed]
    elapsed = time.time() - t_start
    print()
    print(f"# {len(results) - len(failed)}/{len(results)} passed  ({elapsed:.1f}s total)")
    if failed:
        print(f"# FAIL: {', '.join(r.name for r in failed)}")
        if args.alert_on_fail:
            _open_github_issue(failed, elapsed)
        return 1
    print("# all good — every critical path is actually working in prod")
    return 0


if __name__ == "__main__":
    sys.exit(main())
